import Foundation
import Glibc
import NNC
import PNG

// This target has no PythonKit / NNCPythonConversion dependency.
@main
enum Flux3Generate {
  static func main() throws {
    setbuf(stdout, nil)
    let env = ProcessInfo.processInfo.environment
    let variant = Flux3Variant(environment: env)
    let checkpoints = Flux3Checkpoints(environment: env)
    let output = env["FLUX3_OUTPUT"] ?? "/slow/Data/\(variant.checkpointStem)_swift.png"
    let prompt =
      env["FLUX3_PROMPT"]
      ?? "A red ceramic teapot on a wooden table beside a window, soft morning light, a small white cup and a folded linen napkin, realistic photograph."
    let width = Int(env["FLUX3_WIDTH"] ?? "512")!
    let height = Int(env["FLUX3_HEIGHT"] ?? "512")!
    let steps = Int(env["FLUX3_STEPS"] ?? "\(variant.defaultSteps)")!
    let guidance = Float(env["FLUX3_GUIDANCE"] ?? "\(variant.defaultGuidance)")!
    let seed = UInt64(env["FLUX3_SEED"] ?? "42")!
    let device = Int(env["FLUX3_DEVICE"] ?? "1")!
    let precision = env["FLUX3_DTYPE"] ?? "bfloat16"
    precondition(["float16", "bfloat16"].contains(precision), "Unknown FLUX3_DTYPE")
    precondition(width > 0 && height > 0 && width % 16 == 0 && height % 16 == 0)
    precondition(steps > 0)
    let start = Date()
    let tokenizer = try Flux3Tokenizer(path: checkpoints.tokenizer)
    let encoded = tokenizer.prompt(prompt)
    print("Swift-only FLUX.3 generation:", width, "x", height, "steps", steps, "seed", seed)
    print("Prompt:", prompt)
    print("Tokens:", encoded.valid, "real,", encoded.tokens.count, "padded")
    print("DiT precision:", precision, "residual:", precision == "float16" ? "float32" : precision)
    let graph = DynamicGraph()
    graph.maxConcurrency = .limit(1)
    let context = encode(
      graph: graph, device: device, checkpoint: checkpoints.text, encoded: encoded)
    let unconditional =
      variant.flowMap
      ? nil
      : encode(
        graph: graph, device: device, checkpoint: checkpoints.text, encoded: tokenizer.prompt(""))
    let latent: Tensor<Float>
    if precision == "float16" {
      latent = denoise(
        Float16.self, precision: precision, graph: graph, device: device,
        checkpoint: checkpoints.dit, context: context,
        unconditional: unconditional, variant: variant, guidance: guidance,
        width: width, height: height, steps: steps, seed: seed)
    } else {
      latent = denoise(
        BFloat16.self, precision: precision, graph: graph, device: device,
        checkpoint: checkpoints.dit, context: context,
        unconditional: unconditional, variant: variant, guidance: guidance,
        width: width, height: height, steps: steps, seed: seed)
    }
    let decoded = decode(
      graph: graph, device: device, checkpoint: checkpoints.vae, latent: latent,
      width: width, height: height)
    try save(decoded, width: width, height: height, path: output)
    let elapsed = Date().timeIntervalSince(start)
    let metadata: [String: Any] = [
      "checkpoints": ["text": checkpoints.text, "dit": checkpoints.dit, "vae": checkpoints.vae],
      "tokenizer": checkpoints.tokenizer, "prompt": prompt, "width": width,
      "height": height, "steps": steps, "seed": seed, "seconds": elapsed,
      "variant": variant.rawValue, "guidance": guidance,
      "python": false, "text_dtype": "float16", "dit_dtype": precision, "vae_dtype": "float32",
      "dit_residual_dtype": precision == "float16" ? "float32" : precision,
      "dit_text_scale": precision == "float16" ? Flux3Config.fp16TextScale : 1,
      "dit_ffn_scale": precision == "float16" ? Flux3Config.fp16FFNScale : 1,
      "noise_rng": "SplitMix64 + Box-Muller", "output": output,
    ]
    try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: output + ".json"))
    print("Saved", output, "in", String(format: "%.1f", elapsed), "seconds")
  }

  static func encode(
    graph: DynamicGraph, device: Int, checkpoint: String,
    encoded: (tokens: [Int32], valid: Int)
  ) -> Tensor<BFloat16> {
    graph.withNoGrad {
      let (ids, positions) = flux3TokenInputs(encoded.tokens, valid: encoded.valid)
      let tokens = graph.variable(ids.toGPU(device))
      let rotary = graph.variable(flux3TextRotary(Float16.self, positions: positions).toGPU(device))
      // Keep the same full-encoder graph as the exported Krea-style store.
      let (model, _, _) = Flux3Text(
        Float16.self, length: encoded.tokens.count,
        validLength: encoded.valid, completeEncoder: true)
      model.maxConcurrency = .limit(1)
      model.compile(inputs: tokens, rotary)
      try! graph.openStore(checkpoint, flags: .readOnly) {
        try! $0.read("text_model", model: model, strict: true)
      }.get()
      let outputs = model(inputs: tokens, rotary)
      var features = Tensor<Float>(.CPU, .HWC(1, encoded.tokens.count, 20_480))
      for tap in 0..<8 {
        let values = Tensor<Float>(from: outputs[tap].as(of: Float16.self).rawValue.toCPU())
        for i in encoded.tokens.indices {
          for c in 0..<2_560 {
            precondition(values[i, c].isFinite, "Non-finite text feature")
            features[0, i, tap * 2_560 + c] = values[i, c]
          }
        }
      }
      print("Text encoding complete")
      return Tensor<BFloat16>(from: features)
    }
  }

  static func denoise<T: TensorNumeric>(
    _ type: T.Type, precision: String, graph: DynamicGraph, device: Int, checkpoint: String,
    context: Tensor<BFloat16>, unconditional: Tensor<BFloat16>?, variant: Flux3Variant,
    guidance: Float, width: Int, height: Int, steps: Int, seed: UInt64
  ) -> Tensor<Float> {
    graph.withNoGrad {
      let count = width / 16 * (height / 16)
      let length = context.shape[1]
      let schedule = flux3Schedule(steps: steps)
      let noise = Tensor<T>(from: flux3Noise(seed: seed, count: count))
      var current = graph.variable(noise.toGPU(device))
      // Keep Base's Euler state in FP32; perform CFG there as well.
      var fullCurrent = graph.variable(
        Tensor<Float>(from: Tensor<BFloat16>(from: flux3Noise(seed: seed, count: count))).toGPU(
          device))
      if !variant.flowMap { current = DynamicGraph.Tensor<T>(from: fullCurrent) }
      let text = graph.variable(Tensor<T>(from: Tensor<Float>(from: context)).toGPU(device))
      let rotary = graph.variable(
        flux3Rotary(flux3IDs(text: length, height: height / 16, width: width / 16)).toGPU(device))
      func time(_ values: [Float]) -> DynamicGraph.Tensor<T> {
        graph.variable(
          Tensor<T>(from: flux3Time(values, precision: precision))
            .toGPU(device))
      }
      let t = time([schedule[0], 0])
      let r = time([schedule[1]])
      func makeModel(
        _ text: DynamicGraph.Tensor<T>, _ rotary: DynamicGraph.Tensor<Float>, _ length: Int
      ) -> Model {
        let (model, _, _) = Flux3DiT(
          type, textLength: length, imageLength: count, referenceLength: 0, flowMap: variant.flowMap
        )
        model.maxConcurrency = .limit(1)
        var inputs: [DynamicGraph.AnyTensor] = [current, text, rotary, t]
        if variant.flowMap { inputs.append(r) }
        model.compile(inputs: inputs)
        try! graph.openStore(checkpoint, flags: .readOnly) { store in
          try! store.read("dit", model: model, strict: true) { name, dataType, _, _ in
            guard let tensor = store.read(name) else { return .fail }
            if tensor.dataType == dataType { return .final(tensor) }
            // The store's automatic conversion only supports FP16 <-> FP32.
            let value = Tensor<Float>(from: tensor).toCPU()
            if dataType == .BFloat16 { return .final(Tensor<BFloat16>(from: value)) }
            if dataType == .Float16 { return .final(Tensor<Float16>(from: value)) }
            precondition(dataType == .Float32)
            return .final(value)
          }
        }.get()
        return model
      }
      let model = makeModel(text, rotary, length)
      var negative: (Model, DynamicGraph.Tensor<T>, DynamicGraph.Tensor<Float>)?
      if let unconditional = unconditional {
        let negativeText = graph.variable(
          Tensor<T>(from: Tensor<Float>(from: unconditional)).toGPU(device))
        let negativeLength = unconditional.shape[1]
        let negativeRotary = graph.variable(
          flux3Rotary(flux3IDs(text: negativeLength, height: height / 16, width: width / 16)).toGPU(
            device))
        // Preserve the empty prompt's own padded length, as in the reference.
        let negativeModel =
          negativeLength == length ? model : makeModel(negativeText, negativeRotary, negativeLength)
        negative = (negativeModel, negativeText, negativeRotary)
      }
      for i in 0..<steps {
        let begin = Date()
        let t = time([schedule[i], 0])
        if let (negativeModel, negativeText, negativeRotary) = negative {
          let uncond = DynamicGraph.Tensor<Float>(
            from:
              negativeModel(inputs: current, negativeText, negativeRotary, t)[0].as(of: T.self))
          let cond = DynamicGraph.Tensor<Float>(
            from:
              model(inputs: current, text, rotary, t)[0].as(of: T.self))
          let velocity = uncond + (cond - uncond) * guidance
          fullCurrent = fullCurrent + velocity * (schedule[i + 1] - schedule[i])
          current = DynamicGraph.Tensor<T>(from: fullCurrent)
        } else {
          let velocity = model(
            inputs: current, text, rotary, t, time([schedule[i + 1]]))[0].as(of: T.self)
          current = current + velocity * (schedule[i + 1] - schedule[i])
        }
        // Synchronize and record a finite sample at every step.
        let sample =
          variant.flowMap
          ? Tensor<Float>(from: current.rawValue.toCPU()) : fullCurrent.rawValue.toCPU()
        var peak: Float = 0
        for token in 0..<count {
          for c in 0..<128 {
            precondition(sample[0, token, c].isFinite, "Non-finite latent at step \(i + 1)")
            peak = max(peak, abs(sample[0, token, c]))
          }
        }
        print(
          "Step", i + 1, "/", steps, "t", schedule[i], "->", schedule[i + 1],
          "peak", peak, "seconds", String(format: "%.2f", Date().timeIntervalSince(begin)))
      }
      return variant.flowMap
        ? Tensor<Float>(from: current.rawValue.toCPU()) : fullCurrent.rawValue.toCPU()
    }
  }

  static func decode(
    graph: DynamicGraph, device: Int, checkpoint: String,
    latent: Tensor<Float>, width: Int, height: Int
  ) -> Tensor<Float> {
    graph.withNoGrad {
      let (mean, std) = try! graph.openStore(checkpoint, flags: .readOnly) {
        (
          Tensor<Float>(from: $0.read("latent_mean")!).toCPU(),
          Tensor<Float>(from: $0.read("latent_std")!).toCPU()
        )
      }.get()
      let unpacked = flux3DecodeLatents(latent, mean: mean, std: std, height: height, width: width)
      let input = graph.variable(unpacked.toGPU(device))
      let (_, model) = Flux3VAEDecoder(
        channels: [128, 256, 512, 512], numRepeat: 2,
        batchSize: 1, startWidth: width / 8, startHeight: height / 8)
      model.maxConcurrency = .limit(1)
      model.compile(inputs: input)
      try! graph.openStore(checkpoint, flags: .readOnly) {
        try! $0.read("decoder", model: model, strict: true)
      }.get()
      return model(inputs: input)[0].as(of: Float.self).rawValue.toCPU()
    }
  }

  static func save(_ pixels: Tensor<Float>, width: Int, height: Int, path: String) throws {
    var rgba = [PNG.RGBA<UInt8>](repeating: .init(0, 0, 0, 255), count: width * height)
    func byte(_ x: Float) -> UInt8 {
      precondition(x.isFinite, "Non-finite decoded image")
      return UInt8(min(max(Int((x + 1) * 127.5), 0), 255))
    }
    for y in 0..<height {
      for x in 0..<width {
        rgba[y * width + x] = .init(
          byte(pixels[0, 0, y, x]), byte(pixels[0, 1, y, x]), byte(pixels[0, 2, y, x]), 255)
      }
    }
    let image = PNG.Data.Rectangular(
      packing: rgba, size: (width, height),
      layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
    try image.compress(path: path, level: 4)
  }
}
