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
    let checkpoints = Flux3Checkpoints(environment: env)
    let output = env["FLUX3_OUTPUT"] ?? "/slow/Data/flux_3_image_klein_swift.png"
    let prompt =
      env["FLUX3_PROMPT"]
      ?? "A red ceramic teapot on a wooden table beside a window, soft morning light, a small white cup and a folded linen napkin, realistic photograph."
    let width = Int(env["FLUX3_WIDTH"] ?? "512")!
    let height = Int(env["FLUX3_HEIGHT"] ?? "512")!
    let steps = Int(env["FLUX3_STEPS"] ?? "16")!
    let seed = UInt64(env["FLUX3_SEED"] ?? "42")!
    let device = Int(env["FLUX3_DEVICE"] ?? "1")!
    precondition(width > 0 && height > 0 && width % 16 == 0 && height % 16 == 0)
    precondition(steps > 0)
    let start = Date()
    let tokenizer = try Flux3Tokenizer(path: checkpoints.tokenizer)
    let encoded = tokenizer.prompt(prompt)
    print("Swift-only FLUX.3 generation:", width, "x", height, "steps", steps, "seed", seed)
    print("Prompt:", prompt)
    print("Tokens:", encoded.valid, "real,", encoded.tokens.count, "padded")
    let graph = DynamicGraph()
    graph.maxConcurrency = .limit(1)
    let context = encode(
      graph: graph, device: device, checkpoint: checkpoints.text, encoded: encoded)
    let latent = denoise(
      graph: graph, device: device, checkpoint: checkpoints.dit, context: context,
      width: width, height: height, steps: steps, seed: seed)
    let decoded = decode(
      graph: graph, device: device, checkpoint: checkpoints.vae, latent: latent,
      width: width, height: height)
    try save(decoded, width: width, height: height, path: output)
    let elapsed = Date().timeIntervalSince(start)
    let metadata: [String: Any] = [
      "checkpoints": ["text": checkpoints.text, "dit": checkpoints.dit, "vae": checkpoints.vae],
      "tokenizer": checkpoints.tokenizer, "prompt": prompt, "width": width,
      "height": height, "steps": steps, "seed": seed, "seconds": elapsed,
      "python": false, "text_dtype": "float16", "dit_dtype": "bfloat16", "vae_dtype": "float32",
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

  static func denoise(
    graph: DynamicGraph, device: Int, checkpoint: String,
    context: Tensor<BFloat16>, width: Int, height: Int, steps: Int, seed: UInt64
  ) -> Tensor<Float> {
    graph.withNoGrad {
      let count = width / 16 * (height / 16)
      let length = context.shape[1]
      let schedule = flux3Schedule(steps: steps)
      let noise = Tensor<BFloat16>(from: flux3Noise(seed: seed, count: count))
      var current = graph.variable(noise.toGPU(device))
      let text = graph.variable(context.toGPU(device))
      let rotary = graph.variable(
        flux3Rotary(flux3IDs(text: length, height: height / 16, width: width / 16)).toGPU(device))
      func time(_ values: [Float]) -> DynamicGraph.Tensor<BFloat16> {
        graph.variable(
          Tensor<BFloat16>(from: flux3Time(values, precision: "bfloat16")).toGPU(device))
      }
      let t = time([schedule[0], 0])
      let r = time([schedule[1]])
      let (model, _, _) = Flux3DiT(
        BFloat16.self, textLength: length, imageLength: count, referenceLength: 0)
      model.maxConcurrency = .limit(1)
      model.compile(inputs: current, text, rotary, t, r)
      try! graph.openStore(checkpoint, flags: .readOnly) {
        try! $0.read("dit", model: model, strict: true)
      }.get()
      for i in 0..<steps {
        let begin = Date()
        let velocity = model(
          inputs: current, text, rotary, time([schedule[i], 0]), time([schedule[i + 1]]))[0].as(
            of: BFloat16.self)
        current = current + velocity * (schedule[i + 1] - schedule[i])
        // Synchronize and record a finite sample at every step.
        let sample = Tensor<Float>(from: current.rawValue.toCPU())
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
      return Tensor<Float>(from: current.rawValue.toCPU())
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
