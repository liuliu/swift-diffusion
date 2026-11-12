import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

setbuf(stdout, nil)
let env = ProcessInfo.processInfo.environment
let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let sys = Python.import("sys")
let site = Python.import("site")
for path in [String(site.getusersitepackages())!, "/usr/lib/python3/dist-packages"] {
  if !(Bool(sys.path.__contains__(path)) ?? false) { sys.path.append(path) }
}
sys.path.insert(0, directory + "/artifacts/python")
sys.path.insert(0, directory)
let reference = Python.import("reference")
let device = Int(env["FIBO_DEVICE"] ?? "1")!
let pythonDevice = Int(env["FIBO_REFERENCE_DEVICE"] ?? "0")!
let mode = CommandLine.arguments.dropFirst().first ?? "parity-vae"
let precision = env["FIBO_DTYPE"] ?? (mode == "export-dit-f16" ? "float16" : "float32")
// Keep Python's report metadata consistent with the export mode's default.
if mode == "export-dit-f16" { Python.import("os").environ.__setitem__("FIBO_DTYPE", precision) }
let referencePrecision =
  env["FIBO_REFERENCE_DTYPE"] ?? (precision == "float16" ? "float32" : precision)
if env["FIBO_VERBOSE"] == "1" { DynamicGraph.logLevel = .verbose }

func cpu(_ x: PythonObject) -> Tensor<Float> { try! Tensor<Float>(numpy: x) }
func load<T: TensorNumeric>(
  _ type: T.Type, _ bindings: [FiboBinding], _ state: PythonObject, vae: Bool = false
) {
  for (i, binding) in bindings.enumerated() {
    let value = Tensor<T>(
      from: cpu(
        reference.weight(
          state, binding.key, binding.transform, binding.heads, vae, binding.rowStart,
          binding.rowCount)))
    if binding.isBias {
      binding.model.bias.copy(from: value)
    } else if binding.parameter || binding.transform == "embedding" {
      binding.model.parameters.copy(from: value)
    } else {
      binding.model.weight.copy(from: value)
    }
    if i % 128 == 0 { print("Loaded", i + 1, "/", bindings.count) }
  }
}

func compare<T: TensorNumeric>(
  _ name: String, _ value: DynamicGraph.Tensor<T>, _ expected: PythonObject, tolerance: Double
) {
  let tensor = Tensor<Float>(from: value.rawValue.toCPU())
  precondition(
    Bool(reference.metrics(name, tensor.makeNumpyArray(), expected, tolerance))!,
    "Parity failed: \(name)")
}

func normalizeVAE(_ moments: Tensor<Float>, mean: Tensor<Float>, invstd: Tensor<Float>) -> Tensor<
  Float
> {
  let h = moments.shape[2]
  let w = moments.shape[3]
  var result = Tensor<Float>(.CPU, .NCHW(1, 32, h, w))
  for c in 0..<32 {
    for y in 0..<h {
      for x in 0..<w {
        let packed = c * 4 + (y % 2) * 2 + x % 2
        result[0, c, y, x] = (moments[0, c, y, x] - mean[packed]) * invstd[packed]
      }
    }
  }
  return result
}

func runVAE() {
  let state = reference.State("vae")
  for (index, shape) in [(32, 48), (64, 64), (96, 128)].enumerated() {
    if let selected = env["FIBO_CASE"], Int(selected) != index { continue }
    let (h, w) = shape
    let pack = reference.vae_reference(h, w, 42 + index, pythonDevice)
    let graph = DynamicGraph()
    graph.maxConcurrency = .limit(1)
    graph.withNoGrad {
      let image = graph.variable(cpu(pack["image"]).toGPU(device)).reshaped(.NCHW(1, 3, h, w))
      let (encoderBindings, encoder) = FiboVAEEncoder(
        channels: [128, 256, 512, 512],
        numRepeat: 2, batchSize: 1, startWidth: w / 8, startHeight: h / 8)
      encoder.compile(inputs: image)
      load(Float.self, encoderBindings, state, vae: true)
      let moments = encoder(inputs: image)[0].as(of: Float.self)
      compare("vae/case\(index)/encoder", moments, pack["moments"], tolerance: 0.001)
      let normalized = normalizeVAE(
        moments.rawValue.toCPU(), mean: cpu(pack["mean"]), invstd: cpu(pack["invstd"]))
      precondition(
        Bool(
          reference.metrics(
            "vae/case\(index)/normalized", normalized.makeNumpyArray(), pack["normalized"], 0.001))!
      )
      let latent = graph.variable(cpu(pack["latent"]).toGPU(device)).reshaped(
        .NCHW(1, 32, h / 8, w / 8))
      let (decoderBindings, decoder) = FiboVAEDecoder(
        channels: [128, 256, 512, 512],
        numRepeat: 2, batchSize: 1, startWidth: w / 8, startHeight: h / 8)
      decoder.compile(inputs: latent)
      load(Float.self, decoderBindings, state, vae: true)
      compare(
        "vae/case\(index)/decoder", decoder(inputs: latent)[0].as(of: Float.self), pack["decoded"],
        tolerance: 0.001)
      reference.check_keys(state, (encoderBindings + decoderBindings).map { $0.key }, "vae")
    }
  }
  if !Bool(reference.finish("vae_float32"))! { exit(1) }
}

func loadVision<T: TensorNumeric>(
  _ type: T.Type, _ bindings: [FiboVisionBinding], _ state: PythonObject
) {
  var keys = Set<String>()
  for binding in bindings {
    let kind: String
    switch binding.kind {
    case .affine: kind = "affine"
    case .embedding: kind = "embedding"
    case .patch: kind = "patch"
    case .query: kind = "query"
    case .key: kind = "key"
    case .value: kind = "value"
    }
    let weight = Tensor<T>(from: cpu(reference.vision_weight(state, binding.prefix, kind, false)))
    keys.insert(binding.prefix + ".weight")
    if binding.kind == .embedding {
      binding.model.parameters.copy(from: weight)
    } else {
      binding.model.weight.copy(from: weight)
      let bias = Tensor<T>(from: cpu(reference.vision_weight(state, binding.prefix, kind, true)))
      if let parameter = binding.biasParameter {
        parameter.parameters.copy(from: bias)
      } else {
        binding.model.bias.copy(from: bias)
      }
      keys.insert(binding.prefix + ".bias")
    }
  }
  reference.check_keys(state, Array(keys), "vision")
}

func evaluateText<T: TensorNumeric>(
  _ type: T.Type, index: Int, pack: PythonObject, expected: PythonObject
) -> [Tensor<Float>] {
  let state = reference.State("text_encoder")
  let rawIDs = try! Tensor<Int32>(numpy: pack["tokens"])
  let length = rawIDs.shape[0]
  let spans = [[Int]](pack["spans"])!.map {
    FiboVisualSpan(start: $0[0], height: $0[1], width: $0[2])
  }
  let positions = fiboMultimodalPositions(length: length, visual: spans)
  precondition(
    Bool(
      reference.metrics(
        "text/case\(index)/positions",
        positions.makeNumpyArray(), pack["positions"], 0))!)
  let graph = DynamicGraph()
  graph.maxConcurrency = .limit(1)
  print("Text case", index, "length", length, "dtype", precision)
  return graph.withNoGrad {
    let tokens = graph.variable(rawIDs.reshaped(format: .NHWC, shape: rawIDs.shape).toGPU(device))
    let rotary = graph.variable(fiboMultimodalRotary(type, positions).toGPU(device))
    let mask = graph.variable(cpu(pack["mask"]).toGPU(device))
    var inputs: [DynamicGraph.AnyTensor] =
      [tokens, rotary] + (T.dataType == .Float32 ? [mask] : [])
    if !spans.isEmpty {
      let grids = [[Int]](pack["grids"])!.map { (t: $0[0], h: $0[1], w: $0[2]) }
      let positions = fiboVisionPositions(grids)
      let patches = graph.variable(fiboRoundedInput(type, cpu(pack["patches"])).toGPU(device))
      let ids = graph.variable(positions.ids.toGPU(device))
      let weights = graph.variable(positions.weights.toGPU(device))
      let rot = graph.variable(positions.rotary.toGPU(device))
      let diagnostic = env["FIBO_VISION_TRACE"] == "1"
      let (vision, bindings) = FiboVisionModel(
        type, grids: grids, device: device, diagnostic: diagnostic)
      vision.compile(inputs: patches, ids, weights, rot)
      loadVision(type, bindings, state)
      let results = vision(inputs: patches, ids, weights, rot)
      if diagnostic {
        for i in 0..<26 {
          compare(
            "vision/stage\(i)", results[i + 4].as(of: T.self),
            expected["vision_stage_\(i)"], tolerance: 0.05)
        }
      }
      for i in 0..<4 {
        compare(
          "vision/\(precision)/case\(index)/output\(i)", results[i].as(of: T.self),
          expected["vision_\(i)"], tolerance: T.dataType == .Float32 ? 0.001 : 0.05)
      }
      inputs += results.prefix(4).map { $0.as(of: T.self) }
    }
    let (model, bindings, names) = FiboText(type, length: length, device: device, visual: spans)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: inputs)
    reference.check_keys(state, bindings.map { $0.key }, "text")
    load(type, bindings, state)
    let results = model(inputs: tokens, Array(inputs.dropFirst()))
    for (i, name) in names.enumerated() {
      compare(
        "text/\(precision)/case\(index)/\(name)", results[i].as(of: T.self),
        expected[name], tolerance: T.dataType == .Float32 ? 0.001 : 0.05)
    }
    return results.suffix(6).map { Tensor<Float>(from: $0.as(of: T.self).rawValue.toCPU()) }
  }
}

func runText<T: TensorNumeric>(_ type: T.Type) {
  for index in 0..<4 {
    if let selected = env["FIBO_CASE"], Int(selected) != index { continue }
    let pack = reference.text_case(index)
    let expected = reference.text_reference(pack, referencePrecision, pythonDevice)
    _ = evaluateText(type, index: index, pack: pack, expected: expected)
  }
  if !Bool(reference.finish("text_\(precision)"))! { exit(1) }
}

func runDiT<T: TensorNumeric>(_ type: T.Type, exportPath: String? = nil) {
  let state = reference.State("transformer")
  for index in 0..<(precision == "float16" ? 4 : 3) {
    if let selected = env["FIBO_CASE"], Int(selected) != index { continue }
    let pack = reference.dit_case(index, precision)
    let expected = reference.dit_reference(pack, referencePrecision, pythonDevice)
    let h = Int(pack["height"])!
    let w = Int(pack["width"])!
    let textLength = Int(pack["text_length"])!
    let gist = Int(pack["gist"])!
    let imageLength = h * w / 4
    let coordinates = [[Int]](pack["coordinates"])!
    let gistCoordinates = [[Int]](pack["gist_coordinates"])!
    let referenceLength = coordinates.count - imageLength
    let graph = DynamicGraph()
    graph.maxConcurrency = .limit(1)
    print(
      "DiT case", index, "image", imageLength, "references", referenceLength, "gist", gist,
      precision)
    graph.withNoGrad {
      let x = graph.variable(Tensor<T>(from: cpu(pack["packed"])).toGPU(device))
      var inputs: [DynamicGraph.AnyTensor] = [x]
      for i in 0..<6 {
        inputs.append(graph.variable(Tensor<T>(from: cpu(pack["context"][0, i])).toGPU(device)))
      }
      inputs.append(graph.variable(fiboRotary(coordinates).toGPU(device)))
      inputs.append(graph.variable(fiboRotary(gistCoordinates + coordinates).toGPU(device)))
      inputs.append(
        graph.variable(fiboRoundedInput(type, fiboTime(Float(pack["timestep"])!)).toGPU(device)))
      let visualIndices = (0..<coordinates.count).map { Int32($0 < imageLength ? 0 : 1) }
      let indices = Array(repeating: Int32(1), count: gist) + visualIndices
      inputs.append(
        graph.variable(Tensor<Int32>(visualIndices, .CPU, .C(visualIndices.count)).toGPU(device)))
      inputs.append(graph.variable(Tensor<Int32>(indices, .CPU, .C(indices.count)).toGPU(device)))
      let (model, bindings, names) = FiboDiT(
        type, imageLength: imageLength, referenceLength: referenceLength,
        textLength: textLength, gistCount: gist, device: device)
      model.maxConcurrency = .limit(1)
      model.compile(inputs: inputs)
      reference.check_dit_bindings(
        state, bindings.map { $0.key }, bindings.map { $0.rowStart }, bindings.map { $0.rowCount })
      load(type, bindings, state)
      let results = model(inputs: x, Array(inputs.dropFirst()))
      for (i, name) in names.enumerated() {
        compare(
          "dit/\(precision)/case\(index)/\(name)", results[i].as(of: T.self),
          expected[name], tolerance: T.dataType == .Float32 ? 0.001 : 0.05)
      }
      if index == 3, let exportPath {
        // All four cases, including every layer, must pass before writing.
        if !Bool(reference.finish("dit_\(precision)"))! { exit(1) }
        reference.release()
        exportDiTCheckpoint(
          graph: graph, model: model, inputs: inputs,
          expected: Tensor<Float>(from: results.last!.as(of: T.self).rawValue.toCPU()),
          imageLength: imageLength, referenceLength: referenceLength,
          textLength: textLength, gistCount: gist, path: exportPath)
      }
    }
  }
  if exportPath == nil, !Bool(reference.finish("dit_\(precision)"))! { exit(1) }
}

func exportDiTCheckpoint(
  graph: DynamicGraph, model: Model, inputs: [DynamicGraph.AnyTensor],
  expected: Tensor<Float>, imageLength: Int, referenceLength: Int,
  textLength: Int, gistCount: Int, path: String
) {
  let staging = path + ".partial"
  var keys = [String]()
  print("Writing FP16 DiT:", staging)
  try! graph.openStore(staging) { store in
    try! store.write("dit", model: model, strict: true) { name, tensor in
      precondition(tensor.dataType == .Float16)
      keys.append(name)
      return .continue(name)
    }
  }.get()
  let inventory = reference.check_dit_store(staging, keys)

  // The PyTorch model has been released; reuse its GPU for a fresh reload.
  let reloadGraph = DynamicGraph()
  reloadGraph.maxConcurrency = .limit(1)
  reloadGraph.withNoGrad {
    precondition(inputs.count == 12)
    let copied: [DynamicGraph.AnyTensor] = inputs.enumerated().map { index, input in
      if index == 7 || index == 8 {
        return reloadGraph.variable(input.as(of: Float.self).rawValue.toCPU().toGPU(pythonDevice))
      } else if index >= 10 {
        return reloadGraph.variable(input.as(of: Int32.self).rawValue.toCPU().toGPU(pythonDevice))
      }
      return reloadGraph.variable(input.as(of: Float16.self).rawValue.toCPU().toGPU(pythonDevice))
    }
    let (fresh, _, _) = FiboDiT(
      Float16.self, imageLength: imageLength, referenceLength: referenceLength,
      textLength: textLength, gistCount: gistCount, device: pythonDevice, diagnostic: false)
    fresh.maxConcurrency = .limit(1)
    fresh.compile(inputs: copied)
    try! reloadGraph.openStore(staging, flags: .readOnly) { store in
      try! store.read("dit", model: fresh, strict: true)
    }.get()
    let output = fresh(inputs: copied[0], Array(copied.dropFirst()))[0].as(of: Float16.self)
    compare(
      "checkpoint/dit/reload", output, expected.makeNumpyArray(), tolerance: 1e-6)
  }
  if !Bool(reference.finish("export_dit_float16"))! { exit(1) }
  precondition(rename(staging, path) == 0, "Cannot finalize checkpoint: \(path)")
  reference.record_dit_export(path, inventory)
  print("FP16 DiT export complete:", path)
}

func packLatents(_ latents: [Tensor<Float>]) -> Tensor<Float> {
  let count = latents.reduce(0) { $0 + $1.shape[2] * $1.shape[3] / 4 }
  var result = Tensor<Float>(.CPU, .WC(count, 128))
  var offset = 0
  for latent in latents {
    let h = latent.shape[2]
    let w = latent.shape[3]
    for y in 0..<h {
      for x in 0..<w {
        for c in 0..<32 {
          result[offset + (y / 2) * (w / 2) + x / 2, c * 4 + (y % 2) * 2 + x % 2] =
            latent[0, c, y, x]
        }
      }
    }
    offset += h * w / 4
  }
  return result
}

func runPipeline<T: TensorNumeric>(_ type: T.Type) {
  let tolerance: Double = T.dataType == .Float32 ? 0.001 : 0.05
  let pack = reference.pipeline_reference(pythonDevice)
  let positive = evaluateText(
    type, index: 2, pack: pack["positive_pack"], expected: pack["positive"])
  let guidance = Float(pack["guidance"])!
  let negative =
    guidance > 1
    ? evaluateText(type, index: 4, pack: pack["negative_pack"], expected: pack["negative"])
    : []
  let mean = cpu(pack["mean"])
  let invstd = cpu(pack["invstd"])
  let vaeState = reference.State("vae")
  let referenceLatent: Tensor<Float> = {
    let graph = DynamicGraph()
    return graph.withNoGrad {
      let image = graph.variable(cpu(pack["image"]).toGPU(device))
      let (bindings, encoder) = FiboVAEEncoder(
        channels: [128, 256, 512, 512],
        numRepeat: 2, batchSize: 1, startWidth: 12, startHeight: 8)
      encoder.compile(inputs: image)
      load(Float.self, bindings, vaeState, vae: true)
      let moments = encoder(inputs: image)[0].as(of: Float.self)
      compare("pipeline/encoder", moments, pack["moments"], tolerance: 0.001)
      let normalized = normalizeVAE(moments.rawValue.toCPU(), mean: mean, invstd: invstd)
      precondition(
        Bool(
          reference.metrics(
            "pipeline/reference_latent",
            normalized.makeNumpyArray(), pack["references"][0], 0.001))!)
      return normalized
    }
  }()
  var latent = cpu(pack["initial"])
  let steps = Int(pack["steps"])!
  // Native FlowMatchEuler schedule: dynamic exponential shift at 24 image tokens.
  let slope = (1.15 - 0.5) / Double(4096 - 256)
  let mu = slope * 24 + 0.5 - slope * 256
  let shifted = Float(exp(mu))
  var sigmas = Tensor<Float>(.CPU, .C(steps + 1))
  for i in 0..<steps {
    let sigma = Float(1.0 + (1.0 / Double(steps) - 1.0) * Double(i) / Double(steps - 1))
    sigmas[i] = shifted / (shifted + (1 / sigma - 1))
  }
  sigmas[steps] = 0
  precondition(
    Bool(reference.metrics("pipeline/sigmas", sigmas.makeNumpyArray(), pack["sigmas"], 1e-6))!)
  let graph = DynamicGraph()
  graph.maxConcurrency = .limit(1)
  graph.withNoGrad {
    let state = reference.State("transformer")
    let coordinates = (0..<2).flatMap { plane in
      (0..<4).flatMap { y in (0..<6).map { [513 + plane, y, $0] } }
    }
    func inputs(_ context: [Tensor<Float>], _ t: Float) -> [DynamicGraph.AnyTensor] {
      let gist = fiboGistCount(context[0].shape[0])
      let positions = (0..<gist).map { [$0 + 1, 0, 0] }
      let selection = Array(repeating: Int32(0), count: 24) + Array(repeating: Int32(1), count: 24)
      let allSelection = Array(repeating: Int32(1), count: gist) + selection
      var result: [DynamicGraph.AnyTensor] = [
        graph.variable(
          fiboRoundedInput(type, packLatents([latent, referenceLatent])).toGPU(device))
      ]
      result += context.map { graph.variable(fiboRoundedInput(type, $0).toGPU(device)) }
      result += [
        graph.variable(fiboRotary(coordinates).toGPU(device)),
        graph.variable(fiboRotary(positions + coordinates).toGPU(device)),
        graph.variable(fiboRoundedInput(type, fiboTime(t)).toGPU(device)),
        graph.variable(Tensor<Int32>(selection, .CPU, .C(selection.count)).toGPU(device)),
        graph.variable(Tensor<Int32>(allSelection, .CPU, .C(allSelection.count)).toGPU(device)),
      ]
      return result
    }
    func denoiser(_ context: [Tensor<Float>], shared: Model? = nil) -> Model {
      let (model, bindings, _) = FiboDiT(
        type, imageLength: 24, referenceLength: 24,
        textLength: context[0].shape[0], gistCount: fiboGistCount(context[0].shape[0]),
        device: device, diagnostic: false)
      model.maxConcurrency = .limit(1)
      model.compile(inputs: inputs(context, 1))
      if let shared {
        model.parameters.share(from: shared.parameters)
      } else {
        load(type, bindings, state)
      }
      return model
    }
    let positiveModel = denoiser(positive)
    let negativeModel = negative.isEmpty ? nil : denoiser(negative, shared: positiveModel)
    var call = 0
    for step in 0..<steps {
      // Match the pipeline's FP32 sigma -> timestep -> normalized timestep round trip.
      let t = (sigmas[step] * 1000) / 1000
      let positiveInputs = inputs(positive, t)
      let output = positiveModel(inputs: positiveInputs[0], Array(positiveInputs.dropFirst()))[0]
        .as(of: T.self)
      let expected = reference.array(
        reference.pack_latent(Python.import("torch").from_numpy(pack["velocities"][call])))
      compare("pipeline/step\(step)/positive_velocity", output, expected, tolerance: tolerance)
      var velocity = Tensor<Float>(from: output.rawValue.toCPU())
      call += 1
      if let negativeModel {
        let negativeInputs = inputs(negative, t)
        let uncond = negativeModel(inputs: negativeInputs[0], Array(negativeInputs.dropFirst()))[0]
          .as(of: T.self)
        let expected = reference.array(
          reference.pack_latent(Python.import("torch").from_numpy(pack["velocities"][call])))
        compare("pipeline/step\(step)/negative_velocity", uncond, expected, tolerance: tolerance)
        let u = Tensor<Float>(from: uncond.rawValue.toCPU())
        for row in 0..<24 {
          for c in 0..<128 {
            velocity[row, c] = u[row, c] + guidance * (velocity[row, c] - u[row, c])
          }
        }
        call += 1
      }
      let delta = sigmas[step + 1] - sigmas[step]
      for c in 0..<32 {
        for y in 0..<8 {
          for x in 0..<12 {
            let v = velocity[(y / 2) * 6 + x / 2, c * 4 + (y % 2) * 2 + x % 2]
            latent[0, c, y, x] = latent[0, c, y, x] + delta * v
          }
        }
      }
      precondition(
        Bool(
          reference.metrics(
            "pipeline/step\(step)/latent",
            latent.makeNumpyArray(), pack["latents"][step], tolerance))!)
    }
  }
  for c in 0..<32 {
    for y in 0..<8 {
      for x in 0..<12 {
        let channel = c * 4 + (y % 2) * 2 + x % 2
        latent[0, c, y, x] = latent[0, c, y, x] / invstd[channel] + mean[channel]
      }
    }
  }
  graph.withNoGrad {
    let input = graph.variable(latent.toGPU(device))
    let (bindings, decoder) = FiboVAEDecoder(
      channels: [128, 256, 512, 512],
      numRepeat: 2, batchSize: 1, startWidth: 12, startHeight: 8)
    decoder.compile(inputs: input)
    load(Float.self, bindings, vaeState, vae: true)
    let output = decoder(inputs: input)[0].as(of: Float.self)
    var pixels = output.rawValue.toCPU()
    // Record both final image metrics before enforcing the unchanged threshold.
    _ = reference.metrics("pipeline/decoded", pixels.makeNumpyArray(), pack["decoded"], tolerance)
    for c in 0..<3 {
      for y in 0..<64 {
        for x in 0..<96 { pixels[0, c, y, x] = min(1, max(0, pixels[0, c, y, x] / 2 + 0.5)) }
      }
    }
    _ = reference.metrics("pipeline/pixels", pixels.makeNumpyArray(), pack["pixels"], tolerance)
  }
  if !Bool(reference.finish("pipeline_\(precision)", "float32"))! { exit(1) }
}

switch mode {
case "export-dit-f16":
  precondition(precision == "float16" && referencePrecision == "float32")
  precondition(env["FIBO_CASE"] == nil, "Export requires all four parity cases")
  let variant = String(reference.VARIANT)!
  let filename = variant == "dev" ? "fibo_2_dev_f16.ckpt" : "fibo_2_edit_turbo_f16.ckpt"
  let path = env["FIBO_EXPORT_PATH"] ?? "/slow/Data/" + filename
  precondition(
    env["FIBO_REPLACE_CHECKPOINT"] == "1" || !FileManager.default.fileExists(atPath: path),
    "Checkpoint already exists: \(path); use FIBO_REPLACE_CHECKPOINT=1 to replace after validation")
  precondition(
    !FileManager.default.fileExists(atPath: path + ".partial"),
    "Incomplete checkpoint already exists: \(path).partial")
  runDiT(Float16.self, exportPath: path)
case "parity-vae": runVAE()
case "parity-text":
  if precision == "float32" {
    runText(Float.self)
  } else if precision == "bfloat16" {
    runText(BFloat16.self)
  } else if precision == "float16" {
    runText(Float16.self)
  } else {
    fatalError("Use FIBO_DTYPE=float32, float16, or bfloat16")
  }
case "parity-dit":
  if precision == "float32" {
    runDiT(Float.self)
  } else if precision == "bfloat16" {
    runDiT(BFloat16.self)
  } else if precision == "float16" {
    runDiT(Float16.self)
  } else {
    fatalError("Use FIBO_DTYPE=float32, float16, or bfloat16")
  }
case "parity-pipeline":
  if precision == "float32" {
    runPipeline(Float.self)
  } else if precision == "float16" {
    runPipeline(Float16.self)
  } else {
    fatalError("The pipeline comparison supports float32 and float16.")
  }
default:
  fatalError(
    "Usage: bash examples/fibo2/run.sh [parity-vae|parity-text|parity-dit|parity-pipeline|export-dit-f16]"
  )
}
