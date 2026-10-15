import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

setbuf(stdout, nil)
let env = ProcessInfo.processInfo.environment
let sys = Python.import("sys")
let site = Python.import("site")
for path in [String(site.getusersitepackages())!, "/usr/lib/python3/dist-packages"] {
  if !(Bool(sys.path.__contains__(path)) ?? false) { sys.path.append(path) }
}
let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
sys.path.insert(0, directory)
let reference = Python.import("reference")
let torch = Python.import("torch")
let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)
let device = Int(env["FLUX3_DEVICE"] ?? "1")!
let pythonDevice = Int(env["FLUX3_REFERENCE_DEVICE"] ?? "0")!
let precision = env["FLUX3_DTYPE"] ?? "float32"

func cpu(_ value: PythonObject) -> Tensor<Float> { try! Tensor<Float>(numpy: value) }

func load<T: TensorNumeric>(_ type: T.Type, _ bindings: [Flux3Binding], _ state: PythonObject) {
  for binding in bindings {
    let value = cpu(reference.weight(state, binding.key, binding.transform, binding.heads))
    if binding.isBias {
      binding.model.bias.copy(from: Tensor<T>(from: value))
    } else if binding.transform == "float32" {
      binding.model.weight.copy(from: value)
    } else if binding.transform == "embedding" {
      binding.model.parameters.copy(from: Tensor<T>(from: value))
    } else {
      binding.model.weight.copy(from: Tensor<T>(from: value))
    }
  }
}

@discardableResult
func compare<T: TensorNumeric>(
  _ label: String, _ actual: DynamicGraph.Tensor<T>, _ expected: PythonObject, tolerance: Double
) -> Bool {
  let value = Tensor<Float>(from: actual.rawValue.toCPU())
  let ok = Bool(reference.metrics(label, value.makeNumpyArray(), expected, tolerance))!
  precondition(ok, "Numeric parity failed: \(label)")
  return ok
}

func runText<T: TensorNumeric>(_ type: T.Type, name: String) {
  let state = reference.state("text")
  let complete = true
  let cases = ["random", "A red ceramic teapot on a wooden table."]
  for (index, prompt) in cases.enumerated() {
    let pack = reference.text_case(prompt, index == 0 ? 32 : 80, 42)
    let expected = reference.text_reference(state, pack, name, pythonDevice, complete)
    let ids = try! Tensor<Int32>(numpy: pack["tokens"])
    let positions = try! Tensor<Int32>(numpy: pack["positions"])
    graph.withNoGrad {
      let tokens = graph.variable(ids.reshaped(format: .NHWC, shape: ids.shape).toGPU(device))
      let rotary = graph.variable(flux3TextRotary(type, positions: positions).toGPU(device))
      let mask = graph.variable(
        cpu(pack["mask"]).reshaped(.NHWC(1, 1, ids.shape[0], ids.shape[0])).toGPU(device))
      let (model, bindings, names) = Flux3Text(
        type, length: ids.shape[0], validLength: Int(pack["attention"].sum())!,
        completeEncoder: complete)
      model.maxConcurrency = .limit(1)
      let inputs: [DynamicGraph.AnyTensor] =
        [tokens, rotary] + (T.dataType == .Float32 ? [mask] : [])
      model.compile(inputs: inputs)
      reference.check_keys(state, bindings.map { $0.key }, "text", complete)
      load(type, bindings, state)
      let results = model(inputs: tokens, Array(inputs.dropFirst()))
      for (i, output) in results.enumerated() {
        compare(
          "text/\(name)/case\(index)/\(names[i])", output.as(of: T.self), expected[names[i]],
          tolerance: name == "float32" ? 0.001 : 0.05)
      }
      if exporting && index == cases.count - 1 {
        let (fresh, _, _) = Flux3Text(
          type, length: ids.shape[0],
          validLength: Int(pack["attention"].sum())!, completeEncoder: complete)
        exportAndVerify(
          type, key: "text_model", model: model, fresh: fresh,
          inputs: inputs, expected: results)
      }
    }
  }
}

func runDiT<T: TensorNumeric>(_ type: T.Type, name: String) {
  let state = reference.state("dit")
  // Different aspect ratios, no reference, multiple references, and destination times.
  let cases: [(Int, Int, [[Int]], Float, Float)] = [
    (2, 3, [], 0.75, 0.5), (3, 2, [[2, 2], [1, 3]], 0.625, 0.12345),
  ]
  for (index, spec) in cases.enumerated() {
    let textLength = 16
    let pack = reference.dit_case(
      textLength, spec.0, spec.1, spec.2, 42 + index, name, spec.3, spec.4)
    let expected = reference.dit_reference(state, pack, name, pythonDevice)
    let refLength = spec.2.reduce(0) { $0 + $1[0] * $1[1] }
    graph.withNoGrad {
      let x = graph.variable(Tensor<T>(from: cpu(pack["x"])).toGPU(device))
        .reshaped(.HWC(1, spec.0 * spec.1, 128))
      let context = graph.variable(Tensor<T>(from: cpu(pack["context"])).toGPU(device))
        .reshaped(.HWC(1, textLength, 20_480))
      let rot = graph.variable(flux3Rotary(cpu(pack["ids"])).toGPU(device))
      let time = graph.variable(
        Tensor<T>(from: flux3Time([Float(pack["time"])!, 0], precision: name)).toGPU(device))
      let target = graph.variable(
        Tensor<T>(from: flux3Time([Float(pack["target"])!], precision: name)).toGPU(device))
      var inputs: [DynamicGraph.AnyTensor] = [x, context]
      if refLength > 0 {
        inputs.append(
          graph.variable(Tensor<T>(from: cpu(pack["reference"])).toGPU(device))
            .reshaped(.HWC(1, refLength, 128)))
      }
      inputs += [rot, time, target]
      let (model, bindings, names) = Flux3DiT(
        type, textLength: textLength,
        imageLength: spec.0 * spec.1, referenceLength: refLength, debug: true)
      model.maxConcurrency = .limit(1)
      model.compile(inputs: inputs)
      reference.check_keys(state, bindings.map { $0.key }, "dit", refLength > 0)
      print("Loading DiT weights:", name, "case", index)
      load(type, bindings, state)
      let results = model(inputs: x, Array(inputs.dropFirst()))
      for (i, output) in results.enumerated() {
        compare(
          "dit/\(name)/case\(index)/\(names[i])", output.as(of: T.self), expected[names[i]],
          tolerance: name == "float32" ? 0.001 : 0.05)
      }
      if exporting && index == cases.count - 1 {
        let (fresh, _, _) = Flux3DiT(
          type, textLength: textLength,
          imageLength: spec.0 * spec.1, referenceLength: refLength, debug: true)
        exportAndVerify(
          type, key: "dit", model: model, fresh: fresh,
          inputs: inputs, expected: results)
      }
    }
  }
}

func runVAE() {
  let state = reference.state("vae")
  for (height, width) in [(32, 48), (64, 64)] {
    let pack = reference.vae_reference(state, height, width, 42, pythonDevice)
    graph.withNoGrad {
      let image = graph.variable(cpu(pack["image"]).toGPU(device)).reshaped(
        .NCHW(1, 3, height, width))
      let (encoderReader, encoder) = Flux3VAEEncoder(
        channels: [128, 256, 512, 512],
        numRepeat: 2, batchSize: 1, startWidth: width / 8, startHeight: height / 8)
      encoder.compile(inputs: image)
      load(Float.self, encoderReader, state)
      let moments = encoder(inputs: image)[0].as(of: Float.self)
      compare("vae/\(height)x\(width)/encoder", moments, pack["moments"], tolerance: 0.001)
      let normalized = reference.normalize_moments(moments.rawValue.toCPU().makeNumpyArray(), state)
      precondition(
        Bool(
          reference.metrics(
            "vae/\(height)x\(width)/encode_normalized", normalized,
            pack["encoded_tokens"], 0.001))!)
      let latent = graph.variable(cpu(pack["latent"]).toGPU(device))
        .reshaped(.NCHW(1, 32, height / 8, width / 8))
      let (decoderReader, decoder) = Flux3VAEDecoder(
        channels: [128, 256, 512, 512],
        numRepeat: 2, batchSize: 1, startWidth: width / 8, startHeight: height / 8)
      decoder.compile(inputs: latent)
      load(Float.self, decoderReader, state)
      compare(
        "vae/\(height)x\(width)/decoder", decoder(inputs: latent)[0].as(of: Float.self),
        pack["decoded"], tolerance: 0.001)
      let denormalized = reference.denormalize_tokens(pack["tokens"], state, height, width)
      let tokens = graph.variable(cpu(denormalized).toGPU(device))
        .reshaped(.NCHW(1, 32, height / 8, width / 8))
      compare(
        "vae/\(height)x\(width)/decode_normalized", decoder(inputs: tokens)[0].as(of: Float.self),
        pack["decoded_tokens"], tolerance: 0.001)
      if exporting && height == 64 {
        let (_, freshEncoder) = Flux3VAEEncoder(
          channels: [128, 256, 512, 512],
          numRepeat: 2, batchSize: 1, startWidth: width / 8, startHeight: height / 8)
        exportAndVerify(
          Float.self, key: "encoder", model: encoder, fresh: freshEncoder,
          inputs: [image], expected: [moments])
        let (_, freshDecoder) = Flux3VAEDecoder(
          channels: [128, 256, 512, 512],
          numRepeat: 2, batchSize: 1, startWidth: width / 8, startHeight: height / 8)
        exportAndVerify(
          Float.self, key: "decoder", model: decoder, fresh: freshDecoder,
          inputs: [latent], expected: decoder(inputs: latent))
      }
    }
  }
}

func swiftTextFeatures(_ pack: PythonObject) -> Tensor<Float> {
  let state = reference.state("text")
  let ids = try! Tensor<Int32>(numpy: pack["tokens"])
  let positions = try! Tensor<Int32>(numpy: pack["positions"])
  return graph.withNoGrad {
    let tokens = graph.variable(ids.reshaped(format: .NHWC, shape: ids.shape).toGPU(device))
    let rotary = graph.variable(flux3TextRotary(Float.self, positions: positions).toGPU(device))
    let mask = graph.variable(
      cpu(pack["mask"]).reshaped(.NHWC(1, 1, ids.shape[0], ids.shape[0])).toGPU(device))
    let (model, bindings, _) = Flux3Text(Float.self, length: ids.shape[0])
    model.compile(inputs: tokens, rotary, mask)
    load(Float.self, bindings, state)
    let outputs = model(inputs: tokens, rotary, mask)
    let arrays = outputs.map { $0.as(of: Float.self).rawValue.toCPU().makeNumpyArray() }
    return cpu(reference.concatenate_features(arrays))
  }
}

func runPipeline() {
  print(
    "Full pipeline FP32 reference: padded prompt, reference image, two denoising steps, decoder")
  let pack = reference.pipeline_case(pythonDevice)
  let height = Int(pack["height"])!
  let width = Int(pack["width"])!
  let context = swiftTextFeatures(pack["text"])
  precondition(
    Bool(reference.metrics("pipeline/context", context.makeNumpyArray(), pack["context"], 0.001))!)
  let vaeState = reference.state("vae")
  let referenceTokens: Tensor<Float> = graph.withNoGrad {
    let image = graph.variable(cpu(pack["reference_image"]).toGPU(device)).reshaped(
      .NCHW(1, 3, 64, 64))
    let (reader, encoder) = Flux3VAEEncoder(
      channels: [128, 256, 512, 512], numRepeat: 2,
      batchSize: 1, startWidth: 8, startHeight: 8)
    encoder.compile(inputs: image)
    load(Float.self, reader, vaeState)
    let moments = encoder(inputs: image)[0].as(of: Float.self).rawValue.toCPU()
    return cpu(reference.normalize_moments(moments.makeNumpyArray(), vaeState))
  }
  precondition(
    Bool(
      reference.metrics(
        "pipeline/reference_tokens", referenceTokens.makeNumpyArray(),
        pack["reference_tokens"], 0.001))!)
  let state = reference.state("dit")
  let finalTokens: Tensor<Float> = graph.withNoGrad {
    let imageLength = height / 16 * (width / 16)
    let x = graph.variable(cpu(pack["initial"]).toGPU(device)).reshaped(.HWC(1, imageLength, 128))
    let text = graph.variable(context.toGPU(device)).reshaped(.HWC(1, context.shape[0], 20_480))
    let refs = graph.variable(referenceTokens.toGPU(device)).reshaped(
      .HWC(1, referenceTokens.shape[0], 128))
    let rotary = graph.variable(flux3Rotary(cpu(pack["ids"])).toGPU(device))
    let schedule = cpu(pack["schedule"])
    let t = graph.variable(flux3Time([schedule[0], 0]).toGPU(device))
    let r = graph.variable(flux3Time([schedule[1]]).toGPU(device))
    let (model, bindings, _) = Flux3DiT(
      Float.self, textLength: context.shape[0],
      imageLength: imageLength, referenceLength: referenceTokens.shape[0])
    model.compile(inputs: x, text, refs, rotary, t, r)
    load(Float.self, bindings, state)
    var current = x
    for i in 0..<(schedule.shape[0] - 1) {
      let time = graph.variable(flux3Time([schedule[i], 0]).toGPU(device))
      let target = graph.variable(flux3Time([schedule[i + 1]]).toGPU(device))
      let velocity = model(inputs: current, text, refs, rotary, time, target)[0].as(of: Float.self)
      compare(
        "pipeline/velocity_\(i)", velocity.reshaped(.NC(imageLength, 128)),
        pack["velocities"][i], tolerance: 0.002)
      current = current + velocity * (schedule[i + 1] - schedule[i])
      compare(
        "pipeline/latent_\(i)", current.reshaped(.NC(imageLength, 128)),
        pack["latents"][i], tolerance: 0.002)
    }
    return current.reshaped(.NC(imageLength, 128)).rawValue.toCPU()
  }
  graph.withNoGrad {
    let latent = cpu(
      reference.denormalize_tokens(finalTokens.makeNumpyArray(), vaeState, height, width))
    let input = graph.variable(latent.toGPU(device)).reshaped(.NCHW(1, 32, height / 8, width / 8))
    let (reader, decoder) = Flux3VAEDecoder(
      channels: [128, 256, 512, 512], numRepeat: 2,
      batchSize: 1, startWidth: width / 8, startHeight: height / 8)
    decoder.compile(inputs: input)
    load(Float.self, reader, vaeState)
    let result = decoder(inputs: input)[0].as(of: Float.self)
    compare("pipeline/decoded", result, pack["decoded"], tolerance: 0.005)
    print(reference.save_image(result.rawValue.toCPU().makeNumpyArray(), "pipeline-swift.png"))
    print(reference.save_image(pack["decoded"], "pipeline-reference.png"))
  }
}

let mode = CommandLine.arguments.dropFirst().first ?? "parity"
precondition(["float32", "float16", "bfloat16"].contains(precision), "Unknown FLUX3_DTYPE")
switch mode {
case "export":
  exportCheckpoint()
  exit(0)
case "export-dit":
  exportDiTCheckpoint()
  exit(0)
case "parity-dit":
  precondition(
    precision != "float16", "Use FP32 or BF16 for DiT until activation scaling is implemented")
  if precision == "bfloat16" {
    runDiT(BFloat16.self, name: precision)
  } else {
    runDiT(Float.self, name: "float32")
  }
case "parity-text":
  if precision == "bfloat16" {
    runText(BFloat16.self, name: precision)
  } else if precision == "float32" {
    runText(Float.self, name: precision)
  } else {
    runText(Float16.self, name: "float16")
  }
case "parity-vae": runVAE()
case "parity-pipeline": runPipeline()
case "parity":
  runText(Float16.self, name: "float16")
  runVAE()
  runDiT(Float.self, name: "float32")
  runPipeline()
case "--help", "help":
  print(
    "Usage: bash examples/flux3/run.sh [export|export-dit|parity|parity-text|parity-vae|parity-dit|parity-pipeline]"
  )
  print("FLUX3_DTYPE=float32|bfloat16 for DiT; float32|float16|bfloat16 for text.")
  print("FLUX3_DEVICE=1, FLUX3_REFERENCE_DEVICE=0; export reruns parity before writing.")
  print(
    "FLUX3_TEXT_CHECKPOINT, FLUX3_DIT_CHECKPOINT, FLUX3_VAE_CHECKPOINT override the three files.")
  exit(0)
default: fatalError("Unknown mode: \(mode)")
}
let reportPrecision =
  mode == "parity"
  ? "mixed" : (["parity-vae", "parity-pipeline"].contains(mode) ? "float32" : precision)
reference.finish("\(mode)-\(reportPrecision)")
