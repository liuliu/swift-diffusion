import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

setbuf(stdout, nil)
let env = ProcessInfo.processInfo.environment
let variant = env["MOSS_VARIANT"] ?? "sft"
precondition(["sft", "base"].contains(variant), "Unknown MOSS_VARIANT: \(variant)")
let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let sys = Python.import("sys")
let site = Python.import("site")
for path in [String(site.getusersitepackages())!, "/usr/lib/python3/dist-packages"] {
  if !(Bool(sys.path.__contains__(path)) ?? false) { sys.path.append(path) }
}
sys.path.insert(0, directory)
let reference = Python.import("reference")
let device = Int(env["MOSS_DEVICE"] ?? "1")!
let referenceDevice = Int(env["MOSS_REFERENCE_DEVICE"] ?? "0")!
let precision = env["MOSS_DTYPE"] ?? "float32"

func cpu(_ value: PythonObject) -> Tensor<Float> { try! Tensor<Float>(numpy: value) }

func load<T: TensorNumeric>(
  _ type: T.Type, bindings: [MOSSBinding], state: PythonObject, component: String,
  fp16Weights: Bool = false
) {
  for (index, binding) in bindings.enumerated() {
    let value = cpu(
      reference.weight(
        state, binding.key, binding.transform, binding.heads,
        binding.rowStart, binding.rowCount, component, fp16Weights && !binding.float32))
    if binding.parameter {
      if binding.float32 {
        binding.model.parameters.copy(from: value)
      } else {
        binding.model.parameters.copy(from: Tensor<T>(from: value))
      }
    } else if binding.bias {
      if binding.float32 {
        binding.model.bias.copy(from: value)
      } else {
        binding.model.bias.copy(from: Tensor<T>(from: value))
      }
    } else if binding.float32 {
      binding.model.weight.copy(from: value)
    } else {
      binding.model.weight.copy(from: Tensor<T>(from: value))
    }
    if index % 64 == 0 { print("Loaded", index + 1, "/", bindings.count, component, "parameters") }
  }
}

func compare<T: TensorNumeric>(
  _ label: String, actual: DynamicGraph.Tensor<T>, expected: PythonObject,
  maxRelative: Double = 1e-3, relativeL2: Double = 1e-3
) {
  let value = Tensor<Float>(from: actual.rawValue.toCPU())
  _ = reference.metrics(label, value.makeNumpyArray(), expected, maxRelative, relativeL2)
}

func runText<T: TensorNumeric>(_ type: T.Type) {
  let state = reference.State("text_encoder")
  let cases: [PythonObject] = [
    reference.text_case("random", 17, 42),
    reference.text_case("A red teapot on a wooden table beside a window, soft morning light."),
    reference.text_case("random", 65, 123),
  ]
  for (index, pack) in cases.enumerated() {
    if let selected = env["MOSS_CASE"], Int(selected) != index { continue }
    let idsFromNumpy = try! Tensor<Int32>(numpy: pack["tokens"])
    let ids = idsFromNumpy.reshaped(format: .NHWC, shape: idsFromNumpy.shape)
    let length = ids.shape[0]
    print("Text case", index, "tokens", length, "dtype", precision)
    let graph = DynamicGraph()
    graph.maxConcurrency = .limit(1)
    graph.withNoGrad {
      let tokens = graph.variable(ids.toGPU(device))
      let rotaryCPU = mossTextRotary(type, length: length)
      let components = T.dataType == .Float32 ? nil : mossSplitRotary(rotaryCPU)
      let rotary = graph.variable((components?.0 ?? rotaryCPU).toGPU(device))
      let sineRotary = components.map { graph.variable($0.1.toGPU(device)) }
      var causal = Tensor<Float>(.CPU, .NHWC(1, 1, length, length))
      for i in 0..<length {
        for j in 0..<length { causal[0, 0, i, j] = j > i ? -.infinity : 0 }
      }
      let mask = graph.variable(causal.toGPU(device))
      let zeros = Tensor<Float>(
        Array(repeating: 0, count: 32 * 128 * 128), .CPU, .HWC(32, 128, 128))
      let zeroState = graph.variable(zeros.toGPU(device))
      let convZeros = Tensor<T>(
        Array(repeating: T.zero, count: 3 * 8_192), .CPU, .NHWC(1, 3, 1, 8_192))
      let convPadding = graph.variable(convZeros.toGPU(device))
      var triangle = Tensor<Float>(.CPU, .NHWC(7, 32, 64, 64))
      for level in 0..<7 {
        let span = 1 << level
        for head in 0..<32 {
          for i in 0..<64 {
            for j in 0..<64 {
              let include =
                level == 0
                ? j <= i
                : i / span == j / span && i % span >= span / 2 && j % span < span / 2
              triangle[level, head, i, j] = include ? 1 : 0
            }
          }
        }
      }
      let lowerTriangle = graph.variable(triangle.toGPU(device))
      let inputs: [DynamicGraph.AnyTensor] =
        [tokens, rotary]
        + (T.dataType == .Float32 ? [mask] : [sineRotary!]) + [
          zeroState, convPadding, lowerTriangle,
        ]
      let (model, bindings, names) = MOSSTextEncoder(
        type, length: length, device: device, diagnostic: true)
      model.maxConcurrency = .limit(1)
      print("Compile Swift encoder")
      model.compile(inputs: inputs)
      reference.check_text_names(model.parameters.filter { _ in true }.map { $0.name })
      load(type, bindings: bindings, state: state, component: "text")
      let expected = reference.text_reference(pack, precision, referenceDevice)
      print("Run Swift encoder")
      let outputs = model(inputs: tokens, Array(inputs.dropFirst()))
      for (i, output) in outputs.enumerated() {
        compare(
          "text/\(precision)/case\(index)/\(names[i])", actual: output.as(of: T.self),
          expected: expected[names[i]], maxRelative: T.dataType == .Float32 ? 1e-3 : 0.05,
          relativeL2: T.dataType == .Float32 ? 1e-3 : 0.03)
        if index == 1 && names[i] == "output" {
          reference.save_text_features(
            Tensor<Float>(from: output.as(of: T.self).rawValue.toCPU()).makeNumpyArray(), precision)
        }
      }
    }
    reference.release()
  }
  reference.finish("text_\(precision)")
}

func runDiT<T: TensorNumeric>(
  _ type: T.Type, exportPath: String? = nil, fp16Weights: Bool = false
) {
  let state = reference.State("transformer")
  let reportName = fp16Weights ? "fp16_weights" : precision
  for index in 0..<3 {
    if let selected = env["MOSS_CASE"], Int(selected) != index { continue }
    let pack = reference.dit_case(index, precision)
    let height = Int(pack["height"])!
    let width = Int(pack["width"])!
    let contextCPU = cpu(pack["context"])
    let textLength = contextCPU.shape[0]
    let skip = Set([Int](pack["skip_layers"])!)
    print(
      "DiT case", index, "grid", height, width, "text", textLength, "dtype", precision, "skip", skip
    )
    let expected = reference.dit_reference(pack, precision, referenceDevice)
    let graph = DynamicGraph()
    graph.maxConcurrency = .limit(1)
    graph.withNoGrad {
      let x = graph.variable(Tensor<T>(from: cpu(pack["x"])).toGPU(device)).reshaped(
        .WC(height * width, 128))
      let context = graph.variable(Tensor<T>(from: contextCPU).toGPU(device)).reshaped(
        .WC(textLength, 4_096))
      let rotaryCPU = mossDiTRotary(type, height: height, width: width, textLength: textLength)
      let components = T.dataType == .Float32 ? nil : mossSplitRotary(rotaryCPU)
      let rotary = graph.variable((components?.0 ?? rotaryCPU).toGPU(device))
      let sineRotary = components.map { graph.variable($0.1.toGPU(device)) }
      let time = graph.variable(
        Tensor<T>(from: mossTimeEmbedding(Float(pack["timestep"])!)).toGPU(device))
      let (model, bindings, names) = MOSSDiT(
        type, textLength: textLength, height: height, width: width, device: device,
        skipLayers: skip, diagnostic: true)
      model.maxConcurrency = .limit(1)
      let inputs: [DynamicGraph.AnyTensor] =
        [x, context, rotary]
        + (sineRotary.map { [$0] } ?? []) + [time]
      model.compile(inputs: inputs)
      reference.check_dit_keys(state, bindings.map { $0.key }, Array(skip))
      load(type, bindings: bindings, state: state, component: "dit", fp16Weights: fp16Weights)
      print("Run Swift DiT")
      let outputs = model(inputs: x, Array(inputs.dropFirst()))
      for (i, output) in outputs.enumerated() {
        let label = "dit/\(reportName)/case\(index)/\(names[i])"
        if names[i] == "time_embedding" {
          compare(
            label, actual: output.as(of: Float.self), expected: expected[names[i]],
            maxRelative: T.dataType == .Float32 ? 1e-3 : 0.05,
            relativeL2: T.dataType == .Float32 ? 1e-3 : 0.03)
        } else {
          compare(
            label, actual: output.as(of: T.self), expected: expected[names[i]],
            maxRelative: T.dataType == .Float32 ? 1e-3 : 0.05,
            relativeL2: T.dataType == .Float32 ? 1e-3 : 0.03)
        }
      }
      if index == 2 {
        let composed = reference.dit_reference(pack, precision, referenceDevice, true)
        compare(
          "composed/\(reportName)/text_and_dit", actual: outputs.last!.as(of: T.self),
          expected: composed["output"], maxRelative: T.dataType == .Float32 ? 2e-3 : 0.06,
          relativeL2: T.dataType == .Float32 ? 2e-3 : 0.04)
        if let exportPath {
          // All three cases must pass in this session before writing any weights.
          reference.finish("dit_\(reportName)")
          exportDiTCheckpoint(
            type, graph: graph, model: model, inputs: inputs,
            expected: Tensor<Float>(from: outputs.last!.as(of: T.self).rawValue.toCPU()),
            height: height, width: width, textLength: textLength, path: exportPath,
            fp16Storage: fp16Weights)
        }
      }
    }
    reference.release()
  }
  reference.finish(exportPath == nil ? "dit_\(reportName)" : "export_dit_\(reportName)")
}

func exportDiTCheckpoint<T: TensorNumeric>(
  _ type: T.Type, graph: DynamicGraph, model: Model, inputs: [DynamicGraph.AnyTensor],
  expected: Tensor<Float>, height: Int, width: Int, textLength: Int, path: String,
  fp16Storage: Bool = false
) {
  precondition(T.dataType == .Float32)
  let staging = path + ".partial"
  let reloadDevice = Int(env["MOSS_RELOAD_DEVICE"] ?? "2")!
  let reloadGraph = DynamicGraph()
  reloadGraph.maxConcurrency = .limit(1)
  var keys = [String]()
  print("Exporting DiT to", staging)
  try! graph.openStore(staging) { store in
    try! store.write("dit", model: model, strict: true) { name, tensor in
      keys.append(name)
      if fp16Storage {
        let value = Tensor<Float>(from: tensor).toCPU()
        if mossDiTFP32StoreNames.contains(name) {
          try! store.write(name, tensor: value, strict: true)
        } else {
          // Values were already rounded to nearest FP16 before parity. The CPU
          // conversion here is exact even though ccv's FP32->FP16 converter truncates.
          try! store.write(name, tensor: Tensor<Float16>(from: value), strict: true)
        }
        return .skip
      }
      return .continue(name)
    }
  }.get()
  reference.check_dit_store(staging, keys, fp16Storage)
  reloadGraph.withNoGrad {
    let copied: [DynamicGraph.AnyTensor] = inputs.map {
      reloadGraph.variable($0.as(of: T.self).rawValue.toCPU().toGPU(reloadDevice))
    }
    let (fresh, _, _) = MOSSDiT(
      type, textLength: textLength, height: height, width: width, device: reloadDevice)
    fresh.maxConcurrency = .limit(1)
    fresh.compile(inputs: copied)
    try! reloadGraph.openStore(staging, flags: .readOnly) {
      try! $0.read("dit", model: fresh, strict: true)
    }.get()
    let output = fresh(inputs: copied[0], Array(copied.dropFirst()))[0].as(of: T.self)
    compare(
      "checkpoint/dit/output", actual: output, expected: expected.makeNumpyArray(),
      maxRelative: 1e-6, relativeL2: 1e-6)
  }
  reference.finish(fp16Storage ? "export_dit_fp16_weights" : "export_dit_\(precision)")
  precondition(rename(staging, path) == 0, "Cannot finalize DiT checkpoint")
  print("DiT export complete:", path)
}

let mode = CommandLine.arguments.dropFirst().first ?? "parity-text"
if env["MOSS_INSPECT"] == "1" {
  print("Construct encoder")
  let (_, bindings, _) = MOSSTextEncoder(Float.self, length: 17, device: device, diagnostic: true)
  print("Constructed encoder bindings:", bindings.count)
  exit(0)
}
precondition(
  ["parity-text", "parity-dit", "generate", "export-dit", "export-dit-f16"].contains(mode),
  "Usage: moss_image [parity-text|parity-dit|generate|export-dit|export-dit-f16]")
if mode == "generate" {
  runGeneration()
  exit(0)
}
if mode == "export-dit" || mode == "export-dit-f16" {
  precondition(precision == "float32", "Export validation uses FP32 arithmetic")
  precondition(env["MOSS_CASE"] == nil, "Export must run all three parity cases")
  let fp16Weights = mode == "export-dit-f16"
  let path =
    env["MOSS_DIT_CHECKPOINT"]
    ?? (fp16Weights
      ? "/slow/Data/moss_image_\(variant)_f16.ckpt"
      : "/slow/Data/moss_image_\(variant)_dit_f32.ckpt")
  precondition(
    !FileManager.default.fileExists(atPath: path + ".partial"),
    "Incomplete checkpoint already exists: \(path).partial")
  precondition(
    env["MOSS_REPLACE_CHECKPOINT"] == "1" || !FileManager.default.fileExists(atPath: path),
    "Checkpoint already exists: \(path)")
  runDiT(Float.self, exportPath: path, fp16Weights: fp16Weights)
  exit(0)
}
switch precision {
case "float32": if mode == "parity-text" { runText(Float.self) } else { runDiT(Float.self) }
case "float16": if mode == "parity-text" { runText(Float16.self) } else { runDiT(Float16.self) }
case "bfloat16": if mode == "parity-text" { runText(BFloat16.self) } else { runDiT(BFloat16.self) }
default: fatalError("Unknown MOSS_DTYPE: \(precision)")
}
