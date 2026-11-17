import Foundation
import NNC
import NNCPythonConversion
import PythonKit

enum PikaH3FlexConfig {
  static let sourceRoot = "/slow/Data/Pika-H3-Flex"
  static let adapterRank = 128
  static let adapterAlpha = 128
  static let samplingKnots = [1000, 750, 500, 250, 100, 40, 0]
  static let videoShift: Float = 12
  static let audioShift: Float = 3
}

func runPikaH3FlexParity(export: Bool = false) {
  let task = environment["PIKA_H3_TASK"] ?? "fl2va"
  precondition(task == "fl2va" || task == "ref2va")
  precondition(environment["H3_FP32_ATTENTION_CONTROL"] == "1")
  precondition(Bool(torch.cuda.is_available()) == true, "Run Pika parity outside the sandbox")
  let trunk = task == "fl2va" ? "transformer" : "transformer_ref"
  let stem = task == "fl2va" ? "pika_h3_flex_dit" : "pika_h3_flex_ref2va_dit"
  let exportPath = "/slow/Data/\(stem)_f16.ckpt"
  let stagingPath = exportPath + ".partial"
  if export {
    for path in [exportPath, stagingPath, exportPath + ".inference.json"] {
      precondition(!FileManager.default.fileExists(atPath: path), "Refusing to overwrite \(path)")
    }
  }
  torch.set_grad_enabled(false)
  torch.set_num_threads(8)
  torch.backends.cuda.matmul.allow_tf32 = false
  let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().appendingPathComponent("minimax_h3_assets").path
  movePythonPathToFront(directory)
  let reference = Python.import("pika_h3_reference")
  let state = reference.MergedState(task, deviceID)
  precondition(Int(state.manifest["rank"]) == PikaH3FlexConfig.adapterRank)
  precondition(Int(state.manifest["alpha"]) == PikaH3FlexConfig.adapterAlpha)
  precondition([Int](state.manifest["sampling"]["knots"]) == PikaH3FlexConfig.samplingKnots)
  precondition(Float(state.manifest["sampling"]["video_shift"]) == PikaH3FlexConfig.videoShift)
  precondition(Float(state.manifest["sampling"]["audio_shift"]) == PikaH3FlexConfig.audioShift)
  precondition(String(state.manifest["sampling"]["method"]) == "renoise")

  // FP16-vs-FP32 storage differences were measured and accepted by the user.
  // Conversion parity still compares Swift and Python with identical FP16-rounded
  // weights and FP32 arithmetic; no native-FP32 storage exceptions are enabled.
  precondition(Bool(state.keep_native_fp32) == false)
  state.rounded = true
  print("Pika \(task): Python reference with FP16-rounded weights, FP32 arithmetic")
  let pack = h3Reference.load_one_block_shell(
    PikaH3FlexConfig.sourceRoot, deviceID, trunk, 1, true, state)
  let test = h3Reference.run_full_shell_case(pack, H3Config.layers, 52, deviceID, true)
  h3Reference.release_one_block_shell(pack)

  graph.withNoGrad {
    let video = graph.variable(tensorFromPython(test["video"]).toGPU(deviceID))
    let audio = graph.variable(tensorFromPython(test["audio"]).toGPU(deviceID))
    let text = graph.variable(tensorFromPython(test["text"]).toGPU(deviceID))
    let rotary = graph.variable(
      tensorFromPython(test["rotary"]).toGPU(deviceID).reshaped(.NHWC(1, 6, 1, H3Config.headDim)))
    let adaln = graph.variable(tensorFromPython(test["adaln_selection"]).toGPU(deviceID))
    let timestep = graph.variable(tensorFromPython(test["timestep_selection"]).toGPU(deviceID))
    let frequencies = graph.variable(tensorFromPython(test["frequencies"]).toGPU(deviceID))
    let (model, reader) = H3JointTransformer(
      textLength: 2, audioLength: 2, videoLength: 2, timestepCount: 2)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: video, audio, text, rotary, adaln, timestep, frequencies)
    state.accessed.clear()
    print("Pika \(task): loading FP16-rounded weights into the existing H3 graph, unified memory")
    reader(state)
    reference.MergedState.validate_coverage(
      state, (0..<model.parameters.count).map { model.parameters[$0].name })
    state.release()
    torch.cuda.empty_cache()
    print("Pika \(task): executing Swift 50-layer transformer")
    let outputs = model(inputs: video, audio, text, rotary, adaln, timestep, frequencies)
    var before = [Tensor<Float>]()
    var passed = true
    for (index, name, channels) in [(0, "video", 96), (1, "audio", 32)] {
      let actual = outputs[index].as(of: Float.self).rawValue.toCPU().reshaped(.WC(2, channels))
      let expected = tensorFromPython(test["\(name)_output"])
      precondition(actual.shape == expected.shape)
      let parity = tokenParityMetrics(actual, expected)
      printMetrics("Pika \(task) \(name): Swift vs Python, FP16 weights", parity)
      passed =
        parity.allFinite && parity.maxRelativeDifference < 1e-3
        && parity.normalizedRootMeanSquareError < 1e-3
        && parity.minimumCosineSimilarity > 0.99999 && passed
      before.append(actual)
    }
    precondition(passed, "Pika same-precision Swift/Python parity failed; export disabled")
    if export {
      print("Pika \(task): parity passed; writing FP16 weights to", stagingPath)
      // Same streaming writer used by Flux3/MOSS; no intermediate FP32 checkpoint.
      try! graph.openStore(stagingPath) { store in
        try! store.write("dit", model: model, strict: true) { name, tensor in
          let value = Tensor<Float>(from: tensor).toCPU()
          try! store.write(name, tensor: Tensor<Float16>(from: value), strict: true)
          return .skip
        }
      }.get()
      print("Pika \(task): strict reload into the same graph")
      try! graph.openStore(stagingPath, flags: .readOnly) {
        try! $0.read("dit", model: model, strict: true)
      }.get()
      let reloaded = model(inputs: video, audio, text, rotary, adaln, timestep, frequencies)
      for (index, channels) in [(0, 96), (1, 32)] {
        let after = reloaded[index].as(of: Float.self).rawValue.toCPU().reshaped(.WC(2, channels))
        let metrics = tokenParityMetrics(after, before[index])
        printMetrics("Pika \(task) checkpoint round-trip \(index)", metrics)
        precondition(metrics.allFinite && metrics.maxAbsoluteDifference == 0)
      }
      print("Pika \(task): auditing all saved tensors against FP16 merged-weight hashes")
      reference.audit_export(stagingPath, state)
      try! FileManager.default.moveItem(atPath: stagingPath, toPath: exportPath)
      reference.write_sidecar(exportPath, state)
      print("Pika \(task): export complete:", exportPath)
    }
  }
  print("Pika \(task): full-model parity and FP16 storage checks passed")
}
