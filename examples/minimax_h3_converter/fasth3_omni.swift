import Foundation
import NNC
import NNCPythonConversion
import PythonKit

enum FastH3OmniConfig {
  static let pddSteps = 32
  static let sourceRoot = "/slow/Data/FastH3-Omni"
  static let exportPath = "/slow/Data/fasth3_omni_dit_f32.ckpt"
}

func fastH3MetricsPass(_ metrics: TokenParityMetrics) -> Bool {
  metrics.allFinite && metrics.maxRelativeDifference < 0.02
    && metrics.normalizedRootMeanSquareError < 0.01
    && metrics.minimumCosineSimilarity > 0.999
}

func runFastH3OmniParity(
  export: Bool = false, headsOnly: Bool = false, layers: Int = H3Config.layers
) {
  precondition(layers > 0 && layers <= H3Config.layers)
  precondition(!export || (!headsOnly && layers == H3Config.layers))
  precondition(Bool(torch.cuda.is_available()) == true, "Run FastH3 parity outside the sandbox")
  torch.set_grad_enabled(false)
  torch.set_num_threads(8)
  torch.backends.cuda.matmul.allow_tf32 = false
  let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().appendingPathComponent("minimax_h3_assets").path
  movePythonPathToFront(directory)
  let reference = Python.import("fasth3_omni_reference")
  let root = environment["FASTH3_OMNI_MODEL"] ?? FastH3OmniConfig.sourceRoot
  let exportPath = environment["FASTH3_OMNI_EXPORT_PATH"] ?? FastH3OmniConfig.exportPath
  let stagingPath = exportPath + ".partial"
  if export {
    precondition(
      !FileManager.default.fileExists(atPath: exportPath),
      "Refusing to overwrite an existing export")
    precondition(
      !FileManager.default.fileExists(atPath: stagingPath),
      "Inspect the previous partial export before retrying")
  }
  if !headsOnly { reference.audit_source(root, layers) }
  // Both widened heads are in the first shard, allowing a real-weight submodel
  // check while the remaining transformer shards are still downloading.
  let state =
    headsOnly
    ? h3Reference.SingleStateDict(
      osPath.join(root, "transformer_ref", "diffusion_pytorch_model-00001-of-00014.safetensors"))
    : h3Reference.ShardedStateDict(osPath.join(root, "transformer_ref"))
  let headCase = reference.head_case(root, state, deviceID)
  graph.withNoGrad {
    let x = graph.variable(tensorFromPython(headCase["x"]).toGPU(deviceID))
    for (name, prefix, channels) in [("video", "proj_out", 96), ("audio", "audio_proj_out", 32)] {
      let input = Input()
      let projection = Dense(count: channels * FastH3OmniConfig.pddSteps, name: prefix)
      let model = Model([input], [projection(input)])
      model.compile(inputs: x)
      copyFloatDense(projection, state: state, weight: "\(prefix).weight", bias: "\(prefix).bias")
      let actual = model(inputs: x)[0].as(of: Float.self).rawValue.toCPU()
      let expected = tensorFromPython(headCase[name])
      precondition(actual.shape == expected.shape)
      let metrics = tokenParityMetrics(
        actual.reshaped(.WC(7 * 32, channels)), expected.reshaped(.WC(7 * 32, channels)))
      printMetrics("FastH3 isolated \(name) all 32 heads", metrics)
      precondition(
        metrics.allFinite && metrics.maxRelativeDifference < 1e-4
          && metrics.minimumCosineSimilarity > 0.99999)
      // Verify source head order against the official weighted-parameter fusion
      // at every one of the eight scheduled inference intervals.
      let weights = tensorFromPython(headCase["\(name)_weights"].float())
      let fusedExpected = tensorFromPython(headCase["\(name)_fused"])
      var fused = Tensor<Float>(.CPU, .WC(7 * 8, channels))
      for token in 0..<7 {
        for step in 0..<8 {
          var total: Double = 0
          for head in (step * 4)..<(step * 4 + 4) { total += Double(weights[head]) }
          for channel in 0..<channels {
            var value: Double = 0
            for head in (step * 4)..<(step * 4 + 4) {
              value +=
                Double(weights[head]) / total * Double(actual[token, head * channels + channel])
            }
            fused[token * 8 + step, channel] = Float(value)
          }
        }
      }
      let fusedMetrics = tokenParityMetrics(fused, fusedExpected.reshaped(.WC(7 * 8, channels)))
      printMetrics("FastH3 \(name) official eight PDD fusions", fusedMetrics)
      precondition(
        fusedMetrics.allFinite && fusedMetrics.maxRelativeDifference < 1e-4
          && fusedMetrics.minimumCosineSimilarity > 0.99999)
    }
  }
  state.release()
  if headsOnly {
    print("FastH3 PDD head parity passed; full transformer validation still required")
    return
  }
  let pack = h3Reference.load_one_block_shell(root, deviceID, "transformer_ref", 32, true)
  let attentionPrecision =
    environment["H3_FP32_ATTENTION_CONTROL"] == "1"
    ? "FP32 GEMM/softmax attention control" : "BF16 only in SDPA"
  print("FastH3: Python \(layers)-layer FP32 reference (\(attentionPrecision))")
  let test = h3Reference.run_full_shell_case(pack, layers, 52, deviceID, true, true)
  // Free the Python GPU shell before compiling the complete unified-memory graph.
  let fullState = pack["state"]
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
      textLength: 2, audioLength: 2, videoLength: 2, timestepCount: 2,
      layers: layers, includeHidden: !export, pddSteps: FastH3OmniConfig.pddSteps,
      includeCompressionGates: true)
    model.maxConcurrency = .limit(1)
    print("FastH3: compiling Swift full transformer")
    model.compile(inputs: video, audio, text, rotary, adaln, timestep, frequencies)
    if layers == H3Config.layers {
      reference.check_model_names((0..<model.parameters.count).map { model.parameters[$0].name })
    }
    fullState.accessed.clear()
    print("FastH3: loading full transformer into unified memory")
    reader(fullState)
    reference.check_coverage(fullState, layers)
    fullState.release()
    print("FastH3: executing Swift full transformer")
    let results = model(inputs: video, audio, text, rotary, adaln, timestep, frequencies)
    var passed = true
    for (index, name, channels) in [(0, "video", 96), (1, "audio", 32)] {
      let actual = results[index].as(of: Float.self).rawValue.toCPU().reshaped(
        .WC(2, channels * 32))
      let expected = tensorFromPython(test["\(name)_output"])
      precondition(actual.shape == expected.shape)
      let metrics = tokenParityMetrics(
        actual.reshaped(.WC(64, channels)), expected.reshaped(.WC(64, channels)))
      printMetrics("FastH3 full \(name), token x head", metrics)
      passed = fastH3MetricsPass(metrics) && passed
      for head in 0..<32 {
        var headActual = Tensor<Float>(.CPU, .WC(2, channels))
        var headExpected = Tensor<Float>(.CPU, .WC(2, channels))
        for token in 0..<2 {
          for channel in 0..<channels {
            headActual[token, channel] = actual[token, head * channels + channel]
            headExpected[token, channel] = expected[token, head * channels + channel]
          }
        }
        let headMetrics = tokenParityMetrics(headActual, headExpected)
        printMetrics("FastH3 full \(name) head \(head)", headMetrics)
        passed = fastH3MetricsPass(headMetrics) && passed
      }
    }
    if !export {
      for (index, name) in [(2, "pre_norm"), (3, "hidden")] {
        let actual = results[index].as(of: Float.self).rawValue.toCPU().reshaped(
          .WC(6, H3Config.hiddenSize))
        printMetrics("FastH3 \(name)", tokenParityMetrics(actual, tensorFromPython(test[name])))
      }
    }
    for layer in 0..<layers {
      let actual = results[(export ? 2 : 4) + layer].as(of: Float.self).rawValue.toCPU()
        .reshaped(.WC(6, H3Config.innerAttentionSize))
      let expected = tensorFromPython(test["compression_gates"][layer])
      let metrics = tokenParityMetrics(actual, expected)
      printMetrics("FastH3 compression gate \(layer)", metrics)
      passed = fastH3MetricsPass(metrics) && passed
    }
    precondition(passed, "FastH3 full transformer parity failed; export disabled")
    if export {
      let before = (0..<2).map { index in
        results[index].as(of: Float.self).rawValue.toCPU()
          .reshaped(.WC(64, index == 0 ? 96 : 32))
      }
      print("FastH3 parity passed; writing dit to", exportPath)
      graph.openStore(stagingPath) { $0.write("dit", model: model) }
      // Reload into the same compiled model to avoid holding two 140-GB copies.
      graph.openStore(stagingPath) { try! $0.read("dit", model: model, strict: true) }
      let reloaded = model(inputs: video, audio, text, rotary, adaln, timestep, frequencies)
      for (index, channels) in [(0, 96), (1, 32)] {
        let after = reloaded[index].as(of: Float.self).rawValue.toCPU().reshaped(.WC(64, channels))
        let metrics = tokenParityMetrics(after, before[index])
        printMetrics("FastH3 store round-trip \(index)", metrics)
        precondition(metrics.allFinite && metrics.maxAbsoluteDifference == 0)
      }
      reference.audit_export(stagingPath, fullState)
      fullState.release()
      try! FileManager.default.moveItem(atPath: stagingPath, toPath: exportPath)
      try! FileManager.default.copyItem(
        atPath: root + "/fastvideo_inference.json", toPath: exportPath + ".inference.json")
    }
  }
  print("FastH3 Omni: \(layers)-layer parity checks passed")
}
