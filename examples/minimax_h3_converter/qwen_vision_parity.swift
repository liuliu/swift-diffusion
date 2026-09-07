import Foundation
import NNC
import NNCPythonConversion
import PythonKit

func h3QwenVisionMetricsPass(
  _ metrics: TokenParityMetrics, float16: Bool, publicOutput: Bool
) -> Bool {
  if !float16 {
    return metrics.allFinite && metrics.maxRelativeDifference < 0.001
      && metrics.minimumCosineSimilarity > 0.999
      && metrics.normalizedRootMeanSquareError < 0.001
  }
  // Match the existing Qwen FP16 export's output criteria. Last-block residuals
  // contain ~10k outliers, and their relative maxima are diagnostic before the
  // merger norm. Still require finite values and strong cosine / NRMSE there.
  return metrics.allFinite && (!publicOutput || metrics.maxRelativeDifference < 0.02)
    && metrics.minimumCosineSimilarity > 0.99 && metrics.meanCosineSimilarity > 0.999
    && metrics.normalizedRootMeanSquareError < 0.02
}

func h3QwenVisionReference() -> PythonObject {
  // Resolve relative to the checked-in converter, not the caller's working directory.
  let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().appendingPathComponent("minimax_h3_assets").path
  movePythonPathToFront(directory)
  return Python.import("qwen_vision_reference")
}

func loadH3QwenVision<T: TensorNumeric>(
  _ type: T.Type, bindings: [H3QwenVisionBinding], state: PythonObject
) -> Set<String> {
  var loaded = Set<String>()
  for binding in bindings {
    let key = "\(binding.prefix).weight"
    var weight = state[key]
    loaded.insert(key)
    if binding.kind == .embedding {
      binding.model.parameters.copy(from: Tensor<T>(from: tensorFromPython(weight)))
      binding.model.parameters.to(.unifiedMemory)
      continue
    }
    let biasKey = "\(binding.prefix).bias"
    var bias = state[biasKey]
    loaded.insert(biasKey)
    switch binding.kind {
    case .patch:
      weight = weight.flatten(1)
    case .query, .key, .value:
      let width = H3QwenVisionConfig.hiddenSize
      let index = binding.kind == .query ? 0 : (binding.kind == .key ? 1 : 2)
      weight = weight.narrow(0, index * width, width)
      bias = bias.narrow(0, index * width, width)
      if binding.kind != .value {
        weight = weight.reshape(H3QwenVisionConfig.heads, 2, H3QwenVisionConfig.headDim / 2, width)
          .transpose(1, 2).reshape(width, width).contiguous()
        bias = bias.reshape(H3QwenVisionConfig.heads, 2, H3QwenVisionConfig.headDim / 2)
          .transpose(1, 2).reshape(width).contiguous()
      }
    case .affine, .embedding: break
    }
    binding.model.weight.copy(from: Tensor<T>(from: tensorFromPython(weight)))
    binding.model.bias.copy(from: Tensor<T>(from: tensorFromPython(bias)))
    binding.model.weight.to(.unifiedMemory)
    binding.model.bias.to(.unifiedMemory)
  }
  return loaded
}

func runH3QwenVisionCase<T: TensorNumeric>(
  _ type: T.Type, reference: PythonObject, pack: PythonObject,
  grids: [H3QwenVisionGrid], seed: Int, exportPath: String? = nil
) -> Bool {
  let count = h3QwenVisionTokenCount(grids)
  let test = reference.run_case(pack, grids.map { [$0.t, $0.h, $0.w] }, seed)
  print("Qwen3-VL vision case:", T.dataType, grids, "patch tokens:", count)
  return graph.withNoGrad {
    let positions = h3QwenVisionPositions(grids)
    let patches = graph.variable(Tensor<T>(from: tensorFromPython(test["patches"])).toGPU(deviceID))
    let ids = graph.variable(positions.ids.toGPU(deviceID))
    let weights = graph.variable(positions.weights.toGPU(deviceID))
    let rotary = graph.variable(positions.rotary.toGPU(deviceID))
    // The final export graph has only four public outputs, and is validated as
    // such before writing. Diagnostic runs additionally check intermediate blocks.
    let (model, bindings) = H3QwenVisionModel(type, grids: grids, debugOutputs: exportPath == nil)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: patches, ids, weights, rotary)
    let loaded = loadH3QwenVision(type, bindings: bindings, state: pack["state"])
    reference.check_coverage(pack, loaded.sorted())
    let results = model(inputs: patches, ids, weights, rotary)
    let labels = [
      "merger", "deepstack0", "deepstack1", "deepstack2", "stem", "block8", "block16", "block24",
      "block26",
    ]
    var passed = true
    var cpuResults = [Tensor<Float>]()
    for index in 0..<results.count {
      let actual = Tensor<Float>(from: results[index].as(of: T.self).rawValue.toCPU())
      let expected = tensorFromPython(test["outputs"][index])
      print("\(labels[index]) Swift / Python shape:", actual.shape, expected.shape)
      let metrics = tokenParityMetrics(actual, expected)
      printMetrics("\(T.dataType) \(labels[index])", metrics)
      passed =
        h3QwenVisionMetricsPass(
          metrics, float16: T.dataType == .Float16, publicOutput: index < 4) && passed
      if T.dataType == .Float16 {
        let baseline = tokenParityMetrics(actual, tensorFromPython(test["baseline"][index]))
        printMetrics("Float16 vs FP32 \(labels[index])", baseline)
        let pythonBaseline = tokenParityMetrics(
          expected, tensorFromPython(test["baseline"][index]))
        printMetrics("Python Float16 vs FP32 \(labels[index])", pythonBaseline)
        passed =
          h3QwenVisionMetricsPass(
            baseline, float16: true, publicOutput: index < 4) && passed
      }
      cpuResults.append(actual)
    }
    guard passed else { return false }
    if let exportPath {
      precondition(T.dataType == .Float16)
      let previousTextNames = reference.text_store_names(exportPath)
      print("Qwen3-VL vision: parity passed; writing vision_model to", exportPath)
      graph.openStore(exportPath) { $0.write("vision_model", model: model) }
      let (reloaded, _) = H3QwenVisionModel(type, grids: grids)
      reloaded.maxConcurrency = .limit(1)
      reloaded.compile(inputs: patches, ids, weights, rotary)
      graph.openStore(exportPath) { try! $0.read("vision_model", model: reloaded, strict: true) }
      let roundTrip = reloaded(inputs: patches, ids, weights, rotary)
      for index in 0..<roundTrip.count {
        let actual = Tensor<Float>(from: roundTrip[index].as(of: T.self).rawValue.toCPU())
        let metrics = tokenParityMetrics(actual, cpuResults[index])
        printMetrics("Reload \(labels[index])", metrics)
        precondition(
          metrics.allFinite && metrics.maxAbsoluteDifference == 0, "Vision store round trip failed")
      }
      reference.audit_store(exportPath, previousTextNames)
    }
    return true
  }
}

func runH3QwenVisionParity(export: Bool = false) {
  precondition(Bool(torch.cuda.is_available()) == true, "Run vision parity outside the sandbox")
  let reference = h3QwenVisionReference()
  let cases: [[H3QwenVisionGrid]] = [[(1, 8, 8)], [(2, 8, 12), (1, 6, 10)], [(1, 28, 28)]]
  for precision in ["float32", "float16"] {
    let pack = reference.load(modelRoot, deviceID, precision)
    reference.check_fp16_weights(pack)
    var allPassed = true
    for (index, grids) in cases.enumerated() {
      let passed =
        precision == "float32"
        ? runH3QwenVisionCase(
          Float.self, reference: reference, pack: pack, grids: grids, seed: 43 + index)
        : runH3QwenVisionCase(
          Float16.self, reference: reference, pack: pack, grids: grids, seed: 43 + index)
      allPassed = passed && allPassed
    }
    precondition(allPassed, "Qwen3-VL vision numeric parity failed")
    if export && precision == "float16" {
      let path =
        environment["MINIMAX_H3_QWEN_VISION_EXPORT_PATH"]
        ?? "/slow/Data/minimax_h3_qwen_3_vl_with_vision_f16.ckpt"
      let staging = String(reference.prepare_combined_store(qwenExportPath, path))!
      precondition(
        runH3QwenVisionCase(
          Float16.self, reference: reference, pack: pack, grids: cases[0], seed: 43,
          exportPath: staging))
      reference.finish_combined_store(qwenExportPath, staging, path)
    }
    reference.release(pack)
  }
  print("Qwen3-VL vision: all FP32 and FP16 parity cases passed")
}
