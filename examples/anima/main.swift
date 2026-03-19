import Diffusion
import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

typealias FloatType = Float

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

let batchSize = 1
let latentFrames = 1

func envInt(_ name: String, _ defaultValue: Int) -> Int {
  if let value = ProcessInfo.processInfo.environment[name], let intValue = Int(value) {
    return intValue
  }
  return defaultValue
}

func envFlag(_ name: String) -> Bool {
  guard let value = ProcessInfo.processInfo.environment[name] else { return false }
  return value == "1" || value.lowercased() == "true" || value.lowercased() == "yes"
}

let swiftDevice = envInt("ANIMA_SWIFT_DEVICE", 2)

enum PythonPaths {
  static let venvSitePackages =
    "/home/liu/workspace/swift-diffusion/examples/anima/venv/lib/python3.10/site-packages"
  static let zImageSitePackages =
    "/home/liu/workspace/Z-Image/_env/lib/python3.10/site-packages"
  static let systemDistPackages = "/usr/lib/python3/dist-packages"
  static let referenceRoot = "/home/liu/workspace/swift-diffusion/examples/anima/reference"
  static let adapterModuleRoot =
    "/home/liu/workspace/swift-diffusion/examples/anima/reference/llm_adapter"
  static let filteredUserSiteFragment = ".local/lib/python3.10/site-packages"
}

enum AnimaTextConfig {
  static let vocabularySize = 151_936
  static let hiddenSize = 1_024
  static let intermediateSize = 3_072
  static let layers = 28
  static let attentionHeads = 16
  static let keyValueHeads = 8
  static let headDimension = 128
  static let rmsNormEpsilon: Float = 1e-6
  static let ropeTheta = 1_000_000.0
  static let maxPositionEmbeddings = 32_768
}

enum AnimaAdapterConfig {
  static let sourceDimension = 1_024
  static let targetDimension = 1_024
  static let modelDimension = 1_024
  static let layers = 6
  static let attentionHeads = 16
  static let headDimension = 64
  static let mlpRatio = 4
  static let intermediateSize = 4_096
  static let vocabularySize = 32_128
  static let rmsNormEpsilon: Float = 1e-6
  static let ropeTheta = 10_000.0
}

enum AnimaDiTConfig {
  static let inChannels = 16
  static let outChannels = 16
  static let attentionHeads = 16
  static let headDimension = 128
  static let hiddenSize = attentionHeads * headDimension
  static let layers = 28
  static let mlpRatio = 4
  static let textEmbedDimension = 1_024
  static let adaLoraDimension = 256
  static let patchFrames = 1
  static let patchHeight = 2
  static let patchWidth = 2
  static let patchEmbedInChannels = inChannels + 1
  static let patchChannels = patchEmbedInChannels * patchFrames * patchHeight * patchWidth
  static let projOutChannels = patchFrames * patchHeight * patchWidth * outChannels
  static let ropeScale = (temporal: 1.0, height: 4.0, width: 4.0)
  static let maxSize = (frames: 128, height: 240, width: 240)
}

enum AnimaExportConfig {
  static let maxSequenceLength = envInt("ANIMA_MAX_SEQUENCE_LENGTH", 512)
  static let latentHeight = envInt("ANIMA_LATENT_HEIGHT", 96)
  static let latentWidth = envInt("ANIMA_LATENT_WIDTH", 170)
}

enum AnimaParityConfig {
  static let maxSequenceLength = envInt("ANIMA_PARITY_SEQUENCE_LENGTH", 32)
  static let latentHeight = envInt("ANIMA_PARITY_LATENT_HEIGHT", 16)
  static let latentWidth = envInt("ANIMA_PARITY_LATENT_WIDTH", 16)
  static let timestep: Float = 1
}

enum AnimaParityThresholds {
  // The Qwen text path is accepted at a looser tolerance than the adapter / DiT.
  static let textMaxAbs: Float = 0.5
  static let textRelative: Float = 0.02
  static let adapterMaxAbs: Float = 1e-3
  static let adapterRelative: Float = 1e-3
  static let ditMaxAbs: Float = 1e-3
  static let ditRelative: Float = 1e-3
}

enum AnimaPaths {
  static let referenceRoot = PythonPaths.referenceRoot
  static let outputRoot = "/home/liu/workspace/swift-diffusion"
  static let textEncoderOutputPath = "\(outputRoot)/anima_qwen3_text_encoder_f32.ckpt"
  static let mainOutputPath = "\(outputRoot)/anima_f32.ckpt"
  static let textStateDictPath = "\(referenceRoot)/text_encoder/model.safetensors"
  static let adapterStateDictPath =
    "\(referenceRoot)/llm_adapter/diffusion_pytorch_model.safetensors"
  static let transformerStateDictPath =
    "\(referenceRoot)/transformer/diffusion_pytorch_model.safetensors"
}

func validateConfig(sequenceLength: Int, latentHeight: Int, latentWidth: Int) {
  precondition(sequenceLength > 0)
  precondition(latentHeight % AnimaDiTConfig.patchHeight == 0)
  precondition(latentWidth % AnimaDiTConfig.patchWidth == 0)
}

func insertPythonPath(_ path: String, sys: PythonObject) {
  guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return }
  if !Bool(sys.path.__contains__(path))! {
    sys.path.insert(0, path)
  }
}

func configurePythonPaths() {
  let sys = Python.import("sys")
  insertPythonPath(PythonPaths.zImageSitePackages, sys: sys)
  insertPythonPath(PythonPaths.venvSitePackages, sys: sys)
  insertPythonPath(PythonPaths.systemDistPackages, sys: sys)
  insertPythonPath(PythonPaths.referenceRoot, sys: sys)
  insertPythonPath(PythonPaths.adapterModuleRoot, sys: sys)
  let builtins = Python.import("builtins")
  let filteredPaths = builtins.list()
  let pathCount = Int(sys.path.__len__()) ?? 0
  for i in 0..<pathCount {
    let path = sys.path[i]
    let pathString = String(path) ?? ""
    if pathString.contains(PythonPaths.filteredUserSiteFragment)
      && pathString != PythonPaths.zImageSitePackages
    {
      continue
    }
    _ = filteredPaths.append(path)
  }
  sys.path = filteredPaths
}

func installFlexAttentionStub() {
  let sys = Python.import("sys")
  if Bool(sys.modules.__contains__("torch.nn.attention.flex_attention")) ?? false {
    return
  }
  let types = Python.import("types")
  let torch = Python.import("torch")
  let module = types.ModuleType("torch.nn.attention.flex_attention")
  module._DEFAULT_SPARSE_BLOCK_SIZE = 128
  module.BlockMask = torch.Tensor
  module.create_block_mask = Python.None
  module.flex_attention = Python.None
  sys.modules["torch.nn.attention.flex_attention"] = module
}

_ = setenv("PYTHONNOUSERSITE", "1", 1)

configurePythonPaths()

let torch = Python.import("torch")
let safetensorsTorch = Python.import("safetensors.torch")
let pythonGC = Python.import("gc")

torch.set_grad_enabled(false)
torch.manual_seed(42)
installFlexAttentionStub()

let textStateDict = safetensorsTorch.load_file(AnimaPaths.textStateDictPath)
let adapterStateDict = safetensorsTorch.load_file(AnimaPaths.adapterStateDictPath)
let transformerStateDict = safetensorsTorch.load_file(AnimaPaths.transformerStateDictPath)

validateConfig(
  sequenceLength: AnimaExportConfig.maxSequenceLength, latentHeight: AnimaExportConfig.latentHeight,
  latentWidth: AnimaExportConfig.latentWidth)
validateConfig(
  sequenceLength: AnimaParityConfig.maxSequenceLength, latentHeight: AnimaParityConfig.latentHeight,
  latentWidth: AnimaParityConfig.latentWidth)

func makeHalfSplitRotary(tokenLength: Int, headDimension: Int, theta: Double) -> Tensor<Float> {
  precondition(headDimension % 2 == 0)
  let half = headDimension / 2
  var rotary = Tensor<Float>(.CPU, .NHWC(1, tokenLength, 1, headDimension))
  if envFlag("ANIMA_ZERO_ROTARY") {
    for i in 0..<tokenLength {
      for k in 0..<half {
        rotary[0, i, 0, k * 2] = 1
        rotary[0, i, 0, k * 2 + 1] = 0
      }
    }
    return rotary
  }
  for i in 0..<tokenLength {
    for k in 0..<half {
      let freq = Double(i) / pow(theta, Double(2 * k) / Double(headDimension))
      rotary[0, i, 0, k * 2] = Float(cos(freq))
      rotary[0, i, 0, k * 2 + 1] = Float(sin(freq))
    }
  }
  return rotary
}

func makeCosmosRotary(
  frames: Int, height: Int, width: Int, headDimension: Int, ropeScale: (Double, Double, Double)
) -> Tensor<Float> {
  precondition(headDimension % 2 == 0)
  let dimH = headDimension / 6 * 2
  let dimW = headDimension / 6 * 2
  let dimT = headDimension - dimH - dimW
  let half = headDimension / 2
  precondition(half == dimT / 2 + dimH / 2 + dimW / 2)
  let hNTKFactor = pow(ropeScale.1, Double(dimH) / Double(dimH - 2))
  let wNTKFactor = pow(ropeScale.2, Double(dimW) / Double(dimW - 2))
  let tNTKFactor = pow(ropeScale.0, Double(dimT) / Double(dimT - 2))
  let hTheta = AnimaAdapterConfig.ropeTheta * hNTKFactor
  let wTheta = AnimaAdapterConfig.ropeTheta * wNTKFactor
  let tTheta = AnimaAdapterConfig.ropeTheta * tNTKFactor
  let temporalFreqs = (0..<(dimT / 2)).map { 1.0 / pow(tTheta, Double(2 * $0) / Double(dimT)) }
  let heightFreqs = (0..<(dimH / 2)).map { 1.0 / pow(hTheta, Double(2 * $0) / Double(dimH)) }
  let widthFreqs = (0..<(dimW / 2)).map { 1.0 / pow(wTheta, Double(2 * $0) / Double(dimW)) }
  var rotary = Tensor<Float>(.CPU, .NHWC(1, frames * height * width, 1, headDimension))
  for t in 0..<frames {
    for y in 0..<height {
      for x in 0..<width {
        let token = t * height * width + y * width + x
        var i = 0
        for freq in temporalFreqs {
          let theta = Double(t) * freq
          rotary[0, token, 0, i * 2] = Float(cos(theta))
          rotary[0, token, 0, i * 2 + 1] = Float(sin(theta))
          i += 1
        }
        for freq in heightFreqs {
          let theta = Double(y) * freq
          rotary[0, token, 0, i * 2] = Float(cos(theta))
          rotary[0, token, 0, i * 2 + 1] = Float(sin(theta))
          i += 1
        }
        for freq in widthFreqs {
          let theta = Double(x) * freq
          rotary[0, token, 0, i * 2] = Float(cos(theta))
          rotary[0, token, 0, i * 2 + 1] = Float(sin(theta))
          i += 1
        }
      }
    }
  }
  return rotary
}

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timesteps
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

func qkvInterleavedWeight(_ tensor: PythonObject, heads: Int, headDimension: Int) -> Tensor<Float> {
  let numpy = tensor.type(torch.float).view(heads, 2, headDimension / 2, -1).transpose(1, 2).cpu()
    .numpy()
  return try! Tensor<Float>(numpy: numpy)
}

func qkvInterleavedNorm(_ tensor: PythonObject, headDimension: Int) -> Tensor<Float> {
  let numpy = tensor.type(torch.float).view(2, headDimension / 2).transpose(0, 1).cpu().numpy()
  return try! Tensor<Float>(numpy: numpy)
}

func qkvInterleavedRepeatedWeight(
  _ tensor: PythonObject, sourceHeads: Int, targetHeads: Int, headDimension: Int
) -> Tensor<Float> {
  precondition(targetHeads % sourceHeads == 0)
  let repeats = targetHeads / sourceHeads
  let numpy = tensor.type(torch.float)
    .view(sourceHeads, 2, headDimension / 2, -1)
    .transpose(1, 2)
    .repeat_interleave(repeats, dim: 0)
    .reshape(targetHeads * headDimension, -1)
    .cpu()
    .numpy()
  return try! Tensor<Float>(numpy: numpy)
}

func repeatedHeadWeight(
  _ tensor: PythonObject, sourceHeads: Int, targetHeads: Int, headDimension: Int
)
  -> Tensor<Float>
{
  precondition(targetHeads % sourceHeads == 0)
  let repeats = targetHeads / sourceHeads
  let numpy = tensor.type(torch.float)
    .view(sourceHeads, headDimension, -1)
    .repeat_interleave(repeats, dim: 0)
    .reshape(targetHeads * headDimension, -1)
    .cpu()
    .numpy()
  return try! Tensor<Float>(numpy: numpy)
}

func tensorFromPython(_ tensor: PythonObject) -> Tensor<Float> {
  try! Tensor<Float>(numpy: tensor.type(torch.float).cpu().numpy())
}

func tensorInt32FromPython(_ tensor: PythonObject) -> Tensor<Int32> {
  try! Tensor<Int32>(numpy: tensor.type(torch.int32).cpu().numpy())
}

func graphVariableNHWCFloat(_ tensor: Tensor<Float>) -> DynamicGraph.Tensor<FloatType> {
  switch tensor.shape.count {
  case 2:
    let variable = graph.variable(
      .CPU, format: .NHWC, shape: [tensor.shape[0], tensor.shape[1]], of: FloatType.self)
    for i in 0..<tensor.shape[0] {
      for j in 0..<tensor.shape[1] {
        variable[i, j] = FloatType(tensor[i, j])
      }
    }
    return variable
  case 3:
    let variable = graph.variable(
      .CPU, format: .NHWC, shape: [tensor.shape[0], tensor.shape[1], tensor.shape[2]],
      of: FloatType.self)
    for i in 0..<tensor.shape[0] {
      for j in 0..<tensor.shape[1] {
        for k in 0..<tensor.shape[2] {
          variable[i, j, k] = FloatType(tensor[i, j, k])
        }
      }
    }
    return variable
  case 4:
    let variable = graph.variable(
      .CPU, format: .NHWC,
      shape: [tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]],
      of: FloatType.self)
    for i in 0..<tensor.shape[0] {
      for j in 0..<tensor.shape[1] {
        for k in 0..<tensor.shape[2] {
          for l in 0..<tensor.shape[3] {
            variable[i, j, k, l] = FloatType(tensor[i, j, k, l])
          }
        }
      }
    }
    return variable
  case 5:
    let variable = graph.variable(
      .CPU, format: .NHWC,
      shape: [tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4]],
      of: FloatType.self)
    for i in 0..<tensor.shape[0] {
      for j in 0..<tensor.shape[1] {
        for k in 0..<tensor.shape[2] {
          for l in 0..<tensor.shape[3] {
            for m in 0..<tensor.shape[4] {
              variable[i, j, k, l, m] = FloatType(tensor[i, j, k, l, m])
            }
          }
        }
      }
    }
    return variable
  default:
    fatalError("Unsupported float tensor rank \(tensor.shape.count)")
  }
}

func graphVariableNHWCInt32(_ tensor: Tensor<Int32>) -> DynamicGraph.Tensor<Int32> {
  switch tensor.shape.count {
  case 1:
    let variable = graph.variable(.CPU, format: .NHWC, shape: [tensor.shape[0]], of: Int32.self)
    for i in 0..<tensor.shape[0] {
      variable[i] = tensor[i]
    }
    return variable
  case 2:
    let variable = graph.variable(
      .CPU, format: .NHWC, shape: [tensor.shape[0], tensor.shape[1]], of: Int32.self)
    for i in 0..<tensor.shape[0] {
      for j in 0..<tensor.shape[1] {
        variable[i, j] = tensor[i, j]
      }
    }
    return variable
  default:
    fatalError("Unsupported int tensor rank \(tensor.shape.count)")
  }
}

func maxAbsAndRelativeDiff2D<T: TensorNumeric & BinaryFloatingPoint>(
  _ swiftTensor: DynamicGraph.Tensor<T>, _ torchTensor: Tensor<Float>
) -> (
  Float, Float
) {
  precondition(swiftTensor.shape.count == 2)
  precondition(torchTensor.shape.count == 2)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  var maxDiff: Float = 0
  var maxReferenceAbs: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for j in 0..<torchTensor.shape[1] {
      let referenceAbs = abs(torchTensor[i, j])
      if referenceAbs > maxReferenceAbs {
        maxReferenceAbs = referenceAbs
      }
      let diff = abs(Float(swiftTensor[i, j]) - torchTensor[i, j])
      if diff > maxDiff {
        maxDiff = diff
      }
    }
  }
  return (maxDiff, maxDiff / max(1e-6, maxReferenceAbs))
}

func printRowDiffSummary2D<T: TensorNumeric & BinaryFloatingPoint>(
  _ swiftTensor: DynamicGraph.Tensor<T>, _ torchTensor: Tensor<Float>, maxRows: Int = 4
) {
  let rows = min(maxRows, torchTensor.shape[0])
  for i in 0..<rows {
    var rowMaxDiff: Float = 0
    var rowMaxReferenceAbs: Float = 0
    for j in 0..<torchTensor.shape[1] {
      let referenceAbs = abs(torchTensor[i, j])
      if referenceAbs > rowMaxReferenceAbs {
        rowMaxReferenceAbs = referenceAbs
      }
      let diff = abs(Float(swiftTensor[i, j]) - torchTensor[i, j])
      if diff > rowMaxDiff {
        rowMaxDiff = diff
      }
    }
    print(
      "anima text row", i, "max-abs diff:", rowMaxDiff, "relative:",
      rowMaxDiff / max(1e-6, rowMaxReferenceAbs))
  }
}

func debugPrintValueSamples2D<T: TensorNumeric & BinaryFloatingPoint>(
  _ swiftTensor: DynamicGraph.Tensor<T>, _ torchTensor: Tensor<Float>, maxRows: Int = 2,
  maxColumns: Int = 8
) {
  let rows = min(maxRows, torchTensor.shape[0])
  let columns = min(maxColumns, torchTensor.shape[1])
  print("anima text swift sample")
  debugPrint(swiftTensor[0..<rows, 0..<columns].copied())
  print("anima text reference sample")
  debugPrint(torchTensor[0..<rows, 0..<columns])
  var diff = Tensor<Float>(.CPU, .NC(rows, columns))
  for i in 0..<rows {
    for j in 0..<columns {
      diff[i, j] = Float(swiftTensor[i, j]) - torchTensor[i, j]
    }
  }
  print("anima text diff sample")
  debugPrint(diff)
}

func maxAbsAndRelativeDiff5D<T: TensorNumeric & BinaryFloatingPoint>(
  _ swiftTensor: DynamicGraph.Tensor<T>, _ torchTensor: Tensor<Float>
) -> (
  Float, Float
) {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  precondition(swiftTensor.shape[4] == torchTensor.shape[4])
  var maxDiff: Float = 0
  var maxReferenceAbs: Float = 0
  for b in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for t in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let referenceAbs = abs(torchTensor[b, c, t, h, w])
            if referenceAbs > maxReferenceAbs {
              maxReferenceAbs = referenceAbs
            }
            let diff = abs(Float(swiftTensor[b, c, t, h, w]) - torchTensor[b, c, t, h, w])
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
  }
  return (maxDiff, maxDiff / max(1e-6, maxReferenceAbs))
}

func makeReferenceTextEncoder() -> PythonObject {
  let transformers = Python.import("transformers")
  let config = transformers.Qwen3Config(
    vocab_size: AnimaTextConfig.vocabularySize,
    hidden_size: AnimaTextConfig.hiddenSize,
    intermediate_size: AnimaTextConfig.intermediateSize,
    num_hidden_layers: AnimaTextConfig.layers,
    num_attention_heads: AnimaTextConfig.attentionHeads,
    num_key_value_heads: AnimaTextConfig.keyValueHeads,
    head_dim: AnimaTextConfig.headDimension,
    hidden_act: "silu",
    max_position_embeddings: AnimaTextConfig.maxPositionEmbeddings,
    rms_norm_eps: Double(AnimaTextConfig.rmsNormEpsilon),
    rope_theta: AnimaTextConfig.ropeTheta,
    attention_bias: false,
    attention_dropout: 0.0,
    use_cache: false,
    tie_word_embeddings: false
  )
  let model = transformers.Qwen3Model(config)
  let remappedStateDict = Python.dict()
  for keyObject in textStateDict.keys() {
    let key = String(keyObject) ?? ""
    let remappedKey = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
    remappedStateDict[PythonObject(remappedKey)] = textStateDict[keyObject]
  }
  _ = model.load_state_dict(remappedStateDict, strict: true)
  _ = model.eval()
  return model
}

func makeReferenceAdapter() -> PythonObject {
  let adapterModule = Python.import("modeling_llm_adapter")
  let model = adapterModule.AnimaLLMAdapter(
    source_dim: AnimaAdapterConfig.sourceDimension,
    target_dim: AnimaAdapterConfig.targetDimension,
    model_dim: AnimaAdapterConfig.modelDimension,
    num_layers: AnimaAdapterConfig.layers,
    num_heads: AnimaAdapterConfig.attentionHeads,
    mlp_ratio: Double(AnimaAdapterConfig.mlpRatio),
    vocab_size: AnimaAdapterConfig.vocabularySize,
    use_self_attn: true
  )
  _ = model.load_state_dict(adapterStateDict, strict: true)
  _ = model.eval()
  return model
}

func makeReferenceTransformer() -> PythonObject {
  let diffusersModels = Python.import("diffusers.models")
  let model = diffusersModels.CosmosTransformer3DModel(
    in_channels: AnimaDiTConfig.inChannels,
    out_channels: AnimaDiTConfig.outChannels,
    num_attention_heads: AnimaDiTConfig.attentionHeads,
    attention_head_dim: AnimaDiTConfig.headDimension,
    num_layers: AnimaDiTConfig.layers,
    mlp_ratio: Double(AnimaDiTConfig.mlpRatio),
    text_embed_dim: AnimaDiTConfig.textEmbedDimension,
    adaln_lora_dim: AnimaDiTConfig.adaLoraDimension,
    max_size: [
      AnimaDiTConfig.maxSize.frames, AnimaDiTConfig.maxSize.height, AnimaDiTConfig.maxSize.width,
    ],
    patch_size: [
      AnimaDiTConfig.patchFrames, AnimaDiTConfig.patchHeight, AnimaDiTConfig.patchWidth,
    ],
    rope_scale: [
      AnimaDiTConfig.ropeScale.temporal, AnimaDiTConfig.ropeScale.height,
      AnimaDiTConfig.ropeScale.width,
    ],
    concat_padding_mask: true,
    extra_pos_embed_type: Python.None
  )
  _ = model.load_state_dict(transformerStateDict, strict: true)
  _ = model.eval()
  return model
}

func requireExportApproval() {
  precondition(
    ProcessInfo.processInfo.environment["ANIMA_ALLOW_EXPORT"] == "1",
    "Export is disabled until parity passes and you explicitly approve the export run."
  )
}

func releasePythonReferenceModel(_ model: inout PythonObject) {
  _ = model.cpu()
  model = Python.None
  _ = pythonGC.collect()
}

func QwenFeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let up = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  let down = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  let out = down(up(x) .* gate(x).swish())
  return (gate, down, up, Model([x], [out]))
}

func manualAttentionNHWC(
  queries: Model.IO, keys: Model.IO, values: Model.IO, queryLength: Int, keyLength: Int, heads: Int,
  headDimension: Int
) -> Model.IO {
  let query = queries.transposed(1, 2).contiguous()
  let key = keys.transposed(1, 2).contiguous()
  let value = values.transposed(1, 2).contiguous()
  var dot = (1.0 / Float(headDimension).squareRoot()) * Matmul(transposeB: (2, 3))(query, key)
  dot = dot.reshaped([batchSize * heads * queryLength, keyLength]).softmax()
  dot = dot.reshaped([batchSize, heads, queryLength, keyLength])
  return (dot * value).transposed(1, 2)
}

func Qwen3SelfAttention(
  prefix: String, width: Int, headDimension: Int, heads: Int, kvHeads: Int, tokenLength: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let toKeys = Dense(count: headDimension * kvHeads, noBias: true, name: "k_proj")
  let toQueries = Dense(count: headDimension * heads, noBias: true, name: "q_proj")
  let toValues = Dense(count: headDimension * kvHeads, noBias: true, name: "v_proj")
  var keys = toKeys(x).reshaped([batchSize, tokenLength, kvHeads, headDimension])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries = toQueries(x).reshaped([batchSize, tokenLength, heads, headDimension])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = toValues(x).reshaped([batchSize, tokenLength, kvHeads, headDimension])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(
    scale: 1.0 / Float(headDimension).squareRoot(), isCausal: true)(
      queries, keys, values
    ).reshaped([batchSize * tokenLength, heads * headDimension])
  let unifyHeads = Dense(count: width, noBias: true, name: "out_proj")
  out = unifyHeads(out)
  let reader: (PythonObject) -> Void = { stateDict in
    toQueries.weight.copy(
      from: Tensor<Float16>(
        from: qkvInterleavedWeight(
          stateDict["\(prefix).self_attn.q_proj.weight"], heads: heads,
          headDimension: headDimension)))
    normQ.weight.copy(
      from: Tensor<Float16>(
        from: qkvInterleavedNorm(
          stateDict["\(prefix).self_attn.q_norm.weight"], headDimension: headDimension)))
    toKeys.weight.copy(
      from: Tensor<Float16>(
        from: qkvInterleavedWeight(
          stateDict["\(prefix).self_attn.k_proj.weight"], heads: kvHeads,
          headDimension: headDimension)))
    normK.weight.copy(
      from: Tensor<Float16>(
        from: qkvInterleavedNorm(
          stateDict["\(prefix).self_attn.k_norm.weight"], headDimension: headDimension)))
    toValues.weight.copy(
      from: Tensor<Float16>(from: tensorFromPython(stateDict["\(prefix).self_attn.v_proj.weight"])))
    unifyHeads.weight.copy(
      from: Tensor<Float16>(from: tensorFromPython(stateDict["\(prefix).self_attn.o_proj.weight"])))
  }
  return (Model([x, rot], [out]), reader)
}

func Qwen3TransformerBlock(
  prefix: String, width: Int, headDimension: Int, heads: Int, kvHeads: Int, tokenLength: Int,
  intermediateSize: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x).to(.Float16)
  let (attention, attentionReader) = Qwen3SelfAttention(
    prefix: prefix, width: width, headDimension: headDimension, heads: heads, kvHeads: kvHeads,
    tokenLength: tokenLength)
  out = x + attention(out, rot).to(of: x)
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(out).to(.Float16)
  let (gate, down, up, feedForward) = QwenFeedForward(
    hiddenSize: width, intermediateSize: intermediateSize, name: "mlp")
  out = residual + feedForward(out).to(of: residual)
  let reader: (PythonObject) -> Void = { stateDict in
    attentionReader(stateDict)
    norm1.weight.copy(
      from: Tensor<Float32>(from: tensorFromPython(stateDict["\(prefix).input_layernorm.weight"])))
    norm2.weight.copy(
      from: Tensor<Float32>(
        from: tensorFromPython(stateDict["\(prefix).post_attention_layernorm.weight"])))
    gate.weight.copy(
      from: Tensor<Float16>(from: tensorFromPython(stateDict["\(prefix).mlp.gate_proj.weight"])))
    up.weight.copy(
      from: Tensor<Float16>(from: tensorFromPython(stateDict["\(prefix).mlp.up_proj.weight"])))
    down.weight.copy(
      from: Tensor<Float16>(from: tensorFromPython(stateDict["\(prefix).mlp.down_proj.weight"])))
  }
  return (Model([x, rot], [out]), reader)
}

func Qwen3TextEmbedding(vocabularySize: Int, embeddingSize: Int) -> (Model, (PythonObject) -> Void)
{
  let tokens = Input()
  let tokenEmbed = Embedding(
    Float16.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize,
    name: "tok_embeddings")
  let reader: (PythonObject) -> Void = { stateDict in
    tokenEmbed.parameters.copy(
      from: Tensor<Float16>(from: tensorFromPython(stateDict["model.embed_tokens.weight"])))
  }
  return (Model([tokens], [tokenEmbed(tokens)]), reader)
}

func Qwen3TextTransformer(tokenLength: Int) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embeddingReader) = Qwen3TextEmbedding(
    vocabularySize: AnimaTextConfig.vocabularySize, embeddingSize: AnimaTextConfig.hiddenSize)
  var out = embedding(tokens).to(.Float32)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<AnimaTextConfig.layers {
    let (block, reader) = Qwen3TransformerBlock(
      prefix: "model.layers.\(i)", width: AnimaTextConfig.hiddenSize,
      headDimension: AnimaTextConfig.headDimension, heads: AnimaTextConfig.attentionHeads,
      kvHeads: AnimaTextConfig.keyValueHeads, tokenLength: tokenLength,
      intermediateSize: AnimaTextConfig.intermediateSize)
    out = block(out, rot)
    readers.append(reader)
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out).to(.Float16)
  let reader: (PythonObject) -> Void = { stateDict in
    embeddingReader(stateDict)
    for blockReader in readers {
      blockReader(stateDict)
    }
    norm.weight.copy(from: Tensor<Float32>(from: tensorFromPython(stateDict["model.norm.weight"])))
  }
  return (Model([tokens, rot], [out]), reader)
}

func AdapterFeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize, name: "\(name)_fc1")
  let fc2 = Dense(count: hiddenSize, name: "\(name)_fc2")
  let out = fc2(fc1(x).GELU())
  return (fc1, fc2, fc2, Model([x], [out]))
}

func AdapterSelfAttention(
  prefix: String, width: Int, headDimension: Int, heads: Int, tokenLength: Int
)
  -> (Model, (PythonObject) -> Void)
{
  let x = Input()
  let rot = Input()
  let toKeys = Dense(count: headDimension * heads, noBias: true, name: "k_proj")
  let toQueries = Dense(count: headDimension * heads, noBias: true, name: "q_proj")
  let toValues = Dense(count: headDimension * heads, noBias: true, name: "v_proj")
  var keys = toKeys(x).reshaped(.NHWC(batchSize, tokenLength, heads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = Functional.cmul(left: normK(keys), right: rot)
  var queries = toQueries(x).reshaped(.NHWC(batchSize, tokenLength, heads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = Functional.cmul(left: normQ(queries), right: rot)
  let values = toValues(x).reshaped(.NHWC(batchSize, tokenLength, heads, headDimension))
  let out = manualAttentionNHWC(
    queries: queries, keys: keys, values: values, queryLength: tokenLength, keyLength: tokenLength,
    heads: heads, headDimension: headDimension
  ).reshaped([batchSize * tokenLength, width])
  let unifyHeads = Dense(count: width, noBias: true, name: "out_proj")
  let projected = unifyHeads(out)
  let reader: (PythonObject) -> Void = { stateDict in
    toQueries.weight.copy(
      from: qkvInterleavedWeight(
        stateDict["\(prefix).q_proj.weight"], heads: heads, headDimension: headDimension))
    normQ.weight.copy(
      from: qkvInterleavedNorm(
        stateDict["\(prefix).q_norm.weight"], headDimension: headDimension))
    toKeys.weight.copy(
      from: qkvInterleavedWeight(
        stateDict["\(prefix).k_proj.weight"], heads: heads, headDimension: headDimension))
    normK.weight.copy(
      from: qkvInterleavedNorm(
        stateDict["\(prefix).k_norm.weight"], headDimension: headDimension))
    toValues.weight.copy(from: tensorFromPython(stateDict["\(prefix).v_proj.weight"]))
    unifyHeads.weight.copy(from: tensorFromPython(stateDict["\(prefix).o_proj.weight"]))
  }
  return (Model([x, rot], [projected]), reader)
}

func AdapterCrossAttention(
  prefix: String, queryDim: Int, contextDim: Int, headDimension: Int, heads: Int,
  tokenLength: Int, contextLength: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let context = Input()
  let queryRot = Input()
  let contextRot = Input()
  let toKeys = Dense(count: headDimension * heads, noBias: true, name: "k_proj")
  let toQueries = Dense(count: headDimension * heads, noBias: true, name: "q_proj")
  let toValues = Dense(count: headDimension * heads, noBias: true, name: "v_proj")
  var keys = toKeys(context).reshaped(.NHWC(batchSize, contextLength, heads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = Functional.cmul(left: normK(keys), right: contextRot)
  var queries = toQueries(x).reshaped(.NHWC(batchSize, tokenLength, heads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = Functional.cmul(left: normQ(queries), right: queryRot)
  let values = toValues(context).reshaped(.NHWC(batchSize, contextLength, heads, headDimension))
  let out = manualAttentionNHWC(
    queries: queries, keys: keys, values: values, queryLength: tokenLength,
    keyLength: contextLength,
    heads: heads, headDimension: headDimension
  ).reshaped([batchSize * tokenLength, queryDim])
  let unifyHeads = Dense(count: queryDim, noBias: true, name: "out_proj")
  let projected = unifyHeads(out)
  let reader: (PythonObject) -> Void = { stateDict in
    toQueries.weight.copy(
      from: qkvInterleavedWeight(
        stateDict["\(prefix).q_proj.weight"], heads: heads, headDimension: headDimension))
    normQ.weight.copy(
      from: qkvInterleavedNorm(
        stateDict["\(prefix).q_norm.weight"], headDimension: headDimension))
    toKeys.weight.copy(
      from: qkvInterleavedWeight(
        stateDict["\(prefix).k_proj.weight"], heads: heads, headDimension: headDimension))
    normK.weight.copy(
      from: qkvInterleavedNorm(
        stateDict["\(prefix).k_norm.weight"], headDimension: headDimension))
    toValues.weight.copy(from: tensorFromPython(stateDict["\(prefix).v_proj.weight"]))
    unifyHeads.weight.copy(from: tensorFromPython(stateDict["\(prefix).o_proj.weight"]))
  }
  return (Model([x, context, queryRot, contextRot], [projected]), reader)
}

func AnimaAdapterBlock(tokenLength: Int, contextLength: Int, prefix: String) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let context = Input()
  let targetRot = Input()
  let sourceRot = Input()
  let normSelf = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_self")
  let (selfAttention, selfAttentionReader) = AdapterSelfAttention(
    prefix: "\(prefix).self_attn", width: AnimaAdapterConfig.modelDimension,
    headDimension: AnimaAdapterConfig.headDimension, heads: AnimaAdapterConfig.attentionHeads,
    tokenLength: tokenLength)
  var out = x + selfAttention(normSelf(x), targetRot)
  let normCross = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_cross")
  let (crossAttention, crossAttentionReader) = AdapterCrossAttention(
    prefix: "\(prefix).cross_attn", queryDim: AnimaAdapterConfig.modelDimension,
    contextDim: AnimaAdapterConfig.sourceDimension, headDimension: AnimaAdapterConfig.headDimension,
    heads: AnimaAdapterConfig.attentionHeads, tokenLength: tokenLength, contextLength: contextLength
  )
  out = out + crossAttention(normCross(out), context, targetRot, sourceRot)
  let normMlp = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_mlp")
  let fc1 = Dense(count: AnimaAdapterConfig.intermediateSize, name: "mlp_fc1")
  let fc2 = Dense(count: AnimaAdapterConfig.modelDimension, name: "mlp_fc2")
  out = out + fc2(fc1(normMlp(out)).GELU())
  let reader: (PythonObject) -> Void = { stateDict in
    selfAttentionReader(stateDict)
    crossAttentionReader(stateDict)
    normSelf.weight.copy(from: tensorFromPython(stateDict["\(prefix).norm_self_attn.weight"]))
    normCross.weight.copy(from: tensorFromPython(stateDict["\(prefix).norm_cross_attn.weight"]))
    normMlp.weight.copy(from: tensorFromPython(stateDict["\(prefix).norm_mlp.weight"]))
    fc1.weight.copy(from: tensorFromPython(stateDict["\(prefix).mlp.0.weight"]))
    fc1.bias.copy(from: tensorFromPython(stateDict["\(prefix).mlp.0.bias"]))
    fc2.weight.copy(from: tensorFromPython(stateDict["\(prefix).mlp.2.weight"]))
    fc2.bias.copy(from: tensorFromPython(stateDict["\(prefix).mlp.2.bias"]))
  }
  return (Model([x, context, targetRot, sourceRot], [out]), reader)
}

func AnimaLLMAdapter(tokenLength: Int, contextLength: Int) -> (Model, (PythonObject) -> Void) {
  let sourceHiddenStates = Input()
  let targetInputIDs = Input()
  let targetRot = Input()
  let sourceRot = Input()
  let embed = Embedding(
    FloatType.self, vocabularySize: AnimaAdapterConfig.vocabularySize,
    embeddingSize: AnimaAdapterConfig.targetDimension, name: "token_embedding")
  var out = embed(targetInputIDs)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<AnimaAdapterConfig.layers {
    let (block, reader) = AnimaAdapterBlock(
      tokenLength: tokenLength, contextLength: contextLength, prefix: "blocks.\(i)")
    out = block(out, sourceHiddenStates, targetRot, sourceRot)
    readers.append(reader)
  }
  let outProj = Dense(count: AnimaAdapterConfig.targetDimension, name: "out_proj")
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(outProj(out))
  let reader: (PythonObject) -> Void = { stateDict in
    embed.parameters.copy(from: tensorFromPython(stateDict["embed.weight"]))
    for blockReader in readers {
      blockReader(stateDict)
    }
    outProj.weight.copy(from: tensorFromPython(stateDict["out_proj.weight"]))
    outProj.bias.copy(from: tensorFromPython(stateDict["out_proj.bias"]))
    norm.weight.copy(from: tensorFromPython(stateDict["norm.weight"]))
  }
  return (Model([sourceHiddenStates, targetInputIDs, targetRot, sourceRot], [out]), reader)
}

func CosmosAdaLayerNormZero(
  prefix: String, hiddenSize: Int, hiddenFeatures: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let embeddedTimestep = Input()
  let tembShift = Input()
  let tembScale = Input()
  let tembGate = Input()
  let norm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let linear1 = Dense(count: hiddenFeatures, noBias: true, name: "linear_1")
  let shift = Dense(count: hiddenSize, noBias: true, name: "shift")
  let scale = Dense(count: hiddenSize, noBias: true, name: "scale")
  let gate = Dense(count: hiddenSize, noBias: true, name: "gate")
  let hidden = linear1(embeddedTimestep.swish())
  let shiftOut = shift(hidden) + tembShift
  let scaleOut = scale(hidden) + tembScale
  let gateOut = gate(hidden) + tembGate
  let out =
    (1 + scaleOut.reshaped([batchSize, 1, hiddenSize])).to(of: x) .* norm(x)
    + shiftOut.reshaped([batchSize, 1, hiddenSize]).to(of: x)
  let reader: (PythonObject) -> Void = { stateDict in
    linear1.weight.copy(from: tensorFromPython(stateDict["\(prefix).linear_1.weight"]))
    let linear2 = stateDict["\(prefix).linear_2.weight"].type(torch.float).cpu()
    shift.weight.copy(from: try! Tensor<Float>(numpy: linear2[..<hiddenSize, ...].numpy()))
    scale.weight.copy(
      from: try! Tensor<Float>(numpy: linear2[hiddenSize..<(2 * hiddenSize), ...].numpy()))
    gate.weight.copy(
      from: try! Tensor<Float>(numpy: linear2[(2 * hiddenSize)..<(3 * hiddenSize), ...].numpy()))
  }
  return (Model([x, embeddedTimestep, tembShift, tembScale, tembGate], [out, gateOut]), reader)
}

func CosmosAdaLayerNorm(prefix: String, hiddenSize: Int, hiddenFeatures: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let embeddedTimestep = Input()
  let tembShift = Input()
  let tembScale = Input()
  let norm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let linear1 = Dense(count: hiddenFeatures, noBias: true, name: "linear_1")
  let shift = Dense(count: hiddenSize, noBias: true, name: "shift")
  let scale = Dense(count: hiddenSize, noBias: true, name: "scale")
  let hidden = linear1(embeddedTimestep.swish())
  let shiftOut = shift(hidden) + tembShift
  let scaleOut = scale(hidden) + tembScale
  let out =
    (1 + scaleOut.reshaped([batchSize, 1, hiddenSize])).to(of: x) .* norm(x)
    + shiftOut.reshaped([batchSize, 1, hiddenSize]).to(of: x)
  let reader: (PythonObject) -> Void = { stateDict in
    linear1.weight.copy(from: tensorFromPython(stateDict["\(prefix).linear_1.weight"]))
    let linear2 = stateDict["\(prefix).linear_2.weight"].type(torch.float).cpu()
    shift.weight.copy(from: try! Tensor<Float>(numpy: linear2[..<hiddenSize, ...].numpy()))
    scale.weight.copy(
      from: try! Tensor<Float>(numpy: linear2[hiddenSize..<(2 * hiddenSize), ...].numpy()))
  }
  return (Model([x, embeddedTimestep, tembShift, tembScale], [out]), reader)
}

func CosmosSelfAttention(prefix: String, tokenLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let toKeys = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "k_proj")
  let toQueries = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "q_proj")
  let toValues = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "v_proj")
  var keys = toKeys(x).reshaped(
    .NHWC(batchSize, tokenLength, AnimaDiTConfig.attentionHeads, AnimaDiTConfig.headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = Functional.cmul(left: normK(keys), right: rot)
  var queries = toQueries(x).reshaped(
    .NHWC(batchSize, tokenLength, AnimaDiTConfig.attentionHeads, AnimaDiTConfig.headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = Functional.cmul(left: normQ(queries), right: rot)
  let values = toValues(x).reshaped(
    .NHWC(batchSize, tokenLength, AnimaDiTConfig.attentionHeads, AnimaDiTConfig.headDimension))
  let out = manualAttentionNHWC(
    queries: queries, keys: keys, values: values, queryLength: tokenLength, keyLength: tokenLength,
    heads: AnimaDiTConfig.attentionHeads, headDimension: AnimaDiTConfig.headDimension
  ).reshaped(
    [batchSize, tokenLength, AnimaDiTConfig.hiddenSize])
  let unifyHeads = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "out_proj")
  let projected = unifyHeads(out)
  let reader: (PythonObject) -> Void = { stateDict in
    toQueries.weight.copy(
      from: qkvInterleavedWeight(
        stateDict["\(prefix).to_q.weight"], heads: AnimaDiTConfig.attentionHeads,
        headDimension: AnimaDiTConfig.headDimension))
    normQ.weight.copy(
      from: qkvInterleavedNorm(
        stateDict["\(prefix).norm_q.weight"], headDimension: AnimaDiTConfig.headDimension))
    toKeys.weight.copy(
      from: qkvInterleavedWeight(
        stateDict["\(prefix).to_k.weight"], heads: AnimaDiTConfig.attentionHeads,
        headDimension: AnimaDiTConfig.headDimension))
    normK.weight.copy(
      from: qkvInterleavedNorm(
        stateDict["\(prefix).norm_k.weight"], headDimension: AnimaDiTConfig.headDimension))
    toValues.weight.copy(from: tensorFromPython(stateDict["\(prefix).to_v.weight"]))
    unifyHeads.weight.copy(from: tensorFromPython(stateDict["\(prefix).to_out.0.weight"]))
  }
  return (Model([x, rot], [projected]), reader)
}

func CosmosCrossAttention(prefix: String, tokenLength: Int, contextLength: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let context = Input()
  let toKeys = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "k_proj")
  let toQueries = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "q_proj")
  let toValues = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "v_proj")
  var keys = toKeys(context).reshaped(
    .NHWC(batchSize, contextLength, AnimaDiTConfig.attentionHeads, AnimaDiTConfig.headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries = toQueries(x).reshaped(
    .NHWC(batchSize, tokenLength, AnimaDiTConfig.attentionHeads, AnimaDiTConfig.headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = toValues(context).reshaped(
    .NHWC(batchSize, contextLength, AnimaDiTConfig.attentionHeads, AnimaDiTConfig.headDimension))
  let out = manualAttentionNHWC(
    queries: queries, keys: keys, values: values, queryLength: tokenLength,
    keyLength: contextLength,
    heads: AnimaDiTConfig.attentionHeads, headDimension: AnimaDiTConfig.headDimension
  ).reshaped(
    [batchSize, tokenLength, AnimaDiTConfig.hiddenSize])
  let unifyHeads = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "out_proj")
  let projected = unifyHeads(out)
  let reader: (PythonObject) -> Void = { stateDict in
    toQueries.weight.copy(from: tensorFromPython(stateDict["\(prefix).to_q.weight"]))
    normQ.weight.copy(from: tensorFromPython(stateDict["\(prefix).norm_q.weight"]))
    toKeys.weight.copy(from: tensorFromPython(stateDict["\(prefix).to_k.weight"]))
    normK.weight.copy(from: tensorFromPython(stateDict["\(prefix).norm_k.weight"]))
    toValues.weight.copy(from: tensorFromPython(stateDict["\(prefix).to_v.weight"]))
    unifyHeads.weight.copy(from: tensorFromPython(stateDict["\(prefix).to_out.0.weight"]))
  }
  return (Model([x, context], [projected]), reader)
}

func CosmosFeedForward(prefix: String) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let fc1 = Dense(
    count: AnimaDiTConfig.hiddenSize * AnimaDiTConfig.mlpRatio, noBias: true, name: "fc1")
  let fc2 = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "fc2")
  let out = fc2(fc1(x).GELU())
  let reader: (PythonObject) -> Void = { stateDict in
    fc1.weight.copy(from: tensorFromPython(stateDict["\(prefix).net.0.proj.weight"]))
    fc2.weight.copy(from: tensorFromPython(stateDict["\(prefix).net.2.weight"]))
  }
  return (Model([x], [out]), reader)
}

func CosmosTransformerBlock(prefix: String, tokenLength: Int, contextLength: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let context = Input()
  let embeddedTimestep = Input()
  let tembShift = Input()
  let tembScale = Input()
  let tembGate = Input()
  let rot = Input()
  let (norm1, norm1Reader) = CosmosAdaLayerNormZero(
    prefix: "\(prefix).norm1", hiddenSize: AnimaDiTConfig.hiddenSize,
    hiddenFeatures: AnimaDiTConfig.adaLoraDimension)
  let norm1Out = norm1(x, embeddedTimestep, tembShift, tembScale, tembGate)
  let (selfAttention, selfAttentionReader) = CosmosSelfAttention(
    prefix: "\(prefix).attn1", tokenLength: tokenLength)
  var out =
    x
    + norm1Out[1].reshaped([batchSize, 1, AnimaDiTConfig.hiddenSize]).to(of: x)
    .* selfAttention(norm1Out[0], rot)
  let (norm2, norm2Reader) = CosmosAdaLayerNormZero(
    prefix: "\(prefix).norm2", hiddenSize: AnimaDiTConfig.hiddenSize,
    hiddenFeatures: AnimaDiTConfig.adaLoraDimension)
  let norm2Out = norm2(out, embeddedTimestep, tembShift, tembScale, tembGate)
  let (crossAttention, crossAttentionReader) = CosmosCrossAttention(
    prefix: "\(prefix).attn2", tokenLength: tokenLength, contextLength: contextLength)
  out =
    out
    + norm2Out[1].reshaped([batchSize, 1, AnimaDiTConfig.hiddenSize]).to(of: out)
    .* crossAttention(norm2Out[0], context)
  let (norm3, norm3Reader) = CosmosAdaLayerNormZero(
    prefix: "\(prefix).norm3", hiddenSize: AnimaDiTConfig.hiddenSize,
    hiddenFeatures: AnimaDiTConfig.adaLoraDimension)
  let norm3Out = norm3(out, embeddedTimestep, tembShift, tembScale, tembGate)
  let (feedForward, feedForwardReader) = CosmosFeedForward(prefix: "\(prefix).ff")
  out =
    out
    + norm3Out[1].reshaped([batchSize, 1, AnimaDiTConfig.hiddenSize]).to(of: out)
    .* feedForward(norm3Out[0])
  let reader: (PythonObject) -> Void = { stateDict in
    norm1Reader(stateDict)
    selfAttentionReader(stateDict)
    norm2Reader(stateDict)
    crossAttentionReader(stateDict)
    norm3Reader(stateDict)
    feedForwardReader(stateDict)
  }
  return (
    Model([x, context, embeddedTimestep, tembShift, tembScale, tembGate, rot], [out]), reader
  )
}

func CosmosTransformer(latentHeight: Int, latentWidth: Int, textLength: Int) -> (
  Model, (PythonObject) -> Void
) {
  let tokenHeight = latentHeight / AnimaDiTConfig.patchHeight
  let tokenWidth = latentWidth / AnimaDiTConfig.patchWidth
  let tokenLength = latentFrames * tokenHeight * tokenWidth
  let hiddenStates = Input()
  let paddingMask = Input()
  let context = Input()
  let timestepProjection = Input()
  let rot = Input()
  let patchEmbed = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "patch_embed")
  let paddedHiddenStates = Functional.concat(
    axis: 1, hiddenStates, paddingMask.reshaped([batchSize, 1, 1, latentHeight, latentWidth]))
  var out = paddedHiddenStates.reshaped(
    [
      batchSize, AnimaDiTConfig.patchEmbedInChannels, latentFrames, AnimaDiTConfig.patchFrames,
      tokenHeight, AnimaDiTConfig.patchHeight, tokenWidth, AnimaDiTConfig.patchWidth,
    ]
  ).permuted(0, 2, 4, 6, 1, 3, 5, 7).contiguous().reshaped(
    [batchSize, tokenLength, AnimaDiTConfig.patchChannels], format: .NHWC)
  out = patchEmbed(out)
  let timeNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "time_norm")
  let timeLinear1 = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "time_linear_1")
  let timeShift = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "time_shift")
  let timeScale = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "time_scale")
  let timeGate = Dense(count: AnimaDiTConfig.hiddenSize, noBias: true, name: "time_gate")
  let embeddedTimestep = timeNorm(timestepProjection)
  let timeHidden = timeLinear1(timestepProjection).swish()
  let tembShift = timeShift(timeHidden)
  let tembScale = timeScale(timeHidden)
  let tembGate = timeGate(timeHidden)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<AnimaDiTConfig.layers {
    let (block, reader) = CosmosTransformerBlock(
      prefix: "transformer_blocks.\(i)", tokenLength: tokenLength, contextLength: textLength)
    out = block(out, context, embeddedTimestep, tembShift, tembScale, tembGate, rot)
    readers.append(reader)
  }
  let (normOut, normOutReader) = CosmosAdaLayerNorm(
    prefix: "norm_out", hiddenSize: AnimaDiTConfig.hiddenSize,
    hiddenFeatures: AnimaDiTConfig.adaLoraDimension)
  out = normOut(out, embeddedTimestep, tembShift, tembScale)
  let projOut = Dense(count: AnimaDiTConfig.projOutChannels, noBias: true, name: "proj_out")
  out = projOut(out).reshaped(
    [
      batchSize, latentFrames, tokenHeight, tokenWidth, AnimaDiTConfig.patchHeight,
      AnimaDiTConfig.patchWidth, AnimaDiTConfig.patchFrames, AnimaDiTConfig.outChannels,
    ])
    .permuted(0, 7, 1, 6, 2, 4, 3, 5).contiguous().reshaped(
      [batchSize, AnimaDiTConfig.outChannels, latentFrames, latentHeight, latentWidth])
  let reader: (PythonObject) -> Void = { stateDict in
    patchEmbed.weight.copy(from: tensorFromPython(stateDict["patch_embed.proj.weight"]))
    timeNorm.weight.copy(from: tensorFromPython(stateDict["time_embed.norm.weight"]))
    timeLinear1.weight.copy(
      from: tensorFromPython(stateDict["time_embed.t_embedder.linear_1.weight"]))
    let timeLinear2 = stateDict["time_embed.t_embedder.linear_2.weight"].type(torch.float).cpu()
    timeShift.weight.copy(
      from: try! Tensor<Float>(numpy: timeLinear2[..<AnimaDiTConfig.hiddenSize, ...].numpy()))
    timeScale.weight.copy(
      from: try! Tensor<Float>(
        numpy: timeLinear2[
          AnimaDiTConfig.hiddenSize..<(2 * AnimaDiTConfig.hiddenSize), ...
        ].numpy()))
    timeGate.weight.copy(
      from: try! Tensor<Float>(
        numpy: timeLinear2[
          (2 * AnimaDiTConfig.hiddenSize)..<(3 * AnimaDiTConfig.hiddenSize), ...
        ].numpy()))
    for blockReader in readers {
      blockReader(stateDict)
    }
    normOutReader(stateDict)
    projOut.weight.copy(from: tensorFromPython(stateDict["proj_out.weight"]))
  }
  return (Model([hiddenStates, paddingMask, context, timestepProjection, rot], [out]), reader)
}

@discardableResult
func runTextParity(tokenLength: Int = AnimaParityConfig.maxSequenceLength) -> Bool {
  return graph.withNoGrad {
    print("anima text parity: start")
    DynamicGraph.logLevel = envFlag("ANIMA_VERBOSE_GRAPH") ? .verbose : .none
    let tokenIDs = torch.randint(
      AnimaTextConfig.vocabularySize, [batchSize, tokenLength], dtype: torch.int64)
    let expected: Tensor<Float> = {
      var referenceModel = makeReferenceTextEncoder()
      let referenceOutputs: PythonObject
      if envFlag("ANIMA_ZERO_ROTARY") {
        let zeroPositionIDs = torch.zeros([batchSize, tokenLength], dtype: torch.int64)
        referenceOutputs = referenceModel(input_ids: tokenIDs, position_ids: zeroPositionIDs)
      } else {
        referenceOutputs = referenceModel(input_ids: tokenIDs)
      }
      let expected = try! Tensor<Float>(
        numpy: referenceOutputs.last_hidden_state.squeeze(0).to(torch.float).cpu().numpy())
      releasePythonReferenceModel(&referenceModel)
      return expected
    }()
    let (textModel, reader) = Qwen3TextTransformer(tokenLength: tokenLength)
    let swiftTokensCPU = graph.variable(.CPU, format: .NHWC, shape: [tokenLength], of: Int32.self)
    let tokenIDsCPU = tensorInt32FromPython(tokenIDs[0])
    for i in 0..<tokenLength {
      swiftTokensCPU[i] = tokenIDsCPU[i]
    }
    let rotCPU = graph.variable(
      .CPU, format: .NHWC, shape: [1, tokenLength, 1, AnimaTextConfig.headDimension],
      of: Float.self)
    let rotValues = makeHalfSplitRotary(
      tokenLength: tokenLength, headDimension: AnimaTextConfig.headDimension,
      theta: AnimaTextConfig.ropeTheta)
    for i in 0..<tokenLength {
      for j in 0..<AnimaTextConfig.headDimension {
        rotCPU[0, i, 0, j] = rotValues[0, i, 0, j]
      }
    }
    let swiftTokens = swiftTokensCPU.toGPU(swiftDevice)
    let rot = DynamicGraph.Tensor<Float16>(from: rotCPU).toGPU(swiftDevice)
    textModel.compile(inputs: swiftTokens, rot)
    reader(textStateDict)
    let swiftOutput = textModel(inputs: swiftTokens, rot)[0].as(of: Float16.self).toCPU()
    print("anima text parity shapes:", swiftOutput.shape, expected.shape)
    let (maxAbsDiff, relativeDiff) = maxAbsAndRelativeDiff2D(swiftOutput, expected)
    printRowDiffSummary2D(swiftOutput, expected)
    debugPrintValueSamples2D(swiftOutput, expected)
    print("anima text max-abs diff:", maxAbsDiff, "relative:", relativeDiff)
    return maxAbsDiff <= AnimaParityThresholds.textMaxAbs
      && relativeDiff <= AnimaParityThresholds.textRelative
  }
}

@discardableResult
func runAdapterParity(
  tokenLength: Int = AnimaParityConfig.maxSequenceLength,
  contextLength: Int = AnimaParityConfig.maxSequenceLength
) -> Bool {
  return graph.withNoGrad {
    print("anima adapter parity: start")
    DynamicGraph.logLevel = envFlag("ANIMA_VERBOSE_GRAPH") ? .verbose : .none
    let sourceHiddenStatesTorch = torch.randn(
      [batchSize, contextLength, AnimaAdapterConfig.sourceDimension], dtype: torch.float32)
    let targetInputIDsTorch = torch.randint(
      AnimaAdapterConfig.vocabularySize, [batchSize, tokenLength], dtype: torch.int64)
    let expected: Tensor<Float> = {
      var referenceModel = makeReferenceAdapter()
      let expected = try! Tensor<Float>(
        numpy: referenceModel(
          source_hidden_states: sourceHiddenStatesTorch, target_input_ids: targetInputIDsTorch
        ).squeeze(0).to(torch.float).cpu().numpy())
      releasePythonReferenceModel(&referenceModel)
      return expected
    }()
    let (adapter, adapterReader) = AnimaLLMAdapter(
      tokenLength: tokenLength, contextLength: contextLength)
    let sourceHiddenStates = graph.variable(
      .CPU, format: .NHWC, shape: [contextLength, AnimaAdapterConfig.sourceDimension],
      of: FloatType.self
    )
    .toGPU(swiftDevice)
    let targetInputIDs = graph.variable(.CPU, format: .NHWC, shape: [tokenLength], of: Int32.self)
      .toGPU(swiftDevice)
    let targetRot = graph.variable(
      makeHalfSplitRotary(
        tokenLength: tokenLength, headDimension: AnimaAdapterConfig.headDimension,
        theta: AnimaAdapterConfig.ropeTheta)
    ).toGPU(swiftDevice)
    let sourceRot = graph.variable(
      makeHalfSplitRotary(
        tokenLength: contextLength, headDimension: AnimaAdapterConfig.headDimension,
        theta: AnimaAdapterConfig.ropeTheta)
    ).toGPU(swiftDevice)
    adapter.compile(inputs: sourceHiddenStates, targetInputIDs, targetRot, sourceRot)
    adapterReader(adapterStateDict)
    let swiftSourceHiddenStates = graphVariableNHWCFloat(
      tensorFromPython(sourceHiddenStatesTorch[0])
    ).toGPU(swiftDevice)
    let swiftTargetInputIDs = graphVariableNHWCInt32(
      tensorInt32FromPython(targetInputIDsTorch[0])
    ).toGPU(swiftDevice)
    let swiftOutput = adapter(
      inputs: swiftSourceHiddenStates, swiftTargetInputIDs, targetRot, sourceRot)[0]
      .as(of: Float.self)
      .toCPU()
    print("anima adapter parity shapes:", swiftOutput.shape, expected.shape)
    let (maxAbsDiff, relativeDiff) = maxAbsAndRelativeDiff2D(swiftOutput, expected)
    print("anima adapter max-abs diff:", maxAbsDiff, "relative:", relativeDiff)
    return maxAbsDiff <= AnimaParityThresholds.adapterMaxAbs
      && relativeDiff <= AnimaParityThresholds.adapterRelative
  }
}

@discardableResult
func runTransformerParity(
  textLength: Int = AnimaParityConfig.maxSequenceLength,
  latentHeight: Int = AnimaParityConfig.latentHeight,
  latentWidth: Int = AnimaParityConfig.latentWidth
) -> Bool {
  return graph.withNoGrad {
    print("anima dit parity: start")
    DynamicGraph.logLevel = envFlag("ANIMA_VERBOSE_GRAPH") ? .verbose : .none
    let hiddenStatesTorch = torch.randn(
      [batchSize, AnimaDiTConfig.inChannels, latentFrames, latentHeight, latentWidth],
      dtype: torch.float32)
    let paddingMaskTorch = torch.zeros(
      [batchSize, 1, latentHeight, latentWidth], dtype: torch.float32)
    let contextTorch = torch.randn(
      [batchSize, textLength, AnimaDiTConfig.textEmbedDimension], dtype: torch.float32)
    let timestepTorch = torch.tensor([AnimaParityConfig.timestep], dtype: torch.float32)
    let expected: Tensor<Float> = {
      var referenceModel = makeReferenceTransformer()
      let expected = try! Tensor<Float>(
        numpy: referenceModel(
          hidden_states: hiddenStatesTorch,
          timestep: timestepTorch,
          encoder_hidden_states: contextTorch,
          padding_mask: paddingMaskTorch,
          return_dict: false
        )[0].to(torch.float).cpu().numpy())
      releasePythonReferenceModel(&referenceModel)
      return expected
    }()
    let (dit, ditReader) = CosmosTransformer(
      latentHeight: latentHeight, latentWidth: latentWidth, textLength: textLength)
    let hiddenStates = graph.variable(
      .CPU, format: .NHWC,
      shape: [batchSize, AnimaDiTConfig.inChannels, latentFrames, latentHeight, latentWidth],
      of: FloatType.self
    ).toGPU(swiftDevice)
    let paddingMask = graph.variable(
      .CPU, format: .NHWC, shape: [batchSize, 1, latentHeight, latentWidth], of: FloatType.self
    ).toGPU(swiftDevice)
    let context = graph.variable(
      .CPU, format: .NHWC, shape: [batchSize, textLength, AnimaDiTConfig.textEmbedDimension],
      of: FloatType.self
    )
    .toGPU(swiftDevice)
    let timestepProjection = graph.variable(
      timeEmbedding(
        timesteps: AnimaParityConfig.timestep, batchSize: batchSize,
        embeddingSize: AnimaDiTConfig.hiddenSize,
        maxPeriod: Int(AnimaAdapterConfig.ropeTheta))
    ).toGPU(swiftDevice)
    let rot = graph.variable(
      makeCosmosRotary(
        frames: latentFrames, height: latentHeight / AnimaDiTConfig.patchHeight,
        width: latentWidth / AnimaDiTConfig.patchWidth, headDimension: AnimaDiTConfig.headDimension,
        ropeScale: (
          AnimaDiTConfig.ropeScale.temporal, AnimaDiTConfig.ropeScale.height,
          AnimaDiTConfig.ropeScale.width
        ))
    ).toGPU(swiftDevice)
    dit.compile(inputs: hiddenStates, paddingMask, context, timestepProjection, rot)
    ditReader(transformerStateDict)
    let swiftHiddenStates = graphVariableNHWCFloat(tensorFromPython(hiddenStatesTorch)).toGPU(
      swiftDevice)
    let swiftPaddingMask = graphVariableNHWCFloat(tensorFromPython(paddingMaskTorch)).toGPU(
      swiftDevice)
    let swiftContext = graphVariableNHWCFloat(tensorFromPython(contextTorch)).toGPU(swiftDevice)
    let swiftOutput = dit(
      inputs: swiftHiddenStates, swiftPaddingMask, swiftContext, timestepProjection, rot)[0]
      .as(of: Float.self)
      .toCPU()
    print("anima dit parity shapes:", swiftOutput.shape, expected.shape)
    let (maxAbsDiff, relativeDiff) = maxAbsAndRelativeDiff5D(swiftOutput, expected)
    print("anima dit max-abs diff:", maxAbsDiff, "relative:", relativeDiff)
    return maxAbsDiff <= AnimaParityThresholds.ditMaxAbs
      && relativeDiff <= AnimaParityThresholds.ditRelative
  }
}

func exportTextModel() {
  graph.withNoGrad {
    requireExportApproval()
    precondition(runTextParity(), "Text parity must pass before export.")
    let sequenceLength = AnimaExportConfig.maxSequenceLength
    let (textModel, reader) = Qwen3TextTransformer(tokenLength: sequenceLength)
    let tokens = graph.variable(.CPU, format: .NHWC, shape: [sequenceLength], of: Int32.self)
    let rot = graph.variable(
      Tensor<Float16>(
        from: makeHalfSplitRotary(
          tokenLength: sequenceLength, headDimension: AnimaTextConfig.headDimension,
          theta: AnimaTextConfig.ropeTheta))
    )
    textModel.compile(inputs: tokens, rot)
    reader(textStateDict)
    graph.openStore(AnimaPaths.textEncoderOutputPath) {
      $0.write("text_model", model: textModel)
    }
    print("Wrote \(AnimaPaths.textEncoderOutputPath)")
  }
}

func exportMainModels() {
  graph.withNoGrad {
    requireExportApproval()
    precondition(runAdapterParity(), "Adapter parity must pass before export.")
    precondition(runTransformerParity(), "DiT parity must pass before export.")
    let sequenceLength = AnimaExportConfig.maxSequenceLength
    let latentHeight = AnimaExportConfig.latentHeight
    let latentWidth = AnimaExportConfig.latentWidth
    let (adapter, adapterReader) = AnimaLLMAdapter(
      tokenLength: sequenceLength, contextLength: sequenceLength)
    let sourceHiddenStates = graph.variable(
      .CPU, format: .NHWC, shape: [sequenceLength, AnimaAdapterConfig.sourceDimension],
      of: FloatType.self)
    let targetInputIDs = graph.variable(
      .CPU, format: .NHWC, shape: [sequenceLength], of: Int32.self)
    let adapterRot = graph.variable(
      makeHalfSplitRotary(
        tokenLength: sequenceLength, headDimension: AnimaAdapterConfig.headDimension,
        theta: AnimaAdapterConfig.ropeTheta))
    adapter.compile(inputs: sourceHiddenStates, targetInputIDs, adapterRot, adapterRot)
    adapterReader(adapterStateDict)
    let (dit, ditReader) = CosmosTransformer(
      latentHeight: latentHeight, latentWidth: latentWidth, textLength: sequenceLength)
    let hiddenStates = graph.variable(
      .CPU, format: .NHWC,
      shape: [batchSize, AnimaDiTConfig.inChannels, latentFrames, latentHeight, latentWidth],
      of: FloatType.self)
    let paddingMask = graph.variable(
      .CPU, format: .NHWC, shape: [batchSize, 1, latentHeight, latentWidth], of: FloatType.self)
    let context = graph.variable(
      .CPU, format: .NHWC, shape: [batchSize, sequenceLength, AnimaDiTConfig.textEmbedDimension],
      of: FloatType.self)
    let timestepProjection = graph.variable(
      timeEmbedding(
        timesteps: AnimaParityConfig.timestep, batchSize: batchSize,
        embeddingSize: AnimaDiTConfig.hiddenSize,
        maxPeriod: Int(AnimaAdapterConfig.ropeTheta)))
    let rot = graph.variable(
      makeCosmosRotary(
        frames: latentFrames, height: latentHeight / AnimaDiTConfig.patchHeight,
        width: latentWidth / AnimaDiTConfig.patchWidth, headDimension: AnimaDiTConfig.headDimension,
        ropeScale: (
          AnimaDiTConfig.ropeScale.temporal, AnimaDiTConfig.ropeScale.height,
          AnimaDiTConfig.ropeScale.width
        )))
    dit.compile(inputs: hiddenStates, paddingMask, context, timestepProjection, rot)
    ditReader(transformerStateDict)
    graph.openStore(AnimaPaths.mainOutputPath) {
      $0.write("llm_adapter", model: adapter)
      $0.write("dit", model: dit)
    }
    print("Wrote \(AnimaPaths.mainOutputPath)")
  }
}

let mode = CommandLine.arguments.dropFirst().first ?? "parity"

switch mode {
case "parity-text":
  if !runTextParity() { exit(2) }
case "parity-adapter":
  if !runAdapterParity() { exit(2) }
case "parity-dit":
  if !runTransformerParity() { exit(2) }
case "parity":
  if !runTextParity() { exit(2) }
  if !runAdapterParity() { exit(2) }
  if !runTransformerParity() { exit(2) }
case "text":
  exportTextModel()
case "main":
  exportMainModels()
case "all":
  exportTextModel()
  exportMainModels()
default:
  fputs("Usage: anima [parity-text|parity-adapter|parity-dit|parity|text|main|all]\n", stderr)
  exit(1)
}
