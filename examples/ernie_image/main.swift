import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit

typealias FloatType = Float16
typealias TextFloatType = Float16

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

let site = Python.import("site")
let sys = Python.import("sys")
let osPath = Python.import("os.path")

func movePythonPathToFront(_ path: String) {
  while Bool(sys.path.__contains__(path)) ?? false {
    sys.path.remove(path)
  }
  sys.path.insert(0, path)
}

func movePythonPathToBack(_ path: String) {
  while Bool(sys.path.__contains__(path)) ?? false {
    sys.path.remove(path)
  }
  sys.path.append(path)
}

var insertedVirtualEnvSitePackages = false
var preferredVirtualEnvSitePackages: String? = nil
if let virtualEnv = ProcessInfo.processInfo.environment["VIRTUAL_ENV"] {
  let libRoot = URL(fileURLWithPath: virtualEnv).appendingPathComponent("lib")
  if let pythonLibDirs = try? FileManager.default.contentsOfDirectory(
    at: libRoot, includingPropertiesForKeys: nil)
  {
    for pythonLibDir in pythonLibDirs.sorted(by: { $0.path < $1.path }) {
      let sitePackagesDir = pythonLibDir.appendingPathComponent("site-packages").path
      if FileManager.default.fileExists(atPath: sitePackagesDir) {
        movePythonPathToFront(sitePackagesDir)
        insertedVirtualEnvSitePackages = true
        preferredVirtualEnvSitePackages = sitePackagesDir
      }
    }
  }
}
let userSitePackages = String(site.getusersitepackages()) ?? ""
if Bool(osPath.isdir(userSitePackages)) ?? false {
  if insertedVirtualEnvSitePackages {
    movePythonPathToBack(userSitePackages)
  } else if (Bool(sys.path.__contains__(userSitePackages)) ?? false) == false {
    movePythonPathToFront(userSitePackages)
  }
}
let systemDistPackages = "/usr/lib/python3/dist-packages"
if insertedVirtualEnvSitePackages {
  movePythonPathToBack(systemDistPackages)
} else if (Bool(sys.path.__contains__(systemDistPackages)) ?? false) == false {
  movePythonPathToFront(systemDistPackages)
}
if let preferredVirtualEnvSitePackages {
  let currentPath = String(sys.path[0]) ?? ""
  if currentPath != preferredVirtualEnvSitePackages {
    movePythonPathToFront(preferredVirtualEnvSitePackages)
  }
}

let os = Python.import("os")
let builtins = Python.import("builtins")
let numpy = Python.import("numpy")
let tokenizers = Python.import("tokenizers")
let torch = Python.import("torch")
let transformers = Python.import("transformers")
let diffusers = Python.import("diffusers")
let diffusersModels = Python.import("diffusers.models")
let huggingfaceHub = Python.import("huggingface_hub")

if ProcessInfo.processInfo.environment["ERNIE_IMAGE_DEBUG_IMPORTS"] == "1" {
  print("ERNIE debug VIRTUAL_ENV:", ProcessInfo.processInfo.environment["VIRTUAL_ENV"] ?? "nil")
  print("ERNIE debug PYTHONHOME:", ProcessInfo.processInfo.environment["PYTHONHOME"] ?? "nil")
  print("ERNIE debug PYTHONPATH:", ProcessInfo.processInfo.environment["PYTHONPATH"] ?? "nil")
  print("ERNIE debug sys.executable:", sys.executable)
  print("ERNIE debug sys.prefix:", sys.__dict__["prefix"])
  print("ERNIE debug sys.path:", sys.path)
  print("ERNIE debug torch:", torch.__file__)
  print("ERNIE debug transformers:", transformers.__file__)
  print("ERNIE debug diffusers:", diffusers.__file__)
  exit(0)
}

torch.set_grad_enabled(false)
torch.manual_seed(42)
let hasCUDA = Bool(torch.cuda.is_available()) ?? false
if hasCUDA {
  torch.cuda.manual_seed_all(42)
} else {
  print("CUDA is not visible to Python in the current environment.")
  print("Run this target outside the sandbox or in a GPU-visible session for parity.")
  exit(1)
}

let modelRoot =
  ProcessInfo.processInfo.environment["ERNIE_IMAGE_MODEL"] ?? "baidu/ERNIE-Image-Turbo"
let prompt =
  ProcessInfo.processInfo.environment["ERNIE_IMAGE_PROMPT"]
  ?? "A poster on a city wall shows the words ERNIE IMAGE TURBO in large white letters."
let textReferenceTorchDType: PythonObject = torch.bfloat16
let exportModels = ProcessInfo.processInfo.environment["ERNIE_IMAGE_EXPORT"] == "1"
let textExportPath =
  ProcessInfo.processInfo.environment["ERNIE_IMAGE_TEXT_EXPORT_PATH"]
  ?? "/home/liu/workspace/swift-diffusion/ernie_image_turbo_text_model_f16.ckpt"
let ditExportPath =
  ProcessInfo.processInfo.environment["ERNIE_IMAGE_DIT_EXPORT_PATH"]
  ?? "/home/liu/workspace/swift-diffusion/ernie_image_turbo_dit_f16.ckpt"
let torchDevice = torch.device("cuda")
let isLocalModelPath = Bool(os.path.isdir(modelRoot)) ?? false
let hasErnieImageTransformer =
  Bool(builtins.hasattr(diffusersModels, "ErnieImageTransformer2DModel")) ?? false

func localOrHubPath(_ subpath: String) -> String {
  if isLocalModelPath {
    return "\(modelRoot)/\(subpath)"
  }
  return "\(huggingfaceHub.hf_hub_download(repo_id: modelRoot, filename: subpath))"
}

func copiedToCPU(_ tensor: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
  tensor.as(of: Float.self).rawValue.toCPU()
}

func maxAbsDiff2D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var maxDiff: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let diff = abs(Float(lhs[i, j]) - Float(rhs[i, j]))
      if diff > maxDiff {
        maxDiff = diff
      }
    }
  }
  return maxDiff
}

func maxRelativeDiff2D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var maxAbsDiff: Float = 0
  var maxAbsRef: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let ref = Float(rhs[i, j])
      let absDiff = abs(Float(lhs[i, j]) - ref)
      if absDiff > maxAbsDiff {
        maxAbsDiff = absDiff
      }
      let absRef = abs(ref)
      if absRef > maxAbsRef {
        maxAbsRef = absRef
      }
    }
  }
  return maxAbsDiff / max(maxAbsRef, 1e-6)
}

func maxAbsDiff4D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 4)
  precondition(rhs.shape.count == 4)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  precondition(lhs.shape[2] == rhs.shape[2])
  precondition(lhs.shape[3] == rhs.shape[3])
  var maxDiff: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      for k in 0..<lhs.shape[2] {
        for l in 0..<lhs.shape[3] {
          let diff = abs(Float(lhs[i, j, k, l]) - Float(rhs[i, j, k, l]))
          if diff > maxDiff {
            maxDiff = diff
          }
        }
      }
    }
  }
  return maxDiff
}

func maxRelativeDiff4D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 4)
  precondition(rhs.shape.count == 4)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  precondition(lhs.shape[2] == rhs.shape[2])
  precondition(lhs.shape[3] == rhs.shape[3])
  var maxAbsDiff: Float = 0
  var maxAbsRef: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      for k in 0..<lhs.shape[2] {
        for l in 0..<lhs.shape[3] {
          let ref = Float(rhs[i, j, k, l])
          let absDiff = abs(Float(lhs[i, j, k, l]) - ref)
          if absDiff > maxAbsDiff {
            maxAbsDiff = absDiff
          }
          let absRef = abs(ref)
          if absRef > maxAbsRef {
            maxAbsRef = absRef
          }
        }
      }
    }
  }
  return maxAbsDiff / max(maxAbsRef, 1e-6)
}

func diffusersTimestepEmbedding(timestep: Float, embeddingSize: Int, maxPeriod: Int = 10_000)
  -> Tensor<Float>
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .C(embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let exponent = -log(Float(maxPeriod)) * Float(i) / Float(half)
    let freq = exp(exponent)
    let value = timestep * freq
    embedding[i] = sin(value)
    embedding[i + half] = cos(value)
  }
  return embedding
}

func makeErnieImageRotTensor(textLength: Int, height: Int, width: Int)
  -> DynamicGraph.Tensor<FloatType>
{
  let totalTokens = textLength + height * width
  let rotTensor = graph.variable(.CPU, .NHWC(1, totalTokens, 1, 128), of: Float.self)
  for i in 0..<textLength {
    for k in 0..<16 {
      let theta = Double(i) / pow(256.0, Double(k) / 16.0)
      rotTensor[0, i, 0, k * 2] = Float(cos(theta))
      rotTensor[0, i, 0, k * 2 + 1] = Float(sin(theta))
    }
    for k in 0..<24 {
      rotTensor[0, i, 0, (k + 16) * 2] = 1
      rotTensor[0, i, 0, (k + 16) * 2 + 1] = 0
      rotTensor[0, i, 0, (k + 16 + 24) * 2] = 1
      rotTensor[0, i, 0, (k + 16 + 24) * 2 + 1] = 0
    }
  }
  for y in 0..<height {
    for x in 0..<width {
      let i = textLength + y * width + x
      for k in 0..<16 {
        let theta = Double(textLength) / pow(256.0, Double(k) / 16.0)
        rotTensor[0, i, 0, k * 2] = Float(cos(theta))
        rotTensor[0, i, 0, k * 2 + 1] = Float(sin(theta))
      }
      for k in 0..<24 {
        let thetaY = Double(y) / pow(256.0, Double(k) / 24.0)
        rotTensor[0, i, 0, (k + 16) * 2] = Float(cos(thetaY))
        rotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sin(thetaY))
      }
      for k in 0..<24 {
        let thetaX = Double(x) / pow(256.0, Double(k) / 24.0)
        rotTensor[0, i, 0, (k + 16 + 24) * 2] = Float(cos(thetaX))
        rotTensor[0, i, 0, (k + 16 + 24) * 2 + 1] = Float(sin(thetaX))
      }
    }
  }
  return DynamicGraph.Tensor<FloatType>(from: rotTensor).toGPU(0)
}

func TextFeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let up = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = up(x) .* gate(x).swish()
  let down = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = down(out)
  return (gate, down, up, Model([x], [out]))
}

func TextSelfAttention(prefix: String, width: Int, headDim: Int, heads: Int, kvHeads: Int, t: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: headDim * kvHeads, noBias: true, name: "k_proj")
  let toqueries = Dense(count: headDim * heads, noBias: true, name: "q_proj")
  let tovalues = Dense(count: headDim * kvHeads, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([1, t, kvHeads, headDim])
  var queries = toqueries(x).reshaped([1, t, heads, headDim])
  let values = tovalues(x).reshaped([1, t, kvHeads, headDim])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(headDim).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([t, heads * headDim])
  let unifyheads = Dense(count: width, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let qWeight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).view(
      heads, 2, headDim / 2, width
    ).transpose(1, 2).cpu().numpy()
    toqueries.weight.copy(from: Tensor<TextFloatType>(from: try! Tensor<Float>(numpy: qWeight)))
    let kWeight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).view(
      kvHeads, 2, headDim / 2, width
    ).transpose(1, 2).cpu().numpy()
    tokeys.weight.copy(from: Tensor<TextFloatType>(from: try! Tensor<Float>(numpy: kWeight)))
    let vWeight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu().numpy()
    tovalues.weight.copy(from: Tensor<TextFloatType>(from: try! Tensor<Float>(numpy: vWeight)))
    let projWeight = state_dict["\(prefix).self_attn.o_proj.weight"].type(torch.float).cpu().numpy()
    unifyheads.weight.copy(from: Tensor<TextFloatType>(from: try! Tensor<Float>(numpy: projWeight)))
  }
  return (Model([x, rot], [out]), reader)
}

func TextTransformerBlock(
  prefix: String, width: Int, headDim: Int, heads: Int, kvHeads: Int, t: Int, mlp: Int
) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x).to(TextFloatType.dataType)
  let (attention, attentionReader) = TextSelfAttention(
    prefix: prefix, width: width, headDim: headDim, heads: heads, kvHeads: kvHeads, t: t)
  out = attention(out, rot).to(of: x) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "post_attention_layernorm")
  out = norm2(out).to(TextFloatType.dataType)
  let (w1, w2, w3, ffn) = TextFeedForward(hiddenSize: width, intermediateSize: mlp, name: "mlp")
  out = residual + ffn(out).to(of: residual)
  let reader: (PythonObject) -> Void = { state_dict in
    attentionReader(state_dict)
    let norm1Weight = state_dict["\(prefix).input_layernorm.weight"].type(torch.float).cpu().numpy()
    norm1.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm1Weight)))
    let norm2Weight = state_dict["\(prefix).post_attention_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm2.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm2Weight)))
    let w1Weight = state_dict["\(prefix).mlp.gate_proj.weight"].type(torch.float).cpu().numpy()
    w1.weight.copy(from: Tensor<TextFloatType>(from: try! Tensor<Float>(numpy: w1Weight)))
    let w2Weight = state_dict["\(prefix).mlp.down_proj.weight"].type(torch.float).cpu().numpy()
    w2.weight.copy(from: Tensor<TextFloatType>(from: try! Tensor<Float>(numpy: w2Weight)))
    let w3Weight = state_dict["\(prefix).mlp.up_proj.weight"].type(torch.float).cpu().numpy()
    w3.weight.copy(from: Tensor<TextFloatType>(from: try! Tensor<Float>(numpy: w3Weight)))
  }
  return (Model([x, rot], [out]), reader)
}

func MistralTextModel(
  vocabularySize: Int, tokenLength: Int, width: Int, layers: Int, mlp: Int, heads: Int,
  kvHeads: Int, headDim: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let tokenEmbed = Embedding(
    TextFloatType.self, vocabularySize: vocabularySize, embeddingSize: width, name: "tok_embeddings"
  )
  var out = tokenEmbed(tokens).to(.Float32)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (layer, reader) = TextTransformerBlock(
      prefix: "layers.\(i)", width: width, headDim: headDim, heads: heads, kvHeads: kvHeads,
      t: tokenLength, mlp: mlp)
    out = layer(out, rot)
    readers.append(reader)
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  let finalHidden = norm(out).to(.Float32)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["embed_tokens.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: Tensor<TextFloatType>(from: try! Tensor<Float>(numpy: vocab)))
    for reader in readers {
      reader(state_dict)
    }
    let normWeight = state_dict["norm.weight"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: normWeight)))
  }
  return (Model([tokens, rot], [finalHidden]), reader)
}

func ErnieImageFeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let up = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = up(x) .* GELU()(gate(x))
  let down = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = down(out).to(.Float32)
  return (gate, down, up, Model([x], [out]))
}

func ErnieImageBlock(prefix: String, hiddenSize: Int, k: Int, h: Int, tokenLength: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let shiftMSA = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let shiftMLP = Input()
  let scaleMLP = Input()
  let gateMLP = Input()

  let attentionNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "attention_norm1")
  var out = attentionNorm(x)
  out = ((1 + scaleMSA).to(of: out) .* out + shiftMSA.to(of: out)).to(.Float16)
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  var keys = tokeys(out).reshaped([1, tokenLength, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries = toqueries(out).reshaped([1, tokenLength, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = tovalues(out).reshaped([1, tokenLength, h, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var attentionOut = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
    queries, keys, values
  ).reshaped([tokenLength, hiddenSize])
  let attentionOutProj = Dense(count: hiddenSize, noBias: true, name: "o")
  attentionOut = attentionOutProj(attentionOut).to(.Float32)
  out = x + gateMSA.to(of: attentionOut) .* attentionOut

  let mlpNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "attention_norm2")
  var mlpIn = mlpNorm(out)
  mlpIn = ((1 + scaleMLP).to(of: mlpIn) .* mlpIn + shiftMLP.to(of: mlpIn)).to(.Float16)
  let (w1, w2, w3, ffn) = ErnieImageFeedForward(
    hiddenSize: hiddenSize, intermediateSize: 12_288, name: "ffn")
  let mlpOut = ffn(mlpIn)
  out = out + gateMLP.to(of: mlpOut) .* mlpOut.to(of: out)

  let reader: (PythonObject) -> Void = { state_dict in
    let attentionNormWeight = state_dict["\(prefix).adaLN_sa_ln.weight"].type(torch.float).cpu()
      .numpy()
    attentionNorm.weight.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: attentionNormWeight)))
    let qWeight = state_dict["\(prefix).self_attention.to_q.weight"].type(torch.float).view(
      h, 2, k / 2, hiddenSize
    ).transpose(1, 2).cpu().numpy()
    toqueries.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: qWeight)))
    let kWeight = state_dict["\(prefix).self_attention.to_k.weight"].type(torch.float).view(
      h, 2, k / 2, hiddenSize
    ).transpose(1, 2).cpu().numpy()
    tokeys.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: kWeight)))
    let vWeight = state_dict["\(prefix).self_attention.to_v.weight"].type(torch.float).cpu().numpy()
    tovalues.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: vWeight)))
    let normKWeight = state_dict["\(prefix).self_attention.norm_k.weight"].type(torch.float).cpu()
      .numpy()
    normK.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: normKWeight)))
    let normQWeight = state_dict["\(prefix).self_attention.norm_q.weight"].type(torch.float).cpu()
      .numpy()
    normQ.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: normQWeight)))
    let outProjWeight = state_dict["\(prefix).self_attention.to_out.0.weight"].type(torch.float)
      .cpu().numpy()
    attentionOutProj.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: outProjWeight)))
    let mlpNormWeight = state_dict["\(prefix).adaLN_mlp_ln.weight"].type(torch.float).cpu().numpy()
    mlpNorm.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: mlpNormWeight)))
    let gateWeight = state_dict["\(prefix).mlp.gate_proj.weight"].type(torch.float).cpu().numpy()
    w1.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: gateWeight)))
    let upWeight = state_dict["\(prefix).mlp.up_proj.weight"].type(torch.float).cpu().numpy()
    w3.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: upWeight)))
    let downWeight = state_dict["\(prefix).mlp.linear_fc2.weight"].type(torch.float).cpu().numpy()
    w2.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: downWeight)))
  }
  return (
    Model([x, rot, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP], [out]),
    reader
  )
}

func ErnieImageModel(height: Int, width: Int, textLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let txt = Input()
  let rot = Input()
  let t = Input()

  let xEmbedder = Convolution(
    groups: 1, filters: 4_096, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    name: "x_embedder")
  var img = xEmbedder(x).permuted(0, 2, 3, 1).reshaped([height * width, 4_096]).to(.Float32)
  let textProj = Dense(count: 4_096, noBias: true, name: "c_embedder")
  let text = textProj(txt).to(.Float32)

  let timeMlp0 = Dense(count: 4_096, name: "time_embedder_0")
  let timeMlp2 = Dense(count: 4_096, name: "time_embedder_1")
  let c = timeMlp2(timeMlp0(t).swish())
  let temb = c.swish()
  let adaLNs = (0..<6).map { Dense(count: 4_096, name: "ada_ln_\($0)") }
  let shiftMSA = adaLNs[0](temb)
  let scaleMSA = adaLNs[1](temb)
  let gateMSA = adaLNs[2](temb)
  let shiftMLP = adaLNs[3](temb)
  let scaleMLP = adaLNs[4](temb)
  let gateMLP = adaLNs[5](temb)

  var out = Functional.concat(axis: 0, img, text)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<36 {
    let (block, reader) = ErnieImageBlock(
      prefix: "layers.\(i)", hiddenSize: 4_096, k: 128, h: 32,
      tokenLength: height * width + textLength)
    out = block(out, rot, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP)
    readers.append(reader)
  }

  let finalNorm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let finalScale = Dense(count: 4_096, name: "scale")
  let finalShift = Dense(count: 4_096, name: "shift")
  out = ((1 + finalScale(c)).to(of: out) .* finalNorm(out) + finalShift(c).to(of: out)).to(.Float16)
  let finalLinear = Dense(count: 128, name: "linear")
  out = finalLinear(out)
  let imageOnly = out.reshaped(
    [height * width, 128], offset: [0, 0], strides: [128, 1]
  ).reshaped([1, height, width, 128]).permuted(0, 3, 1, 2).copied().to(.Float32)

  let reader: (PythonObject) -> Void = { state_dict in
    let xWeight = state_dict["x_embedder.proj.weight"].type(torch.float).cpu().numpy()
    xEmbedder.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: xWeight)))
    let xBias = state_dict["x_embedder.proj.bias"].type(torch.float).cpu().numpy()
    xEmbedder.bias.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: xBias)))
    let textWeight = state_dict["text_proj.weight"].type(torch.float).cpu().numpy()
    textProj.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: textWeight)))
    let time0Weight = state_dict["time_embedding.linear_1.weight"].type(torch.float).cpu().numpy()
    timeMlp0.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: time0Weight)))
    let time0Bias = state_dict["time_embedding.linear_1.bias"].type(torch.float).cpu().numpy()
    timeMlp0.bias.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: time0Bias)))
    let time2Weight = state_dict["time_embedding.linear_2.weight"].type(torch.float).cpu().numpy()
    timeMlp2.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: time2Weight)))
    let time2Bias = state_dict["time_embedding.linear_2.bias"].type(torch.float).cpu().numpy()
    timeMlp2.bias.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: time2Bias)))
    let modulationWeight = state_dict["adaLN_modulation.1.weight"].type(torch.float).cpu().numpy()
    let modulationBias = state_dict["adaLN_modulation.1.bias"].type(torch.float).cpu().numpy()
    for i in 0..<6 {
      adaLNs[i].weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: modulationWeight[(4_096 * i)..<(4_096 * (i + 1)), ...])))
      adaLNs[i].bias.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: modulationBias[(4_096 * i)..<(4_096 * (i + 1))])))
    }
    for reader in readers {
      reader(state_dict)
    }
    let finalLinearWeight = state_dict["final_norm.linear.weight"].type(torch.float).cpu().numpy()
    let finalLinearBias = state_dict["final_norm.linear.bias"].type(torch.float).cpu().numpy()
    finalScale.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: finalLinearWeight[..<4_096, ...])))
    finalScale.bias.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: finalLinearBias[..<4_096])))
    finalShift.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: finalLinearWeight[4_096..<(4_096 * 2), ...])))
    finalShift.bias.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: finalLinearBias[4_096..<(4_096 * 2)])))
    let finalWeight = state_dict["final_linear.weight"].type(torch.float).cpu().numpy()
    finalLinear.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: finalWeight)))
    let finalBias = state_dict["final_linear.bias"].type(torch.float).cpu().numpy()
    finalLinear.bias.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: finalBias)))
  }
  return (Model([x, txt, rot, t], [imageOnly]), reader)
}

let textEncoder: PythonObject
if isLocalModelPath {
  textEncoder = transformers.AutoModel.from_pretrained(
    "\(modelRoot)/text_encoder", torch_dtype: textReferenceTorchDType, low_cpu_mem_usage: false)
} else {
  textEncoder = transformers.AutoModel.from_pretrained(
    modelRoot, subfolder: "text_encoder", torch_dtype: textReferenceTorchDType,
    low_cpu_mem_usage: false)
}
textEncoder.eval()
textEncoder.to("cuda")

var transformer: PythonObject = Python.None
if hasErnieImageTransformer {
  if isLocalModelPath {
    transformer = diffusersModels.ErnieImageTransformer2DModel.from_pretrained(
      "\(modelRoot)/transformer", torch_dtype: torch.bfloat16, low_cpu_mem_usage: false)
  } else {
    transformer = diffusersModels.ErnieImageTransformer2DModel.from_pretrained(
      modelRoot, subfolder: "transformer", torch_dtype: torch.bfloat16, low_cpu_mem_usage: false)
  }
  transformer.eval()
  transformer.to("cuda")
}

let backendTokenizer = tokenizers.Tokenizer.from_file(localOrHubPath("tokenizer/tokenizer.json"))
let encodedPrompt = backendTokenizer.encode(prompt, add_special_tokens: true)
let tokenIds = encodedPrompt.ids
let tokenIdsTensorCPU = try! Tensor<Int32>(numpy: numpy.array(tokenIds, dtype: numpy.int32))
let tokenCount = tokenIdsTensorCPU.shape[0]
let inputIds = torch.tensor(tokenIds, dtype: torch.long).unsqueeze(0).to(torchDevice)

let textOutputs = textEncoder(input_ids: inputIds, output_hidden_states: true)
let textFinalOutputReference = try! Tensor<Float>(
  numpy: textOutputs.hidden_states[-1][0].to(torch.float).cpu().numpy())
let textDitConditioningReference = try! Tensor<Float>(
  numpy: textOutputs.hidden_states[-2][0].to(torch.float).cpu().numpy())
let languageModel = textEncoder.language_model
let textForDit = textOutputs.hidden_states[-2].to(torch.bfloat16)
let textStateDict = languageModel.state_dict()
let positionIds = torch.arange(tokenCount, dtype: torch.long).unsqueeze(0).to(torchDevice)
let dummyHidden = torch.zeros([1, tokenCount, 3_072], dtype: torch.bfloat16).to(torchDevice)
let positionEmbeddings = languageModel.rotary_emb(dummyHidden, positionIds)
let textCos = try! Tensor<Float>(numpy: positionEmbeddings[0].to(torch.float).cpu().numpy())
let textSin = try! Tensor<Float>(numpy: positionEmbeddings[1].to(torch.float).cpu().numpy())

let textRotTensor = graph.variable(.CPU, .NHWC(1, tokenCount, 1, 128), of: Float.self)
for i in 0..<tokenCount {
  for k in 0..<64 {
    textRotTensor[0, i, 0, k * 2] = textCos[0, i, k]
    textRotTensor[0, i, 0, k * 2 + 1] = textSin[0, i, k]
  }
}

graph.withNoGrad {
  let tokenIdsTensor = graph.variable(.CPU, format: .NHWC, shape: [tokenCount], of: Int32.self)
  for i in 0..<tokenCount {
    tokenIdsTensor[i] = tokenIdsTensorCPU[i]
  }
  let tokenIdsTensorGPU = tokenIdsTensor.toGPU(0)
  let (textModel, textReader) = MistralTextModel(
    vocabularySize: 131_072, tokenLength: tokenCount, width: 3_072, layers: 26, mlp: 9_216,
    heads: 32, kvHeads: 8, headDim: 128)
  let textRotTensorGPU = DynamicGraph.Tensor<TextFloatType>(from: textRotTensor).toGPU(0)
  textModel.compile(inputs: tokenIdsTensorGPU, textRotTensorGPU)
  textReader(textStateDict)
  let swiftTextOutput = copiedToCPU(
    textModel(inputs: tokenIdsTensorGPU, textRotTensorGPU)[0].as(of: Float.self))
  print("ERNIE text encoder token count:", tokenCount)
  print("ERNIE text encoder output shape:", swiftTextOutput.shape)
  print("ERNIE text encoder max abs diff:", maxAbsDiff2D(swiftTextOutput, textFinalOutputReference))
  print(
    "ERNIE text encoder max rel diff:", maxRelativeDiff2D(swiftTextOutput, textFinalOutputReference)
  )
  if exportModels {
    graph.openStore(textExportPath) {
      $0.write("text_model", model: textModel)
    }
    print("Exported ERNIE text model:", textExportPath)
  }
}

if hasErnieImageTransformer {
  let ditStateDict = transformer.state_dict()
  let latentTorch = torch.randn([1, 128, 64, 64]).to(torch.bfloat16).to(torchDevice)
  let timestepValue: Float = 0.2
  let timestepTorch = torch.tensor([timestepValue], dtype: torch.bfloat16).to(torchDevice)
  let textLensTorch = torch.tensor([tokenCount], dtype: torch.long).to(torchDevice)
  let ditReference = try! Tensor<Float>(
    numpy: transformer(
      hidden_states: latentTorch, timestep: timestepTorch, text_bth: textForDit,
      text_lens: textLensTorch,
      return_dict: false
    )[0].to(torch.float).cpu().numpy())

  graph.withNoGrad {
    let xTensor = graph.variable(
      Tensor<FloatType>(from: try! Tensor<Float>(numpy: latentTorch.to(torch.float).cpu().numpy()))
        .toGPU(0)
    ).reshaped(.NCHW(1, 128, 64, 64))
    let txtTensor = graph.variable(
      Tensor<FloatType>(from: textDitConditioningReference).toGPU(0)
    ).reshaped(.WC(tokenCount, 3_072))
    let rotTensorGPU = makeErnieImageRotTensor(textLength: tokenCount, height: 64, width: 64)
    let tTensor = graph.variable(
      Tensor<FloatType>(
        from: diffusersTimestepEmbedding(
          timestep: timestepValue, embeddingSize: 4_096
        )
      ).toGPU(0)
    ).reshaped(.WC(1, 4_096))

    let (dit, ditReader) = ErnieImageModel(height: 64, width: 64, textLength: tokenCount)
    dit.compile(inputs: xTensor, txtTensor, rotTensorGPU, tTensor)
    ditReader(ditStateDict)
    let swiftDitOutput = copiedToCPU(
      dit(inputs: xTensor, txtTensor, rotTensorGPU, tTensor)[0].as(of: Float.self))
    print("ERNIE DiT output shape:", swiftDitOutput.shape)
    print("ERNIE DiT max abs diff:", maxAbsDiff4D(swiftDitOutput, ditReference))
    print("ERNIE DiT max rel diff:", maxRelativeDiff4D(swiftDitOutput, ditReference))
    if exportModels {
      graph.openStore(ditExportPath) {
        $0.write("dit", model: dit)
      }
      print("Exported ERNIE DiT:", ditExportPath)
    }
  }
} else {
  print(
    "Skipping ERNIE DiT parity: current diffusers install does not expose ErnieImageTransformer2DModel."
  )
  print("Upgrade diffusers to 0.36+ for the DiT phase.")
}
