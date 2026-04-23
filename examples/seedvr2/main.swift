import Diffusion
import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PNG
import PythonKit

typealias FloatType = Float

setbuf(stdout, nil)

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

func envInt(_ name: String, _ defaultValue: Int) -> Int {
  if let value = ProcessInfo.processInfo.environment[name], let intValue = Int(value) {
    return intValue
  }
  return defaultValue
}

func envFlag(_ name: String) -> Bool {
  ProcessInfo.processInfo.environment[name] == "1"
}

func envString(_ name: String, _ defaultValue: String) -> String {
  ProcessInfo.processInfo.environment[name] ?? defaultValue
}

let sys = Python.import("sys")
let osPath = Python.import("os.path")
let site = Python.import("site")

let repoRoot = "/home/liu/workspace/swift-diffusion"
let seedVRRoot = "/home/liu/workspace/SeedVR"
let seedVRExampleRoot = "\(repoRoot)/examples/seedvr2"

enum SeedVR2MLPKind {
  case swiglu
  case gelu
}

enum SeedVR2RotaryKind {
  case mmrope3d
  case pixelVideoOnly
}

struct SeedVR2DiTConfig {
  let name: String
  let configDir: String
  let checkpointRoot: String
  let ditCheckpoint: String
  let hiddenSize: Int
  let heads: Int
  let headDim: Int
  let layers: Int
  let sharedWeightStartLayer: Int?
  let lastLayerVidOnly: Bool
  let outputNormAda: Bool
  let rotaryKind: SeedVR2RotaryKind
  let rotaryDim: Int
  let mlpHiddenSize: Int
  let mlpKind: SeedVR2MLPKind

  var embeddingSize: Int { hiddenSize * 6 }

  func sharedWeights(layerIndex: Int) -> Bool {
    guard let sharedWeightStartLayer = sharedWeightStartLayer else {
      return false
    }
    return layerIndex >= sharedWeightStartLayer
  }
}

func seedVR2DiTConfig() -> SeedVR2DiTConfig {
  let modelSize = envString("SEEDVR2_MODEL_SIZE", "3B").uppercased()
  switch modelSize {
  case "7B":
    let checkpoint = envString("SEEDVR2_DIT_CHECKPOINT", "seedvr2_ema_7b.pth")
    return SeedVR2DiTConfig(
      name: "7B", configDir: "configs_7b", checkpointRoot: "\(seedVRRoot)/SeedVR2-7B",
      ditCheckpoint: checkpoint, hiddenSize: 3072, heads: 24, headDim: 128, layers: 36,
      sharedWeightStartLayer: nil, lastLayerVidOnly: false, outputNormAda: false,
      rotaryKind: .pixelVideoOnly, rotaryDim: 60, mlpHiddenSize: 12288, mlpKind: .gelu)
  default:
    let checkpoint = envString("SEEDVR2_DIT_CHECKPOINT", "seedvr2_ema_3b.pth")
    return SeedVR2DiTConfig(
      name: "3B", configDir: "configs_3b", checkpointRoot: "\(seedVRRoot)/SeedVR2-3B",
      ditCheckpoint: checkpoint, hiddenSize: 2560, heads: 20, headDim: 128, layers: 32,
      sharedWeightStartLayer: 10, lastLayerVidOnly: true, outputNormAda: true,
      rotaryKind: .mmrope3d, rotaryDim: 126, mlpHiddenSize: 6912, mlpKind: .swiglu)
  }
}

let seedVR2Config = seedVR2DiTConfig()
let seedVR2VAEExportPath = envString(
  "SEEDVR2_VAE_CKPT", "/fast/Data/seedvr2_3b_vae_f32.ckpt")
let seedVR2DiTExportPath = envString(
  "SEEDVR2_DIT_CKPT", "/fast/Data/seedvr2_\(seedVR2Config.name.lowercased())_dit_f32.ckpt")

let userSitePackages = site.getusersitepackages()
if (Bool(osPath.isdir(userSitePackages)) ?? false)
  && (Bool(sys.path.__contains__(userSitePackages)) ?? false) == false
{
  sys.path.insert(0, userSitePackages)
}
let systemDistPackages = "/usr/lib/python3/dist-packages"
if (Bool(osPath.isdir(systemDistPackages)) ?? false)
  && (Bool(sys.path.__contains__(systemDistPackages)) ?? false) == false
{
  sys.path.insert(0, systemDistPackages)
}
if (Bool(sys.path.__contains__(seedVRExampleRoot)) ?? false) == false {
  sys.path.insert(0, seedVRExampleRoot)
}
if (Bool(sys.path.__contains__(seedVRRoot)) ?? false) == false {
  sys.path.insert(0, seedVRRoot)
}

let torch = Python.import("torch")
let random = Python.import("random")
let numpy = Python.import("numpy")
let seedvr2Reference = Python.import("reference")
let processInfo = ProcessInfo.processInfo
let forceCPU = processInfo.environment["SEEDVR2_FORCE_CPU"] == "1"

torch.set_grad_enabled(false)

let hasCUDA = !forceCPU && (Bool(torch.cuda.is_available()) ?? false)
let torchDevice = hasCUDA ? torch.device("cuda") : torch.device("cpu")
let disableTF32 = processInfo.environment["SEEDVR2_DISABLE_TF32"] == "1"

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
if hasCUDA {
  torch.cuda.manual_seed_all(42)
  if disableTF32 {
    torch.backends.cuda.matmul.allow_tf32 = false
    torch.backends.cudnn.allow_tf32 = false
  }
}

let swiftDevice = 0

func logStep(_ message: String) {
  FileHandle.standardError.write(Data((message + "\n").utf8))
}

func placeOnDevice(_ tensor: DynamicGraph.Tensor<Float>) -> DynamicGraph.Tensor<Float> {
  if hasCUDA {
    return tensor.toGPU(swiftDevice)
  }
  return tensor
}

func placeOnDevice(_ tensor: DynamicGraph.Tensor<Float16>) -> DynamicGraph.Tensor<Float16> {
  if hasCUDA {
    return tensor.toGPU(swiftDevice)
  }
  return tensor
}

func copiedToCPU(_ tensor: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
  tensor.as(of: Float.self).rawValue.toCPU()
}

func copiedToCPU(_ tensor: DynamicGraph.Tensor<Float16>) -> Tensor<Float> {
  Tensor<Float>(from: tensor.as(of: Float16.self).rawValue.toCPU())
}

func rematerializeOnDevice(_ graph: DynamicGraph, _ tensor: Tensor<Float>)
  -> DynamicGraph.Tensor<Float>
{
  placeOnDevice(graph.variable(tensor))
}

func rematerializeOnDeviceFloat16(_ graph: DynamicGraph, _ tensor: Tensor<Float>)
  -> DynamicGraph.Tensor<Float16>
{
  placeOnDevice(graph.variable(Tensor<Float16>(from: tensor)))
}

func maxAbsDiff5D(_ swiftTensor: Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  precondition(swiftTensor.shape[4] == torchTensor.shape[4])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - Float(torchTensor[i, c, d, h, w]))
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
  }
  return maxDiff
}

func maxGlobalRelativeDiff5D(_ swiftTensor: Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  precondition(swiftTensor.shape[4] == torchTensor.shape[4])
  var maxDiff: Float = 0
  var maxMagnitude: Float = 1e-6
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let swift = Float(swiftTensor[i, c, d, h, w])
            let reference = Float(torchTensor[i, c, d, h, w])
            maxDiff = max(maxDiff, abs(swift - reference))
            maxMagnitude = max(maxMagnitude, abs(swift), abs(reference))
          }
        }
      }
    }
  }
  return maxDiff / maxMagnitude
}

func maxAbsDiff4DTensor(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 4)
  precondition(lhs.shape == rhs.shape)
  var maxDiff: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      for k in 0..<lhs.shape[2] {
        for l in 0..<lhs.shape[3] {
          maxDiff = max(maxDiff, abs(Float(lhs[i, j, k, l]) - Float(rhs[i, j, k, l])))
        }
      }
    }
  }
  return maxDiff
}

func maxGlobalRelativeDiff4DTensor(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 4)
  precondition(lhs.shape == rhs.shape)
  var maxDiff: Float = 0
  var maxMagnitude: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      for k in 0..<lhs.shape[2] {
        for l in 0..<lhs.shape[3] {
          let left = Float(lhs[i, j, k, l])
          let right = Float(rhs[i, j, k, l])
          maxDiff = max(maxDiff, abs(left - right))
          maxMagnitude = max(maxMagnitude, abs(left), abs(right))
        }
      }
    }
  }
  return maxDiff / max(maxMagnitude, 1e-6)
}

func seedVR2RotaryTensor(from freqs: Tensor<Float>) -> Tensor<Float> {
  precondition(freqs.shape.count == 2)
  let seqLen = freqs.shape[0]
  let rotDim = freqs.shape[1]
  precondition(rotDim % 2 == 0)
  var rot = Tensor<Float>(.CPU, .NHWC(1, seqLen, 1, rotDim))
  for i in 0..<seqLen {
    for j in 0..<(rotDim / 2) {
      let angle = Float(freqs[i, j * 2])
      rot[0, i, 0, j * 2] = cosf(angle)
      rot[0, i, 0, j * 2 + 1] = sinf(angle)
    }
  }
  return rot
}

func seedVR2FillMMRotaryAxis(
  _ rot: inout Tensor<Float>, token: Int, axis: Int, position: Int, rotaryDim: Int
) {
  precondition(rotaryDim % 6 == 0)
  let axisDim = rotaryDim / 3
  let axisOffset = axis * axisDim
  for k in 0..<(axisDim / 2) {
    let theta = Double(position) * 1.0 / pow(10_000, Double(k) * 2 / Double(axisDim))
    let sintheta = sin(theta)
    let costheta = cos(theta)
    rot[0, token, 0, axisOffset + k * 2] = Float(costheta)
    rot[0, token, 0, axisOffset + k * 2 + 1] = Float(sintheta)
  }
}

func seedVR2MMRotary3DWindowTensors(
  windows: [SeedVR2Window3D], txtLen: Int, rotaryDim: Int
) -> (vid: [Tensor<Float>], txt: [Tensor<Float>]) {
  var vid = [Tensor<Float>]()
  var txt = [Tensor<Float>]()
  for window in windows {
    var vidRot = Tensor<Float>(.CPU, .NHWC(1, window.tokenLength, 1, rotaryDim))
    var token = 0
    for t in 0..<window.tLength {
      for h in 0..<window.hLength {
        for w in 0..<window.wLength {
          seedVR2FillMMRotaryAxis(
            &vidRot, token: token, axis: 0, position: txtLen + t, rotaryDim: rotaryDim)
          seedVR2FillMMRotaryAxis(
            &vidRot, token: token, axis: 1, position: h, rotaryDim: rotaryDim)
          seedVR2FillMMRotaryAxis(
            &vidRot, token: token, axis: 2, position: w, rotaryDim: rotaryDim)
          token += 1
        }
      }
    }
    vid.append(vidRot)

    var txtRot = Tensor<Float>(.CPU, .NHWC(1, txtLen, 1, rotaryDim))
    for textIndex in 0..<txtLen {
      for axis in 0..<3 {
        seedVR2FillMMRotaryAxis(
          &txtRot, token: textIndex, axis: axis, position: textIndex, rotaryDim: rotaryDim)
      }
    }
    txt.append(txtRot)
  }
  return (vid, txt)
}

func seedVR2FillIdentityRotary(_ rot: inout Tensor<Float>, token: Int, rotaryDim: Int) {
  for k in 0..<(rotaryDim / 2) {
    rot[0, token, 0, k * 2] = 1
    rot[0, token, 0, k * 2 + 1] = 0
  }
}

func seedVR2PixelRotaryPosition(_ index: Int, length: Int) -> Double {
  if length <= 1 {
    return -1
  }
  return -1 + 2 * Double(index) / Double(length - 1)
}

func seedVR2FillPixelRotaryAxis(
  _ rot: inout Tensor<Float>, token: Int, axis: Int, index: Int, length: Int, rotaryDim: Int
) {
  precondition(rotaryDim % 6 == 0)
  let axisDim = rotaryDim / 3
  let axisOffset = axis * axisDim
  let pairs = axisDim / 2
  let position = seedVR2PixelRotaryPosition(index, length: length)
  for k in 0..<pairs {
    let multiplier = pairs == 1 ? 1 : 1 + 127 * Double(k) / Double(pairs - 1)
    let theta = position * multiplier * Double.pi
    let sintheta = sin(theta)
    let costheta = cos(theta)
    rot[0, token, 0, axisOffset + k * 2] = Float(costheta)
    rot[0, token, 0, axisOffset + k * 2 + 1] = Float(sintheta)
  }
}

func seedVR2PixelVideoRotaryWindowTensors(
  windows: [SeedVR2Window3D], txtLen: Int, rotaryDim: Int
) -> (vid: [Tensor<Float>], txt: [Tensor<Float>]) {
  var vid = [Tensor<Float>]()
  var txt = [Tensor<Float>]()
  for window in windows {
    var vidRot = Tensor<Float>(.CPU, .NHWC(1, window.tokenLength, 1, rotaryDim))
    var token = 0
    for t in 0..<window.tLength {
      for h in 0..<window.hLength {
        for w in 0..<window.wLength {
          seedVR2FillPixelRotaryAxis(
            &vidRot, token: token, axis: 0, index: t, length: window.tLength,
            rotaryDim: rotaryDim)
          seedVR2FillPixelRotaryAxis(
            &vidRot, token: token, axis: 1, index: h, length: window.hLength,
            rotaryDim: rotaryDim)
          seedVR2FillPixelRotaryAxis(
            &vidRot, token: token, axis: 2, index: w, length: window.wLength,
            rotaryDim: rotaryDim)
          token += 1
        }
      }
    }
    vid.append(vidRot)

    var txtRot = Tensor<Float>(.CPU, .NHWC(1, txtLen, 1, rotaryDim))
    for textIndex in 0..<txtLen {
      seedVR2FillIdentityRotary(&txtRot, token: textIndex, rotaryDim: rotaryDim)
    }
    txt.append(txtRot)
  }
  return (vid, txt)
}

func seedVR2RotaryWindowTensors(
  config: SeedVR2DiTConfig, windows: [SeedVR2Window3D], txtLen: Int
) -> (vid: [Tensor<Float>], txt: [Tensor<Float>]) {
  switch config.rotaryKind {
  case .mmrope3d:
    return seedVR2MMRotary3DWindowTensors(
      windows: windows, txtLen: txtLen, rotaryDim: config.rotaryDim)
  case .pixelVideoOnly:
    return seedVR2PixelVideoRotaryWindowTensors(
      windows: windows, txtLen: txtLen, rotaryDim: config.rotaryDim)
  }
}

func seedVR2PrintRotaryParity(
  _ name: String, swift: [Tensor<Float>], reference: [Tensor<Float>]
) {
  precondition(swift.count == reference.count)
  var maxAbs: Float = 0
  var maxRel: Float = 0
  for index in 0..<swift.count {
    maxAbs = max(maxAbs, maxAbsDiff4DTensor(swift[index], reference[index]))
    maxRel = max(maxRel, maxGlobalRelativeDiff4DTensor(swift[index], reference[index]))
  }
  print("\(name) rotary max abs diff:", maxAbs)
  print("\(name) rotary global max rel diff:", maxRel)
}

func seedVR2LoadWindowFreqs(
  _ probe: PythonObject, kind: String, count: Int
) -> [Tensor<Float>] {
  var freqs = [Tensor<Float>]()
  for index in 0..<count {
    let torchTensor = probe["window\(index)_\(kind)_freqs"].to(torch.float)
    freqs.append(seedVR2RotaryTensor(from: try! Tensor<Float>(numpy: torchTensor.cpu().numpy())))
  }
  return freqs
}

func maxAbsDiff2DTensor(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
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

func maxGlobalRelativeDiff2DTensor(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var maxDiff: Float = 0
  var maxRef: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let ref = Float(rhs[i, j])
      let diff = abs(Float(lhs[i, j]) - ref)
      if diff > maxDiff {
        maxDiff = diff
      }
      let refAbs = abs(ref)
      if refAbs > maxRef {
        maxRef = refAbs
      }
    }
  }
  return maxDiff / max(maxRef, 1e-6)
}

func seedVR2FiniteSummary(_ name: String, _ tensor: Tensor<Float>) {
  var finiteCount = 0
  var nonFiniteCount = 0
  var minValue = Float.greatestFiniteMagnitude
  var maxValue = -Float.greatestFiniteMagnitude

  func record(_ value: Float) {
    if value.isFinite {
      finiteCount += 1
      minValue = min(minValue, value)
      maxValue = max(maxValue, value)
    } else {
      nonFiniteCount += 1
    }
  }

  if tensor.shape.count == 2 {
    for i in 0..<tensor.shape[0] {
      for j in 0..<tensor.shape[1] {
        record(tensor[i, j])
      }
    }
  } else {
    precondition(tensor.shape.count == 5)
    for n in 0..<tensor.shape[0] {
      for c in 0..<tensor.shape[1] {
        for d in 0..<tensor.shape[2] {
          for h in 0..<tensor.shape[3] {
            for w in 0..<tensor.shape[4] {
              record(tensor[n, c, d, h, w])
            }
          }
        }
      }
    }
  }

  print(
    "\(name) finite:", finiteCount, "non-finite:", nonFiniteCount, "min:",
    finiteCount > 0 ? minValue : .nan, "max:", finiteCount > 0 ? maxValue : .nan)
}

func copyParameterFloat16(_ parameter: Model.Parameters, from tensor: Tensor<Float>) {
  parameter.copy(from: Tensor<Float16>(from: tensor))
}

func seedVR2TimeEmbedding(timestep: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int)
  -> Tensor<
    Float
  >
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timestep
    let sinFreq = sin(freq)
    let cosFreq = cos(freq)
    for j in 0..<batchSize {
      embedding[j, i] = sinFreq
      embedding[j, i + half] = cosFreq
    }
  }
  return embedding
}

func seedVR2TemporalReplicatePad(
  _ x: Model.IO, batchSize: Int, channels: Int, depth: Int, height: Int, width: Int, left: Int
) -> Model.IO {
  if left == 0 {
    return x
  }
  return x.padded(.replicate, begin: [0, 0, left, 0, 0], end: [0, 0, 0, 0, 0])
}

func seedVR2FrameWiseGroupNorm(
  _ x: Model.IO, channels: Int, depth: Int, height: Int, width: Int, name: String
) -> (GroupNorm, Model.IO) {
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: name)
  let frameWise = x.permuted(0, 2, 1, 3, 4).copied().reshaped([depth, channels, height, width])
  let normalized = norm(frameWise)
  let restored = normalized.reshaped([1, depth, channels, height, width]).permuted(0, 2, 1, 3, 4)
    .copied()
  return (norm, restored)
}

func SeedVR2ResnetBlock3D(
  prefix: String, inChannels: Int, outChannels: Int, depth: Int, height: Int, width: Int,
  flattenOutput: Bool = false
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (norm1, norm1Out) = seedVR2FrameWiseGroupNorm(
    x, channels: inChannels, depth: depth, height: height, width: width, name: "norm1")
  var out = norm1Out
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv1")
  out = conv1(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: inChannels, depth: depth, height: height, width: width, left: 2))
  let (norm2, norm2Out) = seedVR2FrameWiseGroupNorm(
    out, channels: outChannels, depth: depth, height: height, width: width, name: "norm2")
  out = norm2Out
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv2")
  out = conv2(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: outChannels, depth: depth, height: height, width: width, left: 2)
  )
  let convShortcut: Convolution?
  if inChannels != outChannels {
    let shortcut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "conv_shortcut")
    out = shortcut(x) + out
    convShortcut = shortcut
  } else {
    out = x + out
    convShortcut = nil
  }
  let reader: (PythonObject) -> Void = { stateDict in
    let norm1Weight = stateDict["\(prefix).norm1.weight"].to(torch.float).cpu().numpy()
    let norm1Bias = stateDict["\(prefix).norm1.bias"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1Weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1Bias))
    let conv1Weight = stateDict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1Bias = stateDict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1Weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1Bias))
    let norm2Weight = stateDict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2Bias = stateDict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2Weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2Bias))
    let conv2Weight = stateDict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2Bias = stateDict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2Weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2Bias))
    if let convShortcut = convShortcut {
      let shortcutWeight = stateDict["\(prefix).conv_shortcut.weight"].to(torch.float).cpu().numpy()
      let shortcutBias = stateDict["\(prefix).conv_shortcut.bias"].to(torch.float).cpu().numpy()
      convShortcut.weight.copy(from: try! Tensor<Float>(numpy: shortcutWeight))
      convShortcut.bias.copy(from: try! Tensor<Float>(numpy: shortcutBias))
    }
  }
  let output: Model.IO
  if flattenOutput {
    output = out.reshaped(.NC(1, outChannels * depth * height * width))
  } else {
    output = out
  }
  return (reader, Model([x], [output.copied()]))
}

func SeedVR2AttentionBlock2D(
  prefix: String, channels: Int, batchSize: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: "group_norm")
  var out = norm(x)
  let hw = height * width
  let toQueries = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), name: "to_q")
  let toKeys = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), name: "to_k")
  let toValues = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), name: "to_v")
  let q = ((1.0 / Float(channels).squareRoot()) * toQueries(out)).reshaped([
    batchSize, channels, hw,
  ])
  let k = toKeys(out).reshaped([batchSize, channels, hw])
  let v = toValues(out).reshaped([batchSize, channels, hw])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let toOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    name: "to_out")
  out = x + toOut(out.reshaped([batchSize, channels, height, width]))
  let reader: (PythonObject) -> Void = { stateDict in
    let normWeight = stateDict["\(prefix).group_norm.weight"].to(torch.float).cpu().numpy()
    let normBias = stateDict["\(prefix).group_norm.bias"].to(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: normWeight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: normBias))
    let qWeight = stateDict["\(prefix).to_q.weight"].to(torch.float).cpu().numpy()
    let qBias = stateDict["\(prefix).to_q.bias"].to(torch.float).cpu().numpy()
    toQueries.weight.copy(from: try! Tensor<Float>(numpy: qWeight))
    toQueries.bias.copy(from: try! Tensor<Float>(numpy: qBias))
    let kWeight = stateDict["\(prefix).to_k.weight"].to(torch.float).cpu().numpy()
    let kBias = stateDict["\(prefix).to_k.bias"].to(torch.float).cpu().numpy()
    toKeys.weight.copy(from: try! Tensor<Float>(numpy: kWeight))
    toKeys.bias.copy(from: try! Tensor<Float>(numpy: kBias))
    let vWeight = stateDict["\(prefix).to_v.weight"].to(torch.float).cpu().numpy()
    let vBias = stateDict["\(prefix).to_v.bias"].to(torch.float).cpu().numpy()
    toValues.weight.copy(from: try! Tensor<Float>(numpy: vWeight))
    toValues.bias.copy(from: try! Tensor<Float>(numpy: vBias))
    let outWeight = stateDict["\(prefix).to_out.0.weight"].to(torch.float).cpu().numpy()
    let outBias = stateDict["\(prefix).to_out.0.bias"].to(torch.float).cpu().numpy()
    toOut.weight.copy(from: try! Tensor<Float>(numpy: outWeight))
    toOut.bias.copy(from: try! Tensor<Float>(numpy: outBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DecoderMidBlock3D(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (resnet0Reader, resnet0) = SeedVR2ResnetBlock3D(
    prefix: "decoder.mid_block.resnets.0", inChannels: 512, outChannels: 512, depth: depth,
    height: height, width: width)
  var out = resnet0(x)
  let (attnReader, attn) = SeedVR2AttentionBlock2D(
    prefix: "decoder.mid_block.attentions.0", channels: 512, batchSize: depth, height: height,
    width: width)
  let frameWise = out.permuted(0, 2, 1, 3, 4).contiguous().reshaped([depth, 512, height, width])
  out = attn(frameWise).reshaped([1, depth, 512, height, width]).permuted(0, 2, 1, 3, 4)
    .contiguous()
  let (resnet1Reader, resnet1) = SeedVR2ResnetBlock3D(
    prefix: "decoder.mid_block.resnets.1", inChannels: 512, outChannels: 512, depth: depth,
    height: height, width: width)
  out = resnet1(out)
  let reader: (PythonObject) -> Void = { stateDict in
    resnet0Reader(stateDict)
    attnReader(stateDict)
    resnet1Reader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Upsample3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int, temporalUp: Bool,
  spatialUp: Bool
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let temporalRatio = temporalUp ? 2 : 1
  let spatialRatio = spatialUp ? 2 : 1
  let upscaleRatio = temporalRatio * spatialRatio * spatialRatio
  let upscaleConv = Convolution(
    groups: 1, filters: channels * upscaleRatio, filterSize: [1, 1, 1],
    hint: Hint(stride: [1, 1, 1]),
    name: "conv_up1")
  var out = upscaleConv(x)
  out = out.reshaped([1, spatialRatio, spatialRatio, temporalRatio, channels, depth, height, width])
    .permuted(0, 4, 5, 3, 6, 1, 7, 2).contiguous()
  let upDepthRaw = depth * temporalRatio
  let upHeight = height * spatialRatio
  let upWidth = width * spatialRatio
  out = out.reshaped([1, channels, upDepthRaw, upHeight, upWidth])
  if temporalUp {
    let first = out.reshaped(
      [1, channels, 1, upHeight, upWidth],
      strides: [
        channels * upDepthRaw * upHeight * upWidth, upDepthRaw * upHeight * upWidth,
        upHeight * upWidth, upWidth, 1,
      ]
    ).contiguous()
    if depth == 1 {
      out = first
    } else {
      let rest = out.reshaped(
        [1, channels, upDepthRaw - 2, upHeight, upWidth],
        offset: [0, 0, 2, 0, 0],
        strides: [
          channels * upDepthRaw * upHeight * upWidth, upDepthRaw * upHeight * upWidth,
          upHeight * upWidth, upWidth, 1,
        ]
      ).contiguous()
      out = Functional.concat(axis: 2, first, rest)
    }
  }
  let upDepth = temporalUp ? 1 + (depth - 1) * temporalRatio : upDepthRaw
  let conv = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_up2")
  out = conv(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: channels, depth: upDepth, height: upHeight, width: upWidth,
      left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let upscaleWeight = stateDict["\(prefix).upscale_conv.weight"].to(torch.float).cpu().numpy()
    let upscaleBias = stateDict["\(prefix).upscale_conv.bias"].to(torch.float).cpu().numpy()
    upscaleConv.weight.copy(from: try! Tensor<Float>(numpy: upscaleWeight))
    upscaleConv.bias.copy(from: try! Tensor<Float>(numpy: upscaleBias))
    let convWeight = stateDict["\(prefix).conv.weight"].to(torch.float).cpu().numpy()
    let convBias = stateDict["\(prefix).conv.bias"].to(torch.float).cpu().numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: convWeight))
    conv.bias.copy(from: try! Tensor<Float>(numpy: convBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DiTTextIn(hiddenSize: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let txtIn = Dense(count: hiddenSize, name: "c_embedder")
  let out = txtIn(x)
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["txt_in.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["txt_in.bias"].to(torch.float).cpu().numpy()
    txtIn.weight.copy(from: try! Tensor<Float>(numpy: weight))
    txtIn.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DiTTimeEmbedding(hiddenSize: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let projIn = Dense(count: hiddenSize, name: "time_proj_in")
  let projHid = Dense(count: hiddenSize, name: "time_proj_hid")
  let projOut = Dense(count: hiddenSize * 6, name: "time_proj_out")
  var out = projIn(x).swish()
  out = projHid(out).swish()
  out = projOut(out)
  let reader: (PythonObject) -> Void = { stateDict in
    let projInWeight = stateDict["emb_in.proj_in.weight"].to(torch.float).cpu().numpy()
    let projInBias = stateDict["emb_in.proj_in.bias"].to(torch.float).cpu().numpy()
    projIn.weight.copy(from: try! Tensor<Float>(numpy: projInWeight))
    projIn.bias.copy(from: try! Tensor<Float>(numpy: projInBias))
    let projHidWeight = stateDict["emb_in.proj_hid.weight"].to(torch.float).cpu().numpy()
    let projHidBias = stateDict["emb_in.proj_hid.bias"].to(torch.float).cpu().numpy()
    projHid.weight.copy(from: try! Tensor<Float>(numpy: projHidWeight))
    projHid.bias.copy(from: try! Tensor<Float>(numpy: projHidBias))
    let projOutWeight = stateDict["emb_in.proj_out.weight"].to(torch.float).cpu().numpy()
    let projOutBias = stateDict["emb_in.proj_out.bias"].to(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: projOutWeight))
    projOut.bias.copy(from: try! Tensor<Float>(numpy: projOutBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DiTPatchIn(
  hiddenSize: Int, frames: Int, latentHeight: Int, latentWidth: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let proj = Dense(count: hiddenSize, name: "x_embedder")
  var out = x.reshaped([frames, latentHeight, latentWidth, 33], format: .NHWC)
  out = out.reshaped(
    [frames, latentHeight / 2, 2, latentWidth / 2, 2, 33], format: .NHWC
  ).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    frames * (latentHeight / 2) * (latentWidth / 2), 132,
  ])
  out = proj(out)
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["vid_in.proj.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["vid_in.proj.bias"].to(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [out.copied()]))
}

func seedVR2EmbRow(_ x: Model.IO, offset: Int, width: Int) -> Model.IO {
  x.reshaped([1, width], offset: [offset, 0], strides: [1, 6]).contiguous()
}

func seedVR2FlattenHeads(_ x: Model.IO, seqLen: Int, heads: Int, headDim: Int) -> Model.IO {
  x.contiguous().reshaped([seqLen, heads * headDim])
}

func seedVR2ApplyRotary3D(
  _ x: Model.IO, rot: Model.IO, seqLen: Int, heads: Int, headDim: Int = 128, rotDim: Int = 126
) -> Model.IO {
  let x = x.contiguous().copied()
  let xRot = x.reshaped(
    [seqLen, heads, rotDim], offset: [0, 0, 0],
    strides: [heads * headDim, headDim, 1]
  ).contiguous().reshaped([1, seqLen, heads, rotDim])
  let xPass = x.reshaped(
    [seqLen, heads, headDim - rotDim], offset: [0, 0, rotDim],
    strides: [heads * headDim, headDim, 1]
  ).contiguous()
  let rotated = Functional.cmul(left: xRot, right: rot).reshaped([seqLen, heads, rotDim])
    .contiguous()
  return Functional.concat(axis: 2, rotated, xPass).contiguous()
}

func seedVR2StateTensorNC(_ stateDict: PythonObject, _ key: String, width: Int) -> Tensor<Float> {
  try! Tensor<Float>(numpy: stateDict[key].to(torch.float).view(1, width).cpu().numpy())
}

func seedVR2StateTensor(_ stateDict: PythonObject, _ key: String) -> Tensor<Float> {
  try! Tensor<Float>(numpy: stateDict[key].to(torch.float).cpu().numpy())
}

func seedVR2DiTBlockMod(
  blockIndex: Int, branch: String, layer: String, emb: Model.IO, hiddenSize: Int,
  sharedWeights: Bool, includeGate: Bool = true
) -> ((PythonObject) -> Void, [Model.IO]) {
  let layerOffset = layer == "attn" ? 0 : 3
  let shift = Parameter<Float>(
    .GPU(swiftDevice), .NC(1, hiddenSize), trainable: false,
    name: "block\(blockIndex)_\(branch)_\(layer)_shift")
  let scale = Parameter<Float>(
    .GPU(swiftDevice), .NC(1, hiddenSize), trainable: false,
    name: "block\(blockIndex)_\(branch)_\(layer)_scale")
  let gate =
    includeGate
    ? Parameter<Float>(
      .GPU(swiftDevice), .NC(1, hiddenSize), trainable: false,
      name: "block\(blockIndex)_\(branch)_\(layer)_gate")
    : nil
  var mod = [
    seedVR2EmbRow(emb, offset: layerOffset, width: hiddenSize) + shift,
    seedVR2EmbRow(emb, offset: layerOffset + 1, width: hiddenSize) + scale,
  ]
  if let gate = gate {
    mod.append(seedVR2EmbRow(emb, offset: layerOffset + 2, width: hiddenSize) + gate)
  }
  let reader: (PythonObject) -> Void = { stateDict in
    let specificPrefix = "blocks.\(blockIndex).ada.\(branch)"
    let sharedPrefix = "blocks.\(blockIndex).ada.all"
    let statePrefix = sharedWeights ? sharedPrefix : specificPrefix
    shift.weight.copy(
      from: seedVR2StateTensorNC(stateDict, "\(statePrefix).\(layer)_shift", width: hiddenSize))
    scale.weight.copy(
      from: seedVR2StateTensorNC(stateDict, "\(statePrefix).\(layer)_scale", width: hiddenSize))
    if let gate = gate {
      gate.weight.copy(
        from: seedVR2StateTensorNC(stateDict, "\(statePrefix).\(layer)_gate", width: hiddenSize))
    }
  }
  return (reader, mod)
}

func seedVR2DiTOutputMod(
  emb: Model.IO, hiddenSize: Int
) -> ((PythonObject) -> Void, [Model.IO]) {
  let shift = Parameter<Float>(
    .GPU(swiftDevice), .NC(1, hiddenSize), trainable: false, name: "vid_out_ada_shift")
  let scale = Parameter<Float>(
    .GPU(swiftDevice), .NC(1, hiddenSize), trainable: false, name: "vid_out_ada_scale")
  let mod = [
    seedVR2EmbRow(emb, offset: 0, width: hiddenSize) + shift,
    seedVR2EmbRow(emb, offset: 1, width: hiddenSize) + scale,
  ]
  let reader: (PythonObject) -> Void = { stateDict in
    shift.weight.copy(
      from: seedVR2StateTensorNC(stateDict, "vid_out_ada.out_shift", width: hiddenSize))
    scale.weight.copy(
      from: seedVR2StateTensorNC(stateDict, "vid_out_ada.out_scale", width: hiddenSize))
  }
  return (reader, mod)
}

struct SeedVR2Window3D {
  let tStart: Int
  let tLength: Int
  let hStart: Int
  let hLength: Int
  let wStart: Int
  let wLength: Int

  var tokenLength: Int { tLength * hLength * wLength }
}

func seedVR2FlatWindowSlice(
  _ x: Model.IO, start: Int, length: Int, heads: Int, headDim: Int
) -> Model.IO {
  x.reshaped(
    [length, heads, headDim], offset: [start, 0, 0], strides: [heads * headDim, headDim, 1]
  ).contiguous().copied()
}

func seedVR2WindowSlice3D(
  _ x: Model.IO, window: SeedVR2Window3D, frames: Int, height: Int, width: Int, heads: Int,
  headDim: Int
) -> Model.IO {
  x.reshaped(
    [window.tLength, window.hLength, window.wLength, heads, headDim],
    offset: [window.tStart, window.hStart, window.wStart, 0, 0],
    strides: [
      height * width * heads * headDim, width * heads * headDim, heads * headDim, headDim, 1,
    ]
  ).contiguous().copied().reshaped([window.tokenLength, heads, headDim])
    .contiguous()
}

func seedVR2Concat(_ xs: [Model.IO], axis: Int) -> Model.IO {
  precondition(!xs.isEmpty)
  if xs.count == 1 {
    return xs[0].copied()
  }
  return Concat(axis: axis)(xs.map { $0.copied() }).contiguous()
}

func seedVR2ReverseWindowOutputs(
  _ windowOutputs: [Model.IO], windows: [SeedVR2Window3D],
  frames: Int, height: Int, width: Int, heads: Int, headDim: Int
) -> Model.IO {
  precondition(windowOutputs.count == windows.count)
  var windowIndex = 0
  var widthTiles = [Model.IO]()
  while windowIndex < windows.count {
    let wStart = windows[windowIndex].wStart
    let wLength = windows[windowIndex].wLength
    var heightTiles = [Model.IO]()
    while windowIndex < windows.count && windows[windowIndex].wStart == wStart
      && windows[windowIndex].wLength == wLength
    {
      let hStart = windows[windowIndex].hStart
      let hLength = windows[windowIndex].hLength
      var timeTiles = [Model.IO]()
      while windowIndex < windows.count && windows[windowIndex].wStart == wStart
        && windows[windowIndex].wLength == wLength && windows[windowIndex].hStart == hStart
        && windows[windowIndex].hLength == hLength
      {
        let window = windows[windowIndex]
        timeTiles.append(
          windowOutputs[windowIndex].reshaped(
            [window.tLength, window.hLength, window.wLength, heads, headDim]
          ).contiguous())
        windowIndex += 1
      }
      heightTiles.append(seedVR2Concat(timeTiles, axis: 0))
    }
    widthTiles.append(seedVR2Concat(heightTiles, axis: 1))
  }
  return seedVR2Concat(widthTiles, axis: 2).reshaped([frames * height * width, heads, headDim])
    .contiguous()
}

func seedVR2BatchedWindowAttentionSDPA(
  vidQ: [Model.IO], vidK: [Model.IO], vidV: [Model.IO],
  txtQ: [Model.IO], txtK: [Model.IO], txtV: Model.IO,
  windowLen: Int, txtLen: Int, heads: Int, headDim: Int, returnText: Bool = true
) -> (Model.IO, Model.IO?) {
  let batch = vidQ.count
  precondition(batch > 0)
  precondition(vidK.count == batch && vidV.count == batch)
  precondition(txtK.count == batch)
  precondition(!returnText || txtQ.count == batch)
  let batchedVidQ = seedVR2Concat(vidQ, axis: 0).reshaped(
    [batch, windowLen, heads, headDim]
  ).contiguous()
  let batchedVidK = seedVR2Concat(vidK, axis: 0).reshaped(
    [batch, windowLen, heads, headDim]
  ).contiguous()
  let batchedVidV = seedVR2Concat(vidV, axis: 0).reshaped(
    [batch, windowLen, heads, headDim]
  ).contiguous()
  let batchedTxtK = seedVR2Concat(txtK, axis: 0).reshaped(
    [batch, txtLen, heads, headDim]
  ).contiguous()
  let batchedTxtV = seedVR2Concat(vidQ.map { _ in txtV }, axis: 0).reshaped(
    [batch, txtLen, heads, headDim]
  ).contiguous()
  let k = Functional.concat(axis: 1, batchedVidK, batchedTxtK).contiguous()
  let v = Functional.concat(axis: 1, batchedVidV, batchedTxtV).contiguous()
  if !returnText {
    let vidOnlyOut = ScaledDotProductAttention(
      scale: 1.0 / Float(headDim).squareRoot(), flags: [.Float16])(
        batchedVidQ, k, v
      ).contiguous().reshaped([batch * windowLen, heads, headDim]).contiguous()
    return (vidOnlyOut, nil)
  }
  let batchedTxtQ = seedVR2Concat(txtQ, axis: 0).reshaped(
    [batch, txtLen, heads, headDim]
  ).contiguous()
  let q = Functional.concat(axis: 1, batchedVidQ, batchedTxtQ).contiguous()
  let out = ScaledDotProductAttention(scale: 1.0 / Float(headDim).squareRoot(), flags: [.Float16])(
    q, k, v
  ).contiguous()
  let queryLen = windowLen + txtLen
  let vidOut = out.reshaped(
    [batch, windowLen, heads, headDim], offset: [0, 0, 0, 0],
    strides: [queryLen * heads * headDim, heads * headDim, headDim, 1]
  ).contiguous().copied().reshaped([batch * windowLen, heads, headDim])
    .contiguous()
  let txtOut = out.reshaped(
    [batch, txtLen, heads, headDim], offset: [0, windowLen, 0, 0],
    strides: [queryLen * heads * headDim, heads * headDim, headDim, 1]
  ).contiguous().copied().reduced(.mean, axis: [0]).copied()
  return (vidOut, txtOut)
}

func seedVR2WindowAttentionSDPA(
  vidQ: Model.IO, vidK: Model.IO, vidV: Model.IO,
  txtQ: Model.IO, txtK: Model.IO, txtV: Model.IO,
  vidFreqs: [Model.IO], txtFreqs: [Model.IO],
  windows: [SeedVR2Window3D], frames: Int, height: Int, width: Int, txtLen: Int,
  heads: Int, headDim: Int, rotaryDim: Int, rotateText: Bool, returnText: Bool = true
) -> (Model.IO, Model.IO?) {
  let windowCount = windows.count
  precondition(windowCount > 0)
  precondition(vidFreqs.count == windowCount)
  precondition(!rotateText || txtFreqs.count == windowCount)

  var windowVidQ = [Model.IO]()
  var windowVidK = [Model.IO]()
  var windowVidV = [Model.IO]()
  var windowTxtQ = [Model.IO]()
  var windowTxtK = [Model.IO]()
  for windowIndex in 0..<windowCount {
    let window = windows[windowIndex]
    windowVidQ.append(
      seedVR2ApplyRotary3D(
        seedVR2WindowSlice3D(
          vidQ, window: window, frames: frames, height: height, width: width, heads: heads,
          headDim: headDim),
        rot: vidFreqs[windowIndex], seqLen: window.tokenLength, heads: heads,
        headDim: headDim, rotDim: rotaryDim))
    windowVidK.append(
      seedVR2ApplyRotary3D(
        seedVR2WindowSlice3D(
          vidK, window: window, frames: frames, height: height, width: width, heads: heads,
          headDim: headDim),
        rot: vidFreqs[windowIndex], seqLen: window.tokenLength, heads: heads,
        headDim: headDim, rotDim: rotaryDim))
    windowVidV.append(
      seedVR2WindowSlice3D(
        vidV, window: window, frames: frames, height: height, width: width, heads: heads,
        headDim: headDim))
    if rotateText {
      if returnText {
        windowTxtQ.append(
          seedVR2ApplyRotary3D(
            txtQ, rot: txtFreqs[windowIndex], seqLen: txtLen, heads: heads, headDim: headDim,
            rotDim: rotaryDim))
      }
      windowTxtK.append(
        seedVR2ApplyRotary3D(
          txtK, rot: txtFreqs[windowIndex], seqLen: txtLen, heads: heads, headDim: headDim,
          rotDim: rotaryDim))
    } else {
      if returnText {
        windowTxtQ.append(txtQ)
      }
      windowTxtK.append(txtK)
    }
  }

  var windowVidOuts = [Model.IO]()
  var groupTxtOuts = [Model.IO]()
  var groupCounts = [Int]()
  var groupStart = 0
  while groupStart < windowCount {
    let windowLen = windows[groupStart].tokenLength
    var groupEnd = groupStart + 1
    while groupEnd < windowCount && windows[groupEnd].tokenLength == windowLen {
      groupEnd += 1
    }
    let groupRange = groupStart..<groupEnd
    let (groupVidOut, groupTxtOut) = seedVR2BatchedWindowAttentionSDPA(
      vidQ: groupRange.map { windowVidQ[$0] },
      vidK: groupRange.map { windowVidK[$0] },
      vidV: groupRange.map { windowVidV[$0] },
      txtQ: returnText ? groupRange.map { windowTxtQ[$0] } : [],
      txtK: groupRange.map { windowTxtK[$0] },
      txtV: txtV,
      windowLen: windowLen, txtLen: txtLen, heads: heads, headDim: headDim,
      returnText: returnText)
    for groupWindowIndex in 0..<(groupEnd - groupStart) {
      windowVidOuts.append(
        seedVR2FlatWindowSlice(
          groupVidOut, start: groupWindowIndex * windowLen, length: windowLen, heads: heads,
          headDim: headDim))
    }
    if let groupTxtOut = groupTxtOut {
      groupTxtOuts.append(groupTxtOut)
      groupCounts.append(groupEnd - groupStart)
    }
    groupStart = groupEnd
  }

  let txtOut: Model.IO?
  if returnText {
    var reducedTxtOut = groupTxtOuts[0] * (Float(groupCounts[0]) / Float(windowCount))
    for groupIndex in 1..<groupTxtOuts.count {
      reducedTxtOut =
        reducedTxtOut + groupTxtOuts[groupIndex]
        * (Float(groupCounts[groupIndex]) / Float(windowCount))
    }
    txtOut = reducedTxtOut.copied()
  } else {
    txtOut = nil
  }
  return (
    seedVR2ReverseWindowOutputs(
      windowVidOuts, windows: windows, frames: frames, height: height, width: width, heads: heads,
      headDim: headDim),
    txtOut
  )
}

func seedVR2CeilDiv(_ x: Int, _ y: Int) -> Int {
  (x + y - 1) / y
}

func seedVR2PythonRoundToInt(_ x: Double) -> Int {
  let lower = floor(x)
  let fraction = x - lower
  if fraction < 0.5 {
    return Int(lower)
  }
  if fraction > 0.5 {
    return Int(lower + 1)
  }
  let lowerInt = Int(lower)
  return lowerInt % 2 == 0 ? lowerInt : lowerInt + 1
}

func seedVR2WindowRanges(size: Int, windowSize: Int, shifted: Bool) -> [(start: Int, length: Int)] {
  if shifted {
    if windowSize >= size {
      return [(0, size)]
    }
    let shift = 0.5
    let windowCount = Int(ceil((Double(size) - shift) / Double(windowSize))) + 1
    var ranges = [(start: Int, length: Int)]()
    for windowIndex in 0..<windowCount {
      let start = max(Int((Double(windowIndex) - shift) * Double(windowSize)), 0)
      let end = min(Int((Double(windowIndex) - shift + 1) * Double(windowSize)), size)
      if end > start {
        ranges.append((start, end - start))
      }
    }
    return ranges
  }
  var ranges = [(start: Int, length: Int)]()
  let windowCount = seedVR2CeilDiv(size, windowSize)
  for windowIndex in 0..<windowCount {
    let start = windowIndex * windowSize
    let end = min((windowIndex + 1) * windowSize, size)
    if end > start {
      ranges.append((start, end - start))
    }
  }
  return ranges
}

func seedVR2WindowSpec3D(frames: Int, height: Int, width: Int, shifted: Bool) -> [SeedVR2Window3D] {
  let scale = sqrt(Double(45 * 80) / Double(height * width))
  let resizedHeight = seedVR2PythonRoundToInt(Double(height) * scale)
  let resizedWidth = seedVR2PythonRoundToInt(Double(width) * scale)
  let windowFrames = seedVR2CeilDiv(min(frames, 30), 4)
  let windowHeight = seedVR2CeilDiv(resizedHeight, 3)
  let windowWidth = seedVR2CeilDiv(resizedWidth, 3)
  let timeRanges = seedVR2WindowRanges(size: frames, windowSize: windowFrames, shifted: shifted)
  let heightRanges = seedVR2WindowRanges(size: height, windowSize: windowHeight, shifted: shifted)
  let widthRanges = seedVR2WindowRanges(size: width, windowSize: windowWidth, shifted: shifted)
  var windows = [SeedVR2Window3D]()
  for widthRange in widthRanges {
    for heightRange in heightRanges {
      for timeRange in timeRanges {
        windows.append(
          SeedVR2Window3D(
            tStart: timeRange.start, tLength: timeRange.length,
            hStart: heightRange.start, hLength: heightRange.length,
            wStart: widthRange.start, wLength: widthRange.length))
      }
    }
  }
  return windows
}

func SeedVR2DiTBlock(
  config: SeedVR2DiTConfig, layerIndex: Int, windows: [SeedVR2Window3D], frames: Int,
  height: Int, width: Int, txtLen: Int, lastLayer: Bool = false, returnText: Bool = true
) -> ((PythonObject) -> Void, Model) {
  precondition(!windows.isEmpty)
  let vid = Input()
  let txt = Input()
  let emb = Input()
  let windowVidFreqs = windows.map { _ in Input() }
  let rotateText = config.rotaryKind == .mmrope3d
  let windowTxtFreqs = rotateText ? windows.map { _ in Input() } : []
  let hiddenSize = config.hiddenSize
  let heads = config.heads
  let headDim = config.headDim
  let vidLen = frames * height * width
  let sharedWeights = config.sharedWeights(layerIndex: layerIndex)
  var readers = [(PythonObject) -> Void]()

  let (vidAttnModReader, vidAttnMod) = seedVR2DiTBlockMod(
    blockIndex: layerIndex, branch: "vid", layer: "attn", emb: emb, hiddenSize: hiddenSize,
    sharedWeights: sharedWeights)
  readers.append(vidAttnModReader)
  let (vidMlpModReader, vidMlpMod) = seedVR2DiTBlockMod(
    blockIndex: layerIndex, branch: "vid", layer: "mlp", emb: emb, hiddenSize: hiddenSize,
    sharedWeights: sharedWeights)
  readers.append(vidMlpModReader)

  let vidAttnNorm = RMSNorm(
    epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "vid_attn_norm")
  let txtAttnNorm = RMSNorm(
    epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "txt_attn_norm")
  let vidAttnIn = (vidAttnNorm(vid) .* vidAttnMod[1] + vidAttnMod[0]).copied()
  let txtAttnIn: Model.IO
  let txtAttnGate: Model.IO?
  let txtMlpMod: [Model.IO]?
  if lastLayer {
    txtAttnIn = txtAttnNorm(txt).copied()
    txtAttnGate = nil
    txtMlpMod = nil
  } else {
    let (txtAttnModReader, txtAttnMod) = seedVR2DiTBlockMod(
      blockIndex: layerIndex, branch: "txt", layer: "attn", emb: emb, hiddenSize: hiddenSize,
      sharedWeights: sharedWeights, includeGate: returnText)
    readers.append(txtAttnModReader)
    txtAttnIn = (txtAttnNorm(txt) .* txtAttnMod[1] + txtAttnMod[0]).copied()
    txtAttnGate = returnText ? txtAttnMod[2] : nil
    if returnText {
      let (txtMlpModReader, txtMlpModValue) = seedVR2DiTBlockMod(
        blockIndex: layerIndex, branch: "txt", layer: "mlp", emb: emb, hiddenSize: hiddenSize,
        sharedWeights: sharedWeights)
      readers.append(txtMlpModReader)
      txtMlpMod = txtMlpModValue
    } else {
      txtMlpMod = nil
    }
  }

  let vidQProj = Dense(count: hiddenSize, noBias: true, name: "vid_q")
  let vidKProj = Dense(count: hiddenSize, noBias: true, name: "vid_k")
  let vidVProj = Dense(count: hiddenSize, noBias: true, name: "vid_v")
  let txtQProj = Dense(count: hiddenSize, noBias: true, name: "txt_q")
  let txtKProj = Dense(count: hiddenSize, noBias: true, name: "txt_k")
  let txtVProj = Dense(count: hiddenSize, noBias: true, name: "txt_v")
  let vidNormQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "vid_norm_q")
  let vidNormK = RMSNorm(epsilon: 1e-5, axis: [2], name: "vid_norm_k")
  let txtNormQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "txt_norm_q")
  let txtNormK = RMSNorm(epsilon: 1e-5, axis: [2], name: "txt_norm_k")

  let vidAttnProjIn = vidAttnIn.to(.Float16).copied()
  let txtAttnProjIn = txtAttnIn.to(.Float16).copied()
  let vidQ = vidNormQ(vidQProj(vidAttnProjIn).reshaped([vidLen, heads, headDim])).copied()
  let vidK = vidNormK(vidKProj(vidAttnProjIn).reshaped([vidLen, heads, headDim])).copied()
  let vidV = vidVProj(vidAttnProjIn).reshaped([vidLen, heads, headDim]).copied()
  let txtK = txtNormK(txtKProj(txtAttnProjIn).reshaped([txtLen, heads, headDim])).copied()
  let txtV = txtVProj(txtAttnProjIn).reshaped([txtLen, heads, headDim]).copied()
  let txtQ =
    returnText
    ? txtNormQ(txtQProj(txtAttnProjIn).reshaped([txtLen, heads, headDim])).copied() : txtK

  let (vidAttn3D, txtAttn3D) = seedVR2WindowAttentionSDPA(
    vidQ: vidQ, vidK: vidK, vidV: vidV, txtQ: txtQ, txtK: txtK, txtV: txtV,
    vidFreqs: windowVidFreqs, txtFreqs: windowTxtFreqs, windows: windows, frames: frames,
    height: height, width: width, txtLen: txtLen, heads: heads, headDim: headDim,
    rotaryDim: config.rotaryDim, rotateText: rotateText, returnText: returnText)
  let vidAttnFlat = seedVR2FlattenHeads(vidAttn3D, seqLen: vidLen, heads: heads, headDim: headDim)
    .copied()
  let vidAttnOut = Dense(count: hiddenSize, name: "vid_attn_out")
  let vidAttnProjected = vidAttnOut(vidAttnFlat).to(.Float32).copied()
  let txtAttnOut: Model?
  let txtAfterAttn: Model.IO?
  if let txtAttn3D = txtAttn3D {
    let txtAttnFlat = seedVR2FlattenHeads(
      txtAttn3D, seqLen: txtLen, heads: heads, headDim: headDim
    ).copied()
    let projection = Dense(count: hiddenSize, name: "txt_attn_out")
    let txtAttnProjected = projection(txtAttnFlat).to(.Float32).copied()
    txtAfterAttn = (txt + txtAttnProjected .* txtAttnGate!).copied()
    txtAttnOut = projection
  } else {
    txtAfterAttn = nil
    txtAttnOut = nil
  }
  let vidAfterAttn = (vid + vidAttnProjected .* vidAttnMod[2]).copied()

  let vidMlpNorm = RMSNorm(epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "vid_mlp_norm")
  let txtMlpNorm = RMSNorm(epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "txt_mlp_norm")
  let vidMlpIn = (vidMlpNorm(vidAfterAttn) .* vidMlpMod[1] + vidMlpMod[0]).copied()
  let txtMlpIn: Model.IO?
  if let txtAfterAttn = txtAfterAttn, let txtMlpMod = txtMlpMod {
    txtMlpIn = (txtMlpNorm(txtAfterAttn) .* txtMlpMod[1] + txtMlpMod[0]).copied()
  } else {
    txtMlpIn = nil
  }

  let vidFinal: Model.IO
  let txtFinal: Model.IO?
  switch config.mlpKind {
  case .swiglu:
    let vidMlpGate = Dense(count: config.mlpHiddenSize, noBias: true, name: "vid_mlp_gate")
    let vidMlpInProj = Dense(count: config.mlpHiddenSize, noBias: true, name: "vid_mlp_in")
    let vidMlpOutProj = Dense(count: hiddenSize, noBias: true, name: "vid_mlp_out")
    let vidMlpProjIn = vidMlpIn.to(.Float16).copied()
    let vidGate = vidMlpGate(vidMlpProjIn).copied()
    let vidInner = vidMlpInProj(vidMlpProjIn).copied()
    let vidMlpOut = vidMlpOutProj(((vidGate .* vidGate.sigmoid()) .* vidInner).copied())
      .to(.Float32).copied()
    vidFinal = (vidAfterAttn + vidMlpOut .* vidMlpMod[2]).to(of: vidAfterAttn)

    let txtMlpGate: Model?
    let txtMlpInProj: Model?
    let txtMlpOutProj: Model?
    if let txtMlpIn = txtMlpIn, let txtMlpMod = txtMlpMod {
      let gate = Dense(count: config.mlpHiddenSize, noBias: true, name: "txt_mlp_gate")
      let inProj = Dense(count: config.mlpHiddenSize, noBias: true, name: "txt_mlp_in")
      let outProj = Dense(count: hiddenSize, noBias: true, name: "txt_mlp_out")
      let txtMlpProjIn = txtMlpIn.to(.Float16).copied()
      let txtGate = gate(txtMlpProjIn).copied()
      let txtInner = inProj(txtMlpProjIn).copied()
      let txtMlpOut = outProj(((txtGate .* txtGate.sigmoid()) .* txtInner).copied())
        .to(.Float32).copied()
      txtFinal = (txtAfterAttn! + txtMlpOut .* txtMlpMod[2]).to(of: txtAfterAttn!)
      txtMlpGate = gate
      txtMlpInProj = inProj
      txtMlpOutProj = outProj
    } else {
      txtFinal = nil
      txtMlpGate = nil
      txtMlpInProj = nil
      txtMlpOutProj = nil
    }

    readers.append { stateDict in
      let prefix = "blocks.\(layerIndex)"
      if sharedWeights {
        let allQKV = stateDict["\(prefix).attn.proj_qkv.all.weight"].to(torch.float).cpu()
        let qWeight = try! Tensor<Float>(numpy: allQKV[..<hiddenSize, ...].numpy())
        let kWeight = try! Tensor<Float>(numpy: allQKV[hiddenSize..<(2 * hiddenSize), ...].numpy())
        let vWeight = try! Tensor<Float>(numpy: allQKV[(2 * hiddenSize)..., ...].numpy())
        let allNormQ = seedVR2StateTensor(stateDict, "\(prefix).attn.norm_q.all.weight")
        let allNormK = seedVR2StateTensor(stateDict, "\(prefix).attn.norm_k.all.weight")
        copyParameterFloat16(vidQProj.weight, from: qWeight)
        copyParameterFloat16(vidKProj.weight, from: kWeight)
        copyParameterFloat16(vidVProj.weight, from: vWeight)
        if returnText {
          copyParameterFloat16(txtQProj.weight, from: qWeight)
        }
        copyParameterFloat16(txtKProj.weight, from: kWeight)
        copyParameterFloat16(txtVProj.weight, from: vWeight)
        copyParameterFloat16(vidNormQ.weight, from: allNormQ)
        copyParameterFloat16(vidNormK.weight, from: allNormK)
        if returnText {
          copyParameterFloat16(txtNormQ.weight, from: allNormQ)
        }
        copyParameterFloat16(txtNormK.weight, from: allNormK)
      } else {
        let vidQKV = stateDict["\(prefix).attn.proj_qkv.vid.weight"].to(torch.float).cpu()
        let txtQKV = stateDict["\(prefix).attn.proj_qkv.txt.weight"].to(torch.float).cpu()
        copyParameterFloat16(
          vidQProj.weight, from: try! Tensor<Float>(numpy: vidQKV[..<hiddenSize, ...].numpy()))
        copyParameterFloat16(
          vidKProj.weight,
          from: try! Tensor<Float>(numpy: vidQKV[hiddenSize..<(2 * hiddenSize), ...].numpy()))
        copyParameterFloat16(
          vidVProj.weight, from: try! Tensor<Float>(numpy: vidQKV[(2 * hiddenSize)..., ...].numpy())
        )
        if returnText {
          copyParameterFloat16(
            txtQProj.weight, from: try! Tensor<Float>(numpy: txtQKV[..<hiddenSize, ...].numpy()))
        }
        copyParameterFloat16(
          txtKProj.weight,
          from: try! Tensor<Float>(numpy: txtQKV[hiddenSize..<(2 * hiddenSize), ...].numpy()))
        copyParameterFloat16(
          txtVProj.weight, from: try! Tensor<Float>(numpy: txtQKV[(2 * hiddenSize)..., ...].numpy())
        )
        copyParameterFloat16(
          vidNormQ.weight, from: seedVR2StateTensor(stateDict, "\(prefix).attn.norm_q.vid.weight"))
        copyParameterFloat16(
          vidNormK.weight, from: seedVR2StateTensor(stateDict, "\(prefix).attn.norm_k.vid.weight"))
        if returnText {
          copyParameterFloat16(
            txtNormQ.weight, from: seedVR2StateTensor(stateDict, "\(prefix).attn.norm_q.txt.weight")
          )
        }
        copyParameterFloat16(
          txtNormK.weight, from: seedVR2StateTensor(stateDict, "\(prefix).attn.norm_k.txt.weight"))
      }

      let vidProjPrefix =
        sharedWeights ? "\(prefix).attn.proj_out.all" : "\(prefix).attn.proj_out.vid"
      let txtProjPrefix =
        sharedWeights ? "\(prefix).attn.proj_out.all" : "\(prefix).attn.proj_out.txt"
      copyParameterFloat16(
        vidAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidProjPrefix).weight"].to(torch.float).cpu().numpy()))
      copyParameterFloat16(
        vidAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidProjPrefix).bias"].to(torch.float).cpu().numpy()))
      if let txtAttnOut = txtAttnOut {
        copyParameterFloat16(
          txtAttnOut.weight,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtProjPrefix).weight"].to(torch.float).cpu().numpy()))
        copyParameterFloat16(
          txtAttnOut.bias,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtProjPrefix).bias"].to(torch.float).cpu().numpy()))
      }

      let vidMlpPrefix = sharedWeights ? "\(prefix).mlp.all" : "\(prefix).mlp.vid"
      copyParameterFloat16(
        vidMlpGate.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidMlpPrefix).proj_in_gate.weight"].to(torch.float).cpu().numpy()))
      copyParameterFloat16(
        vidMlpInProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidMlpPrefix).proj_in.weight"].to(torch.float).cpu().numpy()))
      copyParameterFloat16(
        vidMlpOutProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidMlpPrefix).proj_out.weight"].to(torch.float).cpu().numpy()))
      if let txtMlpGate = txtMlpGate, let txtMlpInProj = txtMlpInProj,
        let txtMlpOutProj = txtMlpOutProj
      {
        let txtMlpPrefix = sharedWeights ? "\(prefix).mlp.all" : "\(prefix).mlp.txt"
        copyParameterFloat16(
          txtMlpGate.weight,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtMlpPrefix).proj_in_gate.weight"].to(torch.float).cpu().numpy()))
        copyParameterFloat16(
          txtMlpInProj.weight,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtMlpPrefix).proj_in.weight"].to(torch.float).cpu().numpy()))
        copyParameterFloat16(
          txtMlpOutProj.weight,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtMlpPrefix).proj_out.weight"].to(torch.float).cpu().numpy()))
      }
    }
  case .gelu:
    let vidMlpInProj = Dense(count: config.mlpHiddenSize, name: "vid_mlp_in")
    let vidMlpOutProj = Dense(count: hiddenSize, name: "vid_mlp_out")
    let vidMlpOut = vidMlpOutProj(
      vidMlpInProj(vidMlpIn.to(.Float16).copied()).GELU(approximate: .tanh)
    ).to(.Float32).copied()
    vidFinal = (vidAfterAttn + vidMlpOut .* vidMlpMod[2]).to(of: vidAfterAttn)

    let txtMlpInProj: Model?
    let txtMlpOutProj: Model?
    if let txtMlpIn = txtMlpIn, let txtMlpMod = txtMlpMod {
      let inProj = Dense(count: config.mlpHiddenSize, name: "txt_mlp_in")
      let outProj = Dense(count: hiddenSize, name: "txt_mlp_out")
      let txtMlpOut = outProj(
        inProj(txtMlpIn.to(.Float16).copied()).GELU(approximate: .tanh)
      ).to(.Float32).copied()
      txtFinal = (txtAfterAttn! + txtMlpOut .* txtMlpMod[2]).to(of: txtAfterAttn!)
      txtMlpInProj = inProj
      txtMlpOutProj = outProj
    } else {
      txtFinal = nil
      txtMlpInProj = nil
      txtMlpOutProj = nil
    }

    readers.append { stateDict in
      let prefix = "blocks.\(layerIndex)"
      if sharedWeights {
        let allQKV = stateDict["\(prefix).attn.proj_qkv.all.weight"].to(torch.float).cpu()
        let qWeight = try! Tensor<Float>(numpy: allQKV[..<hiddenSize, ...].numpy())
        let kWeight = try! Tensor<Float>(numpy: allQKV[hiddenSize..<(2 * hiddenSize), ...].numpy())
        let vWeight = try! Tensor<Float>(numpy: allQKV[(2 * hiddenSize)..., ...].numpy())
        let allNormQ = seedVR2StateTensor(stateDict, "\(prefix).attn.norm_q.all.weight")
        let allNormK = seedVR2StateTensor(stateDict, "\(prefix).attn.norm_k.all.weight")
        copyParameterFloat16(vidQProj.weight, from: qWeight)
        copyParameterFloat16(vidKProj.weight, from: kWeight)
        copyParameterFloat16(vidVProj.weight, from: vWeight)
        if returnText {
          copyParameterFloat16(txtQProj.weight, from: qWeight)
        }
        copyParameterFloat16(txtKProj.weight, from: kWeight)
        copyParameterFloat16(txtVProj.weight, from: vWeight)
        copyParameterFloat16(vidNormQ.weight, from: allNormQ)
        copyParameterFloat16(vidNormK.weight, from: allNormK)
        if returnText {
          copyParameterFloat16(txtNormQ.weight, from: allNormQ)
        }
        copyParameterFloat16(txtNormK.weight, from: allNormK)
      } else {
        let vidQKV = stateDict["\(prefix).attn.proj_qkv.vid.weight"].to(torch.float).cpu()
        let txtQKV = stateDict["\(prefix).attn.proj_qkv.txt.weight"].to(torch.float).cpu()
        copyParameterFloat16(
          vidQProj.weight, from: try! Tensor<Float>(numpy: vidQKV[..<hiddenSize, ...].numpy()))
        copyParameterFloat16(
          vidKProj.weight,
          from: try! Tensor<Float>(numpy: vidQKV[hiddenSize..<(2 * hiddenSize), ...].numpy()))
        copyParameterFloat16(
          vidVProj.weight, from: try! Tensor<Float>(numpy: vidQKV[(2 * hiddenSize)..., ...].numpy())
        )
        if returnText {
          copyParameterFloat16(
            txtQProj.weight, from: try! Tensor<Float>(numpy: txtQKV[..<hiddenSize, ...].numpy()))
        }
        copyParameterFloat16(
          txtKProj.weight,
          from: try! Tensor<Float>(numpy: txtQKV[hiddenSize..<(2 * hiddenSize), ...].numpy()))
        copyParameterFloat16(
          txtVProj.weight, from: try! Tensor<Float>(numpy: txtQKV[(2 * hiddenSize)..., ...].numpy())
        )
        copyParameterFloat16(
          vidNormQ.weight, from: seedVR2StateTensor(stateDict, "\(prefix).attn.norm_q.vid.weight"))
        copyParameterFloat16(
          vidNormK.weight, from: seedVR2StateTensor(stateDict, "\(prefix).attn.norm_k.vid.weight"))
        if returnText {
          copyParameterFloat16(
            txtNormQ.weight, from: seedVR2StateTensor(stateDict, "\(prefix).attn.norm_q.txt.weight")
          )
        }
        copyParameterFloat16(
          txtNormK.weight, from: seedVR2StateTensor(stateDict, "\(prefix).attn.norm_k.txt.weight"))
      }

      let vidProjPrefix =
        sharedWeights ? "\(prefix).attn.proj_out.all" : "\(prefix).attn.proj_out.vid"
      let txtProjPrefix =
        sharedWeights ? "\(prefix).attn.proj_out.all" : "\(prefix).attn.proj_out.txt"
      copyParameterFloat16(
        vidAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidProjPrefix).weight"].to(torch.float).cpu().numpy()))
      copyParameterFloat16(
        vidAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidProjPrefix).bias"].to(torch.float).cpu().numpy()))
      if let txtAttnOut = txtAttnOut {
        copyParameterFloat16(
          txtAttnOut.weight,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtProjPrefix).weight"].to(torch.float).cpu().numpy()))
        copyParameterFloat16(
          txtAttnOut.bias,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtProjPrefix).bias"].to(torch.float).cpu().numpy()))
      }

      let vidMlpPrefix = sharedWeights ? "\(prefix).mlp.all" : "\(prefix).mlp.vid"
      copyParameterFloat16(
        vidMlpInProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidMlpPrefix).proj_in.weight"].to(torch.float).cpu().numpy()))
      copyParameterFloat16(
        vidMlpInProj.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidMlpPrefix).proj_in.bias"].to(torch.float).cpu().numpy()))
      copyParameterFloat16(
        vidMlpOutProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidMlpPrefix).proj_out.weight"].to(torch.float).cpu().numpy()))
      copyParameterFloat16(
        vidMlpOutProj.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(vidMlpPrefix).proj_out.bias"].to(torch.float).cpu().numpy()))
      if let txtMlpInProj = txtMlpInProj, let txtMlpOutProj = txtMlpOutProj {
        let txtMlpPrefix = sharedWeights ? "\(prefix).mlp.all" : "\(prefix).mlp.txt"
        copyParameterFloat16(
          txtMlpInProj.weight,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtMlpPrefix).proj_in.weight"].to(torch.float).cpu().numpy()))
        copyParameterFloat16(
          txtMlpInProj.bias,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtMlpPrefix).proj_in.bias"].to(torch.float).cpu().numpy()))
        copyParameterFloat16(
          txtMlpOutProj.weight,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtMlpPrefix).proj_out.weight"].to(torch.float).cpu().numpy()))
        copyParameterFloat16(
          txtMlpOutProj.bias,
          from: try! Tensor<Float>(
            numpy: stateDict["\(txtMlpPrefix).proj_out.bias"].to(torch.float).cpu().numpy()))
      }
    }
  }

  let reader: (PythonObject) -> Void = { stateDict in
    for reader in readers {
      reader(stateDict)
    }
  }
  let outputs = returnText ? [vidFinal, txtFinal!] : [vidFinal]
  return (reader, Model([vid, txt, emb] + windowVidFreqs + windowTxtFreqs, outputs))
}

func seedVR2DiTOutputIO(
  config: SeedVR2DiTConfig, frames: Int, latentHeight: Int, latentWidth: Int, x: Model.IO,
  emb: Model.IO
) -> ((PythonObject) -> Void, Model.IO) {
  let patchHeight = latentHeight / 2
  let patchWidth = latentWidth / 2
  if config.outputNormAda {
    let (modReader, mod) = seedVR2DiTOutputMod(emb: emb, hiddenSize: config.hiddenSize)
    let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm_out")
    let proj = Dense(count: 64, name: "linear")
    let afterAda = (norm(x) .* mod[1] + mod[0]).to(.Float16)
    var out = proj(afterAda).to(.Float32).reshaped(
      [frames, patchHeight, patchWidth, 2, 2, 16], format: .NHWC)
    out = out.permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
      frames * latentHeight * latentWidth, 16,
    ])
    let reader: (PythonObject) -> Void = { stateDict in
      modReader(stateDict)
      norm.weight.copy(
        from: try! Tensor<Float>(
          numpy: stateDict["vid_out_norm.weight"].to(torch.float).cpu().numpy()))
      proj.weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: stateDict["vid_out.proj.weight"].to(torch.float).cpu().numpy())))
      proj.bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: stateDict["vid_out.proj.bias"].to(torch.float).cpu().numpy())))
    }
    return (reader, out.copied())
  }

  let proj = Dense(count: 64, name: "linear")
  var out = proj(x.to(.Float16)).to(.Float32).reshaped(
    [frames, patchHeight, patchWidth, 2, 2, 16], format: .NHWC)
  out = out.permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    frames * latentHeight * latentWidth, 16,
  ])
  let reader: (PythonObject) -> Void = { stateDict in
    proj.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(
          numpy: stateDict["vid_out.proj.weight"].to(torch.float).cpu().numpy())))
    proj.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(
          numpy: stateDict["vid_out.proj.bias"].to(torch.float).cpu().numpy())))
  }
  return (reader, out.copied())
}

func SeedVR2DiT(
  config: SeedVR2DiTConfig, frames: Int, latentHeight: Int, latentWidth: Int, txtLen: Int
) -> ((PythonObject) -> Void, Model) {
  let vid = Input()
  let txt = Input()
  let timestep = Input()
  let patchHeight = latentHeight / 2
  let patchWidth = latentWidth / 2
  let regularWindows = seedVR2WindowSpec3D(
    frames: frames, height: patchHeight, width: patchWidth, shifted: false)
  let shiftedWindows = seedVR2WindowSpec3D(
    frames: frames, height: patchHeight, width: patchWidth, shifted: true)
  let rotateText = config.rotaryKind == .mmrope3d
  let regularVidFreqs = regularWindows.map { _ in Input() }
  let regularTxtFreqs = rotateText ? regularWindows.map { _ in Input() } : []
  let shiftedVidFreqs = shiftedWindows.map { _ in Input() }
  let shiftedTxtFreqs = rotateText ? shiftedWindows.map { _ in Input() } : []

  let (txtInReader, txtIn) = SeedVR2DiTTextIn(hiddenSize: config.hiddenSize)
  let (patchInReader, patchIn) = SeedVR2DiTPatchIn(
    hiddenSize: config.hiddenSize, frames: frames, latentHeight: latentHeight,
    latentWidth: latentWidth)
  let (embReader, embIn) = SeedVR2DiTTimeEmbedding(hiddenSize: config.hiddenSize)
  var txtOut = txtIn(txt).to(.Float32).copied()
  var vidOut = patchIn(vid).to(.Float32).copied()
  let emb = embIn(timestep).to(.Float32).copied()

  var readers: [(PythonObject) -> Void] = [txtInReader, patchInReader, embReader]
  for layerIndex in 0..<config.layers {
    let isFinalLayer = layerIndex == config.layers - 1
    let lastLayer = config.lastLayerVidOnly && isFinalLayer
    let returnText = !isFinalLayer
    let useShifted = layerIndex % 2 == 1 || lastLayer
    let windows = useShifted ? shiftedWindows : regularWindows
    let (blockReader, block) = SeedVR2DiTBlock(
      config: config, layerIndex: layerIndex, windows: windows, frames: frames,
      height: patchHeight, width: patchWidth, txtLen: txtLen, lastLayer: lastLayer,
      returnText: returnText)
    let blockFreqs =
      useShifted ? (shiftedVidFreqs + shiftedTxtFreqs) : (regularVidFreqs + regularTxtFreqs)
    let blockOut = block([vidOut, txtOut, emb] + blockFreqs)
    vidOut = blockOut[0].copied()
    if returnText {
      txtOut = blockOut[1].copied()
    }
    readers.append(blockReader)
  }

  let (outReader, out) = seedVR2DiTOutputIO(
    config: config, frames: frames, latentHeight: latentHeight, latentWidth: latentWidth,
    x: vidOut, emb: emb)
  readers.append(outReader)

  let reader: (PythonObject) -> Void = { stateDict in
    for reader in readers {
      reader(stateDict)
    }
  }
  return (
    reader,
    Model(
      [vid, txt, timestep] + regularVidFreqs + regularTxtFreqs + shiftedVidFreqs + shiftedTxtFreqs,
      [out.copied()])
  )
}

func SeedVR2DecoderConvIn(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_in")
  let out = convIn(
    seedVR2TemporalReplicatePad(
      x, batchSize: 1, channels: 16, depth: depth, height: height, width: width, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let convInWeight = stateDict["decoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let convInBias = stateDict["decoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: convInWeight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: convInBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2UpDecoderBlock3D(
  prefix: String, inChannels: Int, outChannels: Int, depth: Int, height: Int, width: Int,
  addUpsample: Bool, temporalUp: Bool, spatialUp: Bool
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (resnet0Reader, resnet0) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.0", inChannels: inChannels, outChannels: outChannels, depth: depth,
    height: height, width: width)
  var out = resnet0(x)
  let (resnet1Reader, resnet1) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.1", inChannels: outChannels, outChannels: outChannels,
    depth: depth, height: height, width: width)
  out = resnet1(out)
  let (resnet2Reader, resnet2) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.2", inChannels: outChannels, outChannels: outChannels,
    depth: depth, height: height, width: width)
  out = resnet2(out)
  let upsampleReader: ((PythonObject) -> Void)?
  if addUpsample {
    let (reader, upsample) = SeedVR2Upsample3D(
      prefix: "\(prefix).upsamplers.0", channels: outChannels, depth: depth, height: height,
      width: width, temporalUp: temporalUp, spatialUp: spatialUp)
    out = upsample(out)
    upsampleReader = reader
  } else {
    upsampleReader = nil
  }
  let reader: (PythonObject) -> Void = { stateDict in
    resnet0Reader(stateDict)
    resnet1Reader(stateDict)
    resnet2Reader(stateDict)
    upsampleReader?(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DecoderPostProcess(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let convNormOut = GroupNorm(
    axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: "conv_norm_out")
  var out = x.permuted(0, 2, 1, 3, 4).copied().reshaped([depth, 128, height, width])
  out = convNormOut(out).reshaped([1, depth, 128, height, width]).permuted(0, 2, 1, 3, 4)
    .copied()
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_out")
  out = convOut(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: 128, depth: depth, height: height, width: width, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let normWeight = stateDict["decoder.conv_norm_out.weight"].to(torch.float).cpu().numpy()
    let normBias = stateDict["decoder.conv_norm_out.bias"].to(torch.float).cpu().numpy()
    convNormOut.weight.copy(from: try! Tensor<Float>(numpy: normWeight))
    convNormOut.bias.copy(from: try! Tensor<Float>(numpy: normBias))
    let convWeight = stateDict["decoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let convBias = stateDict["decoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: convWeight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: convBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Decoder3D(startDepth: Int, startHeight: Int, startWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (convInReader, convIn) = SeedVR2DecoderConvIn(
    depth: startDepth, height: startHeight, width: startWidth)
  var out = convIn(x)

  let (midBlockReader, midBlock) = SeedVR2DecoderMidBlock3D(
    depth: startDepth, height: startHeight, width: startWidth)
  out = midBlock(out)

  let (upBlock0Reader, upBlock0) = SeedVR2UpDecoderBlock3D(
    prefix: "decoder.up_blocks.0", inChannels: 512, outChannels: 512, depth: startDepth,
    height: startHeight, width: startWidth, addUpsample: true, temporalUp: true, spatialUp: true)
  out = upBlock0(out)
  let depth1 = 1 + (startDepth - 1) * 2
  let height1 = startHeight * 2
  let width1 = startWidth * 2

  let (upBlock1Reader, upBlock1) = SeedVR2UpDecoderBlock3D(
    prefix: "decoder.up_blocks.1", inChannels: 512, outChannels: 512, depth: depth1,
    height: height1, width: width1, addUpsample: true, temporalUp: true, spatialUp: true)
  out = upBlock1(out)
  let depth2 = 1 + (depth1 - 1) * 2
  let height2 = height1 * 2
  let width2 = width1 * 2

  let (upBlock2Reader, upBlock2) = SeedVR2UpDecoderBlock3D(
    prefix: "decoder.up_blocks.2", inChannels: 512, outChannels: 256, depth: depth2,
    height: height2, width: width2, addUpsample: true, temporalUp: false, spatialUp: true)
  out = upBlock2(out)
  let depth3 = depth2
  let height3 = height2 * 2
  let width3 = width2 * 2

  let (upBlock3Reader, upBlock3) = SeedVR2UpDecoderBlock3D(
    prefix: "decoder.up_blocks.3", inChannels: 256, outChannels: 128, depth: depth3,
    height: height3, width: width3, addUpsample: false, temporalUp: false, spatialUp: false)
  out = upBlock3(out)

  let (postReader, postProcess) = SeedVR2DecoderPostProcess(
    depth: depth3, height: height3, width: width3)
  out = postProcess(out)

  let reader: (PythonObject) -> Void = { stateDict in
    convInReader(stateDict)
    midBlockReader(stateDict)
    upBlock0Reader(stateDict)
    upBlock1Reader(stateDict)
    upBlock2Reader(stateDict)
    upBlock3Reader(stateDict)
    postReader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Decoder3DNHWC(startDepth: Int, startHeight: Int, startWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (reader, decoder) = SeedVR2Decoder3D(
    startDepth: startDepth, startHeight: startHeight, startWidth: startWidth)
  let nchw = x.permuted(0, 4, 1, 2, 3).contiguous().copied().reshaped(
    [1, 16, startDepth, startHeight, startWidth], format: .NCHW)
  let out = decoder(nchw).permuted(0, 2, 3, 4, 1).contiguous().copied()
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2EncoderConvIn(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_in")
  let out = convIn(
    seedVR2TemporalReplicatePad(
      x, batchSize: 1, channels: 3, depth: depth, height: height, width: width, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["encoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["encoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Downsample3D(
  prefix: String, channels: Int, temporalDown: Bool, spatialDown: Bool, depth: Int, height: Int,
  width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let temporalRatio = temporalDown ? 2 : 1
  let spatialRatio = spatialDown ? 2 : 1
  let temporalKernel = temporalDown ? 3 : 1
  let spatialKernel = spatialDown ? 3 : 1
  var out: Model.IO = x
  if spatialDown {
    out = out.padded(.zero, begin: [0, 0, 0, 0, 0], end: [0, 0, 0, 1, 1])
  }
  let conv = Convolution(
    groups: 1, filters: channels, filterSize: [temporalKernel, spatialKernel, spatialKernel],
    hint: Hint(stride: [temporalRatio, spatialRatio, spatialRatio]),
    name: "conv_down")
  if temporalDown {
    out = conv(
      seedVR2TemporalReplicatePad(
        out, batchSize: 1, channels: channels, depth: depth, height: height + (spatialDown ? 1 : 0),
        width: width + (spatialDown ? 1 : 0), left: 2))
  } else {
    out = conv(out)
  }
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["\(prefix).conv.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["\(prefix).conv.bias"].to(torch.float).cpu().numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: weight))
    conv.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DownEncoderBlock3D(
  prefix: String, inChannels: Int, outChannels: Int, addDownsample: Bool, temporalDown: Bool,
  spatialDown: Bool, depth: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (resnet0Reader, resnet0) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.0", inChannels: inChannels, outChannels: outChannels, depth: depth,
    height: height, width: width)
  var out = resnet0(x)
  let (resnet1Reader, resnet1) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.1", inChannels: outChannels, outChannels: outChannels,
    depth: depth, height: height, width: width)
  out = resnet1(out)
  let downsampleReader: ((PythonObject) -> Void)?
  if addDownsample {
    let (reader, downsample) = SeedVR2Downsample3D(
      prefix: "\(prefix).downsamplers.0", channels: outChannels, temporalDown: temporalDown,
      spatialDown: spatialDown, depth: depth, height: height, width: width)
    out = downsample(out)
    downsampleReader = reader
  } else {
    downsampleReader = nil
  }
  let reader: (PythonObject) -> Void = { stateDict in
    resnet0Reader(stateDict)
    resnet1Reader(stateDict)
    downsampleReader?(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2EncoderMidBlock3D(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (resnet0Reader, resnet0) = SeedVR2ResnetBlock3D(
    prefix: "encoder.mid_block.resnets.0", inChannels: 512, outChannels: 512, depth: depth,
    height: height, width: width)
  var out = resnet0(x)
  let (attnReader, attn) = SeedVR2AttentionBlock2D(
    prefix: "encoder.mid_block.attentions.0", channels: 512, batchSize: depth, height: height,
    width: width)
  let frameWise = out.permuted(0, 2, 1, 3, 4).contiguous().reshaped([depth, 512, height, width])
  out = attn(frameWise).reshaped([1, depth, 512, height, width]).permuted(0, 2, 1, 3, 4)
    .contiguous()
  let (resnet1Reader, resnet1) = SeedVR2ResnetBlock3D(
    prefix: "encoder.mid_block.resnets.1", inChannels: 512, outChannels: 512, depth: depth,
    height: height, width: width)
  out = resnet1(out)
  let reader: (PythonObject) -> Void = { stateDict in
    resnet0Reader(stateDict)
    attnReader(stateDict)
    resnet1Reader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2EncoderPostProcess(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let convNormOut = GroupNorm(
    axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: "conv_norm_out")
  var out = x.permuted(0, 2, 1, 3, 4).copied().reshaped([depth, 512, height, width])
  out = convNormOut(out).reshaped([1, depth, 512, height, width]).permuted(0, 2, 1, 3, 4)
    .copied()
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_out")
  out = convOut(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: 512, depth: depth, height: height, width: width, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let normWeight = stateDict["encoder.conv_norm_out.weight"].to(torch.float).cpu().numpy()
    let normBias = stateDict["encoder.conv_norm_out.bias"].to(torch.float).cpu().numpy()
    convNormOut.weight.copy(from: try! Tensor<Float>(numpy: normWeight))
    convNormOut.bias.copy(from: try! Tensor<Float>(numpy: normBias))
    let convWeight = stateDict["encoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let convBias = stateDict["encoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: convWeight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: convBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Encoder3D(startDepth: Int, startHeight: Int, startWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (convInReader, convIn) = SeedVR2EncoderConvIn(
    depth: startDepth, height: startHeight, width: startWidth)
  var out = convIn(x)

  let (downBlock0Reader, downBlock0) = SeedVR2DownEncoderBlock3D(
    prefix: "encoder.down_blocks.0", inChannels: 128, outChannels: 128, addDownsample: true,
    temporalDown: false, spatialDown: true, depth: startDepth, height: startHeight,
    width: startWidth)
  out = downBlock0(out)
  let depth1 = startDepth
  let height1 = startHeight / 2
  let width1 = startWidth / 2

  let (downBlock1Reader, downBlock1) = SeedVR2DownEncoderBlock3D(
    prefix: "encoder.down_blocks.1", inChannels: 128, outChannels: 256, addDownsample: true,
    temporalDown: true, spatialDown: true, depth: depth1, height: height1, width: width1)
  out = downBlock1(out)
  let depth2 = (depth1 + 1) / 2
  let height2 = height1 / 2
  let width2 = width1 / 2

  let (downBlock2Reader, downBlock2) = SeedVR2DownEncoderBlock3D(
    prefix: "encoder.down_blocks.2", inChannels: 256, outChannels: 512, addDownsample: true,
    temporalDown: true, spatialDown: true, depth: depth2, height: height2, width: width2)
  out = downBlock2(out)
  let depth3 = (depth2 + 1) / 2
  let height3 = height2 / 2
  let width3 = width2 / 2

  let (downBlock3Reader, downBlock3) = SeedVR2DownEncoderBlock3D(
    prefix: "encoder.down_blocks.3", inChannels: 512, outChannels: 512, addDownsample: false,
    temporalDown: false, spatialDown: false, depth: depth3, height: height3, width: width3)
  out = downBlock3(out)

  let (midBlockReader, midBlock) = SeedVR2EncoderMidBlock3D(
    depth: depth3, height: height3, width: width3)
  out = midBlock(out)

  let (postReader, postProcess) = SeedVR2EncoderPostProcess(
    depth: depth3, height: height3, width: width3)
  out = postProcess(out)

  let reader: (PythonObject) -> Void = { stateDict in
    convInReader(stateDict)
    downBlock0Reader(stateDict)
    downBlock1Reader(stateDict)
    downBlock2Reader(stateDict)
    downBlock3Reader(stateDict)
    midBlockReader(stateDict)
    postReader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Encoder3DNHWC(startDepth: Int, startHeight: Int, startWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (reader, encoder) = SeedVR2Encoder3D(
    startDepth: startDepth, startHeight: startHeight, startWidth: startWidth)
  let nchw = x.permuted(0, 4, 1, 2, 3).contiguous().copied().reshaped(
    [1, 3, startDepth, startHeight, startWidth], format: .NCHW)
  let out = encoder(nchw).permuted(0, 2, 3, 4, 1).contiguous().copied()
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2MomentsToLatent(depth: Int, height: Int, width: Int) -> Model {
  let x = Input()
  let latent = x.reshaped(
    [1, 16, depth, height, width],
    strides: [32 * depth * height * width, depth * height * width, height * width, width, 1]
  ).contiguous()
  return Model([x], [latent])
}

if envFlag("SEEDVR2_RUN_VAE") || envFlag("SEEDVR2_EXPORT_VAE") {
  let vaeReference = seedvr2Reference.SeedVR2Reference(
    repo_root: seedVRRoot,
    checkpoint_root: seedVR2Config.checkpointRoot,
    config_dir: seedVR2Config.configDir,
    dit_checkpoint: seedVR2Config.ditCheckpoint,
    device: hasCUDA ? "cuda" : "cpu",
    load_dit: false,
    load_vae: true)
  logStep("SeedVR2 vae state_dict start")
  let vaeStateDict = vaeReference.runner.vae.state_dict()
  logStep("SeedVR2 vae encoder reference start")
  let vaeEncoderReference = vaeReference.make_encoder_probe(
    depth: 5, height: 96, width: 160, seed: 42)
  logStep("SeedVR2 vae decoder reference start")
  let vaeDecoderReference = vaeReference.make_decoder_probe(
    depth: 2, height: 12, width: 20, seed: 42)
  logStep("SeedVR2 vae mode reference start")
  let vaeModeReference = vaeReference.make_vae_mode_probe(
    frames: 5, height: 96, width: 160, seed: 42)

  let graph = DynamicGraph()
  graph.maxConcurrency = .limit(1)
  graph.withNoGrad {
    func materialize(_ value: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
      copiedToCPU(value)
    }

    func loadInput(_ probe: PythonObject, _ key: String) -> DynamicGraph.Tensor<Float> {
      placeOnDevice(
        graph.variable(try! Tensor<Float>(numpy: probe[key].to(torch.float).cpu().numpy())))
    }

    func loadTensor(_ probe: PythonObject, _ key: String) -> Tensor<Float> {
      try! Tensor<Float>(numpy: probe[key].to(torch.float).cpu().numpy())
    }

    func printParity(_ name: String, _ swift: Tensor<Float>, _ reference: Tensor<Float>) {
      print("\(name) max abs diff:", maxAbsDiff5D(swift, reference))
      print("\(name) global max rel diff:", maxGlobalRelativeDiff5D(swift, reference))
    }

    logStep("SeedVR2 vae validation start")
    let encoderInput = loadInput(vaeEncoderReference, "input")
    let (encoderReader, encoder) = SeedVR2Encoder3D(
      startDepth: 5, startHeight: 96, startWidth: 160)
    encoder.compile(inputs: encoderInput)
    encoderReader(vaeStateDict)
    let swiftEncoderOutput = materialize(encoder(inputs: encoderInput)[0].as(of: Float.self))
    let torchEncoderOutput = loadTensor(vaeEncoderReference, "output")
    printParity("SeedVR2 vae.encoder", swiftEncoderOutput, torchEncoderOutput)

    let encoderNHWCInput = encoderInput.permuted(0, 2, 3, 4, 1).contiguous()
    let (encoderNHWCReader, encoderNHWC) = SeedVR2Encoder3DNHWC(
      startDepth: 5, startHeight: 96, startWidth: 160)
    encoderNHWC.compile(inputs: encoderNHWCInput)
    encoderNHWCReader(vaeStateDict)
    let swiftEncoderNHWCOutput = materialize(
      encoderNHWC(inputs: encoderNHWCInput)[0].as(of: Float.self).permuted(0, 4, 1, 2, 3)
        .contiguous())
    printParity("SeedVR2 vae.encoder nhwc", swiftEncoderNHWCOutput, torchEncoderOutput)

    let decoderInput = loadInput(vaeDecoderReference, "input")
    let (decoderReader, decoder) = SeedVR2Decoder3D(
      startDepth: 2, startHeight: 12, startWidth: 20)
    decoder.compile(inputs: decoderInput)
    decoderReader(vaeStateDict)
    let swiftDecoderOutput = materialize(decoder(inputs: decoderInput)[0].as(of: Float.self))
    let torchDecoderOutput = loadTensor(vaeDecoderReference, "output")
    printParity("SeedVR2 vae.decoder", swiftDecoderOutput, torchDecoderOutput)

    let decoderNHWCInput = decoderInput.permuted(0, 2, 3, 4, 1).contiguous()
    let (decoderNHWCReader, decoderNHWC) = SeedVR2Decoder3DNHWC(
      startDepth: 2, startHeight: 12, startWidth: 20)
    decoderNHWC.compile(inputs: decoderNHWCInput)
    decoderNHWCReader(vaeStateDict)
    let swiftDecoderNHWCOutput = materialize(
      decoderNHWC(inputs: decoderNHWCInput)[0].as(of: Float.self).permuted(0, 4, 1, 2, 3)
        .contiguous())
    printParity("SeedVR2 vae.decoder nhwc", swiftDecoderNHWCOutput, torchDecoderOutput)

    let vaeModeInput = loadInput(vaeModeReference, "input")
    let swiftModeMoments = materialize(encoder(inputs: vaeModeInput)[0].as(of: Float.self))
    let torchModeMoments = loadTensor(vaeModeReference, "moments")
    printParity("SeedVR2 vae.mode moments", swiftModeMoments, torchModeMoments)

    let torchModeLatent = loadTensor(vaeModeReference, "latent")
    let momentsToLatentInput = rematerializeOnDevice(graph, swiftModeMoments)
    let momentsToLatent = SeedVR2MomentsToLatent(depth: 2, height: 12, width: 20)
    momentsToLatent.compile(inputs: momentsToLatentInput)
    let swiftModeLatent = materialize(
      momentsToLatent(inputs: momentsToLatentInput)[0].as(of: Float.self))
    printParity("SeedVR2 vae.mode latent", swiftModeLatent, torchModeLatent)

    let torchModeDecoded = loadTensor(vaeModeReference, "output")
    let swiftModeDecoded = materialize(
      decoder(inputs: rematerializeOnDevice(graph, swiftModeLatent))[0].as(of: Float.self))
    printParity("SeedVR2 vae.mode decoded", swiftModeDecoded, torchModeDecoded)

    printParity("SeedVR2 vae full", swiftModeDecoded, torchModeDecoded)

    if envFlag("SEEDVR2_EXPORT_VAE") {
      logStep("SeedVR2 vae export compile")
      let decoderInput = placeOnDevice(
        graph.variable(.CPU, format: .NHWC, shape: [1, 2, 12, 20, 16], of: Float.self))
      let (decoderReader, decoder) = SeedVR2Decoder3DNHWC(
        startDepth: 2, startHeight: 12, startWidth: 20)
      decoder.compile(inputs: decoderInput)
      decoderReader(vaeStateDict)

      let encoderInput = placeOnDevice(
        graph.variable(.CPU, format: .NHWC, shape: [1, 5, 96, 160, 3], of: Float.self))
      let (encoderReader, encoder) = SeedVR2Encoder3DNHWC(
        startDepth: 5, startHeight: 96, startWidth: 160)
      encoder.compile(inputs: encoderInput)
      encoderReader(vaeStateDict)

      graph.openStore(seedVR2VAEExportPath) {
        $0.write("decoder", model: decoder)
        $0.write("encoder", model: encoder)
      }
      print("Wrote \(seedVR2VAEExportPath)")
    }
  }
  exit(0)
}

func seedVR2PositiveEmbeddingTensor(_ reference: PythonObject) -> Tensor<Float> {
  let positive = reference.positive_text_embeddings().to(torch.float)
  return try! Tensor<Float>(numpy: positive.cpu().numpy())
}

func seedVR2NegativeEmbeddingTensor(_ reference: PythonObject) -> Tensor<Float> {
  let negative = reference.negative_text_embeddings().to(torch.float)
  return try! Tensor<Float>(numpy: negative.cpu().numpy())
}

func seedVR2WriteDiTExport(reference: PythonObject, dit: Model) {
  graph.openStore(seedVR2DiTExportPath) {
    $0.write("positive_embedding", tensor: seedVR2PositiveEmbeddingTensor(reference))
    $0.write("negative_embedding", tensor: seedVR2NegativeEmbeddingTensor(reference))
    $0.write("dit", model: dit)
  }
  print("Wrote \(seedVR2DiTExportPath)")
}

func seedVR2LoadImageTensor(path: String, height: Int, width: Int) -> Tensor<Float> {
  let imageModule = Python.import("PIL.Image")
  let image = imageModule.open(path).convert("RGB").resize(PythonObject(tupleOf: width, height))
  let array = ((numpy.array(image).astype(numpy.float32) / 127.5) - 1.0).transpose([2, 0, 1])
    .reshape([1, 3, 1, height, width])
  return try! Tensor<Float>(numpy: array)
}

func seedVR2Byte(_ value: Float) -> UInt8 {
  if !value.isFinite {
    return 0
  }
  return UInt8(min(max(Int((value + 1) * 127.5 + 0.5), 0), 255))
}

func seedVR2SaveImageTensor(_ tensor: Tensor<Float>, path: String) {
  precondition(tensor.shape.count == 5)
  precondition(tensor.shape[0] == 1)
  precondition(tensor.shape[4] == 3)
  let height = tensor.shape[2]
  let width = tensor.shape[3]
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: width * height)
  for y in 0..<height {
    for x in 0..<width {
      let offset = y * width + x
      rgba[offset].r = seedVR2Byte(Float(tensor[0, 0, y, x, 0]))
      rgba[offset].g = seedVR2Byte(Float(tensor[0, 0, y, x, 1]))
      rgba[offset].b = seedVR2Byte(Float(tensor[0, 0, y, x, 2]))
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (width, height),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! image.compress(path: path, level: 4)
}

func seedVR2ScaleTensor5D(_ tensor: Tensor<Float>, by scale: Float) -> Tensor<Float> {
  precondition(tensor.shape.count == 5)
  var out = tensor
  for n in 0..<tensor.shape[0] {
    for c in 0..<tensor.shape[1] {
      for d in 0..<tensor.shape[2] {
        for h in 0..<tensor.shape[3] {
          for w in 0..<tensor.shape[4] {
            out[n, c, d, h, w] = tensor[n, c, d, h, w] * scale
          }
        }
      }
    }
  }
  return out
}

func seedVR2BuildRawDiTInput(
  noise: Tensor<Float>, conditionLatent: Tensor<Float>, depth: Int, height: Int, width: Int
) -> Tensor<Float> {
  precondition(conditionLatent.shape.count == 5)
  precondition(conditionLatent.shape[0] == 1)
  precondition(conditionLatent.shape[1] == depth)
  precondition(conditionLatent.shape[2] == height)
  precondition(conditionLatent.shape[3] == width)
  precondition(conditionLatent.shape[4] >= 16)
  let tokenCount = depth * height * width
  precondition(noise.shape == [tokenCount, 16])
  var input = Tensor<Float>(.CPU, .NC(tokenCount, 33))
  var token = 0
  for t in 0..<depth {
    for h in 0..<height {
      for w in 0..<width {
        for c in 0..<16 {
          input[token, c] = noise[token, c]
          input[token, 16 + c] = conditionLatent[0, t, h, w, c]
        }
        input[token, 32] = 1
        token += 1
      }
    }
  }
  return input
}

func seedVR2BuildDecoderLatent(
  noise: Tensor<Float>, prediction: Tensor<Float>, depth: Int, height: Int, width: Int,
  scalingFactor: Float
) -> Tensor<Float> {
  let tokenCount = depth * height * width
  precondition(noise.shape == [tokenCount, 16])
  precondition(prediction.shape == [tokenCount, 16])
  var latent = Tensor<Float>(.CPU, format: .NHWC, shape: [1, depth, height, width, 16])
  var token = 0
  for t in 0..<depth {
    for h in 0..<height {
      for w in 0..<width {
        for c in 0..<16 {
          latent[0, t, h, w, c] = (noise[token, c] - prediction[token, c]) / scalingFactor
        }
        token += 1
      }
    }
  }
  return latent
}

if envFlag("SEEDVR2_RUN_UPSCALE") {
  precondition(seedVR2Config.name == "3B", "SeedVR2 upscale smoke path currently targets 3B.")
  let inputPath = envString("SEEDVR2_UPSCALE_INPUT", "\(repoRoot)/generated.png")
  let outputPath = envString(
    "SEEDVR2_UPSCALE_OUTPUT", "\(repoRoot)/seedvr2_\(seedVR2Config.name.lowercased())_upscale.png")
  let imageHeight = envInt("SEEDVR2_UPSCALE_HEIGHT", 256)
  let imageWidth = envInt("SEEDVR2_UPSCALE_WIDTH", 256)
  precondition(imageHeight % 16 == 0 && imageWidth % 16 == 0)
  let latentDepth = 1
  let latentHeight = imageHeight / 8
  let latentWidth = imageWidth / 8
  let patchHeight = latentHeight / 2
  let patchWidth = latentWidth / 2
  let scalingFactor: Float = 0.9152
  let seed = envInt("SEEDVR2_UPSCALE_SEED", 666)

  print("SeedVR2 upscale input:", inputPath)
  print("SeedVR2 upscale output:", outputPath)
  print("SeedVR2 upscale size:", imageHeight, imageWidth)
  print("SeedVR2 upscale precision: VAE Float16, DiT mixed Float16/Float32")

  let positiveEmbedding = try! graph.openStore(seedVR2DiTExportPath, flags: [.readOnly]) {
    Tensor<Float>(from: $0.read("positive_embedding")!)
  }.get()
  let txtLen = positiveEmbedding.shape[0]
  let regularWindows = seedVR2WindowSpec3D(
    frames: latentDepth, height: patchHeight, width: patchWidth, shifted: false)
  let shiftedWindows = seedVR2WindowSpec3D(
    frames: latentDepth, height: patchHeight, width: patchWidth, shifted: true)

  let regularRotary = seedVR2RotaryWindowTensors(
    config: seedVR2Config, windows: regularWindows, txtLen: txtLen)
  let shiftedRotary = seedVR2RotaryWindowTensors(
    config: seedVR2Config, windows: shiftedWindows, txtLen: txtLen)

  torch.manual_seed(seed)
  let noise = try! Tensor<Float>(
    numpy: torch.randn([latentDepth * latentHeight * latentWidth, 16], dtype: torch.float32).cpu()
      .numpy())
  let imageTensor = seedVR2LoadImageTensor(path: inputPath, height: imageHeight, width: imageWidth)

  graph.withNoGrad {
    logStep("SeedVR2 upscale encoder compile/load Float16")
    let encoderInput = placeOnDevice(graph.variable(Tensor<Float16>(from: imageTensor)))
      .permuted(0, 2, 3, 4, 1).contiguous()
    let (_, encoder) = SeedVR2Encoder3DNHWC(
      startDepth: 1, startHeight: imageHeight, startWidth: imageWidth)
    encoder.compile(inputs: encoderInput)
    graph.openStore(seedVR2VAEExportPath, flags: [.readOnly]) {
      $0.read("encoder", model: encoder)
    }
    logStep("SeedVR2 upscale encode Float16")
    let conditionLatentCPU = seedVR2ScaleTensor5D(
      copiedToCPU(encoder(inputs: encoderInput)[0].as(of: Float16.self)), by: scalingFactor)

    let rawVidInputCPU = seedVR2BuildRawDiTInput(
      noise: noise, conditionLatent: conditionLatentCPU, depth: latentDepth, height: latentHeight,
      width: latentWidth)
    let rawVidInput = placeOnDevice(graph.variable(Tensor<Float16>(from: rawVidInputCPU)))
    let rawTxtInput = placeOnDevice(graph.variable(Tensor<Float16>(from: positiveEmbedding)))
    let timeInput = placeOnDevice(
      graph.variable(
        Tensor<Float16>(
          from: seedVR2TimeEmbedding(
            timestep: 1000.0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)))
    )
    let regularVidFreqInputs = regularRotary.vid.map { rematerializeOnDeviceFloat16(graph, $0) }
    let regularTxtFreqInputs = regularRotary.txt.map { rematerializeOnDeviceFloat16(graph, $0) }
    let shiftedVidFreqInputs = shiftedRotary.vid.map { rematerializeOnDeviceFloat16(graph, $0) }
    let shiftedTxtFreqInputs = shiftedRotary.txt.map { rematerializeOnDeviceFloat16(graph, $0) }
    let rotaryInputs =
      seedVR2Config.rotaryKind == .mmrope3d
      ? regularVidFreqInputs + regularTxtFreqInputs + shiftedVidFreqInputs + shiftedTxtFreqInputs
      : regularVidFreqInputs + shiftedVidFreqInputs
    let ditInputs: [DynamicGraph_Any] =
      [
        rawVidInput as DynamicGraph_Any, rawTxtInput as DynamicGraph_Any,
        timeInput as DynamicGraph_Any,
      ]
      + rotaryInputs.map { $0 as DynamicGraph_Any }

    logStep("SeedVR2 upscale dit compile/load mixed")
    let (_, dit) = SeedVR2DiT(
      config: seedVR2Config, frames: latentDepth, latentHeight: latentHeight,
      latentWidth: latentWidth, txtLen: txtLen)
    dit.compile(inputs: ditInputs)
    graph.openStore(seedVR2DiTExportPath, flags: [.readOnly]) {
      $0.read("dit", model: dit)
    }

    logStep("SeedVR2 upscale dit forward mixed")
    let prediction = dit(inputs: rawVidInput, Array(ditInputs.dropFirst()))[0].as(of: Float.self)
    let predictionCPU = copiedToCPU(prediction)
    seedVR2FiniteSummary("SeedVR2 upscale prediction", predictionCPU)
    let decoderLatent = seedVR2BuildDecoderLatent(
      noise: noise, prediction: predictionCPU, depth: latentDepth, height: latentHeight,
      width: latentWidth, scalingFactor: scalingFactor)

    logStep("SeedVR2 upscale decoder compile/load Float16")
    let decoderInput = placeOnDevice(graph.variable(Tensor<Float16>(from: decoderLatent)))
    let (_, decoder) = SeedVR2Decoder3DNHWC(
      startDepth: latentDepth, startHeight: latentHeight, startWidth: latentWidth)
    decoder.compile(inputs: decoderInput)
    graph.openStore(seedVR2VAEExportPath, flags: [.readOnly]) {
      $0.read("decoder", model: decoder)
    }

    logStep("SeedVR2 upscale decode Float16")
    let decoded = copiedToCPU(decoder(inputs: decoderInput)[0].as(of: Float16.self))
    seedVR2FiniteSummary("SeedVR2 upscale decoded", decoded)
    seedVR2SaveImageTensor(decoded, path: outputPath)
    print("Wrote \(outputPath)")
  }
  exit(0)
}

if envFlag("SEEDVR2_RUN_DIT") || envFlag("SEEDVR2_RUN_DIT_WINDOW_FULL")
  || envFlag("SEEDVR2_EXPORT_DIT")
{
  let ditReference = seedvr2Reference.SeedVR2Reference(
    repo_root: seedVRRoot,
    checkpoint_root: seedVR2Config.checkpointRoot,
    config_dir: seedVR2Config.configDir,
    dit_checkpoint: seedVR2Config.ditCheckpoint,
    device: hasCUDA ? "cuda" : "cpu",
    load_dit: true,
    load_vae: false)
  let ditFrames = envInt("SEEDVR2_DIT_FRAMES", 5)
  let ditLatentHeight = envInt("SEEDVR2_DIT_LATENT_HEIGHT", 6)
  let ditLatentWidth = envInt("SEEDVR2_DIT_LATENT_WIDTH", 6)
  let patchHeight = ditLatentHeight / 2
  let patchWidth = ditLatentWidth / 2
  let ditTxtLen = 58
  let regularWindows = seedVR2WindowSpec3D(
    frames: ditFrames, height: patchHeight, width: patchWidth, shifted: false)
  let shiftedWindows = seedVR2WindowSpec3D(
    frames: ditFrames, height: patchHeight, width: patchWidth, shifted: true)

  print(
    "SeedVR2 dit shape frames/latent:",
    ditFrames, ditLatentHeight, ditLatentWidth)
  print("SeedVR2 dit regular window count:", regularWindows.count)
  print("SeedVR2 dit shifted window count:", shiftedWindows.count)

  logStep("SeedVR2 dit regular rotary start")
  let regularProbe = ditReference.make_dit_block_probe(
    layer_idx: 0, frames: ditFrames, latent_height: ditLatentHeight, latent_width: ditLatentWidth,
    timestep: 500.0, disable_rope: false)
  logStep("SeedVR2 dit shifted rotary start")
  let shiftedProbe = ditReference.make_dit_block_probe(
    layer_idx: 1, frames: ditFrames, latent_height: ditLatentHeight, latent_width: ditLatentWidth,
    timestep: 500.0, disable_rope: false)
  logStep("SeedVR2 dit reference start")
  let officialProbe = ditReference.make_dit_body_probe_official(
    frames: ditFrames, latent_height: ditLatentHeight, latent_width: ditLatentWidth,
    timestep: 500.0)
  if hasCUDA {
    _ = ditReference.runner.dit.to(torch.device("cpu"))
    torch.cuda.empty_cache()
  }
  logStep("SeedVR2 dit state_dict start")
  let ditStateDict = ditReference.runner.dit.state_dict()

  graph.withNoGrad {
    func loadTensor(_ probe: PythonObject, _ key: String) -> Tensor<Float> {
      try! Tensor<Float>(numpy: probe[key].to(torch.float).cpu().numpy())
    }

    func loadDeviceTensor(_ probe: PythonObject, _ key: String) -> DynamicGraph.Tensor<Float> {
      rematerializeOnDevice(graph, loadTensor(probe, key))
    }

    func loadWindowFreqs(
      _ probe: PythonObject, kind: String, count: Int
    ) -> [Tensor<Float>] {
      var freqs = [Tensor<Float>]()
      for index in 0..<count {
        let tensor = seedVR2RotaryTensor(from: loadTensor(probe, "window\(index)_\(kind)_freqs"))
        freqs.append(tensor)
      }
      return freqs
    }

    let regularReferenceVidFreqs = loadWindowFreqs(
      regularProbe, kind: "vid", count: regularWindows.count)
    let regularReferenceTxtFreqs = loadWindowFreqs(
      regularProbe, kind: "txt", count: regularWindows.count)
    let shiftedReferenceVidFreqs = loadWindowFreqs(
      shiftedProbe, kind: "vid", count: shiftedWindows.count)
    let shiftedReferenceTxtFreqs = loadWindowFreqs(
      shiftedProbe, kind: "txt", count: shiftedWindows.count)
    let regularRotary = seedVR2RotaryWindowTensors(
      config: seedVR2Config, windows: regularWindows, txtLen: ditTxtLen)
    let shiftedRotary = seedVR2RotaryWindowTensors(
      config: seedVR2Config, windows: shiftedWindows, txtLen: ditTxtLen)
    seedVR2PrintRotaryParity(
      "SeedVR2 dit regular vid", swift: regularRotary.vid, reference: regularReferenceVidFreqs)
    seedVR2PrintRotaryParity(
      "SeedVR2 dit regular txt", swift: regularRotary.txt, reference: regularReferenceTxtFreqs)
    seedVR2PrintRotaryParity(
      "SeedVR2 dit shifted vid", swift: shiftedRotary.vid, reference: shiftedReferenceVidFreqs)
    seedVR2PrintRotaryParity(
      "SeedVR2 dit shifted txt", swift: shiftedRotary.txt, reference: shiftedReferenceTxtFreqs)

    let rawVidInput = loadDeviceTensor(officialProbe, "vid_input")
    let rawTxtInput = loadDeviceTensor(officialProbe, "txt_input")
    let timeInput = placeOnDevice(
      graph.variable(
        seedVR2TimeEmbedding(timestep: 500.0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)))
    let regularVidFreqInputs = regularRotary.vid.map { rematerializeOnDeviceFloat16(graph, $0) }
    let regularTxtFreqInputs = regularRotary.txt.map { rematerializeOnDeviceFloat16(graph, $0) }
    let shiftedVidFreqInputs = shiftedRotary.vid.map { rematerializeOnDeviceFloat16(graph, $0) }
    let shiftedTxtFreqInputs = shiftedRotary.txt.map { rematerializeOnDeviceFloat16(graph, $0) }
    let rotaryInputs =
      seedVR2Config.rotaryKind == .mmrope3d
      ? regularVidFreqInputs + regularTxtFreqInputs + shiftedVidFreqInputs + shiftedTxtFreqInputs
      : regularVidFreqInputs + shiftedVidFreqInputs
    let ditInputs: [DynamicGraph_Any] =
      [
        rawVidInput as DynamicGraph_Any, rawTxtInput as DynamicGraph_Any,
        timeInput as DynamicGraph_Any,
      ]
      + rotaryInputs.map { $0 as DynamicGraph_Any }
    let ditRestInputs = Array(ditInputs.dropFirst())

    logStep("SeedVR2 dit compile")
    let (ditReader, dit) = SeedVR2DiT(
      config: seedVR2Config, frames: ditFrames, latentHeight: ditLatentHeight,
      latentWidth: ditLatentWidth, txtLen: ditTxtLen)
    dit.compile(inputs: ditInputs)
    logStep("SeedVR2 dit load")
    ditReader(ditStateDict)
    logStep("SeedVR2 dit forward")
    let ditOutputs = dit(inputs: rawVidInput, ditRestInputs)
    let swiftFullOutput = copiedToCPU(ditOutputs[0].as(of: Float.self))
    let torchFullOutput = loadTensor(officialProbe, "output")
    print(
      "SeedVR2 dit output max abs diff:",
      maxAbsDiff2DTensor(swiftFullOutput, torchFullOutput))
    print(
      "SeedVR2 dit output global max rel diff:",
      maxGlobalRelativeDiff2DTensor(swiftFullOutput, torchFullOutput))

    if envFlag("SEEDVR2_EXPORT_DIT") {
      seedVR2WriteDiTExport(reference: ditReference, dit: dit)
    }
  }
  exit(0)
}

exit(0)
