import Diffusion
import Foundation
import NNC
import PNG

public typealias FloatType = Float16

public struct DiffusionModel {
  public var linearStart: Float
  public var linearEnd: Float
  public var timesteps: Int
  public var steps: Int
}

extension DiffusionModel {
  public var betas: [Float] {  // Linear for now.
    var betas = [Float]()
    let start = linearStart.squareRoot()
    let length = linearEnd.squareRoot() - start
    for i in 0..<timesteps {
      let beta = start + Float(i) * length / Float(timesteps - 1)
      betas.append(beta * beta)
    }
    return betas
  }
  public var alphasCumprod: [Float] {
    var cumprod: Float = 1
    return betas.map {
      cumprod *= 1 - $0
      return cumprod
    }
  }
  // This is Karras scheduler sigmas.
  public func karrasSigmas(_ range: ClosedRange<Float>, rho: Float = 7.0) -> [Float] {
    let minInvRho = pow(range.lowerBound, 1.0 / rho)
    let maxInvRho = pow(range.upperBound, 1.0 / rho)
    var sigmas = [Float]()
    for i in 0..<steps {
      sigmas.append(pow(maxInvRho + Float(i) * (minInvRho - maxInvRho) / Float(steps - 1), rho))
    }
    sigmas.append(0)
    return sigmas
  }

  public func fixedStepSigmas(_ range: ClosedRange<Float>, sigmas sigmasForTimesteps: [Float])
    -> [Float]
  {
    var sigmas = [Float]()
    for i in 0..<steps {
      let timestep = Float(steps - 1 - i) / Float(steps - 1) * Float(timesteps - 1)
      let lowIdx = Int(floor(timestep))
      let highIdx = min(lowIdx + 1, timesteps - 1)
      let w = timestep - Float(lowIdx)
      let logSigma =
        (1 - w) * log(sigmasForTimesteps[lowIdx]) + w * log(sigmasForTimesteps[highIdx])
      sigmas.append(exp(logSigma))
    }
    sigmas.append(0)
    return sigmas
  }

  public static func sigmas(from alphasCumprod: [Float]) -> [Float] {
    return alphasCumprod.map { ((1 - $0) / $0).squareRoot() }
  }

  public static func timestep(from sigma: Float, sigmas: [Float]) -> Float {
    guard sigma > sigmas[0] else {
      return 0
    }
    guard sigma < sigmas[sigmas.count - 1] else {
      return Float(sigmas.count - 1)
    }
    // Find in between which sigma resides.
    var highIdx: Int = sigmas.count - 1
    var lowIdx: Int = 0
    while lowIdx < highIdx - 1 {
      let midIdx = lowIdx + (highIdx - lowIdx) / 2
      if sigma < sigmas[midIdx] {
        highIdx = midIdx
      } else {
        lowIdx = midIdx
      }
    }
    assert(sigma >= sigmas[highIdx - 1] && sigma <= sigmas[highIdx])
    let low = log(sigmas[highIdx - 1])
    let high = log(sigmas[highIdx])
    let logSigma = log(sigma)
    let w = min(max((low - logSigma) / (low - high), 0), 1)
    return (1.0 - w) * Float(highIdx - 1) + w * Float(highIdx)
  }
}

DynamicGraph.setSeed(40)
DynamicGraph.memoryEfficient = true

let unconditionalGuidanceScale: Float = 7.5
let scaleFactor: Float = 0.18215
let strength: Float = 0.75
var startWidth: Int = 64
var startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 30)
let alphasCumprod = model.alphasCumprod
let sigmasForTimesteps = DiffusionModel.sigmas(from: alphasCumprod)
let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let alphas = alphasCumprod.map { $0.squareRoot() }
let sigmas = alphasCumprod.map { (1 - $0).squareRoot() }
let lambdas = zip(alphas, sigmas).map { log($0) - log($1) }

let workDir = CommandLine.arguments[1]
let text =
  CommandLine.arguments.count > 2
  ? CommandLine.arguments.suffix(from: 2).joined(separator: " ") : ""

let tokens = tokenizer.tokenize(text: text, truncation: true, maxLength: 77)

var initImg = Tensor<FloatType>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
if let image = try PNG.Data.Rectangular.decompress(path: workDir + "/init_img.png") {
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  // print(rgba)

  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let pixel = rgba[y * startWidth * 8 + x]
      initImg[0, 0, y, x] = FloatType(Float(pixel.r) / 255 * 2 - 1)
      initImg[0, 1, y, x] = FloatType(Float(pixel.g) / 255 * 2 - 1)
      initImg[0, 2, y, x] = FloatType(Float(pixel.b) / 255 * 2 - 1)
    }
  }
}

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

let tokensTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
for i in 0..<77 {
  tokensTensor[i] = tokens[i]
  positionTensor[i] = Int32(i)
}

let casualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
  }
}

let unet = ModelBuilder {
  let startWidth = $0[0].shape[3]
  let startHeight = $0[0].shape[2]
  return LoRAUNet(batchSize: 1, startWidth: startWidth, startHeight: startHeight)
}

unet.maxConcurrency = .limit(1)

let decoder = ModelBuilder {
  let startWidth = $0[0].shape[3]
  let startHeight = $0[0].shape[2]
  return Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
    startHeight: startHeight)
}

graph.workspaceSize = 0  // 1_024 * 1_024 * 1_024

let latents = graph.withNoGrad {
  let encoder = ModelBuilder {
    let startWidth = $0[0].shape[3] / 8
    let startHeight = $0[0].shape[2] / 8
    return Encoder(
      channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
      startHeight: startHeight)
  }
  let initImgGPU = graph.variable(initImg.toGPU(0))
  encoder.compile(inputs: initImgGPU)
  graph.openStore("/fast/Data/SD/swift-diffusion/vae_ft_mse_840000_f32.ckpt") {
    $0.read("encoder", model: encoder)
  }
  let encoded = encoder(inputs: initImgGPU)[0].as(of: FloatType.self)
  return scaleFactor * encoded[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
}

let tokensTensorGPU = tokensTensor.toGPU(0)
let positionTensorGPU = positionTensor.toGPU(0)
let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)

let textModel = LoRACLIPTextModel(
  FloatType.self,
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
  batchSize: 1, intermediateSize: 3072)

textModel.maxConcurrency = .limit(1)
textModel.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
graph.openStore("/fast/Data/SD/swift-diffusion/sd-v1.5.ckpt") { store in
  store.read("text_model", model: textModel) { name, dataType, format, shape in
    if name.contains("lora_up") {
      precondition(dataType == .Float32)
      var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<Float32>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return .final(tensor)
    }
    return .continue(name)
  }
}

let c = textModel(
  inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
    of: FloatType.self
  ).reshaped(.CHW(1, 77, 768))

let ts = timeEmbedding(timestep: 0, batchSize: 1, embeddingSize: 320, maxPeriod: 10_000).toGPU(0)
unet.compile(inputs: latents, graph.variable(Tensor<FloatType>(from: ts)), c)
graph.openStore("/fast/Data/SD/swift-diffusion/sd-v1.5.ckpt") { store in
  store.read("unet", model: unet) { name, dataType, format, shape in
    if name.contains("lora_up") {
      precondition(dataType == .Float32)
      var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<Float32>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return .final(tensor)
    }
    return .continue(name)
  }
}

var adamWOptimizer = AdamWOptimizer(
  graph, rate: 0.0001, betas: (0.9, 0.999), decay: 0.001, epsilon: 1e-8)
adamWOptimizer.parameters = [unet.parameters]
let startTime = Date()
var accumulateGradSteps = 0
let minSNRGamma: Float = 1
var scaler = GradScaler()
for epoch in 0..<1000 {
  let noise = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
  noise.randn(std: 1, mean: 0)
  let timestep = Int.random(in: 0...999)
  let c = textModel(
    inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
      of: FloatType.self
    ).reshaped(.CHW(1, 77, 768))
  let sqrtAlphasCumprod = alphasCumprod[timestep].squareRoot()
  let sqrtOneMinusAlphasCumprod = (1 - alphasCumprod[timestep]).squareRoot()
  let snr = alphasCumprod[timestep] / (1 - alphasCumprod[timestep])
  let gammaOverSNR = minSNRGamma / snr
  let snrWeight = min(gammaOverSNR, 1)
  let noisyLatents = sqrtAlphasCumprod * latents + sqrtOneMinusAlphasCumprod * noise
  let ts = timeEmbedding(timestep: timestep, batchSize: 1, embeddingSize: 320, maxPeriod: 10_000)
    .toGPU(0)
  let t = graph.variable(Tensor<FloatType>(from: ts))
  let et = unet(inputs: noisyLatents, t, c)[0].as(of: FloatType.self)  // .reshaped(
  //  .NC(1, 4 * startWidth * startHeight))
  // let loss = snrWeight * MSELoss()(et, target: noise.reshaped(.NC(1, 4 * startWidth * startHeight)))[0].as(
  //   of: FloatType.self)
  let d = et - noise
  let loss = snrWeight * (d .* d).reduced(.mean, axis: [1, 2, 3])
  scaler.scale(loss).backward(to: [latents, tokensTensorGPU])
  let value = loss.toCPU()[0, 0, 0, 0]
  if accumulateGradSteps == 5 {
    scaler.step(&adamWOptimizer)
    accumulateGradSteps = 0
  } else {
    accumulateGradSteps += 1
  }
  print(
    "epoch: \(epoch), \(timestep), loss: \(value), step \(adamWOptimizer.step), scale \(scaler.scale)"
  )
  if value.isNaN {
    fatalError()
  }
}
graph.openStore(workDir + "/lora_training.ckpt") {
  $0.write("lora_unet", model: unet)
  $0.write("lora_text_model", model: textModel)
}
print("Total time \(Date().timeIntervalSince(startTime))")
