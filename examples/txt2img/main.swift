import Diffusion
import Foundation
import NNC
import PNG

public typealias UseFloatingPoint = Float16

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
let startWidth: Int = 64
let startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 30)
let alphasCumprod = model.alphasCumprod
let sigmasForTimesteps = DiffusionModel.sigmas(from: alphasCumprod)
// This is for Karras scheduler (used in DPM++ 2M Karras)
let sigmas = model.karrasSigmas(sigmasForTimesteps[0]...sigmasForTimesteps[999])
// This is for Euler Ancestral
// let sigmas = model.fixedStepSigmas(
//   sigmasForTimesteps[0]...sigmasForTimesteps[999], sigmas: sigmasForTimesteps)
let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let workDir = CommandLine.arguments[1]
let text =
  CommandLine.arguments.count > 2
  ? CommandLine.arguments.suffix(from: 2).joined(separator: " ") : ""

let unconditionalTokens = tokenizer.tokenize(text: "", truncation: true, maxLength: 77)
let tokens = tokenizer.tokenize(text: text, truncation: true, maxLength: 77)

let graph = DynamicGraph()

let textModel = CLIPTextModel(
  UseFloatingPoint.self,
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
  batchSize: 2, intermediateSize: 3072)

let tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
for i in 0..<77 {
  tokensTensor[i] = unconditionalTokens[i]
  tokensTensor[i + 77] = tokens[i]
  positionTensor[i] = Int32(i)
  positionTensor[i + 77] = Int32(i)
}

let casualAttentionMask = graph.variable(Tensor<UseFloatingPoint>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -UseFloatingPoint.greatestFiniteMagnitude
  }
}

let unet = UNet(batchSize: 2, startWidth: startWidth, startHeight: startHeight)
let decoder = Decoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
  startHeight: startHeight)

graph.workspaceSize = 1_024 * 1_024 * 1_024

graph.withNoGrad {
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  textModel.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore(workDir + "/sd-v1.4.ckpt") {
    $0.read("text_model", model: textModel)
  }
  let c: DynamicGraph.AnyTensor = textModel(
    inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
      of: UseFloatingPoint.self
    ).reshaped(.CHW(2, 77, 768))
  let x_T = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  x_T.randn(std: 1, mean: 0)
  var x = x_T
  var xIn = graph.variable(.GPU(0), .NCHW(2, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  let ts = timeEmbedding(timestep: 0, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000).toGPU(0)
  unet.compile(inputs: xIn, graph.variable(Tensor<UseFloatingPoint>(from: ts)), c)
  decoder.compile(inputs: x)
  graph.openStore(workDir + "/sd-v1.4.ckpt") {
    $0.read("unet", model: unet)
    $0.read("decoder", model: decoder)
  }
  var oldDenoised: DynamicGraph.Tensor<UseFloatingPoint>? = nil
  let startTime = Date()
  DynamicGraph.setProfiler(true)
  // Now do DPM++ 2M Karras sampling. (DPM++ 2S a Karras requires two denoising per step, not ideal for my use case).
  x = sigmas[0] * x
  for i in 0..<model.steps {
    let sigma = sigmas[i]
    let timestep = DiffusionModel.timestep(from: sigma, sigmas: sigmasForTimesteps)
    let ts = timeEmbedding(timestep: timestep, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000)
      .toGPU(0)
    let t = graph.variable(Tensor<UseFloatingPoint>(from: ts))
    let cIn = 1.0 / (sigma * sigma + 1).squareRoot()
    let cOut = -sigma
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = cIn * x
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = cIn * x
    var et = unet(inputs: xIn, t, c)[0].as(of: UseFloatingPoint.self)
    var etUncond = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
    var etCond = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
    etUncond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
    etCond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[1..<2, 0..<4, 0..<startHeight, 0..<startWidth]
    et = etUncond + unconditionalGuidanceScale * (etCond - etUncond)
    /* // Below is the Euler ancestral sampling implementation.
    let sigmaUp = min(
      sigmas[i + 1],
      1.0
        * ((sigmas[i + 1] * sigmas[i + 1]) * (sigma * sigma - sigmas[i + 1] * sigmas[i + 1])
        / (sigma * sigma)).squareRoot())
    let sigmaDown = (sigmas[i + 1] * sigmas[i + 1] - sigmaUp * sigmaUp).squareRoot()
    let dt = sigmaDown - sigma  // Notice this is already a negative.
    x = x + dt * et
    let noise = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
    noise.randn(std: 1, mean: 0)
    if i < model.steps - 1 {
      x = x + sigmaUp * noise
    }
    */
    // Below is the DPM++ 2M Karras sampling implementation.
    let denoised = x + cOut * et
    let h = log(sigmas[i]) - log(sigmas[i + 1])
    if let oldDenoised = oldDenoised, i < model.steps - 1 {
      let hLast = log(sigmas[i - 1]) - log(sigmas[i])
      let r = (h / hLast) / 2
      let denoisedD = (1 + r) * denoised - r * oldDenoised
      let w = sigmas[i + 1] / sigmas[i]
      x = w * x - (w - 1) * denoisedD
    } else if i == model.steps - 1 {
      x = denoised
    } else {
      let w = sigmas[i + 1] / sigmas[i]
      x = w * x - (w - 1) * denoised
    }
    oldDenoised = denoised
  }
  let z = 1.0 / scaleFactor * x
  let img = DynamicGraph.Tensor<Float>(from: decoder(inputs: z)[0].as(of: UseFloatingPoint.self))
    .toCPU()
  print("Total time \(Date().timeIntervalSince(startTime))")
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: startWidth * 8 * startHeight * 8)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let (r, g, b) = (img[0, 0, y, x], img[0, 1, y, x], img[0, 2, y, x])
      rgba[y * startWidth * 8 + x].r = UInt8(
        min(max(Int(Float((r + 1) / 2) * 255), 0), 255))
      rgba[y * startWidth * 8 + x].g = UInt8(
        min(max(Int(Float((g + 1) / 2) * 255), 0), 255))
      rgba[y * startWidth * 8 + x].b = UInt8(
        min(max(Int(Float((b + 1) / 2) * 255), 0), 255))
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (startWidth * 8, startHeight * 8),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! image.compress(path: workDir + "/txt2img.png", level: 4)
}
