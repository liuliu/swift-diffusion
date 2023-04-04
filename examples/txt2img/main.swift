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
var startWidth: Int = 64
var startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 30)
let alphasCumprod = model.alphasCumprod
let sigmasForTimesteps = DiffusionModel.sigmas(from: alphasCumprod)
// This is for Karras scheduler (used in DPM++ 2M Karras)
// let sigmas = model.karrasSigmas(sigmasForTimesteps[0]...sigmasForTimesteps[999])
// This is for Euler Ancestral
// let sigmas = model.fixedStepSigmas(
//   sigmasForTimesteps[0]...sigmasForTimesteps[999], sigmas: sigmasForTimesteps)
let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let alphas = alphasCumprod.map { $0.squareRoot() }
let sigmas = alphasCumprod.map { (1 - $0).squareRoot() }
let lambdas = zip(alphas, sigmas).map { log($0) - log($1) }

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

let unet = ModelBuilder {
  let startWidth = $0[0].shape[3]
  let startHeight = $0[0].shape[2]
  return UNet(batchSize: 2, startWidth: startWidth, startHeight: startHeight)
}
let decoder = ModelBuilder {
  let startWidth = $0[0].shape[3]
  let startHeight = $0[0].shape[2]
  return Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
    startHeight: startHeight)
}

graph.workspaceSize = 1_024 * 1_024 * 1_024

func uniPBhUpdate(
  mt: DynamicGraph.Tensor<UseFloatingPoint>, prevTimestep t: Int,
  sample x: DynamicGraph.Tensor<UseFloatingPoint>, timestepList: [Int],
  outputList: [DynamicGraph.Tensor<UseFloatingPoint>], lambdas: [Float], alphas: [Float],
  sigmas: [Float]
) -> DynamicGraph.Tensor<UseFloatingPoint> {
  let s0 = timestepList[timestepList.count - 1]
  let m0 = outputList[outputList.count - 1]
  let lambdat = lambdas[t]
  let lambdas0 = lambdas[s0]
  let alphat = alphas[t]
  let sigmat = sigmas[t]
  let sigmas0 = sigmas[s0]
  let h = lambdat - lambdas0
  let D1: DynamicGraph.Tensor<UseFloatingPoint>?
  if timestepList.count >= 2 && outputList.count >= 2 {
    let si = timestepList[timestepList.count - 2]
    let mi = outputList[outputList.count - 2]
    let lambdasi = lambdas[si]
    let rk = (lambdasi - lambdas0) / h
    D1 = (mi - m0) / rk
  } else {
    D1 = nil
  }
  let hh = -h
  let hPhi1 = exp(hh) - 1
  let Bh = hPhi1
  let rhosP: Float = 0.5
  let xt_ = Functional.add(
    left: x, right: m0, leftScalar: sigmat / sigmas0, rightScalar: -alphat * hPhi1)
  if let D1 = D1 {
    let xt = Functional.add(left: xt_, right: D1, leftScalar: 1, rightScalar: alphat * Bh * rhosP)
    return xt
  } else {
    return xt_
  }
}

func uniCBhUpdate(
  mt: DynamicGraph.Tensor<UseFloatingPoint>, timestep t: Int,
  lastSample x: DynamicGraph.Tensor<UseFloatingPoint>, timestepList: [Int],
  outputList: [DynamicGraph.Tensor<UseFloatingPoint>], lambdas: [Float], alphas: [Float],
  sigmas: [Float]
) -> DynamicGraph.Tensor<UseFloatingPoint> {
  let s0 = timestepList[timestepList.count - 1]
  let m0 = outputList[outputList.count - 1]
  let lambdat = lambdas[t]
  let lambdas0 = lambdas[s0]
  let alphat = alphas[t]
  let sigmat = sigmas[t]
  let sigmas0 = sigmas[s0]
  let h = lambdat - lambdas0
  let hh = -h
  let hPhi1 = exp(hh) - 1
  let hPhik = hPhi1 / hh - 1
  let Bh = hPhi1
  let D1: DynamicGraph.Tensor<UseFloatingPoint>?
  let rhosC0: Float
  let rhosC1: Float
  if timestepList.count >= 2 && outputList.count >= 2 {
    let si = timestepList[timestepList.count - 2]
    let mi = outputList[outputList.count - 2]
    let lambdasi = lambdas[si]
    let rk = (lambdasi - lambdas0) / h
    D1 = (mi - m0) / rk
    let b0 = hPhik / Bh
    let b1 = (hPhik / hh - 0.5) * 2 / Bh
    rhosC0 = (b0 - b1) / (1 - rk)
    rhosC1 = b0 - rhosC0
  } else {
    D1 = nil
    rhosC0 = 0.5
    rhosC1 = 0.5
  }
  print("rhos_c \(rhosC0), rhos_c \(rhosC1)")
  let xt_ = Functional.add(
    left: x, right: m0, leftScalar: sigmat / sigmas0, rightScalar: -alphat * hPhi1)
  let D1t = mt - m0
  let D1s: DynamicGraph.Tensor<UseFloatingPoint>
  if let D1 = D1 {
    D1s = Functional.add(left: D1, right: D1t, leftScalar: rhosC0, rightScalar: rhosC1)
  } else {
    D1s = rhosC1 * D1t
  }
  let xt = Functional.add(left: xt_, right: D1s, leftScalar: 1, rightScalar: -alphat * Bh)
  return xt
}

graph.withNoGrad {
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  textModel.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/fast/Data/SD/swift-diffusion/sd-v1.5.ckpt") { store in
    store.read("text_model", model: textModel)
  }
  /*
  graph.openStore(workDir + "/moxin_v1.0_lora_f16.ckpt") { lora in
    let keys = Set(lora.keys)
    graph.openStore(workDir + "/sd-v1.5.ckpt") { store in
      store.read("text_model", model: textModel) { name, _, _, _ in
        if keys.contains(name + "__up__") {
          let original = graph.variable(Tensor<UseFloatingPoint>(from: store.read(name)!)).toGPU(0)
          let up = graph.variable(Tensor<UseFloatingPoint>(lora.read(name + "__up__")!)).toGPU(0)
          let down = graph.variable(Tensor<UseFloatingPoint>(lora.read(name + "__down__")!)).toGPU(0)
          let final = original + 0.8 * (up * down)
          return .final(final.rawValue.toCPU())
        }
        return .continue(name)
      }
    }
  }
  */
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
  graph.openStore("/fast/Data/SD/swift-diffusion/sd-v1.5.ckpt") { store in
    store.read("unet", model: unet)
    store.read("decoder", model: decoder)
  }
  /*
  graph.openStore(workDir + "/moxin_v1.0_lora_f16.ckpt") { lora in
    let keys = Set(lora.keys)
    graph.openStore(workDir + "/sd-v1.5.ckpt") { store in
      store.read("unet", model: unet) { name, _, _, _ in
        if keys.contains(name + "__up__") {
          let original = graph.variable(Tensor<UseFloatingPoint>(from: store.read(name)!)).toGPU(0)
          let up: DynamicGraph.Tensor<UseFloatingPoint>
          let down: DynamicGraph.Tensor<UseFloatingPoint>
          let result: DynamicGraph.Tensor<UseFloatingPoint>
          if original.shape.count == 4 {
            let loraUp = Tensor<UseFloatingPoint>(lora.read(name + "__up__")!)
            up = graph.variable(loraUp.reshaped(.NC(loraUp.shape[0], loraUp.shape[1] * loraUp.shape[2] * loraUp.shape[3]))).toGPU(0)
            let loraDown = Tensor<UseFloatingPoint>(lora.read(name + "__down__")!)
            down = graph.variable(loraDown.reshaped(.NC(loraDown.shape[0], loraDown.shape[1] * loraDown.shape[2] * loraDown.shape[3]))).toGPU(0)
            result = original + 0.8 * (up * down).reshaped(format: .NCHW, shape: original.shape)
          } else {
            up = graph.variable(Tensor<UseFloatingPoint>(lora.read(name + "__up__")!)).toGPU(0)
            down = graph.variable(Tensor<UseFloatingPoint>(lora.read(name + "__down__")!)).toGPU(0)
            result = original + 0.8 * (up * down)
          }
          return .final(result.rawValue)
        }
        return .continue(name)
      }
      store.read("decoder", model: decoder)
    }
  }
  */
  // var oldDenoised: DynamicGraph.Tensor<UseFloatingPoint>? = nil
  var timestepList = [Int]()
  var outputList = [DynamicGraph.Tensor<UseFloatingPoint>]()
  let startTime = Date()
  var lastSample: DynamicGraph.Tensor<UseFloatingPoint>? = nil
  // Now do DPM++ 2M Karras sampling. (DPM++ 2S a Karras requires two denoising per step, not ideal for my use case).
  // x = sigmas[0] * x
  for i in 0..<model.steps {
    // let sigma = sigmas[i]
    // let timestep = DiffusionModel.timestep(from: sigma, sigmas: sigmasForTimesteps)
    let timestep = model.timesteps - model.timesteps / model.steps * i - 1
    let ts = timeEmbedding(timestep: timestep, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000)
      .toGPU(0)
    let t = graph.variable(Tensor<UseFloatingPoint>(from: ts))
    // let cIn = 1.0 / (sigma * sigma + 1).squareRoot()
    // let cOut = -sigma
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = x  // cIn * x
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = x  // cIn * x
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
    // UniPC sampler.
    let mt = Functional.add(
      left: x, right: et, leftScalar: 1.0 / alphas[timestep],
      rightScalar: -sigmas[timestep] / alphas[timestep])
    let useCorrector = lastSample != nil
    if useCorrector, let lastSample = lastSample {
      x = uniCBhUpdate(
        mt: mt, timestep: timestep, lastSample: lastSample, timestepList: timestepList,
        outputList: outputList, lambdas: lambdas, alphas: alphas, sigmas: sigmas)
    }
    if timestepList.count < 2 {
      timestepList.append(timestep)
    } else {
      timestepList[0] = timestepList[1]
      timestepList[1] = timestep
    }
    if outputList.count < 2 {
      outputList.append(mt)
    } else {
      outputList[0] = outputList[1]
      outputList[1] = mt
    }
    let prevTimestep = max(0, model.timesteps - model.timesteps / model.steps * (i + 1) - 1)
    lastSample = x
    x = uniPBhUpdate(
      mt: mt, prevTimestep: prevTimestep, sample: x, timestepList: timestepList,
      outputList: outputList, lambdas: lambdas, alphas: alphas, sigmas: sigmas)
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
    /* // Below is the DPM++ 2M Karras sampling implementation.
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
    oldDenoised = denoised */
  }
  /*
  startWidth = 96
  startHeight = 96
  x = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  x.randn(std: 1, mean: 0)
  xIn = graph.variable(.GPU(0), .NCHW(2, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  x = sigmas[0] * x
  oldDenoised = nil
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
  */
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
