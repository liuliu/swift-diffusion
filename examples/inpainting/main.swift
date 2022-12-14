import C_ccv
import Diffusion
import Foundation
import NNC
import PNG

// Unlike img2img and txt2img, CompVis repo's inpaint is not what most people use. This inpaint
// implementation is in spirit with outpainting implemented in AUTOMATIC1111's repo:
// https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/outpainting_mk_2.py
// i.e. we simply mask out noise with the masks for input latent and run img2img pipeline.
// Note that this is partially implemented in ddim_sampling (as shown with the mask parameter).

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
}

DynamicGraph.setSeed(38)

let unconditionalGuidanceScale: Float = 10
let scaleFactor: Float = 0.18215
let strength: Float = 0.75
let startWidth: Int = 64
let startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 50)
let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let workDir = CommandLine.arguments[1]
var text =
  CommandLine.arguments.count > 2
  ? CommandLine.arguments.suffix(from: 2).joined(separator: " ") : ""

var initImg = Tensor<UseFloatingPoint>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
var initMask = Tensor<UseFloatingPoint>(.CPU, .NCHW(1, 1, startHeight, startWidth))
var initImage = ccv_dense_matrix_new(
  Int32(startHeight * 8), Int32(startWidth * 8), Int32(CCV_8U | CCV_C3), nil, 0)!
if let image = try PNG.Data.Rectangular.decompress(path: workDir + "/init_inpainting.png") {
  for y in 0..<startHeight {
    for x in 0..<startWidth {
      initMask[0, 0, y, x] = 0
    }
  }
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let pixel = rgba[y * startWidth * 8 + x]
      if pixel.g == 255 && pixel.r == 0 && pixel.b == 0 {
        initMask[0, 0, y / 8, x / 8] = 1
        initImg[0, 0, y, x] = 0
        initImg[0, 1, y, x] = 0
        initImg[0, 2, y, x] = 0
        initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3] = 123
        initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 1] = 117
        initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 2] = 104
      } else {
        initImg[0, 0, y, x] = UseFloatingPoint(Float(pixel.r) / 255 * 2 - 1)
        initImg[0, 1, y, x] = UseFloatingPoint(Float(pixel.g) / 255 * 2 - 1)
        initImg[0, 2, y, x] = UseFloatingPoint(Float(pixel.b) / 255 * 2 - 1)
        initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3] = pixel.r
        initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 1] = pixel.g
        initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 2] = pixel.b
      }
    }
  }
}

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

var ts = [Tensor<Float>]()
for i in 0..<model.steps {
  let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
  ts.append(
    timeEmbedding(timestep: timestep, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000).toGPU(0))
}
let unet = UNet(batchSize: 2, startWidth: startWidth, startHeight: startHeight)
let encoder = Encoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
  startHeight: startHeight)
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
  let c = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
    of: UseFloatingPoint.self
  ).reshaped(.CHW(2, 77, 768))
  let noise = graph.variable(
    .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  noise.randn(std: 1, mean: 0)
  var x = noise
  var xIn = graph.variable(.GPU(0), .NCHW(2, 9, startHeight, startWidth), of: UseFloatingPoint.self)
  unet.compile(inputs: xIn, graph.variable(Tensor<UseFloatingPoint>(from: ts[0])), c)
  decoder.compile(inputs: x)
  let initImg = graph.variable(initImg.toGPU(0))
  encoder.compile(inputs: initImg)
  graph.openStore(workDir + "/sd-v1.4.ckpt") {
    $0.read("decoder", model: decoder)
    $0.read("encoder", model: encoder)
  }
  graph.openStore(workDir + "/sd-v1.5-inpainting.ckpt") {
    $0.read("unet", model: unet)
  }
  let parameters = encoder(inputs: initImg)[0].as(of: UseFloatingPoint.self)
  let mean = parameters[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
  let logvar = parameters[0..<1, 4..<8, 0..<startHeight, 0..<startWidth].clamped(-30...20)
  let initMask = graph.variable(initMask.toGPU(0))
  let std = Functional.exp(0.5 * logvar) .* initMask
  let n = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  n.randn(std: 1, mean: 0)
  let sample = scaleFactor * (mean + std .* n)
  let alphasCumprod = model.alphasCumprod
  let startTime = Date()
  let tEnc = Int(strength * Float(model.steps))
  let initTimestep = model.timesteps - model.timesteps / model.steps * (model.steps - tEnc) + 1
  let sqrtAlphasCumprod = alphasCumprod[initTimestep].squareRoot()
  let sqrtOneMinusAlphasCumprod = (1 - alphasCumprod[initTimestep]).squareRoot()
  var initNegMask = graph.variable(
    .GPU(0), .NCHW(1, 1, startHeight, startWidth), of: UseFloatingPoint.self)
  initNegMask.full(1)
  initNegMask = initNegMask - initMask
  let maskedImg = scaleFactor * mean
  let zEnc = sqrtAlphasCumprod * sample + sqrtOneMinusAlphasCumprod * noise
  x = zEnc .* initNegMask + noise .* initMask
  DynamicGraph.setProfiler(true)
  // Now do DDIM sampling.
  for i in (model.steps - tEnc)..<model.steps {
    let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
    let t = graph.variable(Tensor<UseFloatingPoint>(from: ts[i]))
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = x
    xIn[0..<1, 4..<5, 0..<startHeight, 0..<startWidth] = initMask
    xIn[0..<1, 5..<9, 0..<startHeight, 0..<startWidth] = maskedImg
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = x
    xIn[1..<2, 4..<5, 0..<startHeight, 0..<startWidth] = initMask
    xIn[1..<2, 5..<9, 0..<startHeight, 0..<startWidth] = maskedImg
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
    let alpha = alphasCumprod[timestep]
    let alphaPrev = alphasCumprod[max(timestep - model.timesteps / model.steps, 0)]
    let predX0 = (1 / alpha.squareRoot()) * (x - (1 - alpha).squareRoot() * et)
    let dirXt = (1 - alphaPrev).squareRoot() * et
    let xPrev = alphaPrev.squareRoot() * predX0 + dirXt
    x = xPrev
    if i < model.steps - 1 {
      // Apply mask repeatedly during the diffusion process. Do it or not in the last step doesn't
      // practically matter since the alpha is so small. Don't do it to match CompVis repo.
      noise.randn(std: 1, mean: 0)
      let qSample = alpha.squareRoot() * sample + (1 - alpha).squareRoot() * noise
      x = qSample .* initNegMask + x .* initMask
    }
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
  try! image.compress(path: workDir + "/inpainting.png", level: 4)
}
