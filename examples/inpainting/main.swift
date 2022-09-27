import C_ccv
import Diffusion
import Foundation
import NNC

// Unlike img2img and txt2img, CompVis repo's inpaint is not what most people use. This inpaint
// implementation is in spirit with outpainting implemented in AUTOMATIC1111's repo:
// https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/scripts/outpainting_mk_2.py
// i.e. we simply mask out noise with the masks for input latent and run img2img pipeline.
// Note that this is partially implemented in ddim_sampling (as shown with the mask parameter).

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

DynamicGraph.setSeed(42)

let unconditionalGuidanceScale: Float = 5
let scaleFactor: Float = 0.18215
let strength: Float = 0.75
let startWidth: Int = 64
let startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 50)
let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let workDir = CommandLine.arguments[1]
let text = CommandLine.arguments.suffix(2).joined(separator: " ")

var initImage: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
let _ = (workDir + "/init_inpainting.png").withCString {
  ccv_read_impl($0, &initImage, Int32(CCV_IO_ANY_FILE), 0, 0, 0)
}
var initImg = Tensor<Float>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
var initMask = Tensor<Float>(.CPU, .NCHW(1, 1, startHeight, startWidth))
if let initImage = initImage {
  for y in 0..<startHeight {
    for x in 0..<startWidth {
      initMask[0, 0, y, x] = 0
    }
  }
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let r = initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3]
      let g = initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 1]
      let b = initImage.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 2]
      if g == 255 && r == 0 && b == 0 {
        initMask[0, 0, y / 8, x / 8] = 1
        initImg[0, 0, y, x] = 0
        initImg[0, 1, y, x] = 0
        initImg[0, 2, y, x] = 0
      } else {
        initImg[0, 0, y, x] = Float(r) / 255 * 2 - 1
        initImg[0, 1, y, x] = Float(g) / 255 * 2 - 1
        initImg[0, 2, y, x] = Float(b) / 255 * 2 - 1
      }
    }
  }
}

let unconditionalTokens = tokenizer.tokenize(text: "", truncation: true, maxLength: 77)
let tokens = tokenizer.tokenize(text: text, truncation: true, maxLength: 77)

let graph = DynamicGraph()

let textModel = CLIPTextModel(
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

let casualAttentionMask = graph.variable(Tensor<Float>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -Float.greatestFiniteMagnitude
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
  let _ = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore(workDir + "/sd-v1.4.ckpt") {
    $0.read("text_model", model: textModel)
  }
  let c = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
    of: Float.self
  ).reshaped(.CHW(2, 77, 768))
  let noise = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
  noise.randn(std: 1, mean: 0)
  var x = noise
  var xIn = graph.variable(.GPU(0), .NCHW(2, 4, startHeight, startWidth), of: Float.self)
  let _ = unet(inputs: xIn, graph.variable(ts[0]), c)
  let _ = decoder(inputs: x)
  let initImg = graph.variable(initImg.toGPU(0))
  let _ = encoder(inputs: initImg)
  graph.openStore(workDir + "/sd-v1.4.ckpt") {
    $0.read("unet", model: unet)
    $0.read("decoder", model: decoder)
    $0.read("encoder", model: encoder)
  }
  let parameters = encoder(inputs: initImg)[0].as(of: Float.self)
  let mean = parameters[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
  let logvar = parameters[0..<1, 4..<8, 0..<startHeight, 0..<startWidth].clamped(-30...20)
  let initMask = graph.variable(initMask.toGPU(0))
  let std = Functional.exp(0.5 * logvar) .* initMask
  let n = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
  n.randn(std: 1, mean: 0)
  let sample = scaleFactor * (mean + std .* n)
  let alphasCumprod = model.alphasCumprod
  let startTime = Date()
  let tEnc = Int(strength * Float(model.steps))
  let initTimestep = model.timesteps - model.timesteps / model.steps * (model.steps - tEnc) + 1
  let sqrtAlphasCumprod = alphasCumprod[initTimestep].squareRoot()
  let sqrtOneMinusAlphasCumprod = (1 - alphasCumprod[initTimestep]).squareRoot()
  var initNegMask = graph.variable(.GPU(0), .NCHW(1, 1, startHeight, startWidth), of: Float.self)
  initNegMask.full(1)
  initNegMask = initNegMask - initMask
  let zEnc = sqrtAlphasCumprod * sample + sqrtOneMinusAlphasCumprod * noise
  x = zEnc .* initNegMask + noise .* initMask
  DynamicGraph.setProfiler(true)
  // Now do DDIM sampling.
  for i in (model.steps - tEnc)..<model.steps {
    let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
    let t = graph.variable(ts[i])
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = x
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = x
    var et = unet(inputs: xIn, t, c)[0].as(of: Float.self)
    var etUncond = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
    var etCond = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
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
  let img = decoder(inputs: z)[0].as(of: Float.self).toCPU()
  print("Total time \(Date().timeIntervalSince(startTime))")
  let image = ccv_dense_matrix_new(
    Int32(startHeight * 8), Int32(startWidth * 8), Int32(CCV_8U | CCV_C3), nil, 0)
  // I have better way to copy this out (basically, transpose and then ccv_shift). Doing this just for fun.
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let (r, g, b) = (img[0, 0, y, x], img[0, 1, y, x], img[0, 2, y, x])
      image!.pointee.data.u8[y * startWidth * 8 * 3 + x * 3] = UInt8(
        min(max(Int(Float((r + 1) / 2) * 255), 0), 255))
      image!.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 1] = UInt8(
        min(max(Int(Float((g + 1) / 2) * 255), 0), 255))
      image!.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 2] = UInt8(
        min(max(Int(Float((b + 1) / 2) * 255), 0), 255))
    }
  }
  let _ = (workDir + "/inpainting.png").withCString {
    ccv_write(image, UnsafeMutablePointer(mutating: $0), nil, Int32(CCV_IO_PNG_FILE), nil)
  }
}
