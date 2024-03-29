import C_ccv
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
}

DynamicGraph.setSeed(43)

let unconditionalGuidanceScale: Float = 7.5
let scaleFactor: Float = 0.18215
let strength: Float = 0.99
let startWidth: Int = 64
let startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 50)
let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let workDir = CommandLine.arguments[1]
let text =
  CommandLine.arguments.count > 2
  ? CommandLine.arguments.suffix(from: 2).joined(separator: " ") : ""

var initImg = Tensor<UseFloatingPoint>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
var hintImg = Tensor<UseFloatingPoint>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
var u8Img = ccv_dense_matrix_new(
  Int32(startHeight * 8), Int32(startWidth * 8), Int32(CCV_8U | CCV_C1), nil, 0)!
if let image = try PNG.Data.Rectangular.decompress(path: workDir + "/init_img.png") {
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let pixel = rgba[y * startWidth * 8 + x]
      u8Img.pointee.data.u8[y * startWidth * 8 + x] = UInt8(
        (Int32(pixel.r) * 6969 + Int32(pixel.g) * 23434 + Int32(pixel.b) * 2365) >> 15)
      initImg[0, 0, y, x] = UseFloatingPoint(Float(pixel.r) / 255 * 2 - 1)
      initImg[0, 1, y, x] = UseFloatingPoint(Float(pixel.g) / 255 * 2 - 1)
      initImg[0, 2, y, x] = UseFloatingPoint(Float(pixel.b) / 255 * 2 - 1)
    }
  }
}
var cannyImg: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
ccv_canny(u8Img, &cannyImg, 0, 3, 100, 200)
var canny = [PNG.RGBA<UInt8>](repeating: .init(0), count: startWidth * 8 * startHeight * 8)
for y in 0..<startHeight * 8 {
  for x in 0..<startWidth * 8 {
    if cannyImg!.pointee.data.u8[y * startWidth * 8 + x] == 0 {
      hintImg[0, 0, y, x] = 0
      hintImg[0, 1, y, x] = 0
      hintImg[0, 2, y, x] = 0
      canny[y * startWidth * 8 + x].r = 255
      canny[y * startWidth * 8 + x].g = 255
      canny[y * startWidth * 8 + x].b = 255
    } else {
      hintImg[0, 0, y, x] = 1  // Black lines.
      hintImg[0, 1, y, x] = 1
      hintImg[0, 2, y, x] = 1
      canny[y * startWidth * 8 + x].r = 0
      canny[y * startWidth * 8 + x].g = 0
      canny[y * startWidth * 8 + x].b = 0
    }
  }
}
let cannyPNG = PNG.Data.Rectangular(
  packing: canny, size: (startWidth * 8, startHeight * 8),
  layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
try! cannyPNG.compress(path: workDir + "/canny.png", level: 4)

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
let unet = UNet(batchSize: 2, startWidth: startWidth, startHeight: startHeight, control: true)
let controlnet = ControlNet(batchSize: 2)
let hintnet = HintNet()
let encoder = Encoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
  startHeight: startHeight)
let decoder = Decoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
  startHeight: startHeight)
// let adapternet = Adapter(channels: [320, 640, 1280, 1280], numRepeat: 2)
// let adapternet = AdapterLight(channels: [320, 640, 1280, 1280], numRepeat: 4)

graph.workspaceSize = 1_024 * 1_024 * 1_024

graph.withNoGrad {
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  textModel.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/fast/Data/SD/swift-diffusion/clip_vit_l14_f16.ckpt") {
    $0.read("text_model", model: textModel)
  }
  let c = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
    of: UseFloatingPoint.self
  ).reshaped(.CHW(2, 77, 768))
  let noise = graph.variable(
    .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  noise.randn(std: 1, mean: 0)
  /*
  let hint = graph.variable(hintImg[0..<1, 0..<1, 0..<startHeight * 8, 0..<startWidth * 8].toGPU(0))
  let hintIn = hint.reshaped(format: .NCHW, shape: [1, 1, startHeight, 8, startWidth, 8]).permuted(
    0, 1, 3, 5, 2, 4
  ).copied().reshaped(.NCHW(1, 64, startHeight, startWidth))
  adapternet.compile(inputs: hintIn)
  graph.openStore(workDir + "/t2iadapter_color_1.x_f32.ckpt") {
    $0.read("adapter", model: adapternet)
  }
  let adapters = adapternet(inputs: hintIn).map { $0.as(of: UseFloatingPoint.self) }
  graph.openStore(workDir + "/t2iadapter_color_1.x_f16.ckpt") {
    $0.write("adapter", model: adapternet)
  }
  */
  let hint = graph.variable(hintImg.toGPU(0))
  hintnet.compile(inputs: hint)
  graph.openStore(workDir + "/controlnet_mlsd_1.x_v1.1_f32.ckpt") {
    $0.read("hintnet", model: hintnet)
  }
  let guidance = hintnet(inputs: hint)[0].as(of: UseFloatingPoint.self)
  var x = noise
  var xIn = graph.variable(.GPU(0), .NCHW(2, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  let ts0 = graph.variable(Tensor<UseFloatingPoint>(from: ts[0]))
  controlnet.compile(inputs: xIn, guidance, ts0, c)
  graph.openStore(workDir + "/controlnet_mlsd_1.x_v1.1_f32.ckpt") {
    $0.read("controlnet", model: controlnet)
  }
  graph.openStore(workDir + "/controlnet_mlsd_1.x_v1.1_f16.ckpt") {
    $0.write("hintnet", model: hintnet)
    $0.write("controlnet", model: controlnet)
  }
  var controls = controlnet(inputs: xIn, guidance, ts0, c).map { $0.as(of: UseFloatingPoint.self) }
  // var controls = adapters
  controls.insert(contentsOf: [xIn, ts0, c], at: 0)
  unet.compile(inputs: controls)
  decoder.compile(inputs: x)
  let initImg = graph.variable(initImg.toGPU(0))
  encoder.compile(inputs: initImg)
  graph.openStore("/fast/Data/SD/swift-diffusion/sd_v1.5_f16.ckpt") {
    $0.read("unet", model: unet)
  }
  graph.openStore("/fast/Data/SD/swift-diffusion/vae_ft_mse_840000_f32.ckpt") {
    $0.read("decoder", model: decoder)
    $0.read("encoder", model: encoder)
  }
  let parameters = encoder(inputs: initImg)[0].as(of: UseFloatingPoint.self)
  let mean = parameters[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
  let logvar = parameters[0..<1, 4..<8, 0..<startHeight, 0..<startWidth].clamped(-30...20)
  let std = Functional.exp(0.5 * logvar)
  let n = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: UseFloatingPoint.self)
  n.randn(std: 1, mean: 0)
  let sample = scaleFactor * (mean + std .* n)
  let alphasCumprod = model.alphasCumprod
  let startTime = Date()
  let tEnc = Int(strength * Float(model.steps))
  let initTimestep = model.timesteps - model.timesteps / model.steps * (model.steps - tEnc) + 1
  let sqrtAlphasCumprod = alphasCumprod[initTimestep].squareRoot()
  let sqrtOneMinusAlphasCumprod = (1 - alphasCumprod[initTimestep]).squareRoot()
  let zEnc = sqrtAlphasCumprod * sample + sqrtOneMinusAlphasCumprod * noise
  x = zEnc
  DynamicGraph.setProfiler(true)
  // Now do DDIM sampling.
  for i in (model.steps - tEnc)..<model.steps {
    let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
    let t = graph.variable(Tensor<UseFloatingPoint>(from: ts[i]))
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = x
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = x
    var controls = controlnet(inputs: xIn, guidance, t, c).map { $0.as(of: UseFloatingPoint.self) }
    // var controls = adapters
    controls.insert(contentsOf: [t, c], at: 0)
    var et = unet(inputs: xIn, controls)[0].as(of: UseFloatingPoint.self)
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
  try! image.compress(path: workDir + "/canny2img.png", level: 4)
}
