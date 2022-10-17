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

DynamicGraph.setSeed(40)
DynamicGraph.memoryEfficient = true

let unconditionalGuidanceScale: Float = 7.5
let scaleFactor: Float = 0.18215
let startWidth: Int = 64
let startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 50)
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

var ts = [Tensor<Float>]()
for i in 0..<model.steps {
  let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
  ts.append(
    timeEmbedding(timestep: timestep, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000).toGPU(0))
}
let unet = UNet(batchSize: 2, startWidth: startWidth, startHeight: startHeight)
let decoder = Decoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
  startHeight: startHeight)

func xPrevAndPredX0(
  x: DynamicGraph.Tensor<UseFloatingPoint>, et: DynamicGraph.Tensor<UseFloatingPoint>, alpha: Float,
  alphaPrev: Float
) -> (DynamicGraph.Tensor<UseFloatingPoint>, DynamicGraph.Tensor<UseFloatingPoint>) {
  let predX0 = (1 / alpha.squareRoot()) * (x - (1 - alpha).squareRoot() * et)
  let dirXt = (1 - alphaPrev).squareRoot() * et
  let xPrev = alphaPrev.squareRoot() * predX0 + dirXt
  return (xPrev, predX0)
}

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
  let x_T = graph.variable(.GPU(0), .NHWC(1, startHeight, startWidth, 4), of: UseFloatingPoint.self)
  x_T.randn(std: 1, mean: 0)
  var x = x_T
  var xIn = graph.variable(.GPU(0), .NHWC(2, startHeight, startWidth, 4), of: UseFloatingPoint.self)
  unet.compile(inputs: xIn, graph.variable(Tensor<UseFloatingPoint>(from: ts[0])), c)
  decoder.compile(inputs: x)
  graph.openStore(workDir + "/sd-v1.4.ckpt") {
    $0.read("unet", model: unet)
    $0.read("decoder", model: decoder)
  }
  let alphasCumprod = model.alphasCumprod
  var oldEps = [DynamicGraph.Tensor<UseFloatingPoint>]()
  let startTime = Date()
  DynamicGraph.setProfiler(true)
  // Now do PLMS sampling.
  for i in 0..<model.steps {
    let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
    let t = graph.variable(Tensor<UseFloatingPoint>(from: ts[i]))
    let tNext = Tensor<UseFloatingPoint>(from: ts[min(i + 1, ts.count - 1)])
    xIn[0..<1, 0..<startHeight, 0..<startWidth, 0..<4] = x
    xIn[1..<2, 0..<startHeight, 0..<startWidth, 0..<4] = x
    var et = unet(inputs: xIn, t, c)[0].as(of: UseFloatingPoint.self)
    var etUncond = graph.variable(
      .GPU(0), .NHWC(1, startHeight, startWidth, 4), of: UseFloatingPoint.self)
    var etCond = graph.variable(
      .GPU(0), .NHWC(1, startHeight, startWidth, 4), of: UseFloatingPoint.self)
    etUncond[0..<1, 0..<startHeight, 0..<startWidth, 0..<4] =
      et[0..<1, 0..<startHeight, 0..<startWidth, 0..<4]
    etCond[0..<1, 0..<startHeight, 0..<startWidth, 0..<4] =
      et[1..<2, 0..<startHeight, 0..<startWidth, 0..<4]
    et = etUncond + unconditionalGuidanceScale * (etCond - etUncond)
    let alpha = alphasCumprod[timestep]
    let alphaPrev = alphasCumprod[max(timestep - model.timesteps / model.steps, 0)]
    let etPrime: DynamicGraph.Tensor<UseFloatingPoint>
    switch oldEps.count {
    case 0:
      let (xPrev, _) = xPrevAndPredX0(x: x, et: et, alpha: alpha, alphaPrev: alphaPrev)
      // Compute etNext.
      xIn[0..<1, 0..<startHeight, 0..<startWidth, 0..<4] = xPrev
      xIn[1..<2, 0..<startHeight, 0..<startWidth, 0..<4] = xPrev
      var etNext = unet(inputs: xIn, graph.variable(tNext), c)[0].as(of: UseFloatingPoint.self)
      var etNextUncond = graph.variable(
        .GPU(0), .NHWC(1, startHeight, startWidth, 4), of: UseFloatingPoint.self)
      var etNextCond = graph.variable(
        .GPU(0), .NHWC(1, startHeight, startWidth, 4), of: UseFloatingPoint.self)
      etNextUncond[0..<1, 0..<startHeight, 0..<startWidth, 0..<4] =
        etNext[0..<1, 0..<startHeight, 0..<startWidth, 0..<4]
      etNextCond[0..<1, 0..<startHeight, 0..<startWidth, 0..<4] =
        etNext[1..<2, 0..<startHeight, 0..<startWidth, 0..<4]
      etNext = etNextUncond + unconditionalGuidanceScale * (etNextCond - etNextUncond)
      etPrime = 0.5 * (et + etNext)
    case 1:
      etPrime = 0.5 * (3 * et - oldEps[0])
    case 2:
      etPrime =
        Float(1) / Float(12) * (Float(23) * et - Float(16) * oldEps[1] + Float(5) * oldEps[0])
    case 3:
      etPrime =
        Float(1) / Float(24)
        * (Float(55) * et - Float(59) * oldEps[2] + Float(37) * oldEps[1] - Float(9) * oldEps[0])
    default:
      fatalError()
    }
    let (xPrev, _) = xPrevAndPredX0(x: x, et: etPrime, alpha: alpha, alphaPrev: alphaPrev)
    x = xPrev
    oldEps.append(et)
    if oldEps.count > 3 {
      oldEps.removeFirst()
    }
  }
  let z = 1.0 / scaleFactor * x
  let img = DynamicGraph.Tensor<Float>(from: decoder(inputs: z)[0].as(of: UseFloatingPoint.self))
    .toCPU()
  print("Total time \(Date().timeIntervalSince(startTime))")
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: startWidth * 8 * startHeight * 8)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let (r, g, b) = (img[0, y, x, 0], img[0, y, x, 1], img[0, y, x, 2])
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
