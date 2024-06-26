import Diffusion
import Foundation
import NNC
import PNG
import SentencePiece

typealias FloatType = Float
struct PythonObject {}

func T5TextEmbedding(vocabularySize: Int, embeddingSize: Int, name: String) -> Model {
  let tokenEmbed = Embedding(
    FloatType.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: name)
  return tokenEmbed
}

func T5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let positionBias = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  // No scaling the queries.
  let queries = toqueries(x).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + positionBias
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: outFeatures, noBias: true, name: "o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, positionBias], [out]))
}

func T5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true, name: "w0")
  let wi_1 = Dense(count: intermediateSize, noBias: true, name: "w1")
  var out = wi_1(x).to(.Float32) .* wi_0(x).GELU(approximate: .tanh).to(.Float32)
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  // Need to apply a scale factor if T5 has to work with Float16.
  let scaleFactor: Float = 8
  out = scaleFactor * wo(((1 / scaleFactor) * out).to(of: x)).to(.Float32)
  return (wi_0, wi_1, wo, Model([x], [out]))
}

func T5Block(
  prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let positionBias = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (tokeys, toqueries, tovalues, unifyheads, attention) = T5LayerSelfAttention(
    k: k, h: h, b: b, t: t, outFeatures: outFeatures)
  var out = x + attention(norm1(x).to(FloatType.dataType), positionBias).to(of: x)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm2")
  let (wi_0, wi_1, wo, ff) = T5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize)
  out = out + ff(norm2(out).to(FloatType.dataType))
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, positionBias], [out]))
}

func T5ForConditionalGeneration(b: Int, t: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let relativePositionBuckets = Input()
  let textEmbed = T5TextEmbedding(vocabularySize: 32_128, embeddingSize: 4_096, name: "shared")
  var out = textEmbed(x).to(.Float32)
  let relativePositionEmbedding = Embedding(
    FloatType.self, vocabularySize: 32, embeddingSize: 64, name: "relative_position_embedding")
  let positionBias = relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, 64])
    .permuted(0, 3, 1, 2).contiguous()
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = T5Block(
      prefix: "encoder.block.\(i)", k: 64, h: 64, b: b, t: t, outFeatures: 4_096,
      intermediateSize: 10_240)
    out = block(out, positionBias)
    readers.append(reader)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out).to(FloatType.dataType)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, relativePositionBuckets], [out]))
}

func sinCos2DPositionEmbedding(height: Int, width: Int, embeddingSize: Int) -> Tensor<Float> {
  precondition(embeddingSize % 4 == 0)
  var embedding = Tensor<Float>(.CPU, .HWC(height, width, embeddingSize))
  let halfOfHalf = embeddingSize / 4
  let omega: [Double] = (0..<halfOfHalf).map {
    pow(Double(1.0 / 10000), Double($0) / Double(halfOfHalf))
  }
  for i in 0..<height {
    let y = Double(i) / 2
    for j in 0..<width {
      let x = Double(j) / 2
      for k in 0..<halfOfHalf {
        let xFreq = x * omega[k]
        embedding[i, j, k] = Float(sin(xFreq))
        embedding[i, j, k + halfOfHalf] = Float(cos(xFreq))
        let yFreq = y * omega[k]
        embedding[i, j, k + 2 * halfOfHalf] = Float(sin(yFreq))
        embedding[i, j, k + 3 * halfOfHalf] = Float(cos(yFreq))
      }
    }
  }
  return embedding
}

func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLP(hiddenSize: Int, intermediateSize: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize, name: "\(name)_fc1")
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = Dense(count: hiddenSize, name: "\(name)_fc2")
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func SelfAttention(k: Int, h: Int, b: Int, t: Int) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let tokeys = Dense(count: k * h, name: "k")
  let toqueries = Dense(count: k * h, name: "q")
  let tovalues = Dense(count: k * h, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
  // No scaling the queries.
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .transposed(1, 2)
  let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
  let unifyheads = Dense(count: k * h, name: "o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func CrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model, Model, Model)
{
  let x = Input()
  let context = Input()
  let tokeys = Dense(count: k * h, name: "c_k")
  let toqueries = Dense(count: k * h, name: "c_q")
  let tovalues = Dense(count: k * h, name: "c_v")
  let keys = tokeys(context).reshaped([b, t, h, k]).transposed(1, 2)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2)
  let values = tovalues(context).reshaped([b, t, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h, name: "c_o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, context], [out]))
}

func PixArtMSBlock(prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let context = Input()
  let shiftMsa = Input()
  let scaleMsa = Input()
  let gateMsa = Input()
  let shiftMlp = Input()
  let scaleMlp = Input()
  let gateMlp = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn) = SelfAttention(k: k, h: h, b: b, t: hw)
  let shiftMsaShift = Parameter<FloatType>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_0")
  let scaleMsaShift = Parameter<FloatType>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_1")
  let gateMsaShift = Parameter<FloatType>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_2")
  var out =
    x + (gateMsa + gateMsaShift)
    .* attn(norm1(x) .* (scaleMsa + scaleMsaShift) + (shiftMsa + shiftMsaShift))
  let (tokeys2, toqueries2, tovalues2, unifyheads2, crossAttn) = CrossAttention(
    k: k, h: h, b: b, hw: hw, t: t)
  out = out + crossAttn(out, context)
  let (fc1, fc2, mlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4, name: "mlp")
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shiftMlpShift = Parameter<FloatType>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_3")
  let scaleMlpShift = Parameter<FloatType>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_4")
  let gateMlpShift = Parameter<FloatType>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_5")
  out = out + (gateMlp + gateMlpShift)
    .* mlp(norm2(out) .* (scaleMlp + scaleMlpShift) + (shiftMlp + shiftMlpShift))
  let reader: (PythonObject) -> Void = { _ in
  }
  return (
    reader, Model([x, context, shiftMsa, scaleMsa, gateMsa, shiftMlp, scaleMlp, gateMlp], [out])
  )
}

func PixArt(b: Int, h: Int, w: Int, t: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let posEmbed = Input()
  let ts = Input()
  let y = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 1152, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 1152, h * w]).transposed(1, 2) + posEmbed
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 1152)
  let t0 = tEmbedder(ts)
  let t1 = t0.swish().reshaped([b, 1, 1152])
  let tBlock = (0..<6).map { Dense(count: 1152, name: "t_block_\($0)") }
  var adaln = tBlock.map { $0(t1) }
  adaln[1] = 1 + adaln[1]
  adaln[4] = 1 + adaln[4]
  let (fc1, fc2, yEmbedder) = MLP(hiddenSize: 1152, intermediateSize: 1152, name: "y_embedder")
  let y0 = yEmbedder(y)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<28 {
    let (reader, block) = PixArtMSBlock(
      prefix: "blocks.\(i)", k: 72, h: 16, b: 2, hw: h * w, t: t)
    out = block(out, y0, adaln[0], adaln[1], adaln[2], adaln[3], adaln[4], adaln[5])
    readers.append(reader)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shiftShift = Parameter<FloatType>(
    .GPU(0), .CHW(1, 1, 1152), name: "final_scale_shift_table_0")
  let scaleShift = Parameter<FloatType>(
    .GPU(0), .CHW(1, 1, 1152), name: "final_scale_shift_table_1")
  let tt = t0.reshaped([1, 1, 1152])  // PixArt uses chunk, but that always assumes t0 is the same, which is true.
  out = (scaleShift + 1 + tt) .* normFinal(out) + (shiftShift + tt)
  let linear = Dense(count: 2 * 2 * 8, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 8]).permuted(0, 5, 1, 3, 2, 4).contiguous().reshaped([
    b, 8, h * 2, w * 2,
  ])
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x, posEmbed, ts, y], [out]))
}

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
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

let graph = DynamicGraph()

let prompt =
  "A miniature tooth fairy woman is holding a pick axe and mining diamonds in a bedroom at night. The fairy has an angry expression."
let negativePrompt = ""

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/swift-diffusion/examples/pixart/spiece.model")
var tokens2 = sentencePiece.encode(prompt).map { return $0.id }
tokens2.append(1)

var tokensTensor2 = graph.variable(.CPU, .C(tokens2.count), of: Int32.self)
for i in 0..<tokens2.count {
  tokensTensor2[i] = tokens2[i]
}

func relativePositionBuckets(sequenceLength: Int, numBuckets: Int, maxDistance: Int) -> Tensor<
  Int32
> {
  // isBidirectional = true.
  let numBuckets = numBuckets / 2
  let maxExact = numBuckets / 2
  var relativePositionBuckets = Tensor<Int32>(.CPU, .C(sequenceLength * sequenceLength))
  for i in 0..<sequenceLength {
    for j in 0..<sequenceLength {
      var relativePositionBucket = j > i ? numBuckets : 0
      let relativePosition = abs(i - j)
      let isSmall = relativePosition < maxExact
      if isSmall {
        relativePositionBucket += relativePosition
      } else {
        let relativePositionIfLarge = min(
          numBuckets - 1,
          maxExact
            + Int(
              (log(Double(relativePosition) / Double(maxExact))
                / log(Double(maxDistance) / Double(maxExact)) * Double(numBuckets - maxExact))
                .rounded(.down)))
        relativePositionBucket += relativePositionIfLarge
      }
      relativePositionBuckets[i * sequenceLength + j] = Int32(relativePositionBucket)
    }
  }
  return relativePositionBuckets
}

let c = graph.withNoGrad {
  let (_, textModel) = T5ForConditionalGeneration(b: 1, t: tokens2.count)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: tokens2.count, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor2.toGPU(0)
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, relativePositionBucketsGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/t5_xxl_encoder_q6p.ckpt") {
    $0.read("text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7])
  }
  let output = textModel(inputs: tokensTensorGPU, relativePositionBucketsGPU)[0].as(
    of: FloatType.self)
  return output
}

public struct DiffusionModel {
  public var linearStart: Float
  public var linearEnd: Float
  public var timesteps: Int
  public var steps: Int
}

extension DiffusionModel {
  public var betas: [Float] {  // Linear for now.
    var betas = [Float]()
    let start = linearStart
    let length = linearEnd - start
    for i in 0..<timesteps {
      let beta = start + Float(i) * length / Float(timesteps - 1)
      betas.append(beta)
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

  public func fixedStepSigmas(sigmas sigmasForTimesteps: [Float])
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

DynamicGraph.setSeed(120)

let unconditionalGuidanceScale: Float = 5
let scaleFactor: Float = 0.13025
let startHeight = 128
let startWidth = 128
let model = DiffusionModel(linearStart: 0.0001, linearEnd: 0.02, timesteps: 1_000, steps: 30)
let alphasCumprod = model.alphasCumprod
let sigmasForTimesteps = DiffusionModel.sigmas(from: alphasCumprod)
// This is for Karras scheduler (used in DPM++ 2M Karras)
let sigmas = model.fixedStepSigmas(sigmas: sigmasForTimesteps)

let (reader, dit) = PixArt(b: 2, h: 64, w: 64, t: tokens2.count)

let z = graph.withNoGrad {
  var cTensor = graph.variable(.GPU(0), .CHW(2, tokens2.count, 4096), of: FloatType.self)
  cTensor.full(0)
  cTensor[1..<2, 0..<tokens2.count, 0..<4096] = c
  let x_T = graph.variable(.GPU(0), .NCHW(1, 4, 128, 128), of: FloatType.self)
  x_T.randn(std: 1, mean: 0)
  var x = x_T
  var xIn = graph.variable(.GPU(0), .NCHW(2, 4, 128, 128), of: FloatType.self)
  let tTensor = graph.variable(
    timeEmbedding(timesteps: 666, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000).toGPU(0))
  let posEmbedTensor = graph.variable(
    sinCos2DPositionEmbedding(height: 64, width: 64, embeddingSize: 1152).toGPU(0)
  ).reshaped(.CHW(1, 4096, 1152))
  dit.compile(inputs: xIn, posEmbedTensor, tTensor, cTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/pixart_sigma_xl_2_1024_ms_f32.ckpt") {
    $0.read("dit", model: dit)
  }
  var oldDenoised: DynamicGraph.Tensor<FloatType>? = nil
  // Now do DPM++ 2M Karras sampling. (DPM++ 2S a Karras requires two denoising per step, not ideal for my use case).
  x = sigmas[0] * x
  for i in 0..<model.steps {
    let sigma = sigmas[i]
    let timestep = DiffusionModel.timestep(from: sigma, sigmas: sigmasForTimesteps)
    let ts = timeEmbedding(
      timestep: timestep, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000
    ).toGPU(0)
    let tTensor = graph.variable(Tensor<FloatType>(from: ts))
    let cIn = 1.0 / (sigma * sigma + 1).squareRoot()
    let cOut = -sigma
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = cIn * x
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = cIn * x
    var et = dit(inputs: xIn, posEmbedTensor, tTensor, cTensor)[0].as(of: FloatType.self)
    var etUncond = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
    var etCond = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
    etUncond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
    etCond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[1..<2, 0..<4, 0..<startHeight, 0..<startWidth]
    et = etUncond + unconditionalGuidanceScale * (etCond - etUncond)
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
    debugPrint(x)
  }
  return 1.0 / scaleFactor * x
}

graph.withNoGrad {
  let decoder = ModelBuilder {
    let startWidth = $0[0].shape[3]
    let startHeight = $0[0].shape[2]
    return Decoder(
      channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
      startHeight: startHeight)
  }
  let z32 = DynamicGraph.Tensor<Float>(from: z)
  decoder.compile(inputs: z32)
  graph.openStore("/home/liu/workspace/swift-diffusion/sdxl_vae_v1.0_f16.ckpt") {
    $0.read("decoder", model: decoder)
  }
  let img = decoder(inputs: z32)[0].as(of: Float.self)
    .toCPU()
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
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/txt2img.png", level: 4)
}
