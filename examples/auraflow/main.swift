import Diffusion
import Foundation
import NNC
import PNG
import SentencePiece

typealias FloatType = Float16
struct PythonObject {}

func UMT5TextEmbedding(vocabularySize: Int, embeddingSize: Int, name: String) -> Model {
  let tokenEmbed = Embedding(
    FloatType.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: name)
  return tokenEmbed
}

func UMT5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int) -> (
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

func UMT5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model)
{
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true, name: "w0")
  let wi_1 = Dense(count: intermediateSize, noBias: true, name: "w1")
  var out = wi_1(x).to(.Float32) .* wi_0(x).GELU(approximate: .tanh).to(.Float32)
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  let scaleFactor: Float = 8
  out = scaleFactor * wo(((1 / scaleFactor) * out).to(of: x)).to(.Float32)
  return (wi_0, wi_1, wo, Model([x], [out]))
}

func UMT5Block(
  prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let relativePositionEmbedding = Embedding(
    FloatType.self, vocabularySize: 32, embeddingSize: 32, name: "relative_position_embedding")
  let positionBias =
    relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, 32])
    .permuted(0, 3, 1, 2) + attentionMask
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (tokeys, toqueries, tovalues, unifyheads, attention) = UMT5LayerSelfAttention(
    k: k, h: h, b: b, t: t, outFeatures: outFeatures)
  var out = x + attention(norm1(x).to(FloatType.dataType), positionBias).to(of: x)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (wi_0, wi_1, wo, ff) = UMT5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize)
  out = out + ff(norm2(out).to(FloatType.dataType))
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, attentionMask, relativePositionBuckets], [out]))
}

func UMT5ForConditionalGeneration(b: Int, t: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let textEmbed = UMT5TextEmbedding(vocabularySize: 32_128, embeddingSize: 2_048, name: "shared")
  var out = textEmbed(x).to(.Float32)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = UMT5Block(
      prefix: "encoder.block.\(i)", k: 64, h: 32, b: b, t: t, outFeatures: 2_048,
      intermediateSize: 5_120)
    out = block(out, attentionMask, relativePositionBuckets)
    readers.append(reader)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out).to(FloatType.dataType)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, attentionMask, relativePositionBuckets], [out]))
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
// debugPrint(timeEmbedding(timesteps: 1_000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000))

func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_linear1")
  let linear2 = Dense(count: intermediateSize, noBias: true, name: "\(name)_linear2")
  var out = linear1(x).swish() .* linear2(x)
  let outProjection = Dense(count: hiddenSize, noBias: true, name: "\(name)_out_proj")
  out = outProjection(out)
  return (linear1, linear2, outProjection, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, noBias: true, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, noBias: true, name: "c_k")
  let contextToQueries = Dense(count: k * h, noBias: true, name: "c_q")
  let contextToValues = Dense(count: k * h, noBias: true, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, noBias: true, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  keys = keys.permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .permuted(0, 2, 1, 3)
  values = values.permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = Dense(count: k * h, noBias: true, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = x + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextLinear2: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextLinear2, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
    contextOut = context + contextChunks[5]
      .* contextFF(contextNorm2(contextOut) .* (1 + contextChunks[4]) + contextChunks[3])
  } else {
    contextLinear1 = nil
    contextLinear2 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xLinear2, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  xOut = x + xChunks[5] .* xFF(xNorm2(xOut) .* (1 + xChunks[4]) + xChunks[3])
  let reader: (PythonObject) -> Void = { state_dict in
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, c], [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, c], [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let c = Input()
  let xAdaLNs = (0..<6).map { Dense(count: k * h, noBias: true, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  let xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  // Now run attention.
  keys = keys.permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .permuted(0, 2, 1, 3)
  values = values.permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  out = xUnifyheads(out)
  out = xIn + xChunks[2] .* out
  // Attentions are now. Now run MLP.
  let (xLinear1, xLinear2, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  out = xIn + xChunks[5] .* xFF(xNorm2(out) .* (1 + xChunks[4]) + xChunks[3])
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, c], [out]))
}

func MMDiT(b: Int, h: Int, w: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t = Input()
  let contextIn = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 3072, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 3072, h * w]).transposed(1, 2)
  let posEmbed = Parameter<FloatType>(.GPU(0), .NHWC(1, 64, 64, 3072), name: "pos_embed")
  let spatialPosEmbed = posEmbed.reshaped(
    [1, h, w, 3072], offset: [0, (64 - h) / 2, (64 - w) / 2, 0],
    strides: [64 * 64 * 3072, 64 * 3072, 3072, 1]
  ).contiguous().reshaped([1, h * w, 3072])
  out = spatialPosEmbed + out
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 3072)
  let c = tEmbedder(t).reshaped([b, 1, 3072]).swish()
  let contextEmbedder = Dense(count: 3072, noBias: true, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  let registerTokens = Parameter<FloatType>(.GPU(0), .HWC(1, 8, 3072), name: "register_tokens")
  context = Functional.concat(
    axis: 1, Concat(axis: 0)(Array(repeating: registerTokens, count: b)), context)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<4 {
    let (reader, block) = JointTransformerBlock(
      prefix: "joint_transformer_blocks.\(i)", k: 256, h: 12, b: b, t: 264, hw: h * w,
      contextBlockPreOnly: false)
    let blockOut = block(context, out, c)
    context = blockOut[0]
    out = blockOut[1]
    readers.append(reader)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<32 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_transformer_blocks.\(i)", k: 256, h: 12, b: b, t: 264, hw: h * w,
      contextBlockPreOnly: i == 31)
    out = block(out, c)
    readers.append(reader)
  }
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  out = (1 + scale(c)) .* out + shift(c)
  let projOut = Dense(count: 2 * 2 * 4, name: "linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 4]).permuted(0, 5, 1, 3, 2, 4).contiguous().reshaped([
    b, 4, h * 2, w * 2,
  ])
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, t, contextIn], [out]))
}

let graph = DynamicGraph()

let prompt =
  // "Professional photograph of an astronaut riding a horse on the moon with view of Earth in the background."
  // "a smiling indian man with a google t-shirt next to a frowning asian man with a shirt saying nexus at a meeting table facing each other, photograph, detailed, 8k"
  // "photo of a young woman with long, wavy brown hair sleeping in grassfield, top down shot, summer, warm, laughing, joy, fun"
  // "35mm analogue full-body portrait of a beautiful woman wearing black sheer dress, catwalking in a busy market, soft colour grading, infinity cove, shadows, kodak, contax t2"
  "A miniature tooth fairy woman is holding a pick axe and mining diamonds in a bedroom at night. The fairy has an angry expression."
let negativePrompt = ""

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/swift-diffusion/tokenizer.model")
var tokens = sentencePiece.encode(prompt).map { return $0.id }
tokens.append(2)
var negativeTokens = sentencePiece.encode(negativePrompt).map { return $0.id }
negativeTokens.append(2)

let tokensTensor = graph.variable(.CPU, .C(256 * 2), of: Int32.self)
for i in 0..<256 {
  tokensTensor[i] = i < negativeTokens.count ? negativeTokens[i] : 1
  tokensTensor[256 + i] = i < tokens.count ? tokens[i] : 1
}

let encoderHiddenStates = graph.withNoGrad {
  let (_, textModel) = UMT5ForConditionalGeneration(b: 2, t: 256)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: 256, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor.toGPU(0)
  var attentionMask = Tensor<FloatType>(.CPU, .NCHW(2, 1, 1, 256))
  for i in 0..<256 {
    attentionMask[0, 0, 0, i] = i < negativeTokens.count ? 0 : -FloatType.greatestFiniteMagnitude
    attentionMask[1, 0, 0, i] = i < tokens.count ? 0 : -FloatType.greatestFiniteMagnitude
  }
  let attentionMaskGPU = graph.variable(attentionMask.toGPU(0))
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/pile_t5_xl_encoder_q8p.ckpt") {
    $0.read("text_model", model: textModel, codec: [.q4p, .q6p, .q8p, .ezm7])
  }
  let output = textModel(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[0]
    .as(of: FloatType.self).reshaped(.CHW(2, 256, 2048))
  var encoderMask = Tensor<FloatType>(.CPU, .CHW(2, 256, 1))
  for i in 0..<256 {
    encoderMask[0, i, 0] = i < negativeTokens.count ? 1 : 0
    encoderMask[1, i, 0] = i < tokens.count ? 1 : 0
  }
  return output .* graph.variable(encoderMask.toGPU(0))
}

debugPrint(encoderHiddenStates)

let (_, dit) = MMDiT(b: 2, h: 64, w: 64)

DynamicGraph.setSeed(42)

graph.withNoGrad {
  var z = graph.variable(.GPU(0), .NCHW(1, 4, 128, 128), of: FloatType.self)
  z.randn()
  var input = graph.variable(.GPU(0), .NCHW(2, 4, 128, 128), of: FloatType.self)
  let tTensor = graph.variable(
    Tensor<FloatType>(
      from: timeEmbedding(timesteps: 1000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000)
        .toGPU(0)))
  dit.compile(inputs: input, tTensor, encoderHiddenStates)
  graph.openStore("/home/liu/workspace/swift-diffusion/auraflow_v0.1_f16.ckpt") {
    $0.read("dit", model: dit, codec: [.q8p, .q6p, .q4p, .ezm7])
  }
  let samplingSteps = 30
  for i in (1...samplingSteps).reversed() {
    print("\(i)")
    let t = Float(i) / Float(samplingSteps) * 1_000
    input[0..<1, 0..<4, 0..<128, 0..<128] = z
    input[1..<2, 0..<4, 0..<128, 0..<128] = z
    let tTensor = graph.variable(
      Tensor<FloatType>(
        from: timeEmbedding(timesteps: t, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000)
          .toGPU(0)))
    // cfg = 4.5
    let vcu = dit(inputs: input, tTensor, encoderHiddenStates)[0].as(of: FloatType.self)
    let vu = vcu[0..<1, 0..<4, 0..<128, 0..<128]
    let vc = vcu[1..<2, 0..<4, 0..<128, 0..<128]
    let v = vu + 4.5 * (vc - vu)
    z = z - (1 / Float(samplingSteps)) * v
    debugPrint(z)
  }
  /*
  let truth = graph.variable(like: z)
  graph.openStore("/home/liu/workspace/auraflow_output.ckpt") {
    // $0.write("fp16_2", variable: z)
    $0.read("fp16_2", variable: truth)
  }
  let error = DynamicGraph.Tensor<Float>(from: z) - DynamicGraph.Tensor<Float>(from: truth)
  let mse = (error .* error).reduced(.mean, axis: [1, 2, 3])
  debugPrint(mse)
  */
  // Already processed out.
  z = (1.0 / 0.13025) * z
  let decoder = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  decoder.compile(inputs: z)
  graph.openStore("/home/liu/workspace/swift-diffusion/sdxl_vae_v1.0_f16.ckpt") {
    $0.read("decoder", model: decoder)
  }
  let img = decoder(inputs: z)[0].as(of: FloatType.self).toCPU()
  let startHeight = 128
  let startWidth = 128
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
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/4_txt2img.png", level: 4)
}
