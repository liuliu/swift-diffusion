import Diffusion
import Foundation
import NNC
import PNG
import SentencePiece

typealias FloatType = Float
struct PythonObject {}

func ResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  out = Swish()(out)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = norm2(out)
  out = Swish()(out)
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = nin(x) + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x], [out]))
}

func AttnBlock(prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let hw = width * height
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let k = tokeys(out).reshaped([batchSize, inChannels, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    batchSize, inChannels, hw,
  ])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let v = tovalues(out).reshaped([batchSize, inChannels, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = x + projOut(out.reshaped([batchSize, inChannels, height, width]))
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x], [out]))
}

func Encoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var readers = [(PythonObject) -> Void]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (reader, block) = ResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", outChannels: channel,
        shortcut: previousChannel != channel)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      // Conv always pad left first, then right, and pad top first then bottom.
      // Thus, we cannot have (0, 1, 0, 1) (left 0, right 1, top 0, bottom 1) padding as in
      // Stable Diffusion. Instead, we pad to (2, 1, 2, 1) and simply discard the first row and first column.
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])))
      out = conv2d(out).reshaped(
        [batchSize, channel, height, width], offset: [0, 0, 1, 1],
        strides: [channel * (height + 1) * (width + 1), (height + 1) * (width + 1), width + 1, 1])
      let downLayer = i
      let reader: (PythonObject) -> Void = { state_dict in
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "encoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x], [out]))
}

func Decoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "decoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "decoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  var readers = [(PythonObject) -> Void]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", outChannels: channel,
        shortcut: previousChannel != channel)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
      let upLayer = i
      let reader: (PythonObject) -> Void = { state_dict in
      }
      readers.append(reader)
    }
  }
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = Swish()(out)
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x], [out]))
}

let graph = DynamicGraph()

func CLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    FloatType.self, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 {
      penultimate = out
    }
    let encoderLayer =
      CLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [penultimate!, out])
}

func OpenCLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    FloatType.self, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 {
      penultimate = out
    }
    let encoderLayer =
      OpenCLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [penultimate!, out])
}

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
  var out = wi_1(x) .* wi_0(x).GELU(approximate: .tanh)
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  out = wo(out)
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
  var out = x + attention(norm1(x), positionBias)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (wi_0, wi_1, wo, ff) = T5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize)
  out = out + ff(norm2(out))
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, positionBias], [out]))
}

func T5ForConditionalGeneration(b: Int, t: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let relativePositionBuckets = Input()
  let textEmbed = T5TextEmbedding(vocabularySize: 32_128, embeddingSize: 4_096, name: "shared")
  var out = textEmbed(x)
  let relativePositionEmbedding = Embedding(
    FloatType.self, vocabularySize: 32, embeddingSize: 64, name: "relative_position_embedding")
  let positionBias = relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, 64])
    .permuted(0, 3, 1, 2)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = T5Block(
      prefix: "encoder.block.\(i)", k: 64, h: 64, b: b, t: t, outFeatures: 4_096,
      intermediateSize: 10_240)
    out = block(out, positionBias)
    readers.append(reader)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, relativePositionBuckets], [out]))
}

let tokenizer0 = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let tokenizer1 = CLIPTokenizer(
  vocabulary: "examples/open_clip/vocab_16e6.json",
  merges: "examples/open_clip/bpe_simple_vocab_16e6.txt")

let prompt =
  "photo of a young woman with long, wavy brown hair lying down in grass, top down shot, summer, warm, laughing, joy, fun"
let negativePrompt = ""

let tokens0 = tokenizer0.tokenize(text: prompt, truncation: true, maxLength: 77)
let tokens1 = tokenizer1.tokenize(text: prompt, truncation: true, maxLength: 77, paddingToken: 0)
let unconditionalTokens0 = tokenizer0.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77)
let unconditionalTokens1 = tokenizer1.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77, paddingToken: 0)
let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/swift-diffusion/examples/sd3/spiece.model")
var tokens2 = sentencePiece.encode(prompt).map { return $0.id }
tokens2.append(1)

let tokensTensor0 = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensTensor1 = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensTensor2 = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
for i in 0..<77 {
  tokensTensor0[i] = tokens0[i]
  tokensTensor1[i] = tokens1[i]
  tokensTensor2[i] = i < tokens2.count ? tokens2[i] : 0
  positionTensor[i] = Int32(i)
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

let casualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
  }
}

let (c0, c0Pooled) = graph.withNoGrad {
  let tokensTensorGPU = tokensTensor0.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  let textModel0 = CLIPTextModel(
    vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
    batchSize: 1, intermediateSize: 3072)
  textModel0.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/clip_vit_l14_f32.ckpt") {
    $0.read("text_model", model: textModel0)
  }
  let c = textModel0(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU).map {
    $0.as(of: FloatType.self)
  }
  var pooled = graph.variable(.GPU(0), .NC(1, 768), of: FloatType.self)
  let c1 = c[0].reshaped(.CHW(1, 77, 768))
  for (i, token) in tokens0.enumerated() {
    if token == tokenizer0.endToken {
      pooled[0..<1, 0..<768] = c[1][i..<(i + 1), 0..<768]
      break
    }
  }
  return (c1, pooled)
}

let (c1, c1Pooled) = graph.withNoGrad {
  let tokensTensorGPU = tokensTensor1.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  let textModel1 = OpenCLIPTextModel(
    vocabularySize: 49408, maxLength: 77, embeddingSize: 1280, numLayers: 32, numHeads: 20,
    batchSize: 1, intermediateSize: 5120)
  let textProjection = graph.variable(.GPU(0), .NC(1280, 1280), of: FloatType.self)
  textModel1.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_vit_bigg14_f16.ckpt") {
    $0.read("text_model", model: textModel1)
    $0.read("text_projection", variable: textProjection)
  }
  let c = textModel1(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU).map {
    $0.as(of: FloatType.self)
  }
  var pooled = graph.variable(.GPU(0), .NC(1, 1280), of: FloatType.self)
  let c1 = c[0].reshaped(.CHW(1, 77, 1280))
  for (i, token) in tokens1.enumerated() {
    if token == tokenizer1.endToken {
      pooled[0..<1, 0..<1280] = c[1][i..<(i + 1), 0..<1280] * textProjection
      break
    }
  }
  return (c1, pooled)
}

let c2 = graph.withNoGrad {
  let (_, textModel) = T5ForConditionalGeneration(b: 1, t: 77)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: 77, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor2.toGPU(0)
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, relativePositionBucketsGPU)
  graph.openStore("/home/liu/workspace/swift-llm/t5_xxl_encoder_f32.ckpt") {
    $0.read("text_model", model: textModel)
  }
  let output = textModel(inputs: tokensTensorGPU, relativePositionBucketsGPU)[0].as(of: Float.self)
  return output
}

let (c, pooled) = graph.withNoGrad {
  var pooled = graph.variable(.GPU(0), .NC(1, 2048), of: FloatType.self)
  pooled[0..<1, 0..<768] = c0Pooled
  pooled[0..<1, 768..<2048] = c1Pooled
  var c = graph.variable(.GPU(0), .CHW(1, 154, 4096), of: FloatType.self)
  c.full(0)
  c[0..<1, 0..<77, 0..<768] = c0
  c[0..<1, 0..<77, 768..<2048] = c1
  c[0..<1, 77..<154, 0..<4096] = c2
  return (c, pooled)
}

func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func VectorEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "y_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "y_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLP(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  let contextK = contextToKeys(contextOut)
  let contextQ = contextToQueries(contextOut)
  let contextV = contextToValues(contextOut)
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  let xK = xToKeys(xOut)
  let xQ = xToQueries(xOut)
  let xV = xToValues(xOut)
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  keys = keys.reshaped([b, t + hw, h, k]).permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
    .permuted(0, 2, 1, 3)
  values = values.reshaped([b, t + hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = Dense(count: k * h)
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h)
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = x + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextFc1: Model?
  let contextFc2: Model?
  if !contextBlockPreOnly {
    let contextMlp: Model
    (contextFc1, contextFc2, contextMlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4)
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut = contextOut + contextChunks[5]
      .* contextMlp(contextNorm2(contextOut) .* (1 + contextChunks[4]) + contextChunks[3])
  } else {
    contextFc1 = nil
    contextFc2 = nil
  }
  let (xFc1, xFc2, xMlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut = xOut + xChunks[5] .* xMlp(xNorm2(xOut) .* (1 + xChunks[4]) + xChunks[3])
  let reader: (PythonObject) -> Void = { state_dict in
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, c], [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, c], [xOut]))
  }
}

func MMDiT(b: Int, h: Int, w: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t = Input()
  let y = Input()
  let contextIn = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 1536, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 1536, h * w]).transposed(1, 2)
  let posEmbed = Parameter<FloatType>(.GPU(0), .NHWC(1, 192, 192, 1536))
  let spatialPosEmbed = posEmbed.reshaped(
    [1, h, w, 1536], offset: [0, (192 - h) / 2, (192 - w) / 2, 0],
    strides: [192 * 192 * 1536, 192 * 1536, 1536, 1]
  ).contiguous().reshaped([1, h * w, 1536])
  out = spatialPosEmbed + out
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 1536)
  let (yMlp0, yMlp2, yEmbedder) = VectorEmbedder(channels: 1536)
  let c = (tEmbedder(t) + yEmbedder(y)).reshaped([b, 1, 1536]).swish()
  let contextEmbedder = Dense(count: 1536, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = JointTransformerBlock(
      prefix: "diffusion_model.joint_blocks.\(i)", k: 64, h: 24, b: b, t: 154, hw: h * w,
      contextBlockPreOnly: i == 23)
    let blockOut = block(context, out, c)
    if i == 23 {
      out = blockOut
    } else {
      context = blockOut[0]
      out = blockOut[1]
    }
    readers.append(reader)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shift = Dense(count: 1536, name: "ada_ln_0")
  let scale = Dense(count: 1536, name: "ada_ln_1")
  out = (1 + scale(c)) .* normFinal(out) + shift(c)
  let linear = Dense(count: 2 * 2 * 16, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 16]).permuted(0, 5, 1, 3, 2, 4).contiguous().reshaped([
    b, 16, h * 2, w * 2,
  ])
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, t, contextIn, y], [out]))
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

let (reader, dit) = MMDiT(b: 2, h: 64, w: 64)

DynamicGraph.setSeed(42)

graph.withNoGrad {
  var z = graph.variable(.GPU(0), .NCHW(1, 16, 128, 128), of: FloatType.self)
  z.randn()
  var input = graph.variable(.GPU(0), .NCHW(2, 16, 128, 128), of: FloatType.self)
  let tTensor = graph.variable(
    timeEmbedding(timesteps: 1000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000).toGPU(0))
  var cTensor = graph.variable(.GPU(0), .CHW(2, 154, 4096), of: FloatType.self)
  cTensor.full(0)
  cTensor[1..<2, 0..<154, 0..<4096] = c
  var yTensor = graph.variable(.GPU(0), .NC(2, 2048), of: FloatType.self)
  yTensor.full(0)
  yTensor[1..<2, 0..<2048] = pooled
  dit.compile(inputs: input, tTensor, cTensor, yTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/sd3_medium_f32.ckpt") {
    $0.read("dit", model: dit)
  }
  let samplingSteps = 30
  for i in (1...samplingSteps).reversed() {
    print("\(i)")
    let t = Float(i) / Float(samplingSteps) * 1_000
    input[0..<1, 0..<16, 0..<128, 0..<128] = z
    input[1..<2, 0..<16, 0..<128, 0..<128] = z
    let tTensor = graph.variable(
      timeEmbedding(timesteps: t, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000).toGPU(0))
    // cfg = 4.5
    let vcu = dit(inputs: input, tTensor, cTensor, yTensor)[0].as(of: FloatType.self)
    let vu = vcu[0..<1, 0..<16, 0..<128, 0..<128]
    let vc = vcu[1..<2, 0..<16, 0..<128, 0..<128]
    let v = vu + 4.5 * (vc - vu)
    z = z - (1 / Float(samplingSteps)) * v
    debugPrint(z)
  }
  // Already processed out.
  z = (1.0 / 1.5305) * z + 0.0609
  let (_, decoder) = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  decoder.compile(inputs: z)
  graph.openStore("/home/liu/workspace/swift-diffusion/sd3_vae_f32.ckpt") {
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
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/txt2img.png", level: 4)
}
