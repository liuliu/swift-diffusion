import Diffusion
import Foundation
import NNC
import PNG
import SentencePiece

typealias FloatType = Float16
struct PythonObject {}

let graph = DynamicGraph()

DynamicGraph.setSeed(42)

let prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"."

let filename = "hidream_i1_dev"

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k]).transposed(1, 2)
  queries = ((1.0 / Float(k).squareRoot()) * Functional.cmul(left: queries, right: rot)).transposed(
    1, 2)
  keys = Functional.cmul(left: keys, right: rot).transposed(1, 2)
  var outs = [Model.IO]()
  for i in 0..<hk {
    let query = queries.reshaped(
      [b, h / hk, t, k], offset: [0, i * (h / hk), 0, 0], strides: [h * t * k, t * k, k, 1])
    let key = keys.reshaped(
      [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1])
    let value = values.reshaped(
      [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1])
    var dot = Matmul(transposeB: (2, 3))(query, key) + causalAttentionMask
    if let last = outs.last {
      dot.add(dependencies: [last])
    }
    dot = dot.reshaped([b * (h / hk) * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h / hk, t, t])
    let out = dot * value
    outs.append(out)
  }
  var out = Concat(axis: 1)(outs).reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { _ in }
  return (Model([x, rot, causalAttentionMask], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

func TransformerBlock(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot, causalAttentionMask) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
  }
  return (Model([x, rot, causalAttentionMask], [out]), reader)
}

func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { _ in }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Bool, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let (embedding, embedReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  var hiddenStates = [Model.IO]()
  for i in 0..<layers {
    if i > 0 && outputHiddenStates {
      hiddenStates.append(out)
    }
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: heads / 4, b: batchSize,
      t: tokenLength,
      MLP: MLP)
    out = layer(out, rot, causalAttentionMask)
    readers.append(reader)
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([tokens, rot, causalAttentionMask], hiddenStates + [out]), reader)
}

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

let tokenizer0 = CLIPTokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/clip/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/clip/merges.txt")

let tokenizer1 = CLIPTokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/open_clip/vocab_16e6.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/open_clip/bpe_simple_vocab_16e6.txt")

let tokens0 = tokenizer0.tokenize(text: prompt, truncation: true, maxLength: 77)

let tokens1 = tokenizer1.tokenize(text: prompt, truncation: true, maxLength: 77, paddingToken: 0)

let tokenizer2 = SentencePiece(
  file: "/home/liu/workspace/swift-diffusion/examples/sd3/spiece.model")

var tokens2 = tokenizer2.encode(prompt).map { return $0.id }
tokens2.append(1)

let tokenizer3 = GPT2Tokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/hunyuan/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/hunyuan/merges.txt",
  specialTokens: [
    "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
    "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
  ])

let tokens3 = tokenizer3.tokenize(text: prompt, addSpecialTokens: true)

let tokensTensor0 = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensTensor1 = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensTensor2 = graph.variable(.CPU, .C(128), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
for i in 0..<77 {
  tokensTensor0[i] = tokens0[i]
  tokensTensor1[i] = tokens1[i]
  positionTensor[i] = Int32(i)
}
for i in 0..<128 {
  tokensTensor2[i] = i < tokens2.count ? tokens2[i] : 0
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
  graph.openStore("/fast/Data/SD/clip_vit_l14_f32.ckpt") {
    try! $0.read("text_model", model: textModel0, strict: true)
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
  graph.openStore("/fast/Data/SD/open_clip_vit_bigg14_f16.ckpt") {
    try! $0.read("text_model", model: textModel1, strict: true)
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

let pooledPromptEmbedTensor = graph.withNoGrad {
  var pooled = graph.variable(.GPU(0), .NC(1, 2048), of: FloatType.self)
  pooled.full(0)
  // pooled[0..<1, 0..<768] = c0Pooled
  // pooled[0..<1, 768..<2048] = c1Pooled
  return pooled
}

let promptEmbed3Tensor = graph.withNoGrad {
  let (_, textModel) = T5ForConditionalGeneration(b: 1, t: 128)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: 128, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor2.toGPU(0)
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, relativePositionBucketsGPU)
  graph.openStore("/fast/Data/SD/t5_xxl_encoder_q6p.ckpt") {
    try! $0.read("text_model", model: textModel, strict: true, codec: [.q8p, .q6p, .q4p, .ezm7])
  }
  let output = textModel(inputs: tokensTensorGPU, relativePositionBucketsGPU)[0].as(
    of: FloatType.self)
  return output.reshaped(.HWC(1, 128, 4096))
}

let promptEmbed4Tensors = graph.withNoGrad {
  let (transformer, reader) = Transformer(
    Float16.self, vocabularySize: 128_256, maxLength: 128, width: 4_096, tokenLength: 128,
    layers: 32, MLP: 14336, heads: 32, outputHiddenStates: true, batchSize: 1)
  let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [128], of: Int32.self)
  for i in 0..<tokens3.count {
    tokensTensor[i] = tokens3[i]
  }
  for i in tokens3.count..<128 {
    tokensTensor[i] = 128009
  }
  let rotTensor = graph.variable(.CPU, .NHWC(1, 128, 1, 128), of: Float.self)
  let invFreqLlama = (0..<64).map { k in
    let lowFreqWavelen = Double(8_192) / 1.0
    let highFreqWavelen = Double(8_192) / 4.0
    let invFreq = 1.0 / pow(500_000, Double(k) * 2 / 128)
    let wavelen = 2.0 * .pi / invFreq
    var invFreqLlama: Double
    if wavelen > lowFreqWavelen {
      invFreqLlama = invFreq / 8.0
    } else {
      invFreqLlama = invFreq
    }
    let smoothFactor = (Double(8_192) / wavelen - 1.0) / (4.0 - 1.0)
    let smoothInvFreq = (1 - smoothFactor) * invFreqLlama / 8.0 + smoothFactor * invFreqLlama
    if wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen {
      invFreqLlama = smoothInvFreq
    }
    return invFreqLlama
  }
  for i in 0..<128 {
    for k in 0..<64 {
      let theta = Double(i) * invFreqLlama[k]
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let causalAttentionMask = graph.variable(Tensor<Float16>(.CPU, .NHWC(1, 1, 128, 128)))
  causalAttentionMask.full(0)
  for i in 0..<128 {
    for j in min(i + 1, tokens3.count)..<128 {
      causalAttentionMask[0, 0, i, j] = -Float16.greatestFiniteMagnitude
    }
  }
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
  let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
  transformer.compile(inputs: tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/llama_3.1_8b_instruct_f16.ckpt") {
    $0.read("text_model", model: transformer)
  }
  let outputHiddenStates = transformer(
    inputs: tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU)
  return outputHiddenStates.map { $0.as(of: Float16.self).reshaped(.HWC(1, 128, 4096)) }
}

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
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

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w1")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w3")
  var out = w1(x).swish() .* w3(x)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 4
    out = (1 / scaleFactor) * out
  }
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out)
  if upcast {
    let scaleFactor: Float = 4
    out = out.to(.Float32) * scaleFactor
  } else {
    out = out.to(.Float32)
  }
  return (w1, w2, w3, Model([x], [out]))
}

func MoEFeedForward(
  segments: Int, tokenLength: Int, hiddenSize: Int, intermediateSize: Int, upcast: Bool,
  name: String
) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: segments, noBias: true, name: "\(name)_gate")
  let route = gate(x).reshaped([tokenLength, 4]).softmax().partitioned(
    kth: 2, axis: 1, descending: true)
  var weights = route[0].reshaped([tokenLength * 2])
  let experts = route[1].reshaped([tokenLength * 2])  // This is to select into experts.
  let sort = experts.sorted(axis: 0, descending: false)
  let sortIndices = sort[1]
  weights = IndexSelect()(weights, sortIndices)  // Reorder the weights by the sorting order.
  let expertIds = sort[0].uniqueConsecutive(bincount: segments)
  let indices = 0.5 * sortIndices  // Scale it to 0..<tokenLength.
  let gathered = IndexSelect()(x.reshaped([tokenLength, hiddenSize]), indices)
  let w1 = SegmentedDense(
    segments: segments, count: intermediateSize, noBias: true, name: "\(name)_w1")
  let w3 = SegmentedDense(
    segments: segments, count: intermediateSize, noBias: true, name: "\(name)_w3")
  var out = w1(gathered, expertIds).swish() .* w3(gathered, expertIds)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 4
    out = (1 / scaleFactor) * out
  }
  let w2 = SegmentedDense(segments: segments, count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out, expertIds)
  // Out is tokenLength * 2, now multiply weights and scale back.
  out = out .* weights.reshaped([tokenLength * 2, 1])
  out = Functional.scatterAdd(bincount: tokenLength, out, index: indices)
  if upcast {
    let scaleFactor: Float = 4
    out = out.to(.Float32) * scaleFactor
  } else {
    out = out.to(.Float32)
  }
  return (gate, w1, w2, w3, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let rot = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context).to(.Float16) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut)
  let normAddedK = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_k")
  contextK = normAddedK(contextK).reshaped([b, t.1, h, k])
  var contextQ = contextToQueries(contextOut)
  let normAddedQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_q")
  contextQ = normAddedQ(contextQ).reshaped([b, t.1, h, k])
  let contextV = contextToValues(contextOut).reshaped([b, t.1, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  let values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    queries, keys, values
  ).reshaped([b, t.1 + hw, h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t.0, h * k], offset: [0, hw, 0], strides: [(t.1 + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut =
      context.reshaped([b, t.0, h * k], strides: [t.1 * h * k, h * k, 1]).contiguous()
      + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: 6912, upcast: upcast, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + (contextChunks[5].to(of: contextOut)
      .* contextFF(
        contextNorm2(contextOut).to(.Float16) .* (1 + contextChunks[4]) + contextChunks[3])).to(
        of: contextOut)
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: 3584, upcast: upcast, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = MoEFeedForward(
    segments: 4, tokenLength: hw, hiddenSize: k * h, intermediateSize: 6912, upcast: upcast,
    name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xIn = xNorm2(xOut).to(.Float16) .* (1 + xChunks[4]) + xChunks[3]
  xOut =
    xOut
    + (xChunks[5].to(of: xOut) .* (xSharedFF(xIn) + xMoEFF(xIn))).to(of: xOut)
  let reader: (PythonObject) -> Void = { _ in }
  if !contextBlockPreOnly {
    return (reader, Model([x, context, c, rot], [xOut, contextOut]))
  } else {
    return (reader, Model([x, context, c, rot], [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let c = Input()
  let rot = Input()
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw + t.1, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw + t.1, h, k])
  let xV = xToValues(xOut).reshaped([b, hw + t.1, h, k])
  xQ = Functional.cmul(left: xQ, right: rot)
  xK = Functional.cmul(left: xK, right: rot)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    xQ, xK, xV
  ).reshaped([b, t.1 + hw, h * k])
  var xIn: Model.IO = x
  let xLength: Int
  if contextBlockPreOnly {
    xOut = out.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xIn = xIn.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xLength = hw
  } else {
    xOut = out.reshaped([b, hw + t.0, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xIn = xIn.reshaped([b, hw + t.0, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xLength = hw + t.0
  }
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  xOut = xIn + (xChunks[2] .* xOut).to(of: xIn)
  // Attentions are now. Now run MLP.
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: 3584, upcast: upcast, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = MoEFeedForward(
    segments: 4, tokenLength: xLength, hiddenSize: k * h, intermediateSize: 6912, upcast: upcast,
    name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xFFIn = xNorm2(xOut).to(.Float16) .* (1 + xChunks[4]) + xChunks[3]
  xOut =
    xOut + (xChunks[5].to(of: xOut) .* (xSharedFF(xFFIn) + xMoEFF(xFFIn))).to(of: xOut)
  let reader: (PythonObject) -> Void = { _ in }
  return (reader, Model([x, c, rot], [xOut]))
}

func HiDream(height: Int, width: Int, textLength: (Int, Int), layers: (Int, Int)) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let imgIn = Dense(count: 2_560, name: "x_embedder")
  var out = imgIn(x).to(.Float32)
  let t = Input()
  let vector = Input()
  let (tMlp0, tMlp2, timeEmbedder) = MLPEmbedder(channels: 2_560, name: "t")
  let (pMlp0, pMlp2, pooledEmbedder) = MLPEmbedder(channels: 2_560, name: "p")
  var vec = timeEmbedder(t) + pooledEmbedder(vector)
  let t5EncoderHiddenStates = Input()
  let llamaEncoderHiddenStates = (0..<32).map { _ in Input() }
  let captionProjections = (0..<49).map { _ in
    Dense(count: 2_560, noBias: true, name: "caption_projection")
  }
  var encoderHiddenStates = [Model.IO]()
  for i in 0..<48 {
    encoderHiddenStates.append(
      captionProjections[i](llamaEncoderHiddenStates[min(i, llamaEncoderHiddenStates.count - 1)]))
  }
  encoderHiddenStates.append(captionProjections[48](t5EncoderHiddenStates))
  var context = Functional.concat(
    axis: 1, encoderHiddenStates[encoderHiddenStates.count - 1],
    encoderHiddenStates[encoderHiddenStates.count - 2]
  ).to(.Float32)
  let h = height / 2
  let w = width / 2
  vec = vec.reshaped([1, 1, 2_560]).swish()
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers.0 {
    let contextIn = Functional.concat(axis: 1, context, encoderHiddenStates[i].to(.Float32))
    let (reader, block) = JointTransformerBlock(
      prefix: "double_stream_blocks.\(i).block", k: 128, h: 20, b: 1,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: false, upcast: i > 12)
    readers.append(reader)
    let blockOut = block(out, contextIn, vec, rot)
    out = blockOut[0]
    context = blockOut[1]
  }
  out = Functional.concat(axis: 1, out, context)
  for i in 0..<layers.1 {
    let xIn = Functional.concat(axis: 1, out, encoderHiddenStates[layers.0 + i].to(.Float32))
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_stream_blocks.\(i).block", k: 128, h: 20, b: 1,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: i == layers.1 - 1, upcast: false)
    readers.append(reader)
    out = block(xIn, vec, rot)
  }
  let scale = Dense(count: 2_560, name: "ada_ln_0")
  let shift = Dense(count: 2_560, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (
    Model([x, rot, t, vector, t5EncoderHiddenStates] + llamaEncoderHiddenStates, [out]), reader
  )
}

let z = graph.withNoGrad {
  let rotTensor = graph.variable(.CPU, .NHWC(1, 4096 + 128 * 3, 1, 128), of: Float.self)
  for y in 0..<64 {
    for x in 0..<64 {
      let i = y * 64 + x
      for k in 0..<32 {
        let theta = 0 * 1.0 / pow(10_000, Double(k) / 32)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(y) * 1.0 / pow(10_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 32) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 32) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(x) * 1.0 / pow(10_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 32 + 16) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 32 + 16) * 2 + 1] = Float(sintheta)
      }
    }
  }
  for i in 0..<(128 * 3) {
    for k in 0..<32 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 32)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, 4096 + i, 0, k * 2] = Float(costheta)
      rotTensor[0, 4096 + i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, 4096 + i, 0, (k + 32) * 2] = Float(costheta)
      rotTensor[0, 4096 + i, 0, (k + 32) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, 4096 + i, 0, (k + 32 + 16) * 2] = Float(costheta)
      rotTensor[0, 4096 + i, 0, (k + 32 + 16) * 2 + 1] = Float(sintheta)
    }
  }
  let timestep = graph.variable(
    Tensor<Float16>(
      from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(0))
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
  let (hiDream, reader) = HiDream(height: 128, width: 128, textLength: (128, 128), layers: (16, 32))
  var z = graph.variable(.GPU(0), .HWC(1, 4096, 64), of: FloatType.self)
  z.randn()
  hiDream.compile(
    inputs: [z, rotTensorGPU, timestep, pooledPromptEmbedTensor, promptEmbed3Tensor]
      + promptEmbed4Tensors)
  graph.openStore("/home/liu/workspace/swift-diffusion/hidream_i1_dev_f16.ckpt") {
    try! $0.read("dit", model: hiDream, strict: true)
  }
  let samplingSteps = 30
  for i in (1...samplingSteps).reversed() {
    print("\(i)")
    let t = Float(i) / Float(samplingSteps) * 1_000
    let tTensor = graph.variable(
      Tensor<FloatType>(
        from: timeEmbedding(timesteps: t, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
          .toGPU(0)))
    let v = hiDream(
      inputs: z,
      [rotTensorGPU, tTensor, pooledPromptEmbedTensor, promptEmbed3Tensor] + promptEmbed4Tensors)[0]
      .as(of: FloatType.self)
    z = z + (1 / Float(samplingSteps)) * v
    debugPrint(z)
  }
  return z.reshaped(format: .NCHW, shape: [1, 64, 64, 2, 2, 16]).permuted(
    0, 5, 1, 3, 2, 4
  )
  .contiguous().reshaped(format: .NCHW, shape: [1, 16, 128, 128])
}

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
  let f32 = out.to(.Float32)
  let hw = width * height
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let k = tokeys(f32).reshaped([batchSize, inChannels, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(f32)).reshaped([
    batchSize, inChannels, hw,
  ])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw]).to(of: out)
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

graph.withNoGrad {
  // Already processed out.
  let zTensor = (1.0 / 0.3611) * z + 0.11590  // DynamicGraph.Tensor<Float>(from: (1.0 / 0.3611) * z + 0.11590)
  let (_, decoder) = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  decoder.compile(inputs: zTensor)
  graph.openStore("/fast/Data/SD/flux_1_vae_f16.ckpt") {
    try! $0.read("decoder", model: decoder, strict: true)
  }
  let img = decoder(inputs: zTensor)[0].as(of: FloatType.self).toCPU()
  let startHeight = 128
  let startWidth = 128
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: startWidth * 8 * startHeight * 8)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let (r, g, b) = (img[0, 0, y, x], img[0, 1, y, x], img[0, 2, y, x])
      rgba[y * startWidth * 8 + x].r = UInt8(
        min(max(Int(Float(r.isFinite ? (r + 1) / 2 : 0) * 255), 0), 255))
      rgba[y * startWidth * 8 + x].g = UInt8(
        min(max(Int(Float(g.isFinite ? (g + 1) / 2 : 0) * 255), 0), 255))
      rgba[y * startWidth * 8 + x].b = UInt8(
        min(max(Int(Float(b.isFinite ? (b + 1) / 2 : 0) * 255), 0), 255))
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (startWidth * 8, startHeight * 8),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/\(filename).png", level: 4)
}
