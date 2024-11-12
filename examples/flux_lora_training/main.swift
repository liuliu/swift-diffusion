import Diffusion
import Foundation
import NNC
import PNG
import SentencePiece

typealias FloatType = Float16

struct PythonObject {}

DynamicGraph.setSeed(42)

let textEncodingLength = 512

let prompt =
  "liuliu"
// "Professional photograph of an astronaut riding a horse on the moon with view of Earth in the background."
// "a smiling indian man with a google t-shirt next to a frowning asian man with a shirt saying nexus at a meeting table facing each other, photograph, detailed, 8k"
// "photo of a young woman with long, wavy brown hair sleeping in grassfield, top down shot, summer, warm, laughing, joy, fun"
// "35mm analogue full-body portrait of a beautiful woman wearing black sheer dress, catwalking in a busy market, soft colour grading, infinity cove, shadows, kodak, contax t2"
// "A miniature tooth fairy woman is holding a pick axe and mining diamonds in a bedroom at night. The fairy has an angry expression."
let filename = "lora_f16"
let model = "flux_1_dev_f16"

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

func LoRAFeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = LoRADense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / 8) * out
  }
  let outProjection = LoRADense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool, upcast: Bool
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
  let contextToKeys = LoRADense(count: k * h, name: "c_k")
  let contextToQueries = LoRADense(count: k * h, name: "c_q")
  let contextToValues = LoRADense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = LoRADense(count: k * h, name: "x_k")
  let xToQueries = LoRADense(count: k * h, name: "x_q")
  let xToValues = LoRADense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(
    scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
  let out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  scaledDotProductAttention.gradientCheckpointing = false
  /*
  keys = keys.permuted(0, 2, 1, 3).contiguous()
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .permuted(0, 2, 1, 3).contiguous()
  values = values.permuted(0, 2, 1, 3).contiguous()
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  */
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = LoRADense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = LoRADense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = LoRAFeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "c")
    contextFF.gradientCheckpointing = true
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + ((upcast ? contextChunks[5].to(of: contextOut) : contextChunks[5])
      .* contextFF(
        contextNorm2(contextOut).to(.Float16) .* (1 + contextChunks[4]) + contextChunks[3])).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "x")
  xFF.gradientCheckpointing = true
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + ((upcast ? xChunks[5].to(of: xOut) : xChunks[5])
    .* xFF(xNorm2(xOut).to(.Float16) .* (1 + xChunks[4]) + xChunks[3])).to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, c, rot], [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, c, rot], [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let c = Input()
  let rot = Input()
  let xAdaLNs = (0..<3).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = ((1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0])
  let xToKeys = LoRADense(count: k * h, name: "x_k")
  let xToQueries = LoRADense(count: k * h, name: "x_q")
  let xToValues = LoRADense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(
    scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  scaledDotProductAttention.gradientCheckpointing = false
  /*
  keys = keys.permuted(0, 2, 1, 3).contiguous()
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .permuted(0, 2, 1, 3).contiguous()
  values = values.permuted(0, 2, 1, 3).contiguous()
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  */
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1]
    )
  }
  let xUnifyheads = LoRADense(count: k * h, noBias: true, name: "x_o")
  let (_, _, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: false, name: "x")
  xFF.gradientCheckpointing = true
  out = xUnifyheads(out) + xFF(xOut)
  out = xIn + (xChunks[2] .* out).to(of: xIn)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, c, rot], [out]))
}

func MMDiT(b: Int, h: Int, w: Int, guidanceEmbed: Bool) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t = Input()
  let y = Input()
  let contextIn = Input()
  let rot = Input()
  let guidance: Input?
  let xEmbedder = Dense(count: 3072, name: "x_embedder")
  var out = xEmbedder(x).to(.Float32)
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: 3072, name: "t")
  var vec = tEmbedder(t)
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = MLPEmbedder(channels: 3072, name: "guidance")
    let g = Input()
    vec = vec + gEmbedder(g)
    guidance = g
    gMlp0 = mlp0
    gMlp2 = mlp2
  } else {
    gMlp0 = nil
    gMlp2 = nil
    guidance = nil
  }
  let (yMlp0, yMlp2, yEmbedder) = MLPEmbedder(channels: 3072, name: "vector")
  vec = vec + yEmbedder(y)
  let contextEmbedder = Dense(count: 3072, name: "context_embedder")
  var context = contextEmbedder(contextIn).to(.Float32)
  let c = vec.reshaped([b, 1, 3072]).swish()
  var readers = [(PythonObject) -> Void]()
  var c32: Model.IO = c
  var rot32: Model.IO = rot
  for i in 0..<19 {
    let (reader, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: 24, b: b, t: textEncodingLength, hw: h * w,
      contextBlockPreOnly: false, upcast: i > 16)  // Just last layer should be enough, but to be safe, for the last 2 layers, we will do upcast.
    let blockOut = block(context, out, c32, rot32)
    // block.gradientCheckpointing = true
    context = blockOut[0]
    out = blockOut[1]
    readers.append(reader)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<38 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: 24, b: b, t: textEncodingLength, hw: h * w,
      contextBlockPreOnly: i == 37)
    // block.gradientCheckpointing = true
    out = block(out, c, rot)
    readers.append(reader)
  }
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(c)) .* normFinal(out).to(.Float16) + shift(c)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (
    reader,
    Model([x, t, y, contextIn, rot] + (guidance.map { [$0] } ?? []), [out], trainable: false)
  )
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

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")
let tokens0 = tokenizer.tokenize(text: prompt, truncation: true, maxLength: 77)
let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/swift-diffusion/examples/sd3/spiece.model")
var tokens1 = sentencePiece.encode(prompt).map { return $0.id }
tokens1.append(1)
let tokensTensor0 = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensTensor1 = graph.variable(.CPU, .C(textEncodingLength), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
for i in 0..<77 {
  tokensTensor0[i] = tokens0[i]
  positionTensor[i] = Int32(i)
}
for i in 0..<textEncodingLength {
  tokensTensor1[i] = i < tokens1.count ? tokens1[i] : 0
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

let pooled = graph.withNoGrad {
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
  for (i, token) in tokens0.enumerated() {
    if token == tokenizer.endToken {
      pooled[0..<1, 0..<768] = c[1][i..<(i + 1), 0..<768]
      break
    }
  }
  return pooled
}

let c1 = graph.withNoGrad {
  let (_, textModel) = T5ForConditionalGeneration(b: 1, t: textEncodingLength)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: textEncodingLength, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor1.toGPU(0)
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, relativePositionBucketsGPU)
  graph.openStore("/home/liu/workspace/swift-llm/t5_xxl_encoder_f32.ckpt") {
    $0.read("text_model", model: textModel, codec: [.q8p, .q6p, .q4p, .ezm7])
  }
  let output = textModel(inputs: tokensTensorGPU, relativePositionBucketsGPU)[0].as(
    of: FloatType.self
  ).reshaped(.CHW(1, textEncodingLength, 4096))
  return output
}

let startHeight = 64
let startWidth = 64

var initImg = Tensor<FloatType>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
if let image = try PNG.Data.Rectangular.decompress(
  path: "/home/liu/workspace/swift-diffusion/IMG_2778.png")
{
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)

  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let pixel = rgba[y * startWidth * 8 + x]
      initImg[0, 0, y, x] = FloatType(Float(pixel.r) / 255 * 2 - 1)
      initImg[0, 1, y, x] = FloatType(Float(pixel.g) / 255 * 2 - 1)
      initImg[0, 2, y, x] = FloatType(Float(pixel.b) / 255 * 2 - 1)
    }
  }
}

let x = graph.withNoGrad {
  let (_, encoder) = Encoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
    startHeight: startHeight)
  let image = graph.variable(initImg.toGPU(0))
  encoder.compile(inputs: image)
  graph.openStore("/home/liu/workspace/swift-diffusion/flux_1_vae_f16.ckpt") {
    $0.read("encoder", model: encoder)
  }
  let z =
    (encoder(inputs: image)[0].as(of: FloatType.self)[
      0..<1, 0..<16, 0..<startHeight, 0..<startWidth
    ].copied() - 0.11590) * 0.3611
  return z.reshaped(format: .NHWC, shape: [1, 16, startHeight / 2, 2, startWidth / 2, 2]).permuted(
    0, 2, 4, 1, 3, 5
  )
  .contiguous().reshaped(.HWC(1, (startHeight / 2) * (startWidth / 2), 64))
}

debugPrint(x)

let (_, dit) = MMDiT(b: 1, h: startHeight / 2, w: startWidth / 2, guidanceEmbed: true)
var rotTensor = Tensor<Float>(
  .CPU, .NHWC(1, (startHeight / 2 * startWidth / 2) + textEncodingLength, 24, 128))
for i in 0..<textEncodingLength {
  for j in 0..<24 {
    for k in 0..<8 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 8)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, j, k * 2] = Float(costheta)
      rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<28 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, j, (k + 8) * 2] = Float(costheta)
      rotTensor[0, i, j, (k + 8) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<28 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, j, (k + 8 + 28) * 2] = Float(costheta)
      rotTensor[0, i, j, (k + 8 + 28) * 2 + 1] = Float(sintheta)
    }
  }
}
for y in 0..<(startHeight / 2) {
  for j in 0..<24 {
    for x in 0..<(startWidth / 2) {
      let i = y * (startWidth / 2) + x + textEncodingLength
      for k in 0..<8 {
        let theta = 0 * 1.0 / pow(10_000, Double(k) / 8)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, k * 2] = Float(costheta)
        rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<28 {
        let theta = Double(y) * 1.0 / pow(10_000, Double(k) / 28)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, (k + 8) * 2] = Float(costheta)
        rotTensor[0, i, j, (k + 8) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<28 {
        let theta = Double(x) * 1.0 / pow(10_000, Double(k) / 28)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, (k + 8 + 28) * 2] = Float(costheta)
        rotTensor[0, i, j, (k + 8 + 28) * 2 + 1] = Float(sintheta)
      }
    }
  }
}
let rotTensorGPU = graph.constant(Tensor<FloatType>(from: rotTensor).toGPU(0))
dit.maxConcurrency = .limit(1)
var z = graph.variable(
  .GPU(0), .HWC(1, (startHeight / 2) * (startWidth / 2), 64), of: FloatType.self)
z.randn()
let tTensor = graph.constant(
  Tensor<FloatType>(
    from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
      .toGPU(0)))
var yTensor = graph.variable(.GPU(0), .WC(1, 768), of: FloatType.self)
yTensor[0..<1, 0..<768] = pooled
yTensor = graph.constant(yTensor.rawValue)
var cTensor = graph.variable(.GPU(0), .HWC(1, textEncodingLength, 4096), of: FloatType.self)
cTensor[0..<1, 0..<textEncodingLength, 0..<4096] = c1
cTensor = graph.constant(cTensor.rawValue)
let gTensor = graph.constant(
  Tensor<FloatType>(
    from: timeEmbedding(timesteps: 3500, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
      .toGPU(0)))
dit.memoryReduction = true
/*
var input = graph.variable(.GPU(0), .CHW(2, 4096, 64), of: FloatType.self)
*/
dit.compile(inputs: z, tTensor, yTensor, cTensor, rotTensorGPU, gTensor)
graph.openStore("/home/liu/workspace/swift-diffusion/\(model).ckpt") {
  $0.read("dit", model: dit, codec: [.q8p, .q6p, .q4p, .ezm7]) { name, dataType, format, shape in
    if name.contains("lora_up") {
      precondition(dataType == .Float32)
      var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<Float32>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return .final(tensor)
    }
    return .continue(name)
  }
}

var optimizer = AdamWOptimizer(graph, rate: 0.0001)
optimizer.parameters = [dit.parameters]
var scaler = GradScaler()
for i in 0..<2000 {
  let t = graph.withNoGrad {
    let t = graph.variable(.CPU, .C(1), of: Float.self)
    t.rand(0...1)
    return t
  }
  let (zt, z1) = graph.withNoGrad {
    let tG = DynamicGraph.Tensor<FloatType>(from: t.toGPU(0))
    let z1 = graph.variable(like: x)
    z1.randn()
    let zt = (1 - tG) .* x + tG .* z1
    return (zt, z1)
  }
  let tTensor = graph.constant(
    Tensor<FloatType>(
      from: timeEmbedding(
        timesteps: t[0] * 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000
      )
      .toGPU(0)))
  let vtheta = dit(inputs: zt, tTensor, yTensor, cTensor, rotTensorGPU, gTensor)[0].as(
    of: FloatType.self)
  let diff = z1 - x - vtheta
  let loss = (diff .* diff).reduced(.mean, axis: [1, 2])
  DynamicGraph.logLevel = .verbose
  scaler.scale(loss).backward(to: [zt])
  scaler.step(&optimizer)
  exit(0)
  let time = t[0]
  let lossC = loss.rawValue.toCPU()[0, 0, 0]
  print("epoch \(i), t \(time), loss \(lossC)")
}

let samplingSteps = 20
z.randn()
let testGTensor = graph.constant(
  Tensor<FloatType>(
    from: timeEmbedding(timesteps: 3500, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
      .toGPU(0)))
for i in (1...samplingSteps).reversed() {
  let t = Float(i) / Float(samplingSteps) * 1_000
  let tTensor = graph.constant(
    Tensor<FloatType>(
      from: timeEmbedding(timesteps: t, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
        .toGPU(0)))
  let v = dit(inputs: z, tTensor, yTensor, cTensor, rotTensorGPU, testGTensor)[0].as(
    of: FloatType.self)
  /*
  let vu = vcu[0..<1, 0..<4096, 0..<64]
  let vc = vcu[1..<2, 0..<4096, 0..<64]
  let v = vu + 7 * (vc - vu)
  */
  z = z - (1 / Float(samplingSteps)) * v
  debugPrint(z)
}
z = z.reshaped(format: .NCHW, shape: [1, startHeight / 2, startWidth / 2, 16, 2, 2]).permuted(
  0, 3, 1, 4, 2, 5
)
.contiguous().reshaped(.NCHW(1, 16, startHeight, startWidth))

graph.withNoGrad {
  /*
  var x = x.reshaped(format: .NCHW, shape: [1, startHeight / 2, startWidth / 2, 16, 2, 2])
  x = x.permuted(0, 3, 1, 4, 2, 5)
  x = x.contiguous().reshaped(.NCHW(1, 16, startHeight, startWidth))
  */
  // Already processed out.
  let zTensor = (1.0 / 0.3611) * z + 0.11590  // DynamicGraph.Tensor<Float>(from: (1.0 / 0.3611) * z + 0.11590)
  let (_, decoder) = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
    startHeight: startHeight)
  decoder.compile(inputs: zTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/flux_1_vae_f16.ckpt") {
    $0.read("decoder", model: decoder)
  }
  let img = decoder(inputs: zTensor)[0].as(of: FloatType.self).toCPU()
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
