import Diffusion
import Foundation
import NNC

struct PythonObject {}

DynamicGraph.setSeed(42)

let tokenizer = GPT2Tokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/hunyuan/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/hunyuan/merges.txt",
  specialTokens: [
    "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
    "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
  ])
let prompt = "A cat walks on the grass, realistic style."
let promptWithTemplate =
  "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\(prompt)<|eot_id|>"
print(promptWithTemplate)
let result = tokenizer.tokenize(text: promptWithTemplate, addSpecialTokens: true)
print(result)

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([x, rot], [out]), reader)
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
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([x, rot], [out]), reader)
}

func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Int?, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embedReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  var hiddenStates: Model.IO? = nil
  for i in 0..<layers {
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: heads / 4, b: batchSize,
      t: tokenLength,
      MLP: MLP)
    out = layer(out, rot)
    readers.append(reader)
    if let outputHiddenStates = outputHiddenStates, outputHiddenStates == i {
      hiddenStates = out
    }
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([tokens, rot], (hiddenStates.map { [$0] } ?? []) + [out]), reader)
}

func CLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let causalAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    Float16.self, batchSize: batchSize,
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
    out = encoderLayer(out, causalAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, causalAttentionMask], [penultimate!, out])
}

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func RefinerSelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let tokeys = Dense(count: k * hk, name: "refiner_k_proj")
  let toqueries = Dense(count: k * h, name: "refiner_q_proj")
  let tovalues = Dense(count: k * hk, name: "refiner_v_proj")
  let keys = tokeys(x).reshaped([b, t, hk, k])
  let queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
    queries, keys, values
  ).reshaped([b, t, h * k])
  let unifyheads = Dense(count: k * h, name: "refiner_out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    // The rotary in Llama is first half and second half, we can be clever and do the extra transpose here to use with cmul.
  }
  return (Model([x], [out]), reader)
}

func IndividualRefinerBlock(prefix: String, t: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let c = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], name: "refiner_norm1")
  let gateMsa = Dense(count: 3_072, name: "refiner_ada_ln_msa")
  let (attention, attentionReader) = RefinerSelfAttention(
    prefix: prefix, k: 128, h: 24, hk: 24, b: 1, t: t)
  var out = x + attention(norm1(x)) .* gateMsa(c)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], name: "refiner_norm2")
  let mlp0 = Dense(count: 3_072 * 4, name: "refiner_mlp_0")
  let mlp1 = Dense(count: 3_072, name: "refiner_mlp_1")
  let gateMlp = Dense(count: 3_072, name: "refiner_ada_ln_mlp")
  out = out + mlp1(mlp0(norm2(out)).swish()) .* gateMlp(c)
  let reader: (PythonObject) -> Void = { state_dict in
    attentionReader(state_dict)
  }
  return (Model([x, c], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
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
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
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
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  xQ = Functional.cmul(left: xQ, right: rot)
  xK = Functional.cmul(left: xK, right: rot)
  let keys = Functional.concat(axis: 1, xK, contextK)
  let values = Functional.concat(axis: 1, xV, contextV)
  let queries = Functional.concat(axis: 1, xQ, contextQ)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    queries, keys, values
  ).reshaped([b, t + hw, h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], offset: [0, hw, 0], strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
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
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "c")
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
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + ((upcast ? xChunks[5].to(of: xOut) : xChunks[5])
    .* xFF(xNorm2(xOut).to(.Float16) .* (1 + xChunks[4]) + xChunks[3])).to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  if !contextBlockPreOnly {
    return (reader, Model([x, context, c, rot], [xOut, contextOut]))
  } else {
    return (reader, Model([x, context, c, rot], [xOut]))
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
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  let queries = Functional.cmul(left: xQ, right: rot)
  let keys = Functional.cmul(left: xK, right: rot)
  let values = xV
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(
    scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xLinear1 = Dense(count: k * h * 4, name: "x_linear1")
  let xOutProjection = Dense(count: k * h, name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + (xChunks[2] .* out).to(of: xIn)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, c, rot], [out]))
}

func Hunyuan(time: Int, height: Int, width: Int, textLength: Int) -> (Model, (PythonObject) -> Void)
{
  let x = Input()
  let rot = Input()
  let imgIn = Dense(count: 3072, name: "x_embedder")
  let txt = Input()
  let t = Input()
  let vector = Input()
  let guidanceEmbed = Input()
  let (tMlp0, tMlp2, timeEmbedder) = MLPEmbedder(channels: 3_072, name: "txt_in_t")
  var c = txt.reduced(.mean, axis: [1])
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: 3_072, name: "c")
  c = timeEmbedder(t) + contextEmbedder(c)
  c = c.reshaped([1, 1, 3072]).swish()
  let inputEmbedder = Dense(count: 3_072, name: "input_embedder")
  var context = inputEmbedder(txt)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<2 {
    let (block, reader) = IndividualRefinerBlock(
      prefix: "txt_in.individual_token_refiner.blocks.\(i)", t: textLength)
    context = block(context, c)
    readers.append(reader)
  }
  context = context.to(.Float32)
  var out = imgIn(x).to(.Float32)
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(channels: 3_072, name: "t")
  let (vMlp0, vMlp2, vectorIn) = MLPEmbedder(channels: 3_072, name: "vector")
  let (gMlp0, gMlp2, guidanceIn) = MLPEmbedder(channels: 3_072, name: "guidance")
  var vec = timeIn(t) + vectorIn(vector) + guidanceIn(guidanceEmbed)
  vec = vec.reshaped([1, 1, 3072]).swish()
  let h = height / 2
  let w = width / 2
  for i in 0..<20 {
    let (reader, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: time * h * w,
      contextBlockPreOnly: false, upcast: i > 5)
    let blockOut = block(out, context, vec, rot)
    out = blockOut[0]
    context = blockOut[1]
    readers.append(reader)
  }
  let rot2 = Input()
  out = Functional.concat(axis: 1, out, context)
  for i in 0..<40 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: time * h * w,
      contextBlockPreOnly: i == 39)
    out = block(out, vec, rot2)
    readers.append(reader)
  }
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x, rot, rot2, txt, t, vector, guidanceEmbed], [out]), reader)
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

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)
graph.withNoGrad {
  let pooled = graph.withNoGrad {
    let tokenizer = CLIPTokenizer(
      vocabulary: "/home/liu/workspace/swift-diffusion/examples/clip/vocab.json",
      merges: "/home/liu/workspace/swift-diffusion/examples/clip/merges.txt")
    let tokens0 = tokenizer.tokenize(text: prompt, truncation: true, maxLength: 77)
    let tokensTensor0 = graph.variable(.CPU, .C(77), of: Int32.self)
    let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
    for i in 0..<77 {
      tokensTensor0[i] = tokens0[i]
      positionTensor[i] = Int32(i)
    }

    let causalAttentionMask = graph.variable(Tensor<Float16>(.CPU, .NHWC(1, 1, 77, 77)))
    causalAttentionMask.full(0)
    for i in 0..<76 {
      for j in (i + 1)..<77 {
        causalAttentionMask[0, 0, i, j] = -Float16.greatestFiniteMagnitude
      }
    }
    let tokensTensorGPU = tokensTensor0.toGPU(0)
    let positionTensorGPU = positionTensor.toGPU(0)
    let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
    let textModel0 = CLIPTextModel(
      vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
      batchSize: 1, intermediateSize: 3072)
    textModel0.compile(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
    graph.openStore("/home/liu/workspace/swift-diffusion/clip_vit_l14_f32.ckpt") {
      $0.read("text_model", model: textModel0)
    }
    let c = textModel0(inputs: tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU).map {
      $0.as(of: Float16.self)
    }
    var pooled = graph.variable(.GPU(0), .WC(1, 768), of: Float16.self)
    for (i, token) in tokens0.enumerated() {
      if token == tokenizer.endToken {
        pooled[0..<1, 0..<768] = c[1][i..<(i + 1), 0..<768]
        break
      }
    }
    return pooled
  }
  debugPrint(pooled)
  let lastHiddenStates = graph.withNoGrad {
    let (transformer, reader) = Transformer(
      Float16.self, vocabularySize: 128_320, maxLength: 351, width: 4_096, tokenLength: 351,
      layers: 32, MLP: 14336, heads: 32, outputHiddenStates: 29, batchSize: 1)
    let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [351], of: Int32.self)
    for i in 0..<result.count {
      tokensTensor[i] = result[i]
    }
    for i in result.count..<351 {
      tokensTensor[i] = 128258
    }
    let rotTensor = graph.variable(.CPU, .NHWC(1, 351, 1, 128), of: Float.self)
    for i in 0..<351 {
      for k in 0..<64 {
        let theta = Double(i) * 1.0 / pow(500_000, Double(k) * 2 / 128)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
    }
    let tokensTensorGPU = tokensTensor.toGPU(0)
    let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
    transformer.compile(inputs: tokensTensorGPU, rotTensorGPU)
    graph.openStore("/home/liu/workspace/swift-diffusion/llava_llama_3_8b_v1.1_f16.ckpt") {
      $0.read("llava", model: transformer)
    }
    return transformer(inputs: tokensTensorGPU, rotTensorGPU)[0].as(of: Float16.self)[
      95..<106, 0..<4096
    ].reshaped(.HWC(1, 11, 4096)).copied()  // We don't need attention mask, just reduce the hidden states.
  }
  debugPrint(lastHiddenStates)
  let timestep = timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  var rotNdTensor = graph.variable(.CPU, .NHWC(1, 33 * 34 * 60, 1, 128), of: Float.self)
  var rotNdTensor2 = graph.variable(.CPU, .NHWC(1, 33 * 34 * 60 + 11, 1, 128), of: Float.self)
  for t in 0..<33 {
    for y in 0..<34 {
      for x in 0..<60 {
        let i = t * 34 * 60 + y * 60 + x
        for k in 0..<8 {
          let theta = Double(t) * 1.0 / pow(256, Double(k) / 8)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, k * 2] = Float(costheta)
          rotNdTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
          rotNdTensor2[0, i, 0, k * 2] = Float(costheta)
          rotNdTensor2[0, i, 0, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<28 {
          let theta = Double(y) * 1.0 / pow(256, Double(k) / 28)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, (k + 8) * 2] = Float(costheta)
          rotNdTensor[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
          rotNdTensor2[0, i, 0, (k + 8) * 2] = Float(costheta)
          rotNdTensor2[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<28 {
          let theta = Double(x) * 1.0 / pow(256, Double(k) / 28)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
          rotNdTensor[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
          rotNdTensor2[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
          rotNdTensor2[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  for i in 33 * 34 * 60..<(33 * 34 * 60 + 11) {
    for k in 0..<64 {
      rotNdTensor2[0, i, 0, k * 2] = 1
      rotNdTensor2[0, i, 0, k * 2 + 1] = 0
    }
  }
  let (hunyuan, hunyuanReader) = Hunyuan(time: 33, height: 68, width: 120, textLength: 11)
  hunyuan.maxConcurrency = .limit(1)
  let tGPU = graph.variable(Tensor<Float16>(from: timestep)).toGPU(0)
  var xTensor = graph.variable(.GPU(0), .HWC(1, 33 * 34 * 60, 64), of: Float16.self)
  xTensor.randn()
  let guidanceEmbed = timeEmbedding(
    timesteps: 3500, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let gGPU = graph.variable(Tensor<Float16>(from: guidanceEmbed)).toGPU(0)
  let vector = pooled
  let rotNdTensorGPU = DynamicGraph.Tensor<Float16>(from: rotNdTensor).toGPU(0)
  let rotNdTensor2GPU = DynamicGraph.Tensor<Float16>(from: rotNdTensor2).toGPU(0)
  hunyuan.compile(
    inputs: xTensor, rotNdTensorGPU, rotNdTensor2GPU, lastHiddenStates, tGPU, vector, gGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/hunyuan_video_t2v_720p_f16.ckpt") {
    $0.read("dit", model: hunyuan)
  }
  debugPrint(
    hunyuan(inputs: xTensor, rotNdTensorGPU, rotNdTensor2GPU, lastHiddenStates, tGPU, vector, gGPU))
}
