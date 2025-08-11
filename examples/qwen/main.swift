import Diffusion
import Foundation
import NNC
import PNG
import TensorBoard

let filename = "qwen_image_f16"

DynamicGraph.setSeed(42)

struct PythonObject {}

let graph = DynamicGraph()

graph.maxConcurrency = .limit(4)

let prompt =
  "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\nA coffee shop entrance features a chalkboard sign reading \"Qwen Coffee 😊 $2 per cup,\" with a neon light beside it displaying \"通义千问\". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written \"π≈3.1415926-53589793-23846264-33832795-02384197\".<|im_end|>\n<|im_start|>assistant\n"
let negativePrompt =
  "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n <|im_end|>\n<|im_start|>assistant\n"

let tokenizer = TiktokenTokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/qwen/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/qwen/merges.txt",
  specialTokens: [
    "<|im_end|>": 151645, "<|im_start|>": 151644, "<|endoftext|>": 151643,
    "<|file_sep|>": 151664, "</tool_call>": 151658,
  ])

let positiveTokens = tokenizer.tokenize(text: prompt, addSpecialTokens: false)
let negativeTokens = tokenizer.tokenize(text: negativePrompt, addSpecialTokens: false)

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
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
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
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
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: 4, b: batchSize,
      t: tokenLength,
      MLP: MLP)
    out = layer(out, rot)
    readers.append(reader)
    if let outputHiddenStates = outputHiddenStates, outputHiddenStates == i {
      hiddenStates = out
    }
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([tokens, rot], (hiddenStates.map { [$0] } ?? []) + [out]), reader)
}

let txt = graph.withNoGrad {
  let rotTensor = graph.variable(.CPU, .NHWC(1, positiveTokens.0.count, 1, 128), of: Float.self)
  for i in 0..<positiveTokens.0.count {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let (transformer, reader) = Transformer(
    Float16.self, vocabularySize: 152_064, maxLength: positiveTokens.0.count, width: 3_584,
    tokenLength: positiveTokens.0.count,
    layers: 28, MLP: 18_944, heads: 28, outputHiddenStates: 28, batchSize: 1)
  let tokensTensor = graph.variable(
    .CPU, format: .NHWC, shape: [positiveTokens.0.count], of: Int32.self)
  for i in 0..<positiveTokens.0.count {
    tokensTensor[i] = positiveTokens.0[i]
  }
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
  transformer.compile(inputs: tokensTensorGPU, rotTensorGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/qwen_2.5_vl_7b_f16.ckpt", flags: [.readOnly])
  {
    $0.read("text_model", model: transformer)
  }
  let lastHiddenStates = transformer(inputs: tokensTensorGPU, rotTensorGPU)[0].as(of: Float16.self)[
    34..<positiveTokens.0.count, 0..<3584
  ].copied()
  return lastHiddenStates
}

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, scaleFactor: Float, name: String)
  -> (
    Model, Model, Model
  )
{
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  out = (1.0 / scaleFactor) * out
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = scaleFactor * outProjection(out).to(.Float32)
  return (linear1, outProjection, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  scaleFactor: (Float, Float)
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let rot = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(
    epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut =
    ((1 + contextChunks[1].to(of: context)) .* contextNorm1(context)
      + contextChunks[0].to(of: context))
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  let downcastContextOut = ((1.0 / 8) * contextOut).to(.Float16)
  var contextK = contextToKeys(downcastContextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6 / 64, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(downcastContextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6 / 64, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(downcastContextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x)
  xOut = ((1 + xChunks[1].to(of: x)) .* xOut + xChunks[0].to(of: x))
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  let downcastXOut = ((1.0 / 8) * xOut).to(.Float16)
  var xK = xToKeys(downcastXOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6 / 64, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(downcastXOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6 / 64, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(downcastXOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    queries, keys, values
  ).reshaped([b, t + hw, h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut =
      (8 * scaleFactor.0) * unifyheads((1.0 / scaleFactor.0) * contextOut).to(of: context)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = (8 * scaleFactor.0) * xUnifyheads((1.0 / scaleFactor.0) * xOut).to(of: x)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2]).to(of: context) .* contextOut
  }
  xOut = x + (xChunks[2]).to(of: x) .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1, name: "c")
    let contextNorm2 = LayerNorm(
      epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + contextChunks[5].to(of: contextOut)
      .* contextFF(
        (contextNorm2(contextOut) .* (1 + contextChunks[4].to(of: contextOut))
          + contextChunks[3].to(of: contextOut)).to(.Float16)
      ).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + xChunks[5].to(of: xOut)
    .* xFF((xNorm2(xOut) .* (1 + xChunks[4].to(of: xOut)) + xChunks[3].to(of: xOut)).to(.Float16))
    .to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  if !contextBlockPreOnly {
    return (reader, Model([x, context, c, rot], [xOut, contextOut]))
  } else {
    return (reader, Model([x, context, c, rot], [xOut]))
  }
}

func QwenImage(height: Int, width: Int, textLength: Int, layers: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let txt = Input()
  let t = Input()
  let imgIn = Dense(count: 3072, name: "x_embedder")
  let txtNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm")
  let txtIn = Dense(count: 3_072, name: "context_embedder")
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(channels: 3_072, name: "t")
  var vec = timeIn(t)
  vec = vec.reshaped([1, 1, 3072]).swish()
  var context = txtIn(txtNorm(txt))
  var readers = [(PythonObject) -> Void]()
  context = context.to(.Float32)
  var out = imgIn(x).to(.Float32)
  let h = height / 2
  let w = width / 2
  for i in 0..<layers {
    let (reader, block) = JointTransformerBlock(
      prefix: "transformer_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: h * w,
      contextBlockPreOnly: i == layers - 1,
      scaleFactor: (i >= layers - 16 ? 16 : 2, i >= layers - 1 ? 256 : 16))
    let blockOut = block(out, context, vec, rot)
    if i == layers - 1 {
      out = blockOut
    } else {
      out = blockOut[0]
      context = blockOut[1]
    }
    readers.append(reader)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x, rot, t, txt], [out]), reader)
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

let z = graph.withNoGrad {
  let tTensor = graph.variable(
    Tensor<Float16>(
      from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(0))
  let textLength = positiveTokens.0.count - 34
  let cTensor = txt.reshaped(.HWC(1, textLength, 3584)).toGPU(0)
  let rotTensor = graph.variable(.CPU, .NHWC(1, 4096 + textLength, 1, 128), of: Float.self)
  let maxImgIdx = max(64 / 2, 64 / 2)
  for i in 0..<textLength {
    for k in 0..<8 {
      let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) / 8)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<28 {
      let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 8) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<28 {
      let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<64 {
    for x in 0..<64 {
      let i = y * 64 + x + textLength
      for k in 0..<8 {
        let theta = 0 * 1.0 / pow(10_000, Double(k) / 8)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<28 {
        let theta = Double(y - (64 - 64 / 2)) * 1.0 / pow(10_000, Double(k) / 28)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 8) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<28 {
        let theta = Double(x - (64 - 64 / 2)) * 1.0 / pow(10_000, Double(k) / 28)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
      }
    }
  }
  debugPrint(rotTensor)
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(0)
  let (dit, reader) = QwenImage(height: 128, width: 128, textLength: textLength, layers: 60)
  dit.maxConcurrency = .limit(4)
  var z = graph.variable(.GPU(0), .HWC(1, 4096, 64), of: Float16.self)
  z.randn()
  dit.compile(inputs: z, rotTensorGPU, tTensor, cTensor)
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/qwen_image_1.0_f16.ckpt", flags: [.readOnly]
  ) {
    $0.read("dit", model: dit)
  }
  let samplingSteps = 50
  for i in (1...samplingSteps).reversed() {
    print("\(i)")
    let t = Float(i) / Float(samplingSteps) * 1_000
    let tTensor = graph.variable(
      Tensor<Float16>(
        from: timeEmbedding(timesteps: t, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
          .toGPU(0)))
    let v = dit(inputs: z, rotTensorGPU, tTensor, cTensor)[0].as(of: Float16.self)
    z = z - (1 / Float(samplingSteps)) * v
    debugPrint(z)
  }
  return z.reshaped(format: .NCHW, shape: [1, 64, 64, 16, 2, 2]).permuted(
    0, 3, 1, 4, 2, 5
  ).contiguous().reshaped(format: .NCHW, shape: [1, 16, 1, 128, 128])
}

func ResnetBlockCausal3D(
  prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool, depth: Int, height: Int,
  width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm1")
  var out = norm1(x.reshaped([inChannels, depth, height, width])).reshaped([
    1, inChannels, depth, height, width,
  ])
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "resnet_conv1")
  out = conv1(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm2")
  out = norm2(out.reshaped([outChannels, depth, height, width])).reshaped([
    1, outChannels, depth, height, width,
  ])
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "resnet_conv2")
  out = conv2(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "resnet_shortcut")
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

func AttnBlockCausal3D(
  prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm = RMSNorm(epsilon: 1e-6, axis: [0], name: "attn_norm")
  var out = norm(x.reshaped([inChannels, depth, height, width])).reshaped([
    1, inChannels, depth, height, width,
  ])
  let hw = width * height
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "to_k")
  let k = tokeys(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "to_q")
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    inChannels, depth, hw,
  ]).transposed(0, 1)
  var dot =
    Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([depth * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([depth, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "to_v")
  let v = tovalues(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "proj_out")
  out = x + projOut(out.transposed(0, 1).reshaped([1, inChannels, depth, height, width]))
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x], [out]))
}

func DecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1, 1], name: "post_quant_conv")
  var out = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_in")
  out = convIn(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let (midBlockReader1, midBlock1) = ResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlockCausal3D(
    prefix: "decoder.mid_block.attentions.0", inChannels: previousChannel, depth: startDepth,
    height: startHeight, width: startWidth)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock2(out)
  var width = startWidth
  var height = startHeight
  var depth = startDepth
  var readers = [(PythonObject) -> Void]()
  var k = 0
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)",
        inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
      k += 1
    }
    if i > 0 {
      if i > 1 && startDepth > 1 {  // Need to bump up on the depth axis.
        let first = out.reshaped(
          [channel, 1, height, width], strides: [depth * height * width, height * width, width, 1]
        ).contiguous()
        let more = out.reshaped(
          [channel, (depth - 1), height, width], offset: [0, 1, 0, 0],
          strides: [depth * height * width, height * width, width, 1]
        ).contiguous().reshaped([1, channel, depth - 1, height, width])
        let timeConv = Convolution(
          groups: 1, filters: channel * 2, filterSize: [3, 1, 1],
          hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
          name: "time_conv")
        var expanded = timeConv(more.padded(.zero, begin: [0, 0, 2, 0, 0], end: [0, 0, 0, 0, 0]))
        let upLayer = channels.count - 1 - i
        let reader: (PythonObject) -> Void = { state_dict in
        }
        readers.append(reader)
        expanded = expanded.reshaped([2, channel, depth - 1, height, width]).permuted(1, 2, 0, 3, 4)
          .contiguous().reshaped([channel, 2 * (depth - 1), height, width])
        out = Functional.concat(axis: 1, first, expanded)
        depth = 1 + (depth - 1) * 2
        out = out.reshaped([1, channel, depth, height, width])
      }
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
        out.reshaped([channel, depth, height, width])
      ).reshaped([1, channel, depth, height * 2, width * 2])
      width *= 2
      height *= 2
      let conv2d = Convolution(
        groups: 1, filters: channel / 2, filterSize: [1, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
        name: "upsample")
      out = conv2d(out)
      previousChannel = channel / 2
      let upLayer = channels.count - 1 - i
      let reader: (PythonObject) -> Void = { state_dict in
      }
      readers.append(reader)
      k += 1
    }
  }
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  out = normOut(out.reshaped([channels[0], depth, height, width])).reshaped([
    1, channels[0], depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let reader: (PythonObject) -> Void = { state_dict in
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x], [out]))
}

func EncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_in")
  var out = convIn(x.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  var readers = [(PythonObject) -> Void]()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  var k = 0
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "encoder.down_blocks.\(k)", inChannels: previousChannel,
        outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
      k += 1
    }
    if i < channels.count - 1 {
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [1, 3, 3],
        hint: Hint(
          stride: [1, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        name: "downsample")
      out = conv2d(out.padded(.zero, begin: [0, 0, 0, 0, 0], end: [0, 0, 0, 1, 1]))
      let downLayer = k
      let reader: (PythonObject) -> Void = { state_dict in
      }
      readers.append(reader)
      if i > 0 && depth > 1 {
        let first = out.reshaped(
          [1, channel, 1, height, width],
          strides: [depth * height * width, depth * height * width, height * width, width, 1]
        ).contiguous()
        let timeConv = Convolution(
          groups: 1, filters: channel, filterSize: [3, 1, 1],
          hint: Hint(
            stride: [2, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
          name: "time_conv")
        let shrunk = timeConv(out)
        let upLayer = k
        let reader: (PythonObject) -> Void = { state_dict in
        }
        readers.append(reader)
        depth = (depth - 1) / 2 + 1
        out = Functional.concat(axis: 2, first, shrunk)
      }
      k += 1
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlockCausal3D(
    prefix: "encoder.mid_block.attentions.0", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock2(out)
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let quantConv = Convolution(groups: 1, filters: 32, filterSize: [1, 1, 1], name: "quant_conv")
  out = quantConv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
  }
  return (reader, Model([x], [out]))
}

graph.withNoGrad {
  let mean = graph.variable(
    Tensor<Float>(
      [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
      ], kind: .CPU, format: .NCHW, shape: [1, 16, 1, 1, 1])
  ).toGPU(0)
  let std = graph.variable(
    Tensor<Float>(
      [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
      ], kind: .CPU, format: .NCHW, shape: [1, 16, 1, 1, 1])
  ).toGPU(0)
  var zTensor = DynamicGraph.Tensor<Float>(from: z)
  zTensor = zTensor .* std + mean
  let (decoderReader, decoder) = DecoderCausal3D(
    channels: [96, 192, 384, 384], numRepeat: 2, startWidth: 128, startHeight: 128, startDepth: 1)
  decoder.compile(inputs: zTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/qwen_image_vae_f32.ckpt", flags: [.readOnly])
  {
    $0.read("decoder", model: decoder)
  }
  let img = decoder(inputs: zTensor)[0].as(of: Float.self).toCPU()
  let startHeight = 128
  let startWidth = 128
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: startWidth * 8 * startHeight * 8)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let (r, g, b) = (img[0, 0, 0, y, x], img[0, 1, 0, y, x], img[0, 2, 0, y, x])
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
