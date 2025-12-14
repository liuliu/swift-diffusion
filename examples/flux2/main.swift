import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

typealias FloatType = Float16

let torch = Python.import("torch")
let PIL = Python.import("PIL")

torch.set_grad_enabled(false)

let torch_device = torch.device("cuda")

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let flux2_util = Python.import("flux2.util")

var mistral = flux2_util.load_mistral_small_embedder()
let model = flux2_util.load_flow_model(model_name: "flux.2-dev", debug_mode: false, device: "cpu")

// print(mistral)

let prompt =
  "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture"

// print(mistral([prompt]))

mistral = mistral.cpu()

let ae = flux2_util.load_ae(model_name: "flux.2-dev").to(torch.float32)

print(ae)

let tokenizer = TiktokenTokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/flux2/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/flux2/merges.txt",
  specialTokens: [
    "[SYSTEM_PROMPT]": 17, "[/SYSTEM_PROMPT]": 18, "[INST]": 3, "[/INST]": 4, "<unk>": 0, "<s>": 1,
    "</s>": 2,
  ], unknownToken: "<unk>", startToken: "<s>", endToken: "</s>")

let promptWithTemplate =
  "<s>[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]\(prompt)[/INST]"
let positiveTokens = tokenizer.tokenize(text: promptWithTemplate, addSpecialTokens: false)

let text_state_dict = mistral.model.model.language_model.state_dict()

func SelfAttention(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
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
  let unifyheads = Dense(count: width, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    // The rotary in Qwen is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    let q_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).view(
      32, 2, 64, 5_120
    ).transpose(1, 2).cpu().numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let k_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).view(
      8, 2, 64, 5_120
    ).transpose(1, 2).cpu().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    let v_weight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: v_weight)))
    let proj_weight = state_dict["\(prefix).self_attn.o_proj.weight"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_weight)))
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
  return (w1, w2, w3, Model([x], [out]))
}

func TransformerBlock(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x).to(.Float16)
  let (attention, attnReader) = SelfAttention(
    prefix: prefix, width: width, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot).to(of: x) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "post_attention_layernorm")
  out = norm2(out).to(.Float16)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: width, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out).to(of: residual)
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
    let norm1_weight = state_dict["\(prefix).input_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm1.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm1_weight)))
    let norm2_weight = state_dict["\(prefix).post_attention_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm2.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm2_weight)))
    let w1_weight = state_dict["\(prefix).mlp.gate_proj.weight"].type(torch.float).cpu()
      .numpy()
    w1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w1_weight)))
    let w2_weight = state_dict["\(prefix).mlp.down_proj.weight"].type(torch.float).cpu()
      .numpy()
    w2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w2_weight)))
    let w3_weight = state_dict["\(prefix).mlp.up_proj.weight"].type(torch.float).cpu()
      .numpy()
    w3.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w3_weight)))
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
    let vocab = state_dict["embed_tokens.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: Tensor<T>(from: try! Tensor<Float>(numpy: vocab)))
  }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: [Int], batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embedReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens).to(.Float32)
  var readers = [(PythonObject) -> Void]()
  var hiddenStates = [Model.IO]()
  for i in 0..<layers {
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", width: width, k: 128, h: heads, hk: 8, b: batchSize,
      t: tokenLength, MLP: MLP)
    out = layer(out, rot)
    readers.append(reader)
    if outputHiddenStates.contains(i) {
      hiddenStates.append(out.to(.Float16))
    }
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out).to(.Float16)
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_weight = state_dict["norm.weight"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm_weight)))
  }
  return (Model([tokens, rot], hiddenStates + [out]), reader)
}

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

let _ = graph.withNoGrad {
  /*
  let positiveRotTensor = graph.variable(
    .CPU, .NHWC(1, positiveTokens.0.count, 1, 128), of: Float.self)
  let endAligned = 512 - positiveTokens.0.count
  for i in 0..<positiveTokens.0.count {
    for k in 0..<64 {
      let theta = Double(endAligned + i) * 1.0 / pow(1_000_000_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      positiveRotTensor[0, i, 0, k * 2] = Float(costheta)
      positiveRotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let (transformer, reader) = Transformer(
    BFloat16.self, vocabularySize: 131_072, maxLength: positiveTokens.0.count, width: 5_120,
    tokenLength: positiveTokens.0.count,
    layers: 40, MLP: 32_768, heads: 32, outputHiddenStates: [9, 19, 29], batchSize: 1
  )
  let positiveTokensTensor = graph.variable(
    .CPU, format: .NHWC, shape: [positiveTokens.0.count], of: Int32.self)
  for i in 0..<positiveTokens.0.count {
    positiveTokensTensor[i] = positiveTokens.0[i]
  }
  let positiveTokensTensorGPU = positiveTokensTensor.toGPU(1)
  let positiveRotTensorGPU = DynamicGraph.Tensor<Float16>(from: positiveRotTensor).toGPU(1)
  transformer.compile(inputs: positiveTokensTensorGPU, positiveRotTensorGPU)
  reader(text_state_dict)
  let positiveLastHiddenStates = transformer(inputs: positiveTokensTensorGPU, positiveRotTensorGPU)
    .map { $0.as(of: Float16.self) }
  debugPrint(positiveLastHiddenStates)
  graph.openStore("/fast/Data/mistral_small_3.2_24b_instruct_2506_f16.ckpt") {
    $0.write("text_model", model: transformer)
  }
  return positiveLastHiddenStates
  */
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
    let norm1_weight = state_dict["\(prefix).norm1.weight"].to(torch.float).cpu().numpy()
    let norm1_bias = state_dict["\(prefix).norm1.bias"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let conv1_weight = state_dict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2_bias = state_dict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).nin_shortcut.weight"].to(torch.float).cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).nin_shortcut.bias"].to(torch.float).cpu()
        .numpy()
      ninShortcut.weight.copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
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
    let norm_weight = state_dict["\(prefix).norm.weight"].to(torch.float).cpu().numpy()
    let norm_bias = state_dict["\(prefix).norm.bias"].to(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: norm_bias))
    let k_weight = state_dict["\(prefix).k.weight"].to(torch.float).cpu().numpy()
    let k_bias = state_dict["\(prefix).k.bias"].to(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["\(prefix).q.weight"].to(torch.float).cpu().numpy()
    let q_bias = state_dict["\(prefix).q.bias"].to(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["\(prefix).v.weight"].to(torch.float).cpu().numpy()
    let v_bias = state_dict["\(prefix).v.bias"].to(torch.float).cpu().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["\(prefix).proj_out.weight"].to(torch.float).cpu().numpy()
    let proj_out_bias = state_dict["\(prefix).proj_out.bias"].to(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
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
        let conv_weight = state_dict["encoder.down.\(downLayer).downsample.conv.weight"].to(
          torch.float
        ).cpu()
          .numpy()
        let conv_bias = state_dict["encoder.down.\(downLayer).downsample.conv.bias"].to(torch.float)
          .cpu().numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth, height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "encoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let quantConv = Convolution(
    groups: 1, filters: 64, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
  out = quantConv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["encoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["encoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    let norm_out_weight = state_dict["encoder.norm_out.weight"].to(torch.float).cpu().numpy()
    let norm_out_bias = state_dict["encoder.norm_out.bias"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["encoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["encoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
    let quant_conv_weight = state_dict["encoder.quant_conv.weight"].to(torch.float).cpu().numpy()
    let quant_conv_bias = state_dict["encoder.quant_conv.bias"].to(torch.float).cpu().numpy()
    quantConv.weight.copy(from: try! Tensor<Float>(numpy: quant_conv_weight))
    quantConv.bias.copy(from: try! Tensor<Float>(numpy: quant_conv_bias))
  }
  return (reader, Model([x], [out]))
}

func Decoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 32, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
  var out = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convIn(out)
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
        let conv_weight = state_dict["decoder.up.\(upLayer).upsample.conv.weight"].to(torch.float)
          .cpu().numpy()
        let conv_bias = state_dict["decoder.up.\(upLayer).upsample.conv.bias"].to(torch.float).cpu()
          .numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
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
    let post_quant_conv_weight = state_dict["decoder.post_quant_conv.weight"].to(torch.float).cpu()
      .numpy()
    let post_quant_conv_bias = state_dict["decoder.post_quant_conv.bias"].to(torch.float).cpu()
      .numpy()
    postQuantConv.weight.copy(from: try! Tensor<Float>(numpy: post_quant_conv_weight))
    postQuantConv.bias.copy(from: try! Tensor<Float>(numpy: post_quant_conv_bias))
    let conv_in_weight = state_dict["decoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["decoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["decoder.norm_out.weight"].to(torch.float).cpu().numpy()
    let norm_out_bias = state_dict["decoder.norm_out.bias"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["decoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["decoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

let z = torch.randn([1, 32, 128, 128]).to(torch.float).cuda()
let image = ae.decoder(z)
print(image)
print(ae.encoder(image))

let vae_state_dict = ae.state_dict()

graph.withNoGrad {
  var zTensor = graph.variable(try! Tensor<Float>(numpy: z.to(torch.float).cpu().numpy())).toGPU(2)
  // Already processed out.
  let (decoderReader, decoder) = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  decoder.compile(inputs: zTensor)
  decoderReader(vae_state_dict)
  let decodedImage = decoder(inputs: zTensor)[0].as(of: Float.self)
  debugPrint(decodedImage)
  let imageTensor = graph.variable(try! Tensor<Float>(numpy: image.to(torch.float).cpu().numpy()))
    .toGPU(2)
  let (encoderReader, encoder) = Encoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  encoder.compile(inputs: imageTensor)
  encoderReader(vae_state_dict)
  let ae = encoder(inputs: imageTensor)[0].as(of: Float.self)
  debugPrint(ae)
  graph.openStore("/home/liu/workspace/swift-diffusion/flux_2_vae_f32.ckpt") {
    $0.write("decoder", model: decoder)
    $0.write("encoder", model: encoder)
  }
}

exit(0)

print(model)

let x = torch.randn([1, 4096, 128]).to(torch.bfloat16).cuda()
let txt = torch.randn([1, 512, 15360]).to(torch.bfloat16).cuda() * 0.01
var img_ids = torch.zeros([64, 64, 4])
img_ids[..., ..., 1] = img_ids[..., ..., 1] + torch.arange(64)[..., Python.None]
img_ids[..., ..., 2] = img_ids[..., ..., 2] + torch.arange(64)[Python.None, ...]
img_ids = img_ids.reshape([1, 4096, 4]).cuda()
var txt_ids = torch.zeros([512, 4])
txt_ids[..., 3] = txt_ids[..., 3] + torch.arange(512)
txt_ids = txt_ids.reshape([1, 512, 4]).cuda()
let t = torch.full([1], 1).to(torch.bfloat16).cuda()
let guidance = torch.full([1], 4.0).to(torch.bfloat16).cuda()

let output = model(x, img_ids, t, txt, txt_ids, guidance)

let state_dict = model.cpu().state_dict()

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
  let fc0 = Dense(count: channels, noBias: true, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, noBias: true, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context).to(.Float16) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, noBias: true, name: "c_k")
  let contextToQueries = Dense(count: k * h, noBias: true, name: "c_q")
  let contextToValues = Dense(count: k * h, noBias: true, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
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
  /*
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
  */
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
    contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: context)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 3, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + (contextChunks[5]
      .* contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3]))
      .to(of: contextOut)
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xW1, xW2, xW3, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 3, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut + (xChunks[5] .* xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])).to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
    let txt_attn_qkv_weight = state_dict["\(prefix).txt_attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[..<(k * h), ...]))
    )
    contextToQueries.weight.to(.unifiedMemory)
    contextToKeys.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[(k * h)..<(2 * k * h), ...])))
    contextToKeys.weight.to(.unifiedMemory)
    contextToValues.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...]))
    )
    contextToValues.weight.to(.unifiedMemory)
    let txt_attn_key_norm_scale = state_dict["\(prefix).txt_attn.norm.key_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normAddedK.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt_attn_key_norm_scale)))
    let txt_attn_query_norm_scale = state_dict["\(prefix).txt_attn.norm.query_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normAddedQ.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt_attn_query_norm_scale)))
    let img_attn_qkv_weight = state_dict["\(prefix).img_attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: img_attn_qkv_weight[..<(k * h), ...]))
    )
    xToQueries.weight.to(.unifiedMemory)
    xToKeys.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: img_attn_qkv_weight[(k * h)..<(2 * k * h), ...])))
    xToKeys.weight.to(.unifiedMemory)
    xToValues.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: img_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...])))
    xToValues.weight.to(.unifiedMemory)
    let img_attn_key_norm_scale = state_dict["\(prefix).img_attn.norm.key_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normK.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: img_attn_key_norm_scale)))
    normK.weight.to(.unifiedMemory)
    let img_attn_query_norm_scale = state_dict["\(prefix).img_attn.norm.query_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: img_attn_query_norm_scale)))
    normQ.weight.to(.unifiedMemory)
    if let contextUnifyheads = contextUnifyheads {
      let attn_to_add_out_weight = state_dict["\(prefix).txt_attn.proj.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: attn_to_add_out_weight)))
      contextUnifyheads.weight.to(.unifiedMemory)
    }
    let attn_to_out_0_weight = state_dict["\(prefix).img_attn.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: attn_to_out_0_weight)))
    xUnifyheads.weight.to(.unifiedMemory)
    if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
      let ff_context_linear_1_weight = state_dict["\(prefix).txt_mlp.0.weight"].to(
        torch.float
      ).cpu().numpy()
      contextW1.weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: ff_context_linear_1_weight[..<(k * h * 3), ...])))
      contextW1.weight.to(.unifiedMemory)
      contextW3.weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(
            numpy: ff_context_linear_1_weight[(k * h * 3)..<(k * h * 6), ...])))
      contextW3.weight.to(.unifiedMemory)
      let ff_context_out_projection_weight = state_dict[
        "\(prefix).txt_mlp.2.weight"
      ].to(
        torch.float
      ).cpu().numpy()
      contextW2.weight.copy(
        from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: ff_context_out_projection_weight)))
      contextW2.weight.to(.unifiedMemory)
    }
    let ff_linear_1_weight = state_dict["\(prefix).img_mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    xW1.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: ff_linear_1_weight[..<(k * h * 3), ...])))
    xW1.weight.to(.unifiedMemory)
    xW3.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: ff_linear_1_weight[(k * h * 3)..<(k * h * 6), ...])))
    xW3.weight.to(.unifiedMemory)
    let ff_out_projection_weight = state_dict["\(prefix).img_mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    xW2.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: ff_out_projection_weight)))
    xW2.weight.to(.unifiedMemory)
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, rot] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, rot] + contextChunks + xChunks, [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
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
  /*
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
  */
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xW1 = Dense(count: k * h * 3, noBias: true, name: "x_w1")
  let xW3 = Dense(count: k * h * 3, noBias: true, name: "x_w3")
  let xW2 = Dense(count: k * h, noBias: true, name: "x_w2")
  out = xUnifyheads(out) + xW2(xW3(xOut) .* xW1(xOut).swish())
  out = xIn + (xChunks[2] .* out).to(of: xIn)
  let reader: (PythonObject) -> Void = { state_dict in
    let linear1_weight = state_dict["\(prefix).linear1.weight"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: linear1_weight[..<(k * h), ...])))
    xToQueries.weight.to(.unifiedMemory)
    xToKeys.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear1_weight[(k * h)..<(2 * k * h), ...])))
    xToKeys.weight.to(.unifiedMemory)
    xToValues.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear1_weight[(2 * k * h)..<(3 * k * h), ...])))
    xToValues.weight.to(.unifiedMemory)
    let key_norm_scale = state_dict["\(prefix).norm.key_norm.scale"].to(torch.float).cpu().numpy()
    normK.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: key_norm_scale)))
    normK.weight.to(.unifiedMemory)
    let query_norm_scale = state_dict["\(prefix).norm.query_norm.scale"].to(torch.float).cpu()
      .numpy()
    normQ.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: query_norm_scale)))
    normQ.weight.to(.unifiedMemory)
    xW1.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear1_weight[(3 * k * h)..<(6 * k * h), ...])))
    xW1.weight.to(.unifiedMemory)
    xW3.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear1_weight[(6 * k * h)..<(9 * k * h), ...])))
    xW3.weight.to(.unifiedMemory)
    let linear2_weight = state_dict["\(prefix).linear2.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: linear2_weight[..., 0..<(k * h)])))
    xUnifyheads.weight.to(.unifiedMemory)
    xW2.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear2_weight[..., (k * h)..<(k * h * 4)])))
    xW2.weight.to(.unifiedMemory)
  }
  return (reader, Model([x, rot] + xChunks, [out]))
}

func Flux2(b: Int, h: Int, w: Int, guidanceEmbed: Bool) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let contextIn = Input()
  let rot = Input()
  let t = Input()
  let xEmbedder = Dense(count: 6144, noBias: true, name: "x_embedder")
  var out = xEmbedder(x).to(.Float32)
  let contextEmbedder = Dense(count: 6144, noBias: true, name: "context_embedder")
  var context = contextEmbedder(contextIn).to(.Float32)
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: 6144, name: "t")
  let (gMlp0, gMlp2, gEmbedder) = MLPEmbedder(channels: 6144, name: "guidance")
  let g = Input()
  var vec = tEmbedder(t)
  vec = vec + gEmbedder(g)
  vec = vec.swish()
  let xAdaLNs = (0..<6).map { Dense(count: 6144, noBias: true, name: "x_ada_ln_\($0)") }
  let contextAdaLNs = (0..<6).map { Dense(count: 6144, noBias: true, name: "context_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(vec) }
  var contextChunks = contextAdaLNs.map { $0(vec) }
  xChunks[1] = 1 + xChunks[1]
  xChunks[4] = 1 + xChunks[4]
  contextChunks[1] = 1 + contextChunks[1]
  contextChunks[4] = 1 + contextChunks[4]
  var readers = [(PythonObject) -> Void]()
  for i in 0..<8 {
    let (reader, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: 48, b: b, t: 512, hw: h * w,
      contextBlockPreOnly: false)
    let blockOut = block([context, out, rot] + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    readers.append(reader)
  }
  let singleAdaLNs = (0..<3).map { Dense(count: 6144, noBias: true, name: "single_ada_ln_\($0)") }
  var singleChunks = singleAdaLNs.map { $0(vec) }
  singleChunks[1] = 1 + singleChunks[1]
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<48 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: 48, b: b, t: 512, hw: h * w,
      contextBlockPreOnly: i == 47)
    out = block([out, rot] + singleChunks)
    readers.append(reader)
  }
  let scale = Dense(count: 6144, noBias: true, name: "ada_ln_0")
  let shift = Dense(count: 6144, noBias: true, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 32, noBias: true, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_weight = state_dict["img_in.weight"].to(
      torch.float
    ).cpu().numpy()
    xEmbedder.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: img_in_weight)))
    xEmbedder.weight.to(.unifiedMemory)
    let txt_in_weight = state_dict["txt_in.weight"].to(
      torch.float
    ).cpu().numpy()
    contextEmbedder.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt_in_weight)))
    contextEmbedder.weight.to(.unifiedMemory)
    let t_embedder_mlp_0_weight = state_dict["time_in.in_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    tMlp0.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight)))
    tMlp0.weight.to(.unifiedMemory)
    let t_embedder_mlp_2_weight = state_dict["time_in.out_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    tMlp2.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight)))
    tMlp2.weight.to(.unifiedMemory)
    let guidance_embedder_mlp_0_weight = state_dict["guidance_in.in_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    gMlp0.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: guidance_embedder_mlp_0_weight)))
    gMlp0.weight.to(.unifiedMemory)
    let guidance_embedder_mlp_2_weight = state_dict["guidance_in.out_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    gMlp2.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: guidance_embedder_mlp_2_weight)))
    gMlp2.weight.to(.unifiedMemory)
    let double_stream_modulation_img_lin_weight = state_dict[
      "double_stream_modulation_img.lin.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let double_stream_modulation_txt_lin_weight = state_dict[
      "double_stream_modulation_txt.lin.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(
            numpy: double_stream_modulation_img_lin_weight[(6144 * i)..<(6144 * (i + 1)), ...])))
      xAdaLNs[i].weight.to(.unifiedMemory)
      contextAdaLNs[i].weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(
            numpy: double_stream_modulation_txt_lin_weight[(6144 * i)..<(6144 * (i + 1)), ...])))
      contextAdaLNs[i].weight.to(.unifiedMemory)
    }
    let single_stream_modulation_lin_weight = state_dict[
      "single_stream_modulation.lin.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<3 {
      singleAdaLNs[i].weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(
            numpy: single_stream_modulation_lin_weight[(6144 * i)..<(6144 * (i + 1)), ...])))
      singleAdaLNs[i].weight.to(.unifiedMemory)
    }
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_adaLN_modulation_weight = state_dict["final_layer.adaLN_modulation.1.weight"]
      .to(torch.float).cpu().numpy()
    shift.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: final_layer_adaLN_modulation_weight[0..<6144, ...])))
    shift.weight.to(.unifiedMemory)
    scale.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: final_layer_adaLN_modulation_weight[6144..<(6144 * 2), ...])
      ))
    scale.weight.to(.unifiedMemory)
    let proj_out_weight = state_dict["final_layer.linear.weight"].to(
      torch.float
    ).cpu().numpy()
    projOut.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: proj_out_weight)))
    projOut.weight.to(.unifiedMemory)
  }
  return (reader, Model([x, contextIn, rot, t, g], [out]))
}

let (reader, dit) = Flux2(b: 1, h: 64, w: 64, guidanceEmbed: true)

graph.withNoGrad {
  let xTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy())).toGPU(2)
  ).reshaped(.HWC(1, 4096, 128))
  let contextTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt.to(torch.float).cpu().numpy())).toGPU(2)
  ).reshaped(.HWC(1, 512, 15360))
  let tTensor = graph.variable(
    Tensor<FloatType>(
      from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(2))
  let gTensor = graph.variable(
    Tensor<FloatType>(
      from: timeEmbedding(timesteps: 4000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(2))
  let rotTensor = graph.variable(.CPU, .NHWC(1, 4096 + 512, 1, 128), of: Float.self)
  for i in 0..<512 {
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 16) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 16 * 2) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 16 * 2) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = Double(i) * 1.0 / pow(2_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 16 * 3) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 16 * 3) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<64 {
    for x in 0..<64 {
      let i = y * 64 + x + 512
      for k in 0..<16 {
        let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(y) * 1.0 / pow(2_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 16) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(x) * 1.0 / pow(2_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 16 * 2) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 16 * 2) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 16 * 3) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 16 * 3) * 2 + 1] = Float(sintheta)
      }
    }
  }
  let rotTensorGPU = DynamicGraph.Tensor<FloatType>(from: rotTensor).toGPU(2)
  dit.maxConcurrency = .limit(1)
  dit.compile(inputs: xTensor, contextTensor, rotTensorGPU, tTensor, gTensor)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, contextTensor, rotTensorGPU, tTensor, gTensor))
  /*
  graph.openStore("/fast/Data/flux_2_dev_f16.ckpt") {
    $0.write("dit", model: dit)
  }
  */
}
