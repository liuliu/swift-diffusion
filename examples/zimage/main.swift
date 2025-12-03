import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

typealias FloatType = Float

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

let diffusers = Python.import("diffusers")

let pipe = diffusers.ZImagePipeline.from_pretrained(
  "Tongyi-MAI/Z-Image-Turbo",
  torch_dtype: torch.bfloat16,
  low_cpu_mem_usage: false
)

print(pipe.transformer)

pipe.to("cuda")

let graph = DynamicGraph()

let prompt =
  "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

let tokenizer = TiktokenTokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/zimage/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/zimage/merges.txt",
  specialTokens: [
    "<|im_end|>": 151645, "<|im_start|>": 151644, "<|endoftext|>": 151643,
    "<|file_sep|>": 151664, "</tool_call>": 151658,
  ])

let promptWithTemplate = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
let positiveTokens = tokenizer.tokenize(text: promptWithTemplate, addSpecialTokens: false)
let text_state_dict = pipe.text_encoder.state_dict()

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
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
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
      32, 2, 64, 2_560
    ).transpose(1, 2).cpu().numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let q_norm_weight = state_dict["\(prefix).self_attn.q_norm.weight"].type(torch.float).view(
      2, 64
    ).transpose(0, 1).cpu()
      .numpy()
    normQ.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_norm_weight)))
    let k_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).view(
      8, 2, 64, 2_560
    ).transpose(1, 2).cpu().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    let k_norm_weight = state_dict["\(prefix).self_attn.k_norm.weight"].type(torch.float).view(
      2, 64
    ).transpose(0, 1).cpu()
      .numpy()
    normK.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_norm_weight)))
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
  return (w1, w2, w3, Model([x], [out], name: name))
}

func TransformerBlock(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(
    prefix: prefix, width: width, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: width, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
    let norm1_weight = state_dict["\(prefix).input_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm1_weight)))
    let norm2_weight = state_dict["\(prefix).post_attention_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm2_weight)))
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
    tokenEmbed.parameters.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vocab)))
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
      prefix: "layers.\(i)", width: width, k: 128, h: heads, hk: 8, b: batchSize,
      t: tokenLength, MLP: MLP)
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
    let norm_weight = state_dict["norm.weight"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_weight)))
  }
  return (Model([tokens, rot], (hiddenStates.map { [$0] } ?? []) + [out]), reader)
}

let _ = graph.withNoGrad {
  /*
  let positiveRotTensor = graph.variable(
    .CPU, .NHWC(1, positiveTokens.0.count, 1, 128), of: Float.self)
  for i in 0..<positiveTokens.0.count {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      positiveRotTensor[0, i, 0, k * 2] = Float(costheta)
      positiveRotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let (transformer, reader) = Transformer(
    Float16.self, vocabularySize: 151_936, maxLength: positiveTokens.0.count, width: 2_560,
    tokenLength: positiveTokens.0.count,
    layers: 36, MLP: 9_728, heads: 32, outputHiddenStates: 34, batchSize: 1
  )
  let positiveTokensTensor = graph.variable(
    .CPU, format: .NHWC, shape: [positiveTokens.0.count], of: Int32.self)
  for i in 0..<positiveTokens.0.count {
    positiveTokensTensor[i] = positiveTokens.0[i]
  }
  let positiveTokensTensorGPU = positiveTokensTensor.toGPU(0)
  let positiveRotTensorGPU = DynamicGraph.Tensor<Float16>(from: positiveRotTensor).toGPU(0)
  transformer.compile(inputs: positiveTokensTensorGPU, positiveRotTensorGPU)
  reader(text_state_dict)
  let positiveLastHiddenStates = transformer(inputs: positiveTokensTensorGPU, positiveRotTensorGPU)[
    0
  ].as(of: Float16.self)
  return positiveLastHiddenStates
  */
}

/*
let image = pipe(
  prompt: prompt,
  height: 1024,
  width: 1024,
  num_inference_steps: 9,  // This actually results in 8 DiT forwards
  guidance_scale: 0.0,  // Guidance should be 0 for the Turbo models
  generator: torch.Generator("cuda").manual_seed(42)
).images[0]
*/

let x = torch.randn([16, 1, 128, 128]).to(torch.bfloat16).cuda()
let txt = torch.randn([64, 2560]).to(torch.bfloat16).cuda()
let t = torch.full([1], 0.2).to(torch.bfloat16).cuda()

let output = pipe.transformer([x], t, [txt])

let state_dict = pipe.transformer.state_dict()

private func MLPEmbedder(channels: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let fc0 = Dense(count: intermediateSize, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, scaleFactor: Float?, name: String = "")
  -> (
    Model, Model, Model, Model
  )
{
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x)
  if let scaleFactor = scaleFactor {
    out = (1 / scaleFactor) * out
  }
  out = out .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out).to(.Float32)
  if let scaleFactor = scaleFactor {
    out = out * scaleFactor
  }
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func ZImageTransformerBlock(
  prefix: String, name: String, k: Int, h: Int, b: Int, t: (Int, Int), scaleFactor: (Float, Float),
  modulation: Bool
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let tEmbed: Input?
  let chunks: [Model.IO]
  let adaLNs: [Model]
  if modulation {
    let t = Input()
    adaLNs = (0..<4).map {
      Dense(count: k * h, name: name.isEmpty ? "ada_ln_\($0)" : "\(name)_ada_ln_\($0)")
    }
    chunks = adaLNs.map { $0(t) }
    tEmbed = t
  } else {
    tEmbed = nil
    adaLNs = []
    chunks = []
  }
  let rot = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: name.isEmpty ? "k" : "\(name)_k")
  let toqueries = Dense(count: k * h, noBias: true, name: name.isEmpty ? "q" : "\(name)_q")
  let tovalues = Dense(count: k * h, noBias: true, name: name.isEmpty ? "v" : "\(name)_v")
  let attentionNorm1 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "attention_norm1" : "\(name)_attention_norm_1")
  var out = attentionNorm1(x)
  if modulation {
    out = (chunks[0] + 1).to(of: out) .* out
  }
  out = out.to(.Float16)
  var keys = tokeys(out).reshaped([b, t.0, h, k])
  var queries = toqueries(out).reshaped([b, t.0, h, k])
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_k" : "\(name)_norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: name.isEmpty ? "norm_q" : "\(name)_norm_q")
  queries = normQ(queries)
  var values = tovalues(out)
  if scaleFactor.0 > 1 {
    values = (1 / scaleFactor.0) * values
  }
  values = values.reshaped([b, t.0, h, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
    queries, keys, values
  )
  let xIn: Model.IO
  if t.0 > t.1 {
    xIn = x.reshaped([b, t.1, h * k], offset: [0, 0, 0], strides: [t.0 * h * k, h * k, 1])
    out = out.reshaped([b, t.1, h * k], offset: [0, 0, 0], strides: [t.0 * h * k, h * k, 1])
  } else {
    xIn = x
    out = out.reshaped([b, t.0, h * k])
  }
  let unifyheads = Dense(count: k * h, noBias: true, name: name.isEmpty ? "o" : "\(name)_o")
  out = unifyheads(out).to(of: xIn)
  if scaleFactor.0 > 1 {
    out = scaleFactor.0 * out
  }
  let attentionNorm2 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "attention_norm2" : "\(name)_attention_norm2")
  out = attentionNorm2(out)
  if modulation {
    out = chunks[1].tanh().to(of: out) .* out
  }
  out = xIn + out
  let (w1, w2, w3, ffn) = FeedForward(
    hiddenSize: h * k, intermediateSize: 10_240, scaleFactor: scaleFactor.1,
    name: name.isEmpty ? "ffn" : "\(name)_ffn")
  let feedForwardNorm1 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "ffn_norm1" : "\(name)_ffn_norm1")
  let feedForwardNorm2 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "ffn_norm2" : "\(name)_ffn_norm2")
  let residual = out
  out = feedForwardNorm1(out)
  if modulation {
    out = (chunks[2] + 1).to(of: out) .* out
  }
  out = out.to(.Float16)
  out = feedForwardNorm2(ffn(out))  // Already converted to Float32.
  if modulation {
    out = chunks[3].tanh().to(of: out) .* out
  }
  out = residual + out
  let reader: (PythonObject) -> Void = { state_dict in
    let q_weight = state_dict["\(prefix).attention.to_q.weight"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let k_weight = state_dict["\(prefix).attention.to_k.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    let norm_q_weight = state_dict["\(prefix).attention.norm_q.weight"].type(torch.float).cpu()
      .numpy()
    normQ.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_q_weight)))
    let norm_k_weight = state_dict["\(prefix).attention.norm_k.weight"].type(torch.float).cpu()
      .numpy()
    normK.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_k_weight)))
    let v_weight = state_dict["\(prefix).attention.to_v.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: v_weight)))
    let proj_weight = state_dict["\(prefix).attention.to_out.0.weight"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_weight)))
    let attention_norm1_weight = state_dict["\(prefix).attention_norm1.weight"].type(torch.float)
      .cpu().numpy()
    attentionNorm1.weight.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: attention_norm1_weight)))
    let attention_norm2_weight = state_dict["\(prefix).attention_norm2.weight"].type(torch.float)
      .cpu().numpy()
    attentionNorm2.weight.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: attention_norm2_weight)))
    let w1_weight = state_dict["\(prefix).feed_forward.w1.weight"].type(torch.float).cpu()
      .numpy()
    w1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w1_weight)))
    let w2_weight = state_dict["\(prefix).feed_forward.w2.weight"].type(torch.float).cpu()
      .numpy()
    w2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w2_weight)))
    let w3_weight = state_dict["\(prefix).feed_forward.w3.weight"].type(torch.float).cpu()
      .numpy()
    w3.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w3_weight)))
    let ffn_norm1_weight = state_dict["\(prefix).ffn_norm1.weight"].type(torch.float).cpu().numpy()
    feedForwardNorm1.weight.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: ffn_norm1_weight)))
    let ffn_norm2_weight = state_dict["\(prefix).ffn_norm2.weight"].type(torch.float)
      .cpu().numpy()
    feedForwardNorm2.weight.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: ffn_norm2_weight)))
    if modulation {
      let adaLN_modulation_0_weight = state_dict["\(prefix).adaLN_modulation.0.weight"].type(
        torch.float
      ).cpu().numpy()
      let adaLN_modulation_0_bias = state_dict["\(prefix).adaLN_modulation.0.bias"].type(
        torch.float
      ).cpu().numpy()
      for (i, adaLN) in adaLNs.enumerated() {
        adaLN.weight.copy(
          from: Tensor<Float16>(
            from: try! Tensor<Float>(
              numpy: adaLN_modulation_0_weight[(k * h * i)..<(k * h * (i + 1)), ...])))
        adaLN.bias.copy(
          from: Tensor<Float16>(
            from: try! Tensor<Float>(
              numpy: adaLN_modulation_0_bias[(k * h * i)..<(k * h * (i + 1))])))
      }
    }
  }
  return (Model([x, rot] + (tEmbed.map { [$0] } ?? []), [out]), reader)
}

func ZImage(height: Int, width: Int, textLength: Int, layers: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let xRot = Input()
  let txt = Input()
  let txtRot = Input()
  let t = Input()
  let imgIn = Dense(count: 3_840, name: "x_embedder")
  let h = height / 2
  let w = width / 2
  var xOut = imgIn(
    x.reshaped([1, 16, h, 2, w, 2]).permuted(0, 2, 4, 3, 5, 1).contiguous()
      .reshaped([1, h * w, 2 * 2 * 16], format: .NHWC)
  ).to(.Float32)
  let txtNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "cap_norm")
  let txtIn = Dense(count: 3_840, name: "cap_embedder")
  var txtOut = txtIn(txtNorm(txt)).to(.Float32)
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(
    channels: 256, intermediateSize: 1024, name: "t")
  let tOut = timeIn(t)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<2 {
    let (block, reader) = ZImageTransformerBlock(
      prefix: "context_refiner.\(i)", name: "context_refiner", k: 128, h: 30, b: 1,
      t: (textLength, textLength), scaleFactor: (2, 2), modulation: false)
    txtOut = block(txtOut, txtRot)
    readers.append(reader)
  }
  for i in 0..<2 {
    let (block, reader) = ZImageTransformerBlock(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", k: 128, h: 30, b: 1, t: (h * w, h * w),
      scaleFactor: (4, 8), modulation: true)
    xOut = block(xOut, xRot, tOut)
    readers.append(reader)
  }
  var out = Functional.concat(axis: 1, xOut, txtOut)
  let rot = Functional.concat(axis: 1, xRot, txtRot)
  for i in 0..<layers {
    let (block, reader) = ZImageTransformerBlock(
      prefix: "layers.\(i)", name: "", k: 128, h: 30, b: 1,
      t: (h * w + textLength, i == layers - 1 ? h * w : h * w + textLength), scaleFactor: (4, 16),
      modulation: true)
    out = block(out, rot, tOut)
    readers.append(reader)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let scale = Dense(count: 3840, name: "ada_ln_final")
  let projOut = Dense(count: 2 * 2 * 16, name: "linear_final")
  out = (1 + scale(tOut.swish())) .* normFinal(out).to(.Float16)
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_weight = state_dict["all_x_embedder.2-1.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_in_bias = state_dict["all_x_embedder.2-1.bias"].to(torch.float)
      .cpu().numpy()
    imgIn.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_in_weight)))
    imgIn.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_in_bias)))
    let cap_embedder_0_weight = state_dict["cap_embedder.0.weight"].to(torch.float).cpu().numpy()
    txtNorm.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cap_embedder_0_weight)))
    let cap_embedder_1_weight = state_dict["cap_embedder.1.weight"].to(torch.float).cpu().numpy()
    let cap_embedder_1_bias = state_dict["cap_embedder.1.bias"].to(torch.float)
      .cpu().numpy()
    txtIn.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cap_embedder_1_weight)))
    txtIn.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cap_embedder_1_bias)))
    let t_embedder_mlp_0_weight = state_dict["t_embedder.mlp.0.weight"].to(torch.float).cpu()
      .numpy()
    let t_embedder_mlp_0_bias = state_dict["t_embedder.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    timeInMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight)))
    timeInMlp0.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias)))
    let t_embedder_mlp_2_weight = state_dict["t_embedder.mlp.2.weight"].to(torch.float).cpu()
      .numpy()
    let t_embedder_mlp_2_bias = state_dict["t_embedder.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    timeInMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight)))
    timeInMlp2.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias)))
    for reader in readers {
      reader(state_dict)
    }
    let adaLN_modulation_1_weight = state_dict["all_final_layer.2-1.adaLN_modulation.1.weight"].to(
      torch.float
    ).cpu().numpy()
    let adaLN_modulation_1_bias = state_dict["all_final_layer.2-1.adaLN_modulation.1.bias"].to(
      torch.float
    ).cpu().numpy()
    scale.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: adaLN_modulation_1_weight)))
    scale.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: adaLN_modulation_1_bias)))
    let linear_weight = state_dict["all_final_layer.2-1.linear.weight"].to(torch.float).cpu()
      .numpy()
    let linear_bias = state_dict["all_final_layer.2-1.linear.bias"].to(torch.float).cpu().numpy()
    projOut.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: linear_weight)))
    projOut.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: linear_bias)))
  }
  return (Model([x, xRot, txt, txtRot, t], [out, txtOut]), reader)
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

graph.withNoGrad {
  let xTensor = graph.variable(
    Tensor<Float16>(
      from: try! Tensor<Float>(numpy: x.view(1, 16, 128, 128).to(torch.float).cpu().numpy())
    ).toGPU(1))
  let txtTensor = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: txt.to(torch.float).cpu().numpy())).toGPU(1)
  ).reshaped(.HWC(1, 64, 2560))
  let (dit, reader) = ZImage(height: 128, width: 128, textLength: 64, layers: 30)
  let txtRotTensor = graph.variable(.CPU, .NHWC(1, 64, 1, 128), of: Float.self)
  for i in 0..<64 {
    for k in 0..<16 {
      let theta = Double(i + 1) * 1.0 / pow(256, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      txtRotTensor[0, i, 0, k * 2] = Float(costheta)
      txtRotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<24 {
      let theta = Double(0) * 1.0 / pow(256, Double(k) / 24)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      txtRotTensor[0, i, 0, (k + 16) * 2] = Float(costheta)
      txtRotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<24 {
      let theta = Double(0) * 1.0 / pow(256, Double(k) / 24)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      txtRotTensor[0, i, 0, (k + 16 + 24) * 2] = Float(costheta)
      txtRotTensor[0, i, 0, (k + 16 + 24) * 2 + 1] = Float(sintheta)
    }
  }
  let xRotTensor = graph.variable(.CPU, .NHWC(1, 4096, 1, 128), of: Float.self)
  for y in 0..<64 {
    for x in 0..<64 {
      let i = y * 64 + x
      for k in 0..<16 {
        let theta = Double(64 + 1) * 1.0 / pow(256, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        xRotTensor[0, i, 0, k * 2] = Float(costheta)
        xRotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<24 {
        let theta = Double(y) * 1.0 / pow(256, Double(k) / 24)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        xRotTensor[0, i, 0, (k + 16) * 2] = Float(costheta)
        xRotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<24 {
        let theta = Double(x) * 1.0 / pow(256, Double(k) / 24)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        xRotTensor[0, i, 0, (k + 16 + 24) * 2] = Float(costheta)
        xRotTensor[0, i, 0, (k + 16 + 24) * 2 + 1] = Float(sintheta)
      }
    }
  }
  let tTensor = graph.variable(
    Tensor<Float16>(
      from: timeEmbedding(timesteps: 200, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(1))
  let xRotTensorGPU = DynamicGraph.Tensor<Float16>(from: xRotTensor).toGPU(1)
  let txtRotTensorGPU = DynamicGraph.Tensor<Float16>(from: txtRotTensor).toGPU(1)
  dit.maxConcurrency = .limit(4)
  dit.compile(inputs: xTensor, xRotTensorGPU, txtTensor, txtRotTensorGPU, tTensor)
  print(state_dict.keys())
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, xRotTensorGPU, txtTensor, txtRotTensorGPU, tTensor))
}
