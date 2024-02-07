import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let torch = Python.import("torch")
let einops = Python.import("einops")
let numpy = Python.import("numpy")
let Image = Python.import("PIL.Image")

let device = torch.device("cuda")

let raw_image = Image.open("/home/liu/workspace/swift-diffusion/kandinsky-512.png").convert("RGB")

let transformers = Python.import("transformers")

torch.set_grad_enabled(false)
let model_id = "vikhyatk/moondream1"
let model = transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code: true)
let tokenizer = transformers.CodeGenTokenizerFast.from_pretrained(model_id)

let enc_image = model.encode_image(raw_image)
print(enc_image)

print(
  model.answer_question(
    enc_image,
    "Describe this image and its style in a very detailed manner, follow the format of describing: what, who, where, when, how. You don't need to fill in all if they are irrelevant. Please remove What, Who, Where, When, How prefixes and make it one paragraph.",
    tokenizer))

var input_ids = tokenizer("hello world ", return_tensors: "pt", add_special_tokens: false).input_ids
input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), input_ids], dim: 1)
// print(input_ids)
let input_embeds = model.text_model.text_emb(input_ids)
/*
print(
  model.text_model.model.generate(
    inputs_embeds: input_embeds, bos_token_id: tokenizer.bos_token_id,
    pad_token_id: tokenizer.pad_token_id, max_new_tokens: 1))
*/
print(input_embeds)

let vision_encoder_state_dict = model.vision_encoder.state_dict()
let text_model_state_dict = model.text_model.state_dict()

func SigLIPSelfAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, tokeys, tovalues, unifyheads, Model([x], [out]))
}

func SigLIPResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int, MLP: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2])
  let (toqueries, tokeys, tovalues, unifyheads, attention) = SigLIPSelfAttention(
    k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [1])
  let fc = Dense(count: MLP)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu().numpy()
    ln1.weight.copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    ln1.bias.copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let in_proj_weight = state_dict["\(prefix).attn.qkv.weight"].type(torch.float).cpu().numpy()
    let in_proj_bias = state_dict["\(prefix).attn.qkv.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[..<(k * h), ...]))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: in_proj_bias[..<(k * h)]))
    tokeys.weight.copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(k * h)..<(2 * k * h), ...]))
    tokeys.bias.copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(k * h)..<(2 * k * h)]))
    tovalues.weight.copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(2 * k * h)..., ...]))
    tovalues.bias.copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(2 * k * h)...]))
    let out_proj_weight = state_dict["\(prefix).attn.proj.weight"].type(torch.float).cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).attn.proj.bias"].type(torch.float).cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: out_proj_bias))
    let ln_2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu().numpy()
    ln2.weight.copy(from: try! Tensor<Float>(numpy: ln_2_weight))
    ln2.bias.copy(from: try! Tensor<Float>(numpy: ln_2_bias))
    let c_fc_weight = state_dict["\(prefix).mlp.fc1.weight"].type(torch.float).cpu().numpy()
    let c_fc_bias = state_dict["\(prefix).mlp.fc1.bias"].type(torch.float).cpu().numpy()
    fc.weight.copy(from: try! Tensor<Float>(numpy: c_fc_weight))
    fc.bias.copy(from: try! Tensor<Float>(numpy: c_fc_bias))
    let c_proj_weight = state_dict["\(prefix).mlp.fc2.weight"].type(torch.float).cpu().numpy()
    let c_proj_bias = state_dict["\(prefix).mlp.fc2.bias"].type(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: c_proj_weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: c_proj_bias))
  }
  return (reader, Model([x], [out]))
}

func SigLIPVisionTransformer(
  gridX: Int, gridY: Int, width: Int, layers: Int, heads: Int, MLP: Int, batchSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let posEmbed = Parameter<Float>(
    .GPU(0), .CHW(1, 27 * 27, width), initBound: 1, name: "pos_embed")
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, gridX * gridY]).transposed(1, 2)
  out = out + posEmbed
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (reader, block) = SigLIPResidualAttentionBlock(
      prefix: "model.encoder.model.visual.blocks.\(i)", k: width / heads, h: heads, b: batchSize,
      t: gridX * gridY, MLP: MLP)
    out = block(out.reshaped([batchSize, gridX * gridY, width]))
    readers.append(reader)
  }
  let lnPost = LayerNorm(epsilon: 1e-6, axis: [1])
  out = lnPost(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let positional_embedding = state_dict["model.encoder.model.visual.pos_embed"].type(torch.float)
      .cpu().numpy()
    posEmbed.weight.copy(from: try! Tensor<Float>(numpy: positional_embedding))
    let conv1_weight = state_dict["model.encoder.model.visual.patch_embed.linear.weight"].type(
      torch.float
    ).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let conv1_bias = state_dict["model.encoder.model.visual.patch_embed.linear.bias"].type(
      torch.float
    ).cpu().numpy()
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    for reader in readers {
      reader(state_dict)
    }
    let ln_post_weight = state_dict["model.encoder.model.visual.norm.weight"].type(torch.float)
      .cpu().numpy()
    let ln_post_bias = state_dict["model.encoder.model.visual.norm.bias"].type(torch.float).cpu()
      .numpy()
    lnPost.weight.copy(from: try! Tensor<Float>(numpy: ln_post_weight))
    lnPost.bias.copy(from: try! Tensor<Float>(numpy: ln_post_bias))
  }
  return (reader, Model([x], [out]))
}

func MoondreamVisionProjection() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let mlp1fc = Dense(count: 2048 * 4)
  let mlp1gelu = GELU()
  let mlp1proj = Dense(count: 2048)
  var out = mlp1proj(mlp1gelu(mlp1fc(x)))
  let ln = LayerNorm(epsilon: 1e-5, axis: [1])
  out = ln(out)
  let mlp2fc = Dense(count: 2048 * 4)
  let mlp2gelu = GELU()
  let mlp2proj = Dense(count: 2048)
  out = out + mlp2proj(mlp2gelu(mlp2fc(out)))
  let reader: (PythonObject) -> Void = { state_dict in
    let mlp1_fc1_weight = state_dict["model.projection.mlp1.fc1.weight"].type(torch.float).cpu()
      .numpy()
    mlp1fc.weight.copy(from: try! Tensor<Float>(numpy: mlp1_fc1_weight))
    let mlp1_fc1_bias = state_dict["model.projection.mlp1.fc1.bias"].type(torch.float).cpu().numpy()
    mlp1fc.bias.copy(from: try! Tensor<Float>(numpy: mlp1_fc1_bias))
    let mlp1_fc2_weight = state_dict["model.projection.mlp1.fc2.weight"].type(torch.float).cpu()
      .numpy()
    mlp1proj.weight.copy(from: try! Tensor<Float>(numpy: mlp1_fc2_weight))
    let mlp1_fc2_bias = state_dict["model.projection.mlp1.fc2.bias"].type(torch.float).cpu().numpy()
    mlp1proj.bias.copy(from: try! Tensor<Float>(numpy: mlp1_fc2_bias))

    let ln_weight = state_dict["model.projection.ln.weight"].type(torch.float).cpu().numpy()
    ln.weight.copy(from: try! Tensor<Float>(numpy: ln_weight))
    let ln_bias = state_dict["model.projection.ln.bias"].type(torch.float).cpu().numpy()
    ln.bias.copy(from: try! Tensor<Float>(numpy: ln_bias))

    let mlp2_fc1_weight = state_dict["model.projection.mlp2.fc1.weight"].type(torch.float).cpu()
      .numpy()
    mlp2fc.weight.copy(from: try! Tensor<Float>(numpy: mlp2_fc1_weight))
    let mlp2_fc1_bias = state_dict["model.projection.mlp2.fc1.bias"].type(torch.float).cpu().numpy()
    mlp2fc.bias.copy(from: try! Tensor<Float>(numpy: mlp2_fc1_bias))
    let mlp2_fc2_weight = state_dict["model.projection.mlp2.fc2.weight"].type(torch.float).cpu()
      .numpy()
    mlp2proj.weight.copy(from: try! Tensor<Float>(numpy: mlp2_fc2_weight))
    let mlp2_fc2_bias = state_dict["model.projection.mlp2.fc2.bias"].type(torch.float).cpu().numpy()
    mlp2proj.bias.copy(from: try! Tensor<Float>(numpy: mlp2_fc2_bias))
  }
  return (reader, Model([x], [out]))
}

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, rotaryDim: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let costheta = Input()
  let sintheta = Input()
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  let keysRot0 = keys.reshaped(
    [b, t, hk, rotaryDim / 2], offset: [0, 0, 0, 0], strides: [t * hk * k, hk * k, k, 1])
  let keysRot1 = keys.reshaped(
    [b, t, hk, rotaryDim / 2], offset: [0, 0, 0, rotaryDim / 2],
    strides: [t * hk * k, hk * k, k, 1])
  let keysPass = keys.reshaped(
    [b, t, hk, k - rotaryDim], offset: [0, 0, 0, rotaryDim], strides: [t * hk * k, hk * k, k, 1])
  let queriesRot0 = queries.reshaped(
    [b, t, h, rotaryDim / 2], offset: [0, 0, 0, 0], strides: [t * h * k, h * k, k, 1])
  let queriesRot1 = queries.reshaped(
    [b, t, h, rotaryDim / 2], offset: [0, 0, 0, rotaryDim / 2], strides: [t * h * k, h * k, k, 1])
  let queriesPass = queries.reshaped(
    [b, t, h, k - rotaryDim], offset: [0, 0, 0, rotaryDim], strides: [t * h * k, h * k, k, 1])
  queries = Functional.concat(
    axis: 3, queriesRot0 .* costheta - queriesRot1 .* sintheta,
    queriesRot0 .* sintheta + queriesRot1 .* costheta, queriesPass)
  keys = Functional.concat(
    axis: 3, keysRot0 .* costheta - keysRot1 .* sintheta,
    keysRot0 .* sintheta + keysRot1 .* costheta, keysPass)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let wqkv_weight = state_dict["\(prefix).mixer.Wqkv.weight"].type(torch.float).cpu().numpy()
    let wqkv_bias = state_dict["\(prefix).mixer.Wqkv.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: wqkv_weight[..<(k * h), ...])))
    toqueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: wqkv_bias[..<(k * h)])))
    tokeys.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: wqkv_weight[(k * h)..<(2 * k * h), ...])))
    tokeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: wqkv_bias[(k * h)..<(2 * k * h)])))
    tovalues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: wqkv_weight[(2 * k * h)..., ...])))
    tovalues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: wqkv_bias[(2 * k * h)...])))
    let proj_weight = state_dict["\(prefix).mixer.out_proj.weight"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_weight)))
    let proj_bias = state_dict["\(prefix).mixer.out_proj.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_bias)))
  }
  return (Model([x, costheta, sintheta], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (Model, Model, Model)
{
  let x = Input()
  let w1 = Dense(count: intermediateSize)
  var out = GELU()(w1(x))
  let w2 = Dense(count: hiddenSize)
  out = w2(out)
  return (w1, w2, Model([x], [out], name: name))
}

func TransformerBlock(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int, rotaryDim: Int
) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let costheta = Input()
  let sintheta = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(
    prefix: prefix, k: k, h: h, hk: hk, b: b, t: t, rotaryDim: rotaryDim)
  let residual = out
  out = attention(out, costheta, sintheta) + x
  let (w1, w2, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "ffn")
  out = out + ffn(residual)
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
    let norm1_weight = state_dict["\(prefix).ln.weight"].type(torch.float)
      .cpu().numpy()
    norm1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm1_weight)))
    let norm1_bias = state_dict["\(prefix).ln.bias"].type(torch.float)
      .cpu().numpy()
    norm1.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm1_bias)))
    let w1_weight = state_dict["\(prefix).mlp.fc1.weight"].type(torch.float).cpu()
      .numpy()
    w1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w1_weight)))
    let w1_bias = state_dict["\(prefix).mlp.fc1.bias"].type(torch.float).cpu()
      .numpy()
    w1.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w1_bias)))
    let w2_weight = state_dict["\(prefix).mlp.fc2.weight"].type(torch.float).cpu()
      .numpy()
    w2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w2_weight)))
    let w2_bias = state_dict["\(prefix).mlp.fc2.bias"].type(torch.float).cpu()
      .numpy()
    w2.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w2_bias)))
  }
  return (Model([x, costheta, sintheta], [out]), reader)
}

public func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["tok_embeddings.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vocab)))
  }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, rotaryDim: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let textEmb = Input()
  let costheta = Input()
  let sintheta = Input()
  var out: Model.IO = textEmb
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (layer, reader) = TransformerBlock(
      prefix: "model.transformer.h.\(i)", k: width / heads, h: heads, hk: heads, b: batchSize,
      t: tokenLength,
      MLP: MLP, rotaryDim: rotaryDim)
    out = layer(out, costheta, sintheta)
    readers.append(reader)
  }
  let norm = LayerNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  let output = Dense(count: vocabularySize, name: "output")
  out = output(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
    let norm_weight = state_dict["model.lm_head.ln.weight"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_weight)))
    let norm_bias = state_dict["model.lm_head.ln.bias"].type(torch.float).cpu().numpy()
    norm.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_bias)))
    let output_weight = state_dict["model.lm_head.linear.weight"].type(torch.float).cpu().numpy()
    output.weight.copy(from: try! Tensor<Float16>(from: Tensor<Float>(numpy: output_weight)))
    let output_bias = state_dict["model.lm_head.linear.bias"].type(torch.float).cpu().numpy()
    output.bias.copy(from: try! Tensor<Float16>(from: Tensor<Float>(numpy: output_bias)))
  }
  return (Model([textEmb, costheta, sintheta], [out]), reader)
}

let random = Python.import("random")
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([1, 3, 378, 378])
print(
  model.vision_encoder.model(
    einops.rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1: 14, p2: 14)))
let graph = DynamicGraph()
let xTensor = graph.variable(try! Tensor<Float>(numpy: x.numpy())).toGPU(0)
let (reader, vit) = SigLIPVisionTransformer(
  gridX: 27, gridY: 27, width: 1152, layers: 27, heads: 16, MLP: 4304, batchSize: 1)
let (projReader, proj) = MoondreamVisionProjection()
let inputsEmbedsTensor = graph.variable(
  Tensor<Float16>(from: try! Tensor<Float>(numpy: input_embeds.numpy()))
).toGPU(0).reshaped(.WC(4, 2048))

let (phi, phiReader) = Transformer(
  Float16.self, vocabularySize: 51_200, width: 2048, tokenLength: 4, layers: 24, MLP: 2048 * 4,
  rotaryDim: 32, heads: 32, batchSize: 1)

graph.withNoGrad {
  let _ = vit(inputs: xTensor)
  reader(vision_encoder_state_dict)
  var outs = vit(inputs: xTensor).map { $0.as(of: Float.self) }
  let _ = proj(inputs: outs[0])
  projReader(vision_encoder_state_dict)
  outs = proj(inputs: outs[0]).map { $0.as(of: Float.self) }
  print(outs)

  let costhetaTensor = graph.variable(.CPU, .NHWC(1, 4, 1, 16), of: Float.self)
  let sinthetaTensor = graph.variable(.CPU, .NHWC(1, 4, 1, 16), of: Float.self)
  for i in 0..<4 {
    for k in 0..<16 {
      let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 32)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      costhetaTensor[0, i, 0, k] = Float(costheta)
      sinthetaTensor[0, i, 0, k] = Float(sintheta)
    }
  }
  let costhetaTensorGPU = DynamicGraph.Tensor<Float16>(from: costhetaTensor).toGPU(0)
  let sinthetaTensorGPU = DynamicGraph.Tensor<Float16>(from: sinthetaTensor).toGPU(0)
  debugPrint(costhetaTensor)
  debugPrint(sinthetaTensor)
  let _ = phi(inputs: inputsEmbedsTensor, costhetaTensorGPU, sinthetaTensorGPU)
  let text_emb = text_model_state_dict["text_emb.weight"].type(torch.float).cpu().numpy()
  let textEmb = Tensor<Float16>(from: try! Tensor<Float>(numpy: text_emb))
  phiReader(text_model_state_dict)
  let output = phi(inputs: inputsEmbedsTensor, costhetaTensorGPU, sinthetaTensorGPU).map {
    $0.as(of: Float16.self)
  }
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/siglip_384_f32.ckpt") {
    $0.write("vit", model: vit)
  }
  graph.openStore("/home/liu/workspace/swift-diffusion/moondream1_f32.ckpt") {
    $0.write("vision_proj", model: proj)
    $0.write("text_emb", tensor: textEmb)
    $0.write("phi", model: phi)
  }
  */
  let digit = output[0].rawValue.toCPU()
  var minVal = digit[3, 0]
  var minS = 0
  for i in 1..<51_200 {
    if digit[3, i] > minVal {
      minVal = digit[3, i]
      minS = i
    }
  }
  print("index \(minS)")
}

/*
let gpt2tokenizer = GPT2Tokenizer(
  vocabulary: "/home/liu/vocab.json", merges: "/home/liu/merges.txt")

print(
  gpt2tokenizer.tokenize(
    text:
      "</image>\n\nQuestion: Describe this image and its style in a very detailed manner, follow the format of describing: what, who, where, when, how. You don't need to fill in all if they are irrelevant. Please remove What, Who, Where, When, How prefixes and make it one paragraph.\n\nAnswer:",
    addSpecialTokens: false))

print(gpt2tokenizer.tokenize(text: "<image>", addSpecialTokens: false))
*/
