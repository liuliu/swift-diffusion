import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let streamlit_helpers = Python.import("scripts.demo.streamlit_helpers")

var version_dict: [String: PythonObject] = [
  "H": 1024,
  "W": 1024,
  "C": 4,
  "f": 8,
  "is_legacy": false,
  "config": "/home/liu/workspace/generative-models/configs/inference/sd_xl_base.yaml",
  "ckpt": "/home/liu/workspace/generative-models/checkpoints/sd_xl_base_0.9.safetensors",
  "is_guided": true,
]

let state = streamlit_helpers.init_st(version_dict)
let init_dict: [String: PythonObject] = [
  "orig_width": 1280,
  "orig_height": 1024,
  "target_width": 1024,
  "target_height": 1024,
]
let prompt = "astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
let negative_prompt = ""
let value_dict = streamlit_helpers.init_embedder_options(
  streamlit_helpers.get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
  init_dict, prompt: prompt, negative_prompt: negative_prompt)
let (num_rows, num_cols, sampler) = streamlit_helpers.init_sampling(use_identity_guider: false)
  .tuple3
let out = streamlit_helpers.do_sample(
  state["model"], sampler, value_dict, 1, 1024, 1024, 4, 8,
  force_uc_zero_embeddings: [PythonObject](), return_latents: false)
print(out)
// let state_dict = state["model"].conditioner.embedders[0].transformer.state_dict()
// let state_dict = state["model"].conditioner.embedders[1].model.state_dict()
// let state_dict = state["model"].model.state_dict()
// print(state_dict.keys())

/* OpenAI CLIP L14 model.
func CLIPTextEmbedding(vocabularySize: Int, maxLength: Int, embeddingSize: Int) -> (
  Model, Model, Model
) {
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(Float.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + positionEmbed(positions)
  return (tokenEmbed, positionEmbed, Model([tokens, positions], [embedding], name: "embeddings"))
}

func CLIPAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + casualAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, casualAttentionMask], [out]))
}

func QuickGELU() -> Model {
  let x = Input()
  let y = x .* Sigmoid()(1.702 * x)
  return Model([x], [y])
}

func CLIPMLP(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = fc1(x)
  out = QuickGELU()(out)
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func CLIPEncoderLayer(k: Int, h: Int, b: Int, t: Int, intermediateSize: Int) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model
) {
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let (tokeys, toqueries, tovalues, unifyheads, attention) = CLIPAttention(k: k, h: h, b: b, t: t)
  out = attention(out, casualAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let (fc1, fc2, mlp) = CLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return (
    layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2,
    Model([x, casualAttentionMask], [out])
  )
}

func CLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> (
  Model, Model, [Model], [Model], [Model], [Model], [Model], [Model], [Model], [Model], Model, Model
) {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let (tokenEmbed, positionEmbed, embedding) = CLIPTextEmbedding(
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  var layerNorm1s = [Model]()
  var tokeyss = [Model]()
  var toqueriess = [Model]()
  var tovaluess = [Model]()
  var unifyheadss = [Model]()
  var layerNorm2s = [Model]()
  var fc1s = [Model]()
  var fc2s = [Model]()
  let k = embeddingSize / numHeads
  for _ in 0..<numLayers {
    let (layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2, encoderLayer) =
      CLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    layerNorm1s.append(layerNorm1)
    tokeyss.append(tokeys)
    toqueriess.append(toqueries)
    tovaluess.append(tovalues)
    unifyheadss.append(unifyheads)
    layerNorm2s.append(layerNorm2)
    fc1s.append(fc1)
    fc2s.append(fc2)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return (
    tokenEmbed, positionEmbed, layerNorm1s, tokeyss, toqueriess, tovaluess, unifyheadss,
    layerNorm2s, fc1s, fc2s, finalLayerNorm, Model([tokens, positions, casualAttentionMask], [out])
  )
}

let transformers = Python.import("transformers")
let tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

let batch_encoding = tokenizer(
  ["a photograph of an astronaut riding a horse"], truncation: true, max_length: 77,
  return_length: true, return_overflowing_tokens: false, padding: "max_length", return_tensors: "pt"
)
let tokens = batch_encoding["input_ids"]

let (
  tokenEmbed, positionEmbed, layerNorm1s, tokeys, toqueries, tovalues, unifyheads, layerNorm2s,
  fc1s, fc2s, finalLayerNorm, textModel
) = CLIPTextModel(
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
  batchSize: 1, intermediateSize: 3072)

let graph = DynamicGraph()
let tokensTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensNumpy = tokens.numpy()
for i in 0..<77 {
  tokensTensor[i] = Int32(tokensNumpy[0, i])!
  positionTensor[i] = Int32(i)
}
let casualAttentionMask = graph.variable(Tensor<Float>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -Float.greatestFiniteMagnitude
  }
}

let _ = textModel(inputs: tokensTensor, positionTensor, casualAttentionMask)

let vocab = state_dict["text_model.embeddings.token_embedding.weight"]
let pos = state_dict["text_model.embeddings.position_embedding.weight"]
tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab.cpu().numpy()))
positionEmbed.parameters.copy(from: try! Tensor<Float>(numpy: pos.cpu().numpy()))

for i in 0..<12 {
  let layer_norm_1_weight = state_dict["text_model.encoder.layers.\(i).layer_norm1.weight"].cpu().numpy()
  let layer_norm_1_bias = state_dict["text_model.encoder.layers.\(i).layer_norm1.bias"].cpu().numpy()
  layerNorm1s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: layer_norm_1_weight))
  layerNorm1s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: layer_norm_1_bias))

  let k_proj_weight = state_dict["text_model.encoder.layers.\(i).self_attn.k_proj.weight"].cpu().numpy()
  let k_proj_bias = state_dict["text_model.encoder.layers.\(i).self_attn.k_proj.bias"].cpu().numpy()
  tokeys[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: k_proj_weight))
  tokeys[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: k_proj_bias))

  let v_proj_weight = state_dict["text_model.encoder.layers.\(i).self_attn.v_proj.weight"].cpu().numpy()
  let v_proj_bias = state_dict["text_model.encoder.layers.\(i).self_attn.v_proj.bias"].cpu().numpy()
  tovalues[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: v_proj_weight))
  tovalues[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: v_proj_bias))

  let q_proj_weight = state_dict["text_model.encoder.layers.\(i).self_attn.q_proj.weight"].cpu().numpy()
  let q_proj_bias = state_dict["text_model.encoder.layers.\(i).self_attn.q_proj.bias"].cpu().numpy()
  toqueries[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: q_proj_weight))
  toqueries[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: q_proj_bias))

  let out_proj_weight = state_dict["text_model.encoder.layers.\(i).self_attn.out_proj.weight"]
    .cpu().numpy()
  let out_proj_bias = state_dict["text_model.encoder.layers.\(i).self_attn.out_proj.bias"].cpu().numpy()
  unifyheads[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_proj_weight))
  unifyheads[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_proj_bias))

  let layer_norm_2_weight = state_dict["text_model.encoder.layers.\(i).layer_norm2.weight"].cpu().numpy()
  let layer_norm_2_bias = state_dict["text_model.encoder.layers.\(i).layer_norm2.bias"].cpu().numpy()
  layerNorm2s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: layer_norm_2_weight))
  layerNorm2s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: layer_norm_2_bias))

  let fc1_weight = state_dict["text_model.encoder.layers.\(i).mlp.fc1.weight"].cpu().numpy()
  let fc1_bias = state_dict["text_model.encoder.layers.\(i).mlp.fc1.bias"].cpu().numpy()
  fc1s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: fc1_weight))
  fc1s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: fc1_bias))

  let fc2_weight = state_dict["text_model.encoder.layers.\(i).mlp.fc2.weight"].cpu().numpy()
  let fc2_bias = state_dict["text_model.encoder.layers.\(i).mlp.fc2.bias"].cpu().numpy()
  fc2s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: fc2_weight))
  fc2s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: fc2_bias))
}

let final_layer_norm_weight = state_dict["text_model.final_layer_norm.weight"].cpu().numpy()
let final_layer_norm_bias = state_dict["text_model.final_layer_norm.bias"].cpu().numpy()
finalLayerNorm.parameters(for: .weight).copy(
  from: try! Tensor<Float>(numpy: final_layer_norm_weight))
finalLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: final_layer_norm_bias))

let c = textModel(inputs: tokensTensor, positionTensor, casualAttentionMask)[0].as(of: Float.self)

graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
  $0.write("text_model", model: textModel)
}
*/

/* OpenCLIP bigG14 model.
func CLIPTextEmbedding(vocabularySize: Int, maxLength: Int, embeddingSize: Int) -> (
  Model, Model, Model
) {
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(Float.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + positionEmbed(positions)
  return (tokenEmbed, positionEmbed, Model([tokens, positions], [embedding], name: "embeddings"))
}

func CLIPAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + casualAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, casualAttentionMask], [out]))
}

func CLIPMLP(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = fc1(x)
  out = GELU()(out)
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func CLIPEncoderLayer(k: Int, h: Int, b: Int, t: Int, intermediateSize: Int) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model
) {
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let (tokeys, toqueries, tovalues, unifyheads, attention) = CLIPAttention(k: k, h: h, b: b, t: t)
  out = attention(out, casualAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let (fc1, fc2, mlp) = CLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return (
    layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2,
    Model([x, casualAttentionMask], [out])
  )
}

func CLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> (
  Model, Model, [Model], [Model], [Model], [Model], [Model], [Model], [Model], [Model], Model, Model
) {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let (tokenEmbed, positionEmbed, embedding) = CLIPTextEmbedding(
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  var layerNorm1s = [Model]()
  var tokeyss = [Model]()
  var toqueriess = [Model]()
  var tovaluess = [Model]()
  var unifyheadss = [Model]()
  var layerNorm2s = [Model]()
  var fc1s = [Model]()
  var fc2s = [Model]()
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 {
      penultimate = out
    }
    let (layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2, encoderLayer) =
      CLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    layerNorm1s.append(layerNorm1)
    tokeyss.append(tokeys)
    toqueriess.append(toqueries)
    tovaluess.append(tovalues)
    unifyheadss.append(unifyheads)
    layerNorm2s.append(layerNorm2)
    fc1s.append(fc1)
    fc2s.append(fc2)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return (
    tokenEmbed, positionEmbed, layerNorm1s, tokeyss, toqueriess, tovaluess, unifyheadss,
    layerNorm2s, fc1s, fc2s, finalLayerNorm, Model([tokens, positions, casualAttentionMask], [penultimate!, out])
  )
}

let open_clip = Python.import("open_clip")
let torch = Python.import("torch")

let tokens = open_clip.tokenize(["a professional photograph of an astronaut riding a horse"])
print(tokens)
let x = state["model"].conditioner.embedders[1].encode_with_transformer(tokens.cuda())
print(x)

let (
  tokenEmbed, positionEmbed, layerNorm1s, tokeys, toqueries, tovalues, unifyheads, layerNorm2s,
  fc1s, fc2s, finalLayerNorm, textModel
) = CLIPTextModel(
  vocabularySize: 49408, maxLength: 77, embeddingSize: 1280, numLayers: 32, numHeads: 20,
  batchSize: 1, intermediateSize: 5120)

let graph = DynamicGraph()
let tokensTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensNumpy = tokens.numpy()
for i in 0..<77 {
  tokensTensor[i] = Int32(tokensNumpy[0, i])!
  positionTensor[i] = Int32(i)
}
let casualAttentionMask = graph.variable(Tensor<Float>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -Float.greatestFiniteMagnitude
  }
}

let tokensTensorGPU = tokensTensor.toGPU(1)
let positionTensorGPU = positionTensor.toGPU(1)
let casualAttentionMaskGPU = casualAttentionMask.toGPU(1)
let _ = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)

let vocab = state_dict["token_embedding.weight"]
let pos = state_dict["positional_embedding"]
tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab.cpu().numpy()))
positionEmbed.parameters.copy(from: try! Tensor<Float>(numpy: pos.cpu().numpy()))
print("\"token_embedding.weight\", \"\(tokenEmbed.parameters.name)\"")
print("\"positional_embedding\", \"\(positionEmbed.parameters.name)\"")

for i in 0..<32 {
  let layer_norm_1_weight = state_dict["transformer.resblocks.\(i).ln_1.weight"].cpu().numpy()
  let layer_norm_1_bias = state_dict["transformer.resblocks.\(i).ln_1.bias"].cpu().numpy()
  layerNorm1s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: layer_norm_1_weight))
  layerNorm1s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: layer_norm_1_bias))
  print("\"transformer.resblocks.\(i).ln_1.weight\", \"\(layerNorm1s[i].parameters(for: .weight).name)\"")
  print("\"transformer.resblocks.\(i).ln_1.bias\", \"\(layerNorm1s[i].parameters(for: .bias).name)\"")

  let in_proj_weight = state_dict["transformer.resblocks.\(i).attn.in_proj_weight"].type(
    torch.float
  ).cpu().numpy()
  let in_proj_bias = state_dict["transformer.resblocks.\(i).attn.in_proj_bias"].type(torch.float)
    .cpu().numpy()
  toqueries[i].parameters(for: .weight).copy(
    from: try! Tensor<Float>(numpy: in_proj_weight[..<(1280), ...]))
  toqueries[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_proj_bias[..<(1280)]))
  print("\"transformer.resblocks.\(i).attn.in_proj_weight\", \"\(toqueries[i].parameters(for: .weight).name)\"")
  print("\"transformer.resblocks.\(i).attn.in_proj_bias\", \"\(toqueries[i].parameters(for: .bias).name)\"")
  tokeys[i].parameters(for: .weight).copy(
    from: try! Tensor<Float>(numpy: in_proj_weight[(1280)..<(2 * 1280), ...]))
  tokeys[i].parameters(for: .bias).copy(
    from: try! Tensor<Float>(numpy: in_proj_bias[(1280)..<(2 * 1280)]))
  print("\"transformer.resblocks.\(i).attn.in_proj_weight\", \"\(tokeys[i].parameters(for: .weight).name)\"")
  print("\"transformer.resblocks.\(i).attn.in_proj_bias\", \"\(tokeys[i].parameters(for: .bias).name)\"")
  tovalues[i].parameters(for: .weight).copy(
    from: try! Tensor<Float>(numpy: in_proj_weight[(2 * 1280)..., ...]))
  tovalues[i].parameters(for: .bias).copy(
    from: try! Tensor<Float>(numpy: in_proj_bias[(2 * 1280)...]))
  print("\"transformer.resblocks.\(i).attn.in_proj_weight\", \"\(tovalues[i].parameters(for: .weight).name)\"")
  print("\"transformer.resblocks.\(i).attn.in_proj_bias\", \"\(tovalues[i].parameters(for: .bias).name)\"")

  let out_proj_weight = state_dict["transformer.resblocks.\(i).attn.out_proj.weight"]
    .cpu().numpy()
  let out_proj_bias = state_dict["transformer.resblocks.\(i).attn.out_proj.bias"].cpu().numpy()
  unifyheads[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_proj_weight))
  unifyheads[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_proj_bias))
  print("\"transformer.resblocks.\(i).attn.out_proj.weight\", \"\(unifyheads[i].parameters(for: .weight).name)\"")
  print("\"transformer.resblocks.\(i).attn.out_proj.bias\", \"\(unifyheads[i].parameters(for: .bias).name)\"")

  let layer_norm_2_weight = state_dict["transformer.resblocks.\(i).ln_2.weight"].cpu().numpy()
  let layer_norm_2_bias = state_dict["transformer.resblocks.\(i).ln_2.bias"].cpu().numpy()
  layerNorm2s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: layer_norm_2_weight))
  layerNorm2s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: layer_norm_2_bias))
  print("\"transformer.resblocks.\(i).ln_2.weight\", \"\(layerNorm2s[i].parameters(for: .weight).name)\"")
  print("\"transformer.resblocks.\(i).ln_2.bias\", \"\(layerNorm2s[i].parameters(for: .bias).name)\"")

  let fc1_weight = state_dict["transformer.resblocks.\(i).mlp.c_fc.weight"].cpu().numpy()
  let fc1_bias = state_dict["transformer.resblocks.\(i).mlp.c_fc.bias"].cpu().numpy()
  fc1s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: fc1_weight))
  fc1s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: fc1_bias))
  print("\"transformer.resblocks.\(i).mlp.c_fc.weight\", \"\(fc1s[i].parameters(for: .weight).name)\"")
  print("\"transformer.resblocks.\(i).mlp.c_fc.bias\", \"\(fc1s[i].parameters(for: .bias).name)\"")

  let fc2_weight = state_dict["transformer.resblocks.\(i).mlp.c_proj.weight"].cpu().numpy()
  let fc2_bias = state_dict["transformer.resblocks.\(i).mlp.c_proj.bias"].cpu().numpy()
  fc2s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: fc2_weight))
  fc2s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: fc2_bias))
  print("\"transformer.resblocks.\(i).mlp.c_proj.weight\", \"\(fc2s[i].parameters(for: .weight).name)\"")
  print("\"transformer.resblocks.\(i).mlp.c_proj.bias\", \"\(fc2s[i].parameters(for: .bias).name)\"")
}

let final_layer_norm_weight = state_dict["ln_final.weight"].cpu().numpy()
let final_layer_norm_bias = state_dict["ln_final.bias"].cpu().numpy()
finalLayerNorm.parameters(for: .weight).copy(
  from: try! Tensor<Float>(numpy: final_layer_norm_weight))
finalLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: final_layer_norm_bias))
print("\"ln_final.weight\", \"\(finalLayerNorm.parameters(for: .weight).name)\"")
print("\"ln_final.bias\", \"\(finalLayerNorm.parameters(for: .bias).name)\"")

graph.withNoGrad {
  let text_projection = state_dict["text_projection"].cpu().numpy()
  let textProjectionTensor = try! Tensor<Float>(numpy: text_projection)
  let textProjection = graph.variable(textProjectionTensor.toGPU(1))
  let c = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU).map { $0.as(of: Float.self) }
  debugPrint(c[0])
  debugPrint(c[1][10..<11, 0..<1280])
  let pooled = c[1][10..<11, 0..<1280] * textProjection
  debugPrint(pooled)
  graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
    $0.write("text_model", model: textModel)
    $0.write("text_projection", tensor: textProjectionTensor)
  }
}
*/

/* SDXL Base UNet
func timeEmbedding(timesteps: Int, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * Float(timesteps)
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

func TimeEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func LabelEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func ResBlock(b: Int, outChannels: Int, skipConnection: Bool) -> (
  Model, Model, Model, Model, Model, Model?, Model
) {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: outChannels)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([b, outChannels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var skipModel: Model? = nil
  if skipConnection {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]))
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = skip
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
  }
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel,
    Model([x, emb], [out])
  )
}

func SelfAttention(k: Int, h: Int, b: Int, hw: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, hw])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func CrossAttentionKeysAndValues(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = Dense(count: k * h, noBias: true)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, unifyheads, Model([x, keys, values], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let fc10 = Dense(count: intermediateSize)
  let fc11 = Dense(count: intermediateSize)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc10, fc11, fc2, Model([x], [out]))
}

func BasicTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (toqueries2, unifyheads2, attn2) = CrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, keys, values) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  let reader: (PythonObject) -> Void = { state_dict in
    let attn1_to_k_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_k.weight"
    ].cpu().numpy()
    tokeys1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_k_weight))
    let attn1_to_q_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_q.weight"
    ].cpu().numpy()
    toqueries1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_q_weight))
    let attn1_to_v_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_v.weight"
    ].cpu().numpy()
    tovalues1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_v_weight))
    let attn1_to_out_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_out.0.weight"
    ].cpu().numpy()
    let attn1_to_out_bias = state_dict[
      "diffusion_model.\(prefix).attn1.to_out.0.bias"
    ].cpu().numpy()
    unifyheads1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attn1_to_out_weight))
    unifyheads1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn1_to_out_bias))
    let ff_net_0_proj_weight = state_dict[
      "diffusion_model.\(prefix).ff.net.0.proj.weight"
    ].cpu().numpy()
    let ff_net_0_proj_bias = state_dict[
      "diffusion_model.\(prefix).ff.net.0.proj.bias"
    ].cpu().numpy()
    fc10.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[..<intermediateSize, ...]))
    fc10.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[..<intermediateSize]))
    fc11.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[intermediateSize..., ...]))
    fc11.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[intermediateSize...]))
    let ff_net_2_weight = state_dict[
      "diffusion_model.\(prefix).ff.net.2.weight"
    ].cpu().numpy()
    let ff_net_2_bias = state_dict[
      "diffusion_model.\(prefix).ff.net.2.bias"
    ].cpu().numpy()
    fc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ff_net_2_weight))
    fc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_2_bias))
    let attn2_to_q_weight = state_dict[
      "diffusion_model.\(prefix).attn2.to_q.weight"
    ].cpu().numpy()
    toqueries2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_q_weight))
    let attn2_to_out_weight = state_dict[
      "diffusion_model.\(prefix).attn2.to_out.0.weight"
    ].cpu().numpy()
    let attn2_to_out_bias = state_dict[
      "diffusion_model.\(prefix).attn2.to_out.0.bias"
    ].cpu().numpy()
    unifyheads2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attn2_to_out_weight))
    unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
    let norm1_weight = state_dict[
      "diffusion_model.\(prefix).norm1.weight"
    ]
    .cpu().numpy()
    let norm1_bias = state_dict[
      "diffusion_model.\(prefix).norm1.bias"
    ]
    .cpu().numpy()
    layerNorm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    layerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let norm2_weight = state_dict[
      "diffusion_model.\(prefix).norm2.weight"
    ]
    .cpu().numpy()
    let norm2_bias = state_dict[
      "diffusion_model.\(prefix).norm2.bias"
    ]
    .cpu().numpy()
    layerNorm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
    layerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let norm3_weight = state_dict[
      "diffusion_model.\(prefix).norm3.weight"
    ]
    .cpu().numpy()
    let norm3_bias = state_dict[
      "diffusion_model.\(prefix).norm3.bias"
    ]
    .cpu().numpy()
    layerNorm3.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm3_weight))
    layerNorm3.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm3_bias))
  }
  return (reader, Model([x, keys, values], [out]))
}

func SpatialTransformer(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).permuted(0, 2, 1)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<depth {
    let keys = Input()
    kvs.append(keys)
    let values = Input()
    kvs.append(values)
    let (reader, block) = BasicTransformerBlock(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    out = block(out, keys, values)
    readers.append(reader)
  }
  out = out.reshaped([b, height, width, k * h]).permuted(0, 3, 1, 2)
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_weight = state_dict["diffusion_model.\(prefix).norm.weight"]
      .cpu().numpy()
    let norm_bias = state_dict["diffusion_model.\(prefix).norm.bias"].cpu().numpy()
    norm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_bias))
    let proj_in_weight = state_dict["diffusion_model.\(prefix).proj_in.weight"]
      .cpu().numpy()
    let proj_in_bias = state_dict["diffusion_model.\(prefix).proj_in.bias"]
      .cpu().numpy()
    projIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_in_weight))
    projIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    let proj_out_weight = state_dict[
      "diffusion_model.\(prefix).proj_out.weight"
    ].cpu().numpy()
    let proj_out_bias = state_dict["diffusion_model.\(prefix).proj_out.bias"]
      .cpu().numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x] + kvs, [out]))
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let emb = Input()
  var kvs = [Model.IO]()
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  var transformerReader: ((PythonObject) -> Void)? = nil
  if attentionBlock > 0 {
    let c = (0..<(attentionBlock * 2)).map { _ in Input() }
    let transformer: Model
    (
      transformerReader, transformer
    ) = SpatialTransformer(
      prefix: "\(prefix).\(layerStart).1",
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let in_layers_0_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.0.weight"
    ].cpu().numpy()
    let in_layers_0_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.0.bias"
    ].cpu().numpy()
    inLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_0_weight))
    inLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_bias))
    let in_layers_2_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.2.weight"
    ].cpu().numpy()
    let in_layers_2_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.2.bias"
    ].cpu().numpy()
    inLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_2_weight))
    inLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_bias))
    let emb_layers_1_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.weight"
    ].cpu().numpy()
    let emb_layers_1_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.bias"
    ].cpu().numpy()
    embLayer.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_1_weight))
    embLayer.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_1_bias))
    let out_layers_0_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.0.weight"
    ].cpu().numpy()
    let out_layers_0_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.0.bias"
    ].cpu().numpy()
    outLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_layers_0_weight))
    outLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_bias))
    let out_layers_3_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.3.weight"
    ].cpu().numpy()
    let out_layers_3_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.3.bias"
    ].cpu().numpy()
    outLayerConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_3_weight))
    outLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_3_bias))
    if let skipModel = skipModel {
      let skip_connection_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).0.skip_connection.weight"
      ].cpu().numpy()
      let skip_connection_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).0.skip_connection.bias"
      ].cpu().numpy()
      skipModel.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: skip_connection_weight))
      skipModel.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: skip_connection_bias))
    }
    if let transformerReader = transformerReader {
      transformerReader(state_dict)
    }
  }
  return (reader, Model([x, emb] + kvs, [out]))
}

func MiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  x: Model.IO, emb: Model.IO
) -> ((PythonObject) -> Void, Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let kvs = [Input(), Input()]
  let (
    transformerReader, transformer
  ) = SpatialTransformer(
    prefix: "middle_block.1",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, depth: 1,
    t: embeddingSize,
    intermediateSize: channels * 4)
  out = transformer([out] + kvs)
  let (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  let reader: (PythonObject) -> Void = { state_dict in
    let in_layers_0_0_weight = state_dict["diffusion_model.middle_block.0.in_layers.0.weight"]
      .cpu().numpy()
    let in_layers_0_0_bias = state_dict["diffusion_model.middle_block.0.in_layers.0.bias"].cpu()
      .numpy()
    inLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_0_0_weight))
    inLayerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_0_bias))
    let in_layers_0_2_weight = state_dict["diffusion_model.middle_block.0.in_layers.2.weight"]
      .cpu().numpy()
    let in_layers_0_2_bias = state_dict["diffusion_model.middle_block.0.in_layers.2.bias"].cpu()
      .numpy()
    inLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_0_2_weight))
    inLayerConv2d1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_2_bias))
    let emb_layers_0_1_weight = state_dict["diffusion_model.middle_block.0.emb_layers.1.weight"]
      .cpu().numpy()
    let emb_layers_0_1_bias = state_dict["diffusion_model.middle_block.0.emb_layers.1.bias"].cpu()
      .numpy()
    embLayer1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_0_1_weight))
    embLayer1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_0_1_bias))
    let out_layers_0_0_weight = state_dict["diffusion_model.middle_block.0.out_layers.0.weight"]
      .cpu().numpy()
    let out_layers_0_0_bias = state_dict[
      "diffusion_model.middle_block.0.out_layers.0.bias"
    ].cpu().numpy()
    outLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_0_weight))
    outLayerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_0_bias))
    let out_layers_0_3_weight = state_dict["diffusion_model.middle_block.0.out_layers.3.weight"]
      .cpu().numpy()
    let out_layers_0_3_bias = state_dict["diffusion_model.middle_block.0.out_layers.3.bias"].cpu()
      .numpy()
    outLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_3_weight))
    outLayerConv2d1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_3_bias))
    transformerReader(state_dict)
    let in_layers_2_0_weight = state_dict["diffusion_model.middle_block.2.in_layers.0.weight"]
      .cpu().numpy()
    let in_layers_2_0_bias = state_dict["diffusion_model.middle_block.2.in_layers.0.bias"].cpu()
      .numpy()
    inLayerNorm2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_2_0_weight))
    inLayerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_0_bias))
    let in_layers_2_2_weight = state_dict["diffusion_model.middle_block.2.in_layers.2.weight"]
      .cpu().numpy()
    let in_layers_2_2_bias = state_dict["diffusion_model.middle_block.2.in_layers.2.bias"].cpu()
      .numpy()
    inLayerConv2d2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_2_2_weight))
    inLayerConv2d2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_2_bias))
    let emb_layers_2_1_weight = state_dict["diffusion_model.middle_block.2.emb_layers.1.weight"]
      .cpu().numpy()
    let emb_layers_2_1_bias = state_dict["diffusion_model.middle_block.2.emb_layers.1.bias"].cpu()
      .numpy()
    embLayer2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_2_1_weight))
    embLayer2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_2_1_bias))
    let out_layers_2_0_weight = state_dict["diffusion_model.middle_block.2.out_layers.0.weight"]
      .cpu().numpy()
    let out_layers_2_0_bias = state_dict[
      "diffusion_model.middle_block.2.out_layers.0.bias"
    ].cpu().numpy()
    outLayerNorm2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_0_weight))
    outLayerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_2_0_bias))
    let out_layers_2_3_weight = state_dict["diffusion_model.middle_block.2.out_layers.3.weight"]
      .cpu().numpy()
    let out_layers_2_3_bias = state_dict["diffusion_model.middle_block.2.out_layers.3.bias"].cpu()
      .numpy()
    outLayerConv2d2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_3_weight))
    outLayerConv2d2.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_3_bias))
  }
  return (reader, out, kvs)
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO
) -> ((PythonObject) -> Void, [Model.IO], Model.IO, [Input]) {
  let conv2d = Convolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      let (reader, inputLayer) = BlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      let c = (0..<(attentionBlock * 2)).map { _ in Input() }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      passLayers.append(out)
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let reader: (PythonObject) -> Void = { state_dict in
        let op_weight = state_dict["diffusion_model.input_blocks.\(downLayer).0.op.weight"].cpu()
          .numpy()
        let op_bias = state_dict["diffusion_model.input_blocks.\(downLayer).0.op.bias"].cpu()
          .numpy()
        downsample.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: op_weight))
        downsample.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: op_bias))
      }
      readers.append(reader)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let input_blocks_0_0_weight = state_dict["diffusion_model.input_blocks.0.0.weight"].cpu()
      .numpy()
    let input_blocks_0_0_bias = state_dict["diffusion_model.input_blocks.0.0.bias"].cpu().numpy()
    conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_weight))
    conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_bias))
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, passLayers, out, kvs)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO,
  inputs: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO, [Input]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let (reader, outputLayer) = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      let c = (0..<(attentionBlock * 2)).map { _ in Input() }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      readers.append(reader)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock > 0 ? 2 : 1
        let reader: (PythonObject) -> Void = { state_dict in
          let op_weight = state_dict[
            "diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight"
          ].cpu().numpy()
          let op_bias = state_dict["diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias"]
            .cpu().numpy()
          conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: op_weight))
          conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: op_bias))
        }
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out, kvs)
}

func UNetXL(batchSize: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  let (timeFc0, timeFc2, timeEmbed) = TimeEmbed(modelChannels: 320)
  let (labelFc0, labelFc2, labelEmbed) = LabelEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let attentionRes: [Int: Int] = [2: 2, 4: 10]

  let (inputReader, inputs, inputBlocks, inputKVs) = InputBlocks(
    channels: [320, 640, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: 128,
    startWidth: 128, embeddingSize: 77, attentionRes: attentionRes, x: x, emb: emb)
  var out = inputBlocks
  let (middleReader, middleBlock, middleKVs) = MiddleBlock(
    channels: 1280, numHeadChannels: 64, batchSize: batchSize, height: 32, width: 32,
    embeddingSize: 77,
    x: out, emb: emb)
  out = middleBlock
  let (outputReader, outputBlocks, outputKVs) = OutputBlocks(
    channels: [320, 640, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: 128,
    startWidth: 128, embeddingSize: 77, attentionRes: attentionRes, x: out, emb: emb,
    inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let time_embed_0_weight = state_dict["diffusion_model.time_embed.0.weight"].cpu().numpy()
    let time_embed_0_bias = state_dict["diffusion_model.time_embed.0.bias"].cpu().numpy()
    let time_embed_2_weight = state_dict["diffusion_model.time_embed.2.weight"].cpu().numpy()
    let time_embed_2_bias = state_dict["diffusion_model.time_embed.2.bias"].cpu().numpy()
    timeFc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_0_weight))
    timeFc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_0_bias))
    timeFc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_2_weight))
    timeFc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_2_bias))
    let label_emb_0_0_weight = state_dict["diffusion_model.label_emb.0.0.weight"].cpu().numpy()
    let label_emb_0_0_bias = state_dict["diffusion_model.label_emb.0.0.bias"].cpu().numpy()
    let label_emb_0_2_weight = state_dict["diffusion_model.label_emb.0.2.weight"].cpu().numpy()
    let label_emb_0_2_bias = state_dict["diffusion_model.label_emb.0.2.bias"].cpu().numpy()
    labelFc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: label_emb_0_0_weight))
    labelFc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: label_emb_0_0_bias))
    labelFc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: label_emb_0_2_weight))
    labelFc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: label_emb_0_2_bias))
    inputReader(state_dict)
    middleReader(state_dict)
    outputReader(state_dict)
    let out_0_weight = state_dict["diffusion_model.out.0.weight"].cpu().numpy()
    let out_0_bias = state_dict["diffusion_model.out.0.bias"].cpu().numpy()
    outNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_0_weight))
    outNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_0_bias))
    let out_2_weight = state_dict["diffusion_model.out.2.weight"].cpu().numpy()
    let out_2_bias = state_dict["diffusion_model.out.2.bias"].cpu().numpy()
    outConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_2_weight))
    outConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_2_bias))
  }
  return (reader, Model([x, t_emb, y] + inputKVs + middleKVs + outputKVs, [out]))
}

func CrossAttentionFixed(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model) {
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
  return (tokeys, tovalues, Model([c], [keys, values]))
}

func BasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(k: k, h: h, b: b, hw: hw, t: t)
  let reader: (PythonObject) -> Void = { state_dict in
    let attn2_to_k_weight = state_dict[
      "diffusion_model.\(prefix).attn2.to_k.weight"
    ].cpu().numpy()
    tokeys2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_k_weight))
    let attn2_to_v_weight = state_dict[
      "diffusion_model.\(prefix).attn2.to_v.weight"
    ].cpu().numpy()
    tovalues2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
  }
  return (reader, attn2)
}

func SpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  let hw = height * width
  for i in 0..<depth {
    let (reader, block) = BasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(block(c))
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([c], outs))
}

func BlockLayerFixed(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (transformerReader, transformer) = SpatialTransformerFixed(
    prefix: "\(prefix).\(layerStart).1",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingSize,
    intermediateSize: channels * 4)
  return (transformerReader, transformer)
}

func MiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  c: Model.IO
) -> ((PythonObject) -> Void, Model.IO) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (
    transformerReader, transformer
  ) = SpatialTransformerFixed(
    prefix: "middle_block.1",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, depth: 1,
    t: embeddingSize,
    intermediateSize: channels * 4)
  let out = transformer(c)
  return (transformerReader, out)
}

func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO
) -> ((PythonObject) -> Void, [Model.IO]) {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      if attentionBlock > 0 {
        let (reader, inputLayer) = BlockLayerFixed(
          prefix: "input_blocks",
          layerStart: layerStart, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        previousChannel = channel
        outs.append(inputLayer(c))
        readers.append(reader)
      }
      layerStart += 1
    }
    if i != channels.count - 1 {
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, outs)
}

func OutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO
) -> ((PythonObject) -> Void, [Model.IO]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<(numRepeat + 1) {
      if attentionBlock > 0 {
        let (reader, outputLayer) = BlockLayerFixed(
          prefix: "output_blocks",
          layerStart: layerStart, skipConnection: true,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        outs.append(outputLayer(c))
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, outs)
}

func UNetXLFixed(batchSize: Int) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  let attentionRes: [Int: Int] = [2: 2, 4: 10]
  let (inputReader, inputBlocks) = InputBlocksFixed(
    channels: [320, 640, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: 128, startWidth: 128, embeddingSize: 77, attentionRes: attentionRes, c: c)
  var out = inputBlocks
  let (middleReader, middleBlock) = MiddleBlockFixed(
    channels: 1280, numHeadChannels: 64, batchSize: batchSize, height: 32, width: 32,
    embeddingSize: 77, c: c)
  out.append(middleBlock)
  let (outputReader, outputBlocks) = OutputBlocksFixed(
    channels: [320, 640, 1280], numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: 128, startWidth: 128, embeddingSize: 77, attentionRes: attentionRes, c: c)
  out.append(contentsOf: outputBlocks)
  let reader: (PythonObject) -> Void = { state_dict in
    inputReader(state_dict)
    middleReader(state_dict)
    outputReader(state_dict)
  }
  return (reader, Model([c], out))
}

let random = Python.import("random")
let numpy = Python.import("numpy")
let torch = Python.import("torch")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([2, 4, 128, 128])
let c = torch.randn([2, 77, 2048])
let t = torch.full([1], 981)
let y = torch.randn([2, 2816])

let ret = state["model"].model(x.cuda(), t.cuda(), ["crossattn": c.cuda(), "vector": y.cuda()])
print(ret)

let graph = DynamicGraph()

let t_emb = graph.variable(
  timeEmbedding(timesteps: 981, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000)
).toGPU(1)
let xTensor = graph.variable(try! Tensor<Float>(numpy: x.numpy())).toGPU(1)
let cTensor = graph.variable(try! Tensor<Float>(numpy: c.numpy())).toGPU(1)
let yTensor = graph.variable(try! Tensor<Float>(numpy: y.numpy())).toGPU(1)
let (readerFixed, unetFixed) = UNetXLFixed(batchSize: 2)
let (reader, unet) = UNetXL(batchSize: 2)
graph.workspaceSize = 1_024 * 1_024 * 1_024
graph.withNoGrad {
  let _ = unetFixed(inputs: cTensor)
  readerFixed(state_dict)
  let kvs = unetFixed(inputs: cTensor).map { $0.as(of: Float.self) }
  let _ = unet(inputs: xTensor, [t_emb, yTensor] + kvs)
  reader(state_dict)
  let attnOut = unet(inputs: xTensor, [t_emb, yTensor] + kvs)[0].as(of: Float.self)
  debugPrint(attnOut)
  graph.openStore("/home/liu/workspace/swift-diffusion/unet.ckpt") {
    $0.write("unet_fixed", model: unetFixed)
    $0.write("unet", model: unet)
  }
}
*/
