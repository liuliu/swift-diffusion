import NNC
import NNCPythonConversion
import PythonKit

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

let open_clip = Python.import("open_clip")
let torch = Python.import("torch")

let (model, _, _) = open_clip.create_model_and_transforms(
  "ViT-H-14", device: torch.device("cpu"), pretrained: "laion2b_s32b_b79k"
).tuple3

let tokens = open_clip.tokenize([""])  // a professional photograph of an astronaut riding a horse"])
var x = model.token_embedding(tokens)
x = x + model.positional_embedding
x = x.permute(1, 0, 2)
for (i, r) in model.transformer.resblocks.enumerated() {
  if i == Int(Python.len(model.transformer.resblocks))! - 1 {
    break
  }
  x = r(x, attn_mask: model.attn_mask)
}
x = x.permute(1, 0, 2)
x = model.ln_final(x)
print(x)
// let state_dict = model.state_dict()

let pl_sd = torch.load(
  "/home/liu/workspace/stablediffusion/models/stable-diffusion-v2/redshift-diffusion-768.ckpt",
  map_location: "cpu")
let state_dict = pl_sd["state_dict"]

let (
  tokenEmbed, positionEmbed, layerNorm1s, tokeys, toqueries, tovalues, unifyheads, layerNorm2s,
  fc1s, fc2s, finalLayerNorm, textModel
) = CLIPTextModel(
  vocabularySize: 49408, maxLength: 77, embeddingSize: 1024, numLayers: 23, numHeads: 16,
  batchSize: 1, intermediateSize: 4096)

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

let vocab = state_dict["cond_stage_model.model.token_embedding.weight"].float().numpy()
let pos = state_dict["cond_stage_model.model.positional_embedding"].float().numpy()
tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab))
positionEmbed.parameters.copy(from: try! Tensor<Float>(numpy: pos))

for i in 0..<23 {
  let layer_norm_1_weight = state_dict[
    "cond_stage_model.model.transformer.resblocks.\(i).ln_1.weight"
  ].float().numpy()
  let layer_norm_1_bias = state_dict["cond_stage_model.model.transformer.resblocks.\(i).ln_1.bias"]
    .float().numpy()
  layerNorm1s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: layer_norm_1_weight))
  layerNorm1s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: layer_norm_1_bias))

  let in_proj_weight = state_dict[
    "cond_stage_model.model.transformer.resblocks.\(i).attn.in_proj_weight"
  ].float().numpy()
  let in_proj_bias = state_dict[
    "cond_stage_model.model.transformer.resblocks.\(i).attn.in_proj_bias"
  ].float().numpy()
  toqueries[i].parameters(for: .weight).copy(
    from: try! Tensor<Float>(numpy: in_proj_weight[..<(1024), ...]))
  toqueries[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_proj_bias[..<(1024)]))
  tokeys[i].parameters(for: .weight).copy(
    from: try! Tensor<Float>(numpy: in_proj_weight[(1024)..<(2 * 1024), ...]))
  tokeys[i].parameters(for: .bias).copy(
    from: try! Tensor<Float>(numpy: in_proj_bias[(1024)..<(2 * 1024)]))
  tovalues[i].parameters(for: .weight).copy(
    from: try! Tensor<Float>(numpy: in_proj_weight[(2 * 1024)..., ...]))
  tovalues[i].parameters(for: .bias).copy(
    from: try! Tensor<Float>(numpy: in_proj_bias[(2 * 1024)...]))

  let out_proj_weight = state_dict[
    "cond_stage_model.model.transformer.resblocks.\(i).attn.out_proj.weight"
  ].float()
    .numpy()
  let out_proj_bias = state_dict[
    "cond_stage_model.model.transformer.resblocks.\(i).attn.out_proj.bias"
  ].float().numpy()
  unifyheads[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_proj_weight))
  unifyheads[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_proj_bias))

  let layer_norm_2_weight = state_dict[
    "cond_stage_model.model.transformer.resblocks.\(i).ln_2.weight"
  ].float().numpy()
  let layer_norm_2_bias = state_dict["cond_stage_model.model.transformer.resblocks.\(i).ln_2.bias"]
    .float().numpy()
  layerNorm2s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: layer_norm_2_weight))
  layerNorm2s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: layer_norm_2_bias))

  let fc1_weight = state_dict["cond_stage_model.model.transformer.resblocks.\(i).mlp.c_fc.weight"]
    .float().numpy()
  let fc1_bias = state_dict["cond_stage_model.model.transformer.resblocks.\(i).mlp.c_fc.bias"]
    .float().numpy()
  fc1s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: fc1_weight))
  fc1s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: fc1_bias))

  let fc2_weight = state_dict["cond_stage_model.model.transformer.resblocks.\(i).mlp.c_proj.weight"]
    .float().numpy()
  let fc2_bias = state_dict["cond_stage_model.model.transformer.resblocks.\(i).mlp.c_proj.bias"]
    .float().numpy()
  fc2s[i].parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: fc2_weight))
  fc2s[i].parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: fc2_bias))
}

let final_layer_norm_weight = state_dict["cond_stage_model.model.ln_final.weight"].float().numpy()
let final_layer_norm_bias = state_dict["cond_stage_model.model.ln_final.bias"].float().numpy()
finalLayerNorm.parameters(for: .weight).copy(
  from: try! Tensor<Float>(numpy: final_layer_norm_weight))
finalLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: final_layer_norm_bias))

let c = textModel(inputs: tokensTensor, positionTensor, casualAttentionMask)[0].as(of: Float.self)

graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
  $0.write("text_model", model: textModel)
}
