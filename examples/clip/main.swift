import NNC
import NNCPythonConversion
import PythonKit

func CLIPTextEmbedding() -> (Model, Model, Model) {
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(Float.self, vocabularySize: 49408, embeddingSize: 768)
  let positionEmbed = Embedding(Float.self, vocabularySize: 77, embeddingSize: 768)
  let embedding = tokenEmbed(tokens) + positionEmbed(positions)
  return (tokenEmbed, positionEmbed, Model([tokens, positions], [embedding], name: "embeddings"))
}

func CLIPAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let casualAttentionMask = Input()
  let multiheads = x.reshaped([b * t, k * h])
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(multiheads).reshaped([t, b, h, k]).transposed(0, 2).reshaped([b * h, t, k])
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(multiheads)).reshaped([t, b, h, k])
    .transposed(0, 2).reshaped([b * h, t, k])
  let values = tovalues(multiheads).reshaped([t, b, h, k]).transposed(0, 2).reshaped([b * h, t, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys) + casualAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b * h, t, t])
  var out = dot * values
  out = out.reshaped([h, b, t, k]).transposed(0, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out).reshaped([t, b, k * h])
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, casualAttentionMask], [out]))
}

let transformers = Python.import("transformers")

let tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
let transformer = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

let batch_encoding = tokenizer(
  ["a photograph of an astronaut riding a horse"], truncation: true, max_length: 77,
  return_length: true, return_overflowing_tokens: false, padding: "max_length", return_tensors: "pt"
)
let tokens = batch_encoding["input_ids"]
let outputs = transformer(input_ids: tokens)
let state_dict = transformer.state_dict()

let (tokenEmbed, positionEmbed, textEmbedding) = CLIPTextEmbedding()

let graph = DynamicGraph()
let tokensTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensNumpy = tokens.numpy()
for i in 0..<77 {
  tokensTensor[i] = Int32(tokensNumpy[0, i])!
  positionTensor[i] = Int32(i)
}

let _ = textEmbedding(inputs: tokensTensor, positionTensor)

// let vocab = state_dict["text_model.embeddings.token_embedding.weight"]
// let pos = state_dict["text_model.embeddings.position_embedding.weight"]
// tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab.numpy()))
// positionEmbed.parameters.copy(from: try! Tensor<Float>(numpy: pos.numpy()))

graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
  $0.read("text_model", model: textEmbedding)
}

let outputTensor = textEmbedding(inputs: tokensTensor, positionTensor)[0].as(of: Float.self)

let casualAttentionMask = graph.variable(Tensor<Float>(.CPU, .HWC(1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, i, j] = -Float.greatestFiniteMagnitude
  }
}

// graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
//   $0.write("text_model", model: textEmbedding)
// }
// print(state_dict.keys())
let layer_norm_1_weight = state_dict["text_model.encoder.layers.0.layer_norm1.weight"].numpy()
let layer_norm_1_bias = state_dict["text_model.encoder.layers.0.layer_norm1.bias"].numpy()
let layerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
let _ = layerNorm(outputTensor)
layerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: layer_norm_1_weight))
layerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: layer_norm_1_bias))
let normOutput = layerNorm(outputTensor)
let (tokeys, toqueries, tovalues, unifyheads, attention) = CLIPAttention(k: 64, h: 12, b: 1, t: 77)
let _ = attention(inputs: normOutput, casualAttentionMask)
let k_proj_weight = state_dict["text_model.encoder.layers.0.self_attn.k_proj.weight"].numpy()
let k_proj_bias = state_dict["text_model.encoder.layers.0.self_attn.k_proj.bias"].numpy()
let v_proj_weight = state_dict["text_model.encoder.layers.0.self_attn.v_proj.weight"].numpy()
let v_proj_bias = state_dict["text_model.encoder.layers.0.self_attn.v_proj.bias"].numpy()
let q_proj_weight = state_dict["text_model.encoder.layers.0.self_attn.q_proj.weight"].numpy()
let q_proj_bias = state_dict["text_model.encoder.layers.0.self_attn.q_proj.bias"].numpy()
let out_proj_weight = state_dict["text_model.encoder.layers.0.self_attn.out_proj.weight"].numpy()
let out_proj_bias = state_dict["text_model.encoder.layers.0.self_attn.out_proj.bias"].numpy()
tokeys.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: k_proj_weight))
tokeys.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: k_proj_bias))
tovalues.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: v_proj_weight))
tovalues.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: v_proj_bias))
toqueries.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: q_proj_weight))
toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: q_proj_bias))
unifyheads.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_proj_weight))
unifyheads.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_proj_bias))
let attnOutput = attention(inputs: normOutput, casualAttentionMask)[0].as(of: Float.self)
for i in 0..<6 {
  let x = i < 3 ? i : 71 + i
  for j in 0..<6 {
    let y = j < 3 ? j : 762 + j
    print("\(x) \(y) \(attnOutput[0, x, y])")
  }
}
