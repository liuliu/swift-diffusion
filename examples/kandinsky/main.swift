import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let kandinsky2 = Python.import("kandinsky2")
let torch = Python.import("torch")
let random = Python.import("random")
let numpy = Python.import("numpy")

let model = kandinsky2.get_kandinsky2(
  "cuda", task_type: "text2img", model_version: "2.1", use_flash_attention: false)
let state_dict = model.text_encoder.state_dict()
print(model.text_encoder)
print(state_dict.keys())

func XLMRobertaTextEmbedding(
  prefix: String, vocabularySize: Int, maxLength: Int, tokenTypes: Int, embeddingSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let tokens = Input()
  let tokenType = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let tokenTypeEmbed = Embedding(
    Float.self, vocabularySize: tokenTypes, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(Float.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + tokenTypeEmbed(tokenType) + positionEmbed(positions)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  let out = layerNorm(embedding)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["\(prefix).word_embeddings.weight"].type(torch.float).cpu().numpy()
    let token_type = state_dict["\(prefix).token_type_embeddings.weight"].type(torch.float).cpu()
      .numpy()
    let pos = state_dict["\(prefix).position_embeddings.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab))
    tokenTypeEmbed.parameters.copy(from: try! Tensor<Float>(numpy: token_type))
    positionEmbed.parameters.copy(from: try! Tensor<Float>(numpy: pos))
    let layer_norm_weight = state_dict["\(prefix).LayerNorm.weight"].type(torch.float).cpu().numpy()
    let layer_norm_bias = state_dict["\(prefix).LayerNorm.bias"].type(torch.float).cpu().numpy()
    layerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: layer_norm_weight))
    layerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: layer_norm_bias))
  }
  return (reader, Model([tokens, positions, tokenType], [out], name: "embeddings"))
}

func XLMRobertaSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
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
  let reader: (PythonObject) -> Void = { state_dict in
    let self_key_weight = state_dict["\(prefix).self.key.weight"].type(torch.float).cpu().numpy()
    let self_key_bias = state_dict["\(prefix).self.key.bias"].type(torch.float).cpu().numpy()
    tokeys.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: self_key_weight))
    tokeys.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: self_key_bias))
    let self_query_weight = state_dict["\(prefix).self.query.weight"].type(torch.float).cpu()
      .numpy()
    let self_query_bias = state_dict["\(prefix).self.query.bias"].type(torch.float).cpu().numpy()
    toqueries.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: self_query_weight))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: self_query_bias))
    let self_value_weight = state_dict["\(prefix).self.value.weight"].type(torch.float).cpu()
      .numpy()
    let self_value_bias = state_dict["\(prefix).self.value.bias"].type(torch.float).cpu().numpy()
    tovalues.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: self_value_weight))
    tovalues.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: self_value_bias))
    let output_dense_weight = state_dict["\(prefix).output.dense.weight"].type(torch.float).cpu()
      .numpy()
    let output_dense_bias = state_dict["\(prefix).output.dense.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: output_dense_weight))
    unifyheads.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: output_dense_bias))
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

func XLMRobertaLayer(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let casualAttentionMask = Input()
  let (selfAttentionReader, selfAttention) = XLMRobertaSelfAttention(
    prefix: "\(prefix).attention", k: k, h: h, b: b, t: t)
  var out = selfAttention(x, casualAttentionMask)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm(out + x)
  let intermediate = Dense(count: k * h * 4)
  let ff = out
  out = intermediate(out).GELU()
  let output = Dense(count: k * h)
  out = output(out)
  let layerNormFinal = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNormFinal(out + ff)
  let reader: (PythonObject) -> Void = { state_dict in
    selfAttentionReader(state_dict)
    let attention_output_layerNorm_weight = state_dict[
      "\(prefix).attention.output.LayerNorm.weight"
    ].type(torch.float).cpu().numpy()
    let attention_output_layerNorm_bias = state_dict["\(prefix).attention.output.LayerNorm.bias"]
      .type(torch.float).cpu().numpy()
    layerNorm.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attention_output_layerNorm_weight))
    layerNorm.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: attention_output_layerNorm_bias))
    let intermediate_dense_weight = state_dict["\(prefix).intermediate.dense.weight"].type(
      torch.float
    ).cpu().numpy()
    let intermediate_dense_bias = state_dict["\(prefix).intermediate.dense.bias"].type(torch.float)
      .cpu().numpy()
    intermediate.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: intermediate_dense_weight))
    intermediate.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: intermediate_dense_bias))
    let output_dense_weight = state_dict["\(prefix).output.dense.weight"].type(torch.float).cpu()
      .numpy()
    let output_dense_bias = state_dict["\(prefix).output.dense.bias"].type(torch.float).cpu()
      .numpy()
    output.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: output_dense_weight))
    output.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: output_dense_bias))
    let output_layerNorm_weight = state_dict["\(prefix).output.LayerNorm.weight"].type(torch.float)
      .cpu().numpy()
    let output_layerNorm_bias = state_dict["\(prefix).output.LayerNorm.bias"].type(torch.float)
      .cpu().numpy()
    layerNormFinal.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: output_layerNorm_weight))
    layerNormFinal.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: output_layerNorm_bias))
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

func XLMRobertaModel(numberOfLayers: Int, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  var readers = [(PythonObject) -> Void]()
  let x = Input()
  let casualAttentionMask = Input()
  var out: Model.IO = x
  for i in 0..<numberOfLayers {
    let (reader, layer) = XLMRobertaLayer(
      prefix: "model.transformer.encoder.layer.\(i)", k: k, h: h, b: b, t: t)
    out = layer(out, casualAttentionMask)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

let (reader, textEncoder) = XLMRobertaTextEmbedding(
  prefix: "model.transformer.embeddings", vocabularySize: 250_002, maxLength: 514, tokenTypes: 1,
  embeddingSize: 1_024)
let graph = DynamicGraph()
var tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
for i in 0..<154 {
  tokensTensor[i] = 1
}
tokensTensor[0] = 0
tokensTensor[1] = 4842
tokensTensor[2] = 7515
tokensTensor[3] = 4
tokensTensor[4] = 201
tokensTensor[5] = 92
tokensTensor[6] = 16186
tokensTensor[7] = 2
tokensTensor[77] = 0
tokensTensor[78] = 2
var tokenTypesTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
var positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
for i in 0..<154 {
  tokenTypesTensor[i] = 0
  positionTensor[i] = 1
}
for i in 0..<8 {
  positionTensor[i] = Int32(i + 2)
}
positionTensor[77] = 2
positionTensor[78] = 3
textEncoder.compile(inputs: tokensTensor, positionTensor, tokenTypesTensor)
reader(state_dict)
let embeddings = textEncoder(inputs: tokensTensor, positionTensor, tokenTypesTensor)[0].as(
  of: Float.self)
let (layerReader, layer) = XLMRobertaModel(numberOfLayers: 24, k: 64, h: 16, b: 2, t: 77)
let attentionMask = graph.variable(.CPU, .NCHW(2, 1, 1, 77), of: Float.self)
attentionMask.full(0)
for i in 8..<77 {
  attentionMask[0, 0, 0, i] = -Float.greatestFiniteMagnitude
}
for i in 2..<77 {
  attentionMask[1, 0, 0, i] = -Float.greatestFiniteMagnitude
}
layer.compile(inputs: embeddings, attentionMask)
layerReader(state_dict)
let output = layer(inputs: embeddings, attentionMask)[0].as(of: Float.self)
debugPrint(output.reshaped(.CHW(2, 77, 1024)))
let images = model.generate_text2img(
  "red cat, 4k photo", num_steps: 100, batch_size: 1, guidance_scale: 4, h: 768, w: 768,
  sampler: "p_sampler", prior_cf_scale: 4, prior_steps: "5")
images[0].save("/home/liu/workspace/swift-diffusion/kandinsky.png")
