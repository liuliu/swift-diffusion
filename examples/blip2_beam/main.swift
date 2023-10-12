import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

public typealias FloatType = Float16

let torch = Python.import("torch")
let Image = Python.import("PIL.Image")

let device = torch.device("cuda")

let raw_image = Image.open("/home/liu/workspace/blip2/merlion.png").convert("RGB")

let lavis = Python.import("lavis")

let (model, vis_processors, _) = lavis.models.load_model_and_preprocess(
  name: "blip2_opt", model_type: "caption_coco_opt2.7b", is_eval: true, device: device
).tuple3

let image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
print(image.shape)

func EvaSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let tokeys = Dense(count: k * h, noBias: true)
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
  let reader: (PythonObject) -> Void = { state_dict in
    let qkv = state_dict["\(prefix).attn.qkv.weight"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: qkv[0..<(k * h), 0..<(k * h)]))
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: qkv[(k * h)..<(2 * k * h), 0..<(k * h)]))
    tovalues.weight.copy(
      from: try! Tensor<Float>(numpy: qkv[(2 * k * h)..<(3 * k * h), 0..<(k * h)]))
    let q_bias = state_dict["\(prefix).attn.q_bias"].type(torch.float).cpu().numpy()
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_bias = state_dict["\(prefix).attn.v_bias"].type(torch.float).cpu().numpy()
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_weight = state_dict["\(prefix).attn.proj.weight"].type(torch.float).cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: proj_weight))
    let proj_bias = state_dict["\(prefix).attn.proj.bias"].type(torch.float).cpu().numpy()
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: proj_bias))
  }
  return (Model([x], [out]), reader)
}

func EvaResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int, MLP: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2])
  let (attention, attnReader) = EvaSelfAttention(prefix: prefix, k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [1])
  let fc = Dense(count: MLP)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
    let norm1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu().numpy()
    let norm1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu().numpy()
    ln1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    ln1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu().numpy()
    let norm2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu().numpy()
    ln2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    ln2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let fc1_weight = state_dict["\(prefix).mlp.fc1.weight"].type(torch.float).cpu().numpy()
    let fc1_bias = state_dict["\(prefix).mlp.fc1.bias"].type(torch.float).cpu().numpy()
    fc.weight.copy(from: try! Tensor<Float>(numpy: fc1_weight))
    fc.bias.copy(from: try! Tensor<Float>(numpy: fc1_bias))
    let fc2_weight = state_dict["\(prefix).mlp.fc2.weight"].type(torch.float).cpu().numpy()
    let fc2_bias = state_dict["\(prefix).mlp.fc2.bias"].type(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: fc2_weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: fc2_bias))
  }
  return (Model([x], [out]), reader)
}

public func EvaVisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type,
  grid: Int, width: Int, MLP: Int, layers: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  let classEmbedding = Parameter<T>(.GPU(0), .CHW(1, 1, width))
  let positionalEmbedding = Parameter<T>(.GPU(0), .CHW(1, grid * grid + 1, width))
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (block, reader) = EvaResidualAttentionBlock(
      prefix: "blocks.\(i)",
      k: width / heads, h: heads, b: batchSize, t: grid * grid + 1, MLP: MLP)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let patch_embed_proj_weight = state_dict["patch_embed.proj.weight"].type(torch.float).cpu()
      .numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: patch_embed_proj_weight))
    let patch_embed_proj_bias = state_dict["patch_embed.proj.bias"].type(torch.float).cpu().numpy()
    conv1.bias.copy(from: try! Tensor<Float>(numpy: patch_embed_proj_bias))
    let class_embedding = state_dict["cls_token"].type(torch.float).cpu().numpy()
    classEmbedding.weight.copy(from: try! Tensor<Float>(numpy: class_embedding))
    let positional_embedding = state_dict["pos_embed"].type(torch.float).cpu().numpy()
    positionalEmbedding.weight.copy(from: try! Tensor<Float>(numpy: positional_embedding))
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x], [out]), reader)
}

func BertSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
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
  let reader: (PythonObject) -> Void = { state_dict in
    let q_weight = state_dict["\(prefix).attention.self.query.weight"].type(torch.float).cpu()
      .numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    let k_weight = state_dict["\(prefix).attention.self.key.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    let v_weight = state_dict["\(prefix).attention.self.value.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    let q_bias = state_dict["\(prefix).attention.self.query.bias"].type(torch.float).cpu().numpy()
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let k_bias = state_dict["\(prefix).attention.self.key.bias"].type(torch.float).cpu().numpy()
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_bias))
    let v_bias = state_dict["\(prefix).attention.self.value.bias"].type(torch.float).cpu().numpy()
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_weight = state_dict["\(prefix).attention.output.dense.weight"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: proj_weight))
    let proj_bias = state_dict["\(prefix).attention.output.dense.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: proj_bias))
  }
  return (Model([x], [out]), reader)
}

func BertCrossAttention(
  prefix: String, k: Int, h: Int, b: Int, queryEmbeddingLength: Int, imageEmbeddingLength: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let c = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(c).reshaped([b, imageEmbeddingLength, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([
    b, queryEmbeddingLength, h, k,
  ])
  .permuted(0, 2, 1, 3)
  let values = tovalues(c).reshaped([b, imageEmbeddingLength, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * queryEmbeddingLength, imageEmbeddingLength])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, queryEmbeddingLength, imageEmbeddingLength])
  var out = dot * values
  out = out.reshaped([b, h, queryEmbeddingLength, k]).transposed(1, 2).reshaped([
    b * queryEmbeddingLength, h * k,
  ])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let q_weight = state_dict["\(prefix).crossattention.self.query.weight"].type(torch.float).cpu()
      .numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    let k_weight = state_dict["\(prefix).crossattention.self.key.weight"].type(torch.float).cpu()
      .numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    let v_weight = state_dict["\(prefix).crossattention.self.value.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    let q_bias = state_dict["\(prefix).crossattention.self.query.bias"].type(torch.float).cpu()
      .numpy()
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let k_bias = state_dict["\(prefix).crossattention.self.key.bias"].type(torch.float).cpu()
      .numpy()
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_bias))
    let v_bias = state_dict["\(prefix).crossattention.self.value.bias"].type(torch.float).cpu()
      .numpy()
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_weight = state_dict["\(prefix).crossattention.output.dense.weight"].type(torch.float)
      .cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: proj_weight))
    let proj_bias = state_dict["\(prefix).crossattention.output.dense.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: proj_bias))
  }
  return (Model([x, c], [out]), reader)
}

func BertLayer(
  prefix: String, k: Int, h: Int, b: Int, queryEmbeddingLength: Int, imageEmbeddingLength: Int,
  MLP: Int, hasCrossAttention: Bool
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let c: Model.IO?
  let (attention1, attnReader1) = BertSelfAttention(
    prefix: prefix, k: k, h: h, b: b, t: queryEmbeddingLength)
  var out = x.reshaped([b * queryEmbeddingLength, h * k]) + attention1(x)
  let ln1 = LayerNorm(epsilon: 1e-12, axis: [1])
  out = ln1(out)
  let attnReader2: ((PythonObject) -> Void)?
  let ln2: Model?
  if hasCrossAttention {
    let lc = Input()
    let (attention2, lattnReader2) = BertCrossAttention(
      prefix: prefix, k: k, h: h, b: b, queryEmbeddingLength: queryEmbeddingLength,
      imageEmbeddingLength: imageEmbeddingLength)
    out = out.reshaped([b * queryEmbeddingLength, h * k]) + attention2(out, lc)
    let lln2 = LayerNorm(epsilon: 1e-12, axis: [1])
    out = lln2(out)
    c = lc
    attnReader2 = lattnReader2
    ln2 = lln2
  } else {
    c = nil
    attnReader2 = nil
    ln2 = nil
  }
  let fc = Dense(count: MLP)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  let ln3 = LayerNorm(epsilon: 1e-12, axis: [1])
  out = ln3(out + proj(gelu(fc(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader1(state_dict)
    let norm1_weight = state_dict["\(prefix).attention.output.LayerNorm.weight"].type(torch.float)
      .cpu().numpy()
    let norm1_bias = state_dict["\(prefix).attention.output.LayerNorm.bias"].type(torch.float).cpu()
      .numpy()
    ln1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    ln1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    attnReader2?(state_dict)
    if let ln2 = ln2 {
      let norm2_weight = state_dict["\(prefix).crossattention.output.LayerNorm.weight"].type(
        torch.float
      ).cpu().numpy()
      let norm2_bias = state_dict["\(prefix).crossattention.output.LayerNorm.bias"].type(
        torch.float
      ).cpu().numpy()
      ln2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
      ln2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    }
    let fc1_weight = state_dict["\(prefix).intermediate_query.dense.weight"].type(torch.float).cpu()
      .numpy()
    let fc1_bias = state_dict["\(prefix).intermediate_query.dense.bias"].type(torch.float).cpu()
      .numpy()
    fc.weight.copy(from: try! Tensor<Float>(numpy: fc1_weight))
    fc.bias.copy(from: try! Tensor<Float>(numpy: fc1_bias))
    let fc2_weight = state_dict["\(prefix).output_query.dense.weight"].type(torch.float).cpu()
      .numpy()
    let fc2_bias = state_dict["\(prefix).output_query.dense.bias"].type(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: fc2_weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: fc2_bias))
    let ln3_weight = state_dict["\(prefix).output_query.LayerNorm.weight"].type(torch.float).cpu()
      .numpy()
    let ln3_bias = state_dict["\(prefix).output_query.LayerNorm.bias"].type(torch.float).cpu()
      .numpy()
    ln3.weight.copy(from: try! Tensor<Float>(numpy: ln3_weight))
    ln3.bias.copy(from: try! Tensor<Float>(numpy: ln3_bias))
  }
  if let c = c {
    return (Model([x, c], [out]), reader)
  } else {
    return (Model([x], [out]), reader)
  }
}

func BertModel(
  width: Int, queryEmbeddingLength: Int, imageEmbeddingLength: Int, MLP: Int, layers: Int,
  heads: Int, batchSize: Int, crossAttentionFreq: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let c = Input()
  let ln = LayerNorm(epsilon: 1e-12, axis: [1])
  var out = ln(x)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let hasCrossAttention = (i % crossAttentionFreq) == 0
    let (layer, reader) = BertLayer(
      prefix: "encoder.layer.\(i)", k: width / heads, h: heads, b: batchSize,
      queryEmbeddingLength: queryEmbeddingLength, imageEmbeddingLength: imageEmbeddingLength,
      MLP: MLP, hasCrossAttention: hasCrossAttention)
    if hasCrossAttention {
      out = layer(out, c)
    } else {
      out = layer(out)
    }
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_weight = state_dict["embeddings.LayerNorm.weight"].type(torch.float).cpu().numpy()
    let ln_bias = state_dict["embeddings.LayerNorm.bias"].type(torch.float).cpu().numpy()
    ln.weight.copy(from: try! Tensor<Float>(numpy: ln_weight))
    ln.bias.copy(from: try! Tensor<Float>(numpy: ln_bias))
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x, c], [out]), reader)
}

func OPTSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * h, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * h, name: "v_proj")
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let q_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).cpu()
      .numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    let k_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    let v_weight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    let q_bias = state_dict["\(prefix).self_attn.q_proj.bias"].type(torch.float).cpu().numpy()
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let k_bias = state_dict["\(prefix).self_attn.k_proj.bias"].type(torch.float).cpu().numpy()
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_bias))
    let v_bias = state_dict["\(prefix).self_attn.v_proj.bias"].type(torch.float).cpu().numpy()
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_weight = state_dict["\(prefix).self_attn.out_proj.weight"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: proj_weight))
    let proj_bias = state_dict["\(prefix).self_attn.out_proj.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: proj_bias))
  }
  return (Model([x, causalAttentionMask], [out]), reader)
}

func OPTDecodeLayer(prefix: String, k: Int, h: Int, b: Int, t: Int, MLP: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let causalAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1], name: "self_attn_layer_norm")
  var out = layerNorm1(x)
  let (attention, attnReader) = OPTSelfAttention(prefix: prefix, k: k, h: h, b: b, t: t)
  out = attention(out, causalAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1], name: "final_layer_norm")
  out = layerNorm2(out)
  let fc = Dense(count: MLP, name: "fc1")
  let relu = ReLU()
  let proj = Dense(count: k * h, name: "fc2")
  out = residual + proj(relu(fc(out)))
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
    let norm1_weight = state_dict["\(prefix).self_attn_layer_norm.weight"].type(torch.float)
      .cpu().numpy()
    let norm1_bias = state_dict["\(prefix).self_attn_layer_norm.bias"].type(torch.float).cpu()
      .numpy()
    layerNorm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    layerNorm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let norm2_weight = state_dict["\(prefix).final_layer_norm.weight"].type(torch.float)
      .cpu().numpy()
    let norm2_bias = state_dict["\(prefix).final_layer_norm.bias"].type(torch.float).cpu()
      .numpy()
    layerNorm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    layerNorm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let fc1_weight = state_dict["\(prefix).fc1.weight"].type(torch.float).cpu()
      .numpy()
    let fc1_bias = state_dict["\(prefix).fc1.bias"].type(torch.float).cpu()
      .numpy()
    fc.weight.copy(from: try! Tensor<Float>(numpy: fc1_weight))
    fc.bias.copy(from: try! Tensor<Float>(numpy: fc1_bias))
    let fc2_weight = state_dict["\(prefix).fc2.weight"].type(torch.float).cpu()
      .numpy()
    let fc2_bias = state_dict["\(prefix).fc2.bias"].type(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: fc2_weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: fc2_bias))
  }
  return (Model([x, causalAttentionMask], [out]), reader)
}

public func OPTTextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let queryEmbed = Input()
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "embed_tokens")
  let positionEmbed = Embedding(
    T.self, vocabularySize: maxLength, embeddingSize: embeddingSize, name: "embed_positions")
  let embedding =
    Functional.concat(axis: 0, queryEmbed, tokenEmbed(tokens)) + positionEmbed(positions)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["model.decoder.embed_tokens.weight"].type(torch.float).cpu().numpy()
    let pos = state_dict["model.decoder.embed_positions.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab))
    positionEmbed.parameters.copy(from: try! Tensor<Float>(numpy: pos))
  }
  return (Model([queryEmbed, tokens, positions], [embedding], name: "embeddings"), reader)
}

func OPTDecoder<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let queryEmbed = Input()
  let tokens = Input()
  let positions = Input()
  let (embedding, embedReader) = OPTTextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(queryEmbed, tokens, positions)
  let causalAttentionMask = Input()
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (layer, reader) = OPTDecodeLayer(
      prefix: "model.decoder.layers.\(i)", k: width / heads, h: heads, b: batchSize, t: tokenLength,
      MLP: MLP)
    out = layer(out, causalAttentionMask)
    readers.append(reader)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1], name: "final_layer_norm")
  out = finalLayerNorm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_norm_weight = state_dict["model.decoder.final_layer_norm.weight"].type(
      torch.float
    ).cpu().numpy()
    let final_layer_norm_bias = state_dict["model.decoder.final_layer_norm.bias"].type(torch.float)
      .cpu().numpy()
    finalLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: final_layer_norm_weight))
    finalLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: final_layer_norm_bias))
  }
  return (Model([queryEmbed, tokens, positions, causalAttentionMask], [out]), reader)
}

let graph = DynamicGraph()
let (vit, _) = EvaVisionTransformer(
  FloatType.self,
  grid: 26, width: 1408, MLP: 6144, layers: 39, heads: 16, batchSize: 1)
let ln_vision = LayerNorm(epsilon: 1e-5, axis: [1])
let (qformer, _) = BertModel(
  width: 768, queryEmbeddingLength: 32, imageEmbeddingLength: 26 * 26 + 1, MLP: 768 * 4, layers: 12,
  heads: 12, batchSize: 1, crossAttentionFreq: 2)
let (opt, _) = OPTDecoder(
  FloatType.self, vocabularySize: 50272, maxLength: 2050, width: 2560, tokenLength: 47, layers: 32,
  MLP: 10240, heads: 32, batchSize: 1)
graph.withNoGrad {
  let xTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: image.cpu().numpy()))
  ).toGPU(0)
  vit.compile(inputs: xTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/blip2_eva_vit_q8p.ckpt") {
    $0.read("vit", model: vit, codec: [.jit, .q8p, .ezm7])
  }
  var out = vit(inputs: xTensor)[0].as(of: FloatType.self)
  ln_vision.compile(inputs: out)
  graph.openStore("/home/liu/workspace/swift-diffusion/blip2_qformer_f16.ckpt") {
    $0.read("ln_vision", model: ln_vision)
  }
  out = ln_vision(inputs: out)[0].as(of: FloatType.self)
  let queryTokensTensor = graph.variable(.GPU(0), .NC(32, 768), of: FloatType.self)
  qformer.compile(inputs: queryTokensTensor, out)
  graph.openStore("/home/liu/workspace/swift-diffusion/blip2_qformer_f16.ckpt") {
    $0.read("query_tokens", variable: queryTokensTensor)
    $0.read("qformer", model: qformer)
  }
  out = qformer(inputs: queryTokensTensor, out)[0].as(of: FloatType.self)
  let optProj = Dense(count: 2560)
  optProj.compile(inputs: out)
  graph.openStore("/home/liu/workspace/swift-diffusion/blip2_qformer_f16.ckpt") {
    $0.read("opt_proj", model: optProj)
  }
  out = optProj(inputs: out)[0].as(of: FloatType.self)
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/blip2_eva_vit_f16.ckpt") {
    $0.write("vit", model: vit)
  }
  graph.openStore("/home/liu/workspace/swift-diffusion/blip2_qformer_f16.ckpt") {
    $0.write("ln_vision", model: ln_vision)
    $0.write("query_tokens", variable: queryTokensTensor)
    $0.write("qformer", model: qformer)
    $0.write("opt_proj", model: optProj)
  }
  */
  let tokensTensor = graph.variable(.CPU, .C(15), of: Int32.self)
  let positionTensor = graph.variable(.CPU, .C(15 + 32), of: Int32.self)
  tokensTensor[0] = 2
  tokensTensor[1] = 102
  tokensTensor[2] = 1345
  tokensTensor[3] = 9
  tokensTensor[4] = 10
  tokensTensor[5] = 9577
  tokensTensor[6] = 9
  tokensTensor[7] = 10
  tokensTensor[8] = 9374
  tokensTensor[9] = 16355
  tokensTensor[10] = 30652
  tokensTensor[11] = 514
  tokensTensor[12] = 88
  tokensTensor[13] = 5
  tokensTensor[14] = 935
  // tokensTensor[15] = 7886
  for i in 0..<(15 + 32) {
    positionTensor[i] = Int32(2 + i)
  }
  let causalAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 47, 47)))
  causalAttentionMask.full(0)
  for i in 0..<46 {
    for j in (i + 1)..<47 {
      causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
  opt.compile(inputs: out, tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/opt_2.7b_q6p.ckpt") {
    $0.read("opt", model: opt, codec: [.jit, .q6p, .ezm7])
  }
  out = opt(inputs: out, tokensTensorGPU, positionTensorGPU, causalAttentionMaskGPU)[0].as(
    of: FloatType.self)
  let lastOut = out[46..<47, 0..<2560]
  let lmHead = Dense(count: 50272, noBias: true)
  lmHead.compile(inputs: lastOut)
  graph.openStore("/home/liu/workspace/swift-diffusion/opt_2.7b_q6p.ckpt") {
    $0.read("lm_head", model: lmHead, codec: [.jit, .q6p, .ezm7])
  }
  out = lmHead(inputs: lastOut)[0].as(of: FloatType.self).toCPU()
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/opt_2.7b_f16.ckpt") {
    $0.write("opt", model: opt)
    $0.write("lm_head", model: lmHead)
  }
  */
  var maxIdx: Int = -1
  var maxVal: FloatType = -FloatType.greatestFiniteMagnitude
  for i in 0..<50272 {
    if out[0, i] > maxVal {
      maxVal = out[0, i]
      maxIdx = i
    }
  }
  print("next token: \(maxIdx)")
}
