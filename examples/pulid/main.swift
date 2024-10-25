import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let pulid_pipeline_flux = Python.import("pulid.pipeline_flux")

let flux_util = Python.import("flux.util")

let torch = Python.import("torch")

let numpy = Python.import("numpy")

let model = flux_util.load_flow_model("flux-dev", device: "cpu")

let pipeline = pulid_pipeline_flux.PuLIDPipeline(
  model, device: "cpu", weight_dtype: torch.bfloat16, onnx_provider: "gpu")

pipeline.load_pretrain()

torch.set_grad_enabled(false)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([1, 3, 336, 336]).to(torch.float).cuda()
let img = torch.randn([1, 4096, 3072]).to(torch.float).cuda()

pipeline.clip_vision_model = pipeline.clip_vision_model.to(torch.float).cuda()

var (id_cond_vit, id_vit_hidden) = pipeline.clip_vision_model(
  x, return_all_features: false, return_hidden: true, shuffle: false
).tuple2

let id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, true)

id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

print("id_cond_vit \(id_cond_vit) \(id_cond_vit.shape)")

let id_ante_embedding = torch.rand([1, 512]).to(torch.float).cuda()
let id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim: -1)

pipeline.pulid_encoder = pipeline.pulid_encoder.to(torch.float).cuda()
print("id_vit_hidden \(id_vit_hidden)")

let id_embedding = pipeline.pulid_encoder(id_cond, id_vit_hidden)

print("id_embedding \(id_embedding) \(id_embedding.shape)")
pipeline.pulid_ca = pipeline.pulid_ca.to(torch.float).cuda()
print(pipeline.pulid_ca)

for i in 0..<20 {
  let out = pipeline.pulid_ca[i](id_embedding, img)
  print("out \(out) \(out.shape)")
}

func EvaSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, name: "q")
  let tovalues = Dense(count: k * h, name: "v")
  var keys = tokeys(x).reshaped([b, t, h, k])
  var queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
  queries = Functional.cmul(left: queries, right: rot).permuted(0, 2, 1, 3)
  keys = Functional.cmul(left: keys, right: rot).permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let innerAttnLn = LayerNorm(epsilon: 1e-6, axis: [1], name: "inner_attn_ln")
  let unifyheads = Dense(count: k * h, name: "proj")
  out = unifyheads(innerAttnLn(out))
  let reader: (PythonObject) -> Void = { state_dict in
    let q_proj = state_dict["\(prefix).attn.q_proj.weight"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_proj))
    let k_proj = state_dict["\(prefix).attn.k_proj.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_proj))
    let v_proj = state_dict["\(prefix).attn.v_proj.weight"].type(torch.float).cpu().numpy()
    tovalues.weight.copy(
      from: try! Tensor<Float>(numpy: v_proj))
    let q_bias = state_dict["\(prefix).attn.q_bias"].type(torch.float).cpu().numpy()
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_bias = state_dict["\(prefix).attn.v_bias"].type(torch.float).cpu().numpy()
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let inner_attn_ln_weight = state_dict["\(prefix).attn.inner_attn_ln.weight"].type(torch.float)
      .cpu().numpy()
    innerAttnLn.weight.copy(from: try! Tensor<Float>(numpy: inner_attn_ln_weight))
    let inner_attn_ln_bias = state_dict["\(prefix).attn.inner_attn_ln.bias"].type(torch.float).cpu()
      .numpy()
    innerAttnLn.bias.copy(from: try! Tensor<Float>(numpy: inner_attn_ln_bias))
    let proj_weight = state_dict["\(prefix).attn.proj.weight"].type(torch.float).cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: proj_weight))
    let proj_bias = state_dict["\(prefix).attn.proj.bias"].type(torch.float).cpu().numpy()
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: proj_bias))
  }
  return (Model([x, rot], [out]), reader)
}

func EvaResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int, MLP: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2], name: "ln1")
  let (attention, attnReader) = EvaSelfAttention(prefix: prefix, k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x), rot)
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [1], name: "ln2")
  let w1 = Dense(count: MLP, name: "w1")
  let w2 = Dense(count: MLP, name: "w2")
  let ffnLn = LayerNorm(epsilon: 1e-6, axis: [1], name: "ffn_ln")
  let w3 = Dense(count: k * h, name: "w3")
  let residual = out
  out = ln2(out)
  out = residual + w3(ffnLn(w1(out).swish() .* w2(out)))
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
    let w1_weight = state_dict["\(prefix).mlp.w1.weight"].type(torch.float).cpu().numpy()
    let w1_bias = state_dict["\(prefix).mlp.w1.bias"].type(torch.float).cpu().numpy()
    w1.weight.copy(from: try! Tensor<Float>(numpy: w1_weight))
    w1.bias.copy(from: try! Tensor<Float>(numpy: w1_bias))
    let w2_weight = state_dict["\(prefix).mlp.w2.weight"].type(torch.float).cpu().numpy()
    let w2_bias = state_dict["\(prefix).mlp.w2.bias"].type(torch.float).cpu().numpy()
    w2.weight.copy(from: try! Tensor<Float>(numpy: w2_weight))
    w2.bias.copy(from: try! Tensor<Float>(numpy: w2_bias))
    let ffn_ln_weight = state_dict["\(prefix).mlp.ffn_ln.weight"].type(torch.float).cpu().numpy()
    let ffn_ln_bias = state_dict["\(prefix).mlp.ffn_ln.bias"].type(torch.float).cpu().numpy()
    ffnLn.weight.copy(from: try! Tensor<Float>(numpy: ffn_ln_weight))
    ffnLn.bias.copy(from: try! Tensor<Float>(numpy: ffn_ln_bias))
    let w3_weight = state_dict["\(prefix).mlp.w3.weight"].type(torch.float).cpu().numpy()
    let w3_bias = state_dict["\(prefix).mlp.w3.bias"].type(torch.float).cpu().numpy()
    w3.weight.copy(from: try! Tensor<Float>(numpy: w3_weight))
    w3.bias.copy(from: try! Tensor<Float>(numpy: w3_bias))
  }
  return (Model([x, rot], [out]), reader)
}

public func EvaVisionTransformer(
  grid: Int, outputChannels: Int, width: Int, MLP: Int, layers: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]),
    name: "patch_embed")
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  let classEmbedding = Parameter<Float>(.GPU(1), .CHW(1, 1, width), name: "cls_embed")
  let positionalEmbedding = Parameter<Float>(
    .GPU(1), .CHW(1, grid * grid + 1, width), name: "pos_embed")
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  for i in 0..<layers {
    if [4, 8, 12, 16, 20].contains(i) {
      outs.append(out)
    }
    let (block, reader) = EvaResidualAttentionBlock(
      prefix: "blocks.\(i)",
      k: width / heads, h: heads, b: batchSize, t: grid * grid + 1, MLP: MLP)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]), rot)
    readers.append(reader)
  }
  let norm = LayerNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out)
  let head = Dense(count: outputChannels, name: "head")
  out = head(out)
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
    let norm_weight = state_dict["norm.weight"].type(torch.float).cpu().numpy()
    let norm_bias = state_dict["norm.bias"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: norm_bias))
    let head_weight = state_dict["head.weight"].type(torch.float).cpu().numpy()
    let head_bias = state_dict["head.bias"].type(torch.float).cpu().numpy()
    head.weight.copy(from: try! Tensor<Float>(numpy: head_weight))
    head.bias.copy(from: try! Tensor<Float>(numpy: head_bias))
  }
  return (Model([x, rot], [out] + outs), reader)
}

func PerceiverAttention(prefix: String, k: Int, h: Int, outputDim: Int, b: Int, t: (Int, Int)) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let outX = norm1(x)
  let c = Input()
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  let outC = norm2(c)
  let outXC = Functional.concat(axis: 1, outX, outC)
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(outXC).reshaped([b, t.0 + t.1, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(outC)).reshaped([b, t.1, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(outXC).reshaped([b, t.0 + t.1, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t.1, t.0 + t.1])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t.1, t.0 + t.1])
  var out = dot * values
  out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: outputDim, noBias: true)
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).0.norm1.weight"].type(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    let norm1_bias = state_dict["\(prefix).0.norm1.bias"].type(torch.float).cpu().numpy()
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let norm2_weight = state_dict["\(prefix).0.norm2.weight"].type(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    let norm2_bias = state_dict["\(prefix).0.norm2.bias"].type(torch.float).cpu().numpy()
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let to_q_weight = state_dict["\(prefix).0.to_q.weight"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: to_q_weight))
    let to_kv_weight = state_dict["\(prefix).0.to_kv.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: to_kv_weight[0..<(k * h), ...]))
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: to_kv_weight[(k * h)..<(k * h * 2), ...]))
    let to_out_weight = state_dict["\(prefix).0.to_out.weight"].type(torch.float).cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: to_out_weight))
  }
  return (reader, Model([x, c], [out]))
}

func ResamplerLayer(prefix: String, k: Int, h: Int, outputDim: Int, b: Int, t: (Int, Int)) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let c = Input()
  let (attentionReader, attention) = PerceiverAttention(
    prefix: prefix, k: k, h: h, outputDim: outputDim, b: b, t: t)
  var out = c + attention(x, c).reshaped([b, t.1, outputDim])
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
  let fc1 = Dense(count: outputDim * 4, noBias: true)
  let gelu = GELU()
  let fc2 = Dense(count: outputDim, noBias: true)
  out = out + fc2(gelu(fc1(layerNorm(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    attentionReader(state_dict)
    let layerNorm_weight = state_dict["\(prefix).1.0.weight"].type(torch.float).cpu().numpy()
    layerNorm.weight.copy(from: try! Tensor<Float>(numpy: layerNorm_weight))
    let layerNorm_bias = state_dict["\(prefix).1.0.bias"].type(torch.float).cpu().numpy()
    layerNorm.bias.copy(from: try! Tensor<Float>(numpy: layerNorm_bias))
    let fc1_weight = state_dict["\(prefix).1.1.weight"].type(torch.float).cpu().numpy()
    fc1.weight.copy(from: try! Tensor<Float>(numpy: fc1_weight))
    let fc2_weight = state_dict["\(prefix).1.3.weight"].type(torch.float).cpu().numpy()
    fc2.weight.copy(from: try! Tensor<Float>(numpy: fc2_weight))
  }
  return (reader, Model([x, c], [out]))
}

func IDFormerMapping(prefix: String, channels: Int, outputChannels: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let layer0 = Dense(count: channels, name: "\(prefix).0")
  var out = layer0(x)
  let layer1 = LayerNorm(epsilon: 1e-5, axis: [1], name: "\(prefix).1")
  out = layer1(out)
  let layer2 = LeakyReLU(negativeSlope: 0.01)
  out = layer2(out)
  let layer3 = Dense(count: channels, name: "\(prefix).3")
  out = layer3(out)
  let layer4 = LayerNorm(epsilon: 1e-5, axis: [1], name: "\(prefix).4")
  out = layer4(out)
  let layer5 = LeakyReLU(negativeSlope: 0.01)
  out = layer5(out)
  let layer6 = Dense(count: outputChannels, name: "\(prefix).6")
  out = layer6(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let layer0_weight = state_dict["\(prefix).0.weight"].type(torch.float).cpu().numpy()
    layer0.weight.copy(from: try! Tensor<Float>(numpy: layer0_weight))
    let layer0_bias = state_dict["\(prefix).0.bias"].type(torch.float).cpu().numpy()
    layer0.bias.copy(from: try! Tensor<Float>(numpy: layer0_bias))
    let layer1_weight = state_dict["\(prefix).1.weight"].type(torch.float).cpu().numpy()
    layer1.weight.copy(from: try! Tensor<Float>(numpy: layer1_weight))
    print("layer1_weight \(layer1_weight)")
    let layer1_bias = state_dict["\(prefix).1.bias"].type(torch.float).cpu().numpy()
    layer1.bias.copy(from: try! Tensor<Float>(numpy: layer1_bias))
    let layer3_weight = state_dict["\(prefix).3.weight"].type(torch.float).cpu().numpy()
    layer3.weight.copy(from: try! Tensor<Float>(numpy: layer3_weight))
    let layer3_bias = state_dict["\(prefix).3.bias"].type(torch.float).cpu().numpy()
    layer3.bias.copy(from: try! Tensor<Float>(numpy: layer3_bias))
    let layer4_weight = state_dict["\(prefix).4.weight"].type(torch.float).cpu().numpy()
    layer4.weight.copy(from: try! Tensor<Float>(numpy: layer4_weight))
    let layer4_bias = state_dict["\(prefix).4.bias"].type(torch.float).cpu().numpy()
    layer4.bias.copy(from: try! Tensor<Float>(numpy: layer4_bias))
    let layer6_weight = state_dict["\(prefix).6.weight"].type(torch.float).cpu().numpy()
    layer6.weight.copy(from: try! Tensor<Float>(numpy: layer6_weight))
    let layer6_bias = state_dict["\(prefix).6.bias"].type(torch.float).cpu().numpy()
    layer6.bias.copy(from: try! Tensor<Float>(numpy: layer6_bias))
  }
  return (reader, Model([x], [out]))
}

func IDFormer(
  width: Int, outputDim: Int, heads: Int, grid: Int, idQueries: Int, queries: Int, layers: Int,
  depth: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let y = (0..<layers).map { _ in Input() }
  let latents = Parameter<Float>(.GPU(1), .CHW(1, queries, width), name: "latents")
  var readers = [(PythonObject) -> Void]()
  let idEmbeddingMapping = IDFormerMapping(
    prefix: "id_embedding_mapping", channels: 1024, outputChannels: 1024 * idQueries)
  readers.append(idEmbeddingMapping.0)
  var out = idEmbeddingMapping.1(x).reshaped([1, idQueries, 1024])
  let idFeature = out
  out = Functional.concat(axis: 1, latents, out)
  for i in 0..<layers {
    let mapping = IDFormerMapping(prefix: "mapping_\(i)", channels: 1024, outputChannels: 1024)
    readers.append(mapping.0)
    let vitFeature = mapping.1(y[i]).reshaped([1, grid * grid + 1, width])
    let ctxFeature = Functional.concat(axis: 1, idFeature, vitFeature)
    for j in 0..<depth {
      let (reader, layer) = ResamplerLayer(
        prefix: "layers.\(i * depth + j)", k: width / heads, h: heads,
        outputDim: width, b: 1, t: (grid * grid + 1 + idQueries, queries + idQueries)
      )
      readers.append(reader)
      out = layer(ctxFeature, out)
    }
  }
  let projOut = Dense(count: outputDim, noBias: true, name: "proj_out")
  out = projOut(
    out.reshaped([1, queries, width], strides: [(queries + idQueries) * width, width, 1])
      .contiguous())
  let reader: (PythonObject) -> Void = { state_dict in
    let latents_weight = state_dict["latents"].type(torch.float)
      .cpu().numpy()
    latents.weight.copy(from: try! Tensor<Float>(numpy: latents_weight))
    for reader in readers {
      reader(state_dict)
    }
    let proj_out_weight = state_dict["proj_out"].type(torch.float).t().cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
  }
  return (reader, Model([x] + y, [out]))
}

let state_dict = pipeline.clip_vision_model.state_dict()

let face_resampler_state_dict = pipeline.pulid_encoder.state_dict()

let pulid_ca_state_dict = pipeline.pulid_ca.state_dict()

func PuLIDCrossAttentionFixed(prefix: String, name: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_norm1")
  let tokeys = Dense(count: k * h, noBias: true, name: "\(name)_k")
  let tovalues = Dense(count: k * h, noBias: true, name: "\(name)_v")
  let out = norm1(x)
  let keys = tokeys(out).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(out).reshaped([b, t, h, k]).transposed(1, 2)
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    let norm1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu().numpy()
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let to_kv_weight = state_dict["\(prefix).to_kv.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: to_kv_weight[0..<(k * h), ...]))
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: to_kv_weight[(k * h)..<(k * h * 2), ...]))
  }
  return (reader, Model([x], [keys, values]))
}

func PuLIDFixed(queries: Int, double: [Int], single: [Int]) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  var ca: Int = 0
  for i in double {
    let (reader, block) = PuLIDCrossAttentionFixed(
      prefix: "\(ca)", name: "double_\(i)", k: 2048 / 16, h: 16, b: 1, t: queries)
    outs.append(block(x))
    readers.append(reader)
    ca += 1
  }
  for i in single {
    let (reader, block) = PuLIDCrossAttentionFixed(
      prefix: "\(ca)", name: "single_\(i)", k: 2048 / 16, h: 16, b: 1, t: queries)
    outs.append(block(x))
    readers.append(reader)
    ca += 1
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x], outs))
}

func PuLIDCrossAttentionKeysAndValues(
  prefix: String, name: String, outputDim: Int, k: Int, h: Int, b: Int, t: (Int, Int)
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_norm2")
  let toqueries = Dense(count: k * h, noBias: true, name: "\(name)_q")
  let queries = toqueries(norm2(x)).reshaped([b, t.1, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t.1, t.0])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t.1, t.0])
  var out = dot * values
  out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: outputDim, noBias: true, name: "\(name)_to_out")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let norm2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    let norm2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu().numpy()
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let to_q_weight = state_dict["\(prefix).to_q.weight"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: to_q_weight))
    let top_10_max = -numpy.sort(-numpy.partition(-to_q_weight.ravel(), 10)[..<10])
    let top_10_min = numpy.sort(numpy.partition(to_q_weight.ravel(), 10)[..<10])
    print("top 10 max \(top_10_max)")
    print("top 10 min \(top_10_min)")
    let to_out_weight = state_dict["\(prefix).to_out.weight"].type(torch.float).cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: to_out_weight))
  }
  return (reader, Model([x, keys, values], [out]))
}

func PuLID(queries: Int, width: Int, hw: Int, double: [Int], single: [Int]) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  var readers = [(PythonObject) -> Void]()
  var kvs = [Input]()
  var outs = [Model.IO]()
  var ca: Int = 0
  for i in double {
    let (reader, block) = PuLIDCrossAttentionKeysAndValues(
      prefix: "\(ca)", name: "double_\(i)", outputDim: width, k: 2048 / 16, h: 16, b: 1,
      t: (queries, hw))
    let k = Input()
    let v = Input()
    let out = block(x, k, v)
    kvs.append(contentsOf: [k, v])
    outs.append(out)
    readers.append(reader)
    ca += 1
  }
  for i in single {
    let (reader, block) = PuLIDCrossAttentionKeysAndValues(
      prefix: "\(ca)", name: "single_\(i)", outputDim: width, k: 2048 / 16, h: 16, b: 1,
      t: (queries, hw))
    let k = Input()
    let v = Input()
    let out = block(x, k, v)
    kvs.append(contentsOf: [k, v])
    outs.append(out)
    readers.append(reader)
    ca += 1
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x] + kvs, outs))
}

print(pulid_ca_state_dict.keys())

let graph = DynamicGraph()

graph.withNoGrad {
  let rotTensor = graph.variable(.CPU, .NHWC(1, 24 * 24 + 1, 1, 64), of: Float.self)
  for i in 0..<1 {
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 16) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<24 {
    for x in 0..<24 {
      let i = y * 24 + x + 1
      for k in 0..<16 {
        let theta = Double(y) * 16.0 / 24.0 / pow(10_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(x) * 16.0 / 24.0 / pow(10_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 16) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sintheta)
      }
    }
  }
  let xTensor = graph.variable(try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy()).toGPU(1))
  let rotTensorGPU = rotTensor.toGPU(1)
  let (vit, vitReader) = EvaVisionTransformer(
    grid: 24, outputChannels: 768, width: 1024, MLP: 2730, layers: 24, heads: 16, batchSize: 1)
  vit.compile(inputs: xTensor, rotTensorGPU)
  vitReader(state_dict)
  let output = vit(inputs: xTensor, rotTensorGPU).map { $0.as(of: Float.self) }
  var idCondVit = output[0][0..<1, 0..<768].copied()
  let idVitHidden = Array(output[1...])
  let idCondVitNorm = idCondVit.reduced(.norm2, axis: [1])
  idCondVit = idCondVit .* (1 / idCondVitNorm)
  let idAnteEmbedding = graph.variable(
    try! Tensor<Float>(numpy: id_ante_embedding.to(torch.float).cpu().numpy()).toGPU(1))
  let idCond = Functional.concat(axis: 1, idAnteEmbedding, idCondVit)
  let (idFormerReader, idFormer) = IDFormer(
    width: 1024, outputDim: 2048, heads: 16, grid: 24, idQueries: 5, queries: 32, layers: 5,
    depth: 2)
  idFormer.compile(inputs: [idCond] + idVitHidden)
  idFormerReader(face_resampler_state_dict)
  let features = idFormer(inputs: idCond, idVitHidden)[0].as(of: Float.self)

  let (pulidFixedReader, pulidFixed) = PuLIDFixed(
    queries: 32, double: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    single: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
  pulidFixed.compile(inputs: features)
  pulidFixedReader(pulid_ca_state_dict)
  let outs = pulidFixed(inputs: features).map { $0.as(of: Float.self) }

  let (pulidReader, pulid) = PuLID(
    queries: 32, width: 3072, hw: 4096, double: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    single: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
  let imgTensor = graph.variable(
    try! Tensor<Float>(numpy: img.to(torch.float).cpu().numpy()).toGPU(1))
  pulid.compile(inputs: [imgTensor] + outs)
  pulidReader(pulid_ca_state_dict)
  debugPrint(pulid(inputs: imgTensor, outs))
  /*
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/eva02_clip_l14_336_f32.ckpt"
  ) {
    $0.write("vision_model", model: vit)
  }
*/
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/pulid_0.9_eva02_clip_l14_336_f32.ckpt"
  ) {
    $0.write("resampler", model: idFormer)
    $0.write("pulid", model: pulidFixed)
    $0.write("pulid", model: pulid)
  }
}
