import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let pulid_pipeline_flux = Python.import("pulid.pipeline_flux")

let flux_util = Python.import("flux.util")

let torch = Python.import("torch")

let model = flux_util.load_flow_model("flux-dev", device: "cpu")

let pipeline = pulid_pipeline_flux.PuLIDPipeline(
  model, device: "cpu", weight_dtype: torch.bfloat16, onnx_provider: "gpu")

torch.set_grad_enabled(false)

let x = torch.randn([1, 3, 336, 336]).to(torch.float).cuda()

pipeline.clip_vision_model = pipeline.clip_vision_model.to(torch.float).cuda()

let output = pipeline.clip_vision_model(
  x, return_all_features: false, return_hidden: true, shuffle: false)

print(pipeline.clip_vision_model)

print(output)

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
  grid: Int, width: Int, MLP: Int, layers: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  let classEmbedding = Parameter<Float>(.GPU(1), .CHW(1, 1, width))
  let positionalEmbedding = Parameter<Float>(.GPU(1), .CHW(1, grid * grid + 1, width))
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (block, reader) = EvaResidualAttentionBlock(
      prefix: "blocks.\(i)",
      k: width / heads, h: heads, b: batchSize, t: grid * grid + 1, MLP: MLP)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]), rot)
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
  return (Model([x, rot], [out]), reader)
}

let state_dict = pipeline.clip_vision_model.state_dict()

print(state_dict.keys())

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
    grid: 24, width: 1024, MLP: 2730, layers: 24, heads: 16, batchSize: 1)
  vit.compile(inputs: xTensor, rotTensorGPU)
  vitReader(state_dict)
  debugPrint(vit(inputs: xTensor, rotTensorGPU))
}
