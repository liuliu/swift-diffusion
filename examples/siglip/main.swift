import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let torch = Python.import("torch")
let numpy = Python.import("numpy")
let Image = Python.import("PIL.Image")

let device = torch.device("cuda")

let transformers = Python.import("transformers")

let flux_modules_image_embedders = Python.import("flux.modules.image_embedders")

torch.set_grad_enabled(false)

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
  let gelu = GELU(approximate: .tanh)
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_1_weight = state_dict["\(prefix).layer_norm1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).layer_norm1.bias"].type(torch.float).cpu().numpy()
    ln1.weight.copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    ln1.bias.copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let q_proj_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).cpu()
      .numpy()
    let q_proj_bias = state_dict["\(prefix).self_attn.q_proj.bias"].type(torch.float).cpu().numpy()
    let k_proj_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).cpu()
      .numpy()
    let k_proj_bias = state_dict["\(prefix).self_attn.k_proj.bias"].type(torch.float).cpu().numpy()
    let v_proj_weight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu()
      .numpy()
    let v_proj_bias = state_dict["\(prefix).self_attn.v_proj.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(
      from: try! Tensor<Float>(numpy: q_proj_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_proj_bias))
    tokeys.weight.copy(
      from: try! Tensor<Float>(numpy: k_proj_weight))
    tokeys.bias.copy(
      from: try! Tensor<Float>(numpy: k_proj_bias))
    tovalues.weight.copy(
      from: try! Tensor<Float>(numpy: v_proj_weight))
    tovalues.bias.copy(
      from: try! Tensor<Float>(numpy: v_proj_bias))
    let out_proj_weight = state_dict["\(prefix).self_attn.out_proj.weight"].type(torch.float).cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).self_attn.out_proj.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: out_proj_bias))
    let ln_2_weight = state_dict["\(prefix).layer_norm2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).layer_norm2.bias"].type(torch.float).cpu().numpy()
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
    .GPU(0), .CHW(1, gridX * gridY, width), initBound: 1, name: "pos_embed")
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [16, 16], hint: Hint(stride: [16, 16]))
  var out = conv1(x).reshaped([batchSize, width, gridX * gridY]).transposed(1, 2)
  out = out + posEmbed
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (reader, block) = SigLIPResidualAttentionBlock(
      prefix: "encoder.layers.\(i)", k: width / heads, h: heads, b: batchSize,
      t: gridX * gridY, MLP: MLP)
    out = block(out.reshaped([batchSize, gridX * gridY, width]))
    readers.append(reader)
  }
  let lnPost = LayerNorm(epsilon: 1e-6, axis: [1])
  out = lnPost(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let positional_embedding = state_dict["embeddings.position_embedding.weight"].type(torch.float)
      .cpu().numpy()
    posEmbed.weight.copy(from: try! Tensor<Float>(numpy: positional_embedding))
    let conv1_weight = state_dict["embeddings.patch_embedding.weight"].type(
      torch.float
    ).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let conv1_bias = state_dict["embeddings.patch_embedding.bias"].type(
      torch.float
    ).cpu().numpy()
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    for reader in readers {
      reader(state_dict)
    }
    let ln_post_weight = state_dict["post_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    let ln_post_bias = state_dict["post_layernorm.bias"].type(torch.float).cpu()
      .numpy()
    lnPost.weight.copy(from: try! Tensor<Float>(numpy: ln_post_weight))
    lnPost.bias.copy(from: try! Tensor<Float>(numpy: ln_post_bias))
  }
  return (reader, Model([x], [out]))
}

func Redux(channels: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let reduxUp = Dense(count: channels * 3)
  let reduxDown = Dense(count: channels)
  let out = reduxDown(reduxUp(x).swish())
  let reader: (PythonObject) -> Void = { state_dict in
    let redux_up_weight = state_dict["redux_up.weight"].type(torch.float).cpu().numpy()
    reduxUp.weight.copy(from: try! Tensor<Float>(numpy: redux_up_weight))
    let redux_up_bias = state_dict["redux_up.bias"].type(torch.float).cpu().numpy()
    reduxUp.bias.copy(from: try! Tensor<Float>(numpy: redux_up_bias))
    let redux_down_weight = state_dict["redux_down.weight"].type(torch.float).cpu().numpy()
    reduxDown.weight.copy(from: try! Tensor<Float>(numpy: redux_down_weight))
    let redux_down_bias = state_dict["redux_down.bias"].type(torch.float).cpu().numpy()
    reduxDown.bias.copy(from: try! Tensor<Float>(numpy: redux_down_bias))
  }
  return (reader, Model([x], [out]))
}

let x = torch.randn([1, 3, 512, 512])
let siglip = transformers.SiglipVisionModel.from_pretrained("google/siglip2-so400m-patch16-512")
let encoded = siglip(x).last_hidden_state
let vision_encoder_state_dict = siglip.vision_model.state_dict()
print(encoded)

let img_embedder = flux_modules_image_embedders.ReduxImageEncoder(
  torch.device("cpu"),
  redux_path: "/home/liu/workspace/swift-diffusion/flex1_redux_siglip2_512.safetensors",
  dtype: torch.float)
img_embedder.redux_up = img_embedder.redux_up.to(torch.float)
img_embedder.redux_down = img_embedder.redux_down.to(torch.float)
print(img_embedder.redux_down(torch.nn.functional.silu(img_embedder.redux_up(encoded))))
let img_embedder_state_dict = img_embedder.state_dict()

let graph = DynamicGraph()
let xTensor = graph.variable(try! Tensor<Float>(numpy: x.numpy())).toGPU(0)
let (reader, vit) = SigLIPVisionTransformer(
  gridX: 32, gridY: 32, width: 1152, layers: 27, heads: 16, MLP: 4304, batchSize: 1)

let (reduxReader, redux) = Redux(channels: 4096)

graph.withNoGrad {
  let _ = vit(inputs: xTensor)
  reader(vision_encoder_state_dict)
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/siglip_so400m_384_f16.ckpt") {
    $0.read("vit", model: vit, codec: [.q8p, .ezm7])
  }
  */
  var outs = vit(inputs: xTensor).map { $0.as(of: Float.self) }
  debugPrint(outs)
  let _ = redux(inputs: outs[0])
  reduxReader(img_embedder_state_dict)
  outs = redux(inputs: outs[0]).map { $0.as(of: Float.self) }
  debugPrint(outs)
  graph.openStore("/home/liu/workspace/swift-diffusion/siglip2_so400m_512_f32.ckpt") {
    $0.write("vit", model: vit)
  }
  graph.openStore("/home/liu/workspace/swift-diffusion/flex_1_redux_siglip2_512_f32.ckpt") {
    $0.write("redux", model: redux)
  }
}
