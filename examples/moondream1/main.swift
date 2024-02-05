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

let raw_image = Image.open("/home/liu/workspace/blip2/merlion.png").convert("RGB")

let transformers = Python.import("transformers")

torch.set_grad_enabled(false)
let model_id = "vikhyatk/moondream1"
let model = transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code: true)
let tokenizer = transformers.CodeGenTokenizerFast.from_pretrained(model_id)

print(model)

let enc_image = model.encode_image(raw_image)

print(
  model.answer_question(
    enc_image,
    "Describe this image and its style in a very detailed manner, follow the format of describing: what, who, where, when, how. You don't need to fill in all if they are irrelevant. Please remove What, Who, Where, When, How prefixes and make it one paragraph.",
    tokenizer))

let vision_encoder_state_dict = model.vision_encoder.state_dict()
print(vision_encoder_state_dict.keys())

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

graph.withNoGrad {
  let _ = vit(inputs: xTensor)
  reader(vision_encoder_state_dict)
  var outs = vit(inputs: xTensor).map { $0.as(of: Float.self) }
  let _ = proj(inputs: outs[0])
  projReader(vision_encoder_state_dict)
  outs = proj(inputs: outs[0]).map { $0.as(of: Float.self) }
  print(outs)
}
