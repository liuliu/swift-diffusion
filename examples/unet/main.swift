import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let ldm_util = Python.import("ldm.util")
let torch = Python.import("torch")
let omegaconf = Python.import("omegaconf")
let random = Python.import("random")
let numpy = Python.import("numpy")

func timeEmbedding(timesteps: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<Float> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .C(embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * Float(timesteps)
    embedding[i] = cos(freq)
    embedding[i + half] = sin(freq)
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

func ResBlock(b: Int, ch: Int) -> (Model, Model, Model, Model, Model, Model) {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: ch, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: ch)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([b, ch, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: ch, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = x + outLayerConv2d(out)  // This layer should be zero init if training.
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, Model([x, emb], [out])
  )
}

func SelfAttention(k: Int, h: Int, b: Int, hw: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2).reshaped([b * h, hw, k])
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2).reshaped([b * h, hw, k])
  let values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2).reshaped([b * h, hw, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([b * h, hw, hw])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func CrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model, Model, Model)
{
  let x = Input()
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2).reshaped([b * h, t, k])
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2).reshaped([b * h, hw, k])
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2).reshaped([b * h, t, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b * h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, c], [out]))
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

func BasicTransformerBlock(k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model,
  Model
) {
  let x = Input()
  let c = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (tokeys2, toqueries2, tovalues2, unifyheads2, attn2) = CrossAttention(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, c) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  return (
    layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2, toqueries2,
    tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, Model([x, c], [out])
  )
}

func SpatialTransformer(
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, t: Int, intermediateSize: Int
) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model,
  Model, Model, Model, Model
) {
  let x = Input()
  let c = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).transposed(1, 2)
  let (
    layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2, toqueries2,
    tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, block
  ) = BasicTransformerBlock(k: k, h: h, b: b, hw: hw, t: t, intermediateSize: intermediateSize)
  out = block(out, c).reshaped([b, hw, k * h]).transposed(1, 2).reshaped([b, k * h, height, width])
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  return (
    norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
    toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, projOut, Model([x, c], [out])
  )
}

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let config = omegaconf.OmegaConf.load(
  "/home/liu/workspace/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
let pl_sd = torch.load(
  "/home/liu/workspace/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
  map_location: "cpu")
let sd = pl_sd["state_dict"]
let model = ldm_util.instantiate_from_config(config.model)
model.load_state_dict(sd, strict: false)
model.eval()
let state_dict = model.model.state_dict()
let x = torch.randn([1, 4, 64, 64])
let t = torch.full([1], 981)
let c = torch.randn([1, 77, 768])
let ret = model.model.diffusion_model(x, t, c)

let graph = DynamicGraph()

let time_embed_0_weight = state_dict["diffusion_model.time_embed.0.weight"].numpy()
let time_embed_0_bias = state_dict["diffusion_model.time_embed.0.bias"].numpy()
let time_embed_2_weight = state_dict["diffusion_model.time_embed.2.weight"].numpy()
let time_embed_2_bias = state_dict["diffusion_model.time_embed.2.bias"].numpy()

let t_emb = graph.variable(timeEmbedding(timesteps: 981, embeddingSize: 320, maxPeriod: 10_000))
let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
let _ = timeEmbed(inputs: t_emb)
fc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_0_weight))
fc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_0_bias))
fc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_2_weight))
fc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_2_bias))
let emb = timeEmbed(inputs: t_emb)[0].as(of: Float.self).toGPU(0)
let xTensor = graph.variable(try! Tensor<Float>(numpy: x.numpy())).toGPU(0)
let input_blocks_0_0_weight = state_dict["diffusion_model.input_blocks.0.0.weight"].numpy()
let input_blocks_0_0_bias = state_dict["diffusion_model.input_blocks.0.0.bias"].numpy()
let conv2d = Convolution(
  groups: 1, filters: 320, filterSize: [3, 3],
  hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
let _ = conv2d(xTensor)
conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_weight))
conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_bias))
let yTensor = conv2d(xTensor)
let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, resBlock) = ResBlock(
  b: 1, ch: 320)
let _ = resBlock(inputs: yTensor, emb)
let in_layers_0_weight = state_dict["diffusion_model.input_blocks.1.0.in_layers.0.weight"].numpy()
let in_layers_0_bias = state_dict["diffusion_model.input_blocks.1.0.in_layers.0.bias"].numpy()
inLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_0_weight))
inLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_bias))
let in_layers_2_weight = state_dict["diffusion_model.input_blocks.1.0.in_layers.2.weight"].numpy()
let in_layers_2_bias = state_dict["diffusion_model.input_blocks.1.0.in_layers.2.bias"].numpy()
inLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_2_weight))
inLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_bias))
let emb_layers_1_weight = state_dict["diffusion_model.input_blocks.1.0.emb_layers.1.weight"].numpy()
let emb_layers_1_bias = state_dict["diffusion_model.input_blocks.1.0.emb_layers.1.bias"].numpy()
embLayer.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_1_weight))
embLayer.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_1_bias))
let out_layers_0_weight = state_dict["diffusion_model.input_blocks.1.0.out_layers.0.weight"].numpy()
let out_layers_0_bias = state_dict["diffusion_model.input_blocks.1.0.out_layers.0.bias"].numpy()
outLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_layers_0_weight))
outLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_bias))
let out_layers_3_weight = state_dict["diffusion_model.input_blocks.1.0.out_layers.3.weight"].numpy()
let out_layers_3_bias = state_dict["diffusion_model.input_blocks.1.0.out_layers.3.bias"].numpy()
outLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_layers_3_weight))
outLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_3_bias))
let resOut = resBlock(inputs: yTensor, emb)[0].as(of: Float.self)
let (
  norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
  toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, tfc2, projOut, transformer
) = SpatialTransformer(
  ch: 320, k: 40, h: 8, b: 1, height: 64, width: 64, t: 77, intermediateSize: 1280)
let cTensor = graph.variable(try! Tensor<Float>(numpy: c.numpy())).toGPU(0)
let _ = transformer(inputs: resOut, cTensor)
let norm_weight = state_dict["diffusion_model.input_blocks.1.1.norm.weight"].numpy()
let norm_bias = state_dict["diffusion_model.input_blocks.1.1.norm.bias"].numpy()
norm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_weight))
norm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_bias))
let proj_in_weight = state_dict["diffusion_model.input_blocks.1.1.proj_in.weight"].numpy()
let proj_in_bias = state_dict["diffusion_model.input_blocks.1.1.proj_in.bias"].numpy()
projIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_in_weight))
projIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_in_bias))
let attn1_to_k_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight"
].numpy()
tokeys1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_k_weight))
let attn1_to_q_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
].numpy()
toqueries1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_q_weight))
let attn1_to_v_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight"
].numpy()
tovalues1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_v_weight))
let attn1_to_out_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight"
].numpy()
let attn1_to_out_bias = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias"
].numpy()
unifyheads1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_out_weight))
unifyheads1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn1_to_out_bias))
let ff_net_0_proj_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight"
].numpy()
let ff_net_0_proj_bias = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias"
].numpy()
fc10.parameters(for: .weight).copy(
  from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[..<1280, ...]))
fc10.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[..<1280]))
fc11.parameters(for: .weight).copy(
  from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[1280..., ...]))
fc11.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[1280...]))
let ff_net_2_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight"
].numpy()
let ff_net_2_bias = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias"
].numpy()
tfc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ff_net_2_weight))
tfc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_2_bias))
let attn2_to_k_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"
].numpy()
tokeys2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_k_weight))
let attn2_to_q_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight"
].numpy()
toqueries2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_q_weight))
let attn2_to_v_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight"
].numpy()
tovalues2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
let attn2_to_out_weight = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight"
].numpy()
let attn2_to_out_bias = state_dict[
  "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias"
].numpy()
unifyheads2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_out_weight))
unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
let norm1_weight = state_dict["diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.weight"]
  .numpy()
let norm1_bias = state_dict["diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.bias"]
  .numpy()
layerNorm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
layerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
let norm2_weight = state_dict["diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.weight"]
  .numpy()
let norm2_bias = state_dict["diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.bias"]
  .numpy()
layerNorm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
layerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
let norm3_weight = state_dict["diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.weight"]
  .numpy()
let norm3_bias = state_dict["diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.bias"]
  .numpy()
layerNorm3.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm3_weight))
layerNorm3.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm3_bias))
let proj_out_weight = state_dict["diffusion_model.input_blocks.1.1.proj_out.weight"].numpy()
let proj_out_bias = state_dict["diffusion_model.input_blocks.1.1.proj_out.bias"].numpy()
projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
let attnOut = transformer(inputs: resOut, cTensor)[0].as(of: Float.self)

let attnOutCPU = attnOut.toCPU()
print(attnOut)
for i in 0..<6 {
  let x = i < 3 ? i : 314 + i
  for j in 0..<6 {
    let y = j < 3 ? j : 58 + j
    for k in 0..<6 {
      let z = k < 3 ? k : 58 + k
      print("\(x) \(y) \(z) \(attnOutCPU[0, x, y, z])")
    }
  }
}
