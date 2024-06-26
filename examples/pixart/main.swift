import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit
import SentencePiece

typealias FloatType = Float

let torch = Python.import("torch")
let diffusion = Python.import("diffusion")
let tools_download = Python.import("tools.download")

torch.set_grad_enabled(false)

let weight_dtype = torch.float
let device = "cuda"
let model = diffusion.model.nets.PixArtMS_XL_2(
  input_size: 128,
  pe_interpolation: 2,
  micro_condition: false,
  model_max_length: 300
).to(device)
let state_dict = tools_download.find_model(
  "/home/liu/workspace/PixArt-sigma/output/pretrained_models/PixArt-Sigma-XL-2-1024-MS.pth")[
    "state_dict"]
model.load_state_dict(state_dict, strict: false)
model.to(weight_dtype)
model.eval()
print(state_dict.keys())

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([2, 4, 128, 128]).cuda()
let y = torch.randn([2, 1, 300, 4096]).cuda() * 0.001
let t = torch.full([2], 666.0).cuda()

let et = model(x, t, y)
print(et)

func sinCos2DPositionEmbedding(height: Int, width: Int, embeddingSize: Int) -> Tensor<Float> {
  precondition(embeddingSize % 4 == 0)
  var embedding = Tensor<Float>(.CPU, .HWC(height, width, embeddingSize))
  let halfOfHalf = embeddingSize / 4
  let omega: [Double] = (0..<halfOfHalf).map {
    pow(Double(1.0 / 10000), Double($0) / Double(halfOfHalf))
  }
  for i in 0..<height {
    let y = Double(i) / 2
    for j in 0..<width {
      let x = Double(j) / 2
      for k in 0..<halfOfHalf {
        let xFreq = x * omega[k]
        embedding[i, j, k] = Float(sin(xFreq))
        embedding[i, j, k + halfOfHalf] = Float(cos(xFreq))
        let yFreq = y * omega[k]
        embedding[i, j, k + 2 * halfOfHalf] = Float(sin(yFreq))
        embedding[i, j, k + 3 * halfOfHalf] = Float(cos(yFreq))
      }
    }
  }
  return embedding
}

func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLP(hiddenSize: Int, intermediateSize: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize, name: "\(name)_fc1")
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = Dense(count: hiddenSize, name: "\(name)_fc2")
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func SelfAttention(k: Int, h: Int, b: Int, t: Int) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let tokeys = Dense(count: k * h, name: "k")
  let toqueries = Dense(count: k * h, name: "q")
  let tovalues = Dense(count: k * h, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
  // No scaling the queries.
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .transposed(1, 2)
  let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
  let unifyheads = Dense(count: k * h, name: "o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func CrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model, Model, Model)
{
  let x = Input()
  let context = Input()
  let tokeys = Dense(count: k * h, name: "c_k")
  let toqueries = Dense(count: k * h, name: "c_q")
  let tovalues = Dense(count: k * h, name: "c_v")
  let keys = tokeys(context).reshaped([b, t, h, k]).transposed(1, 2)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2)
  let values = tovalues(context).reshaped([b, t, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h, name: "c_o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, context], [out]))
}

func PixArtMSBlock(prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let context = Input()
  let shiftMsa = Input()
  let scaleMsa = Input()
  let gateMsa = Input()
  let shiftMlp = Input()
  let scaleMlp = Input()
  let gateMlp = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn) = SelfAttention(k: k, h: h, b: b, t: hw)
  let shiftMsaShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_0")
  let scaleMsaShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_1")
  let gateMsaShift = Parameter<Float>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_2")
  var out =
    x + (gateMsa + gateMsaShift)
    .* attn(norm1(x) .* (scaleMsa + scaleMsaShift) + (shiftMsa + shiftMsaShift))
  let (tokeys2, toqueries2, tovalues2, unifyheads2, crossAttn) = CrossAttention(
    k: k, h: h, b: b, hw: hw, t: t)
  out = out + crossAttn(out, context)
  let (fc1, fc2, mlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4, name: "mlp")
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shiftMlpShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_3")
  let scaleMlpShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_4")
  let gateMlpShift = Parameter<Float>(.GPU(0), .CHW(1, 1, k * h), name: "scale_shift_table_5")
  out = out + (gateMlp + gateMlpShift)
    .* mlp(norm2(out) .* (scaleMlp + scaleMlpShift) + (shiftMlp + shiftMlpShift))
  let reader: (PythonObject) -> Void = { state_dict in
    let attn_qkv_weight = state_dict["\(prefix).attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    let attn_qkv_bias = state_dict["\(prefix).attn.qkv.bias"].to(torch.float)
      .cpu().numpy()
    toqueries1.weight.copy(
      from: try! Tensor<Float>(numpy: attn_qkv_weight[..<(k * h), ...]))
    toqueries1.bias.copy(from: try! Tensor<Float>(numpy: attn_qkv_bias[..<(k * h)]))
    tokeys1.weight.copy(
      from: try! Tensor<Float>(numpy: attn_qkv_weight[(k * h)..<(2 * k * h), ...]))
    tokeys1.bias.copy(from: try! Tensor<Float>(numpy: attn_qkv_bias[(k * h)..<(2 * k * h)]))
    tovalues1.weight.copy(
      from: try! Tensor<Float>(numpy: attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...]))
    tovalues1.bias.copy(
      from: try! Tensor<Float>(numpy: attn_qkv_bias[(2 * k * h)..<(3 * k * h)]))
    let attn_proj_weight = state_dict["\(prefix).attn.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let attn_proj_bias = state_dict["\(prefix).attn.proj.bias"].to(torch.float)
      .cpu().numpy()
    unifyheads1.weight.copy(from: try! Tensor<Float>(numpy: attn_proj_weight))
    unifyheads1.bias.copy(from: try! Tensor<Float>(numpy: attn_proj_bias))
    let scale_shift_table = state_dict["\(prefix).scale_shift_table"].to(torch.float).cpu().numpy()
    shiftMsaShift.weight.copy(from: try! Tensor<Float>(numpy: scale_shift_table[0, ...]))
    scaleMsaShift.weight.copy(from: try! Tensor<Float>(numpy: scale_shift_table[1, ...]))
    gateMsaShift.weight.copy(from: try! Tensor<Float>(numpy: scale_shift_table[2, ...]))
    let cross_attn_q_linear_weight = state_dict["\(prefix).cross_attn.q_linear.weight"].to(
      torch.float
    ).cpu().numpy()
    let cross_attn_q_linear_bias = state_dict["\(prefix).cross_attn.q_linear.bias"].to(torch.float)
      .cpu().numpy()
    toqueries2.weight.copy(
      from: try! Tensor<Float>(numpy: cross_attn_q_linear_weight))
    toqueries2.bias.copy(from: try! Tensor<Float>(numpy: cross_attn_q_linear_bias))
    let cross_attn_kv_linear_weight = state_dict["\(prefix).cross_attn.kv_linear.weight"].to(
      torch.float
    ).cpu().numpy()
    let cross_attn_kv_linear_bias = state_dict["\(prefix).cross_attn.kv_linear.bias"].to(
      torch.float
    )
    .cpu().numpy()
    tokeys2.weight.copy(
      from: try! Tensor<Float>(numpy: cross_attn_kv_linear_weight[0..<(k * h), ...]))
    tokeys2.bias.copy(from: try! Tensor<Float>(numpy: cross_attn_kv_linear_bias[0..<(k * h)]))
    tovalues2.weight.copy(
      from: try! Tensor<Float>(numpy: cross_attn_kv_linear_weight[(k * h)..<(2 * k * h), ...]))
    tovalues2.bias.copy(
      from: try! Tensor<Float>(numpy: cross_attn_kv_linear_bias[(k * h)..<(2 * k * h)]))
    let cross_attn_proj_weight = state_dict["\(prefix).cross_attn.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let cross_attn_proj_bias = state_dict["\(prefix).cross_attn.proj.bias"].to(torch.float)
      .cpu().numpy()
    unifyheads2.weight.copy(from: try! Tensor<Float>(numpy: cross_attn_proj_weight))
    unifyheads2.bias.copy(from: try! Tensor<Float>(numpy: cross_attn_proj_bias))
    shiftMlpShift.weight.copy(from: try! Tensor<Float>(numpy: scale_shift_table[3, ...]))
    scaleMlpShift.weight.copy(from: try! Tensor<Float>(numpy: scale_shift_table[4, ...]))
    gateMlpShift.weight.copy(from: try! Tensor<Float>(numpy: scale_shift_table[5, ...]))
    let mlp_fc1_weight = state_dict["\(prefix).mlp.fc1.weight"].to(
      torch.float
    ).cpu().numpy()
    let mlp_fc1_bias = state_dict["\(prefix).mlp.fc1.bias"].to(torch.float)
      .cpu().numpy()
    fc1.weight.copy(from: try! Tensor<Float>(numpy: mlp_fc1_weight))
    fc1.bias.copy(from: try! Tensor<Float>(numpy: mlp_fc1_bias))
    let mlp_fc2_weight = state_dict["\(prefix).mlp.fc2.weight"].to(
      torch.float
    ).cpu().numpy()
    let mlp_fc2_bias = state_dict["\(prefix).mlp.fc2.bias"].to(torch.float)
      .cpu().numpy()
    fc2.weight.copy(from: try! Tensor<Float>(numpy: mlp_fc2_weight))
    fc2.bias.copy(from: try! Tensor<Float>(numpy: mlp_fc2_bias))
  }
  return (
    reader, Model([x, context, shiftMsa, scaleMsa, gateMsa, shiftMlp, scaleMlp, gateMlp], [out])
  )
}

func PixArt(b: Int, h: Int, w: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let posEmbed = Input()
  let t = Input()
  let y = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 1152, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 1152, h * w]).transposed(1, 2) + posEmbed
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 1152)
  let t0 = tEmbedder(t)
  let t1 = t0.swish().reshaped([b, 1, 1152])
  let tBlock = (0..<6).map { Dense(count: 1152, name: "t_block_\($0)") }
  var adaln = tBlock.map { $0(t1) }
  adaln[1] = 1 + adaln[1]
  adaln[4] = 1 + adaln[4]
  let (fc1, fc2, yEmbedder) = MLP(hiddenSize: 1152, intermediateSize: 1152, name: "y_embedder")
  let y0 = yEmbedder(y)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<28 {
    let (reader, block) = PixArtMSBlock(
      prefix: "blocks.\(i)", k: 72, h: 16, b: 2, hw: h * w, t: 300)
    out = block(out, y0, adaln[0], adaln[1], adaln[2], adaln[3], adaln[4], adaln[5])
    readers.append(reader)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shiftShift = Parameter<Float>(.GPU(0), .CHW(1, 1, 1152), name: "final_scale_shift_table_0")
  let scaleShift = Parameter<Float>(.GPU(0), .CHW(1, 1, 1152), name: "final_scale_shift_table_1")
  let tt = t0.reshaped([1, 1, 1152])  // PixArt uses chunk, but that always assumes t0 is the same, which is true.
  out = (scaleShift + 1 + tt) .* normFinal(out) + (shiftShift + tt)
  let linear = Dense(count: 2 * 2 * 8, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 8]).permuted(0, 5, 1, 3, 2, 4).contiguous().reshaped([
    b, 8, h * 2, w * 2,
  ])
  let reader: (PythonObject) -> Void = { state_dict in
    let x_embedder_proj_weight = state_dict["x_embedder.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_embedder_proj_bias = state_dict["x_embedder.proj.bias"].to(torch.float)
      .cpu().numpy()
    xEmbedder.weight.copy(from: try! Tensor<Float>(numpy: x_embedder_proj_weight))
    xEmbedder.bias.copy(from: try! Tensor<Float>(numpy: x_embedder_proj_bias))
    let t_embedder_mlp_0_weight = state_dict["t_embedder.mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_0_bias = state_dict["t_embedder.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    tMlp0.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight))
    tMlp0.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias))
    let t_embedder_mlp_2_weight = state_dict["t_embedder.mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_2_bias = state_dict["t_embedder.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    tMlp2.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight))
    tMlp2.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias))
    let t_block_1_weight = state_dict["t_block.1.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_block_1_bias = state_dict["t_block.1.bias"].to(torch.float)
      .cpu().numpy()
    for i in 0..<6 {
      tBlock[i].weight.copy(
        from: try! Tensor<Float>(numpy: t_block_1_weight[(i * 1152)..<((i + 1) * 1152), ...]))
      tBlock[i].bias.copy(
        from: try! Tensor<Float>(numpy: t_block_1_bias[(i * 1152)..<((i + 1) * 1152)]))
    }
    let y_embedder_y_proj_fc1_weight = state_dict["y_embedder.y_proj.fc1.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_y_proj_fc1_bias = state_dict["y_embedder.y_proj.fc1.bias"].to(torch.float)
      .cpu().numpy()
    fc1.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_y_proj_fc1_weight))
    fc1.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_y_proj_fc1_bias))
    let y_embedder_y_proj_fc2_weight = state_dict["y_embedder.y_proj.fc2.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_y_proj_fc2_bias = state_dict["y_embedder.y_proj.fc2.bias"].to(torch.float)
      .cpu().numpy()
    fc2.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_y_proj_fc2_weight))
    fc2.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_y_proj_fc2_bias))
    for reader in readers {
      reader(state_dict)
    }
    let scale_shift_table = state_dict["final_layer.scale_shift_table"].to(torch.float).cpu()
      .numpy()
    shiftShift.weight.copy(from: try! Tensor<Float>(numpy: scale_shift_table[0, ...]))
    scaleShift.weight.copy(from: try! Tensor<Float>(numpy: scale_shift_table[1, ...]))
    let final_layer_linear_weight = state_dict["final_layer.linear.weight"].to(torch.float).cpu()
      .numpy()
    let final_layer_linear_bias = state_dict["final_layer.linear.bias"].to(torch.float).cpu()
      .numpy()
    linear.weight.copy(from: try! Tensor<Float>(numpy: final_layer_linear_weight))
    linear.bias.copy(from: try! Tensor<Float>(numpy: final_layer_linear_bias))
  }
  return (reader, Model([x, posEmbed, t, y], [out]))
}

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timesteps
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

let graph = DynamicGraph()

let (reader, dit) = PixArt(b: 2, h: 64, w: 64)

graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy()).toGPU(0))
  let tTensor = graph.variable(
    timeEmbedding(timesteps: 666, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000).toGPU(0))
  let yTensor = graph.variable(try! Tensor<Float>(numpy: y.to(torch.float).cpu().numpy()).toGPU(0))
    .reshaped(.CHW(2, 300, 4096))
  let posEmbedTensor = graph.variable(
    sinCos2DPositionEmbedding(height: 64, width: 64, embeddingSize: 1152).toGPU(0)
  ).reshaped(.CHW(1, 4096, 1152))
  dit.compile(inputs: xTensor, posEmbedTensor, tTensor, yTensor)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, posEmbedTensor, tTensor, yTensor))
}
