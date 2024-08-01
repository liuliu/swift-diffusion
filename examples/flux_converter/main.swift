import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit
import SentencePiece

typealias FloatType = Float

let torch = Python.import("torch")
let flux_util = Python.import("flux.util")

torch.set_grad_enabled(false)

let torch_device = "cuda"

let t5 = flux_util.load_t5(torch_device, max_length: 256)
let clip = flux_util.load_clip(torch_device)
let model = flux_util.load_flow_model("flux-schnell", device: torch_device)
let ae = flux_util.load_ae("flux-schnell", device: torch_device)

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([1, 4096, 64]).to(torch.bfloat16).cuda()
let y = torch.randn([1, 768]).to(torch.bfloat16).cuda() * 0.01
let txt = torch.randn([1, 256, 4096]).to(torch.bfloat16).cuda() * 0.01
var img_ids = torch.zeros([64, 64, 3])
img_ids[..., ..., 1] = img_ids[..., ..., 1] + torch.arange(64)[..., Python.None]
img_ids[..., ..., 2] = img_ids[..., ..., 2] + torch.arange(64)[Python.None, ...]
img_ids = img_ids.reshape([1, 4096, 3]).cuda()
let txt_ids = torch.zeros([1, 256, 3]).cuda()
let t = torch.full([1], 1).to(torch.bfloat16).cuda()
let guidance = torch.full([1], 3.5).to(torch.bfloat16).cuda()
print(model(x, img_ids, txt, txt_ids, t, y, guidance))

let state_dict = model.state_dict()

let z = torch.randn([1, 16, 128, 128]).to(torch.float).cuda()
print(ae.decode(z))

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

func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func VectorEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "vector_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "vector_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  return (linear1, outProjection, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let rot = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3])
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3])
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3])
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3])
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  keys = keys.permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .permuted(0, 2, 1, 3)
  values = values.permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = x + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut = contextOut + contextChunks[5]
      .* contextFF(contextNorm2(contextOut) .* (1 + contextChunks[4]) + contextChunks[3])
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut = xOut + xChunks[5] .* xFF(xNorm2(xOut) .* (1 + xChunks[4]) + xChunks[3])
  let reader: (PythonObject) -> Void = { state_dict in
    let txt_attn_qkv_weight = state_dict["\(prefix).txt_attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_qkv_bias = state_dict["\(prefix).txt_attn.qkv.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[..<(k * h), ...]))
    contextToQueries.bias.copy(
      from: try! Tensor<Float>(numpy: txt_attn_qkv_bias[..<(k * h)]))
    contextToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[(k * h)..<(2 * k * h), ...]))
    contextToKeys.bias.copy(
      from: try! Tensor<Float>(numpy: txt_attn_qkv_bias[(k * h)..<(2 * k * h)]))
    contextToValues.weight.copy(
      from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...])
    )
    contextToValues.bias.copy(
      from: try! Tensor<Float>(numpy: txt_attn_qkv_bias[(2 * k * h)..<(3 * k * h)]))
    let txt_attn_key_norm_scale = state_dict["\(prefix).txt_attn.norm.key_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normAddedK.weight.copy(from: try! Tensor<Float>(numpy: txt_attn_key_norm_scale))
    let txt_attn_query_norm_scale = state_dict["\(prefix).txt_attn.norm.query_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normAddedQ.weight.copy(from: try! Tensor<Float>(numpy: txt_attn_query_norm_scale))
    let img_attn_qkv_weight = state_dict["\(prefix).img_attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_qkv_bias = state_dict["\(prefix).img_attn.qkv.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: img_attn_qkv_weight[..<(k * h), ...]))
    xToQueries.bias.copy(from: try! Tensor<Float>(numpy: img_attn_qkv_bias[..<(k * h)]))
    xToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: img_attn_qkv_weight[(k * h)..<(2 * k * h), ...]))
    xToKeys.bias.copy(from: try! Tensor<Float>(numpy: img_attn_qkv_bias[(k * h)..<(2 * k * h)]))
    xToValues.weight.copy(
      from: try! Tensor<Float>(numpy: img_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...]))
    xToValues.bias.copy(
      from: try! Tensor<Float>(numpy: img_attn_qkv_bias[(2 * k * h)..<(3 * k * h)]))
    let img_attn_key_norm_scale = state_dict["\(prefix).img_attn.norm.key_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normK.weight.copy(from: try! Tensor<Float>(numpy: img_attn_key_norm_scale))
    let img_attn_query_norm_scale = state_dict["\(prefix).img_attn.norm.query_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normQ.weight.copy(from: try! Tensor<Float>(numpy: img_attn_query_norm_scale))
    if let contextUnifyheads = contextUnifyheads {
      let attn_to_add_out_weight = state_dict["\(prefix).txt_attn.proj.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: try! Tensor<Float>(numpy: attn_to_add_out_weight))
      let attn_to_add_out_bias = state_dict["\(prefix).txt_attn.proj.bias"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.bias.copy(
        from: try! Tensor<Float>(numpy: attn_to_add_out_bias))
    }
    let attn_to_out_0_weight = state_dict["\(prefix).img_attn.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_out_0_weight))
    let attn_to_out_0_bias = state_dict["\(prefix).img_attn.proj.bias"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.bias.copy(
      from: try! Tensor<Float>(numpy: attn_to_out_0_bias))
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      let ff_context_linear_1_weight = state_dict["\(prefix).txt_mlp.0.weight"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.weight.copy(
        from: try! Tensor<Float>(numpy: ff_context_linear_1_weight))
      let ff_context_linear_1_bias = state_dict["\(prefix).txt_mlp.0.bias"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.bias.copy(
        from: try! Tensor<Float>(numpy: ff_context_linear_1_bias))
      let ff_context_out_projection_weight = state_dict[
        "\(prefix).txt_mlp.2.weight"
      ].to(
        torch.float
      ).cpu().numpy()
      contextOutProjection.weight.copy(
        from: try! Tensor<Float>(numpy: ff_context_out_projection_weight))
      let ff_context_out_projection_bias = state_dict[
        "\(prefix).txt_mlp.2.bias"
      ].to(
        torch.float
      ).cpu().numpy()
      contextOutProjection.bias.copy(
        from: try! Tensor<Float>(numpy: ff_context_out_projection_bias))
    }
    let ff_linear_1_weight = state_dict["\(prefix).img_mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.weight.copy(
      from: try! Tensor<Float>(numpy: ff_linear_1_weight))
    let ff_linear_1_bias = state_dict["\(prefix).img_mlp.0.bias"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.bias.copy(
      from: try! Tensor<Float>(numpy: ff_linear_1_bias))
    let ff_out_projection_weight = state_dict["\(prefix).img_mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.weight.copy(
      from: try! Tensor<Float>(numpy: ff_out_projection_weight))
    let ff_out_projection_bias = state_dict["\(prefix).img_mlp.2.bias"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.bias.copy(
      from: try! Tensor<Float>(numpy: ff_out_projection_bias))
    let norm1_context_linear_weight = state_dict[
      "\(prefix).txt_mod.lin.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let norm1_context_linear_bias = state_dict[
      "\(prefix).txt_mod.lin.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<(contextBlockPreOnly ? 2 : 6) {
      contextAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: norm1_context_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
      contextAdaLNs[i].bias.copy(
        from: try! Tensor<Float>(
          numpy: norm1_context_linear_bias[(k * h * i)..<(k * h * (i + 1))]))
    }
    let norm1_linear_weight = state_dict["\(prefix).img_mod.lin.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let norm1_linear_bias = state_dict["\(prefix).img_mod.lin.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: norm1_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
      xAdaLNs[i].bias.copy(
        from: try! Tensor<Float>(
          numpy: norm1_linear_bias[(k * h * i)..<(k * h * (i + 1))]))
    }
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, c, rot], [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, c, rot], [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let c = Input()
  let rot = Input()
  let xAdaLNs = (0..<3).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3])
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3])
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  keys = keys.permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .permuted(0, 2, 1, 3)
  values = values.permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xLinear1 = Dense(count: k * h * 4, name: "x_linear1")
  let xOutProjection = Dense(count: k * h, name: "x_out_proj")
  out = xOutProjection(
    Functional.concat(axis: 2, out, xLinear1(xOut).GELU(approximate: .tanh).contiguous()))
  out = xIn + xChunks[2] .* out
  let reader: (PythonObject) -> Void = { state_dict in
    let linear1_weight = state_dict["\(prefix).linear1.weight"].to(
      torch.float
    ).cpu().numpy()
    let linear1_bias = state_dict["\(prefix).linear1.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: linear1_weight[..<(k * h), ...]))
    xToQueries.bias.copy(from: try! Tensor<Float>(numpy: linear1_bias[..<(k * h)]))
    xToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: linear1_weight[(k * h)..<(2 * k * h), ...]))
    xToKeys.bias.copy(from: try! Tensor<Float>(numpy: linear1_bias[(k * h)..<(2 * k * h)]))
    xToValues.weight.copy(
      from: try! Tensor<Float>(numpy: linear1_weight[(2 * k * h)..<(3 * k * h), ...]))
    xToValues.bias.copy(
      from: try! Tensor<Float>(numpy: linear1_bias[(2 * k * h)..<(3 * k * h)]))
    let key_norm_scale = state_dict["\(prefix).norm.key_norm.scale"].to(torch.float).cpu().numpy()
    normK.weight.copy(from: try! Tensor<Float>(numpy: key_norm_scale))
    let query_norm_scale = state_dict["\(prefix).norm.query_norm.scale"].to(torch.float).cpu()
      .numpy()
    normQ.weight.copy(from: try! Tensor<Float>(numpy: query_norm_scale))
    xLinear1.weight.copy(
      from: try! Tensor<Float>(numpy: linear1_weight[(3 * k * h)..<(7 * k * h), ...]))
    xLinear1.bias.copy(
      from: try! Tensor<Float>(numpy: linear1_bias[(3 * k * h)..<(7 * k * h)]))
    let linear2_weight = state_dict["\(prefix).linear2.weight"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.weight.copy(
      from: try! Tensor<Float>(numpy: linear2_weight))
    let linear2_bias = state_dict["\(prefix).linear2.bias"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.bias.copy(
      from: try! Tensor<Float>(numpy: linear2_bias))
    let norm1_linear_weight = state_dict["\(prefix).modulation.lin.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let norm1_linear_bias = state_dict["\(prefix).modulation.lin.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    for i in 0..<3 {
      xAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: norm1_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
      xAdaLNs[i].bias.copy(
        from: try! Tensor<Float>(
          numpy: norm1_linear_bias[(k * h * i)..<(k * h * (i + 1))]))
    }
  }
  return (reader, Model([x, c, rot], [out]))
}

func MMDiT(b: Int, h: Int, w: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t = Input()
  let y = Input()
  let contextIn = Input()
  let rot = Input()
  let xEmbedder = Dense(count: 3072, name: "x_embedder")
  var out = xEmbedder(x)
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 3072)
  var vec = tEmbedder(t)
  let (yMlp0, yMlp2, yEmbedder) = VectorEmbedder(channels: 3072)
  vec = vec + yEmbedder(y)
  let contextEmbedder = Dense(count: 3072, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  let c = vec.reshaped([b, 1, 3072]).swish()
  var readers = [(PythonObject) -> Void]()
  for i in 0..<19 {
    let (reader, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: 24, b: b, t: 256, hw: h * w,
      contextBlockPreOnly: false)
    let blockOut = block(context, out, c, rot)
    context = blockOut[0]
    out = blockOut[1]
    readers.append(reader)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<38 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: 24, b: b, t: 256, hw: h * w,
      contextBlockPreOnly: i == 37)
    out = block(out, c, rot)
    readers.append(reader)
  }
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(c)) .* normFinal(out) + shift(c)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_weight = state_dict["img_in.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_in_bias = state_dict["img_in.bias"].to(torch.float)
      .cpu().numpy()
    xEmbedder.weight.copy(from: try! Tensor<Float>(numpy: img_in_weight))
    xEmbedder.bias.copy(from: try! Tensor<Float>(numpy: img_in_bias))
    let t_embedder_mlp_0_weight = state_dict["time_in.in_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_0_bias = state_dict["time_in.in_layer.bias"].to(torch.float)
      .cpu().numpy()
    tMlp0.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight))
    tMlp0.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias))
    let t_embedder_mlp_2_weight = state_dict["time_in.out_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_2_bias = state_dict["time_in.out_layer.bias"].to(torch.float)
      .cpu().numpy()
    tMlp2.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight))
    tMlp2.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias))
    let y_embedder_mlp_0_weight = state_dict["vector_in.in_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_mlp_0_bias = state_dict["vector_in.in_layer.bias"].to(torch.float)
      .cpu().numpy()
    yMlp0.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_0_weight))
    yMlp0.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_0_bias))
    let y_embedder_mlp_2_weight = state_dict["vector_in.out_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_mlp_2_bias = state_dict["vector_in.out_layer.bias"].to(torch.float)
      .cpu().numpy()
    yMlp2.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_2_weight))
    yMlp2.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_2_bias))
    let txt_embedder_weight = state_dict["txt_in.weight"].to(
      torch.float
    ).cpu().numpy()
    contextEmbedder.weight.copy(from: try! Tensor<Float>(numpy: txt_embedder_weight))
    let txt_embedder_bias = state_dict["txt_in.bias"].to(
      torch.float
    ).cpu().numpy()
    contextEmbedder.bias.copy(from: try! Tensor<Float>(numpy: txt_embedder_bias))
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_linear_weight = state_dict[
      "final_layer.adaLN_modulation.1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let norm_out_linear_bias = state_dict[
      "final_layer.adaLN_modulation.1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    shift.weight.copy(
      from: try! Tensor<Float>(numpy: norm_out_linear_weight[0..<3072, ...]))
    scale.weight.copy(
      from: try! Tensor<Float>(numpy: norm_out_linear_weight[3072..<(3072 * 2), ...]))
    shift.bias.copy(
      from: try! Tensor<Float>(numpy: norm_out_linear_bias[0..<3072]))
    scale.bias.copy(
      from: try! Tensor<Float>(numpy: norm_out_linear_bias[3072..<(3072 * 2)]))
    let proj_out_weight = state_dict["final_layer.linear.weight"].to(
      torch.float
    ).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    let proj_out_bias = state_dict["final_layer.linear.bias"].to(
      torch.float
    ).cpu().numpy()
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x, t, y, contextIn, rot], [out]))
}

let (reader, dit) = MMDiT(b: 1, h: 64, w: 64)

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy()).toGPU(1))
  let tTensor = graph.variable(
    timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000).toGPU(1))
  let yTensor = graph.variable(
    try! Tensor<Float>(numpy: y.to(torch.float).cpu().numpy()).toGPU(1))
  let cTensor = graph.variable(
    try! Tensor<Float>(numpy: txt.to(torch.float).cpu().numpy()).toGPU(1))
  let rotTensor = graph.variable(.CPU, .NHWC(1, 4096 + 256, 1, 128), of: Float.self)
  for i in 0..<256 {
    for k in 0..<8 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 8)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<28 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 8) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<28 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<64 {
    for x in 0..<64 {
      let i = y * 64 + x + 256
      for k in 0..<8 {
        let theta = 0 * 1.0 / pow(10_000, Double(k) / 8)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<28 {
        let theta = Double(y) * 1.0 / pow(10_000, Double(k) / 28)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 8) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<28 {
        let theta = Double(x) * 1.0 / pow(10_000, Double(k) / 28)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
      }
    }
  }
  let rotTensorGPU = rotTensor.toGPU(1)
  dit.maxConcurrency = .limit(1)
  /*
  dit.compile(inputs: xTensor, tTensor, yTensor, cTensor, rotTensorGPU)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, tTensor, yTensor, cTensor, rotTensorGPU))
  */
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/flux_1_schnell_f32.ckpt") {
    $0.write("dit", model: dit)
  }
  */
}

func ResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  out = Swish()(out)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = norm2(out)
  out = Swish()(out)
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = nin(x) + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).norm1.weight"].to(torch.float).cpu().numpy()
    let norm1_bias = state_dict["\(prefix).norm1.bias"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let conv1_weight = state_dict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2_bias = state_dict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).nin_shortcut.weight"].to(torch.float).cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).nin_shortcut.bias"].to(torch.float).cpu()
        .numpy()
      ninShortcut.weight.copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
  }
  return (reader, Model([x], [out]))
}

func AttnBlock(prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let hw = width * height
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let k = tokeys(out).reshaped([batchSize, inChannels, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    batchSize, inChannels, hw,
  ])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let v = tovalues(out).reshaped([batchSize, inChannels, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = x + projOut(out.reshaped([batchSize, inChannels, height, width]))
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_weight = state_dict["\(prefix).norm.weight"].to(torch.float).cpu().numpy()
    let norm_bias = state_dict["\(prefix).norm.bias"].to(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: norm_bias))
    let k_weight = state_dict["\(prefix).k.weight"].to(torch.float).cpu().numpy()
    let k_bias = state_dict["\(prefix).k.bias"].to(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["\(prefix).q.weight"].to(torch.float).cpu().numpy()
    let q_bias = state_dict["\(prefix).q.bias"].to(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["\(prefix).v.weight"].to(torch.float).cpu().numpy()
    let v_bias = state_dict["\(prefix).v.bias"].to(torch.float).cpu().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["\(prefix).proj_out.weight"].to(torch.float).cpu().numpy()
    let proj_out_bias = state_dict["\(prefix).proj_out.bias"].to(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x], [out]))
}

func Encoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var readers = [(PythonObject) -> Void]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (reader, block) = ResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", outChannels: channel,
        shortcut: previousChannel != channel)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      // Conv always pad left first, then right, and pad top first then bottom.
      // Thus, we cannot have (0, 1, 0, 1) (left 0, right 1, top 0, bottom 1) padding as in
      // Stable Diffusion. Instead, we pad to (2, 1, 2, 1) and simply discard the first row and first column.
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])))
      out = conv2d(out).reshaped(
        [batchSize, channel, height, width], offset: [0, 0, 1, 1],
        strides: [channel * (height + 1) * (width + 1), (height + 1) * (width + 1), width + 1, 1])
      let downLayer = i
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict["encoder.down.\(downLayer).downsample.conv.weight"].to(
          torch.float
        ).cpu()
          .numpy()
        let conv_bias = state_dict["encoder.down.\(downLayer).downsample.conv.bias"].to(torch.float)
          .cpu().numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "encoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["encoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["encoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    let norm_out_weight = state_dict["encoder.norm_out.weight"].to(torch.float).cpu().numpy()
    let norm_out_bias = state_dict["encoder.norm_out.bias"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["encoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["encoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

func Decoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "decoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "decoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  var readers = [(PythonObject) -> Void]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", outChannels: channel,
        shortcut: previousChannel != channel)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
      let upLayer = i
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict["decoder.up.\(upLayer).upsample.conv.weight"].to(torch.float)
          .cpu().numpy()
        let conv_bias = state_dict["decoder.up.\(upLayer).upsample.conv.bias"].to(torch.float).cpu()
          .numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = Swish()(out)
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["decoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["decoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["decoder.norm_out.weight"].to(torch.float).cpu().numpy()
    let norm_out_bias = state_dict["decoder.norm_out.bias"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["decoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["decoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}
let vae_state_dict = ae.state_dict()
graph.withNoGrad {
  var zTensor = graph.variable(try! Tensor<Float>(numpy: z.to(torch.float).cpu().numpy())).toGPU(0)
  // Already processed out.
  zTensor = (1.0 / 0.3611) * zTensor + 0.1159
  debugPrint(zTensor)
  let (decoderReader, decoder) = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  decoder.compile(inputs: zTensor)
  decoderReader(vae_state_dict)
  let image = decoder(inputs: zTensor)[0].as(of: Float.self)
  debugPrint(image)
  let (encoderReader, encoder) = Encoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  encoder.compile(inputs: image)
  encoderReader(vae_state_dict)
  let _ = encoder(inputs: image)[0].as(of: Float.self)
  graph.openStore("/home/liu/workspace/swift-diffusion/flux_1_vae_f32.ckpt") {
    $0.write("encoder", model: encoder)
    $0.write("decoder", model: decoder)
  }
}
