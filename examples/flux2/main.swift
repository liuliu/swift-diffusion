import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

typealias FloatType = Float16

let torch = Python.import("torch")
let PIL = Python.import("PIL")

torch.set_grad_enabled(false)

let torch_device = torch.device("cuda")

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let flux2_util = Python.import("flux2.util")

var mistral = flux2_util.load_mistral_small_embedder()
let model = flux2_util.load_flow_model(model_name: "flux.2-dev", debug_mode: false, device: "cpu")

// print(mistral)
print(model)
mistral = mistral.cpu()

let x = torch.randn([1, 4096, 128]).to(torch.bfloat16).cuda()
let txt = torch.randn([1, 512, 15360]).to(torch.bfloat16).cuda() * 0.01
var img_ids = torch.zeros([64, 64, 4])
img_ids[..., ..., 1] = img_ids[..., ..., 1] + torch.arange(64)[..., Python.None]
img_ids[..., ..., 2] = img_ids[..., ..., 2] + torch.arange(64)[Python.None, ...]
img_ids = img_ids.reshape([1, 4096, 4]).cuda()
var txt_ids = torch.zeros([512, 4])
txt_ids[..., 3] = txt_ids[..., 3] + torch.arange(512)
txt_ids = txt_ids.reshape([1, 512, 4]).cuda()
let t = torch.full([1], 1).to(torch.bfloat16).cuda()
let guidance = torch.full([1], 4.0).to(torch.bfloat16).cuda()

let output = model(x, img_ids, t, txt, txt_ids, guidance)

let state_dict = model.cpu().state_dict()

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
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

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, noBias: true, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, noBias: true, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context).to(.Float16) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, noBias: true, name: "c_k")
  let contextToQueries = Dense(count: k * h, noBias: true, name: "c_q")
  let contextToValues = Dense(count: k * h, noBias: true, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(
    scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
  let out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  /*
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
  */
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = Dense(count: k * h, noBias: true, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: context)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 3, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + (contextChunks[5]
      .* contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3]))
      .to(of: contextOut)
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xW1, xW2, xW3, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 3, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut + (xChunks[5] .* xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])).to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
    let txt_attn_qkv_weight = state_dict["\(prefix).txt_attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[..<(k * h), ...]))
    )
    contextToQueries.weight.to(.unifiedMemory)
    contextToKeys.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[(k * h)..<(2 * k * h), ...])))
    contextToKeys.weight.to(.unifiedMemory)
    contextToValues.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: txt_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...]))
    )
    contextToValues.weight.to(.unifiedMemory)
    let txt_attn_key_norm_scale = state_dict["\(prefix).txt_attn.norm.key_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normAddedK.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt_attn_key_norm_scale)))
    let txt_attn_query_norm_scale = state_dict["\(prefix).txt_attn.norm.query_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normAddedQ.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt_attn_query_norm_scale)))
    let img_attn_qkv_weight = state_dict["\(prefix).img_attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: img_attn_qkv_weight[..<(k * h), ...]))
    )
    xToQueries.weight.to(.unifiedMemory)
    xToKeys.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: img_attn_qkv_weight[(k * h)..<(2 * k * h), ...])))
    xToKeys.weight.to(.unifiedMemory)
    xToValues.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: img_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...])))
    xToValues.weight.to(.unifiedMemory)
    let img_attn_key_norm_scale = state_dict["\(prefix).img_attn.norm.key_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normK.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: img_attn_key_norm_scale)))
    normK.weight.to(.unifiedMemory)
    let img_attn_query_norm_scale = state_dict["\(prefix).img_attn.norm.query_norm.scale"].to(
      torch.float
    ).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: img_attn_query_norm_scale)))
    normQ.weight.to(.unifiedMemory)
    if let contextUnifyheads = contextUnifyheads {
      let attn_to_add_out_weight = state_dict["\(prefix).txt_attn.proj.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: attn_to_add_out_weight)))
      contextUnifyheads.weight.to(.unifiedMemory)
    }
    let attn_to_out_0_weight = state_dict["\(prefix).img_attn.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: attn_to_out_0_weight)))
    xUnifyheads.weight.to(.unifiedMemory)
    if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
      let ff_context_linear_1_weight = state_dict["\(prefix).txt_mlp.0.weight"].to(
        torch.float
      ).cpu().numpy()
      contextW1.weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: ff_context_linear_1_weight[..<(k * h * 3), ...])))
      contextW1.weight.to(.unifiedMemory)
      contextW3.weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(
            numpy: ff_context_linear_1_weight[(k * h * 3)..<(k * h * 6), ...])))
      contextW3.weight.to(.unifiedMemory)
      let ff_context_out_projection_weight = state_dict[
        "\(prefix).txt_mlp.2.weight"
      ].to(
        torch.float
      ).cpu().numpy()
      contextW2.weight.copy(
        from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: ff_context_out_projection_weight)))
      contextW2.weight.to(.unifiedMemory)
    }
    let ff_linear_1_weight = state_dict["\(prefix).img_mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    xW1.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: ff_linear_1_weight[..<(k * h * 3), ...])))
    xW1.weight.to(.unifiedMemory)
    xW3.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: ff_linear_1_weight[(k * h * 3)..<(k * h * 6), ...])))
    xW3.weight.to(.unifiedMemory)
    let ff_out_projection_weight = state_dict["\(prefix).img_mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    xW2.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: ff_out_projection_weight)))
    xW2.weight.to(.unifiedMemory)
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, rot] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, rot] + contextChunks + xChunks, [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(
    scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  /*
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
  */
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
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xW1 = Dense(count: k * h * 3, noBias: true, name: "x_w1")
  let xW3 = Dense(count: k * h * 3, noBias: true, name: "x_w3")
  let xW2 = Dense(count: k * h, noBias: true, name: "x_w2")
  out = xUnifyheads(out) + xW2(xW3(xOut) .* xW1(xOut).swish())
  out = xIn + (xChunks[2] .* out).to(of: xIn)
  let reader: (PythonObject) -> Void = { state_dict in
    let linear1_weight = state_dict["\(prefix).linear1.weight"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: linear1_weight[..<(k * h), ...])))
    xToQueries.weight.to(.unifiedMemory)
    xToKeys.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear1_weight[(k * h)..<(2 * k * h), ...])))
    xToKeys.weight.to(.unifiedMemory)
    xToValues.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear1_weight[(2 * k * h)..<(3 * k * h), ...])))
    xToValues.weight.to(.unifiedMemory)
    let key_norm_scale = state_dict["\(prefix).norm.key_norm.scale"].to(torch.float).cpu().numpy()
    normK.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: key_norm_scale)))
    normK.weight.to(.unifiedMemory)
    let query_norm_scale = state_dict["\(prefix).norm.query_norm.scale"].to(torch.float).cpu()
      .numpy()
    normQ.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: query_norm_scale)))
    normQ.weight.to(.unifiedMemory)
    xW1.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear1_weight[(3 * k * h)..<(6 * k * h), ...])))
    xW1.weight.to(.unifiedMemory)
    xW3.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear1_weight[(6 * k * h)..<(9 * k * h), ...])))
    xW3.weight.to(.unifiedMemory)
    let linear2_weight = state_dict["\(prefix).linear2.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: linear2_weight[..., 0..<(k * h)])))
    xUnifyheads.weight.to(.unifiedMemory)
    xW2.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: linear2_weight[..., (k * h)..<(k * h * 4)])))
    xW2.weight.to(.unifiedMemory)
  }
  return (reader, Model([x, rot] + xChunks, [out]))
}

func Flux2(b: Int, h: Int, w: Int, guidanceEmbed: Bool) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let contextIn = Input()
  let rot = Input()
  let t = Input()
  let xEmbedder = Dense(count: 6144, noBias: true, name: "x_embedder")
  var out = xEmbedder(x).to(.Float32)
  let contextEmbedder = Dense(count: 6144, noBias: true, name: "context_embedder")
  var context = contextEmbedder(contextIn).to(.Float32)
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: 6144, name: "t")
  let (gMlp0, gMlp2, gEmbedder) = MLPEmbedder(channels: 6144, name: "guidance")
  let g = Input()
  var vec = tEmbedder(t)
  vec = vec + gEmbedder(g)
  vec = vec.swish()
  let xAdaLNs = (0..<6).map { Dense(count: 6144, noBias: true, name: "x_ada_ln_\($0)") }
  let contextAdaLNs = (0..<6).map { Dense(count: 6144, noBias: true, name: "context_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(vec) }
  var contextChunks = contextAdaLNs.map { $0(vec) }
  xChunks[1] = 1 + xChunks[1]
  xChunks[4] = 1 + xChunks[4]
  contextChunks[1] = 1 + contextChunks[1]
  contextChunks[4] = 1 + contextChunks[4]
  var readers = [(PythonObject) -> Void]()
  for i in 0..<8 {
    let (reader, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: 48, b: b, t: 512, hw: h * w,
      contextBlockPreOnly: false)
    let blockOut = block([context, out, rot] + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    readers.append(reader)
  }
  let singleAdaLNs = (0..<3).map { Dense(count: 6144, noBias: true, name: "single_ada_ln_\($0)") }
  var singleChunks = singleAdaLNs.map { $0(vec) }
  singleChunks[1] = 1 + singleChunks[1]
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<48 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: 48, b: b, t: 512, hw: h * w,
      contextBlockPreOnly: i == 47)
    out = block([out, rot] + singleChunks)
    readers.append(reader)
  }
  let scale = Dense(count: 6144, noBias: true, name: "ada_ln_0")
  let shift = Dense(count: 6144, noBias: true, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 32, noBias: true, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_weight = state_dict["img_in.weight"].to(
      torch.float
    ).cpu().numpy()
    xEmbedder.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: img_in_weight)))
    xEmbedder.weight.to(.unifiedMemory)
    let txt_in_weight = state_dict["txt_in.weight"].to(
      torch.float
    ).cpu().numpy()
    contextEmbedder.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt_in_weight)))
    contextEmbedder.weight.to(.unifiedMemory)
    let t_embedder_mlp_0_weight = state_dict["time_in.in_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    tMlp0.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight)))
    tMlp0.weight.to(.unifiedMemory)
    let t_embedder_mlp_2_weight = state_dict["time_in.out_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    tMlp2.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight)))
    tMlp2.weight.to(.unifiedMemory)
    let guidance_embedder_mlp_0_weight = state_dict["guidance_in.in_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    gMlp0.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: guidance_embedder_mlp_0_weight)))
    gMlp0.weight.to(.unifiedMemory)
    let guidance_embedder_mlp_2_weight = state_dict["guidance_in.out_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    gMlp2.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: guidance_embedder_mlp_2_weight)))
    gMlp2.weight.to(.unifiedMemory)
    let double_stream_modulation_img_lin_weight = state_dict[
      "double_stream_modulation_img.lin.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let double_stream_modulation_txt_lin_weight = state_dict[
      "double_stream_modulation_txt.lin.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(
            numpy: double_stream_modulation_img_lin_weight[(6144 * i)..<(6144 * (i + 1)), ...])))
      xAdaLNs[i].weight.to(.unifiedMemory)
      contextAdaLNs[i].weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(
            numpy: double_stream_modulation_txt_lin_weight[(6144 * i)..<(6144 * (i + 1)), ...])))
      contextAdaLNs[i].weight.to(.unifiedMemory)
    }
    let single_stream_modulation_lin_weight = state_dict[
      "single_stream_modulation.lin.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<3 {
      singleAdaLNs[i].weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(
            numpy: single_stream_modulation_lin_weight[(6144 * i)..<(6144 * (i + 1)), ...])))
      singleAdaLNs[i].weight.to(.unifiedMemory)
    }
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_adaLN_modulation_weight = state_dict["final_layer.adaLN_modulation.1.weight"]
      .to(torch.float).cpu().numpy()
    shift.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: final_layer_adaLN_modulation_weight[0..<6144, ...])))
    shift.weight.to(.unifiedMemory)
    scale.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: final_layer_adaLN_modulation_weight[6144..<(6144 * 2), ...])
      ))
    scale.weight.to(.unifiedMemory)
    let proj_out_weight = state_dict["final_layer.linear.weight"].to(
      torch.float
    ).cpu().numpy()
    projOut.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: proj_out_weight)))
    projOut.weight.to(.unifiedMemory)
  }
  return (reader, Model([x, contextIn, rot, t, g], [out]))
}

let (reader, dit) = Flux2(b: 1, h: 64, w: 64, guidanceEmbed: true)

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

graph.withNoGrad {
  let xTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy())).toGPU(2)
  ).reshaped(.HWC(1, 4096, 128))
  let contextTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: txt.to(torch.float).cpu().numpy())).toGPU(2)
  ).reshaped(.HWC(1, 512, 15360))
  let tTensor = graph.variable(
    Tensor<FloatType>(
      from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(2))
  let gTensor = graph.variable(
    Tensor<FloatType>(
      from: timeEmbedding(timesteps: 4000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(2))
  let rotTensor = graph.variable(.CPU, .NHWC(1, 4096 + 512, 1, 128), of: Float.self)
  for i in 0..<512 {
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 16) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 16 * 2) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 16 * 2) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = Double(i) * 1.0 / pow(2_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 16 * 3) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 16 * 3) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<64 {
    for x in 0..<64 {
      let i = y * 64 + x + 512
      for k in 0..<16 {
        let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(y) * 1.0 / pow(2_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 16) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 16) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(x) * 1.0 / pow(2_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 16 * 2) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 16 * 2) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = 0 * 1.0 / pow(2_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 16 * 3) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 16 * 3) * 2 + 1] = Float(sintheta)
      }
    }
  }
  let rotTensorGPU = DynamicGraph.Tensor<FloatType>(from: rotTensor).toGPU(2)
  dit.maxConcurrency = .limit(1)
  dit.compile(inputs: xTensor, contextTensor, rotTensorGPU, tTensor, gTensor)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, contextTensor, rotTensorGPU, tTensor, gTensor))
}
