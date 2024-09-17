import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let torch = Python.import("torch")
let diffusers = Python.import("diffusers")

let base_model = "black-forest-labs/FLUX.1-dev"
let controlnet_model_union = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"

let controlnet_union = diffusers.FluxControlNetModel.from_pretrained(
  controlnet_model_union, torch_dtype: torch.bfloat16)
let controlnet = diffusers.models.FluxMultiControlNetModel([controlnet_union])  // we always recommend loading via FluxMultiControlNetModel
let pipe = diffusers.FluxControlNetPipeline.from_pretrained(
  base_model, controlnet: controlnet, torch_dtype: torch.bfloat16)

pipe.to("cuda")

let prompt = "A bohemian-style female travel blogger with sun-kissed skin and messy beach waves."
let control_image_depth = diffusers.utils.load_image(
  "/home/liu/workspace/diffusers/assets_depth.jpg")
let control_mode_depth = 2

let control_image_canny = diffusers.utils.load_image(
  "/home/liu/workspace/diffusers/assets_canny.jpg")
let control_mode_canny = 0

debugPrint(pipe.controlnet)

let (width, height) = control_image_depth.size.tuple2

/*
let image = pipe(
    prompt,
    control_image: [control_image_depth, control_image_canny],
    control_mode: [control_mode_depth, control_mode_canny],
    width: width,
    height: height,
    controlnet_conditioning_scale: [0.2, 0.4],
    num_inference_steps: 24,
    guidance_scale: 3.5,
    generator: torch.manual_seed(42)
).images[0]

image.save("/home/liu/workspace/swift-diffusion/generated.png")
*/

torch.set_grad_enabled(false)

let state_dict = pipe.controlnet.nets[0].state_dict()
print(state_dict.keys())

let x = torch.randn([1, 4096, 64]).to(torch.bfloat16).cuda()
let condition_x = torch.randn([1, 4096, 64]).to(torch.bfloat16).cuda()
let condition_mode = torch.zeros([1, 1]).to(torch.device("cuda"), dtype: torch.long)
let y = torch.randn([1, 768]).to(torch.bfloat16).cuda() * 0.01
let txt = torch.randn([1, 512, 4096]).to(torch.bfloat16).cuda() * 0.01
var img_ids = torch.zeros([64, 64, 3])
img_ids[..., ..., 1] = img_ids[..., ..., 1] + torch.arange(64)[..., Python.None]
img_ids[..., ..., 2] = img_ids[..., ..., 2] + torch.arange(64)[Python.None, ...]
img_ids = img_ids.reshape([4096, 3]).cuda()
let txt_ids = torch.zeros([512, 3]).cuda()
let t = torch.full([1], 1).to(torch.bfloat16).cuda()
let guidance = torch.full([1], 3.5).to(torch.bfloat16).cuda()
let return_dict = pipe.controlnet.nets[0](
  x, condition_x, condition_mode, encoder_hidden_states: txt, pooled_projections: y, timestep: t,
  img_ids: img_ids, txt_ids: txt_ids, guidance: guidance)
print(return_dict)

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

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
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
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
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
    let txt_attn_q_weight = state_dict["\(prefix).attn.add_q_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_q_bias = state_dict["\(prefix).attn.add_q_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: txt_attn_q_weight))
    contextToQueries.bias.copy(
      from: try! Tensor<Float>(numpy: txt_attn_q_bias))
    let txt_attn_k_weight = state_dict["\(prefix).attn.add_k_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_k_bias = state_dict["\(prefix).attn.add_k_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: txt_attn_k_weight))
    contextToKeys.bias.copy(
      from: try! Tensor<Float>(numpy: txt_attn_k_bias))
    let txt_attn_v_weight = state_dict["\(prefix).attn.add_v_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_v_bias = state_dict["\(prefix).attn.add_v_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToValues.weight.copy(
      from: try! Tensor<Float>(numpy: txt_attn_v_weight)
    )
    contextToValues.bias.copy(
      from: try! Tensor<Float>(numpy: txt_attn_v_bias))
    let txt_attn_key_norm_scale = state_dict["\(prefix).attn.norm_added_k.weight"].to(
      torch.float
    ).cpu().numpy()
    normAddedK.weight.copy(from: try! Tensor<Float>(numpy: txt_attn_key_norm_scale))
    let txt_attn_query_norm_scale = state_dict["\(prefix).attn.norm_added_q.weight"].to(
      torch.float
    ).cpu().numpy()
    normAddedQ.weight.copy(from: try! Tensor<Float>(numpy: txt_attn_query_norm_scale))
    let img_attn_q_weight = state_dict["\(prefix).attn.to_q.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_q_bias = state_dict["\(prefix).attn.to_q.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: img_attn_q_weight))
    xToQueries.bias.copy(from: try! Tensor<Float>(numpy: img_attn_q_bias))
    let img_attn_k_weight = state_dict["\(prefix).attn.to_k.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_k_bias = state_dict["\(prefix).attn.to_k.bias"].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: img_attn_k_weight))
    xToKeys.bias.copy(from: try! Tensor<Float>(numpy: img_attn_k_bias))
    let img_attn_v_weight = state_dict["\(prefix).attn.to_v.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_v_bias = state_dict["\(prefix).attn.to_v.bias"].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: try! Tensor<Float>(numpy: img_attn_v_weight))
    xToValues.bias.copy(
      from: try! Tensor<Float>(numpy: img_attn_v_bias))
    let img_attn_key_norm_scale = state_dict["\(prefix).attn.norm_k.weight"].to(
      torch.float
    ).cpu().numpy()
    normK.weight.copy(from: try! Tensor<Float>(numpy: img_attn_key_norm_scale))
    let img_attn_query_norm_scale = state_dict["\(prefix).attn.norm_q.weight"].to(
      torch.float
    ).cpu().numpy()
    normQ.weight.copy(from: try! Tensor<Float>(numpy: img_attn_query_norm_scale))
    if let contextUnifyheads = contextUnifyheads {
      let attn_to_add_out_weight = state_dict["\(prefix).attn.to_add_out.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: try! Tensor<Float>(numpy: attn_to_add_out_weight))
      let attn_to_add_out_bias = state_dict["\(prefix).attn.to_add_out.bias"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.bias.copy(
        from: try! Tensor<Float>(numpy: attn_to_add_out_bias))
    }
    let attn_to_out_0_weight = state_dict["\(prefix).attn.to_out.0.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_out_0_weight))
    let attn_to_out_0_bias = state_dict["\(prefix).attn.to_out.0.bias"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.bias.copy(
      from: try! Tensor<Float>(numpy: attn_to_out_0_bias))
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      let ff_context_linear_1_weight = state_dict["\(prefix).ff_context.net.0.proj.weight"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.weight.copy(
        from: try! Tensor<Float>(numpy: ff_context_linear_1_weight))
      let ff_context_linear_1_bias = state_dict["\(prefix).ff_context.net.0.proj.bias"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.bias.copy(
        from: try! Tensor<Float>(numpy: ff_context_linear_1_bias))
      let ff_context_out_projection_weight = state_dict[
        "\(prefix).ff_context.net.2.weight"
      ].to(
        torch.float
      ).cpu().numpy()
      contextOutProjection.weight.copy(
        from: try! Tensor<Float>(numpy: ff_context_out_projection_weight))
      let ff_context_out_projection_bias = state_dict[
        "\(prefix).ff_context.net.2.bias"
      ].to(
        torch.float
      ).cpu().numpy()
      contextOutProjection.bias.copy(
        from: try! Tensor<Float>(numpy: ff_context_out_projection_bias))
    }
    let ff_linear_1_weight = state_dict["\(prefix).ff.net.0.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.weight.copy(
      from: try! Tensor<Float>(numpy: ff_linear_1_weight))
    let ff_linear_1_bias = state_dict["\(prefix).ff.net.0.proj.bias"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.bias.copy(
      from: try! Tensor<Float>(numpy: ff_linear_1_bias))
    let ff_out_projection_weight = state_dict["\(prefix).ff.net.2.weight"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.weight.copy(
      from: try! Tensor<Float>(numpy: ff_out_projection_weight))
    let ff_out_projection_bias = state_dict["\(prefix).ff.net.2.bias"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.bias.copy(
      from: try! Tensor<Float>(numpy: ff_out_projection_bias))
    let norm1_context_linear_weight = state_dict[
      "\(prefix).norm1_context.linear.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let norm1_context_linear_bias = state_dict[
      "\(prefix).norm1_context.linear.bias"
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
    let norm1_linear_weight = state_dict["\(prefix).norm1.linear.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let norm1_linear_bias = state_dict["\(prefix).norm1.linear.bias"]
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
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xLinear1 = Dense(count: k * h * 4, name: "x_linear1")
  let xOutProjection = Dense(count: k * h, name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + xChunks[2] .* out
  let reader: (PythonObject) -> Void = { state_dict in
    let linear1_q_weight = state_dict["\(prefix).attn.to_q.weight"].to(
      torch.float
    ).cpu().numpy()
    let linear1_q_bias = state_dict["\(prefix).attn.to_q.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: linear1_q_weight))
    xToQueries.bias.copy(from: try! Tensor<Float>(numpy: linear1_q_bias))
    let linear1_k_weight = state_dict["\(prefix).attn.to_k.weight"].to(
      torch.float
    ).cpu().numpy()
    let linear1_k_bias = state_dict["\(prefix).attn.to_k.bias"].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: linear1_k_weight))
    xToKeys.bias.copy(from: try! Tensor<Float>(numpy: linear1_k_bias))
    let linear1_v_weight = state_dict["\(prefix).attn.to_v.weight"].to(
      torch.float
    ).cpu().numpy()
    let linear1_v_bias = state_dict["\(prefix).attn.to_v.bias"].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: try! Tensor<Float>(numpy: linear1_v_weight))
    xToValues.bias.copy(
      from: try! Tensor<Float>(numpy: linear1_v_bias))
    let key_norm_scale = state_dict["\(prefix).attn.norm_k.weight"].to(torch.float).cpu().numpy()
    normK.weight.copy(from: try! Tensor<Float>(numpy: key_norm_scale))
    let query_norm_scale = state_dict["\(prefix).attn.norm_k.weight"].to(torch.float).cpu()
      .numpy()
    normQ.weight.copy(from: try! Tensor<Float>(numpy: query_norm_scale))
    let linear1_proj_weight = state_dict["\(prefix).proj_mlp.weight"].to(
      torch.float
    ).cpu().numpy()
    let linear1_proj_bias = state_dict["\(prefix).proj_mlp.bias"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.weight.copy(
      from: try! Tensor<Float>(numpy: linear1_proj_weight))
    xLinear1.bias.copy(
      from: try! Tensor<Float>(numpy: linear1_proj_bias))
    let linear2_weight = state_dict["\(prefix).proj_out.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: try! Tensor<Float>(numpy: linear2_weight[..., 0..<(k * h)]))
    xOutProjection.weight.copy(
      from: try! Tensor<Float>(numpy: linear2_weight[..., (k * h)..<(k * h * 5)]))
    let linear2_bias = state_dict["\(prefix).proj_out.bias"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.bias.copy(
      from: try! Tensor<Float>(numpy: linear2_bias))
    let norm1_linear_weight = state_dict["\(prefix).norm.linear.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let norm1_linear_bias = state_dict["\(prefix).norm.linear.bias"]
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

func MMDiT(b: Int, h: Int, w: Int, guidanceEmbed: Bool) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t = Input()
  let y = Input()
  let conditionX = Input()
  let conditionMode = Input()
  let contextIn = Input()
  let rot = Input()
  let guidance: Input?
  let xEmbedder = Dense(count: 3072, name: "x_embedder")
  var out = xEmbedder(x)
  let controlnetXEmbedder = Dense(count: 3072, name: "controlnet_x_embedder")
  out = out + controlnetXEmbedder(conditionX)
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: 3072, name: "t")
  var vec = tEmbedder(t)
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = MLPEmbedder(channels: 3072, name: "guidance")
    let g = Input()
    vec = vec + gEmbedder(g)
    guidance = g
    gMlp0 = mlp0
    gMlp2 = mlp2
  } else {
    gMlp0 = nil
    gMlp2 = nil
    guidance = nil
  }
  let (yMlp0, yMlp2, yEmbedder) = MLPEmbedder(channels: 3072, name: "vector")
  vec = vec + yEmbedder(y)
  let contextEmbedder = Dense(count: 3072, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  let controlnetModeEmbedder = Embedding(
    Float.self, vocabularySize: 10, embeddingSize: 3072, name: "controlnet_mode_embedder")
  context = Functional.concat(
    axis: 1, controlnetModeEmbedder(conditionMode).reshaped([1, 1, 3072]), context)
  let c = vec.reshaped([b, 1, 3072]).swish()
  var readers = [(PythonObject) -> Void]()
  var zeroConvs = [Dense]()
  var outs = [Model.IO]()
  for i in 0..<5 {
    let (reader, block) = JointTransformerBlock(
      prefix: "transformer_blocks.\(i)", k: 128, h: 24, b: b, t: 513, hw: h * w,
      contextBlockPreOnly: false)
    let blockOut = block(context, out, c, rot)
    context = blockOut[0]
    out = blockOut[1]
    let zeroConv = Dense(count: 3072, name: "zero_conv")
    outs.append(zeroConv(out))
    zeroConvs.append(zeroConv)
    readers.append(reader)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<10 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_transformer_blocks.\(i)", k: 128, h: 24, b: b, t: 513, hw: h * w,
      contextBlockPreOnly: i == 9)
    out = block(out, c, rot)
    let zeroConv = Dense(count: 3072, name: "zero_conv")
    if i < 9 {
      let input = out.reshaped(
        [b, h * w, 3072], offset: [0, 513, 0], strides: [(513 + h * w) * 3072, 3072, 1]
      )
      .contiguous()
      outs.append(zeroConv(input))
    } else {
      outs.append(zeroConv(out))
    }
    zeroConvs.append(zeroConv)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_weight = state_dict["x_embedder.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_in_bias = state_dict["x_embedder.bias"].to(torch.float)
      .cpu().numpy()
    xEmbedder.weight.copy(from: try! Tensor<Float>(numpy: img_in_weight))
    xEmbedder.bias.copy(from: try! Tensor<Float>(numpy: img_in_bias))
    let controlnet_x_embedder_weight = state_dict["controlnet_x_embedder.weight"].to(
      torch.float
    ).cpu().numpy()
    let controlnet_x_embedder_bias = state_dict["controlnet_x_embedder.bias"].to(torch.float)
      .cpu().numpy()
    controlnetXEmbedder.weight.copy(from: try! Tensor<Float>(numpy: controlnet_x_embedder_weight))
    controlnetXEmbedder.bias.copy(from: try! Tensor<Float>(numpy: controlnet_x_embedder_bias))
    let controlnet_mode_embedder_weight = state_dict["controlnet_mode_embedder.weight"].to(
      torch.float
    ).cpu().numpy()
    controlnetModeEmbedder.weight.copy(
      from: try! Tensor<Float>(numpy: controlnet_mode_embedder_weight))
    let t_embedder_mlp_0_weight = state_dict["time_text_embed.timestep_embedder.linear_1.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let t_embedder_mlp_0_bias = state_dict["time_text_embed.timestep_embedder.linear_1.bias"].to(
      torch.float
    )
    .cpu().numpy()
    tMlp0.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight))
    tMlp0.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias))
    let t_embedder_mlp_2_weight = state_dict["time_text_embed.timestep_embedder.linear_2.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let t_embedder_mlp_2_bias = state_dict["time_text_embed.timestep_embedder.linear_2.bias"].to(
      torch.float
    )
    .cpu().numpy()
    tMlp2.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight))
    tMlp2.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias))
    if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
      let g_embedder_mlp_0_weight = state_dict["time_text_embed.guidance_embedder.linear_1.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      let g_embedder_mlp_0_bias = state_dict["time_text_embed.guidance_embedder.linear_1.bias"].to(
        torch.float
      )
      .cpu().numpy()
      gMlp0.weight.copy(from: try! Tensor<Float>(numpy: g_embedder_mlp_0_weight))
      gMlp0.bias.copy(from: try! Tensor<Float>(numpy: g_embedder_mlp_0_bias))
      let g_embedder_mlp_2_weight = state_dict["time_text_embed.guidance_embedder.linear_2.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      let g_embedder_mlp_2_bias = state_dict["time_text_embed.guidance_embedder.linear_2.bias"].to(
        torch.float
      )
      .cpu().numpy()
      gMlp2.weight.copy(from: try! Tensor<Float>(numpy: g_embedder_mlp_2_weight))
      gMlp2.bias.copy(from: try! Tensor<Float>(numpy: g_embedder_mlp_2_bias))
    }
    let y_embedder_mlp_0_weight = state_dict["time_text_embed.text_embedder.linear_1.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_mlp_0_bias = state_dict["time_text_embed.text_embedder.linear_1.bias"].to(
      torch.float
    )
    .cpu().numpy()
    yMlp0.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_0_weight))
    yMlp0.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_0_bias))
    let y_embedder_mlp_2_weight = state_dict["time_text_embed.text_embedder.linear_2.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_mlp_2_bias = state_dict["time_text_embed.text_embedder.linear_2.bias"].to(
      torch.float
    )
    .cpu().numpy()
    yMlp2.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_2_weight))
    yMlp2.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_2_bias))
    let txt_embedder_weight = state_dict["context_embedder.weight"].to(
      torch.float
    ).cpu().numpy()
    contextEmbedder.weight.copy(from: try! Tensor<Float>(numpy: txt_embedder_weight))
    let txt_embedder_bias = state_dict["context_embedder.bias"].to(
      torch.float
    ).cpu().numpy()
    contextEmbedder.bias.copy(from: try! Tensor<Float>(numpy: txt_embedder_bias))
    for reader in readers {
      reader(state_dict)
    }
    for i in 0..<5 {
      let controlnet_blocks_weight = state_dict["controlnet_blocks.\(i).weight"].to(torch.float)
        .cpu().numpy()
      let controlnet_blocks_bias =
        ((1.0 / 8) * state_dict["controlnet_blocks.\(i).bias"].to(torch.float).cpu())
        .numpy()
      zeroConvs[i].weight.copy(from: try! Tensor<Float>(numpy: controlnet_blocks_weight))
      zeroConvs[i].bias.copy(from: try! Tensor<Float>(numpy: controlnet_blocks_bias))
    }
    for i in 0..<10 {
      let controlnet_single_blocks_weight = state_dict["controlnet_single_blocks.\(i).weight"].to(
        torch.float
      ).cpu().numpy()
      let controlnet_single_blocks_bias =
        ((1.0 / 8)
        * state_dict["controlnet_single_blocks.\(i).bias"].to(
          torch.float
        ).cpu()).numpy()
      zeroConvs[i + 5].weight.copy(from: try! Tensor<Float>(numpy: controlnet_single_blocks_weight))
      zeroConvs[i + 5].bias.copy(from: try! Tensor<Float>(numpy: controlnet_single_blocks_bias))
    }
  }
  return (
    reader,
    Model(
      [x, conditionX, conditionMode, t, y, contextIn, rot] + (guidance.map { [$0] } ?? []), outs)
  )
}

let (reader, dit) = MMDiT(b: 1, h: 64, w: 64, guidanceEmbed: true)

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy()).toGPU(1))
  let conditionXTensor = graph.variable(
    try! Tensor<Float>(numpy: condition_x.to(torch.float).cpu().numpy()).toGPU(1))
  var conditionMode = Tensor<Int32>(.CPU, .C(1))
  conditionMode[0] = 0
  let conditionModeTensor = graph.variable(conditionMode).toGPU(1)
  let tTensor = graph.variable(
    timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000).toGPU(1))
  let yTensor = graph.variable(
    try! Tensor<Float>(numpy: y.to(torch.float).cpu().numpy()).toGPU(1))
  let cTensor = graph.variable(
    try! Tensor<Float>(numpy: txt.to(torch.float).cpu().numpy()).toGPU(1))
  let rotTensor = graph.variable(.CPU, .NHWC(1, 4096 + 513, 1, 128), of: Float.self)
  for i in 0..<513 {
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
      let i = y * 64 + x + 513
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
  let gTensor = graph.variable(
    timeEmbedding(timesteps: 3500, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000).toGPU(1))
  dit.maxConcurrency = .limit(1)
  dit.compile(
    inputs: xTensor, conditionXTensor, conditionModeTensor, tTensor, yTensor, cTensor, rotTensorGPU,
    gTensor)
  reader(state_dict)
  debugPrint(
    dit(
      inputs: xTensor, conditionXTensor, conditionModeTensor, tTensor, yTensor, cTensor,
      rotTensorGPU, gTensor))
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/controlnet_union_pro_flux_1_dev_1.0_f32.ckpt"
  ) {
    $0.write("controlnet", model: dit)
  }
}
