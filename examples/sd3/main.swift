import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let torch = Python.import("torch")
let nodes = Python.import("nodes")
torch.set_grad_enabled(false)
/*
from nodes import (
    ConditioningSetTimestepRange,
    CheckpointLoaderSimple,
    ConditioningCombine,
    VAEDecode,
    CLIPTextEncode,
    NODE_CLASS_MAPPINGS,
    KSampler,
    ConditioningZeroOut,
    init_custom_nodes,
)
*/

nodes.init_custom_nodes()
let triplecliploader = nodes.NODE_CLASS_MAPPINGS["TripleCLIPLoader"]()
let triplecliploader_11 = triplecliploader.load_clip(
  clip_name1: "clip_g.safetensors",
  clip_name2: "clip_l.safetensors",
  clip_name3: "t5xxl_fp8_e4m3fn.safetensors"
)

let cliptextencode = nodes.CLIPTextEncode()
let cliptextencode_71 = cliptextencode.encode(
  text: "", clip: triplecliploader_11[0]
)

let emptysd3latentimage = nodes.NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
let emptysd3latentimage_135 = emptysd3latentimage.generate(
  width: 1024, height: 1024, batch_size: 1
)

let checkpointloadersimple = nodes.CheckpointLoaderSimple()
let checkpointloadersimple_252 = checkpointloadersimple.load_checkpoint(
  ckpt_name: "sdv3/2b_1024/sd3_medium.safetensors"
)

let cliptextencodesd3 = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncodeSD3"]()
let cliptextencodesd3_273 = cliptextencodesd3.encode(
  clip_l:
    "photo of a young woman with long, wavy brown hair lying down in grass, top down shot, summer, warm, laughing, joy, fun",
  clip_g:
    "photo of a young woman with long, wavy brown hair lying down in grass, top down shot, summer, warm, laughing, joy, fun",
  t5xxl: "\n",
  empty_padding: "none",
  clip: triplecliploader_11[0]
)

let modelsamplingsd3 = nodes.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
let conditioningzeroout = nodes.ConditioningZeroOut()
let conditioningsettimesteprange = nodes.ConditioningSetTimestepRange()
let conditioningcombine = nodes.ConditioningCombine()
let ksampler = nodes.KSampler()
let vaedecode = nodes.VAEDecode()

let modelsamplingsd3_13 = modelsamplingsd3.patch(
  shift: 3, model: checkpointloadersimple_252[0]
)

let conditioningzeroout_67 = conditioningzeroout.zero_out(
  conditioning: cliptextencode_71[0]
)

let conditioningsettimesteprange_68 = conditioningsettimesteprange.set_range(
  start: 0.1,
  end: 1,
  conditioning: conditioningzeroout_67[0]
)

let conditioningsettimesteprange_70 = conditioningsettimesteprange.set_range(
  start: 0, end: 0.1, conditioning: cliptextencode_71[0]
)

let conditioningcombine_69 = conditioningcombine.combine(
  conditioning_1: conditioningsettimesteprange_68[0],
  conditioning_2: conditioningsettimesteprange_70[0]
)
/*
let ksampler_271 = ksampler.sample(
  seed: 23,
  steps: 28,
  cfg: 4.5,
  sampler_name: "dpmpp_2m",
  scheduler: "sgm_uniform",
  denoise: 1,
  model: modelsamplingsd3_13[0],
  positive: cliptextencodesd3_273[0],
  negative: conditioningcombine_69[0],
  latent_image: emptysd3latentimage_135[0]
)

let vaedecode_231 = vaedecode.decode(
  samples: ksampler_271[0],
  vae: checkpointloadersimple_252[2]
)

print("vaedecode_231 \(vaedecode_231)")
*/

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([2, 16, 128, 128]).to(torch.float16).cuda()
let y = torch.randn([2, 2048]).to(torch.float16).cuda() * 0.01
let t = torch.full([2], 1000).cuda()
let c = torch.randn([2, 154, 4096]).to(torch.float16).cuda() * 0.01

let diffusion_model = modelsamplingsd3_13[0].model.diffusion_model.cuda()

let out = diffusion_model(x: x, timesteps: t, context: c, y: y)
print(out)

let state_dict = modelsamplingsd3_13[0].model.state_dict()

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
  let fc0 = Dense(count: channels, name: "y_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "y_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLP(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  let contextK = contextToKeys(contextOut)
  let contextQ = contextToQueries(contextOut)
  let contextV = contextToValues(contextOut)
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  let xK = xToKeys(xOut)
  let xQ = xToQueries(xOut)
  let xV = xToValues(xOut)
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  keys = keys.reshaped([b, t + hw, h, k]).permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
    .permuted(0, 2, 1, 3)
  values = values.reshaped([b, t + hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = Dense(count: k * h)
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h)
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = x + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextFc1: Model?
  let contextFc2: Model?
  if !contextBlockPreOnly {
    let contextMlp: Model
    (contextFc1, contextFc2, contextMlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4)
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut = contextOut + contextChunks[5]
      .* contextMlp(contextNorm2(contextOut) .* (1 + contextChunks[4]) + contextChunks[3])
  } else {
    contextFc1 = nil
    contextFc2 = nil
  }
  let (xFc1, xFc2, xMlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut = xOut + xChunks[5] .* xMlp(xNorm2(xOut) .* (1 + xChunks[4]) + xChunks[3])
  let reader: (PythonObject) -> Void = { state_dict in
    let context_block_attn_qkv_weight = state_dict["\(prefix).context_block.attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    let context_block_attn_qkv_bias = state_dict["\(prefix).context_block.attn.qkv.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_weight[..<(k * h), ...]))
    contextToQueries.bias.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_bias[..<(k * h)]))
    contextToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_weight[(k * h)..<(2 * k * h), ...]))
    contextToKeys.bias.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_bias[(k * h)..<(2 * k * h)]))
    contextToValues.weight.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...])
    )
    contextToValues.bias.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_bias[(2 * k * h)..<(3 * k * h)]))
    let x_block_attn_qkv_weight = state_dict["\(prefix).x_block.attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_block_attn_qkv_bias = state_dict["\(prefix).x_block.attn.qkv.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_qkv_weight[..<(k * h), ...]))
    xToQueries.bias.copy(from: try! Tensor<Float>(numpy: x_block_attn_qkv_bias[..<(k * h)]))
    xToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_qkv_weight[(k * h)..<(2 * k * h), ...]))
    xToKeys.bias.copy(from: try! Tensor<Float>(numpy: x_block_attn_qkv_bias[(k * h)..<(2 * k * h)]))
    xToValues.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...]))
    xToValues.bias.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_qkv_bias[(2 * k * h)..<(3 * k * h)]))
    if let contextUnifyheads = contextUnifyheads {
      let context_block_attn_proj_weight = state_dict["\(prefix).context_block.attn.proj.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      let context_block_attn_proj_bias = state_dict["\(prefix).context_block.attn.proj.bias"].to(
        torch.float
      ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: try! Tensor<Float>(numpy: context_block_attn_proj_weight))
      contextUnifyheads.bias.copy(
        from: try! Tensor<Float>(numpy: context_block_attn_proj_bias))
    }
    let x_block_attn_proj_weight = state_dict["\(prefix).x_block.attn.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_block_attn_proj_bias = state_dict["\(prefix).x_block.attn.proj.bias"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_proj_weight))
    xUnifyheads.bias.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_proj_bias))
    if let contextFc1 = contextFc1, let contextFc2 = contextFc2 {
      let context_block_mlp_fc1_weight = state_dict["\(prefix).context_block.mlp.fc1.weight"].to(
        torch.float
      ).cpu().numpy()
      let context_block_mlp_fc1_bias = state_dict["\(prefix).context_block.mlp.fc1.bias"].to(
        torch.float
      ).cpu().numpy()
      contextFc1.weight.copy(
        from: try! Tensor<Float>(numpy: context_block_mlp_fc1_weight))
      contextFc1.bias.copy(
        from: try! Tensor<Float>(numpy: context_block_mlp_fc1_bias))
      let context_block_mlp_fc2_weight = state_dict["\(prefix).context_block.mlp.fc2.weight"].to(
        torch.float
      ).cpu().numpy()
      let context_block_mlp_fc2_bias = state_dict["\(prefix).context_block.mlp.fc2.bias"].to(
        torch.float
      ).cpu().numpy()
      contextFc2.weight.copy(
        from: try! Tensor<Float>(numpy: context_block_mlp_fc2_weight))
      contextFc2.bias.copy(
        from: try! Tensor<Float>(numpy: context_block_mlp_fc2_bias))
    }
    let x_block_mlp_fc1_weight = state_dict["\(prefix).x_block.mlp.fc1.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_block_mlp_fc1_bias = state_dict["\(prefix).x_block.mlp.fc1.bias"].to(
      torch.float
    ).cpu().numpy()
    xFc1.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_mlp_fc1_weight))
    xFc1.bias.copy(
      from: try! Tensor<Float>(numpy: x_block_mlp_fc1_bias))
    let x_block_mlp_fc2_weight = state_dict["\(prefix).x_block.mlp.fc2.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_block_mlp_fc2_bias = state_dict["\(prefix).x_block.mlp.fc2.bias"].to(
      torch.float
    ).cpu().numpy()
    xFc2.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_mlp_fc2_weight))
    xFc2.bias.copy(
      from: try! Tensor<Float>(numpy: x_block_mlp_fc2_bias))
    let context_block_adaln_modulation_weight = state_dict[
      "\(prefix).context_block.adaLN_modulation.1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let context_block_adaln_modulation_bias = state_dict[
      "\(prefix).context_block.adaLN_modulation.1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<(contextBlockPreOnly ? 2 : 6) {
      contextAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: context_block_adaln_modulation_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
      contextAdaLNs[i].bias.copy(
        from: try! Tensor<Float>(
          numpy: context_block_adaln_modulation_bias[(k * h * i)..<(k * h * (i + 1))]))
    }
    let x_block_adaln_modulation_weight = state_dict["\(prefix).x_block.adaLN_modulation.1.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let x_block_adaln_modulation_bias = state_dict["\(prefix).x_block.adaLN_modulation.1.bias"].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: x_block_adaln_modulation_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
      xAdaLNs[i].bias.copy(
        from: try! Tensor<Float>(
          numpy: x_block_adaln_modulation_bias[(k * h * i)..<(k * h * (i + 1))]))
    }
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, c], [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, c], [xOut]))
  }
}

func MMDiT(b: Int, h: Int, w: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t = Input()
  let y = Input()
  let contextIn = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 1536, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 1536, h * w]).transposed(1, 2)
  let posEmbed = Parameter<Float>(.GPU(0), .NHWC(1, 192, 192, 1536))
  let spatialPosEmbed = posEmbed.reshaped(
    [1, h, w, 1536], offset: [0, (192 - h) / 2, (192 - w) / 2, 0],
    strides: [192 * 192 * 1536, 192 * 1536, 1536, 1]
  ).contiguous().reshaped([1, h * w, 1536])
  out = spatialPosEmbed + out
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 1536)
  let (yMlp0, yMlp2, yEmbedder) = VectorEmbedder(channels: 1536)
  let c = (tEmbedder(t) + yEmbedder(y)).reshaped([b, 1, 1536]).swish()
  let contextEmbedder = Dense(count: 1536, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = JointTransformerBlock(
      prefix: "diffusion_model.joint_blocks.\(i)", k: 64, h: 24, b: b, t: 154, hw: h * w,
      contextBlockPreOnly: i == 23)
    let blockOut = block(context, out, c)
    if i == 23 {
      out = blockOut
    } else {
      context = blockOut[0]
      out = blockOut[1]
    }
    readers.append(reader)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shift = Dense(count: 1536, name: "ada_ln_0")
  let scale = Dense(count: 1536, name: "ada_ln_1")
  out = (1 + scale(c)) .* normFinal(out) + shift(c)
  let linear = Dense(count: 2 * 2 * 16, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 16]).permuted(0, 5, 1, 3, 2, 4).contiguous().reshaped([
    b, 16, h * 2, w * 2,
  ])
  let reader: (PythonObject) -> Void = { state_dict in
    let x_embedder_proj_weight = state_dict["diffusion_model.x_embedder.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_embedder_proj_bias = state_dict["diffusion_model.x_embedder.proj.bias"].to(torch.float)
      .cpu().numpy()
    xEmbedder.weight.copy(from: try! Tensor<Float>(numpy: x_embedder_proj_weight))
    xEmbedder.bias.copy(from: try! Tensor<Float>(numpy: x_embedder_proj_bias))
    let pos_embed = state_dict["diffusion_model.pos_embed"].to(torch.float).cpu().numpy()
    posEmbed.weight.copy(from: try! Tensor<Float>(numpy: pos_embed))
    let t_embedder_mlp_0_weight = state_dict["diffusion_model.t_embedder.mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_0_bias = state_dict["diffusion_model.t_embedder.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    tMlp0.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight))
    tMlp0.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias))
    let t_embedder_mlp_2_weight = state_dict["diffusion_model.t_embedder.mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_2_bias = state_dict["diffusion_model.t_embedder.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    tMlp2.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight))
    tMlp2.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias))
    let y_embedder_mlp_0_weight = state_dict["diffusion_model.y_embedder.mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_mlp_0_bias = state_dict["diffusion_model.y_embedder.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    yMlp0.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_0_weight))
    yMlp0.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_0_bias))
    let y_embedder_mlp_2_weight = state_dict["diffusion_model.y_embedder.mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_mlp_2_bias = state_dict["diffusion_model.y_embedder.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    yMlp2.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_2_weight))
    yMlp2.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_2_bias))
    let context_embedder_weight = state_dict["diffusion_model.context_embedder.weight"].to(
      torch.float
    ).cpu().numpy()
    let context_embedder_bias = state_dict["diffusion_model.context_embedder.bias"].to(torch.float)
      .cpu().numpy()
    contextEmbedder.weight.copy(from: try! Tensor<Float>(numpy: context_embedder_weight))
    contextEmbedder.bias.copy(from: try! Tensor<Float>(numpy: context_embedder_bias))
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_adaln_modulation_weight = state_dict[
      "diffusion_model.final_layer.adaLN_modulation.1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let final_layer_adaln_modulation_bias = state_dict[
      "diffusion_model.final_layer.adaLN_modulation.1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    shift.weight.copy(
      from: try! Tensor<Float>(numpy: final_layer_adaln_modulation_weight[0..<1536, ...]))
    shift.bias.copy(from: try! Tensor<Float>(numpy: final_layer_adaln_modulation_bias[0..<1536]))
    scale.weight.copy(
      from: try! Tensor<Float>(numpy: final_layer_adaln_modulation_weight[1536..<(1536 * 2), ...]))
    scale.bias.copy(
      from: try! Tensor<Float>(numpy: final_layer_adaln_modulation_bias[1536..<(1536 * 2)]))
    let final_layer_linear_weight = state_dict["diffusion_model.final_layer.linear.weight"].to(
      torch.float
    ).cpu().numpy()
    let final_layer_linear_bias = state_dict["diffusion_model.final_layer.linear.bias"].to(
      torch.float
    ).cpu().numpy()
    linear.weight.copy(from: try! Tensor<Float>(numpy: final_layer_linear_weight))
    linear.bias.copy(from: try! Tensor<Float>(numpy: final_layer_linear_bias))
  }
  return (reader, Model([x, t, contextIn, y], [out]))
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

let (reader, dit) = MMDiT(b: 2, h: 64, w: 64)

let graph = DynamicGraph()

graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy()).toGPU(0))
  let tTensor = graph.variable(
    timeEmbedding(timesteps: 1000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000).toGPU(0))
  let cTensor = graph.variable(try! Tensor<Float>(numpy: c.to(torch.float).cpu().numpy()).toGPU(0))
  let yTensor = graph.variable(try! Tensor<Float>(numpy: y.to(torch.float).cpu().numpy()).toGPU(0))
  dit.compile(inputs: xTensor, tTensor, cTensor, yTensor)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, tTensor, cTensor, yTensor))
}
