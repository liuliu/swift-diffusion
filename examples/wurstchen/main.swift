import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let inference_utils = Python.import("inference.utils")
let core_utils = Python.import("core.utils")
let train = Python.import("train")

let torch = Python.import("torch")
let yaml = Python.import("yaml")

let device = torch.device("cuda:0")

let loaded_config_c = yaml.safe_load(
  """
  # GLOBAL STUFF
  model_version: 3.6B
  dtype: bfloat16

  effnet_checkpoint_path: /home/liu/workspace/StableCascade/models/effnet_encoder.safetensors
  previewer_checkpoint_path: /home/liu/workspace/StableCascade/models/previewer.safetensors
  generator_checkpoint_path: /home/liu/workspace/StableCascade/models/stage_c_bf16.safetensors
  """)
let core_c = train.WurstCoreC(config_dict: loaded_config_c, device: device, training: false)

let loaded_config_b = yaml.safe_load(
  """
  # GLOBAL STUFF
  model_version: 3B
  dtype: bfloat16

  # For demonstration purposes in reconstruct_images.ipynb
  webdataset_path: file:inference/imagenet_1024.tar
  batch_size: 4
  image_size: 1024
  grad_accum_steps: 1

  effnet_checkpoint_path: /home/liu/workspace/StableCascade/models/effnet_encoder.safetensors
  stage_a_checkpoint_path: /home/liu/workspace/StableCascade/models/stage_a.safetensors
  generator_checkpoint_path: /home/liu/workspace/StableCascade/models/stage_b_bf16.safetensors
  """)
let core_b = train.WurstCoreB(config_dict: loaded_config_b, device: device, training: false)

let extras_c = core_c.setup_extras_pre()
let models_c = core_c.setup_models(extras_c)
models_c.text_model.float().eval()
models_c.generator.float().eval().requires_grad_(false)

let extras_b = core_b.setup_extras_pre()
var models_b = core_b.setup_models(extras_b, skip_clip: true)
models_b = train.WurstCoreB.Models(
  generator: models_b.generator, effnet: models_b.effnet, stage_a: models_b.stage_a,
  tokenizer: models_c.tokenizer, text_model: models_c.text_model
)
models_b.generator.float().eval().requires_grad_(false)

let batch_size = 4
let caption =
  "Cinematic photo of an anthropomorphic polar bear sitting in a cafe reading a book and having a coffee"
let (stage_c_latent_shape, stage_b_latent_shape) = inference_utils.calculate_latent_sizes(
  1024, 1024, batch_size: batch_size
).tuple2

// Stage C Parameters
extras_c.sampling_configs["cfg"] = 4
extras_c.sampling_configs["shift"] = 2
extras_c.sampling_configs["timesteps"] = 20
extras_c.sampling_configs["t_start"] = 1.0

// Stage B Parameters
extras_b.sampling_configs["cfg"] = 1.1
extras_b.sampling_configs["shift"] = 1
extras_b.sampling_configs["timesteps"] = 10
extras_b.sampling_configs["t_start"] = 1.0

// PREPARE CONDITIONS
let batch: [PythonObject: PythonObject] = [
  "captions": PythonObject([caption, caption, caption, caption])
]
let conditions = core_c.get_conditions(
  batch, models_c, extras_c, is_eval: true, is_unconditional: false, eval_image_embeds: false)
let unconditions = core_c.get_conditions(
  batch, models_c, extras_c, is_eval: true, is_unconditional: true, eval_image_embeds: false)
let conditions_b = core_b.get_conditions(
  batch, models_b, extras_b, is_eval: true, is_unconditional: false)
let unconditions_b = core_b.get_conditions(
  batch, models_b, extras_b, is_eval: true, is_unconditional: true)

torch.set_grad_enabled(false)
/*
torch.set_autocast_gpu_dtype(torch.bfloat16)
torch.set_autocast_enabled(true)
torch.autocast_increment_nesting()
torch.set_autocast_cache_enabled(true)
*/
torch.manual_seed(42)
/*
let sampling_c = extras_c.gdf.sample(
  models_c.generator, conditions, stage_c_latent_shape, unconditions, device: device, cfg: extras_c.sampling_configs["cfg"], sampler: extras_c.sampling_configs["sampler"], shift: extras_c.sampling_configs["shift"], timesteps: extras_c.sampling_configs["timesteps"], t_start: extras_c.sampling_configs["t_start"]
)
var sampled_c: PythonObject? = nil
for _ in 0..<Int(extras_c.sampling_configs["timesteps"])! {
  sampled_c = sampling_c.__next__().tuple3.0
}

conditions_b["effnet"] = sampled_c!
unconditions_b["effnet"] = torch.zeros_like(sampled_c)

let sampling_b = extras_b.gdf.sample(
  models_b.generator, conditions_b, stage_b_latent_shape, unconditions_b, device: device, cfg: extras_c.sampling_configs["cfg"], sampler: extras_c.sampling_configs["sampler"], shift: extras_c.sampling_configs["shift"], timesteps: extras_c.sampling_configs["timesteps"], t_start: extras_c.sampling_configs["t_start"]
)
var sampled_b: PythonObject? = nil
for _ in 0..<Int(extras_b.sampling_configs["timesteps"])! {
  sampled_b = sampling_b.__next__().tuple3.0
}
let sampled = models_b.stage_a.decode(sampled_b).float()

inference_utils.save_images(sampled)
*/

let x = torch.randn([2, 16, 24, 24]).cuda()
let clip_text = torch.randn([2, 77, 1280]).cuda()
let clip_text_pooled = torch.zeros([2, 1, 1280]).cuda()
let clip_img = torch.zeros([2, 1, 768]).cuda()
let r = 0.9936 * torch.ones([2]).cuda()
let result = models_c.generator(x, r, clip_text, clip_text_pooled, clip_img)
// print(result)

// First, get weights from core_c.

let state_dict = models_c.generator.state_dict()

// print(state_dict.keys())

func ResBlock(prefix: String, batchSize: Int, channels: Int, skip: Bool) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let depthwise = Convolution(
    groups: channels, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = depthwise(x)
  let norm = LayerNorm(epsilon: 1e-6, axis: [1])
  out = norm(out)
  let xSkip: Input?
  if skip {
    let xSkipIn = Input()
    out = Functional.concat(axis: 1, out, xSkipIn)
    xSkip = xSkipIn
  } else {
    xSkip = nil
  }
  let convIn = Convolution(
    groups: 1, filters: channels * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convIn(out).GELU()
  let Gx = out.reduced(.norm2, axis: [2, 3])
  let Nx = Gx .* (1 / Gx.reduced(.mean, axis: [1])) + 1e-6
  let gamma = Parameter<Float>(.GPU(0), .NCHW(1, channels * 4, 1, 1), initBound: 1)
  let beta = Parameter<Float>(.GPU(0), .NCHW(1, channels * 4, 1, 1), initBound: 1)
  out = gamma .* (out .* Nx) + beta + out
  let convOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out) + x
  let reader: (PythonObject) -> Void = { state_dict in
    let depthwise_weight = state_dict["\(prefix).depthwise.weight"].float().cpu().numpy()
    depthwise.weight.copy(from: try! Tensor<Float>(numpy: depthwise_weight))
    let depthwise_bias = state_dict["\(prefix).depthwise.bias"].float().cpu().numpy()
    depthwise.bias.copy(from: try! Tensor<Float>(numpy: depthwise_bias))
    var layerNorm_weight = Tensor<Float>(.CPU, .NCHW(1, 2048, 1, 1))
    for i in 0..<2048 {
      layerNorm_weight[0, i, 0, 0] = 1
    }
    norm.weight.copy(from: layerNorm_weight)
    let channelwise_0_weight = state_dict["\(prefix).channelwise.0.weight"].float().cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: channelwise_0_weight))
    let channelwise_0_bias = state_dict["\(prefix).channelwise.0.bias"].float().cpu().numpy()
    convIn.bias.copy(from: try! Tensor<Float>(numpy: channelwise_0_bias))
    let channelwise_2_gamma = state_dict["\(prefix).channelwise.2.gamma"].float().cpu().numpy()
    gamma.weight.copy(from: try! Tensor<Float>(numpy: channelwise_2_gamma))
    let channelwise_2_beta = state_dict["\(prefix).channelwise.2.beta"].float().cpu().numpy()
    beta.weight.copy(from: try! Tensor<Float>(numpy: channelwise_2_beta))
    let channelwise_4_weight = state_dict["\(prefix).channelwise.4.weight"].float().cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: channelwise_4_weight))
    let channelwise_4_bias = state_dict["\(prefix).channelwise.4.bias"].float().cpu().numpy()
    convOut.bias.copy(from: try! Tensor<Float>(numpy: channelwise_4_bias))
  }
  if let xSkip = xSkip {
    return (Model([x, xSkip], [out]), reader)
  } else {
    return (Model([x], [out]), reader)
  }
}

func TimestepBlock(prefix: String, batchSize: Int, timeEmbedSize: Int, channels: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rEmbed = Input()
  let mapper = Dense(count: channels * 2)
  var gate = mapper(
    rEmbed.reshaped([batchSize, timeEmbedSize], offset: [0, 0], strides: [timeEmbedSize * 3, 1]))
  let mapperSca = Dense(count: channels * 2)
  gate =
    gate
    + mapperSca(
      rEmbed.reshaped(
        [batchSize, timeEmbedSize], offset: [0, timeEmbedSize], strides: [timeEmbedSize * 3, 1]))
  let mapperCrp = Dense(count: channels * 2)
  gate =
    gate
    + mapperCrp(
      rEmbed.reshaped(
        [batchSize, timeEmbedSize], offset: [0, timeEmbedSize * 2], strides: [timeEmbedSize * 3, 1])
    )
  let out =
    x
    .* (1
      + gate.reshaped(
        [batchSize, channels, 1, 1], offset: [0, 0, 0, 0], strides: [channels * 2, 1, 1, 1]))
    + gate.reshaped(
      [batchSize, channels, 1, 1], offset: [0, channels, 0, 0], strides: [channels * 2, 1, 1, 1])
  let reader: (PythonObject) -> Void = { state_dict in
    let mapper_weight = state_dict["\(prefix).mapper.weight"].float().cpu().numpy()
    mapper.weight.copy(from: try! Tensor<Float>(numpy: mapper_weight))
    let mapper_bias = state_dict["\(prefix).mapper.bias"].float().cpu().numpy()
    mapper.bias.copy(from: try! Tensor<Float>(numpy: mapper_bias))
    let mapper_sca_weight = state_dict["\(prefix).mapper_sca.weight"].float().cpu().numpy()
    mapperSca.weight.copy(from: try! Tensor<Float>(numpy: mapper_sca_weight))
    let mapper_sca_bias = state_dict["\(prefix).mapper_sca.bias"].float().cpu().numpy()
    mapperSca.bias.copy(from: try! Tensor<Float>(numpy: mapper_sca_bias))
    let mapper_crp_weight = state_dict["\(prefix).mapper_crp.weight"].float().cpu().numpy()
    mapperCrp.weight.copy(from: try! Tensor<Float>(numpy: mapper_crp_weight))
    let mapper_crp_bias = state_dict["\(prefix).mapper_crp.bias"].float().cpu().numpy()
    mapperCrp.bias.copy(from: try! Tensor<Float>(numpy: mapper_crp_bias))
  }
  return (Model([x, rEmbed], [out]), reader)
}

func MultiHeadAttention(prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let kv = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(kv).reshaped([b, hw + t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(kv).reshaped([b, hw + t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw + t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, hw + t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let in_proj_weight = state_dict["\(prefix).attention.attn.in_proj_weight"].float().cpu().numpy()
    let in_proj_bias = state_dict["\(prefix).attention.attn.in_proj_bias"].float().cpu().numpy()
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
    let out_proj_weight = state_dict["\(prefix).attention.attn.out_proj.weight"].float().cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).attention.attn.out_proj.bias"].float().cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: out_proj_bias))
  }
  return (Model([x, kv], [out]), reader)
}

func AttnBlock(
  prefix: String, batchSize: Int, channels: Int, nHead: Int, height: Int, width: Int, t: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let kv = Input()
  let kvMapper = Dense(count: channels)
  let kvOut = kvMapper(kv.swish())
  let norm = LayerNorm(epsilon: 1e-6, axis: [1])
  var out = norm(x).reshaped([batchSize, channels, height * width]).transposed(1, 2)
  let xKv = Functional.concat(axis: 1, out, kvOut)
  let k = channels / nHead
  let (multiHeadAttention, multiHeadAttentionReader) = MultiHeadAttention(
    prefix: prefix, k: k, h: nHead, b: batchSize, hw: height * width, t: t)
  out =
    x
    + multiHeadAttention(out, xKv).transposed(1, 2).reshaped([batchSize, channels, height, width])
  let reader: (PythonObject) -> Void = { state_dict in
    var norm_weight = Tensor<Float>(.CPU, .NCHW(1, channels, 1, 1))
    for i in 0..<channels {
      norm_weight[0, i, 0, 0] = 1
    }
    norm.weight.copy(from: norm_weight)
    let kv_mapper_1_weight = state_dict["\(prefix).kv_mapper.1.weight"].float().cpu().numpy()
    kvMapper.weight.copy(from: try! Tensor<Float>(numpy: kv_mapper_1_weight))
    let kv_mapper_1_bias = state_dict["\(prefix).kv_mapper.1.bias"].float().cpu().numpy()
    kvMapper.bias.copy(from: try! Tensor<Float>(numpy: kv_mapper_1_bias))
    multiHeadAttentionReader(state_dict)
  }
  return (Model([x, kv], [out]), reader)
}

func StageC(batchSize: Int, height: Int, width: Int, t: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rEmbed = Input()
  let conv2d = Convolution(
    groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = conv2d(x)
  let normIn = LayerNorm(epsilon: 1e-6, axis: [1])
  out = normIn(out)

  let clipText = Input()
  let clipTextPooled = Input()
  let clipImg = Input()
  let clipTextMapper = Dense(count: 2048)
  let clipTextMapped = clipTextMapper(clipText)
  let clipTextPooledMapper = Dense(count: 2048 * 4)
  let clipTextPooledMapped = clipTextPooledMapper(clipTextPooled).reshaped([batchSize, 4, 2048])
  let clipImgMapper = Dense(count: 2048 * 4)
  let clipImgMapped = clipImgMapper(clipImg).reshaped([batchSize, 4, 2048])
  let clipNorm = LayerNorm(epsilon: 1e-6, axis: [2])
  let clip = clipNorm(
    Functional.concat(axis: 1, clipTextMapped, clipTextPooledMapped, clipImgMapped))

  let blocks: [[Int]] = [[8, 24], [24, 8]]
  var readers: [(PythonObject) -> Void] = []
  var levelOutputs = [Model.IO]()
  for i in 0..<2 {
    if i > 0 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1])
      out = norm(out)
      let downscaler = Convolution(
        groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
      out = downscaler(out)
      readers.append { state_dict in
        var norm_weight = Tensor<Float>(.CPU, .NCHW(1, 2048, 1, 1))
        for i in 0..<2048 {
          norm_weight[0, i, 0, 0] = 1
        }
        norm.weight.copy(from: norm_weight)
        let blocks_0_weight = state_dict["down_downscalers.\(i).1.blocks.0.weight"].float().cpu()
          .numpy()
        downscaler.weight.copy(from: try! Tensor<Float>(numpy: blocks_0_weight))
        let blocks_0_bias = state_dict["down_downscalers.\(i).1.blocks.0.bias"].float().cpu()
          .numpy()
        downscaler.bias.copy(from: try! Tensor<Float>(numpy: blocks_0_bias))
      }
    }
    for j in 0..<blocks[0][i] {
      let (resBlock, resBlockReader) = ResBlock(
        prefix: "down_blocks.\(i).\(j * 3)", batchSize: batchSize, channels: 2048, skip: false)
      readers.append(resBlockReader)
      out = resBlock(out)
      let (timestepBlock, timestepBlockReader) = TimestepBlock(
        prefix: "down_blocks.\(i).\(j * 3 + 1)", batchSize: batchSize, timeEmbedSize: 64,
        channels: 2048)
      readers.append(timestepBlockReader)
      out = timestepBlock(out, rEmbed)
      let (attnBlock, attnBlockReader) = AttnBlock(
        prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        height: height, width: width, t: t)
      readers.append(attnBlockReader)
      out = attnBlock(out, clip)
    }
    if i < 2 - 1 {
      levelOutputs.append(out)
    }
  }

  var skip: Model.IO? = nil
  for i in 0..<2 {
    for j in 0..<blocks[1][i] {
      let (resBlock, resBlockReader) = ResBlock(
        prefix: "up_blocks.\(i).\(j * 3)", batchSize: batchSize, channels: 2048, skip: skip != nil)
      readers.append(resBlockReader)
      if let skip = skip {
        out = resBlock(out, skip)
      } else {
        out = resBlock(out)
      }
      skip = nil
      let (timestepBlock, timestepBlockReader) = TimestepBlock(
        prefix: "up_blocks.\(i).\(j * 3 + 1)", batchSize: batchSize, timeEmbedSize: 64,
        channels: 2048)
      readers.append(timestepBlockReader)
      out = timestepBlock(out, rEmbed)
      let (attnBlock, attnBlockReader) = AttnBlock(
        prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        height: height, width: width, t: t)
      readers.append(attnBlockReader)
      out = attnBlock(out, clip)
    }
    if i < 2 - 1 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1])
      out = norm(out)
      let upscaler = Convolution(
        groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
      out = upscaler(out)
      readers.append { state_dict in
        var norm_weight = Tensor<Float>(.CPU, .NCHW(1, 2048, 1, 1))
        for i in 0..<2048 {
          norm_weight[0, i, 0, 0] = 1
        }
        norm.weight.copy(from: norm_weight)
        let blocks_1_weight = state_dict["up_upscalers.\(i).1.blocks.1.weight"].float().cpu()
          .numpy()
        upscaler.weight.copy(from: try! Tensor<Float>(numpy: blocks_1_weight))
        let blocks_1_bias = state_dict["up_upscalers.\(i).1.blocks.1.bias"].float().cpu().numpy()
        upscaler.bias.copy(from: try! Tensor<Float>(numpy: blocks_1_bias))
      }
      skip = levelOutputs.removeLast()
    }
  }

  let normOut = LayerNorm(epsilon: 1e-6, axis: [1])
  out = normOut(out)
  let convOut = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out)

  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
    let embedding_1_weight = state_dict["embedding.1.weight"].float().cpu().numpy()
    conv2d.weight.copy(from: try! Tensor<Float>(numpy: embedding_1_weight))
    let embedding_1_bias = state_dict["embedding.1.bias"].float().cpu().numpy()
    conv2d.bias.copy(from: try! Tensor<Float>(numpy: embedding_1_bias))
    var norm_weight = Tensor<Float>(.CPU, .NCHW(1, 2048, 1, 1))
    for i in 0..<2048 {
      norm_weight[0, i, 0, 0] = 1
    }
    normIn.weight.copy(from: norm_weight)
    let clip_txt_mapper_weight = state_dict["clip_txt_mapper.weight"].float().cpu().numpy()
    clipTextMapper.weight.copy(from: try! Tensor<Float>(numpy: clip_txt_mapper_weight))
    let clip_txt_mapper_bias = state_dict["clip_txt_mapper.bias"].float().cpu().numpy()
    clipTextMapper.bias.copy(from: try! Tensor<Float>(numpy: clip_txt_mapper_bias))
    let clip_txt_pooled_mapper_weight = state_dict["clip_txt_pooled_mapper.weight"].float().cpu()
      .numpy()
    clipTextPooledMapper.weight.copy(from: try! Tensor<Float>(numpy: clip_txt_pooled_mapper_weight))
    let clip_txt_pooled_mapper_bias = state_dict["clip_txt_pooled_mapper.bias"].float().cpu()
      .numpy()
    clipTextPooledMapper.bias.copy(from: try! Tensor<Float>(numpy: clip_txt_pooled_mapper_bias))
    let clip_img_mapper_weight = state_dict["clip_img_mapper.weight"].float().cpu().numpy()
    clipImgMapper.weight.copy(from: try! Tensor<Float>(numpy: clip_img_mapper_weight))
    let clip_img_mapper_bias = state_dict["clip_img_mapper.bias"].float().cpu().numpy()
    clipImgMapper.bias.copy(from: try! Tensor<Float>(numpy: clip_img_mapper_bias))

    clipNorm.weight.copy(from: norm_weight)

    normOut.weight.copy(from: norm_weight)
    let clf_1_weight = state_dict["clf.1.weight"].float().cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: clf_1_weight))
    let clf_1_bias = state_dict["clf.1.bias"].float().cpu().numpy()
    convOut.bias.copy(from: try! Tensor<Float>(numpy: clf_1_bias))

  }
  return (Model([x, rEmbed, clipText, clipTextPooled, clipImg], [out]), reader)
}

func rEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let r = timesteps * Float(maxPeriod)
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half - 1)) * r
    let sinFreq = sin(freq)
    let cosFreq = cos(freq)
    for j in 0..<batchSize {
      embedding[j, i] = sinFreq
      embedding[j, i + half] = cosFreq
    }
  }
  return embedding
}

let graph = DynamicGraph()
graph.withNoGrad {
  let x = graph.variable(try! Tensor<Float>(numpy: x.float().cpu().numpy())).toGPU(0)
  let clipText = graph.variable(try! Tensor<Float>(numpy: clip_text.float().cpu().numpy())).toGPU(0)
  let clipTextPooled = graph.variable(
    try! Tensor<Float>(numpy: clip_text_pooled.float().cpu().numpy())
  ).toGPU(0)
  let clipImg = graph.variable(try! Tensor<Float>(numpy: clip_img.float().cpu().numpy())).toGPU(0)
  let rTimeEmbed = rEmbedding(timesteps: 0.9936, batchSize: 2, embeddingSize: 64, maxPeriod: 10_000)
  let rZeros = rEmbedding(timesteps: 0, batchSize: 2, embeddingSize: 64, maxPeriod: 10_000)
  var rEmbed = Tensor<Float>(.CPU, .NC(2, 192))
  rEmbed[0..<2, 0..<64] = rTimeEmbed
  rEmbed[0..<2, 64..<128] = rZeros
  rEmbed[0..<2, 128..<192] = rZeros
  let rEmbedVariable = graph.variable(rEmbed).toGPU(0)
  let (stageC, stageCReader) = StageC(batchSize: 2, height: 24, width: 24, t: 77 + 8)
  stageC.compile(inputs: x, rEmbedVariable, clipText, clipTextPooled, clipImg)
  stageCReader(state_dict)
  let out = stageC(inputs: x, rEmbedVariable, clipText, clipTextPooled, clipImg)[0].as(
    of: Float.self)
  debugPrint(out)
}
