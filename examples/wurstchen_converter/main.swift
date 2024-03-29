import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let inference_utils = Python.import("inference.utils")
let core_utils = Python.import("core.utils")
let train = Python.import("train")
let modules = Python.import("modules")

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
  generator_checkpoint_path: /home/liu/workspace/StableCascade/models/stage_c_fp16_fixed.safetensors
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
  stage_a_checkpoint_path: /home/liu/workspace/StableCascade/models/stage_a_ft_hq.safetensors
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
  "A painting of a woman sitting in a field of flowers. She is wearing a red dress and a crown on her head. The woman is surrounded by a variety of flowers, including daisies, roses, and tulips. The painting is done in a very detailed style, capturing the woman's features and the beauty of the flowers in the field."
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
/*
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

torch.set_autocast_gpu_dtype(torch.bfloat16)
torch.set_autocast_enabled(true)
torch.autocast_increment_nesting()
torch.set_autocast_cache_enabled(true)

torch.manual_seed(42)

let sampling_c = extras_c.gdf.sample(
  models_c.generator, conditions, stage_c_latent_shape, unconditions, device: device, cfg: extras_c.sampling_configs["cfg"], sampler: extras_c.sampling_configs["sampler"], shift: extras_c.sampling_configs["shift"], timesteps: extras_c.sampling_configs["timesteps"], t_start: extras_c.sampling_configs["t_start"]
)
var sampled_c: PythonObject? = nil
for _ in 0..<Int(extras_c.sampling_configs["timesteps"])! {
  sampled_c = sampling_c.__next__().tuple3.0
}
// print(sampled_c)

conditions_b["effnet"] = sampled_c!
unconditions_b["effnet"] = torch.zeros_like(sampled_c)

let sampling_b = extras_b.gdf.sample(
  models_b.generator, conditions_b, stage_b_latent_shape, unconditions_b, device: device, cfg: extras_b.sampling_configs["cfg"], sampler: extras_b.sampling_configs["sampler"], shift: extras_b.sampling_configs["shift"], timesteps: extras_b.sampling_configs["timesteps"], t_start: extras_b.sampling_configs["t_start"]
)
var sampled_b: PythonObject? = nil
for _ in 0..<Int(extras_b.sampling_configs["timesteps"])! {
  sampled_b = sampling_b.__next__().tuple3.0
}
// print(sampled_b)

let sampled = models_b.stage_a.decode(sampled_b).float()
// print(sampled)

inference_utils.save_images(sampled)
*/

let previewer = modules.previewer.Previewer().to(device)
previewer.load_state_dict(
  core_utils.load_or_fail("/home/liu/workspace/StableCascade/models/previewer.safetensors"))
previewer.float().eval().requires_grad_(false)
let previewer_state_dict = previewer.state_dict()
let x = torch.randn([2, 16, 24, 24]).cuda()
let previewerResult = previewer(x)
print(previewerResult)
print(previewer)
print(previewer_state_dict.keys())

let effnet = modules.effnet.EfficientNetEncoder().to(device)
effnet.load_state_dict(
  core_utils.load_or_fail("/home/liu/workspace/StableCascade/models/effnet_encoder.safetensors"))
effnet.float().eval().requires_grad_(false)
let effnet_state_dict = effnet.state_dict()

let clip_text = torch.randn([2, 77, 1280]).cuda()
let clip_text_pooled = torch.zeros([2, 1, 1280]).cuda()
let clip_img = torch.zeros([2, 1, 768]).cuda()
let r = 0.9936 * torch.ones([2]).cuda()
let result = models_c.generator(x, r, clip_text, clip_text_pooled, clip_img)
print(result)

// First, get weights from core_c.

let state_dict = models_c.generator.state_dict()

let img = torch.randn([2, 3, 768, 768]).cuda()
let imgResult = effnet(img)
// print(result)

// print(state_dict.keys())

/*
let x = torch.randn([2, 4, 256, 256]).cuda()
let effnet = torch.randn([2, 16, 24, 24]).cuda()
let clip_text_pooled = torch.zeros([2, 1, 1280]).cuda()
let r = 0.9936 * torch.ones([2]).cuda()
let result = models_b.generator(x, r, effnet, clip_text_pooled)
print(result)

let state_dict = models_b.generator.state_dict()

// print(state_dict.keys())
*/
/*
let x = torch.randn([1, 3, 1024, 1024]).cuda()
let (y, _, _, _) = models_b.stage_a.encode(x).tuple4
let result = models_b.stage_a.decode(y)
print(result)

let state_dict = models_b.stage_a.state_dict()

// print(state_dict.keys())
*/
func ResBlock(prefix: String, batchSize: Int, channels: Int, skip: Bool) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let depthwise = Convolution(
    groups: channels, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), name: "resblock")
  var out = depthwise(x)
  let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
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
    groups: 1, filters: channels * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    name: "resblock")
  out = convIn(out).GELU()
  let Gx = out.reduced(.norm2, axis: [2, 3])
  let Nx = Gx .* (1 / Gx.reduced(.mean, axis: [1])) + 1e-6
  let gamma = Parameter<Float>(
    .GPU(0), .NCHW(1, channels * 4, 1, 1), initBound: 1, name: "resblock")
  let beta = Parameter<Float>(.GPU(0), .NCHW(1, channels * 4, 1, 1), initBound: 1, name: "resblock")
  out = gamma .* (out .* Nx) + beta + out
  let convOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out) + x
  let reader: (PythonObject) -> Void = { state_dict in
    let depthwise_weight = state_dict["\(prefix).depthwise.weight"].float().cpu().numpy()
    depthwise.weight.copy(from: try! Tensor<Float>(numpy: depthwise_weight))
    let depthwise_bias = state_dict["\(prefix).depthwise.bias"].float().cpu().numpy()
    depthwise.bias.copy(from: try! Tensor<Float>(numpy: depthwise_bias))
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

func TimestepBlock(
  prefix: String, batchSize: Int, timeEmbedSize: Int, channels: Int, tConds: [String]
) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rEmbed = Input()
  let mapper = Dense(count: channels * 2, name: "timestepblock")
  var gate = mapper(
    rEmbed.reshaped(
      [batchSize, timeEmbedSize], offset: [0, 0], strides: [timeEmbedSize * (tConds.count + 1), 1]))
  var otherMappers = [Model]()
  for i in 0..<tConds.count {
    let otherMapper = Dense(count: channels * 2, name: "timestepblock")
    gate =
      gate
      + otherMapper(
        rEmbed.reshaped(
          [batchSize, timeEmbedSize], offset: [0, timeEmbedSize * (i + 1)],
          strides: [timeEmbedSize * (tConds.count + 1), 1]))
    otherMappers.append(otherMapper)
  }
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
    for (otherMapper, tCond) in zip(otherMappers, tConds) {
      let mapper_t_cond_weight = state_dict["\(prefix).mapper_\(tCond).weight"].float().cpu()
        .numpy()
      otherMapper.weight.copy(from: try! Tensor<Float>(numpy: mapper_t_cond_weight))
      let mapper_t_cond_bias = state_dict["\(prefix).mapper_\(tCond).bias"].float().cpu().numpy()
      otherMapper.bias.copy(from: try! Tensor<Float>(numpy: mapper_t_cond_bias))
    }
  }
  return (Model([x, rEmbed], [out]), reader)
}

func MultiHeadAttention(prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let key = Input()
  let value = Input()
  let tokeys = Dense(count: k * h, name: "\(prefix).keys")
  let toqueries = Dense(count: k * h, name: "queries")
  let tovalues = Dense(count: k * h, name: "\(prefix).values")
  var keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2)
  keys = Functional.concat(axis: 2, keys, key)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2)
  var values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2)
  values = Functional.concat(axis: 2, values, value)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw + t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, hw + t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h, name: "unifyheads")
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
  return (Model([x, key, value], [out]), reader)
}

func AttnBlock(
  prefix: String, batchSize: Int, channels: Int, nHead: Int, height: Int, width: Int, t: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let key = Input()
  let value = Input()
  let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  var out = norm(x).reshaped([batchSize, channels, height * width]).transposed(1, 2)
  let k = channels / nHead
  let (multiHeadAttention, multiHeadAttentionReader) = MultiHeadAttention(
    prefix: prefix, k: k, h: nHead, b: batchSize, hw: height * width, t: t)
  out =
    x
    + multiHeadAttention(out, key, value).transposed(1, 2).reshaped([
      batchSize, channels, height, width,
    ])
  return (Model([x, key, value], [out]), multiHeadAttentionReader)
}

func AttnBlockFixed(prefix: String, batchSize: Int, channels: Int, nHead: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let kv = Input()
  let kvMapper = Dense(count: channels, name: "kv_mapper")
  let kvOut = kvMapper(kv.swish())
  let tokeys = Dense(count: channels, name: "\(prefix).keys")
  let tovalues = Dense(count: channels, name: "\(prefix).values")
  let k = channels / nHead
  let keys = tokeys(kvOut).reshaped([batchSize, t, nHead, k]).transposed(1, 2)
  let values = tovalues(kvOut).reshaped([batchSize, t, nHead, k]).transposed(1, 2)
  let reader: (PythonObject) -> Void = { state_dict in
    let kv_mapper_1_weight = state_dict["\(prefix).kv_mapper.1.weight"].float().cpu().numpy()
    kvMapper.weight.copy(from: try! Tensor<Float>(numpy: kv_mapper_1_weight))
    let kv_mapper_1_bias = state_dict["\(prefix).kv_mapper.1.bias"].float().cpu().numpy()
    kvMapper.bias.copy(from: try! Tensor<Float>(numpy: kv_mapper_1_bias))
    let in_proj_weight = state_dict["\(prefix).attention.attn.in_proj_weight"].float().cpu().numpy()
    let in_proj_bias = state_dict["\(prefix).attention.attn.in_proj_bias"].float().cpu().numpy()
    tokeys.weight.copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[channels..<(2 * channels), ...]))
    tokeys.bias.copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[channels..<(2 * channels)]))
    tovalues.weight.copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(2 * channels)..., ...]))
    tovalues.bias.copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(2 * channels)...]))
  }
  return (Model([kv], [keys, values]), reader)
}

func StageCFixed(batchSize: Int, t: Int) -> (Model, (PythonObject) -> Void) {
  let clipText = Input()
  let clipTextPooled = Input()
  let clipImg = Input()
  let clipTextMapper = Dense(count: 2048, name: "clip_text_mapper")
  let clipTextMapped = clipTextMapper(clipText)
  let clipTextPooledMapper = Dense(count: 2048 * 4, name: "clip_text_pool_mapper")
  let clipTextPooledMapped = clipTextPooledMapper(clipTextPooled).reshaped([batchSize, 4, 2048])
  let clipImgMapper = Dense(count: 2048 * 4, name: "clip_image_mapper")
  let clipImgMapped = clipImgMapper(clipImg).reshaped([batchSize, 4, 2048])
  let clipNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let clip = clipNorm(
    Functional.concat(axis: 1, clipTextMapped, clipTextPooledMapped, clipImgMapped))
  let blocks: [[Int]] = [[8, 24], [24, 8]]
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  for i in 0..<2 {
    for j in 0..<blocks[0][i] {
      let (attnBlockFixed, attnBlockFixedReader) = AttnBlockFixed(
        prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        t: t)
      readers.append(attnBlockFixedReader)
      let out = attnBlockFixed(clip)
      outs.append(out)
    }
  }
  for i in 0..<2 {
    for j in 0..<blocks[1][i] {
      let (attnBlockFixed, attnBlockFixedReader) = AttnBlockFixed(
        prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        t: t)
      readers.append(attnBlockFixedReader)
      let out = attnBlockFixed(clip)
      outs.append(out)
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
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
  }
  return (Model([clipText, clipTextPooled, clipImg], outs), reader)
}

func StageC(batchSize: Int, height: Int, width: Int, t: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rEmbed = Input()
  let conv2d = Convolution(
    groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = conv2d(x)
  let normIn = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normIn(out)

  let blocks: [[Int]] = [[8, 24], [24, 8]]
  var readers: [(PythonObject) -> Void] = []
  var levelOutputs = [Model.IO]()
  var kvs = [Input]()
  for i in 0..<2 {
    if i > 0 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
      out = norm(out)
      let downscaler = Convolution(
        groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
      out = downscaler(out)
      readers.append { state_dict in
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
        channels: 2048, tConds: ["sca", "crp"])
      readers.append(timestepBlockReader)
      out = timestepBlock(out, rEmbed)
      let (attnBlock, attnBlockReader) = AttnBlock(
        prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        height: height, width: width, t: t)
      readers.append(attnBlockReader)
      let key = Input()
      let value = Input()
      out = attnBlock(out, key, value)
      kvs.append(key)
      kvs.append(value)
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
        channels: 2048, tConds: ["sca", "crp"])
      readers.append(timestepBlockReader)
      out = timestepBlock(out, rEmbed)
      let (attnBlock, attnBlockReader) = AttnBlock(
        prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        height: height, width: width, t: t)
      readers.append(attnBlockReader)
      let key = Input()
      let value = Input()
      out = attnBlock(out, key, value)
      kvs.append(key)
      kvs.append(value)
    }
    if i < 2 - 1 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
      out = norm(out)
      let upscaler = Convolution(
        groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
      out = upscaler(out)
      readers.append { state_dict in
        let blocks_1_weight = state_dict["up_upscalers.\(i).1.blocks.1.weight"].float().cpu()
          .numpy()
        upscaler.weight.copy(from: try! Tensor<Float>(numpy: blocks_1_weight))
        let blocks_1_bias = state_dict["up_upscalers.\(i).1.blocks.1.bias"].float().cpu().numpy()
        upscaler.bias.copy(from: try! Tensor<Float>(numpy: blocks_1_bias))
      }
      skip = levelOutputs.removeLast()
    }
  }

  let normOut = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
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

    let clf_1_weight = state_dict["clf.1.weight"].float().cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: clf_1_weight))
    let clf_1_bias = state_dict["clf.1.bias"].float().cpu().numpy()
    convOut.bias.copy(from: try! Tensor<Float>(numpy: clf_1_bias))

  }
  return (Model([x, rEmbed] + kvs, [out]), reader)
}

func SpatialMapper(prefix: String, cHidden: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: cHidden * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = convIn(x)
  let convOut = Convolution(
    groups: 1, filters: cHidden, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out.GELU())
  let normOut = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let effnet_mapper_0_weight = state_dict["\(prefix).0.weight"].float().cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: effnet_mapper_0_weight))
    let effnet_mapper_0_bias = state_dict["\(prefix).0.bias"].float().cpu().numpy()
    convIn.bias.copy(from: try! Tensor<Float>(numpy: effnet_mapper_0_bias))
    let effnet_mapper_2_weight = state_dict["\(prefix).2.weight"].float().cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: effnet_mapper_2_weight))
    let effnet_mapper_2_bias = state_dict["\(prefix).2.bias"].float().cpu().numpy()
    convOut.bias.copy(from: try! Tensor<Float>(numpy: effnet_mapper_2_bias))
  }
  return (Model([x], [out]), reader)
}

func StageBFixed(batchSize: Int, height: Int, width: Int, effnetHeight: Int, effnetWidth: Int) -> (
  Model, (PythonObject) -> Void
) {
  let effnet = Input()
  let pixels = Input()
  let clip = Input()
  let cHidden: [Int] = [320, 640, 1280, 1280]
  let (effnetMapper, effnetMapperReader) = SpatialMapper(
    prefix: "effnet_mapper", cHidden: cHidden[0])
  var out = effnetMapper(
    Upsample(
      .bilinear, widthScale: Float(width / 2) / Float(effnetWidth),
      heightScale: Float(height / 2) / Float(effnetHeight), alignCorners: true)(effnet))
  let (pixelsMapper, pixelsMapperReader) = SpatialMapper(
    prefix: "pixels_mapper", cHidden: cHidden[0])
  out =
    out
    + Upsample(
      .bilinear, widthScale: Float(width / 2) / Float(8), heightScale: Float(height / 2) / Float(8),
      alignCorners: true)(pixelsMapper(pixels))
  var outs = [out]
  let clipMapper = Dense(count: 1280 * 4)
  let clipMapped = clipMapper(clip).reshaped([batchSize, 4, 1280])
  let clipNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let clipNormed = clipNorm(clipMapped)
  let blocks: [[Int]] = [[2, 6, 28, 6], [6, 28, 6, 2]]
  var readers: [(PythonObject) -> Void] = []
  let attentions: [[Bool]] = [[false, false, true, true], [true, true, false, false]]
  for i in 0..<4 {
    let attention = attentions[0][i]
    for j in 0..<blocks[0][i] {
      if attention {
        let (attnBlockFixed, attnBlockFixedReader) = AttnBlockFixed(
          prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[i],
          nHead: 20, t: 4)
        readers.append(attnBlockFixedReader)
        let out = attnBlockFixed(clipNormed)
        outs.append(out)
      }
    }
  }
  for i in 0..<4 {
    let attention = attentions[1][i]
    for j in 0..<blocks[1][i] {
      if attention {
        let (attnBlockFixed, attnBlockFixedReader) = AttnBlockFixed(
          prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[3 - i],
          nHead: 20, t: 4)
        readers.append(attnBlockFixedReader)
        let out = attnBlockFixed(clipNormed)
        outs.append(out)
      }
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    effnetMapperReader(state_dict)
    let clip_mapper_weight = state_dict["clip_mapper.weight"].float().cpu().numpy()
    clipMapper.weight.copy(from: try! Tensor<Float>(numpy: clip_mapper_weight))
    let clip_mapper_bias = state_dict["clip_mapper.bias"].float().cpu().numpy()
    clipMapper.bias.copy(from: try! Tensor<Float>(numpy: clip_mapper_bias))
    pixelsMapperReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([effnet, pixels, clip], outs), reader)
}

func StageB(batchSize: Int, cIn: Int, height: Int, width: Int, effnetHeight: Int, effnetWidth: Int)
  -> (Model, (PythonObject) -> Void)
{
  let x = Input()
  let rEmbed = Input()
  let effnetAndPixels = Input()
  let cHidden: [Int] = [320, 640, 1280, 1280]
  let conv2d = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = conv2d(x)
  let normIn = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normIn(out) + effnetAndPixels
  let blocks: [[Int]] = [[2, 6, 28, 6], [6, 28, 6, 2]]
  var readers: [(PythonObject) -> Void] = []
  let attentions: [[Bool]] = [[false, false, true, true], [true, true, false, false]]
  var levelOutputs = [Model.IO]()
  var height = height / 2
  var width = width / 2
  var kvs = [Input]()
  for i in 0..<4 {
    if i > 0 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
      out = norm(out)
      let downscaler = Convolution(
        groups: 1, filters: cHidden[i], filterSize: [2, 2], hint: Hint(stride: [2, 2]))
      out = downscaler(out)
      readers.append { state_dict in
        let downscalers_1_weight = state_dict["down_downscalers.\(i).1.weight"].float().cpu()
          .numpy()
        downscaler.weight.copy(from: try! Tensor<Float>(numpy: downscalers_1_weight))
        let downscalers_1_bias = state_dict["down_downscalers.\(i).1.bias"].float().cpu()
          .numpy()
        downscaler.bias.copy(from: try! Tensor<Float>(numpy: downscalers_1_bias))
      }
      height = height / 2
      width = width / 2
    }
    let attention = attentions[0][i]
    for j in 0..<blocks[0][i] {
      let (resBlock, resBlockReader) = ResBlock(
        prefix: "down_blocks.\(i).\(j * (attention ? 3 : 2))", batchSize: batchSize,
        channels: cHidden[i], skip: false)
      readers.append(resBlockReader)
      out = resBlock(out)
      let (timestepBlock, timestepBlockReader) = TimestepBlock(
        prefix: "down_blocks.\(i).\(j * (attention ? 3 : 2) + 1)", batchSize: batchSize,
        timeEmbedSize: 64,
        channels: cHidden[i], tConds: ["sca"])
      readers.append(timestepBlockReader)
      out = timestepBlock(out, rEmbed)
      if attention {
        let (attnBlock, attnBlockReader) = AttnBlock(
          prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[i],
          nHead: 20, height: height, width: width, t: 4)
        readers.append(attnBlockReader)
        let key = Input()
        let value = Input()
        out = attnBlock(out, key, value)
        kvs.append(key)
        kvs.append(value)
      }
    }
    if i < 4 - 1 {
      levelOutputs.append(out)
    }
  }
  var skip: Model.IO? = nil
  let blockRepeat: [Int] = [3, 3, 2, 2]
  for i in 0..<4 {
    let cSkip = skip
    skip = nil
    let attention = attentions[1][i]
    var resBlocks = [Model]()
    var timestepBlocks = [Model]()
    var attnBlocks = [Model]()
    var keyAndValue = [(Input, Input)]()
    for j in 0..<blocks[1][i] {
      let (resBlock, resBlockReader) = ResBlock(
        prefix: "up_blocks.\(i).\(j * (attention ? 3 : 2))", batchSize: batchSize,
        channels: cHidden[3 - i], skip: j == 0 && cSkip != nil)
      readers.append(resBlockReader)
      resBlocks.append(resBlock)
      let (timestepBlock, timestepBlockReader) = TimestepBlock(
        prefix: "up_blocks.\(i).\(j * (attention ? 3 : 2) + 1)", batchSize: batchSize,
        timeEmbedSize: 64,
        channels: cHidden[3 - i], tConds: ["sca"])
      readers.append(timestepBlockReader)
      timestepBlocks.append(timestepBlock)
      if attention {
        let (attnBlock, attnBlockReader) = AttnBlock(
          prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[3 - i],
          nHead: 20,
          height: height, width: width, t: 4)
        readers.append(attnBlockReader)
        attnBlocks.append(attnBlock)
        keyAndValue.append((Input(), Input()))
      }
    }
    kvs.append(contentsOf: keyAndValue.flatMap { [$0.0, $0.1] })
    for j in 0..<blockRepeat[i] {
      for k in 0..<blocks[1][i] {
        if k == 0, let cSkip = cSkip {
          out = resBlocks[k](out, cSkip)
        } else {
          out = resBlocks[k](out)
        }
        out = timestepBlocks[k](out, rEmbed)
        if attention {
          out = attnBlocks[k](out, keyAndValue[k].0, keyAndValue[k].1)
        }
      }
      // repmap.
      if j < blockRepeat[i] - 1 {
        let repmap = Convolution(
          groups: 1, filters: cHidden[3 - i], filterSize: [1, 1], hint: Hint(stride: [1, 1]))
        out = repmap(out)
        readers.append { state_dict in
          let up_repeat_mappers_weight = state_dict["up_repeat_mappers.\(i).\(j).weight"].float()
            .cpu()
            .numpy()
          repmap.weight.copy(from: try! Tensor<Float>(numpy: up_repeat_mappers_weight))
          let up_repeat_mappers_bias = state_dict["up_repeat_mappers.\(i).\(j).bias"].float().cpu()
            .numpy()
          repmap.bias.copy(from: try! Tensor<Float>(numpy: up_repeat_mappers_bias))
        }
      }
    }
    if i < 4 - 1 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
      out = norm(out)
      let upscaler = ConvolutionTranspose(
        groups: 1, filters: cHidden[2 - i], filterSize: [2, 2], hint: Hint(stride: [2, 2]))
      out = upscaler(out)
      readers.append { state_dict in
        let upscalers_1_weight = state_dict["up_upscalers.\(i).1.weight"].float().cpu()
          .numpy()
        upscaler.weight.copy(from: try! Tensor<Float>(numpy: upscalers_1_weight))
        let upscalers_1_bias = state_dict["up_upscalers.\(i).1.bias"].float().cpu().numpy()
        upscaler.bias.copy(from: try! Tensor<Float>(numpy: upscalers_1_bias))
      }
      skip = levelOutputs.removeLast()
      height = height * 2
      width = width * 2
    }
  }
  let normOut = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normOut(out)
  let convOut = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out).reshaped([batchSize, 4, 2, 2, height, width]).transposed(3, 4).transposed(4, 5)
    .transposed(2, 3).reshaped([batchSize, 4, height * 2, width * 2])  // This is the same as .permuted(0, 1, 4, 2, 5, 3).

  let reader: (PythonObject) -> Void = { state_dict in
    let embedding_1_weight = state_dict["embedding.1.weight"].float().cpu().numpy()
    conv2d.weight.copy(from: try! Tensor<Float>(numpy: embedding_1_weight))
    let embedding_1_bias = state_dict["embedding.1.bias"].float().cpu().numpy()
    conv2d.bias.copy(from: try! Tensor<Float>(numpy: embedding_1_bias))
    for reader in readers {
      reader(state_dict)
    }

    let clf_1_weight = state_dict["clf.1.weight"].float().cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: clf_1_weight))
    let clf_1_bias = state_dict["clf.1.bias"].float().cpu().numpy()
    convOut.bias.copy(from: try! Tensor<Float>(numpy: clf_1_bias))
  }
  return (Model([x, rEmbed, effnetAndPixels] + kvs, [out]), reader)
}

func StageAResBlock(prefix: String, channels: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let gammas = Parameter<Float>(.GPU(0), .NCHW(1, 1, 1, 6), initBound: 1)
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  var out =
    norm1(x) .* (1 + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 0], strides: [6, 6, 6, 1]))
    + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 1], strides: [6, 6, 6, 1])
  let depthwise = Convolution(
    groups: channels, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1]))
  out = x + depthwise(out.padded(.replication, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]))
    .* gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 2], strides: [6, 6, 6, 1])
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let xTemp =
    norm2(out)
    .* (1 + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 3], strides: [6, 6, 6, 1]))
    + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 4], strides: [6, 6, 6, 1])
  let convIn = Convolution(
    groups: 1, filters: channels * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let convOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = out + convOut(convIn(xTemp).GELU())
    .* gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 5], strides: [6, 6, 6, 1])
  let reader: (PythonObject) -> Void = { state_dict in
    let depthwise_1_weight = state_dict["\(prefix).depthwise.1.weight"].float().cpu().numpy()
    depthwise.weight.copy(from: try! Tensor<Float>(numpy: depthwise_1_weight))
    let depthwise_1_bias = state_dict["\(prefix).depthwise.1.bias"].float().cpu().numpy()
    depthwise.bias.copy(from: try! Tensor<Float>(numpy: depthwise_1_bias))
    let channelwise_0_weight = state_dict["\(prefix).channelwise.0.weight"].float().cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: channelwise_0_weight))
    let channelwise_0_bias = state_dict["\(prefix).channelwise.0.bias"].float().cpu().numpy()
    convIn.bias.copy(from: try! Tensor<Float>(numpy: channelwise_0_bias))
    let channelwise_2_weight = state_dict["\(prefix).channelwise.2.weight"].float().cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: channelwise_2_weight))
    let channelwise_2_bias = state_dict["\(prefix).channelwise.2.bias"].float().cpu().numpy()
    convOut.bias.copy(from: try! Tensor<Float>(numpy: channelwise_2_bias))
    let gammas_weight = state_dict["\(prefix).gammas"].float().cpu().numpy()
    gammas.weight.copy(from: try! Tensor<Float>(numpy: gammas_weight))
  }
  return (Model([x], [out]), reader)
}

func StageAEncoder(batchSize: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let cHidden = [192, 384]
  let convIn = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = convIn(x)
  var readers = [(PythonObject) -> Void]()
  var j = 0
  for i in 0..<cHidden.count {
    if i > 0 {
      let conv2d = Convolution(
        groups: 1, filters: cHidden[i], filterSize: [4, 4],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
      let layer = j
      readers.append { state_dict in
        let down_blocks_weight = state_dict["down_blocks.\(layer).weight"].float().cpu().numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: down_blocks_weight))
        let down_blocks_bias = state_dict["down_blocks.\(layer).bias"].float().cpu().numpy()
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: down_blocks_bias))
      }
      j += 1
    }
    let (resBlock, resBlockReader) = StageAResBlock(
      prefix: "down_blocks.\(j)", channels: cHidden[i])
    out = resBlock(out)
    readers.append(resBlockReader)
    j += 1
  }
  let conv2d = Convolution(groups: 1, filters: 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = conv2d(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let in_block_1_weight = state_dict["in_block.1.weight"].float().cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: in_block_1_weight))
    let in_block_1_bias = state_dict["in_block.1.bias"].float().cpu().numpy()
    convIn.bias.copy(from: try! Tensor<Float>(numpy: in_block_1_bias))
    for reader in readers {
      reader(state_dict)
    }
    let down_blocks_3_0_weight = state_dict["down_blocks.3.0.weight"].float().cpu()
    let down_blocks_3_1_weight = state_dict["down_blocks.3.1.weight"].float().cpu()
    let down_blocks_3_1_running_mean = state_dict["down_blocks.3.1.running_mean"].float().cpu()
    let down_blocks_3_1_running_var = state_dict["down_blocks.3.1.running_var"].float().cpu()
    let down_blocks_3_1_bias = state_dict["down_blocks.3.1.bias"].float().cpu()
    let w_conv = down_blocks_3_0_weight.view(4, -1)
    let w_bn = torch.diag(
      down_blocks_3_1_weight.div(torch.sqrt(1e-5 + down_blocks_3_1_running_var)))
    let fused_weight = torch.mm(w_bn, w_conv).numpy()
    conv2d.weight.copy(from: try! Tensor<Float>(numpy: fused_weight))
    let b_bn =
      down_blocks_3_1_bias
      - down_blocks_3_1_weight.mul(down_blocks_3_1_running_mean).div(
        torch.sqrt(down_blocks_3_1_running_var + 1e-5))
    conv2d.bias.copy(from: try! Tensor<Float>(numpy: b_bn.numpy()))
  }
  return (Model([x], [out]), reader)
}

func StageADecoder(batchSize: Int, height: Int, width: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let cHidden = [384, 192]
  let convIn = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = convIn(x)
  var readers = [(PythonObject) -> Void]()
  var j = 1
  for i in 0..<cHidden.count {
    for _ in 0..<(i == 0 ? 12 : 1) {
      let (resBlock, resBlockReader) = StageAResBlock(
        prefix: "up_blocks.\(j)", channels: cHidden[i])
      out = resBlock(out)
      readers.append(resBlockReader)
      j += 1
    }
    if i < cHidden.count - 1 {
      let conv2d = ConvolutionTranspose(
        groups: 1, filters: cHidden[i + 1], filterSize: [4, 4],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
      let layer = j
      readers.append { state_dict in
        let up_blocks_weight = state_dict["up_blocks.\(layer).weight"].float().cpu().numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: up_blocks_weight))
        let up_blocks_bias = state_dict["up_blocks.\(layer).bias"].float().cpu().numpy()
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: up_blocks_bias))
      }
      j += 1
    }
  }
  let convOut = Convolution(
    groups: 1, filters: 12, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out).reshaped([batchSize, 3, 2, 2, height, width]).transposed(3, 4).transposed(4, 5)
    .transposed(2, 3).reshaped([batchSize, 3, height * 2, width * 2])  // This is the same as .permuted(0, 1, 4, 2, 5, 3).
  let reader: (PythonObject) -> Void = { state_dict in
    let up_blocks_0_0_weight = state_dict["up_blocks.0.0.weight"].float().cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: up_blocks_0_0_weight))
    let up_blocks_0_0_bias = state_dict["up_blocks.0.0.bias"].float().cpu().numpy()
    convIn.bias.copy(from: try! Tensor<Float>(numpy: up_blocks_0_0_bias))
    for reader in readers {
      reader(state_dict)
    }
    let out_block_0_weight = state_dict["out_block.0.weight"].float().cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: out_block_0_weight))
    let out_block_0_bias = state_dict["out_block.0.bias"].float().cpu().numpy()
    convOut.bias.copy(from: try! Tensor<Float>(numpy: out_block_0_bias))
  }
  return (Model([x], [out]), reader)
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

func FusedMBConv(
  prefix: String, outChannels: Int, stride: Int, filterSize: Int, skip: Bool,
  expandChannels: Int? = nil
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  var out: Model.IO = x
  let expandConv: Model?
  let convOut: Model
  if let expandChannels = expandChannels {
    let conv = Convolution(
      groups: 1, filters: expandChannels, filterSize: [filterSize, filterSize],
      hint: Hint(
        stride: [stride, stride],
        border: Hint.Border(
          begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
          end: [(filterSize - 1) / 2, (filterSize - 1) / 2])))
    out = conv(out).swish()
    expandConv = conv
    convOut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = convOut(out)
  } else {
    expandConv = nil
    convOut = Convolution(
      groups: 1, filters: outChannels, filterSize: [filterSize, filterSize],
      hint: Hint(
        stride: [stride, stride],
        border: Hint.Border(
          begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
          end: [(filterSize - 1) / 2, (filterSize - 1) / 2])))
    out = convOut(out).swish()
  }
  if skip {
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
    if let expandConv = expandConv, let expandChannels = expandChannels {
      let block_0_0_weight = state_dict["\(prefix).block.0.0.weight"].float().cpu()
      let block_0_1_weight = state_dict["\(prefix).block.0.1.weight"].float().cpu()
      let block_0_1_running_mean = state_dict["\(prefix).block.0.1.running_mean"].float().cpu()
      let block_0_1_running_var = state_dict["\(prefix).block.0.1.running_var"].float().cpu()
      let block_0_1_bias = state_dict["\(prefix).block.0.1.bias"].float().cpu()
      let w_conv_0 = block_0_0_weight.view(expandChannels, -1)
      let w_bn_0 = torch.diag(
        block_0_1_weight.div(torch.sqrt(1e-3 + block_0_1_running_var)))
      let fused_weight_0 = torch.mm(w_bn_0, w_conv_0).numpy()
      expandConv.weight.copy(from: try! Tensor<Float>(numpy: fused_weight_0))
      let b_bn_0 =
        block_0_1_bias
        - block_0_1_weight.mul(block_0_1_running_mean).div(
          torch.sqrt(block_0_1_running_var + 1e-3))
      expandConv.bias.copy(from: try! Tensor<Float>(numpy: b_bn_0.numpy()))

      let block_1_0_weight = state_dict["\(prefix).block.1.0.weight"].float().cpu()
      let block_1_1_weight = state_dict["\(prefix).block.1.1.weight"].float().cpu()
      let block_1_1_running_mean = state_dict["\(prefix).block.1.1.running_mean"].float().cpu()
      let block_1_1_running_var = state_dict["\(prefix).block.1.1.running_var"].float().cpu()
      let block_1_1_bias = state_dict["\(prefix).block.1.1.bias"].float().cpu()
      let w_conv_1 = block_1_0_weight.view(outChannels, -1)
      let w_bn_1 = torch.diag(
        block_1_1_weight.div(torch.sqrt(1e-3 + block_1_1_running_var)))
      let fused_weight_1 = torch.mm(w_bn_1, w_conv_1).numpy()
      convOut.weight.copy(from: try! Tensor<Float>(numpy: fused_weight_1))
      let b_bn_1 =
        block_1_1_bias
        - block_1_1_weight.mul(block_1_1_running_mean).div(
          torch.sqrt(block_1_1_running_var + 1e-3))
      convOut.bias.copy(from: try! Tensor<Float>(numpy: b_bn_1.numpy()))

    } else {
      let block_0_0_weight = state_dict["\(prefix).block.0.0.weight"].float().cpu()
      let block_0_1_weight = state_dict["\(prefix).block.0.1.weight"].float().cpu()
      let block_0_1_running_mean = state_dict["\(prefix).block.0.1.running_mean"].float().cpu()
      let block_0_1_running_var = state_dict["\(prefix).block.0.1.running_var"].float().cpu()
      let block_0_1_bias = state_dict["\(prefix).block.0.1.bias"].float().cpu()
      let w_conv = block_0_0_weight.view(outChannels, -1)
      let w_bn = torch.diag(
        block_0_1_weight.div(torch.sqrt(1e-3 + block_0_1_running_var)))
      let fused_weight = torch.mm(w_bn, w_conv).numpy()
      convOut.weight.copy(from: try! Tensor<Float>(numpy: fused_weight))
      let b_bn =
        block_0_1_bias
        - block_0_1_weight.mul(block_0_1_running_mean).div(
          torch.sqrt(block_0_1_running_var + 1e-3))
      convOut.bias.copy(from: try! Tensor<Float>(numpy: b_bn.numpy()))
    }
  }
  return (Model([x], [out]), reader)
}

func MBConv(
  prefix: String, stride: Int, filterSize: Int, inChannels: Int, expandChannels: Int,
  outChannels: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  var out: Model.IO = x
  let expandConv: Model?
  if expandChannels != inChannels {
    let conv = Convolution(
      groups: 1, filters: expandChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = conv(out).swish()
    expandConv = conv
  } else {
    expandConv = nil
  }

  let depthwise = Convolution(
    groups: expandChannels, filters: expandChannels, filterSize: [filterSize, filterSize],
    hint: Hint(
      stride: [stride, stride],
      border: Hint.Border(
        begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
        end: [(filterSize - 1) / 2, (filterSize - 1) / 2])))
  out = depthwise(out).swish()

  // Squeeze and Excitation
  var scale = out.reduced(.mean, axis: [2, 3])
  let fc1 = Convolution(
    groups: 1, filters: inChannels / 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  scale = fc1(scale).swish()
  let fc2 = Convolution(
    groups: 1, filters: expandChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  scale = fc2(scale).sigmoid()
  out = scale .* out

  let convOut = Convolution(
    groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out)

  if inChannels == outChannels && stride == 1 {
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let blockStart: Int
    if let expandConv = expandConv {
      let block_0_0_weight = state_dict["\(prefix).block.0.0.weight"].float().cpu()
      let block_0_1_weight = state_dict["\(prefix).block.0.1.weight"].float().cpu()
      let block_0_1_running_mean = state_dict["\(prefix).block.0.1.running_mean"].float().cpu()
      let block_0_1_running_var = state_dict["\(prefix).block.0.1.running_var"].float().cpu()
      let block_0_1_bias = state_dict["\(prefix).block.0.1.bias"].float().cpu()
      let w_conv_0 = block_0_0_weight.view(expandChannels, -1)
      let w_bn_0 = torch.diag(
        block_0_1_weight.div(torch.sqrt(1e-3 + block_0_1_running_var)))
      let fused_weight_0 = torch.mm(w_bn_0, w_conv_0).numpy()
      expandConv.weight.copy(from: try! Tensor<Float>(numpy: fused_weight_0))
      let b_bn_0 =
        block_0_1_bias
        - block_0_1_weight.mul(block_0_1_running_mean).div(
          torch.sqrt(block_0_1_running_var + 1e-3))
      expandConv.bias.copy(from: try! Tensor<Float>(numpy: b_bn_0.numpy()))
      blockStart = 1
    } else {
      blockStart = 0
    }

    let block_1_0_weight = state_dict["\(prefix).block.\(blockStart).0.weight"].float().cpu()
    let block_1_1_weight = state_dict["\(prefix).block.\(blockStart).1.weight"].float().cpu()
    let block_1_1_running_mean = state_dict["\(prefix).block.\(blockStart).1.running_mean"].float()
      .cpu()
    let block_1_1_running_var = state_dict["\(prefix).block.\(blockStart).1.running_var"].float()
      .cpu()
    let block_1_1_bias = state_dict["\(prefix).block.\(blockStart).1.bias"].float().cpu()
    let w_conv_1 = block_1_0_weight.view(expandChannels, -1)
    let w_bn_1 = torch.diag(
      block_1_1_weight.div(torch.sqrt(1e-3 + block_1_1_running_var)))
    let fused_weight_1 = torch.mm(w_bn_1, w_conv_1).numpy()
    depthwise.weight.copy(from: try! Tensor<Float>(numpy: fused_weight_1))
    let b_bn_1 =
      block_1_1_bias
      - block_1_1_weight.mul(block_1_1_running_mean).div(
        torch.sqrt(block_1_1_running_var + 1e-3))
    depthwise.bias.copy(from: try! Tensor<Float>(numpy: b_bn_1.numpy()))

    let block_2_fc1_weight = state_dict["\(prefix).block.\(blockStart + 1).fc1.weight"].float()
      .cpu()
    fc1.weight.copy(from: try! Tensor<Float>(numpy: block_2_fc1_weight.numpy()))
    let block_2_fc1_bias = state_dict["\(prefix).block.\(blockStart + 1).fc1.bias"].float().cpu()
    fc1.bias.copy(from: try! Tensor<Float>(numpy: block_2_fc1_bias.numpy()))
    let block_2_fc2_weight = state_dict["\(prefix).block.\(blockStart + 1).fc2.weight"].float()
      .cpu()
    fc2.weight.copy(from: try! Tensor<Float>(numpy: block_2_fc2_weight.numpy()))
    let block_2_fc2_bias = state_dict["\(prefix).block.\(blockStart + 1).fc2.bias"].float().cpu()
    fc2.bias.copy(from: try! Tensor<Float>(numpy: block_2_fc2_bias.numpy()))

    let block_3_0_weight = state_dict["\(prefix).block.\(blockStart + 2).0.weight"].float().cpu()
    let block_3_1_weight = state_dict["\(prefix).block.\(blockStart + 2).1.weight"].float().cpu()
    let block_3_1_running_mean = state_dict["\(prefix).block.\(blockStart + 2).1.running_mean"]
      .float().cpu()
    let block_3_1_running_var = state_dict["\(prefix).block.\(blockStart + 2).1.running_var"]
      .float().cpu()
    let block_3_1_bias = state_dict["\(prefix).block.\(blockStart + 2).1.bias"].float().cpu()
    let w_conv_3 = block_3_0_weight.view(outChannels, -1)
    let w_bn_3 = torch.diag(
      block_3_1_weight.div(torch.sqrt(1e-3 + block_3_1_running_var)))
    let fused_weight_3 = torch.mm(w_bn_3, w_conv_3).numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: fused_weight_3))
    let b_bn_3 =
      block_3_1_bias
      - block_3_1_weight.mul(block_3_1_running_mean).div(
        torch.sqrt(block_3_1_running_var + 1e-3))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: b_bn_3.numpy()))
  }
  return (Model([x], [out]), reader)
}

func EfficientNetEncoder() -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let conv = Convolution(
    groups: 1, filters: 24, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv(x).swish()
  var readers = [(PythonObject) -> Void]()
  // 1.
  let (backbone_1_0, backbone_1_0_reader) = FusedMBConv(
    prefix: "backbone.1.0", outChannels: 24, stride: 1, filterSize: 3, skip: true)
  out = backbone_1_0(out)
  readers.append(backbone_1_0_reader)
  let (backbone_1_1, backbone_1_1_reader) = FusedMBConv(
    prefix: "backbone.1.1", outChannels: 24, stride: 1, filterSize: 3, skip: true)
  out = backbone_1_1(out)
  readers.append(backbone_1_1_reader)
  // 2.
  let (backbone_2_0, backbone_2_0_reader) = FusedMBConv(
    prefix: "backbone.2.0", outChannels: 48, stride: 2, filterSize: 3, skip: false,
    expandChannels: 96)
  out = backbone_2_0(out)
  readers.append(backbone_2_0_reader)
  for i in 1..<4 {
    let (backbone_2_x, backbone_2_x_reader) = FusedMBConv(
      prefix: "backbone.2.\(i)", outChannels: 48, stride: 1, filterSize: 3, skip: true,
      expandChannels: 192)
    out = backbone_2_x(out)
    readers.append(backbone_2_x_reader)
  }
  // 3.
  let (backbone_3_0, backbone_3_0_reader) = FusedMBConv(
    prefix: "backbone.3.0", outChannels: 64, stride: 2, filterSize: 3, skip: false,
    expandChannels: 192)
  out = backbone_3_0(out)
  readers.append(backbone_3_0_reader)
  for i in 1..<4 {
    let (backbone_3_x, backbone_3_x_reader) = FusedMBConv(
      prefix: "backbone.3.\(i)", outChannels: 64, stride: 1, filterSize: 3, skip: true,
      expandChannels: 256)
    out = backbone_3_x(out)
    readers.append(backbone_3_x_reader)
  }
  // 4.
  let (backbone_4_0, backbone_4_0_reader) = MBConv(
    prefix: "backbone.4.0", stride: 2, filterSize: 3, inChannels: 64, expandChannels: 256,
    outChannels: 128)
  out = backbone_4_0(out)
  readers.append(backbone_4_0_reader)
  for i in 1..<6 {
    let (backbone_4_x, backbone_4_x_reader) = MBConv(
      prefix: "backbone.4.\(i)", stride: 1, filterSize: 3, inChannels: 128, expandChannels: 512,
      outChannels: 128)
    out = backbone_4_x(out)
    readers.append(backbone_4_x_reader)
  }
  // 5.
  let (backbone_5_0, backbone_5_0_reader) = MBConv(
    prefix: "backbone.5.0", stride: 1, filterSize: 3, inChannels: 128, expandChannels: 768,
    outChannels: 160)
  out = backbone_5_0(out)
  readers.append(backbone_5_0_reader)
  for i in 1..<9 {
    let (backbone_5_x, backbone_5_x_reader) = MBConv(
      prefix: "backbone.5.\(i)", stride: 1, filterSize: 3, inChannels: 160, expandChannels: 960,
      outChannels: 160)
    out = backbone_5_x(out)
    readers.append(backbone_5_x_reader)
  }
  // 6.
  let (backbone_6_0, backbone_6_0_reader) = MBConv(
    prefix: "backbone.6.0", stride: 2, filterSize: 3, inChannels: 160, expandChannels: 960,
    outChannels: 256)
  out = backbone_6_0(out)
  readers.append(backbone_6_0_reader)
  for i in 1..<15 {
    let (backbone_6_x, backbone_6_x_reader) = MBConv(
      prefix: "backbone.6.\(i)", stride: 1, filterSize: 3, inChannels: 256, expandChannels: 1536,
      outChannels: 256)
    out = backbone_6_x(out)
    readers.append(backbone_6_x_reader)
  }
  let convOut = Convolution(
    groups: 1, filters: 1280, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out).swish()
  let mapper = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = mapper(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let backbone_0_0_weight = state_dict["backbone.0.0.weight"].float().cpu()
    let backbone_0_1_weight = state_dict["backbone.0.1.weight"].float().cpu()
    let backbone_0_1_running_mean = state_dict["backbone.0.1.running_mean"].float().cpu()
    let backbone_0_1_running_var = state_dict["backbone.0.1.running_var"].float().cpu()
    let backbone_0_1_bias = state_dict["backbone.0.1.bias"].float().cpu()
    let w_conv = backbone_0_0_weight.view(24, -1)
    let w_bn = torch.diag(
      backbone_0_1_weight.div(torch.sqrt(1e-3 + backbone_0_1_running_var)))
    let fused_weight = torch.mm(w_bn, w_conv).numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: fused_weight))
    let b_bn =
      backbone_0_1_bias
      - backbone_0_1_weight.mul(backbone_0_1_running_mean).div(
        torch.sqrt(backbone_0_1_running_var + 1e-3))
    conv.bias.copy(from: try! Tensor<Float>(numpy: b_bn.numpy()))
    for reader in readers {
      reader(state_dict)
    }
    let backbone_7_0_weight = state_dict["backbone.7.0.weight"].float().cpu()
    let backbone_7_1_weight = state_dict["backbone.7.1.weight"].float().cpu()
    let backbone_7_1_running_mean = state_dict["backbone.7.1.running_mean"].float().cpu()
    let backbone_7_1_running_var = state_dict["backbone.7.1.running_var"].float().cpu()
    let backbone_7_1_bias = state_dict["backbone.7.1.bias"].float().cpu()
    let w_conv_7 = backbone_7_0_weight.view(1280, -1)
    let w_bn_7 = torch.diag(
      backbone_7_1_weight.div(torch.sqrt(1e-3 + backbone_7_1_running_var)))
    let fused_weight_7 = torch.mm(w_bn_7, w_conv_7).numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: fused_weight_7))
    let b_bn_7 =
      backbone_7_1_bias
      - backbone_7_1_weight.mul(backbone_7_1_running_mean).div(
        torch.sqrt(backbone_7_1_running_var + 1e-3))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: b_bn_7.numpy()))
    let mapper_0_weight = state_dict["mapper.0.weight"].float().cpu()
    let mapper_1_running_mean = state_dict["mapper.1.running_mean"].float().cpu()
    let mapper_1_running_var = state_dict["mapper.1.running_var"].float().cpu()
    let w_conv_mapper = mapper_0_weight.view(16, -1)
    let w_bn_mapper = torch.diag(
      torch.ones([1]).div(torch.sqrt(1e-5 + mapper_1_running_var)))
    let fused_weight_mapper = torch.mm(w_bn_mapper, w_conv_mapper).numpy()
    mapper.weight.copy(from: try! Tensor<Float>(numpy: fused_weight_mapper))
    let b_bn_mapper =
      -mapper_1_running_mean.div(
        torch.sqrt(mapper_1_running_var + 1e-5))
    mapper.bias.copy(from: try! Tensor<Float>(numpy: b_bn_mapper.numpy()))
  }
  return (Model([x], [out]), reader)
}

func StageCPreviewer() -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 512, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = conv1(x).GELU()
  let norm1 = Convolution(
    groups: 512, filters: 512, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = norm1(out)

  let conv2 = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out).GELU()
  let norm2 = Convolution(
    groups: 512, filters: 512, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = norm2(out)

  let conv3 = ConvolutionTranspose(
    groups: 1, filters: 256, filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  out = conv3(out).GELU()
  let norm3 = Convolution(
    groups: 256, filters: 256, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = norm3(out)

  let conv4 = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv4(out).GELU()
  let norm4 = Convolution(
    groups: 256, filters: 256, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = norm4(out)

  let conv5 = ConvolutionTranspose(
    groups: 1, filters: 128, filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  out = conv5(out).GELU()
  let norm5 = Convolution(
    groups: 128, filters: 128, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = norm5(out)

  let conv6 = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv6(out).GELU()
  let norm6 = Convolution(
    groups: 128, filters: 128, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = norm6(out)

  let conv7 = ConvolutionTranspose(
    groups: 1, filters: 128, filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  out = conv7(out).GELU()
  let norm7 = Convolution(
    groups: 128, filters: 128, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = norm7(out)

  let conv8 = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv8(out).GELU()
  /*
  let norm8 = Convolution(groups: 128, filters: 128, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = norm8(out)
  */

  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out)

  let reader: (PythonObject) -> Void = { state_dict in
    let blocks_0_weight = state_dict["blocks.0.weight"].float().cpu()
    let blocks_0_bias = state_dict["blocks.0.bias"].float().cpu()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: blocks_0_weight.numpy()))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: blocks_0_bias.numpy()))

    let blocks_2_weight = state_dict["blocks.2.weight"].float().cpu()
    let blocks_2_bias = state_dict["blocks.2.bias"].float().cpu()
    let blocks_2_running_var = state_dict["blocks.2.running_var"].float().cpu()
    let blocks_2_running_mean = state_dict["blocks.2.running_mean"].float().cpu()
    let norm1_weight = blocks_2_weight / torch.sqrt(blocks_2_running_var + 1e-5)
    let norm1_bias =
      blocks_2_bias - blocks_2_weight * blocks_2_running_mean
      / torch.sqrt(blocks_2_running_var + 1e-5)
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight.numpy()))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias.numpy()))

    let blocks_3_weight = state_dict["blocks.3.weight"].float().cpu()
    let blocks_3_bias = state_dict["blocks.3.bias"].float().cpu()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: blocks_3_weight.numpy()))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: blocks_3_bias.numpy()))

    let blocks_5_weight = state_dict["blocks.5.weight"].float().cpu()
    let blocks_5_bias = state_dict["blocks.5.bias"].float().cpu()
    let blocks_5_running_var = state_dict["blocks.5.running_var"].float().cpu()
    let blocks_5_running_mean = state_dict["blocks.5.running_mean"].float().cpu()
    let norm2_weight = blocks_5_weight / torch.sqrt(blocks_5_running_var + 1e-5)
    let norm2_bias =
      blocks_5_bias - blocks_5_weight * blocks_5_running_mean
      / torch.sqrt(blocks_5_running_var + 1e-5)
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight.numpy()))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias.numpy()))

    let blocks_6_weight = state_dict["blocks.6.weight"].float().cpu()
    let blocks_6_bias = state_dict["blocks.6.bias"].float().cpu()
    conv3.weight.copy(from: try! Tensor<Float>(numpy: blocks_6_weight.numpy()))
    conv3.bias.copy(from: try! Tensor<Float>(numpy: blocks_6_bias.numpy()))

    let blocks_8_weight = state_dict["blocks.8.weight"].float().cpu()
    let blocks_8_bias = state_dict["blocks.8.bias"].float().cpu()
    let blocks_8_running_var = state_dict["blocks.8.running_var"].float().cpu()
    let blocks_8_running_mean = state_dict["blocks.8.running_mean"].float().cpu()
    let norm3_weight = blocks_8_weight / torch.sqrt(blocks_8_running_var + 1e-5)
    let norm3_bias =
      blocks_8_bias - blocks_8_weight * blocks_8_running_mean
      / torch.sqrt(blocks_8_running_var + 1e-5)
    norm3.weight.copy(from: try! Tensor<Float>(numpy: norm3_weight.numpy()))
    norm3.bias.copy(from: try! Tensor<Float>(numpy: norm3_bias.numpy()))

    let blocks_9_weight = state_dict["blocks.9.weight"].float().cpu()
    let blocks_9_bias = state_dict["blocks.9.bias"].float().cpu()
    conv4.weight.copy(from: try! Tensor<Float>(numpy: blocks_9_weight.numpy()))
    conv4.bias.copy(from: try! Tensor<Float>(numpy: blocks_9_bias.numpy()))

    let blocks_11_weight = state_dict["blocks.11.weight"].float().cpu()
    let blocks_11_bias = state_dict["blocks.11.bias"].float().cpu()
    let blocks_11_running_var = state_dict["blocks.11.running_var"].float().cpu()
    let blocks_11_running_mean = state_dict["blocks.11.running_mean"].float().cpu()
    let norm4_weight = blocks_11_weight / torch.sqrt(blocks_11_running_var + 1e-5)
    let norm4_bias =
      blocks_11_bias - blocks_11_weight * blocks_11_running_mean
      / torch.sqrt(blocks_11_running_var + 1e-5)
    norm4.weight.copy(from: try! Tensor<Float>(numpy: norm4_weight.numpy()))
    norm4.bias.copy(from: try! Tensor<Float>(numpy: norm4_bias.numpy()))

    let blocks_12_weight = state_dict["blocks.12.weight"].float().cpu()
    let blocks_12_bias = state_dict["blocks.12.bias"].float().cpu()
    conv5.weight.copy(from: try! Tensor<Float>(numpy: blocks_12_weight.numpy()))
    conv5.bias.copy(from: try! Tensor<Float>(numpy: blocks_12_bias.numpy()))

    let blocks_14_weight = state_dict["blocks.14.weight"].float().cpu()
    let blocks_14_bias = state_dict["blocks.14.bias"].float().cpu()
    let blocks_14_running_var = state_dict["blocks.14.running_var"].float().cpu()
    let blocks_14_running_mean = state_dict["blocks.14.running_mean"].float().cpu()
    let norm5_weight = blocks_14_weight / torch.sqrt(blocks_14_running_var + 1e-5)
    let norm5_bias =
      blocks_14_bias - blocks_14_weight * blocks_14_running_mean
      / torch.sqrt(blocks_14_running_var + 1e-5)
    norm5.weight.copy(from: try! Tensor<Float>(numpy: norm5_weight.numpy()))
    norm5.bias.copy(from: try! Tensor<Float>(numpy: norm5_bias.numpy()))

    let blocks_15_weight = state_dict["blocks.15.weight"].float().cpu()
    let blocks_15_bias = state_dict["blocks.15.bias"].float().cpu()
    conv6.weight.copy(from: try! Tensor<Float>(numpy: blocks_15_weight.numpy()))
    conv6.bias.copy(from: try! Tensor<Float>(numpy: blocks_15_bias.numpy()))

    let blocks_17_weight = state_dict["blocks.17.weight"].float().cpu()
    let blocks_17_bias = state_dict["blocks.17.bias"].float().cpu()
    let blocks_17_running_var = state_dict["blocks.17.running_var"].float().cpu()
    let blocks_17_running_mean = state_dict["blocks.17.running_mean"].float().cpu()
    let norm6_weight = blocks_17_weight / torch.sqrt(blocks_17_running_var + 1e-5)
    let norm6_bias =
      blocks_17_bias - blocks_17_weight * blocks_17_running_mean
      / torch.sqrt(blocks_17_running_var + 1e-5)
    norm6.weight.copy(from: try! Tensor<Float>(numpy: norm6_weight.numpy()))
    norm6.bias.copy(from: try! Tensor<Float>(numpy: norm6_bias.numpy()))

    let blocks_18_weight = state_dict["blocks.18.weight"].float().cpu()
    let blocks_18_bias = state_dict["blocks.18.bias"].float().cpu()
    conv7.weight.copy(from: try! Tensor<Float>(numpy: blocks_18_weight.numpy()))
    conv7.bias.copy(from: try! Tensor<Float>(numpy: blocks_18_bias.numpy()))

    let blocks_20_weight = state_dict["blocks.20.weight"].float().cpu()
    let blocks_20_bias = state_dict["blocks.20.bias"].float().cpu()
    let blocks_20_running_var = state_dict["blocks.20.running_var"].float().cpu()
    let blocks_20_running_mean = state_dict["blocks.20.running_mean"].float().cpu()
    let norm7_weight = blocks_20_weight / torch.sqrt(blocks_20_running_var + 1e-5)
    let norm7_bias =
      blocks_20_bias - blocks_20_weight * blocks_20_running_mean
      / torch.sqrt(blocks_20_running_var + 1e-5)
    norm7.weight.copy(from: try! Tensor<Float>(numpy: norm7_weight.numpy()))
    norm7.bias.copy(from: try! Tensor<Float>(numpy: norm7_bias.numpy()))

    let blocks_21_weight = state_dict["blocks.21.weight"].float().cpu()
    let blocks_21_bias = state_dict["blocks.21.bias"].float().cpu()
    conv8.weight.copy(from: try! Tensor<Float>(numpy: blocks_21_weight.numpy()))
    conv8.bias.copy(from: try! Tensor<Float>(numpy: blocks_21_bias.numpy()))

    let blocks_23_weight = state_dict["blocks.23.weight"].float().cpu()
    let blocks_23_bias = state_dict["blocks.23.bias"].float().cpu()
    let blocks_23_running_var = state_dict["blocks.23.running_var"].float().cpu()
    let blocks_23_running_mean = state_dict["blocks.23.running_mean"].float().cpu()
    let norm8_weight = blocks_23_weight / torch.sqrt(blocks_23_running_var + 1e-5)
    let norm8_bias =
      blocks_23_bias - blocks_23_weight * blocks_23_running_mean
      / torch.sqrt(blocks_23_running_var + 1e-5)
    /*
    norm8.weight.copy(from: try! Tensor<Float>(numpy: norm8_weight.numpy()))
    norm8.bias.copy(from: try! Tensor<Float>(numpy: norm8_bias.numpy()))
    */

    let blocks_24_weight = state_dict["blocks.24.weight"].float().cpu()
    let blocks_24_bias = state_dict["blocks.24.bias"].float().cpu()
    print(blocks_24_weight.shape)
    print(norm8_weight.shape)
    let conv_out_weight = blocks_24_weight * norm8_weight.view(1, -1, 1, 1)
    print(blocks_24_bias.shape)
    print(blocks_23_bias.shape)
    print(norm8_weight.shape)
    let conv_out_bias = torch.matmul(blocks_24_weight.view(3, -1), norm8_bias) + blocks_24_bias
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight.numpy()))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias.numpy()))
  }
  return (Model([x], [out]), reader)
}

let graph = DynamicGraph()
graph.withNoGrad {
  /*
  let x = graph.variable(try! Tensor<Float>(numpy: x.float().cpu().numpy())).toGPU(0)
  let (stageAEncoder, stageAEncoderReader) = StageAEncoder(batchSize: 1)
  stageAEncoder.compile(inputs: x)
  stageAEncoderReader(state_dict)
  let y = stageAEncoder(inputs: x)[0].as(of: Float.self)
  let (stageADecoder, stageADecoderReader) = StageADecoder(batchSize: 1, height: 512, width: 512)
  stageADecoder.compile(inputs: y)
  stageADecoderReader(state_dict)
  let out = stageADecoder(inputs: y)[0].as(of: Float.self)
  debugPrint(out)
  graph.openStore("/home/liu/workspace/swift-diffusion/wurstchen_3.0_stage_a_hq_f32.ckpt") {
    $0.write("encoder", model: stageAEncoder)
    $0.write("decoder", model: stageADecoder)
  }
  */
  /*
  let rTimeEmbed = rEmbedding(timesteps: 0.9936, batchSize: 2, embeddingSize: 64, maxPeriod: 10_000)
  let rZeros = rEmbedding(timesteps: 0, batchSize: 2, embeddingSize: 64, maxPeriod: 10_000)
  var rEmbed = Tensor<Float>(.CPU, .NC(2, 128))  // 192))
  rEmbed[0..<2, 0..<64] = rTimeEmbed
  rEmbed[0..<2, 64..<128] = rZeros
  // rEmbed[0..<2, 128..<192] = rZeros
  let rEmbedVariable = graph.variable(rEmbed).toGPU(0)
  let x = graph.variable(try! Tensor<Float>(numpy: x.float().cpu().numpy())).toGPU(0)
  let clipTextPooled = graph.variable(
    try! Tensor<Float>(numpy: clip_text_pooled.float().cpu().numpy())
  ).toGPU(0)
  let effnet = graph.variable(try! Tensor<Float>(numpy: effnet.float().cpu().numpy())).toGPU(0)
  let pixels = graph.variable(.GPU(0), .NCHW(2, 3, 8, 8), of: Float.self)
  pixels.full(0)
  let (stageBFixed, stageBFixedReader) = StageBFixed(batchSize: 2, height: 256, width: 256, effnetHeight: 24, effnetWidth: 24)
  stageBFixed.compile(inputs: effnet, pixels, clipTextPooled)
  stageBFixedReader(state_dict)
  let kvs = stageBFixed(inputs: effnet, pixels, clipTextPooled).map { $0.as(of: Float.self) }
  let (stageB, stageBReader) = StageB(
    batchSize: 2, cIn: 4, height: 256, width: 256, effnetHeight: 24, effnetWidth: 24)
  stageB.compile(inputs: [x, rEmbedVariable] + kvs)
  stageBReader(state_dict)
  let out = stageB(inputs: x, [rEmbedVariable] + kvs)[0].as(
    of: Float.self)
  debugPrint(out)
  graph.openStore("/home/liu/workspace/swift-diffusion/wurstchen_3.0_stage_b_f32.ckpt") {
    $0.write("stage_b_fixed", model: stageBFixed)
    $0.write("stage_b", model: stageB)
  }
  */
  let x = graph.variable(try! Tensor<Float>(numpy: x.float().cpu().numpy())).toGPU(0)
  let (previewer, previewerReader) = StageCPreviewer()
  previewer.compile(inputs: x)
  previewerReader(previewer_state_dict)
  let previewOut = previewer(inputs: x)[0].as(of: Float.self)
  debugPrint(previewOut)
  let img = graph.variable(try! Tensor<Float>(numpy: img.float().cpu().numpy())).toGPU(0)
  let (effnet, effnetReader) = EfficientNetEncoder()
  effnet.compile(inputs: img)
  effnetReader(effnet_state_dict)
  let imgOut = effnet(inputs: img)[0].as(of: Float.self)
  debugPrint(imgOut)
  let rTimeEmbed = rEmbedding(timesteps: 0.9936, batchSize: 2, embeddingSize: 64, maxPeriod: 10_000)
  let rZeros = rEmbedding(timesteps: 0, batchSize: 2, embeddingSize: 64, maxPeriod: 10_000)
  var rEmbed = Tensor<Float>(.CPU, .NC(2, 192))
  rEmbed[0..<2, 0..<64] = rTimeEmbed
  rEmbed[0..<2, 64..<128] = rZeros
  rEmbed[0..<2, 128..<192] = rZeros
  let rEmbedVariable = graph.variable(rEmbed).toGPU(0)
  let clipTextPooled = graph.variable(
    try! Tensor<Float>(numpy: clip_text_pooled.float().cpu().numpy())
  ).toGPU(0)
  let clipText = graph.variable(try! Tensor<Float>(numpy: clip_text.float().cpu().numpy())).toGPU(0)
  let clipImg = graph.variable(try! Tensor<Float>(numpy: clip_img.float().cpu().numpy())).toGPU(0)
  let (stageCFixed, stageCFixedReader) = StageCFixed(batchSize: 2, t: 77 + 8)
  stageCFixed.compile(inputs: clipText, clipTextPooled, clipImg)
  stageCFixedReader(state_dict)
  let kvs = stageCFixed(inputs: clipText, clipTextPooled, clipImg).map { $0.as(of: Float.self) }
  let (stageC, stageCReader) = StageC(batchSize: 2, height: 24, width: 24, t: 77 + 8)
  stageC.compile(inputs: [x, rEmbedVariable] + kvs)
  stageCReader(state_dict)
  let out = stageC(inputs: x, [rEmbedVariable] + kvs)[0].as(
    of: Float.self)
  debugPrint(out)
  graph.openStore("/home/liu/workspace/swift-diffusion/wurstchen_3.0_stage_c_f16_f32.ckpt") {
    $0.write("previewer", model: previewer)
    $0.write("effnet", model: effnet)
    $0.write("stage_c_fixed", model: stageCFixed)
    $0.write("stage_c", model: stageC)
  }
}

// print(models_b.stage_a.up_blocks)
