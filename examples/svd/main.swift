import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let streamlit_helpers = Python.import("scripts.demo.streamlit_helpers")
let PIL = Python.import("PIL")
let numpy = Python.import("numpy")
let torch = Python.import("torch")
let pytorch_lightning = Python.import("pytorch_lightning")
let torchvision = Python.import("torchvision")
let kornia = Python.import("kornia")

let version_dict: [String: PythonObject] = [
  "T": 14,
  "H": 512,
  "W": 512,
  "C": 4, "f": 8,
  "config": "/home/liu/workspace/generative-models/configs/inference/svd_image_decoder.yaml",
  "ckpt": "/home/liu/workspace/generative-models/checkpoints/AnimateLCM-SVD-xt-1.1.safetensors",
  "options": [
    "discretization": 1,
    "cfg": 2.5,
    "sigma_min": 0.002,
    "sigma_max": 700.0,
    "rho": 7.0,
    "guider": 2,
    "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
    "num_steps": 25,
  ],
]

let state = streamlit_helpers.init_st(version_dict, load_filter: false)

func load_img_for_prediction(path: String, W: Int, H: Int) -> PythonObject {
  var image = PIL.Image.open(path)
  if image.mode != "RGB" {
    image = image.convert("RGB")
  }
  let (width, height) = image.size.tuple2
  let rfs = Double(
    streamlit_helpers.get_resizing_factor(
      PythonObject(tupleOf: W, H), PythonObject(tupleOf: width, height)))!
  let resize_size = (
    Int((Double(height)! * rfs).rounded(.up)), Int((Double(width)! * rfs).rounded(.up))
  )
  let top = (resize_size.0 - H) / 2
  let left = (resize_size.1 - W) / 2
  image = numpy.array(image).transpose(2, 0, 1)
  image = torch.from_numpy(image).to(dtype: torch.float32) / 255.0
  image = image.unsqueeze(0)
  image = torch.nn.functional.interpolate(
    image, PythonObject(tupleOf: resize_size.0, resize_size.1), mode: "area", antialias: false)
  image = torchvision.transforms.functional.crop(image, top: top, left: left, height: H, width: W)
  return image * 2 - 1
}
let img = load_img_for_prediction(
  path: "/home/liu/workspace/swift-diffusion/kandinsky.png", W: 512, H: 512
).cuda()
let ukeys = Python.set(
  streamlit_helpers.get_unique_embedder_keys_from_conditioner(state["model"].conditioner))
var value_dict = streamlit_helpers.init_embedder_options(ukeys, Python.dict())
pytorch_lightning.seed_everything(23)
value_dict["image_only_indicator"] = 0
value_dict["cond_frames_without_noise"] = img
value_dict["cond_frames"] = img  // + 0.02 * torch.randn_like(img)
value_dict["cond_aug"] = 0.02

func preprocess(image: PythonObject) -> PythonObject {
  let mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
  let std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
  // normalize to [0,1]
  var x = kornia.geometry.resize(
    image,
    PythonObject(tupleOf: 224, 224),
    interpolation: "bicubic",
    align_corners: true,
    antialias: true
  )
  x = (x + 1.0) / 2.0
  // renormalize according to clip
  x = kornia.enhance.normalize(x, mean, std)
  return x
}

let clip_image = preprocess(image: img).detach().cpu()

var options = version_dict["options"]!
options["num_frames"] = 14
/*
let (sampler, num_rows, num_cols) = streamlit_helpers.init_sampling(options: options).tuple3
let num_samples = num_rows * num_cols
let sample = streamlit_helpers.do_sample(
  state["model"], sampler, value_dict, num_samples, 512, 512, 4, 8,
  T: 14, batch2model_input: ["num_video_frames", "image_only_indicator"],
  force_uc_zero_embeddings: options["force_uc_zero_embeddings"], return_latents: false,
  decoding_t: 1)
*/
// streamlit_helpers.save_video_as_grid_and_mp4(sample, "/home/liu/workspace/swift-diffusion/outputs/", 14, fps: value_dict["fps"])
// print(state["model"].conditioner)
// print(state["model"].model.diffusion_model)

func CLIPSelfAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
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

func CLIPResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let (toqueries, tokeys, tovalues, unifyheads, attention) = CLIPSelfAttention(
    k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_1_weight = state_dict["\(prefix).ln_1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).ln_1.bias"].type(torch.float).cpu().numpy()
    ln1.weight.copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    ln1.bias.copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let in_proj_weight = state_dict["\(prefix).attn.in_proj_weight"].type(
      torch.float
    ).cpu().numpy()
    let in_proj_bias = state_dict["\(prefix).attn.in_proj_bias"].type(torch.float)
      .cpu().numpy()
    toqueries.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[..<(1280), ...]))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_proj_bias[..<(1280)]))
    tokeys.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(1280)..<(2 * 1280), ...]))
    tokeys.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(1280)..<(2 * 1280)]))
    tovalues.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(2 * 1280)..., ...]))
    tovalues.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(2 * 1280)...]))
    let out_proj_weight = state_dict["\(prefix).attn.out_proj.weight"].type(torch.float).cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).attn.out_proj.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: out_proj_bias))
    let ln_2_weight = state_dict["\(prefix).ln_2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).ln_2.bias"].type(torch.float).cpu().numpy()
    ln2.weight.copy(from: try! Tensor<Float>(numpy: ln_2_weight))
    ln2.bias.copy(from: try! Tensor<Float>(numpy: ln_2_bias))
    let c_fc_weight = state_dict["\(prefix).mlp.c_fc.weight"].type(torch.float).cpu().numpy()
    let c_fc_bias = state_dict["\(prefix).mlp.c_fc.bias"].type(torch.float).cpu().numpy()
    fc.weight.copy(from: try! Tensor<Float>(numpy: c_fc_weight))
    fc.bias.copy(from: try! Tensor<Float>(numpy: c_fc_bias))
    let c_proj_weight = state_dict["\(prefix).mlp.c_proj.weight"].type(torch.float).cpu().numpy()
    let c_proj_bias = state_dict["\(prefix).mlp.c_proj.bias"].type(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: c_proj_weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: c_proj_bias))
  }
  return (reader, Model([x], [out]))
}

func VisionTransformer(
  grid: Int, width: Int, outputDim: Int, layers: Int, heads: Int, batchSize: Int,
  noFinalLayerNorm: Bool = false
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let classEmbedding = Parameter<Float>(.GPU(1), .CHW(1, 1, width))
  let positionalEmbedding = Parameter<Float>(.GPU(1), .CHW(1, grid * grid + 1, width))
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], noBias: true,
    hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPre(out)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (reader, block) = CLIPResidualAttentionBlock(
      prefix: "open_clip.model.visual.transformer.resblocks.\(i)", k: width / heads, h: heads,
      b: batchSize,
      t: grid * grid + 1)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
    readers.append(reader)
  }
  let finalLayerNorm: Model?
  if !noFinalLayerNorm {
    let lnPost = LayerNorm(epsilon: 1e-5, axis: [1], name: "post_layernorm")
    out = lnPost(out.reshaped([batchSize, width], strides: [width, 1]))
    finalLayerNorm = lnPost
  } else {
    finalLayerNorm = nil
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1_weight = state_dict["open_clip.model.visual.conv1.weight"].type(
      torch.float
    ).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let class_embedding_weight = state_dict["open_clip.model.visual.class_embedding"].type(
      torch.float
    ).cpu().numpy()
    classEmbedding.weight.copy(from: try! Tensor<Float>(numpy: class_embedding_weight))
    let positional_embedding_weight = state_dict[
      "open_clip.model.visual.positional_embedding"
    ].type(torch.float).cpu().numpy()
    positionalEmbedding.weight.copy(from: try! Tensor<Float>(numpy: positional_embedding_weight))
    let ln_pre_weight = state_dict["open_clip.model.visual.ln_pre.weight"].type(torch.float).cpu()
      .numpy()
    let ln_pre_bias = state_dict["open_clip.model.visual.ln_pre.bias"].type(torch.float).cpu()
      .numpy()
    lnPre.weight.copy(from: try! Tensor<Float>(numpy: ln_pre_weight))
    lnPre.bias.copy(from: try! Tensor<Float>(numpy: ln_pre_bias))
    for reader in readers {
      reader(state_dict)
    }
    if let lnPost = finalLayerNorm {
      let ln_post_weight = state_dict["open_clip.model.visual.ln_post.weight"].type(torch.float)
        .cpu()
        .numpy()
      let ln_post_bias = state_dict["open_clip.model.visual.ln_post.bias"].type(torch.float).cpu()
        .numpy()
      lnPost.weight.copy(from: try! Tensor<Float>(numpy: ln_post_weight))
      lnPost.bias.copy(from: try! Tensor<Float>(numpy: ln_post_bias))
    }
  }
  return (reader, Model([x], [out]))
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
    let norm1_weight = state_dict["encoder.\(prefix).norm1.weight"].cpu().numpy()
    let norm1_bias = state_dict["encoder.\(prefix).norm1.bias"].cpu().numpy()
    norm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let conv1_weight = state_dict["encoder.\(prefix).conv1.weight"].cpu().numpy()
    let conv1_bias = state_dict["encoder.\(prefix).conv1.bias"].cpu().numpy()
    conv1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["encoder.\(prefix).norm2.weight"].cpu().numpy()
    let norm2_bias = state_dict["encoder.\(prefix).norm2.bias"].cpu().numpy()
    norm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["encoder.\(prefix).conv2.weight"].cpu().numpy()
    let conv2_bias = state_dict["encoder.\(prefix).conv2.bias"].cpu().numpy()
    conv2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["encoder.\(prefix).nin_shortcut.weight"].cpu().numpy()
      let nin_shortcut_bias = state_dict["encoder.\(prefix).nin_shortcut.bias"].cpu().numpy()
      ninShortcut.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
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
    let norm_weight = state_dict["encoder.\(prefix).norm.weight"].cpu().numpy()
    let norm_bias = state_dict["encoder.\(prefix).norm.bias"].cpu().numpy()
    norm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_bias))
    let k_weight = state_dict["encoder.\(prefix).k.weight"].cpu().numpy()
    let k_bias = state_dict["encoder.\(prefix).k.bias"].cpu().numpy()
    tokeys.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["encoder.\(prefix).q.weight"].cpu().numpy()
    let q_bias = state_dict["encoder.\(prefix).q.bias"].cpu().numpy()
    toqueries.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["encoder.\(prefix).v.weight"].cpu().numpy()
    let v_bias = state_dict["encoder.\(prefix).v.bias"].cpu().numpy()
    tovalues.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["encoder.\(prefix).proj_out.weight"].cpu().numpy()
    let proj_out_bias = state_dict["encoder.\(prefix).proj_out.bias"].cpu().numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
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
        prefix: "down.\(i).block.\(j)", outChannels: channel, shortcut: previousChannel != channel)
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
        let conv_weight = state_dict["encoder.down.\(downLayer).downsample.conv.weight"].cpu()
          .numpy()
        let conv_bias = state_dict["encoder.down.\(downLayer).downsample.conv.bias"].cpu().numpy()
        conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "mid.attn_1", inChannels: previousChannel, batchSize: batchSize, width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 8, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let quantConv2d = Convolution(
    groups: 1, filters: 8, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = quantConv2d(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["encoder.conv_in.weight"].cpu().numpy()
    let conv_in_bias = state_dict["encoder.conv_in.bias"].cpu().numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    let norm_out_weight = state_dict["encoder.norm_out.weight"].cpu().numpy()
    let norm_out_bias = state_dict["encoder.norm_out.bias"].cpu().numpy()
    normOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["encoder.conv_out.weight"].cpu().numpy()
    let conv_out_bias = state_dict["encoder.conv_out.bias"].cpu().numpy()
    convOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_out_bias))
    let quant_conv_weight = state_dict["quant_conv.weight"].cpu().numpy()
    let quant_conv_bias = state_dict["quant_conv.bias"].cpu().numpy()
    quantConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: quant_conv_weight))
    quantConv2d.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: quant_conv_bias))
  }
  return (reader, Model([x], [out]))
}

public func timeEmbedding(timestep: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int)
  -> Tensor<
    Float
  >
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timestep
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
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

func LabelEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func TimePosEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out], name: "time_pos_embed"))
}

func ResBlock(b: Int, outChannels: Int, skipConnection: Bool) -> (
  Model, Model, Model, Model, Model, Model?, Model
) {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: outChannels)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([1, outChannels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var skipModel: Model? = nil
  if skipConnection {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]))
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = skip
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
  }
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel,
    Model([x, emb], [out])
  )
}

func TimeResBlock(b: Int, h: Int, w: Int, channels: Int) -> (
  Model, Model, Model, Model, Model, Model
) {
  let x = Input()
  let emb = Input()
  let y = x.transposed(0, 1).reshaped([1, channels, b, h * w])  // [b, c, h, w] -> [c, b, h, w] -> [1, c, b, h * w]
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(y)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: channels, filterSize: [3, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 0], end: [1, 0])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: channels)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([1, channels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: channels, filterSize: [3, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 0], end: [1, 0])))
  out = y + outLayerConv2d(out)  // This layer should be zero init if training.
  out = out.reshaped([channels, b, h, w]).transposed(0, 1)
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d,
    Model([x, emb], [out], name: "time_stack")
  )
}

func SelfAttention(k: Int, h: Int, b: Int, hw: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, hw])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func CrossAttentionKeysAndValues(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = Dense(count: k * h, noBias: true)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, unifyheads, Model([x, keys, values], [out]))
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

func BasicTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2: Model?
  let toqueries2: Model?
  let unifyheads2: Model?
  let keys: Input?
  if t == 1 {
    out = values + residual
    keys = nil
    layerNorm2 = nil
    toqueries2 = nil
    unifyheads2 = nil
  } else {
    let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
    out = layerNorm(out)
    let (toqueries, unifyheads, attn2) = CrossAttentionKeysAndValues(
      k: k, h: h, b: b, hw: hw, t: t)
    let keys2 = Input()
    out = attn2(out, keys2, values) + residual
    keys = keys2
    layerNorm2 = layerNorm
    toqueries2 = toqueries
    unifyheads2 = unifyheads
  }
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  let reader: (PythonObject) -> Void = { state_dict in
    let attn1_to_k_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_k.weight"
    ].cpu().numpy()
    tokeys1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_k_weight))
    print("\"diffusion_model.\(prefix).attn1.to_k.weight\": [\"\(tokeys1.weight.name)\"],")
    let attn1_to_q_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_q.weight"
    ].cpu().numpy()
    toqueries1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_q_weight))
    print("\"diffusion_model.\(prefix).attn1.to_q.weight\": [\"\(toqueries1.weight.name)\"],")
    let attn1_to_v_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_v.weight"
    ].cpu().numpy()
    tovalues1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_v_weight))
    print("\"diffusion_model.\(prefix).attn1.to_v.weight\": [\"\(tovalues1.weight.name)\"],")
    let attn1_to_out_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_out.0.weight"
    ].cpu().numpy()
    let attn1_to_out_bias = state_dict[
      "diffusion_model.\(prefix).attn1.to_out.0.bias"
    ].cpu().numpy()
    unifyheads1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attn1_to_out_weight))
    print("\"diffusion_model.\(prefix).attn1.to_out.0.weight\": [\"\(unifyheads1.weight.name)\"],")
    unifyheads1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn1_to_out_bias))
    print("\"diffusion_model.\(prefix).attn1.to_out.0.bias\": [\"\(unifyheads1.bias.name)\"],")
    let ff_net_0_proj_weight = state_dict[
      "diffusion_model.\(prefix).ff.net.0.proj.weight"
    ].cpu().numpy()
    let ff_net_0_proj_bias = state_dict[
      "diffusion_model.\(prefix).ff.net.0.proj.bias"
    ].cpu().numpy()
    fc10.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[..<intermediateSize, ...]))
    fc10.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[..<intermediateSize]))
    fc11.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[intermediateSize..., ...]))
    fc11.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[intermediateSize...]))
    print(
      "\"diffusion_model.\(prefix).ff.net.0.proj.weight\": [\"\(fc10.weight.name)\", \"\(fc11.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).ff.net.0.proj.bias\": [\"\(fc10.bias.name)\", \"\(fc11.bias.name)\"],"
    )
    let ff_net_2_weight = state_dict[
      "diffusion_model.\(prefix).ff.net.2.weight"
    ].cpu().numpy()
    let ff_net_2_bias = state_dict[
      "diffusion_model.\(prefix).ff.net.2.bias"
    ].cpu().numpy()
    fc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ff_net_2_weight))
    fc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_2_bias))
    print("\"diffusion_model.\(prefix).ff.net.2.weight\": [\"\(fc2.weight.name)\"],")
    print("\"diffusion_model.\(prefix).ff.net.2.bias\": [\"\(fc2.bias.name)\"],")
    if let layerNorm2 = layerNorm2, let toqueries2 = toqueries2, let unifyheads2 = unifyheads2 {
      let norm2_weight = state_dict[
        "diffusion_model.\(prefix).norm2.weight"
      ]
      .cpu().numpy()
      let norm2_bias = state_dict[
        "diffusion_model.\(prefix).norm2.bias"
      ]
      .cpu().numpy()
      layerNorm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
      layerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
      print("\"diffusion_model.\(prefix).norm2.weight\": [\"\(layerNorm2.weight.name)\"],")
      print("\"diffusion_model.\(prefix).norm2.bias\": [\"\(layerNorm2.bias.name)\"],")
      let attn2_to_q_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_q.weight"
      ].cpu().numpy()
      toqueries2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_q_weight))
      print("\"diffusion_model.\(prefix).attn2.to_q.weight\": [\"\(toqueries2.weight.name)\"],")
      let attn2_to_out_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_out.0.weight"
      ].cpu().numpy()
      let attn2_to_out_bias = state_dict[
        "diffusion_model.\(prefix).attn2.to_out.0.bias"
      ].cpu().numpy()
      unifyheads2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: attn2_to_out_weight))
      unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
      print(
        "\"diffusion_model.\(prefix).attn2.to_out.0.weight\": [\"\(unifyheads2.weight.name)\"],")
      print("\"diffusion_model.\(prefix).attn2.to_out.0.bias\": [\"\(unifyheads2.bias.name)\"],")
    }
    let norm1_weight = state_dict[
      "diffusion_model.\(prefix).norm1.weight"
    ]
    .cpu().numpy()
    let norm1_bias = state_dict[
      "diffusion_model.\(prefix).norm1.bias"
    ]
    .cpu().numpy()
    layerNorm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    layerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    print("\"diffusion_model.\(prefix).norm1.weight\": [\"\(layerNorm1.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm1.bias\": [\"\(layerNorm1.bias.name)\"],")
    let norm3_weight = state_dict[
      "diffusion_model.\(prefix).norm3.weight"
    ]
    .cpu().numpy()
    let norm3_bias = state_dict[
      "diffusion_model.\(prefix).norm3.bias"
    ]
    .cpu().numpy()
    layerNorm3.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm3_weight))
    layerNorm3.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm3_bias))
    print("\"diffusion_model.\(prefix).norm3.weight\": [\"\(layerNorm3.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm3.bias\": [\"\(layerNorm3.bias.name)\"],")
  }
  if let keys = keys {
    return (reader, Model([x, keys, values], [out]))
  } else {
    return (reader, Model([x, values], [out]))
  }
}

func BasicTimeTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let timeEmb = Input()
  let values = Input()
  var out = x.transposed(0, 1) + timeEmb.reshaped([1, b, k * h])
  let normIn = LayerNorm(epsilon: 1e-5, axis: [2])
  let (ffIn10, ffIn11, ffIn2, ffIn) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ffIn(normIn(out)) + out
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: hw, hw: b)
  out = attn1(layerNorm1(out)) + out
  var residual = out
  let layerNorm2: Model?
  let toqueries2: Model?
  let unifyheads2: Model?
  let keys: Input?
  if t == 1 {
    out = values + residual
    keys = nil
    layerNorm2 = nil
    toqueries2 = nil
    unifyheads2 = nil
  } else {
    let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
    out = layerNorm(out)
    let keys2 = Input()
    let (toqueries, unifyheads, attn2) = CrossAttentionKeysAndValues(
      k: k, h: h, b: hw, hw: b, t: t)
    out = attn2(out, keys2, values) + residual
    keys = keys2
    layerNorm2 = layerNorm
    toqueries2 = toqueries
    unifyheads2 = unifyheads
  }
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  out = out.transposed(0, 1)
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_in_weight = state_dict[
      "diffusion_model.\(prefix).norm_in.weight"
    ]
    .cpu().numpy()
    let norm_in_bias = state_dict[
      "diffusion_model.\(prefix).norm_in.bias"
    ]
    .cpu().numpy()
    normIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_in_weight))
    normIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_in_bias))
    print("\"diffusion_model.\(prefix).norm1.weight\": [\"\(layerNorm1.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm1.bias\": [\"\(layerNorm1.bias.name)\"],")
    let ff_in_net_0_proj_weight = state_dict[
      "diffusion_model.\(prefix).ff_in.net.0.proj.weight"
    ].cpu().numpy()
    let ff_in_net_0_proj_bias = state_dict[
      "diffusion_model.\(prefix).ff_in.net.0.proj.bias"
    ].cpu().numpy()
    ffIn10.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_in_net_0_proj_weight[..<intermediateSize, ...]))
    ffIn10.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_in_net_0_proj_bias[..<intermediateSize]))
    ffIn11.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_in_net_0_proj_weight[intermediateSize..., ...]))
    ffIn11.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_in_net_0_proj_bias[intermediateSize...]))
    print(
      "\"diffusion_model.\(prefix).ff_in.net.0.proj.weight\": [\"\(ffIn10.weight.name)\", \"\(ffIn11.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).ff_in.net.0.proj.bias\": [\"\(ffIn10.bias.name)\", \"\(ffIn11.bias.name)\"],"
    )
    let ff_in_net_2_weight = state_dict[
      "diffusion_model.\(prefix).ff_in.net.2.weight"
    ].cpu().numpy()
    let ff_in_net_2_bias = state_dict[
      "diffusion_model.\(prefix).ff_in.net.2.bias"
    ].cpu().numpy()
    ffIn2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ff_in_net_2_weight))
    ffIn2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_in_net_2_bias))
    print("\"diffusion_model.\(prefix).ff_in.net.2.weight\": [\"\(ffIn2.weight.name)\"],")
    print("\"diffusion_model.\(prefix).ff_in.net.2.bias\": [\"\(ffIn2.bias.name)\"],")
    let attn1_to_k_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_k.weight"
    ].cpu().numpy()
    tokeys1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_k_weight))
    print("\"diffusion_model.\(prefix).attn1.to_k.weight\": [\"\(tokeys1.weight.name)\"],")
    let attn1_to_q_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_q.weight"
    ].cpu().numpy()
    toqueries1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_q_weight))
    print("\"diffusion_model.\(prefix).attn1.to_q.weight\": [\"\(toqueries1.weight.name)\"],")
    let attn1_to_v_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_v.weight"
    ].cpu().numpy()
    tovalues1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_v_weight))
    print("\"diffusion_model.\(prefix).attn1.to_v.weight\": [\"\(tovalues1.weight.name)\"],")
    let attn1_to_out_weight = state_dict[
      "diffusion_model.\(prefix).attn1.to_out.0.weight"
    ].cpu().numpy()
    let attn1_to_out_bias = state_dict[
      "diffusion_model.\(prefix).attn1.to_out.0.bias"
    ].cpu().numpy()
    unifyheads1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attn1_to_out_weight))
    print("\"diffusion_model.\(prefix).attn1.to_out.0.weight\": [\"\(unifyheads1.weight.name)\"],")
    unifyheads1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn1_to_out_bias))
    print("\"diffusion_model.\(prefix).attn1.to_out.0.bias\": [\"\(unifyheads1.bias.name)\"],")
    let ff_net_0_proj_weight = state_dict[
      "diffusion_model.\(prefix).ff.net.0.proj.weight"
    ].cpu().numpy()
    let ff_net_0_proj_bias = state_dict[
      "diffusion_model.\(prefix).ff.net.0.proj.bias"
    ].cpu().numpy()
    fc10.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[..<intermediateSize, ...]))
    fc10.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[..<intermediateSize]))
    fc11.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[intermediateSize..., ...]))
    fc11.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[intermediateSize...]))
    print(
      "\"diffusion_model.\(prefix).ff.net.0.proj.weight\": [\"\(fc10.weight.name)\", \"\(fc11.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).ff.net.0.proj.bias\": [\"\(fc10.bias.name)\", \"\(fc11.bias.name)\"],"
    )
    let ff_net_2_weight = state_dict[
      "diffusion_model.\(prefix).ff.net.2.weight"
    ].cpu().numpy()
    let ff_net_2_bias = state_dict[
      "diffusion_model.\(prefix).ff.net.2.bias"
    ].cpu().numpy()
    fc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ff_net_2_weight))
    fc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_2_bias))
    print("\"diffusion_model.\(prefix).ff.net.2.weight\": [\"\(fc2.weight.name)\"],")
    print("\"diffusion_model.\(prefix).ff.net.2.bias\": [\"\(fc2.bias.name)\"],")
    if let layerNorm2 = layerNorm2, let toqueries2 = toqueries2, let unifyheads2 = unifyheads2 {
      let norm2_weight = state_dict[
        "diffusion_model.\(prefix).norm2.weight"
      ]
      .cpu().numpy()
      let norm2_bias = state_dict[
        "diffusion_model.\(prefix).norm2.bias"
      ]
      .cpu().numpy()
      layerNorm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
      layerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
      print("\"diffusion_model.\(prefix).norm2.weight\": [\"\(layerNorm2.weight.name)\"],")
      print("\"diffusion_model.\(prefix).norm2.bias\": [\"\(layerNorm2.bias.name)\"],")
      let attn2_to_q_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_q.weight"
      ].cpu().numpy()
      toqueries2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_q_weight))
      print("\"diffusion_model.\(prefix).attn2.to_q.weight\": [\"\(toqueries2.weight.name)\"],")
      let attn2_to_out_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_out.0.weight"
      ].cpu().numpy()
      let attn2_to_out_bias = state_dict[
        "diffusion_model.\(prefix).attn2.to_out.0.bias"
      ].cpu().numpy()
      unifyheads2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: attn2_to_out_weight))
      unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
      print(
        "\"diffusion_model.\(prefix).attn2.to_out.0.weight\": [\"\(unifyheads2.weight.name)\"],")
      print("\"diffusion_model.\(prefix).attn2.to_out.0.bias\": [\"\(unifyheads2.bias.name)\"],")
    }
    let norm1_weight = state_dict[
      "diffusion_model.\(prefix).norm1.weight"
    ]
    .cpu().numpy()
    let norm1_bias = state_dict[
      "diffusion_model.\(prefix).norm1.bias"
    ]
    .cpu().numpy()
    layerNorm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    layerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    print("\"diffusion_model.\(prefix).norm1.weight\": [\"\(layerNorm1.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm1.bias\": [\"\(layerNorm1.bias.name)\"],")
    let norm3_weight = state_dict[
      "diffusion_model.\(prefix).norm3.weight"
    ]
    .cpu().numpy()
    let norm3_bias = state_dict[
      "diffusion_model.\(prefix).norm3.bias"
    ]
    .cpu().numpy()
    layerNorm3.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm3_weight))
    layerNorm3.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm3_bias))
    print("\"diffusion_model.\(prefix).norm3.weight\": [\"\(layerNorm3.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm3.bias\": [\"\(layerNorm3.bias.name)\"],")
  }
  if let keys = keys {
    return (reader, Model([x, timeEmb, keys, values], [out], name: "time_stack"))
  } else {
    return (reader, Model([x, timeEmb, values], [out], name: "time_stack"))
  }
}

func SpatialTransformer(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let hw = height * width
  let projIn = Dense(count: k * h)
  out = projIn(out.reshaped([b, k * h, hw]).transposed(1, 2))
  var readers = [(PythonObject) -> Void]()
  let timeEmb: Input?
  let mixFactor: Parameter<Float>?
  if depth > 0 {
    let emb = Input()
    kvs.append(emb)
    timeEmb = emb
    mixFactor = Parameter<Float>(.GPU(1), .C(1), name: "time_mixer")
  } else {
    timeEmb = nil
    mixFactor = nil
  }
  for i in 0..<depth {
    if t == 1 {
      let values = Input()
      kvs.append(values)
      let (reader, block) = BasicTransformerBlock(
        prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
        intermediateSize: intermediateSize)
      out = block(out, values)
      readers.append(reader)
    } else {
      let keys = Input()
      kvs.append(keys)
      let values = Input()
      kvs.append(values)
      let (reader, block) = BasicTransformerBlock(
        prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
        intermediateSize: intermediateSize)
      out = block(out, keys, values)
      readers.append(reader)
    }
    if let timeEmb = timeEmb, let mixFactor = mixFactor {
      if t == 1 {
        let values = Input()
        kvs.append(values)
        let (reader, block) = BasicTimeTransformerBlock(
          prefix: "\(prefix).time_stack.\(i)", k: k, h: h, b: b, hw: hw, t: t,
          intermediateSize: intermediateSize)
        out = mixFactor .* out + (1 - mixFactor) .* block(out, timeEmb, values)
        readers.append(reader)
      } else {
        let keys = Input()
        kvs.append(keys)
        let values = Input()
        kvs.append(values)
        let (reader, block) = BasicTimeTransformerBlock(
          prefix: "\(prefix).time_stack.\(i)", k: k, h: h, b: b, hw: hw, t: t,
          intermediateSize: intermediateSize)
        out = mixFactor .* out + (1 - mixFactor) .* block(out, timeEmb, keys, values)
        readers.append(reader)
      }
    }
  }
  out = out.reshaped([b, height, width, k * h]).permuted(0, 3, 1, 2)
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_weight = state_dict["diffusion_model.\(prefix).norm.weight"]
      .cpu().numpy()
    let norm_bias = state_dict["diffusion_model.\(prefix).norm.bias"].cpu().numpy()
    norm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_bias))
    print("\"diffusion_model.\(prefix).norm.weight\": [\"\(norm.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm.bias\": [\"\(norm.bias.name)\"],")
    let proj_in_weight = state_dict["diffusion_model.\(prefix).proj_in.weight"]
      .cpu().numpy()
    let proj_in_bias = state_dict["diffusion_model.\(prefix).proj_in.bias"]
      .cpu().numpy()
    projIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_in_weight))
    projIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_in_bias))
    print("\"diffusion_model.\(prefix).proj_in.weight\": [\"\(projIn.weight.name)\"],")
    print("\"diffusion_model.\(prefix).proj_in.bias\": [\"\(projIn.bias.name)\"],")
    for reader in readers {
      reader(state_dict)
    }
    let proj_out_weight = state_dict[
      "diffusion_model.\(prefix).proj_out.weight"
    ].cpu().numpy()
    let proj_out_bias = state_dict["diffusion_model.\(prefix).proj_out.bias"]
      .cpu().numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
    print("\"diffusion_model.\(prefix).proj_out.weight\": [\"\(projOut.weight.name)\"],")
    print("\"diffusion_model.\(prefix).proj_out.bias\": [\"\(projOut.bias.name)\"],")
    if let mixFactor = mixFactor {
      let mix_factor = torch.sigmoid(
        state_dict[
          "diffusion_model.\(prefix).time_mixer.mix_factor"
        ].cpu()
      ).numpy()
      mixFactor.weight.copy(from: try! Tensor<Float>(numpy: mix_factor))
      print("\"diffusion_model.\(prefix).time_mixer.mix_factor\": [\"\(mixFactor.weight.name)\"],")
    }
  }
  return (reader, Model([x] + kvs, [out]))
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let emb = Input()
  var kvs = [Model.IO]()
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  let (
    timeInLayerNorm, timeInLayerConv2d, timeEmbLayer, timeOutLayerNorm, timeOutLayerConv2d,
    timeResBlock
  ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels)
  let mixFactor = Parameter<Float>(.GPU(1), .C(1), name: "time_mixer")
  out = mixFactor .* out + (1 - mixFactor) .* timeResBlock(out, emb)
  var transformerReader: ((PythonObject) -> Void)? = nil
  if attentionBlock > 0 {
    let c: [Input]
    if embeddingSize == 1 {
      c = (0..<(attentionBlock * 2 + 1)).map { _ in Input() }
    } else {
      c = (0..<(attentionBlock * 4 + 1)).map { _ in Input() }
    }
    let transformer: Model
    (
      transformerReader, transformer
    ) = SpatialTransformer(
      prefix: "\(prefix).\(layerStart).1",
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let in_layers_0_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.0.weight"
    ].cpu().numpy()
    let in_layers_0_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.0.bias"
    ].cpu().numpy()
    inLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_0_weight))
    inLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.in_layers.0.weight\": [\"\(inLayerNorm.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.in_layers.0.bias\": [\"\(inLayerNorm.bias.name)\"],"
    )
    let in_layers_2_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.2.weight"
    ].cpu().numpy()
    let in_layers_2_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.2.bias"
    ].cpu().numpy()
    inLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_2_weight))
    inLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.in_layers.2.weight\": [\"\(inLayerConv2d.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.in_layers.2.bias\": [\"\(inLayerConv2d.bias.name)\"],"
    )
    let emb_layers_1_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.weight"
    ].cpu().numpy()
    let emb_layers_1_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.bias"
    ].cpu().numpy()
    embLayer.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_1_weight))
    embLayer.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_1_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.weight\": [\"\(embLayer.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.bias\": [\"\(embLayer.bias.name)\"],"
    )
    let out_layers_0_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.0.weight"
    ].cpu().numpy()
    let out_layers_0_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.0.bias"
    ].cpu().numpy()
    outLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_layers_0_weight))
    outLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.out_layers.0.weight\": [\"\(outLayerNorm.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.out_layers.0.bias\": [\"\(outLayerNorm.bias.name)\"],"
    )
    let out_layers_3_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.3.weight"
    ].cpu().numpy()
    let out_layers_3_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.3.bias"
    ].cpu().numpy()
    outLayerConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_3_weight))
    outLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_3_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.out_layers.3.weight\": [\"\(outLayerConv2d.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.out_layers.3.bias\": [\"\(outLayerConv2d.bias.name)\"],"
    )
    if let skipModel = skipModel {
      let skip_connection_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).0.skip_connection.weight"
      ].cpu().numpy()
      let skip_connection_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).0.skip_connection.bias"
      ].cpu().numpy()
      skipModel.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: skip_connection_weight))
      skipModel.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: skip_connection_bias))
      print(
        "\"diffusion_model.\(prefix).\(layerStart).0.skip_connection.weight\": [\"\(skipModel.weight.name)\"],"
      )
      print(
        "\"diffusion_model.\(prefix).\(layerStart).0.skip_connection.bias\": [\"\(skipModel.bias.name)\"],"
      )
    }
    let time_stack_in_layers_0_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.in_layers.0.weight"
    ].cpu().numpy()
    let time_stack_in_layers_0_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.in_layers.0.bias"
    ].cpu().numpy()
    timeInLayerNorm.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_0_weight))
    timeInLayerNorm.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_0_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.in_layers.0.weight\": [\"\(timeInLayerNorm.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.in_layers.0.bias\": [\"\(timeInLayerNorm.bias.name)\"],"
    )
    let time_stack_in_layers_2_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.in_layers.2.weight"
    ].cpu().numpy()
    let time_stack_in_layers_2_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.in_layers.2.bias"
    ].cpu().numpy()
    timeInLayerConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_2_weight))
    timeInLayerConv2d.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_2_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.in_layers.2.weight\": [\"\(timeInLayerConv2d.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.in_layers.2.bias\": [\"\(timeInLayerConv2d.bias.name)\"],"
    )
    let time_stack_emb_layers_1_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.emb_layers.1.weight"
    ].cpu().numpy()
    let time_stack_emb_layers_1_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.emb_layers.1.bias"
    ].cpu().numpy()
    timeEmbLayer.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_emb_layers_1_weight))
    timeEmbLayer.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_emb_layers_1_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.emb_layers.1.weight\": [\"\(timeEmbLayer.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.emb_layers.1.bias\": [\"\(timeEmbLayer.bias.name)\"],"
    )
    let time_stack_out_layers_0_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.out_layers.0.weight"
    ].cpu().numpy()
    let time_stack_out_layers_0_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.out_layers.0.bias"
    ].cpu().numpy()
    timeOutLayerNorm.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_0_weight))
    timeOutLayerNorm.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_0_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.out_layers.0.weight\": [\"\(timeOutLayerNorm.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.out_layers.0.bias\": [\"\(timeOutLayerNorm.bias.name)\"],"
    )
    let time_stack_out_layers_3_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.out_layers.3.weight"
    ].cpu().numpy()
    let time_stack_out_layers_3_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.time_stack.out_layers.3.bias"
    ].cpu().numpy()
    timeOutLayerConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_3_weight))
    timeOutLayerConv2d.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_3_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.out_layers.3.weight\": [\"\(timeOutLayerConv2d.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_stack.out_layers.3.bias\": [\"\(timeOutLayerConv2d.bias.name)\"],"
    )
    let mix_factor = torch.sigmoid(
      state_dict[
        "diffusion_model.\(prefix).\(layerStart).0.time_mixer.mix_factor"
      ].cpu()
    ).numpy()
    mixFactor.weight.copy(from: try! Tensor<Float>(numpy: mix_factor))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.time_mixer.mix_factor\": [\"\(mixFactor.weight.name)\"],"
    )
    if let transformerReader = transformerReader {
      transformerReader(state_dict)
    }
  }
  return (reader, Model([x, emb] + kvs, [out]))
}

func MiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, x: Model.IO, emb: Model.IO
) -> ((PythonObject) -> Void, Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let (
    timeInLayerNorm1, timeInLayerConv2d1, timeEmbLayer1, timeOutLayerNorm1, timeOutLayerConv2d1,
    timeResBlock1
  ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels)
  let mixFactor1 = Parameter<Float>(.GPU(1), .C(1), name: "time_mixer")
  out = mixFactor1 .* out + (1 - mixFactor1) .* timeResBlock1(out, emb)
  let kvs: [Input]
  if embeddingSize == 1 {
    kvs = (0..<(attentionBlock * 2 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
  } else {
    kvs = (0..<(attentionBlock * 4 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
  }
  let (
    transformerReader, transformer
  ) = SpatialTransformer(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  out = transformer([out] + kvs)
  let (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  let (
    timeInLayerNorm2, timeInLayerConv2d2, timeEmbLayer2, timeOutLayerNorm2, timeOutLayerConv2d2,
    timeResBlock2
  ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels)
  let mixFactor2 = Parameter<Float>(.GPU(1), .C(1), name: "time_mixer")
  out = mixFactor2 .* out + (1 - mixFactor2) .* timeResBlock2(out, emb)
  let reader: (PythonObject) -> Void = { state_dict in
    let in_layers_0_0_weight = state_dict["diffusion_model.middle_block.0.in_layers.0.weight"]
      .cpu().numpy()
    let in_layers_0_0_bias = state_dict["diffusion_model.middle_block.0.in_layers.0.bias"].cpu()
      .numpy()
    inLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_0_0_weight))
    inLayerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_0_bias))
    print(
      "\"diffusion_model.middle_block.0.in_layers.0.weight\": [\"\(inLayerNorm1.weight.name)\"],")
    print("\"diffusion_model.middle_block.0.in_layers.0.bias\": [\"\(inLayerNorm1.bias.name)\"],")
    let in_layers_0_2_weight = state_dict["diffusion_model.middle_block.0.in_layers.2.weight"]
      .cpu().numpy()
    let in_layers_0_2_bias = state_dict["diffusion_model.middle_block.0.in_layers.2.bias"].cpu()
      .numpy()
    inLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_0_2_weight))
    inLayerConv2d1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_2_bias))
    print(
      "\"diffusion_model.middle_block.0.in_layers.2.weight\": [\"\(inLayerConv2d1.weight.name)\"],")
    print("\"diffusion_model.middle_block.0.in_layers.2.bias\": [\"\(inLayerConv2d1.bias.name)\"],")
    let emb_layers_0_1_weight = state_dict["diffusion_model.middle_block.0.emb_layers.1.weight"]
      .cpu().numpy()
    let emb_layers_0_1_bias = state_dict["diffusion_model.middle_block.0.emb_layers.1.bias"].cpu()
      .numpy()
    embLayer1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_0_1_weight))
    embLayer1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_0_1_bias))
    print("\"diffusion_model.middle_block.0.emb_layers.1.weight\": [\"\(embLayer1.weight.name)\"],")
    print("\"diffusion_model.middle_block.0.emb_layers.1.bias\": [\"\(embLayer1.bias.name)\"],")
    let out_layers_0_0_weight = state_dict["diffusion_model.middle_block.0.out_layers.0.weight"]
      .cpu().numpy()
    let out_layers_0_0_bias = state_dict[
      "diffusion_model.middle_block.0.out_layers.0.bias"
    ].cpu().numpy()
    outLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_0_weight))
    outLayerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_0_bias))
    print(
      "\"diffusion_model.middle_block.0.out_layers.0.weight\": [\"\(outLayerNorm1.weight.name)\"],")
    print("\"diffusion_model.middle_block.0.out_layers.0.bias\": [\"\(outLayerNorm1.bias.name)\"],")
    let out_layers_0_3_weight = state_dict["diffusion_model.middle_block.0.out_layers.3.weight"]
      .cpu().numpy()
    let out_layers_0_3_bias = state_dict["diffusion_model.middle_block.0.out_layers.3.bias"].cpu()
      .numpy()
    outLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_3_weight))
    outLayerConv2d1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_3_bias))
    print(
      "\"diffusion_model.middle_block.0.out_layers.3.weight\": [\"\(outLayerConv2d1.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.0.out_layers.3.bias\": [\"\(outLayerConv2d1.bias.name)\"],")
    transformerReader(state_dict)
    let in_layers_2_0_weight = state_dict["diffusion_model.middle_block.2.in_layers.0.weight"]
      .cpu().numpy()
    let in_layers_2_0_bias = state_dict["diffusion_model.middle_block.2.in_layers.0.bias"].cpu()
      .numpy()
    inLayerNorm2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_2_0_weight))
    inLayerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_0_bias))
    print(
      "\"diffusion_model.middle_block.2.in_layers.0.weight\": [\"\(inLayerNorm2.weight.name)\"],")
    print("\"diffusion_model.middle_block.2.in_layers.0.bias\": [\"\(inLayerNorm2.bias.name)\"],")
    let in_layers_2_2_weight = state_dict["diffusion_model.middle_block.2.in_layers.2.weight"]
      .cpu().numpy()
    let in_layers_2_2_bias = state_dict["diffusion_model.middle_block.2.in_layers.2.bias"].cpu()
      .numpy()
    inLayerConv2d2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_2_2_weight))
    inLayerConv2d2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_2_bias))
    print(
      "\"diffusion_model.middle_block.2.in_layers.2.weight\": [\"\(inLayerConv2d2.weight.name)\"],")
    print("\"diffusion_model.middle_block.2.in_layers.2.bias\": [\"\(inLayerConv2d2.bias.name)\"],")
    let emb_layers_2_1_weight = state_dict["diffusion_model.middle_block.2.emb_layers.1.weight"]
      .cpu().numpy()
    let emb_layers_2_1_bias = state_dict["diffusion_model.middle_block.2.emb_layers.1.bias"].cpu()
      .numpy()
    embLayer2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_2_1_weight))
    embLayer2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_2_1_bias))
    print("\"diffusion_model.middle_block.2.emb_layers.1.weight\": [\"\(embLayer2.weight.name)\"],")
    print("\"diffusion_model.middle_block.2.emb_layers.1.bias\": [\"\(embLayer2.bias.name)\"],")
    let out_layers_2_0_weight = state_dict["diffusion_model.middle_block.2.out_layers.0.weight"]
      .cpu().numpy()
    let out_layers_2_0_bias = state_dict[
      "diffusion_model.middle_block.2.out_layers.0.bias"
    ].cpu().numpy()
    outLayerNorm2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_0_weight))
    outLayerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_2_0_bias))
    print(
      "\"diffusion_model.middle_block.2.out_layers.0.weight\": [\"\(outLayerNorm2.weight.name)\"],")
    print("\"diffusion_model.middle_block.2.out_layers.0.bias\": [\"\(outLayerNorm2.bias.name)\"],")
    let out_layers_2_3_weight = state_dict["diffusion_model.middle_block.2.out_layers.3.weight"]
      .cpu().numpy()
    let out_layers_2_3_bias = state_dict["diffusion_model.middle_block.2.out_layers.3.bias"].cpu()
      .numpy()
    outLayerConv2d2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_3_weight))
    outLayerConv2d2.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_3_bias))
    print(
      "\"diffusion_model.middle_block.2.out_layers.3.weight\": [\"\(outLayerConv2d2.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.2.out_layers.3.bias\": [\"\(outLayerConv2d2.bias.name)\"],")
    let time_stack_in_layers_0_0_weight = state_dict[
      "diffusion_model.middle_block.0.time_stack.in_layers.0.weight"
    ].cpu().numpy()
    let time_stack_in_layers_0_0_bias = state_dict[
      "diffusion_model.middle_block.0.time_stack.in_layers.0.bias"
    ].cpu().numpy()
    timeInLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_0_0_weight))
    timeInLayerNorm1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_0_0_bias))
    print(
      "\"diffusion_model.middle_block.0.time_stack.in_layers.0.weight\": [\"\(timeInLayerNorm1.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.0.time_stack.in_layers.0.bias\": [\"\(timeInLayerNorm1.bias.name)\"],"
    )
    let time_stack_in_layers_0_2_weight = state_dict[
      "diffusion_model.middle_block.0.time_stack.in_layers.2.weight"
    ].cpu().numpy()
    let time_stack_in_layers_0_2_bias = state_dict[
      "diffusion_model.middle_block.0.time_stack.in_layers.2.bias"
    ].cpu().numpy()
    timeInLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_0_2_weight))
    timeInLayerConv2d1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_0_2_bias))
    print(
      "\"diffusion_model.middle_block.0.time_stack.in_layers.2.weight\": [\"\(timeInLayerConv2d1.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.0.time_stack.in_layers.2.bias\": [\"\(timeInLayerConv2d1.bias.name)\"],"
    )
    let time_stack_emb_layers_0_1_weight = state_dict[
      "diffusion_model.middle_block.0.time_stack.emb_layers.1.weight"
    ].cpu().numpy()
    let time_stack_emb_layers_0_1_bias = state_dict[
      "diffusion_model.middle_block.0.time_stack.emb_layers.1.bias"
    ].cpu().numpy()
    timeEmbLayer1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_emb_layers_0_1_weight))
    timeEmbLayer1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_emb_layers_0_1_bias))
    print(
      "\"diffusion_model.middle_block.0.time_stack.emb_layers.1.weight\": [\"\(timeEmbLayer1.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.0.time_stack.emb_layers.1.bias\": [\"\(timeEmbLayer1.bias.name)\"],"
    )
    let time_stack_out_layers_0_0_weight = state_dict[
      "diffusion_model.middle_block.0.time_stack.out_layers.0.weight"
    ].cpu().numpy()
    let time_stack_out_layers_0_0_bias = state_dict[
      "diffusion_model.middle_block.0.time_stack.out_layers.0.bias"
    ].cpu().numpy()
    timeOutLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_0_0_weight))
    timeOutLayerNorm1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_0_0_bias))
    print(
      "\"diffusion_model.middle_block.0.time_stack.out_layers.0.weight\": [\"\(timeOutLayerNorm1.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.0.time_stack.out_layers.0.bias\": [\"\(timeOutLayerNorm1.bias.name)\"],"
    )
    let time_stack_out_layers_0_3_weight = state_dict[
      "diffusion_model.middle_block.0.time_stack.out_layers.3.weight"
    ].cpu().numpy()
    let time_stack_out_layers_0_3_bias = state_dict[
      "diffusion_model.middle_block.0.time_stack.out_layers.3.bias"
    ].cpu().numpy()
    timeOutLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_0_3_weight))
    timeOutLayerConv2d1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_0_3_bias))
    print(
      "\"diffusion_model.middle_block.0.time_stack.out_layers.3.weight\": [\"\(timeOutLayerConv2d1.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.0.time_stack.out_layers.3.bias\": [\"\(timeOutLayerConv2d1.bias.name)\"],"
    )
    let time_stack_in_layers_2_0_weight = state_dict[
      "diffusion_model.middle_block.2.time_stack.in_layers.0.weight"
    ].cpu().numpy()
    let time_stack_in_layers_2_0_bias = state_dict[
      "diffusion_model.middle_block.2.time_stack.in_layers.0.bias"
    ].cpu().numpy()
    timeInLayerNorm2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_2_0_weight))
    timeInLayerNorm2.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_2_0_bias))
    print(
      "\"diffusion_model.middle_block.2.time_stack.in_layers.0.weight\": [\"\(timeInLayerNorm2.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.2.time_stack.in_layers.0.bias\": [\"\(timeInLayerNorm2.bias.name)\"],"
    )
    let time_stack_in_layers_2_2_weight = state_dict[
      "diffusion_model.middle_block.2.time_stack.in_layers.2.weight"
    ].cpu().numpy()
    let time_stack_in_layers_2_2_bias = state_dict[
      "diffusion_model.middle_block.2.time_stack.in_layers.2.bias"
    ].cpu().numpy()
    timeInLayerConv2d2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_2_2_weight))
    timeInLayerConv2d2.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_in_layers_2_2_bias))
    print(
      "\"diffusion_model.middle_block.2.time_stack.in_layers.2.weight\": [\"\(timeInLayerConv2d2.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.2.time_stack.in_layers.2.bias\": [\"\(timeInLayerConv2d2.bias.name)\"],"
    )
    let time_stack_emb_layers_2_1_weight = state_dict[
      "diffusion_model.middle_block.2.time_stack.emb_layers.1.weight"
    ].cpu().numpy()
    let time_stack_emb_layers_2_1_bias = state_dict[
      "diffusion_model.middle_block.2.time_stack.emb_layers.1.bias"
    ].cpu().numpy()
    timeEmbLayer2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_emb_layers_2_1_weight))
    timeEmbLayer2.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_emb_layers_2_1_bias))
    print(
      "\"diffusion_model.middle_block.2.time_stack.emb_layers.1.weight\": [\"\(timeEmbLayer2.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.2.time_stack.emb_layers.1.bias\": [\"\(timeEmbLayer2.bias.name)\"],"
    )
    let time_stack_out_layers_2_0_weight = state_dict[
      "diffusion_model.middle_block.2.time_stack.out_layers.0.weight"
    ].cpu().numpy()
    let time_stack_out_layers_2_0_bias = state_dict[
      "diffusion_model.middle_block.2.time_stack.out_layers.0.bias"
    ].cpu().numpy()
    timeOutLayerNorm2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_2_0_weight))
    timeOutLayerNorm2.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_2_0_bias))
    print(
      "\"diffusion_model.middle_block.2.time_stack.out_layers.0.weight\": [\"\(timeOutLayerNorm2.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.2.time_stack.out_layers.0.bias\": [\"\(timeOutLayerNorm2.bias.name)\"],"
    )
    let time_stack_out_layers_2_3_weight = state_dict[
      "diffusion_model.middle_block.2.time_stack.out_layers.3.weight"
    ].cpu().numpy()
    let time_stack_out_layers_2_3_bias = state_dict[
      "diffusion_model.middle_block.2.time_stack.out_layers.3.bias"
    ].cpu().numpy()
    timeOutLayerConv2d2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_2_3_weight))
    timeOutLayerConv2d2.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: time_stack_out_layers_2_3_bias))
    print(
      "\"diffusion_model.middle_block.2.time_stack.out_layers.3.weight\": [\"\(timeOutLayerConv2d2.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.2.time_stack.out_layers.3.bias\": [\"\(timeOutLayerConv2d2.bias.name)\"],"
    )
    let mix_factor_0 = torch.sigmoid(
      state_dict[
        "diffusion_model.middle_block.0.time_mixer.mix_factor"
      ].cpu()
    ).numpy()
    mixFactor1.weight.copy(from: try! Tensor<Float>(numpy: mix_factor_0))
    print(
      "\"diffusion_model.middle_block.0.time_mixer.mix_factor\": [\"\(mixFactor1.weight.name)\"],")
    let mix_factor_2 = torch.sigmoid(
      state_dict[
        "diffusion_model.middle_block.2.time_mixer.mix_factor"
      ].cpu()
    ).numpy()
    mixFactor2.weight.copy(from: try! Tensor<Float>(numpy: mix_factor_2))
    print(
      "\"diffusion_model.middle_block.2.time_mixer.mix_factor\": [\"\(mixFactor2.weight.name)\"],")
  }
  return (reader, out, kvs)
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO
) -> ((PythonObject) -> Void, [Model.IO], Model.IO, [Input]) {
  let conv2d = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      let (reader, inputLayer) = BlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      let c: [Input]
      if embeddingSize == 1 {
        c = (0..<(attentionBlock * 2 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
      } else {
        c = (0..<(attentionBlock * 4 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
      }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      passLayers.append(out)
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let reader: (PythonObject) -> Void = { state_dict in
        let op_weight = state_dict["diffusion_model.input_blocks.\(downLayer).0.op.weight"].cpu()
          .numpy()
        let op_bias = state_dict["diffusion_model.input_blocks.\(downLayer).0.op.bias"].cpu()
          .numpy()
        downsample.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: op_weight))
        downsample.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: op_bias))
        print(
          "\"diffusion_model.input_blocks.\(downLayer).0.op.weight\": [\"\(downsample.weight.name)\"],"
        )
        print(
          "\"diffusion_model.input_blocks.\(downLayer).0.op.bias\": [\"\(downsample.bias.name)\"],")
      }
      readers.append(reader)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let input_blocks_0_0_weight = state_dict["diffusion_model.input_blocks.0.0.weight"].cpu()
      .numpy()
    let input_blocks_0_0_bias = state_dict["diffusion_model.input_blocks.0.0.bias"].cpu().numpy()
    conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_weight))
    conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_bias))
    print("\"diffusion_model.input_blocks.0.0.weight\": [\"\(conv2d.weight.name)\"],")
    print("\"diffusion_model.input_blocks.0.0.bias\": [\"\(conv2d.bias.name)\"],")
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, passLayers, out, kvs)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO,
  inputs: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO, [Input]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let (reader, outputLayer) = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      let c: [Input]
      if embeddingSize == 1 {
        c = (0..<(attentionBlock * 2 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
      } else {
        c = (0..<(attentionBlock * 4 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
      }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      readers.append(reader)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock > 0 ? 2 : 1
        let reader: (PythonObject) -> Void = { state_dict in
          let op_weight = state_dict[
            "diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight"
          ].cpu().numpy()
          let op_bias = state_dict["diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias"]
            .cpu().numpy()
          conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: op_weight))
          conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: op_bias))
          print(
            "\"diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight\": [\"\(conv2d.weight.name)\"],"
          )
          print(
            "\"diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias\": [\"\(conv2d.bias.name)\"],"
          )
        }
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out, kvs)
}

func UNetXL(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  attentionRes: KeyValuePairs<Int, Int>
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let (timeFc0, timeFc2, timeEmbed) = TimeEmbed(modelChannels: channels[0])
  let (labelFc0, labelFc2, labelEmbed) = LabelEmbed(modelChannels: channels[0])
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (inputReader, inputs, inputBlocks, inputKVs) = InputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 1, attentionRes: attentionRes,
    x: x, emb: emb)
  var out = inputBlocks
  let (middleReader, middleBlock, middleKVs) = MiddleBlock(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 1, attentionBlock: middleBlockAttentionBlock, x: out, emb: emb)
  out = middleBlock
  let (outputReader, outputBlocks, outputKVs) = OutputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 1, attentionRes: attentionRes,
    x: out, emb: emb, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let time_embed_0_weight = state_dict["diffusion_model.time_embed.0.weight"].cpu().numpy()
    let time_embed_0_bias = state_dict["diffusion_model.time_embed.0.bias"].cpu().numpy()
    let time_embed_2_weight = state_dict["diffusion_model.time_embed.2.weight"].cpu().numpy()
    let time_embed_2_bias = state_dict["diffusion_model.time_embed.2.bias"].cpu().numpy()
    timeFc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_0_weight))
    timeFc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_0_bias))
    print("\"diffusion_model.time_embed.0.weight\": [\"\(timeFc0.weight.name)\"],")
    print("\"diffusion_model.time_embed.0.bias\": [\"\(timeFc0.bias.name)\"],")
    timeFc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_2_weight))
    timeFc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_2_bias))
    print("\"diffusion_model.time_embed.2.weight\": [\"\(timeFc2.weight.name)\"],")
    print("\"diffusion_model.time_embed.2.bias\": [\"\(timeFc2.bias.name)\"],")
    let label_emb_0_0_weight = state_dict["diffusion_model.label_emb.0.0.weight"].cpu().numpy()
    let label_emb_0_0_bias = state_dict["diffusion_model.label_emb.0.0.bias"].cpu().numpy()
    let label_emb_0_2_weight = state_dict["diffusion_model.label_emb.0.2.weight"].cpu().numpy()
    let label_emb_0_2_bias = state_dict["diffusion_model.label_emb.0.2.bias"].cpu().numpy()
    labelFc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: label_emb_0_0_weight))
    labelFc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: label_emb_0_0_bias))
    print("\"diffusion_model.label_emb.0.0.weight\": [\"\(labelFc0.weight.name)\"],")
    print("\"diffusion_model.label_emb.0.0.bias\": [\"\(labelFc0.bias.name)\"],")
    labelFc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: label_emb_0_2_weight))
    labelFc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: label_emb_0_2_bias))
    print("\"diffusion_model.label_emb.0.2.weight\": [\"\(labelFc2.weight.name)\"],")
    print("\"diffusion_model.label_emb.0.2.bias\": [\"\(labelFc2.bias.name)\"],")
    inputReader(state_dict)
    middleReader(state_dict)
    outputReader(state_dict)
    let out_0_weight = state_dict["diffusion_model.out.0.weight"].cpu().numpy()
    let out_0_bias = state_dict["diffusion_model.out.0.bias"].cpu().numpy()
    outNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_0_weight))
    outNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_0_bias))
    print("\"diffusion_model.out.0.weight\": [\"\(outNorm.weight.name)\"],")
    print("\"diffusion_model.out.0.bias\": [\"\(outNorm.bias.name)\"],")
    let out_2_weight = state_dict["diffusion_model.out.2.weight"].cpu().numpy()
    let out_2_bias = state_dict["diffusion_model.out.2.bias"].cpu().numpy()
    outConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_2_weight))
    outConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_2_bias))
    print("\"diffusion_model.out.2.weight\": [\"\(outConv2d.weight.name)\"],")
    print("\"diffusion_model.out.2.bias\": [\"\(outConv2d.bias.name)\"],")
  }
  return (reader, Model([x, t_emb, y] + inputKVs + middleKVs + outputKVs, [out]))
}

func CrossAttentionFixed(k: Int, h: Int, b: Int, t: Int, name: String = "") -> (Model, Model, Model)
{
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
  return (tokeys, tovalues, Model([c], [keys, values], name: name))
}

func Attention1Fixed(k: Int, h: Int, b: Int, t: Int, name: String = "") -> (Model, Model, Model) {
  let c = Input()
  let tovalues = Dense(count: k * h, noBias: true)
  let unifyheads = Dense(count: k * h)
  let values = unifyheads(tovalues(c))
  return (tovalues, unifyheads, Model([c], [values], name: name))
}

func BasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  if t == 1 {
    let (tovalues2, unifyheads2, attn2) = Attention1Fixed(k: k, h: h, b: b, t: t)
    let reader: (PythonObject) -> Void = { state_dict in
      // print("\"diffusion_model.\(prefix).attn2.to_k.weight\": [\"\(tokeys2.weight.name)\"],")
      let attn2_to_v_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_v.weight"
      ].cpu().numpy()
      tovalues2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
      // print("\"diffusion_model.\(prefix).attn2.to_v.weight\": [\"\(tovalues2.weight.name)\"],")
      let attn2_to_out_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_out.0.weight"
      ].cpu().numpy()
      let attn2_to_out_bias = state_dict[
        "diffusion_model.\(prefix).attn2.to_out.0.bias"
      ].cpu().numpy()
      unifyheads2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: attn2_to_out_weight))
      unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
      // print("\"diffusion_model.\(prefix).attn2.to_out.0.weight\": [\"\(unifyheads2.weight.name)\"],")
      // print("\"diffusion_model.\(prefix).attn2.to_out.0.bias\": [\"\(unifyheads2.bias.name)\"],")
    }
    return (reader, attn2)
  } else {
    let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(k: k, h: h, b: b, t: t)
    let reader: (PythonObject) -> Void = { state_dict in
      let attn2_to_k_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_k.weight"
      ].cpu().numpy()
      tokeys2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_k_weight))
      // print("\"diffusion_model.\(prefix).attn2.to_k.weight\": [\"\(tokeys2.weight.name)\"],")
      let attn2_to_v_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_v.weight"
      ].cpu().numpy()
      tovalues2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
      // print("\"diffusion_model.\(prefix).attn2.to_v.weight\": [\"\(tovalues2.weight.name)\"],")
    }
    return (reader, attn2)
  }
}

func TimePosEmbedTransformerBlockFixed(
  prefix: String, k: Int, h: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let (timePosFc0, timePosFc2, timePosEmbed) = TimePosEmbed(modelChannels: k * h)
  let reader: (PythonObject) -> Void = { state_dict in
    let time_pos_embed_0_weight = state_dict[
      "diffusion_model.\(prefix).time_pos_embed.0.weight"
    ].cpu().numpy()
    let time_pos_embed_0_bias = state_dict[
      "diffusion_model.\(prefix).time_pos_embed.0.bias"
    ].cpu().numpy()
    timePosFc0.weight.copy(from: try! Tensor<Float>(numpy: time_pos_embed_0_weight))
    timePosFc0.bias.copy(from: try! Tensor<Float>(numpy: time_pos_embed_0_bias))
    let time_pos_embed_2_weight = state_dict[
      "diffusion_model.\(prefix).time_pos_embed.2.weight"
    ].cpu().numpy()
    let time_pos_embed_2_bias = state_dict[
      "diffusion_model.\(prefix).time_pos_embed.2.bias"
    ].cpu().numpy()
    timePosFc2.weight.copy(from: try! Tensor<Float>(numpy: time_pos_embed_2_weight))
    timePosFc2.bias.copy(from: try! Tensor<Float>(numpy: time_pos_embed_2_bias))
  }
  return (reader, timePosEmbed)
}

func BasicTimeTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  if t == 1 {
    let (tovalues2, unifyheads2, attn2) = Attention1Fixed(
      k: k, h: h, b: b, t: t, name: "time_stack")
    let reader: (PythonObject) -> Void = { state_dict in
      // print("\"diffusion_model.\(prefix).attn2.to_k.weight\": [\"\(tokeys2.weight.name)\"],")
      let attn2_to_v_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_v.weight"
      ].cpu().numpy()
      tovalues2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
      // print("\"diffusion_model.\(prefix).attn2.to_v.weight\": [\"\(tovalues2.weight.name)\"],")
      let attn2_to_out_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_out.0.weight"
      ].cpu().numpy()
      let attn2_to_out_bias = state_dict[
        "diffusion_model.\(prefix).attn2.to_out.0.bias"
      ].cpu().numpy()
      unifyheads2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: attn2_to_out_weight))
      unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
      // print("\"diffusion_model.\(prefix).attn2.to_out.0.weight\": [\"\(unifyheads2.weight.name)\"],")
      // print("\"diffusion_model.\(prefix).attn2.to_out.0.bias\": [\"\(unifyheads2.bias.name)\"],")
    }
    return (reader, attn2)
  } else {
    let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(
      k: k, h: h, b: b, t: t, name: "time_stack")
    let reader: (PythonObject) -> Void = { state_dict in
      let attn2_to_k_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_k.weight"
      ].cpu().numpy()
      tokeys2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_k_weight))
      // print("\"diffusion_model.\(prefix).attn2.to_k.weight\": [\"\(tokeys2.weight.name)\"],")
      let attn2_to_v_weight = state_dict[
        "diffusion_model.\(prefix).attn2.to_v.weight"
      ].cpu().numpy()
      tovalues2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
      // print("\"diffusion_model.\(prefix).attn2.to_v.weight\": [\"\(tovalues2.weight.name)\"],")
    }
    return (reader, attn2)
  }
}

func SpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  let numFrames = Input()
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  let hw = height * width
  let (timePosEmbedReader, timePosEmbed) = TimePosEmbedTransformerBlockFixed(
    prefix: prefix, k: k, h: h)
  outs.append(timePosEmbed(numFrames))
  readers.append(timePosEmbedReader)
  for i in 0..<depth {
    let (reader, block) = BasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(block(c))
    readers.append(reader)
    let (timeReader, timeBlock) = BasicTimeTransformerBlockFixed(
      prefix: "\(prefix).time_stack.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(timeBlock(c))
    readers.append(timeReader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([c, numFrames], outs))
}

func BlockLayerFixed(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (transformerReader, transformer) = SpatialTransformerFixed(
    prefix: "\(prefix).\(layerStart).1",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingSize,
    intermediateSize: channels * 4)
  return (transformerReader, transformer)
}

func MiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, c: Model.IO, numFrames: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (
    transformerReader, transformer
  ) = SpatialTransformerFixed(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  let out = transformer(c, numFrames[numFrames.count - 1])
  return (transformerReader, out)
}

func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO, numFrames: [Model.IO]
) -> ((PythonObject) -> Void, [Model.IO]) {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      if attentionBlock > 0 {
        let (reader, inputLayer) = BlockLayerFixed(
          prefix: "input_blocks",
          layerStart: layerStart, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        previousChannel = channel
        outs.append(inputLayer(c, numFrames[i]))
        readers.append(reader)
      }
      layerStart += 1
    }
    if i != channels.count - 1 {
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, outs)
}

func OutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO, numFrames: [Model.IO]
) -> ((PythonObject) -> Void, [Model.IO]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<(numRepeat + 1) {
      if attentionBlock > 0 {
        let (reader, outputLayer) = BlockLayerFixed(
          prefix: "output_blocks",
          layerStart: layerStart, skipConnection: true,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        outs.append(outputLayer(c, numFrames[i]))
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, outs)
}

func UNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  attentionRes: KeyValuePairs<Int, Int>
) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let numFrames = (0..<channels.count).map { _ in Input() }
  let (inputReader, inputBlocks) = InputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 1, attentionRes: attentionRes,
    c: c, numFrames: numFrames)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (middleReader, middleBlock) = MiddleBlockFixed(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 1, attentionBlock: middleBlockAttentionBlock, c: c, numFrames: numFrames)
  out.append(middleBlock)
  let (outputReader, outputBlocks) = OutputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 1, attentionRes: attentionRes,
    c: c, numFrames: numFrames)
  out.append(contentsOf: outputBlocks)
  let reader: (PythonObject) -> Void = { state_dict in
    inputReader(state_dict)
    middleReader(state_dict)
    outputReader(state_dict)
  }
  return (reader, Model([c] + numFrames, out))
}

let embedders_0_state_dict = state["model"].conditioner.embedders[0].state_dict()
let embedders_3_state_dict = state["model"].conditioner.embedders[3].encoder.state_dict()
let model_state_dict = state["model"].model.state_dict()
let x = torch.randn([14, 8, 64, 64]).cuda()
let timesteps = 0.9771 * torch.ones([14]).cuda()
let context = torch.randn([1, 1, 1024]).cuda()
let y = torch.randn([1, 768]).cuda()
let context14 = context.repeat([14, 1, 1])
let y14 = y.repeat([14, 1])
let image_only_indicator = torch.zeros([1, 14]).cuda()
torch.set_grad_enabled(false)
let out = state["model"].model.diffusion_model(
  x, timesteps, context14, y14, num_video_frames: 14, image_only_indicator: image_only_indicator)
print(out)
let graph = DynamicGraph()
graph.withNoGrad {
  let (_, vit) = VisionTransformer(
    grid: 16, width: 1280, outputDim: 1024, layers: 32, heads: 16, batchSize: 1)
  let clip_image = clip_image.type(torch.float).cpu().numpy()
  let clipImageTensor = graph.variable(try! Tensor<Float>(numpy: clip_image)).toGPU(1)
  vit.compile(inputs: clipImageTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_vit_h14_vision_model_f32.ckpt") {
    $0.read("vision_model", model: vit)
  }
  // reader(embedders_0_state_dict)
  let imageEmbeds = vit(inputs: clipImageTensor)[0].as(of: Float.self).reshaped(.CHW(1, 1, 1280))
  let visualProj = Dense(count: 1024, noBias: true)
  let visual_proj = embedders_0_state_dict["open_clip.model.visual.proj"].type(torch.float).T.cpu()
    .numpy()
  visualProj.compile(inputs: imageEmbeds)
  visualProj.weight.copy(from: try! Tensor<Float>(numpy: visual_proj))
  let imageProj = visualProj(inputs: imageEmbeds)[0].as(of: Float.self)
  debugPrint(imageProj)
  let (_, encoder) = Encoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64, startHeight: 64)
  let image = img.type(torch.float).cpu().numpy()
  let imageTensor = graph.variable(try! Tensor<Float>(numpy: image)).reshaped(.NCHW(1, 3, 512, 512))
    .toGPU(1)
  encoder.compile(inputs: imageTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/vae_ft_mse_840000_f16.ckpt") {
    $0.read("encoder", model: encoder)
  }
  // reader(embedders_3_state_dict)
  let parameters = encoder(inputs: imageTensor)[0].as(of: Float.self)
  let mean = parameters[0..<1, 0..<4, 0..<64, 0..<64].copied()
  debugPrint(mean)
  let fpsId = timeEmbedding(timestep: 5, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let motionBucketId = timeEmbedding(
    timestep: 127, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let condAug = timeEmbedding(timestep: 0.02, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let vector = Concat(axis: 1)(
    inputs: graph.variable(fpsId), graph.variable(motionBucketId), graph.variable(condAug))[0].as(
      of: Float.self
    ).toGPU(1)
  debugPrint(vector)
  let x = graph.variable(try! Tensor<Float>(numpy: x.type(torch.float).cpu().numpy())).toGPU(1)
  let context = graph.variable(try! Tensor<Float>(numpy: context.type(torch.float).cpu().numpy()))
    .toGPU(1)
  let y = graph.variable(try! Tensor<Float>(numpy: y.type(torch.float).cpu().numpy())).toGPU(1)
  let t_emb = graph.variable(
    timeEmbedding(timestep: 0.9771, batchSize: 1, embeddingSize: 320, maxPeriod: 10_000)
  ).toGPU(1)
  let numFramesEmb = [320, 640, 1280, 1280].map { embeddingSize in
    let tensors = (0..<14).map {
      graph.variable(
        timeEmbedding(
          timestep: Float($0), batchSize: 1, embeddingSize: embeddingSize, maxPeriod: 10_000)
      ).toGPU(1)
    }
    return Concat(axis: 0)(inputs: tensors[0], Array(tensors[1...]))[0].as(of: Float.self)
  }
  let (readerFixed, unetFixed) = UNetXLFixed(
    batchSize: 1, startHeight: 64, startWidth: 64, channels: [320, 640, 1280, 1280],
    attentionRes: [1: 1, 2: 1, 4: 1])
  unetFixed.compile(inputs: [context] + numFramesEmb)
  readerFixed(model_state_dict)
  let kvs = unetFixed(inputs: context, numFramesEmb).map { $0.as(of: Float.self) }
  let (reader, unet) = UNetXL(
    batchSize: 14, startHeight: 64, startWidth: 64, channels: [320, 640, 1280, 1280],
    attentionRes: [1: 1, 2: 1, 4: 1])
  unet.compile(inputs: [x, t_emb, y] + kvs)
  reader(model_state_dict)
  let pred = unet(inputs: x, [t_emb, y] + kvs)
  debugPrint(pred)
  graph.openStore("/home/liu/workspace/swift-diffusion/animatelcm_svd_i2v_xt_1.1_f32.ckpt") {
    $0.write("visual_proj", model: visualProj)
    $0.write("unet_fixed", model: unetFixed)
    $0.write("unet", model: unet)
  }
}
