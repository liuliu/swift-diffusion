import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let torch = Python.import("torch")
let numpy = Python.import("numpy")
/*
let models = Python.import("extensions.sd-forge-layerdiffuse.lib_layerdiffusion.models")
let ldm_patched_modules_utils = Python.import("ldm_patched.modules.utils")

let sd = ldm_patched_modules_utils.load_torch_file(
  "/home/liu/workspace/Flux-version-LayerDiffuse/models/TransparentVAE.pth"
)

let vae_transparent_decoder = models.TransparentVAEDecoder(sd)
*/
let diffusers = Python.import("diffusers")
let lib_layerdiffuse_vae = Python.import("lib_layerdiffuse.vae")
let pipe = diffusers.FluxPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev", torch_dtype: torch.bfloat16)
let trans_vae = lib_layerdiffuse_vae.TransparentVAE(pipe.vae, pipe.vae.dtype)
trans_vae.load_state_dict(
  torch.load("/home/liu/workspace/Flux-version-LayerDiffuse/models/TransparentVAE.pth"),
  strict: false)
print(trans_vae.decoder)

let random = Python.import("random")
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let pixel = torch.randn([1, 3, 1024, 1024])
let latent = torch.randn([1, 16, 128, 128])

// let model = vae_transparent_decoder.model.model.float().cuda()
let model = trans_vae.decoder.float().cuda()
let out = model(pixel.cuda(), latent.cuda())
let state_dict = model.state_dict()
print(out)

func ResBlock(b: Int, groups: Int, outChannels: Int, skipConnection: Bool) -> (
  Model, Model, Model, Model, Model?, Model
) {
  let x = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: groups, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let outLayerNorm = GroupNorm(axis: 1, groups: groups, epsilon: 1e-5, reduce: [2, 3])
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
    inLayerNorm, inLayerConv2d, outLayerNorm, outLayerConv2d, skipModel, Model([x], [out])
  )
}

private func SelfAttention(k: Int, h: Int, b: Int, hw: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h, name: "to_k")
  let toqueries = Dense(count: k * h, name: "to_q")
  let tovalues = Dense(count: k * h, name: "to_v")
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
  let unifyheads = Dense(count: k * h, name: "to_out")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func DownBlock2D(
  prefix: String, inChannels: Int, outChannels: Int, numRepeat: Int, batchSize: Int, x: Model.IO
) -> ((PythonObject) -> Void, Model.IO, [Model.IO]) {
  var out: Model.IO = x
  var readers = [(PythonObject) -> Void]()
  var inChannels = inChannels
  var hiddenStates = [Model.IO]()
  for i in 0..<numRepeat {
    let (inLayerNorm, inLayerConv2d, outLayerNorm, outLayerConv2d, skipModel, resBlock) = ResBlock(
      b: batchSize, groups: 4, outChannels: outChannels, skipConnection: inChannels != outChannels)
    out = resBlock(out)
    hiddenStates.append(out)
    inChannels = outChannels
    readers.append { state_dict in
      let norm1_weight = state_dict["\(prefix).resnets.\(i).norm1.weight"].type(torch.float).cpu()
        .numpy()
      let norm1_bias = state_dict["\(prefix).resnets.\(i).norm1.bias"].type(torch.float).cpu()
        .numpy()
      inLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
      inLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
      let conv1_weight = state_dict["\(prefix).resnets.\(i).conv1.weight"].type(torch.float).cpu()
        .numpy()
      let conv1_bias = state_dict["\(prefix).resnets.\(i).conv1.bias"].type(torch.float).cpu()
        .numpy()
      inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
      inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
      let norm2_weight = state_dict["\(prefix).resnets.\(i).norm2.weight"].type(torch.float).cpu()
        .numpy()
      let norm2_bias = state_dict["\(prefix).resnets.\(i).norm2.bias"].type(torch.float).cpu()
        .numpy()
      outLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
      outLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
      let conv2_weight = state_dict["\(prefix).resnets.\(i).conv2.weight"].type(torch.float).cpu()
        .numpy()
      let conv2_bias = state_dict["\(prefix).resnets.\(i).conv2.bias"].type(torch.float).cpu()
        .numpy()
      outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
      outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
      if let skipModel = skipModel {
        let conv_shortcut_weight = state_dict["\(prefix).resnets.\(i).conv_shortcut.weight"].type(
          torch.float
        ).cpu().numpy()
        let conv_shortcut_bias = state_dict["\(prefix).resnets.\(i).conv_shortcut.bias"].type(
          torch.float
        ).cpu().numpy()
        skipModel.weight.copy(from: try! Tensor<Float>(numpy: conv_shortcut_weight))
        skipModel.bias.copy(from: try! Tensor<Float>(numpy: conv_shortcut_bias))
      }
    }
  }
  let downsample = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
  readers.append { state_dict in
    let downsamplers_weight = state_dict["\(prefix).downsamplers.0.conv.weight"].type(torch.float)
      .cpu().numpy()
    let downsamplers_bias = state_dict["\(prefix).downsamplers.0.conv.bias"].type(torch.float).cpu()
      .numpy()
    downsample.weight.copy(from: try! Tensor<Float>(numpy: downsamplers_weight))
    downsample.bias.copy(from: try! Tensor<Float>(numpy: downsamplers_bias))
  }
  out = downsample(out)
  hiddenStates.append(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out, hiddenStates)
}

func AttnDownBlock2D(
  prefix: String, inChannels: Int, outChannels: Int, numRepeat: Int, batchSize: Int, numHeads: Int,
  startHeight: Int, startWidth: Int, downsample: Bool, x: Model.IO
) -> ((PythonObject) -> Void, Model.IO, [Model.IO]) {
  var out: Model.IO = x
  var readers = [(PythonObject) -> Void]()
  var inChannels = inChannels
  var hiddenStates = [Model.IO]()
  for i in 0..<numRepeat {
    let (inLayerNorm, inLayerConv2d, outLayerNorm, outLayerConv2d, skipModel, resBlock) = ResBlock(
      b: batchSize, groups: 4, outChannels: outChannels, skipConnection: inChannels != outChannels)
    out = resBlock(out)
    let norm = GroupNorm(axis: 1, groups: 4, epsilon: 1e-5, reduce: [2, 3])
    let residual = out
    out = norm(out)
    let (tokeys, toqueries, tovalues, unifyheads, attn) = SelfAttention(
      k: outChannels / numHeads, h: numHeads, b: batchSize, hw: startHeight * startWidth)
    out =
      attn(out.reshaped([batchSize, outChannels, startHeight * startWidth]).transposed(1, 2))
      .transposed(1, 2).reshaped([batchSize, outChannels, startHeight, startWidth]) + residual
    hiddenStates.append(out)
    inChannels = outChannels
    readers.append { state_dict in
      let norm1_weight = state_dict["\(prefix).resnets.\(i).norm1.weight"].type(torch.float).cpu()
        .numpy()
      let norm1_bias = state_dict["\(prefix).resnets.\(i).norm1.bias"].type(torch.float).cpu()
        .numpy()
      inLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
      inLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
      let conv1_weight = state_dict["\(prefix).resnets.\(i).conv1.weight"].type(torch.float).cpu()
        .numpy()
      let conv1_bias = state_dict["\(prefix).resnets.\(i).conv1.bias"].type(torch.float).cpu()
        .numpy()
      inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
      inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
      let norm2_weight = state_dict["\(prefix).resnets.\(i).norm2.weight"].type(torch.float).cpu()
        .numpy()
      let norm2_bias = state_dict["\(prefix).resnets.\(i).norm2.bias"].type(torch.float).cpu()
        .numpy()
      outLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
      outLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
      let conv2_weight = state_dict["\(prefix).resnets.\(i).conv2.weight"].type(torch.float).cpu()
        .numpy()
      let conv2_bias = state_dict["\(prefix).resnets.\(i).conv2.bias"].type(torch.float).cpu()
        .numpy()
      outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
      outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
      if let skipModel = skipModel {
        let conv_shortcut_weight = state_dict["\(prefix).resnets.\(i).conv_shortcut.weight"].type(
          torch.float
        ).cpu().numpy()
        let conv_shortcut_bias = state_dict["\(prefix).resnets.\(i).conv_shortcut.bias"].type(
          torch.float
        ).cpu().numpy()
        skipModel.weight.copy(from: try! Tensor<Float>(numpy: conv_shortcut_weight))
        skipModel.bias.copy(from: try! Tensor<Float>(numpy: conv_shortcut_bias))
      }
      let group_norm_weight = state_dict["\(prefix).attentions.\(i).group_norm.weight"].type(
        torch.float
      ).cpu().numpy()
      let group_norm_bias = state_dict["\(prefix).attentions.\(i).group_norm.bias"].type(
        torch.float
      ).cpu().numpy()
      norm.weight.copy(from: try! Tensor<Float>(numpy: group_norm_weight))
      norm.bias.copy(from: try! Tensor<Float>(numpy: group_norm_bias))
      let to_k_weight = state_dict["\(prefix).attentions.\(i).to_k.weight"].type(torch.float).cpu()
        .numpy()
      let to_k_bias = state_dict["\(prefix).attentions.\(i).to_k.bias"].type(torch.float).cpu()
        .numpy()
      tokeys.weight.copy(from: try! Tensor<Float>(numpy: to_k_weight))
      tokeys.bias.copy(from: try! Tensor<Float>(numpy: to_k_bias))
      let to_q_weight = state_dict["\(prefix).attentions.\(i).to_q.weight"].type(torch.float).cpu()
        .numpy()
      let to_q_bias = state_dict["\(prefix).attentions.\(i).to_q.bias"].type(torch.float).cpu()
        .numpy()
      toqueries.weight.copy(from: try! Tensor<Float>(numpy: to_q_weight))
      toqueries.bias.copy(from: try! Tensor<Float>(numpy: to_q_bias))
      let to_v_weight = state_dict["\(prefix).attentions.\(i).to_v.weight"].type(torch.float).cpu()
        .numpy()
      let to_v_bias = state_dict["\(prefix).attentions.\(i).to_v.bias"].type(torch.float).cpu()
        .numpy()
      tovalues.weight.copy(from: try! Tensor<Float>(numpy: to_v_weight))
      tovalues.bias.copy(from: try! Tensor<Float>(numpy: to_v_bias))
      let to_out_weight = state_dict["\(prefix).attentions.\(i).to_out.0.weight"].type(torch.float)
        .cpu().numpy()
      let to_out_bias = state_dict["\(prefix).attentions.\(i).to_out.0.bias"].type(torch.float)
        .cpu().numpy()
      unifyheads.weight.copy(from: try! Tensor<Float>(numpy: to_out_weight))
      unifyheads.bias.copy(from: try! Tensor<Float>(numpy: to_out_bias))
    }
  }
  if downsample {
    let downsample = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
    readers.append { state_dict in
      let downsamplers_weight = state_dict["\(prefix).downsamplers.0.conv.weight"].type(torch.float)
        .cpu().numpy()
      let downsamplers_bias = state_dict["\(prefix).downsamplers.0.conv.bias"].type(torch.float)
        .cpu().numpy()
      downsample.weight.copy(from: try! Tensor<Float>(numpy: downsamplers_weight))
      downsample.bias.copy(from: try! Tensor<Float>(numpy: downsamplers_bias))
    }
    out = downsample(out)
    hiddenStates.append(out)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out, hiddenStates)
}

func AttnUpBlock2D(
  prefix: String, channels: Int, numRepeat: Int, batchSize: Int, numHeads: Int, startHeight: Int,
  startWidth: Int, x: Model.IO, hiddenStates: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO) {
  var out: Model.IO = x
  var readers = [(PythonObject) -> Void]()
  for i in 0..<numRepeat {
    let (inLayerNorm, inLayerConv2d, outLayerNorm, outLayerConv2d, skipModel, resBlock) = ResBlock(
      b: batchSize, groups: 4, outChannels: channels, skipConnection: true)
    out = Functional.concat(axis: 1, out, hiddenStates[i])
    out = resBlock(out)
    let norm = GroupNorm(axis: 1, groups: 4, epsilon: 1e-5, reduce: [2, 3])
    let residual = out
    out = norm(out)
    let (tokeys, toqueries, tovalues, unifyheads, attn) = SelfAttention(
      k: channels / numHeads, h: numHeads, b: batchSize, hw: startHeight * startWidth)
    out =
      attn(out.reshaped([batchSize, channels, startHeight * startWidth]).transposed(1, 2))
      .transposed(1, 2).reshaped([batchSize, channels, startHeight, startWidth]) + residual
    readers.append { state_dict in
      let norm1_weight = state_dict["\(prefix).resnets.\(i).norm1.weight"].type(torch.float).cpu()
        .numpy()
      let norm1_bias = state_dict["\(prefix).resnets.\(i).norm1.bias"].type(torch.float).cpu()
        .numpy()
      inLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
      inLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
      let conv1_weight = state_dict["\(prefix).resnets.\(i).conv1.weight"].type(torch.float).cpu()
        .numpy()
      let conv1_bias = state_dict["\(prefix).resnets.\(i).conv1.bias"].type(torch.float).cpu()
        .numpy()
      inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
      inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
      let norm2_weight = state_dict["\(prefix).resnets.\(i).norm2.weight"].type(torch.float).cpu()
        .numpy()
      let norm2_bias = state_dict["\(prefix).resnets.\(i).norm2.bias"].type(torch.float).cpu()
        .numpy()
      outLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
      outLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
      let conv2_weight = state_dict["\(prefix).resnets.\(i).conv2.weight"].type(torch.float).cpu()
        .numpy()
      let conv2_bias = state_dict["\(prefix).resnets.\(i).conv2.bias"].type(torch.float).cpu()
        .numpy()
      outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
      outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
      if let skipModel = skipModel {
        let conv_shortcut_weight = state_dict["\(prefix).resnets.\(i).conv_shortcut.weight"].type(
          torch.float
        ).cpu().numpy()
        let conv_shortcut_bias = state_dict["\(prefix).resnets.\(i).conv_shortcut.bias"].type(
          torch.float
        ).cpu().numpy()
        skipModel.weight.copy(from: try! Tensor<Float>(numpy: conv_shortcut_weight))
        skipModel.bias.copy(from: try! Tensor<Float>(numpy: conv_shortcut_bias))
      }
      let group_norm_weight = state_dict["\(prefix).attentions.\(i).group_norm.weight"].type(
        torch.float
      ).cpu().numpy()
      let group_norm_bias = state_dict["\(prefix).attentions.\(i).group_norm.bias"].type(
        torch.float
      ).cpu().numpy()
      norm.weight.copy(from: try! Tensor<Float>(numpy: group_norm_weight))
      norm.bias.copy(from: try! Tensor<Float>(numpy: group_norm_bias))
      let to_k_weight = state_dict["\(prefix).attentions.\(i).to_k.weight"].type(torch.float).cpu()
        .numpy()
      let to_k_bias = state_dict["\(prefix).attentions.\(i).to_k.bias"].type(torch.float).cpu()
        .numpy()
      tokeys.weight.copy(from: try! Tensor<Float>(numpy: to_k_weight))
      tokeys.bias.copy(from: try! Tensor<Float>(numpy: to_k_bias))
      let to_q_weight = state_dict["\(prefix).attentions.\(i).to_q.weight"].type(torch.float).cpu()
        .numpy()
      let to_q_bias = state_dict["\(prefix).attentions.\(i).to_q.bias"].type(torch.float).cpu()
        .numpy()
      toqueries.weight.copy(from: try! Tensor<Float>(numpy: to_q_weight))
      toqueries.bias.copy(from: try! Tensor<Float>(numpy: to_q_bias))
      let to_v_weight = state_dict["\(prefix).attentions.\(i).to_v.weight"].type(torch.float).cpu()
        .numpy()
      let to_v_bias = state_dict["\(prefix).attentions.\(i).to_v.bias"].type(torch.float).cpu()
        .numpy()
      tovalues.weight.copy(from: try! Tensor<Float>(numpy: to_v_weight))
      tovalues.bias.copy(from: try! Tensor<Float>(numpy: to_v_bias))
      let to_out_weight = state_dict["\(prefix).attentions.\(i).to_out.0.weight"].type(torch.float)
        .cpu().numpy()
      let to_out_bias = state_dict["\(prefix).attentions.\(i).to_out.0.bias"].type(torch.float)
        .cpu().numpy()
      unifyheads.weight.copy(from: try! Tensor<Float>(numpy: to_out_weight))
      unifyheads.bias.copy(from: try! Tensor<Float>(numpy: to_out_bias))
    }
  }
  out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
  let conv = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  readers.append { state_dict in
    let upsamplers_weight = state_dict["\(prefix).upsamplers.0.conv.weight"].type(torch.float).cpu()
      .numpy()
    let upsamplers_bias = state_dict["\(prefix).upsamplers.0.conv.bias"].type(torch.float).cpu()
      .numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: upsamplers_weight))
    conv.bias.copy(from: try! Tensor<Float>(numpy: upsamplers_bias))
  }
  out = conv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out)
}

func UpBlock2D(
  prefix: String, channels: Int, numRepeat: Int, batchSize: Int, upsample: Bool, x: Model.IO,
  hiddenStates: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO) {
  var out: Model.IO = x
  var readers = [(PythonObject) -> Void]()
  for i in 0..<numRepeat {
    let (inLayerNorm, inLayerConv2d, outLayerNorm, outLayerConv2d, skipModel, resBlock) = ResBlock(
      b: batchSize, groups: 4, outChannels: channels, skipConnection: true)
    out = Functional.concat(axis: 1, out, hiddenStates[i])
    out = resBlock(out)
    readers.append { state_dict in
      let norm1_weight = state_dict["\(prefix).resnets.\(i).norm1.weight"].type(torch.float).cpu()
        .numpy()
      let norm1_bias = state_dict["\(prefix).resnets.\(i).norm1.bias"].type(torch.float).cpu()
        .numpy()
      inLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
      inLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
      let conv1_weight = state_dict["\(prefix).resnets.\(i).conv1.weight"].type(torch.float).cpu()
        .numpy()
      let conv1_bias = state_dict["\(prefix).resnets.\(i).conv1.bias"].type(torch.float).cpu()
        .numpy()
      inLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
      inLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
      let norm2_weight = state_dict["\(prefix).resnets.\(i).norm2.weight"].type(torch.float).cpu()
        .numpy()
      let norm2_bias = state_dict["\(prefix).resnets.\(i).norm2.bias"].type(torch.float).cpu()
        .numpy()
      outLayerNorm.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
      outLayerNorm.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
      let conv2_weight = state_dict["\(prefix).resnets.\(i).conv2.weight"].type(torch.float).cpu()
        .numpy()
      let conv2_bias = state_dict["\(prefix).resnets.\(i).conv2.bias"].type(torch.float).cpu()
        .numpy()
      outLayerConv2d.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
      outLayerConv2d.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
      if let skipModel = skipModel {
        let conv_shortcut_weight = state_dict["\(prefix).resnets.\(i).conv_shortcut.weight"].type(
          torch.float
        ).cpu().numpy()
        let conv_shortcut_bias = state_dict["\(prefix).resnets.\(i).conv_shortcut.bias"].type(
          torch.float
        ).cpu().numpy()
        skipModel.weight.copy(from: try! Tensor<Float>(numpy: conv_shortcut_weight))
        skipModel.bias.copy(from: try! Tensor<Float>(numpy: conv_shortcut_bias))
      }
    }
  }
  if upsample {
    out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
    let conv = Convolution(
      groups: 1, filters: channels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    readers.append { state_dict in
      let upsamplers_weight = state_dict["\(prefix).upsamplers.0.conv.weight"].type(torch.float)
        .cpu().numpy()
      let upsamplers_bias = state_dict["\(prefix).upsamplers.0.conv.bias"].type(torch.float).cpu()
        .numpy()
      conv.weight.copy(from: try! Tensor<Float>(numpy: upsamplers_weight))
      conv.bias.copy(from: try! Tensor<Float>(numpy: upsamplers_bias))
    }
    out = conv(out)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out)
}

func MidBlock2D(
  prefix: String, channels: Int, batchSize: Int, numHeads: Int, startHeight: Int, startWidth: Int,
  x: Model.IO
) -> ((PythonObject) -> Void, Model.IO) {
  var out: Model.IO = x
  let (inLayerNorm1, inLayerConv2d1, outLayerNorm1, outLayerConv2d1, _, resBlock1) = ResBlock(
    b: batchSize, groups: 4, outChannels: channels, skipConnection: false)
  out = resBlock1(out)
  let norm = GroupNorm(axis: 1, groups: 4, epsilon: 1e-5, reduce: [2, 3])
  let residual = out
  out = norm(out)
  let (tokeys, toqueries, tovalues, unifyheads, attn) = SelfAttention(
    k: channels / numHeads, h: numHeads, b: batchSize, hw: startHeight * startWidth)
  out =
    attn(out.reshaped([batchSize, channels, startHeight * startWidth]).transposed(1, 2)).transposed(
      1, 2
    ).reshaped([batchSize, channels, startHeight, startWidth]) + residual
  let (inLayerNorm2, inLayerConv2d2, outLayerNorm2, outLayerConv2d2, _, resBlock2) = ResBlock(
    b: batchSize, groups: 4, outChannels: channels, skipConnection: false)
  out = resBlock2(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let norm10_weight = state_dict["\(prefix).resnets.0.norm1.weight"].type(torch.float).cpu()
      .numpy()
    let norm10_bias = state_dict["\(prefix).resnets.0.norm1.bias"].type(torch.float).cpu().numpy()
    inLayerNorm1.weight.copy(from: try! Tensor<Float>(numpy: norm10_weight))
    inLayerNorm1.bias.copy(from: try! Tensor<Float>(numpy: norm10_bias))
    let conv10_weight = state_dict["\(prefix).resnets.0.conv1.weight"].type(torch.float).cpu()
      .numpy()
    let conv10_bias = state_dict["\(prefix).resnets.0.conv1.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d1.weight.copy(from: try! Tensor<Float>(numpy: conv10_weight))
    inLayerConv2d1.bias.copy(from: try! Tensor<Float>(numpy: conv10_bias))
    let norm20_weight = state_dict["\(prefix).resnets.0.norm2.weight"].type(torch.float).cpu()
      .numpy()
    let norm20_bias = state_dict["\(prefix).resnets.0.norm2.bias"].type(torch.float).cpu().numpy()
    outLayerNorm1.weight.copy(from: try! Tensor<Float>(numpy: norm20_weight))
    outLayerNorm1.bias.copy(from: try! Tensor<Float>(numpy: norm20_bias))
    let conv20_weight = state_dict["\(prefix).resnets.0.conv2.weight"].type(torch.float).cpu()
      .numpy()
    let conv20_bias = state_dict["\(prefix).resnets.0.conv2.bias"].type(torch.float).cpu().numpy()
    outLayerConv2d1.weight.copy(from: try! Tensor<Float>(numpy: conv20_weight))
    outLayerConv2d1.bias.copy(from: try! Tensor<Float>(numpy: conv20_bias))
    let group_norm_weight = state_dict["\(prefix).attentions.0.group_norm.weight"].type(torch.float)
      .cpu().numpy()
    let group_norm_bias = state_dict["\(prefix).attentions.0.group_norm.bias"].type(torch.float)
      .cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: group_norm_weight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: group_norm_bias))
    let to_k_weight = state_dict["\(prefix).attentions.0.to_k.weight"].type(torch.float).cpu()
      .numpy()
    let to_k_bias = state_dict["\(prefix).attentions.0.to_k.bias"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: to_k_weight))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: to_k_bias))
    let to_q_weight = state_dict["\(prefix).attentions.0.to_q.weight"].type(torch.float).cpu()
      .numpy()
    let to_q_bias = state_dict["\(prefix).attentions.0.to_q.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: to_q_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: to_q_bias))
    let to_v_weight = state_dict["\(prefix).attentions.0.to_v.weight"].type(torch.float).cpu()
      .numpy()
    let to_v_bias = state_dict["\(prefix).attentions.0.to_v.bias"].type(torch.float).cpu().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: to_v_weight))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: to_v_bias))
    let to_out_weight = state_dict["\(prefix).attentions.0.to_out.0.weight"].type(torch.float).cpu()
      .numpy()
    let to_out_bias = state_dict["\(prefix).attentions.0.to_out.0.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: to_out_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: to_out_bias))
    let norm11_weight = state_dict["\(prefix).resnets.1.norm1.weight"].type(torch.float).cpu()
      .numpy()
    let norm11_bias = state_dict["\(prefix).resnets.1.norm1.bias"].type(torch.float).cpu().numpy()
    inLayerNorm2.weight.copy(from: try! Tensor<Float>(numpy: norm11_weight))
    inLayerNorm2.bias.copy(from: try! Tensor<Float>(numpy: norm11_bias))
    let conv11_weight = state_dict["\(prefix).resnets.1.conv1.weight"].type(torch.float).cpu()
      .numpy()
    let conv11_bias = state_dict["\(prefix).resnets.1.conv1.bias"].type(torch.float).cpu().numpy()
    inLayerConv2d2.weight.copy(from: try! Tensor<Float>(numpy: conv11_weight))
    inLayerConv2d2.bias.copy(from: try! Tensor<Float>(numpy: conv11_bias))
    let norm21_weight = state_dict["\(prefix).resnets.1.norm2.weight"].type(torch.float).cpu()
      .numpy()
    let norm21_bias = state_dict["\(prefix).resnets.1.norm2.bias"].type(torch.float).cpu().numpy()
    outLayerNorm2.weight.copy(from: try! Tensor<Float>(numpy: norm21_weight))
    outLayerNorm2.bias.copy(from: try! Tensor<Float>(numpy: norm21_bias))
    let conv21_weight = state_dict["\(prefix).resnets.1.conv2.weight"].type(torch.float).cpu()
      .numpy()
    let conv21_bias = state_dict["\(prefix).resnets.1.conv2.bias"].type(torch.float).cpu().numpy()
    outLayerConv2d2.weight.copy(from: try! Tensor<Float>(numpy: conv21_weight))
    outLayerConv2d2.bias.copy(from: try! Tensor<Float>(numpy: conv21_bias))
  }
  return (reader, out)
}

func TransparentVAEDecoder(startHeight: Int, startWidth: Int) -> ((PythonObject) -> Void, Model) {
  let pixel = Input()
  let latent = Input()
  precondition(startHeight % 64 == 0)
  precondition(startWidth % 64 == 0)
  let convIn = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), name: "conv_in")
  let latentConvIn = Convolution(
    groups: 1, filters: 64, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    name: "latent_conv_in")
  var out = convIn(pixel)
  var hiddenStates: [Model.IO] = [out]
  var readers = [(PythonObject) -> Void]()
  let channels = [32, 32, 64, 128, 256, 512, 512]
  for i in 0..<4 {
    if i == 3 {
      out = out + latentConvIn(latent)
    }
    let (downReader, downOut, downHiddenStates) = DownBlock2D(
      prefix: "down_blocks.\(i)", inChannels: i > 0 ? channels[i - 1] : 32,
      outChannels: channels[i], numRepeat: 2, batchSize: 1, x: out)
    hiddenStates.append(contentsOf: downHiddenStates)
    out = downOut
    readers.append(downReader)
  }
  var startHeight = startHeight / 16
  var startWidth = startWidth / 16
  for i in 4..<7 {
    let (downReader, downOut, downHiddenStates) = AttnDownBlock2D(
      prefix: "down_blocks.\(i)", inChannels: channels[i - 1], outChannels: channels[i],
      numRepeat: 2, batchSize: 1, numHeads: channels[i] / 8, startHeight: startHeight,
      startWidth: startWidth, downsample: (i < channels.count - 1), x: out)
    hiddenStates.append(contentsOf: downHiddenStates)
    out = downOut
    readers.append(downReader)
    if i < channels.count - 1 {
      startHeight /= 2
      startWidth /= 2
    }
  }
  let (midReader, midOut) = MidBlock2D(
    prefix: "mid_block", channels: 512, batchSize: 1, numHeads: 512 / 8, startHeight: 16,
    startWidth: 16, x: out)
  out = midOut
  readers.append(midReader)
  for i in 0..<3 {
    let (upReader, upOut) = AttnUpBlock2D(
      prefix: "up_blocks.\(i)", channels: channels[channels.count - 1 - i], numRepeat: 3,
      batchSize: 1, numHeads: channels[channels.count - 1 - i] / 8, startHeight: startHeight,
      startWidth: startWidth, x: out,
      hiddenStates: hiddenStates[(hiddenStates.count - 3 * (i + 1))..<(hiddenStates.count - 3 * i)]
        .reversed())
    out = upOut
    readers.append(upReader)
    startHeight *= 2
    startWidth *= 2
  }
  for i in 3..<7 {
    let (upReader, upOut) = UpBlock2D(
      prefix: "up_blocks.\(i)", channels: channels[channels.count - 1 - i], numRepeat: 3,
      batchSize: 1, upsample: (i < channels.count - 1), x: out,
      hiddenStates: hiddenStates[(hiddenStates.count - 3 * (i + 1))..<(hiddenStates.count - 3 * i)]
        .reversed())
    out = upOut
    readers.append(upReader)
  }
  let normOut = GroupNorm(axis: 1, groups: 4, epsilon: 1e-5, reduce: [2, 3], name: "norm_out")
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), name: "conv_out")
  out = convOut(out)

  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["conv_in.weight"].type(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["conv_in.bias"].type(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    let latent_conv_in_weight = state_dict["latent_conv_in.weight"].type(torch.float).cpu().numpy()
    let latent_conv_in_bias = state_dict["latent_conv_in.bias"].type(torch.float).cpu().numpy()
    latentConvIn.weight.copy(from: try! Tensor<Float>(numpy: latent_conv_in_weight))
    latentConvIn.bias.copy(from: try! Tensor<Float>(numpy: latent_conv_in_bias))
    let conv_norm_out_weight = state_dict["conv_norm_out.weight"].type(torch.float).cpu().numpy()
    let conv_norm_out_bias = state_dict["conv_norm_out.bias"].type(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: conv_norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: conv_norm_out_bias))
    var conv_out_weight = state_dict["conv_out.weight"].type(torch.float).cpu()
    var conv_out_bias = state_dict["conv_out.bias"].type(torch.float).cpu()
    // For RedAIGC trained FLUX TransparentVAE, they moved alpha to the last (RGBA) rather than original SDXL's ARGB format.
    // Standardize to ARGB here.
    conv_out_weight = torch.cat([conv_out_weight[3..<4], conv_out_weight[0..<3]], dim: 0)
    conv_out_weight = conv_out_weight.numpy()
    conv_out_bias = torch.cat([conv_out_bias[3..<4], conv_out_bias[0..<3]], dim: 0)
    conv_out_bias = conv_out_bias.numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([pixel, latent], [out]))
}

let graph = DynamicGraph()

graph.withNoGrad {
  let (reader, decoder) = TransparentVAEDecoder(startHeight: 1024, startWidth: 1024)
  let pixelTensor = graph.variable(try! Tensor<Float>(numpy: pixel.numpy())).toGPU(1)
  let latentTensor = graph.variable(try! Tensor<Float>(numpy: latent.numpy())).toGPU(1)
  decoder.compile(inputs: pixelTensor, latentTensor)
  reader(state_dict)
  let out = decoder(inputs: pixelTensor, latentTensor)[0].as(of: Float.self)
  debugPrint(out)
  graph.openStore("/home/liu/workspace/swift-diffusion/flux_1_transparent_vae_decoder_f32.ckpt") {
    $0.write("decoder", model: decoder)
  }
}
