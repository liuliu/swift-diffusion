import Foundation
import NNC
import NNCPythonConversion
import PythonKit

// let ldm_util = Python.import("ldm.util")
let torch = Python.import("torch")
let omegaconf = Python.import("omegaconf")
let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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
    let norm1_weight = state_dict["decoder.\(prefix).norm1.weight"].cpu().numpy()
    let norm1_bias = state_dict["decoder.\(prefix).norm1.bias"].cpu().numpy()
    norm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let conv1_weight = state_dict["decoder.\(prefix).conv1.weight"].cpu().numpy()
    let conv1_bias = state_dict["decoder.\(prefix).conv1.bias"].cpu().numpy()
    conv1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["decoder.\(prefix).norm2.weight"].cpu().numpy()
    let norm2_bias = state_dict["decoder.\(prefix).norm2.bias"].cpu().numpy()
    norm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["decoder.\(prefix).conv2.weight"].cpu().numpy()
    let conv2_bias = state_dict["decoder.\(prefix).conv2.bias"].cpu().numpy()
    conv2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["decoder.\(prefix).nin_shortcut.weight"].cpu().numpy()
      let nin_shortcut_bias = state_dict["decoder.\(prefix).nin_shortcut.bias"].cpu().numpy()
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
    let norm_weight = state_dict["decoder.\(prefix).norm.weight"].cpu().numpy()
    let norm_bias = state_dict["decoder.\(prefix).norm.bias"].cpu().numpy()
    norm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_bias))
    let k_weight = state_dict["decoder.\(prefix).k.weight"].cpu().numpy()
    let k_bias = state_dict["decoder.\(prefix).k.bias"].cpu().numpy()
    tokeys.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["decoder.\(prefix).q.weight"].cpu().numpy()
    let q_bias = state_dict["decoder.\(prefix).q.bias"].cpu().numpy()
    toqueries.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["decoder.\(prefix).v.weight"].cpu().numpy()
    let v_bias = state_dict["decoder.\(prefix).v.bias"].cpu().numpy()
    tovalues.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["decoder.\(prefix).proj_out.weight"].cpu().numpy()
    let proj_out_bias = state_dict["decoder.\(prefix).proj_out.bias"].cpu().numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x], [out]))
}

func Decoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  let postQuantConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = postQuantConv2d(x)
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convIn(out)
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
  var readers = [(PythonObject) -> Void]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlock(
        prefix: "up.\(i).block.\(j)", outChannels: channel, shortcut: previousChannel != channel)
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
        let conv_weight = state_dict["decoder.up.\(upLayer).upsample.conv.weight"].cpu().numpy()
        let conv_bias = state_dict["decoder.up.\(upLayer).upsample.conv.bias"].cpu().numpy()
        conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_bias))
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
    let post_quant_conv_weight = state_dict["post_quant_conv.weight"].cpu().numpy()
    let post_quant_conv_bias = state_dict["post_quant_conv.bias"].cpu().numpy()
    postQuantConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: post_quant_conv_weight))
    postQuantConv2d.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: post_quant_conv_bias))
    let conv_in_weight = state_dict["decoder.conv_in.weight"].cpu().numpy()
    let conv_in_bias = state_dict["decoder.conv_in.bias"].cpu().numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["decoder.norm_out.weight"].cpu().numpy()
    let norm_out_bias = state_dict["decoder.norm_out.bias"].cpu().numpy()
    normOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["decoder.conv_out.weight"].cpu().numpy()
    let conv_out_bias = state_dict["decoder.conv_out.bias"].cpu().numpy()
    convOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

let x = torch.randn([1, 4, 64, 64])
/*
let config = omegaconf.OmegaConf.load(
  "/home/liu/workspace/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
let pl_sd = torch.load(
  "/home/liu/workspace/stable-diffusion/models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt",
  map_location: "cpu")
let sd = pl_sd["state_dict"]
*/
let streamlit_helpers = Python.import("scripts.demo.streamlit_helpers")

var version_dict: [String: PythonObject] = [
  "H": 1024,
  "W": 1024,
  "C": 4,
  "f": 8,
  "is_legacy": false,
  "config": "/home/liu/workspace/generative-models/configs/inference/sd_xl_base.yaml",
  "ckpt": "/home/liu/workspace/generative-models/checkpoints/sd_xl_base_0.9.safetensors",
  "is_guided": true,
]

let state = streamlit_helpers.init_st(version_dict)
let sd = state["model"].first_stage_model.state_dict()
/*
let model = ldm_util.instantiate_from_config(config.model)
model.load_state_dict(sd, strict: false)
model.eval()
let z = 1.0 / model.scale_factor * x
print(model.scale_factor)
let state_dict = model.first_stage_model.state_dict()
let ret = model.decode_first_stage(x)
print(ret)
*/
let z = x
let state_dict = sd

let graph = DynamicGraph()
let zTensor = graph.variable(try! Tensor<Float>(numpy: z.numpy())).toGPU(0)
let (reader, decoder) = Decoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64, startHeight: 64)

graph.withNoGrad {
  let _ = decoder(inputs: zTensor)
  reader(state_dict)
  let quant = decoder(inputs: zTensor)[0].as(of: Float.self)
  let quantCPU = quant.toCPU()
  debugPrint(quantCPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/autoencoder.ckpt") {
    $0.write("decoder", model: decoder)
  }
}
