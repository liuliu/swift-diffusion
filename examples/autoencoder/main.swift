import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let ldm_util = Python.import("ldm.util")
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
    let norm1_weight = state_dict["decoder.\(prefix).norm1.weight"].numpy()
    let norm1_bias = state_dict["decoder.\(prefix).norm1.bias"].numpy()
    norm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let conv1_weight = state_dict["decoder.\(prefix).conv1.weight"].numpy()
    let conv1_bias = state_dict["decoder.\(prefix).conv1.bias"].numpy()
    conv1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["decoder.\(prefix).norm2.weight"].numpy()
    let norm2_bias = state_dict["decoder.\(prefix).norm2.bias"].numpy()
    norm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["decoder.\(prefix).conv2.weight"].numpy()
    let conv2_bias = state_dict["decoder.\(prefix).conv2.bias"].numpy()
    conv2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["decoder.\(prefix).nin_shortcut.weight"].numpy()
      let nin_shortcut_bias = state_dict["decoder.\(prefix).nin_shortcut.bias"].numpy()
      ninShortcut.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
  }
  return (reader, Model([x], [out]))
}

func Decoder() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let postQuantConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = postQuantConv2d(x)
  let convIn = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convIn(out)
  let (midReader1, midBlock1) = ResnetBlock(
    prefix: "mid.block_1", outChannels: 512, shortcut: false)
  out = midBlock1(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let post_quant_conv_weight = state_dict["post_quant_conv.weight"].numpy()
    let post_quant_conv_bias = state_dict["post_quant_conv.bias"].numpy()
    postQuantConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: post_quant_conv_weight))
    postQuantConv2d.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: post_quant_conv_bias))
    let conv_in_weight = state_dict["decoder.conv_in.weight"].numpy()
    let conv_in_bias = state_dict["decoder.conv_in.bias"].numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midReader1(state_dict)
  }
  return (reader, Model([x], [out]))
}

let x = torch.randn([1, 4, 64, 64])

let config = omegaconf.OmegaConf.load(
  "/home/liu/workspace/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
let pl_sd = torch.load(
  "/home/liu/workspace/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
  map_location: "cpu")
let sd = pl_sd["state_dict"]
let model = ldm_util.instantiate_from_config(config.model)
model.load_state_dict(sd, strict: false)
model.eval()
let z = 1.0 / model.scale_factor * x
print(model.scale_factor)
let quant = model.first_stage_model.post_quant_conv(z)
let state_dict = model.first_stage_model.state_dict()
let ret = model.first_stage_model.decoder(quant)
print(state_dict.keys())

let graph = DynamicGraph()
let zTensor = graph.variable(try! Tensor<Float>(numpy: z.numpy())).toGPU(0)
let (reader, decoder) = Decoder()

graph.withNoGrad {
  let _ = decoder(inputs: zTensor)
  reader(state_dict)
  let quant = decoder(inputs: zTensor)[0].as(of: Float.self)
  let quantCPU = quant.toCPU()
  print(quantCPU)
  for i in 0..<6 {
    let x = i < 3 ? i : 506 + i
    for j in 0..<6 {
      let y = j < 3 ? j : 58 + j
      for k in 0..<6 {
        let z = k < 3 ? k : 58 + k
        print("0 \(x) \(y) \(z) \(quantCPU[0, x, y, z])")
      }
    }
  }
}
