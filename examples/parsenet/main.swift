import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let facexlib_parsing = Python.import("facexlib.parsing")
let torch = Python.import("torch")
let numpy = Python.import("numpy")
let random = Python.import("random")

let model = facexlib_parsing.init_parsing_model(
  model_name: "parsenet", device: torch.device("cpu"),
  model_rootpath: "/home/liu/workspace/GFPGAN/gfpgan/weights/")
let state_dict = model.state_dict()
print(state_dict.keys())
model.eval()

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let croppedFace = torch.randn([1, 3, 512, 512])
let ret = model(croppedFace)[0]
print(ret)

func ResidualBlock(prefix: String, outChannels: Int, scaleUp: Bool, scaleDown: Bool, shortcut: Bool)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  let z: Model.IO
  if scaleUp {
    z = Upsample(.nearest, widthScale: 2, heightScale: 2)(x)
  } else {
    z = x
  }
  let ninShortcut: Model?
  let y: Model.IO
  if shortcut {
    let conv: Model
    if scaleDown {
      conv = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    } else {
      conv = Convolution(
        groups: 1, filters: outChannels, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    }
    ninShortcut = conv
    y = conv(z)
  } else {
    ninShortcut = nil
    y = z
  }
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv1(z).leakyReLU(negativeSlope: 0.2)
  let conv2: Model
  if scaleDown {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  } else {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  }
  out = y + conv2(out)
  let reader: (PythonObject) -> Void = { state_dict in
    if let ninShortcut = ninShortcut {
      let shortcut_func_weight = state_dict["\(prefix).shortcut_func.conv2d.weight"].numpy()
      let shortcut_func_bias = state_dict["\(prefix).shortcut_func.conv2d.bias"].numpy()
      ninShortcut.weight.copy(from: try! Tensor<Float>(numpy: shortcut_func_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: shortcut_func_bias))
    }
    let norm1_weight = state_dict["\(prefix).conv1.norm.norm.weight"]
    let norm1_running_var = state_dict["\(prefix).conv1.norm.norm.running_var"]
    let norm1_scale = norm1_weight.div(torch.sqrt(1e-5 + norm1_running_var))
    let conv1_weight = numpy.multiply(
      state_dict["\(prefix).conv1.conv2d.weight"].numpy(),
      numpy.expand_dims(norm1_scale.numpy(), axis: PythonObject(tupleOf: 1, 2, 3)))
    let norm1_bias = state_dict["\(prefix).conv1.norm.norm.bias"]
    let norm1_running_mean = state_dict["\(prefix).conv1.norm.norm.running_mean"]
    let conv1_bias =
      (norm1_bias - norm1_weight.mul(norm1_running_mean).div(torch.sqrt(norm1_running_var + 1e-5)))
      .numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).conv2.norm.norm.weight"]
    let norm2_running_var = state_dict["\(prefix).conv2.norm.norm.running_var"]
    let norm2_scale = norm2_weight.div(torch.sqrt(1e-5 + norm2_running_var))
    let conv2_weight = numpy.multiply(
      state_dict["\(prefix).conv2.conv2d.weight"].numpy(),
      numpy.expand_dims(norm2_scale.numpy(), axis: PythonObject(tupleOf: 1, 2, 3)))
    let norm2_bias = state_dict["\(prefix).conv2.norm.norm.bias"]
    let norm2_running_mean = state_dict["\(prefix).conv2.norm.norm.running_mean"]
    let conv2_bias =
      (norm2_bias - norm2_weight.mul(norm2_running_mean).div(torch.sqrt(norm2_running_var + 1e-5)))
      .numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
  }
  return (reader, Model([x], [out]))
}

func ParseNet() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let conv = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv(x)
  var readers = [(PythonObject) -> Void]()
  let outChannels = [64, 128, 256, 256, 256]
  for i in 1..<5 {
    let (encoderReader, encoder) = ResidualBlock(
      prefix: "encoder.\(i)", outChannels: outChannels[i], scaleUp: false, scaleDown: true,
      shortcut: true)
    out = encoder(out)
    readers.append(encoderReader)
  }
  let feat = out
  for i in 0..<10 {
    let (bodyReader, body) = ResidualBlock(
      prefix: "body.\(i)", outChannels: 256, scaleUp: false, scaleDown: false, shortcut: false)
    out = body(out)
    readers.append(bodyReader)
  }
  out = feat + out
  for i in 0..<4 {
    let (decoderReader, decoder) = ResidualBlock(
      prefix: "decoder.\(i)", outChannels: outChannels[outChannels.count - 2 - i], scaleUp: true,
      scaleDown: false, shortcut: true)
    out = decoder(out)
    readers.append(decoderReader)
  }
  let outMaskConv = Convolution(
    groups: 1, filters: 19, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outMaskConv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_weight = state_dict["encoder.0.conv2d.weight"].numpy()
    let conv_bias = state_dict["encoder.0.conv2d.bias"].numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
    conv.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
    for reader in readers {
      reader(state_dict)
    }
    let out_mask_conv_weight = state_dict["out_mask_conv.conv2d.weight"].numpy()
    let out_mask_conv_bias = state_dict["out_mask_conv.conv2d.bias"].numpy()
    outMaskConv.weight.copy(from: try! Tensor<Float>(numpy: out_mask_conv_weight))
    outMaskConv.bias.copy(from: try! Tensor<Float>(numpy: out_mask_conv_bias))
  }
  return (reader, Model([x], [out]))
}

var initImg = Tensor<Float16>(.CPU, .NCHW(1, 3, 512, 512))
if let image = try PNG.Data.Rectangular.decompress(
  path: "/home/liu/workspace/GFPGAN/inputs/cropped_faces/Justin_Timberlake_crop.png")
{
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  for y in 0..<512 {
    for x in 0..<512 {
      let pixel = rgba[y * 512 + x]
      initImg[0, 0, y, x] = Float16(Float(pixel.r) / 255 * 2 - 1)
      initImg[0, 1, y, x] = Float16(Float(pixel.g) / 255 * 2 - 1)
      initImg[0, 2, y, x] = Float16(Float(pixel.b) / 255 * 2 - 1)
    }
  }
}

let graph = DynamicGraph()
let croppedFaceTensor = graph.variable(try! Tensor<Float>(numpy: croppedFace.numpy())).toGPU(0)
graph.workspaceSize = 1_024 * 1_024 * 1_024
let (reader, parsenet) = ParseNet()
graph.withNoGrad {
  parsenet.compile(inputs: croppedFaceTensor)
  reader(state_dict)
  let result = parsenet(inputs: croppedFaceTensor)[0].as(of: Float.self)
  debugPrint(result)
  graph.openStore("/home/liu/workspace/swift-diffusion/parsenet_v1.0.ckpt") {
    $0.write("parsenet", model: parsenet)
  }
  let (_, parsenet16) = ParseNet()
  let initImgTensor = graph.variable(initImg).toGPU(0)
  parsenet16.compile(inputs: initImgTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/parsenet_v1.0.ckpt") {
    $0.read("parsenet", model: parsenet16)
  }
  let outMask = parsenet16(inputs: initImgTensor)[0].as(
    of: Float16.self
  )
  let idx = Functional.argmax(outMask, axis: 1).toCPU()
  graph.openStore("/home/liu/workspace/swift-diffusion/parsenet_v1.0_f16.ckpt") {
    $0.write("parsenet", model: parsenet16)
  }
  let colors: [(UInt8, UInt8, UInt8)] = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 0, 85), (255, 0, 170), (0, 255, 0),
    (85, 255, 0), (170, 255, 0), (0, 255, 85), (0, 255, 170), (0, 0, 255), (85, 0, 255),
    (170, 0, 255), (0, 85, 255), (0, 170, 255), (255, 255, 0), (255, 255, 85), (255, 255, 170),
    (255, 0, 255), (255, 85, 255), (255, 170, 255), (0, 255, 255), (85, 255, 255), (170, 255, 255),
  ]

  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: 512 * 512)
  for y in 0..<512 {
    for x in 0..<512 {
      let index = Int(idx[0, 0, y, x])
      rgba[y * 512 + x].r = colors[index].0
      rgba[y * 512 + x].g = colors[index].1
      rgba[y * 512 + x].b = colors[index].2
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (512, 512),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/parsenet.png", level: 4)
}
