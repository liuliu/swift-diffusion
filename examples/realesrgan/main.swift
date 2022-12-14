import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let basicsr = Python.import("basicsr")
let torch = Python.import("torch")
let numpy = Python.import("numpy")
let random = Python.import("random")

let model = basicsr.archs.rrdbnet_arch.RRDBNet(
  num_in_ch: 3, num_out_ch: 3, num_feat: 64, num_block: 23, num_grow_ch: 32, scale: 2)
let esrgan = torch.load(
  //  "/home/liu/workspace/Real-ESRGAN/weights/RealESRGAN_x4plus.pth",
  "/home/liu/workspace/Real-ESRGAN/weights/RealESRGAN_x2plus.pth",
  map_location: "cpu")
let state_dict = esrgan["params_ema"]
model.load_state_dict(state_dict, strict: false)

func ResidualDenseBlock(prefix: String, numberOfFeatures: Int, numberOfGrowChannels: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  let x1 = conv1(x).leakyReLU(negativeSlope: 0.2)
  let x01 = Functional.concat(axis: 1, x, x1)
  let conv2 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  let x2 = conv2(x01).leakyReLU(negativeSlope: 0.2)
  let x012 = Functional.concat(axis: 1, x01, x2)
  let conv3 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  let x3 = conv3(x012).leakyReLU(negativeSlope: 0.2)
  let x0123 = Functional.concat(axis: 1, x012, x3)
  let conv4 = Convolution(
    groups: 1, filters: numberOfGrowChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  let x4 = conv4(x0123).leakyReLU(negativeSlope: 0.2)
  let x01234 = Functional.concat(axis: 1, x0123, x4)
  let conv5 = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  let x5 = conv5(x01234)
  let out = 0.2 * x5 + x
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1_weight = state_dict["\(prefix).conv1.weight"].float().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.bias"].float().numpy()
    conv1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let conv2_weight = state_dict["\(prefix).conv2.weight"].float().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.bias"].float().numpy()
    conv2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv2_bias))
    let conv3_weight = state_dict["\(prefix).conv3.weight"].float().numpy()
    let conv3_bias = state_dict["\(prefix).conv3.bias"].float().numpy()
    conv3.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv3_weight))
    conv3.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv3_bias))
    let conv4_weight = state_dict["\(prefix).conv4.weight"].float().numpy()
    let conv4_bias = state_dict["\(prefix).conv4.bias"].float().numpy()
    conv4.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv4_weight))
    conv4.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv4_bias))
    let conv5_weight = state_dict["\(prefix).conv5.weight"].float().numpy()
    let conv5_bias = state_dict["\(prefix).conv5.bias"].float().numpy()
    conv5.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv5_weight))
    conv5.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv5_bias))
  }
  return (Model([x], [out]), reader)
}

func RRDB(prefix: String, numberOfFeatures: Int, numberOfGrowChannels: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let (rdb1, reader1) = ResidualDenseBlock(
    prefix: "\(prefix).rdb1", numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  var out = rdb1(x)
  let (rdb2, reader2) = ResidualDenseBlock(
    prefix: "\(prefix).rdb2", numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  out = rdb2(out)
  let (rdb3, reader3) = ResidualDenseBlock(
    prefix: "\(prefix).rdb3", numberOfFeatures: numberOfFeatures,
    numberOfGrowChannels: numberOfGrowChannels)
  out = 0.2 * rdb3(out) + x
  let reader: (PythonObject) -> Void = { state_dict in
    reader1(state_dict)
    reader2(state_dict)
    reader3(state_dict)
  }
  return (Model([x], [out]), reader)
}

func RRDBNet(
  numberOfOutputChannels: Int, numberOfFeatures: Int, numberOfBlocks: Int, numberOfGrowChannels: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let convFirst = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convFirst(x)
  let feat = out
  var readers = [(PythonObject) -> Void]()
  for i in 0..<numberOfBlocks {
    let (rrdb, reader) = RRDB(
      prefix: "body.\(i)", numberOfFeatures: numberOfFeatures,
      numberOfGrowChannels: numberOfGrowChannels)
    out = rrdb(out)
    readers.append(reader)
  }
  let convBody = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convBody(out)
  out = feat + out
  let convUp1 = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convUp1(Upsample(.nearest, widthScale: 2, heightScale: 2)(out)).leakyReLU(
    negativeSlope: 0.2)
  let convUp2 = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convUp2(Upsample(.nearest, widthScale: 2, heightScale: 2)(out)).leakyReLU(
    negativeSlope: 0.2)
  let convHr = Convolution(
    groups: 1, filters: numberOfFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convHr(out).leakyReLU(negativeSlope: 0.2)
  let convLast = Convolution(
    groups: 1, filters: numberOfOutputChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convLast(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_first_weight = state_dict["conv_first.weight"].float().numpy()
    let conv_first_bias = state_dict["conv_first.bias"].float().numpy()
    convFirst.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_first_weight))
    convFirst.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_first_bias))
    for reader in readers {
      reader(state_dict)
    }
    let conv_body_weight = state_dict["conv_body.weight"].float().numpy()
    let conv_body_bias = state_dict["conv_body.bias"].float().numpy()
    convBody.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_body_weight))
    convBody.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_body_bias))
    let conv_up1_weight = state_dict["conv_up1.weight"].float().numpy()
    let conv_up1_bias = state_dict["conv_up1.bias"].float().numpy()
    convUp1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_up1_weight))
    convUp1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_up1_bias))
    let conv_up2_weight = state_dict["conv_up2.weight"].float().numpy()
    let conv_up2_bias = state_dict["conv_up2.bias"].float().numpy()
    convUp2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_up2_weight))
    convUp2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_up2_bias))
    let conv_hr_weight = state_dict["conv_hr.weight"].float().numpy()
    let conv_hr_bias = state_dict["conv_hr.bias"].float().numpy()
    convHr.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_hr_weight))
    convHr.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_hr_bias))
    let conv_last_weight = state_dict["conv_last.weight"].float().numpy()
    let conv_last_bias = state_dict["conv_last.bias"].float().numpy()
    convLast.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_last_weight))
    convLast.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_last_bias))
  }
  return (Model([x], [out]), reader)
}

let graph = DynamicGraph()
/*
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([1, 3, 32, 32])
let y = model(x)
print(y)
*/
var initImg = Tensor<Float>(.CPU, .NCHW(1, 12, 256, 256))
if let image = try PNG.Data.Rectangular.decompress(
  path: "/home/liu/workspace/swift-diffusion/init_img.png")
{
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  for y in 0..<256 {
    for x in 0..<256 {
      let p0 = rgba[y * 2 * 512 + x * 2]
      let p1 = rgba[y * 2 * 512 + x * 2 + 1]
      let p2 = rgba[(y * 2 + 1) * 512 + x * 2]
      let p3 = rgba[(y * 2 + 1) * 512 + x * 2 + 1]
      initImg[0, 0, y, x] = Float(p0.r) / 255
      initImg[0, 1, y, x] = Float(p1.r) / 255
      initImg[0, 2, y, x] = Float(p2.r) / 255
      initImg[0, 3, y, x] = Float(p3.r) / 255
      initImg[0, 4, y, x] = Float(p0.g) / 255
      initImg[0, 5, y, x] = Float(p1.g) / 255
      initImg[0, 6, y, x] = Float(p2.g) / 255
      initImg[0, 7, y, x] = Float(p3.g) / 255
      initImg[0, 8, y, x] = Float(p0.b) / 255
      initImg[0, 9, y, x] = Float(p1.b) / 255
      initImg[0, 10, y, x] = Float(p2.b) / 255
      initImg[0, 11, y, x] = Float(p3.b) / 255
    }
  }
}
print("loaded image")
let (rrdbnet, reader) = RRDBNet(
  numberOfOutputChannels: 3, numberOfFeatures: 64, numberOfBlocks: 23, numberOfGrowChannels: 32)
graph.withNoGrad {
  let xTensor = graph.variable(initImg).toGPU(0)
  rrdbnet.compile(inputs: xTensor)
  reader(state_dict)
  print("loaded parameters")
  let yTensor = rrdbnet(inputs: xTensor)[0].as(of: Float.self).toCPU()
  graph.openStore("/home/liu/workspace/swift-diffusion/realesrgan_x2plus_f32.ckpt") {
    $0.write("realesrgan_x2plus", model: rrdbnet)
  }
  print("upsampled")
  let yHeight = yTensor.shape[2]
  let yWidth = yTensor.shape[3]
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: yHeight * yWidth)
  for y in 0..<yHeight {
    for x in 0..<yWidth {
      let (r, g, b) = (yTensor[0, 0, y, x], yTensor[0, 1, y, x], yTensor[0, 2, y, x])
      rgba[y * yWidth + x].r = UInt8(
        min(max(Int(r * 255), 0), 255))
      rgba[y * yWidth + x].g = UInt8(
        min(max(Int(g * 255), 0), 255))
      rgba[y * yWidth + x].b = UInt8(
        min(max(Int(b * 255), 0), 255))
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (yHeight, yWidth),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  print("begin compress")
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/upsampler.png", level: 4)
}
