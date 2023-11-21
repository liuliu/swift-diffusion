import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let torch = Python.import("torch")
let diffusers = Python.import("diffusers")
let safetensors = Python.import("safetensors")
let conv_unet_vae = Python.import("conv_unet_vae")
let taesd = Python.import("taesd")
let consistencydecoder = Python.import("consistencydecoder")

// encode with stable diffusion vae
let pipe = diffusers.StableDiffusionPipeline.from_pretrained(
  "runwayml/stable-diffusion-v1-5", torch_dtype: torch.float16
)
pipe.vae.cuda()

let model = taesd.TAESD(
  encoder_path: "/home/liu/workspace/taesd/taesd_encoder.pth",
  decoder_path: "/home/liu/workspace/taesd/taesd_decoder.pth"
).cuda()

let image = consistencydecoder.load_image(
  "/home/liu/workspace/consistencydecoder/assets/gt1.png", size: PythonObject(tupleOf: 256, 256),
  center_crop: true)
let latent = pipe.vae.encode(image.half().cuda()).latent_dist.sample()

// decode with gan
let sample_gan = pipe.vae.decode(latent).sample.detach()
consistencydecoder.save_image(sample_gan, "/home/liu/workspace/swift-diffusion/gan.png")

// decode with conv_unet_vae
let sample_taesd = model.decoder(latent.float()).clamp(0, 1).detach()
consistencydecoder.save_image(sample_taesd, "/home/liu/workspace/swift-diffusion/taesd.png")

let state_dict = model.decoder.state_dict()
print(model.decoder)

func Block(prefix: String, nOut: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let conv = Model([
    Convolution(
      groups: 1, filters: nOut, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1]))),
    ReLU(),
    Convolution(
      groups: 1, filters: nOut, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1]))),
    ReLU(),
    Convolution(
      groups: 1, filters: nOut, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1]))),
  ])
  var out = conv(x)
  out = (out + x).ReLU()
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_0_weight = state_dict["\(prefix).conv.0.weight"].type(torch.float).cpu().numpy()
    let conv_0_bias = state_dict["\(prefix).conv.0.bias"].type(torch.float).cpu().numpy()
    conv.parameters(for: .index(0)).copy(from: try! Tensor<Float>(numpy: conv_0_weight))
    conv.parameters(for: .index(1)).copy(from: try! Tensor<Float>(numpy: conv_0_bias))
    let conv_2_weight = state_dict["\(prefix).conv.2.weight"].type(torch.float).cpu().numpy()
    let conv_2_bias = state_dict["\(prefix).conv.2.bias"].type(torch.float).cpu().numpy()
    conv.parameters(for: .index(2)).copy(from: try! Tensor<Float>(numpy: conv_2_weight))
    conv.parameters(for: .index(3)).copy(from: try! Tensor<Float>(numpy: conv_2_bias))
    let conv_4_weight = state_dict["\(prefix).conv.4.weight"].type(torch.float).cpu().numpy()
    let conv_4_bias = state_dict["\(prefix).conv.4.bias"].type(torch.float).cpu().numpy()
    conv.parameters(for: .index(4)).copy(from: try! Tensor<Float>(numpy: conv_4_weight))
    conv.parameters(for: .index(5)).copy(from: try! Tensor<Float>(numpy: conv_4_bias))
  }
  return (reader, Model([x], [out]))
}

func Decoder() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv1((x * (1.0 / 3)).tanh() * 3).ReLU()
  let (block1Reader, block1) = Block(prefix: "3", nOut: 64)
  out = block1(out)
  let (block2Reader, block2) = Block(prefix: "4", nOut: 64)
  out = block2(out)
  let (block3Reader, block3) = Block(prefix: "5", nOut: 64)
  out = block3(out)
  out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
  let conv2 = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  let (block4Reader, block4) = Block(prefix: "8", nOut: 64)
  out = block4(out)
  let (block5Reader, block5) = Block(prefix: "9", nOut: 64)
  out = block5(out)
  let (block6Reader, block6) = Block(prefix: "10", nOut: 64)
  out = block6(out)
  out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
  let conv3 = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv3(out)
  let (block7Reader, block7) = Block(prefix: "13", nOut: 64)
  out = block7(out)
  let (block8Reader, block8) = Block(prefix: "14", nOut: 64)
  out = block8(out)
  let (block9Reader, block9) = Block(prefix: "15", nOut: 64)
  out = block9(out)
  out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
  let conv4 = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv4(out)
  let (block10Reader, block10) = Block(prefix: "18", nOut: 64)
  out = block10(out)
  let conv5 = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv5(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_1_weight = state_dict["1.weight"].type(torch.float).cpu().numpy()
    let conv_1_bias = state_dict["1.bias"].type(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv_1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv_1_bias))
    block1Reader(state_dict)
    block2Reader(state_dict)
    block3Reader(state_dict)
    let conv_7_weight = state_dict["7.weight"].type(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv_7_weight))
    block4Reader(state_dict)
    block5Reader(state_dict)
    block6Reader(state_dict)
    let conv_12_weight = state_dict["12.weight"].type(torch.float).cpu().numpy()
    conv3.weight.copy(from: try! Tensor<Float>(numpy: conv_12_weight))
    block7Reader(state_dict)
    block8Reader(state_dict)
    block9Reader(state_dict)
    let conv_17_weight = state_dict["17.weight"].type(torch.float).cpu().numpy()
    conv4.weight.copy(from: try! Tensor<Float>(numpy: conv_17_weight))
    block10Reader(state_dict)
    let conv_19_weight = state_dict["19.weight"].type(torch.float).cpu().numpy()
    let conv_19_bias = state_dict["19.bias"].type(torch.float).cpu().numpy()
    conv5.weight.copy(from: try! Tensor<Float>(numpy: conv_19_weight))
    conv5.bias.copy(from: try! Tensor<Float>(numpy: conv_19_bias))
  }
  return (reader, Model([x], [out]))
}

print(sample_taesd)
let f = latent
let graph = DynamicGraph()
graph.withNoGrad {
  let fTensor = graph.variable(try! Tensor<Float>(numpy: f.detach().cpu().float().numpy())).toGPU(0)
  let (reader, decoder) = Decoder()
  decoder.compile(inputs: fTensor)
  reader(state_dict)
  var xStart = decoder(inputs: fTensor)[0].as(of: Float.self)
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/taesd_decoder_f32.ckpt") {
    $0.write("decoder", model: decoder)
  }
  */
  debugPrint(xStart)
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: 256 * 256)
  xStart = xStart.toCPU()
  for y in 0..<256 {
    for x in 0..<256 {
      let (r, g, b) = (xStart[0, 0, y, x], xStart[0, 1, y, x], xStart[0, 2, y, x])
      rgba[y * 256 + x].r = UInt8(
        min(max(Int(Float((r + 1) / 2) * 255), 0), 255))
      rgba[y * 256 + x].g = UInt8(
        min(max(Int(Float((g + 1) / 2) * 255), 0), 255))
      rgba[y * 256 + x].b = UInt8(
        min(max(Int(Float((b + 1) / 2) * 255), 0), 255))
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (256, 256),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/condecode.png", level: 4)
}
