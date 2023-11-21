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
let consistencydecoder = Python.import("consistencydecoder")

// encode with stable diffusion vae
let pipe = diffusers.StableDiffusionPipeline.from_pretrained(
  "runwayml/stable-diffusion-v1-5", torch_dtype: torch.float16
)
pipe.vae.cuda()

// construct original decoder with jitted model
let decoder_consistency = consistencydecoder.ConsistencyDecoder(device: "cuda:0")

// construct UNet code, overwrite the decoder with conv_unet_vae
var model = conv_unet_vae.ConvUNetVAE()
model.load_state_dict(
  conv_unet_vae.rename_state_dict(
    safetensors.torch.load_file(
      "/home/liu/workspace/consistency-decoder-sd15/consistency_decoder.safetensors"),
    safetensors.torch.load_file(
      "/home/liu/workspace/consistency-decoder-sd15/embedding.safetensors")
  )
)
model = model.cuda()
let state_dict = model.state_dict()
decoder_consistency.ckpt = model

let image = consistencydecoder.load_image(
  "/home/liu/workspace/consistencydecoder/assets/gt1.png", size: PythonObject(tupleOf: 256, 256),
  center_crop: true)
let latent = pipe.vae.encode(image.half().cuda()).latent_dist.sample()

// decode with gan
let sample_gan = pipe.vae.decode(latent).sample.detach()
consistencydecoder.save_image(sample_gan, "/home/liu/workspace/swift-diffusion/gan.png")

// decode with conv_unet_vae
let sample_consistency = decoder_consistency(latent)
consistencydecoder.save_image(sample_consistency, "/home/liu/workspace/swift-diffusion/con.png")

func TimestepEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, nTime: Int, nEmb: Int, nOut: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let timeEmbed = Embedding(
    T.self, vocabularySize: nTime, embeddingSize: nEmb)
  let f1 = Dense(count: nOut)
  let f2 = Dense(count: nOut)
  let out = f2(f1(timeEmbed(x)).swish())
  let reader: (PythonObject) -> Void = { state_dict in
    let emb_weight = state_dict["\(prefix).emb.weight"].type(torch.float).cpu().numpy()
    timeEmbed.weight.copy(from: try! Tensor<Float>(numpy: emb_weight))
    let f_1_weight = state_dict["\(prefix).f_1.weight"].type(torch.float).cpu().numpy()
    let f_1_bias = state_dict["\(prefix).f_1.bias"].type(torch.float).cpu().numpy()
    f1.weight.copy(from: try! Tensor<Float>(numpy: f_1_weight))
    f1.bias.copy(from: try! Tensor<Float>(numpy: f_1_bias))
    let f_2_weight = state_dict["\(prefix).f_2.weight"].type(torch.float).cpu().numpy()
    let f_2_bias = state_dict["\(prefix).f_2.bias"].type(torch.float).cpu().numpy()
    f2.weight.copy(from: try! Tensor<Float>(numpy: f_2_weight))
    f2.bias.copy(from: try! Tensor<Float>(numpy: f_2_bias))
  }
  return (Model([x], [out]), reader)
}

func ImageEmbedding(prefix: String, outChannels: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let f = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  let out = f(x)
  let reader: (PythonObject) -> Void = { state_dict in
    let f_weight = state_dict["\(prefix).f.weight"].type(torch.float).cpu().numpy()
    let f_bias = state_dict["\(prefix).f.bias"].type(torch.float).cpu().numpy()
    f.weight.copy(from: try! Tensor<Float>(numpy: f_weight))
    f.bias.copy(from: try! Tensor<Float>(numpy: f_bias))
  }
  return (Model([x], [out]), reader)
}

func ImageUnembedding(prefix: String, outChannels: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let gn = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  let f = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  let out = f(gn(x).swish())
  let reader: (PythonObject) -> Void = { state_dict in
    let gn_weight = state_dict["\(prefix).gn.weight"].type(torch.float).cpu().numpy()
    let gn_bias = state_dict["\(prefix).gn.bias"].type(torch.float).cpu().numpy()
    gn.weight.copy(from: try! Tensor<Float>(numpy: gn_weight))
    gn.bias.copy(from: try! Tensor<Float>(numpy: gn_bias))
    let f_weight = state_dict["\(prefix).f.weight"].type(torch.float).cpu().numpy()
    let f_bias = state_dict["\(prefix).f.bias"].type(torch.float).cpu().numpy()
    f.weight.copy(from: try! Tensor<Float>(numpy: f_weight))
    f.bias.copy(from: try! Tensor<Float>(numpy: f_bias))
  }
  return (Model([x], [out]), reader)
}

func ConvResblock(prefix: String, outFeatures: Int, skip: Bool) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let t = Input()
  let silut = t.swish()
  let ft1 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let t1 = ft1(silut) + 1
  let ft2 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let t2 = ft2(silut)
  let gn1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = gn1(x).swish()
  let f1 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = f1(out)
  let gn2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  let f2 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = f2((gn2(out) .* t1 + t2).swish())
  let fs: Model?
  if skip {
    let conv = Convolution(
      groups: 1, filters: outFeatures, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = conv(x) + out
    fs = conv
  } else {
    fs = nil
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let f_t_weight = state_dict["\(prefix).f_t.weight"].type(torch.float).cpu().numpy()
    let f_t_bias = state_dict["\(prefix).f_t.bias"].type(torch.float).cpu().numpy()
    ft1.weight.copy(from: try! Tensor<Float>(numpy: f_t_weight[0..<outFeatures, ...]))
    ft1.bias.copy(from: try! Tensor<Float>(numpy: f_t_bias[0..<outFeatures]))
    ft2.weight.copy(
      from: try! Tensor<Float>(numpy: f_t_weight[outFeatures..<(outFeatures * 2), ...]))
    ft2.bias.copy(from: try! Tensor<Float>(numpy: f_t_bias[outFeatures..<(outFeatures * 2)]))
    let gn_1_weight = state_dict["\(prefix).gn_1.weight"].type(torch.float).cpu().numpy()
    let gn_1_bias = state_dict["\(prefix).gn_1.bias"].type(torch.float).cpu().numpy()
    gn1.weight.copy(from: try! Tensor<Float>(numpy: gn_1_weight))
    gn1.bias.copy(from: try! Tensor<Float>(numpy: gn_1_bias))
    let f_1_weight = state_dict["\(prefix).f_1.weight"].type(torch.float).cpu().numpy()
    let f_1_bias = state_dict["\(prefix).f_1.bias"].type(torch.float).cpu().numpy()
    f1.weight.copy(from: try! Tensor<Float>(numpy: f_1_weight))
    f1.bias.copy(from: try! Tensor<Float>(numpy: f_1_bias))
    let gn_2_weight = state_dict["\(prefix).gn_2.weight"].type(torch.float).cpu().numpy()
    let gn_2_bias = state_dict["\(prefix).gn_2.bias"].type(torch.float).cpu().numpy()
    gn2.weight.copy(from: try! Tensor<Float>(numpy: gn_2_weight))
    gn2.bias.copy(from: try! Tensor<Float>(numpy: gn_2_bias))
    let f_2_weight = state_dict["\(prefix).f_2.weight"].type(torch.float).cpu().numpy()
    let f_2_bias = state_dict["\(prefix).f_2.bias"].type(torch.float).cpu().numpy()
    f2.weight.copy(from: try! Tensor<Float>(numpy: f_2_weight))
    f2.bias.copy(from: try! Tensor<Float>(numpy: f_2_bias))
    if let fs = fs {
      let f_s_weight = state_dict["\(prefix).f_s.weight"].type(torch.float).cpu().numpy()
      let f_s_bias = state_dict["\(prefix).f_s.bias"].type(torch.float).cpu().numpy()
      fs.weight.copy(from: try! Tensor<Float>(numpy: f_s_weight))
      fs.bias.copy(from: try! Tensor<Float>(numpy: f_s_bias))
    }
  }
  return (Model([x, t], [out]), reader)
}

func Downsample(prefix: String, outFeatures: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let t = Input()
  let silut = t.swish()
  let ft1 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let t1 = ft1(silut) + 1
  let ft2 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let t2 = ft2(silut)
  let gn1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(gn1(x).swish())
  let f1 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = f1(out)
  let gn2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  let f2 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = f2((gn2(out) .* t1 + t2).swish())
  out = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))(x) + out
  let reader: (PythonObject) -> Void = { state_dict in
    let f_t_weight = state_dict["\(prefix).f_t.weight"].type(torch.float).cpu().numpy()
    let f_t_bias = state_dict["\(prefix).f_t.bias"].type(torch.float).cpu().numpy()
    ft1.weight.copy(from: try! Tensor<Float>(numpy: f_t_weight[0..<outFeatures, ...]))
    ft1.bias.copy(from: try! Tensor<Float>(numpy: f_t_bias[0..<outFeatures]))
    ft2.weight.copy(
      from: try! Tensor<Float>(numpy: f_t_weight[outFeatures..<(outFeatures * 2), ...]))
    ft2.bias.copy(from: try! Tensor<Float>(numpy: f_t_bias[outFeatures..<(outFeatures * 2)]))
    let gn_1_weight = state_dict["\(prefix).gn_1.weight"].type(torch.float).cpu().numpy()
    let gn_1_bias = state_dict["\(prefix).gn_1.bias"].type(torch.float).cpu().numpy()
    gn1.weight.copy(from: try! Tensor<Float>(numpy: gn_1_weight))
    gn1.bias.copy(from: try! Tensor<Float>(numpy: gn_1_bias))
    let f_1_weight = state_dict["\(prefix).f_1.weight"].type(torch.float).cpu().numpy()
    let f_1_bias = state_dict["\(prefix).f_1.bias"].type(torch.float).cpu().numpy()
    f1.weight.copy(from: try! Tensor<Float>(numpy: f_1_weight))
    f1.bias.copy(from: try! Tensor<Float>(numpy: f_1_bias))
    let gn_2_weight = state_dict["\(prefix).gn_2.weight"].type(torch.float).cpu().numpy()
    let gn_2_bias = state_dict["\(prefix).gn_2.bias"].type(torch.float).cpu().numpy()
    gn2.weight.copy(from: try! Tensor<Float>(numpy: gn_2_weight))
    gn2.bias.copy(from: try! Tensor<Float>(numpy: gn_2_bias))
    let f_2_weight = state_dict["\(prefix).f_2.weight"].type(torch.float).cpu().numpy()
    let f_2_bias = state_dict["\(prefix).f_2.bias"].type(torch.float).cpu().numpy()
    f2.weight.copy(from: try! Tensor<Float>(numpy: f_2_weight))
    f2.bias.copy(from: try! Tensor<Float>(numpy: f_2_bias))
  }
  return (Model([x, t], [out]), reader)
}

func Upsample(prefix: String, outFeatures: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let t = Input()
  let silut = t.swish()
  let ft1 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let t1 = ft1(silut) + 1
  let ft2 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let t2 = ft2(silut)
  let gn1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = Upsample(.nearest, widthScale: 2, heightScale: 2)(gn1(x).swish())
  let f1 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = f1(out)
  let gn2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  let f2 = Convolution(
    groups: 1, filters: outFeatures, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = f2((gn2(out) .* t1 + t2).swish())
  out = Upsample(.nearest, widthScale: 2, heightScale: 2)(x) + out
  let reader: (PythonObject) -> Void = { state_dict in
    let f_t_weight = state_dict["\(prefix).f_t.weight"].type(torch.float).cpu().numpy()
    let f_t_bias = state_dict["\(prefix).f_t.bias"].type(torch.float).cpu().numpy()
    ft1.weight.copy(from: try! Tensor<Float>(numpy: f_t_weight[0..<outFeatures, ...]))
    ft1.bias.copy(from: try! Tensor<Float>(numpy: f_t_bias[0..<outFeatures]))
    ft2.weight.copy(
      from: try! Tensor<Float>(numpy: f_t_weight[outFeatures..<(outFeatures * 2), ...]))
    ft2.bias.copy(from: try! Tensor<Float>(numpy: f_t_bias[outFeatures..<(outFeatures * 2)]))
    let gn_1_weight = state_dict["\(prefix).gn_1.weight"].type(torch.float).cpu().numpy()
    let gn_1_bias = state_dict["\(prefix).gn_1.bias"].type(torch.float).cpu().numpy()
    gn1.weight.copy(from: try! Tensor<Float>(numpy: gn_1_weight))
    gn1.bias.copy(from: try! Tensor<Float>(numpy: gn_1_bias))
    let f_1_weight = state_dict["\(prefix).f_1.weight"].type(torch.float).cpu().numpy()
    let f_1_bias = state_dict["\(prefix).f_1.bias"].type(torch.float).cpu().numpy()
    f1.weight.copy(from: try! Tensor<Float>(numpy: f_1_weight))
    f1.bias.copy(from: try! Tensor<Float>(numpy: f_1_bias))
    let gn_2_weight = state_dict["\(prefix).gn_2.weight"].type(torch.float).cpu().numpy()
    let gn_2_bias = state_dict["\(prefix).gn_2.bias"].type(torch.float).cpu().numpy()
    gn2.weight.copy(from: try! Tensor<Float>(numpy: gn_2_weight))
    gn2.bias.copy(from: try! Tensor<Float>(numpy: gn_2_bias))
    let f_2_weight = state_dict["\(prefix).f_2.weight"].type(torch.float).cpu().numpy()
    let f_2_bias = state_dict["\(prefix).f_2.bias"].type(torch.float).cpu().numpy()
    f2.weight.copy(from: try! Tensor<Float>(numpy: f_2_weight))
    f2.bias.copy(from: try! Tensor<Float>(numpy: f_2_bias))
  }
  return (Model([x, t], [out]), reader)
}

func ConvUNetVAE<T: TensorNumeric>(_ dataType: T.Type) -> (Model, (PythonObject) -> Void) {
  let embedImage = ImageEmbedding(prefix: "embed_image", outChannels: 320)
  let embedTime = TimestepEmbedding(
    dataType, prefix: "embed_time", nTime: 1024, nEmb: 320, nOut: 1280)
  let down0 = [
    ConvResblock(prefix: "down.0.0", outFeatures: 320, skip: false),
    ConvResblock(prefix: "down.0.1", outFeatures: 320, skip: false),
    ConvResblock(prefix: "down.0.2", outFeatures: 320, skip: false),
    Downsample(prefix: "down.0.3", outFeatures: 320),
  ]
  let down1 = [
    ConvResblock(prefix: "down.1.0", outFeatures: 640, skip: true),
    ConvResblock(prefix: "down.1.1", outFeatures: 640, skip: false),
    ConvResblock(prefix: "down.1.2", outFeatures: 640, skip: false),
    Downsample(prefix: "down.1.3", outFeatures: 640),
  ]
  let down2 = [
    ConvResblock(prefix: "down.2.0", outFeatures: 1024, skip: true),
    ConvResblock(prefix: "down.2.1", outFeatures: 1024, skip: false),
    ConvResblock(prefix: "down.2.2", outFeatures: 1024, skip: false),
    Downsample(prefix: "down.2.3", outFeatures: 1024),
  ]
  let down3 = [
    ConvResblock(prefix: "down.3.0", outFeatures: 1024, skip: false),
    ConvResblock(prefix: "down.3.1", outFeatures: 1024, skip: false),
    ConvResblock(prefix: "down.3.2", outFeatures: 1024, skip: false),
  ]
  let mid = [
    ConvResblock(prefix: "mid.0", outFeatures: 1024, skip: false),
    ConvResblock(prefix: "mid.1", outFeatures: 1024, skip: false),
  ]
  let up3 = [
    ConvResblock(prefix: "up.3.0", outFeatures: 1024, skip: true),
    ConvResblock(prefix: "up.3.1", outFeatures: 1024, skip: true),
    ConvResblock(prefix: "up.3.2", outFeatures: 1024, skip: true),
    ConvResblock(prefix: "up.3.3", outFeatures: 1024, skip: true),
    Upsample(prefix: "up.3.4", outFeatures: 1024),
  ]
  let up2 = [
    ConvResblock(prefix: "up.2.0", outFeatures: 1024, skip: true),
    ConvResblock(prefix: "up.2.1", outFeatures: 1024, skip: true),
    ConvResblock(prefix: "up.2.2", outFeatures: 1024, skip: true),
    ConvResblock(prefix: "up.2.3", outFeatures: 1024, skip: true),
    Upsample(prefix: "up.2.4", outFeatures: 1024),
  ]
  let up1 = [
    ConvResblock(prefix: "up.1.0", outFeatures: 640, skip: true),
    ConvResblock(prefix: "up.1.1", outFeatures: 640, skip: true),
    ConvResblock(prefix: "up.1.2", outFeatures: 640, skip: true),
    ConvResblock(prefix: "up.1.3", outFeatures: 640, skip: true),
    Upsample(prefix: "up.1.4", outFeatures: 640),
  ]
  let up0 = [
    ConvResblock(prefix: "up.0.0", outFeatures: 320, skip: true),
    ConvResblock(prefix: "up.0.1", outFeatures: 320, skip: true),
    ConvResblock(prefix: "up.0.2", outFeatures: 320, skip: true),
    ConvResblock(prefix: "up.0.3", outFeatures: 320, skip: true),
  ]
  let output = ImageUnembedding(prefix: "output", outChannels: 6)
  let x = Input()
  let t = Input()
  let tEmb = embedTime.0(t).reshaped(.NCHW(1, 1280, 1, 1))
  let xEmb = embedImage.0(x)
  var skips = [xEmb]
  var out = xEmb
  for block in down0 {
    out = block.0(out, tEmb)
    skips.append(out)
  }
  for block in down1 {
    out = block.0(out, tEmb)
    skips.append(out)
  }
  for block in down2 {
    out = block.0(out, tEmb)
    skips.append(out)
  }
  for block in down3 {
    out = block.0(out, tEmb)
    skips.append(out)
  }
  for block in mid {
    out = block.0(out, tEmb)
  }
  for block in up3[0..<4] {
    out = block.0(Functional.concat(axis: 1, out, skips.removeLast()), tEmb)
  }
  out = up3[4].0(out, tEmb)
  for block in up2[0..<4] {
    out = block.0(Functional.concat(axis: 1, out, skips.removeLast()), tEmb)
  }
  out = up2[4].0(out, tEmb)
  for block in up1[0..<4] {
    out = block.0(Functional.concat(axis: 1, out, skips.removeLast()), tEmb)
  }
  out = up1[4].0(out, tEmb)
  for block in up0 {
    out = block.0(Functional.concat(axis: 1, out, skips.removeLast()), tEmb)
  }
  assert(skips.count == 0)
  out = output.0(out)
  let reader: (PythonObject) -> Void = { state_dict in
    embedImage.1(state_dict)
    embedTime.1(state_dict)
    for block in down0 {
      block.1(state_dict)
    }
    for block in down1 {
      block.1(state_dict)
    }
    for block in down2 {
      block.1(state_dict)
    }
    for block in down3 {
      block.1(state_dict)
    }
    for block in mid {
      block.1(state_dict)
    }
    for block in up3 {
      block.1(state_dict)
    }
    for block in up2 {
      block.1(state_dict)
    }
    for block in up1 {
      block.1(state_dict)
    }
    for block in up0 {
      block.1(state_dict)
    }
    output.1(state_dict)
  }
  return (Model([x, t], [out]), reader)
}

func betasForAlphaBar(timesteps: Int, maxBeta: Double = 0.999) -> [Double] {
  var betas = [Double]()
  for i in 0..<timesteps {
    let t1 = Double(i) / Double(timesteps)
    let t2 = Double(i + 1) / Double(timesteps)
    var alphaBarT1 = cos((t1 + 0.008) / 1.008 * .pi / 2)
    alphaBarT1 = alphaBarT1 * alphaBarT1
    var alphaBarT2 = cos((t2 + 0.008) / 1.008 * .pi / 2)
    alphaBarT2 = alphaBarT2 * alphaBarT2
    betas.append(min(1 - alphaBarT2 / alphaBarT1, maxBeta))
  }
  return betas
}

func roundTimesteps(timesteps: Int, nDistilledSteps: Int, truncateStart: Bool = false) -> [Int] {
  let space = timesteps / nDistilledSteps
  var roundedTimesteps = [Int]()
  for i in 0..<timesteps {
    var timestep = (i / space + 1) * space
    if timestep == timesteps {
      timestep -= space
    }
    if !truncateStart {
      if timestep == 0 {
        timestep += space
      }
    }
    roundedTimesteps.append(timestep)
  }
  return roundedTimesteps
}

let betas = betasForAlphaBar(timesteps: 1024)
var cumprod: Double = 1
let alphasCumprod = betas.map {
  cumprod *= 1 - $0
  return cumprod
}
let roundedTimesteps = roundTimesteps(timesteps: 1024, nDistilledSteps: 64)
let schedule = [1.0, 0.75, 0.5, 0.25]
let sigmaData = 0.5

let f = latent
let graph = DynamicGraph()
DynamicGraph.setSeed(40)
graph.withNoGrad {
  var fTensor = graph.variable(try! Tensor<Float>(numpy: f.detach().cpu().float().numpy())).toGPU(0)
  fTensor = 0.18215 * fTensor
  let channelMeans = graph.variable(
    Tensor<Float>([-0.38862467, -0.02253063, -0.07381133, 0.0171294], .CPU, .NCHW(1, 4, 1, 1))
      .toGPU(0))
  let channelStds = graph.variable(
    Tensor<Float>(
      [1.0 / 0.9654121, 1.0 / 1.0440036, 1.0 / 0.76147926, 1.0 / 0.77022034], .CPU,
      .NCHW(1, 4, 1, 1)
    ).toGPU(0))
  fTensor = (fTensor + channelMeans) .* channelStds
  let f8Tensor = DynamicGraph.Tensor<Float16>(
    from: Upsample(.nearest, widthScale: 8, heightScale: 8)(inputs: fTensor)[0].as(
      of: Float.self))
  let (convVAE, reader) = ConvUNetVAE(Float16.self)
  let noise = graph.variable(.GPU(0), .NCHW(1, 3, 256, 256), of: Float.self)
  noise.randn()
  var xStart = noise
  let combined = Concat(axis: 1)(
    inputs: DynamicGraph.Tensor<Float16>(from: (1.0002 * 0.9997) * xStart), f8Tensor)[0].as(
      of: Float16.self)
  let tTensor = graph.variable(.CPU, .C(1), of: Int32.self)
  tTensor[0] = 1008
  let tTensorGPU = tTensor.toGPU(0)
  convVAE.compile(inputs: combined, tTensorGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/consistencydecoder_f32.ckpt") {
    $0.read("conv_unet_vae", model: convVAE)
  }
  // reader(state_dict)
  let scheduleTimesteps = schedule.map { Int(((1024 - 1) * $0).rounded()) }
  for (index, i) in scheduleTimesteps.enumerated() {
    let timestep = roundedTimesteps[i]
    let alphaCumprod = alphasCumprod[timestep]
    let sqrtAlphaCumprod = alphaCumprod.squareRoot()
    let sqrtOneMinusAlphaCumprod = (1 - alphaCumprod).squareRoot()
    let sqrtRecipAlphaCumprod = (1.0 / alphaCumprod).squareRoot()
    let sigma = (1.0 / alphaCumprod - 1).squareRoot()
    let cSkip =
      sqrtRecipAlphaCumprod * sigmaData * sigmaData / (sigma * sigma + sigmaData * sigmaData)
    let cOut = sigma * sigmaData / (sigma * sigma + sigmaData * sigmaData).squareRoot()
    let cIn = sqrtRecipAlphaCumprod / (sigma * sigma + sigmaData * sigmaData).squareRoot()
    print(
      "timestep \(timestep) alphaCumprod \(alphaCumprod) sqrtAlphaCumprod \(sqrtAlphaCumprod) cOut \(cOut) cSkip \(cSkip) cIn \(cIn)"
    )
    if index > 0 {
      noise.randn()
    }
    xStart = Functional.add(
      left: xStart, right: noise, leftScalar: Float(sqrtAlphaCumprod),
      rightScalar: Float(sqrtOneMinusAlphaCumprod))
    let combined = Concat(axis: 1)(
      inputs: DynamicGraph.Tensor<Float16>(from: Float(cIn) * xStart), f8Tensor)[0].as(
        of: Float16.self)
    tTensor[0] = Int32(timestep)
    let tTensorGPU = tTensor.toGPU(0)
    let out = DynamicGraph.Tensor<Float>(
      from: convVAE(inputs: combined, tTensorGPU)[0].as(of: Float16.self)[
        0..<1, 0..<3, 0..<256, 0..<256
      ].copied())
    xStart = Functional.add(
      left: out, right: xStart, leftScalar: Float(cOut), rightScalar: Float(cSkip)
    )
    .clamped(-1...1)
  }
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
