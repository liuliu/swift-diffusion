import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit
import SentencePiece

typealias FloatType = Float16

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

let sys = Python.import("sys")
let osPath = Python.import("os.path")
let site = Python.import("site")
let userSitePackages = site.getusersitepackages()
if (Bool(osPath.isdir(userSitePackages)) ?? false)
  && (Bool(sys.path.__contains__(userSitePackages)) ?? false) == false
{
  sys.path.insert(0, userSitePackages)
}
let systemDistPackages = "/usr/lib/python3/dist-packages"
if (Bool(osPath.isdir(systemDistPackages)) ?? false)
  && (Bool(sys.path.__contains__(systemDistPackages)) ?? false) == false
{
  sys.path.insert(0, systemDistPackages)
}

let torch = Python.import("torch")
let json = Python.import("json")
let safetensors = Python.import("safetensors")
let safetensors_torch = Python.import("safetensors.torch")

torch.set_grad_enabled(false)

let hasCUDA = Bool(torch.cuda.is_available()) ?? false
let torch_device = hasCUDA ? torch.device("cuda") : torch.device("cpu")
let pythonDevice = hasCUDA ? "cuda" : "cpu"
let swiftDevice = 1

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
if hasCUDA {
  torch.cuda.manual_seed_all(42)
  torch.backends.cuda.matmul.allow_tf32 = false
  torch.backends.cudnn.allow_tf32 = false
}

let freshLTX2CorePath = "/home/liu/workspace/ltx2/LTX-2/packages/ltx-core/src"
if (Bool(sys.path.__contains__(freshLTX2CorePath)) ?? false) == false {
  sys.path.insert(0, freshLTX2CorePath)
}

let ltx_core_model_upsampler_model_configurator = Python.import(
  "ltx_core.model.upsampler.model_configurator")

func LTX2SpatialResBlock3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    name: "conv1")
  var out = conv1(x)
  let norm1 = GroupNorm(axis: 0, groups: 32, epsilon: 1e-5, reduce: [1, 2, 3], name: "norm1")
  out = norm1(out.reshaped([channels, depth, height, width])).reshaped([
    1, channels, depth, height, width,
  ])
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    name: "conv2")
  out = conv2(out)
  let norm2 = GroupNorm(axis: 0, groups: 32, epsilon: 1e-5, reduce: [1, 2, 3], name: "norm2")
  out = norm2(out.reshaped([channels, depth, height, width])).reshaped([
    1, channels, depth, height, width,
  ])
  out = (out + x).swish()
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1Weight = state_dict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1Bias = state_dict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1Weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1Bias))
    let norm1Weight = state_dict["\(prefix).norm1.weight"].to(torch.float).cpu().numpy()
    let norm1Bias = state_dict["\(prefix).norm1.bias"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1Weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1Bias))
    let conv2Weight = state_dict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2Bias = state_dict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2Weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2Bias))
    let norm2Weight = state_dict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2Bias = state_dict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2Weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2Bias))
  }
  return (reader, Model([x], [out]))
}

enum SpatialUpscalerMode {
  case x2
  case x1_5
}

func LTX2SpatialUpscaler3D(
  inChannels: Int, midChannels: Int, numBlocks: Int, depth: Int, height: Int, width: Int,
  mode: SpatialUpscalerMode
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let initialConv = Convolution(
    groups: 1, filters: midChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    name: "initial_conv")
  var out = initialConv(x)
  let initialNorm = GroupNorm(
    axis: 0, groups: 32, epsilon: 1e-5, reduce: [1, 2, 3], name: "initial_norm")
  out = initialNorm(out.reshaped([midChannels, depth, height, width])).reshaped([
    1, midChannels, depth, height, width,
  ])
  out = out.swish()
  var readers = [(PythonObject) -> Void]()
  for i in 0..<numBlocks {
    let (reader, block) = LTX2SpatialResBlock3D(
      prefix: "res_blocks.\(i)", channels: midChannels, depth: depth, height: height, width: width)
    out = block(out)
    readers.append(reader)
  }

  let upsampleConv: Convolution
  let upsampleWeightKey: String
  let upsampleBiasKey: String
  let postHeight: Int
  let postWidth: Int
  let blurDown: Convolution?
  switch mode {
  case .x2:
    upsampleConv = Convolution(
      groups: 1, filters: midChannels * 4, filterSize: [1, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      name: "upsampler_conv")
    out = upsampleConv(out)
    out = out.reshaped([1, midChannels, 2, 2, depth, height, width]).permuted(
      0, 1, 4, 5, 2, 6, 3
    ).contiguous()
    out = out.reshaped([1, midChannels, depth, height * 2, width * 2])
    upsampleWeightKey = "upsampler.0.weight"
    upsampleBiasKey = "upsampler.0.bias"
    postHeight = height * 2
    postWidth = width * 2
    blurDown = nil
  case .x1_5:
    precondition(height % 2 == 0 && width % 2 == 0)
    upsampleConv = Convolution(
      groups: 1, filters: midChannels * 9, filterSize: [1, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      name: "upsampler_conv")
    out = upsampleConv(out)
    out = out.reshaped([1, midChannels, 3, 3, depth, height, width]).permuted(
      0, 1, 4, 5, 2, 6, 3
    ).contiguous()
    out = out.reshaped([1, midChannels, depth, height * 3, width * 3])
    let blur = Convolution(
      groups: midChannels, filters: midChannels, filterSize: [1, 5, 5],
      hint: Hint(stride: [1, 2, 2], border: Hint.Border(begin: [0, 2, 2], end: [0, 2, 2])),
      name: "upsampler_blur_down")
    out = blur(out)
    upsampleWeightKey = "upsampler.conv.weight"
    upsampleBiasKey = "upsampler.conv.bias"
    postHeight = height * 3 / 2
    postWidth = width * 3 / 2
    blurDown = blur
  }

  for i in 0..<numBlocks {
    let (reader, block) = LTX2SpatialResBlock3D(
      prefix: "post_upsample_res_blocks.\(i)", channels: midChannels, depth: depth,
      height: postHeight, width: postWidth)
    out = block(out)
    readers.append(reader)
  }
  let finalConv = Convolution(
    groups: 1, filters: inChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    name: "final_conv")
  out = finalConv(out)

  let reader: (PythonObject) -> Void = { state_dict in
    let initialConvWeight = state_dict["initial_conv.weight"].to(torch.float).cpu().numpy()
    let initialConvBias = state_dict["initial_conv.bias"].to(torch.float).cpu().numpy()
    initialConv.weight.copy(from: try! Tensor<Float>(numpy: initialConvWeight))
    initialConv.bias.copy(from: try! Tensor<Float>(numpy: initialConvBias))
    let initialNormWeight = state_dict["initial_norm.weight"].to(torch.float).cpu().numpy()
    let initialNormBias = state_dict["initial_norm.bias"].to(torch.float).cpu().numpy()
    initialNorm.weight.copy(from: try! Tensor<Float>(numpy: initialNormWeight))
    initialNorm.bias.copy(from: try! Tensor<Float>(numpy: initialNormBias))
    for reader in readers {
      reader(state_dict)
    }
    let upsampleConvWeight = state_dict[upsampleWeightKey].to(torch.float).unsqueeze(2).cpu()
      .numpy()
    let upsampleConvBias = state_dict[upsampleBiasKey].to(torch.float).cpu().numpy()
    upsampleConv.weight.copy(from: try! Tensor<Float>(numpy: upsampleConvWeight))
    upsampleConv.bias.copy(from: try! Tensor<Float>(numpy: upsampleConvBias))
    if let blurDown = blurDown {
      let blurDownKernel = state_dict["upsampler.blur_down.kernel"].to(torch.float).unsqueeze(2)
      // Materialize broadcasted depthwise kernel explicitly for grouped 3D conv weights.
      let kernelH = Int(blurDownKernel.shape[3])!
      let kernelW = Int(blurDownKernel.shape[4])!
      let blurDownKernelExpanded = blurDownKernel.expand([midChannels, 1, 1, kernelH, kernelW])
        .contiguous()
      blurDown.weight.copy(from: try! Tensor<Float>(numpy: blurDownKernelExpanded.cpu().numpy()))
      let blurDownBias = torch.zeros([midChannels]).to(torch.float).cpu().numpy()
      blurDown.bias.copy(from: try! Tensor<Float>(numpy: blurDownBias))
    }
    let finalConvWeight = state_dict["final_conv.weight"].to(torch.float).cpu().numpy()
    let finalConvBias = state_dict["final_conv.bias"].to(torch.float).cpu().numpy()
    finalConv.weight.copy(from: try! Tensor<Float>(numpy: finalConvWeight))
    finalConv.bias.copy(from: try! Tensor<Float>(numpy: finalConvBias))
  }
  return (reader, Model([x], [out]))
}

func convertLTX2SpatialUpscaler(
  modelPath: String, ckptPath: String, testDepth: Int, testHeight: Int, testWidth: Int
) {
  let upsamplerConfig = json.loads(
    safetensors.safe_open(modelPath, framework: "pt", device: "cpu").metadata()["config"])
  let spatialUpscaler = ltx_core_model_upsampler_model_configurator.LatentUpsamplerConfigurator
    .from_config(upsamplerConfig)
  let spatialUpscalerStateDict = safetensors_torch.load_file(modelPath, device: "cpu")
  _ = spatialUpscaler.load_state_dict(spatialUpscalerStateDict, strict: false)
  spatialUpscaler.to(torch_device)
  spatialUpscaler.eval()
  spatialUpscaler.to(torch.float)

  let inChannels = Int(spatialUpscaler.in_channels)!
  let midChannels = Int(spatialUpscaler.mid_channels)!
  let numBlocks = Int(spatialUpscaler.num_blocks_per_stage)!
  let rationalResampler = Bool(spatialUpscaler.rational_resampler) ?? false
  let spatialScale = Double(spatialUpscaler.spatial_scale) ?? 2.0
  let mode: SpatialUpscalerMode
  let modeLabel: String
  if rationalResampler && abs(spatialScale - 1.5) < 1e-6 {
    mode = .x1_5
    modeLabel = "x1.5"
  } else if !rationalResampler && abs(spatialScale - 2.0) < 1e-6 {
    mode = .x2
    modeLabel = "x2"
  } else {
    fatalError("Unsupported spatial upscaler mode for \(modelPath)")
  }

  let latent = torch.randn([1, inChannels, testDepth, testHeight, testWidth]).to(torch.float).to(
    torch_device)
  let upscaled = spatialUpscaler(latent).to(torch.float)
  print(
    "[\(modeLabel)] input shape: \(latent.shape), torch output shape: \(upscaled.shape), model: \(modelPath)"
  )

  graph.withNoGrad {
    let latentTensor = graph.variable(
      try! Tensor<Float>(numpy: latent.to(torch.float).cpu().numpy())
    ).toGPU(1)
    let (reader, swiftUpscaler) = LTX2SpatialUpscaler3D(
      inChannels: inChannels, midChannels: midChannels, numBlocks: numBlocks, depth: testDepth,
      height: testHeight, width: testWidth, mode: mode)
    swiftUpscaler.compile(inputs: latentTensor)
    reader(spatialUpscaler.state_dict())
    let swiftUpscaled = swiftUpscaler(inputs: latentTensor)[0].as(of: Float.self).toCPU()
    let torchUpscaled = try! Tensor<Float>(numpy: upscaled.cpu().numpy())
    print("[\(modeLabel)] swift output shape: \(swiftUpscaled.shape)")
    debugPrint(swiftUpscaled[0..<1, 0..<1, 0..<1, 0..<2, 0..<2])
    debugPrint(torchUpscaled[0..<1, 0..<1, 0..<1, 0..<2, 0..<2])
    graph.openStore(ckptPath) {
      $0.write("spatial_upsampler", model: swiftUpscaler)
    }
    print("Wrote \(ckptPath)")
  }
}

let testDepth = 16
let testHeight = 16
let testWidth = 16
convertLTX2SpatialUpscaler(
  modelPath: "/fast/Data/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
  ckptPath: "/home/liu/workspace/swift-diffusion/ltx_2.3_spatial_upscaler_x2_1.1_f32.ckpt",
  testDepth: testDepth, testHeight: testHeight, testWidth: testWidth)
exit(0)

let ltx_core_loader_single_gpu_model_builder = Python.import(
  "ltx_core.loader.single_gpu_model_builder")
let ltx_core_model_audio_vae_model_configurator = Python.import(
  "ltx_core.model.audio_vae.model_configurator")
let ltx23ModelPath = "/fast/Data/ltx-2.3-22b-dev.safetensors"
let audio_encoder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: ltx23ModelPath,
  model_class_configurator: ltx_core_model_audio_vae_model_configurator.AudioEncoderConfigurator,
  model_sd_ops: ltx_core_model_audio_vae_model_configurator.AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER
).build(device: pythonDevice)
let audio_decoder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: ltx23ModelPath,
  model_class_configurator: ltx_core_model_audio_vae_model_configurator.AudioDecoderConfigurator,
  model_sd_ops: ltx_core_model_audio_vae_model_configurator.AUDIO_VAE_DECODER_COMFY_KEYS_FILTER
).build(device: pythonDevice)
let audio_vocoder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: ltx23ModelPath,
  model_class_configurator: ltx_core_model_audio_vae_model_configurator.VocoderConfigurator,
  model_sd_ops: ltx_core_model_audio_vae_model_configurator.VOCODER_COMFY_KEYS_FILTER
).build(device: pythonDevice)
let a = torch.randn([1, 8, 121, 16]).to(torch.float).to(torch_device)
audio_decoder.to(torch.float)
audio_decoder.eval()
audio_encoder.to(torch.float)
audio_encoder.eval()
audio_vocoder.to(torch.float)
audio_vocoder.eval()
var decoded_audio: PythonObject = Python.None
var encoded_audio: PythonObject = Python.None
if false {
  decoded_audio = audio_decoder(a)
  encoded_audio = audio_encoder(decoded_audio)
  print("audio_decoder output shape:", decoded_audio.shape)
  print("audio_encoder output shape:", encoded_audio.shape)
}

func ResnetBlockCausal2D(prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-8, axis: [1], elementwiseAffine: false, name: "resnet_norm1")
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    name: "resnet_conv1")
  out = conv1(out.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  let norm2 = RMSNorm(epsilon: 1e-8, axis: [1], elementwiseAffine: false, name: "resnet_norm2")
  out = norm2(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    name: "resnet_conv2")
  out = conv2(out.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      name: "resnet_shortcut")
    out = nin(x) + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1_weight = state_dict["\(prefix).conv1.conv.weight"].to(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.conv.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let conv2_weight = state_dict["\(prefix).conv2.conv.weight"].to(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.conv.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).nin_shortcut.conv.weight"].to(torch.float)
        .cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).nin_shortcut.conv.bias"].to(torch.float).cpu()
        .numpy()
      ninShortcut.weight.copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
  }
  return (reader, Model([x], [out]))
}

func EncoderCausal2D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    name: "conv_in")
  var out = convIn(x.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  var readers = [(PythonObject) -> Void]()
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (reader, block) = ResnetBlockCausal2D(
        prefix: "down.\(i).block.\(j)", inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel)
      out = block(out)
      readers.append(reader)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      let conv = Convolution(
        groups: 1, filters: previousChannel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])),
        name: "upsample")
      out = conv(out.padded(.zero, begin: [0, 0, 2, 0], end: [0, 0, 0, 1]))
      let downBlocks = i
      readers.append { state_dict in
        let conv_weight = state_dict["down.\(downBlocks).downsample.conv.weight"].to(torch.float)
          .cpu().numpy()
        let conv_bias = state_dict["down.\(downBlocks).downsample.conv.bias"].to(torch.float).cpu()
          .numpy()
        conv.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
    }
  }
  let (midResnetBlock1Reader, midResnetBlock1) = ResnetBlockCausal2D(
    prefix: "mid.block_1", inChannels: previousChannel, outChannels: previousChannel,
    shortcut: false)
  out = midResnetBlock1(out)
  readers.append(midResnetBlock1Reader)
  let (midResnetBlock2Reader, midResnetBlock2) = ResnetBlockCausal2D(
    prefix: "mid.block_2", inChannels: previousChannel, outChannels: previousChannel,
    shortcut: false)
  out = midResnetBlock2(out)
  readers.append(midResnetBlock2Reader)
  let normOut = RMSNorm(epsilon: 1e-8, axis: [1], elementwiseAffine: false, name: "norm_out")
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: 16, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  out = out.reshaped(
    [1, 8, startHeight, startWidth], offset: [0, 0, 0, 0],
    strides: [16 * startHeight * startWidth, startHeight * startWidth, startWidth, 1]
  ).contiguous()
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["conv_in.conv.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["conv_in.conv.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    let conv_out_weight = state_dict["conv_out.conv.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["conv_out.conv.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

func DecoderCausal2D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    name: "conv_in")
  var out = convIn(x.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  var readers = [(PythonObject) -> Void]()
  let (midResnetBlock1Reader, midResnetBlock1) = ResnetBlockCausal2D(
    prefix: "mid.block_1", inChannels: previousChannel, outChannels: previousChannel,
    shortcut: false)
  out = midResnetBlock1(out)
  readers.append(midResnetBlock1Reader)
  let (midResnetBlock2Reader, midResnetBlock2) = ResnetBlockCausal2D(
    prefix: "mid.block_2", inChannels: previousChannel, outChannels: previousChannel,
    shortcut: false)
  out = midResnetBlock2(out)
  readers.append(midResnetBlock2Reader)
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat {
      let (reader, block) = ResnetBlockCausal2D(
        prefix: "up.\(i).block.\(j)", inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel)
      out = block(out)
      readers.append(reader)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv = Convolution(
        groups: 1, filters: previousChannel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
        name: "upsample")
      out = conv(out.padded(.zero, begin: [0, 0, 1, 1], end: [0, 0, 0, 1]))
      let upBlocks = i
      readers.append { state_dict in
        let conv_weight = state_dict["up.\(upBlocks).upsample.conv.conv.weight"].to(torch.float)
          .cpu().numpy()
        let conv_bias = state_dict["up.\(upBlocks).upsample.conv.conv.bias"].to(torch.float).cpu()
          .numpy()
        conv.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
    }
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [1], elementwiseAffine: false, name: "norm_out")
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: 2, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["conv_in.conv.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["conv_in.conv.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    let conv_out_weight = state_dict["conv_out.conv.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["conv_out.conv.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

func SnakeBeta(prefix: String, channels: Int, name: String) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let alpha = Parameter<Float>(
    .GPU(swiftDevice), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_alpha")
  let beta = Parameter<Float>(
    .GPU(swiftDevice), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_beta")
  let out = x + beta .* (x .* alpha).sin().pow(2)
  let reader: (PythonObject) -> Void = { state_dict in
    let alphaTensor = state_dict["\(prefix).alpha"].to(torch.float).exp().view(1, -1, 1, 1).cpu()
      .numpy()
    let betaTensor = (state_dict["\(prefix).beta"].to(torch.float).exp() + 1e-9).pow(-1).view(
      1, -1, 1, 1
    ).cpu().numpy()
    alpha.weight.copy(from: try! Tensor<Float>(numpy: alphaTensor))
    beta.weight.copy(from: try! Tensor<Float>(numpy: betaTensor))
  }
  return (reader, Model([x], [out]))
}

func Activation1d(
  prefix: String, channels: Int, width: Int, upRatio: Int = 2, downRatio: Int = 2,
  upKernelSize: Int = 12,
  downKernelSize: Int = 12, name: String = ""
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let upWidth = width * upRatio
  let upPad = upKernelSize / upRatio - 1
  let upInputWidth = width + 2 * upPad
  let upRawWidth = (upInputWidth - 1) * upRatio + upKernelSize
  let outputWidth = upWidth / downRatio
  let upPadLeft = upPad * upRatio + (upKernelSize - upRatio) / 2
  let upsample = ConvolutionTranspose(
    groups: 1, filters: 1, filterSize: [1, upKernelSize], noBias: true,
    hint: Hint(stride: [1, upRatio]),
    name: "\(name)_upsample")
  let (snakeReader, snake) = SnakeBeta(
    prefix: "\(prefix).act", channels: channels, name: "\(name)_snake")
  let downIsEven = downKernelSize % 2 == 0
  let downPadLeft = downKernelSize / 2 - (downIsEven ? 1 : 0)
  let downPadRight = downKernelSize / 2
  let downsample = Convolution(
    groups: 1, filters: 1, filterSize: [1, downKernelSize], noBias: true,
    hint: Hint(stride: [1, downRatio]), name: "\(name)_downsample")
  var out = x.reshaped([channels, 1, 1, width])
  out = out.padded(.replicate, begin: [0, 0, 0, upPad], end: [0, 0, 0, upPad])
  out = Float(upRatio) * upsample(out)
  out = out.reshaped(
    [channels, 1, 1, upWidth], offset: [0, 0, 0, upPadLeft],
    strides: [upRawWidth, upRawWidth, upRawWidth, 1]
  ).contiguous()
  out = out.reshaped([1, channels, 1, upWidth])
  out = snake(out)
  out = out.reshaped([channels, 1, 1, upWidth])
  out = downsample(
    out.padded(.replicate, begin: [0, 0, 0, downPadLeft], end: [0, 0, 0, downPadRight]))
  out = out.reshaped([1, channels, 1, outputWidth])
  let reader: (PythonObject) -> Void = { state_dict in
    snakeReader(state_dict)
    let upFilter = state_dict["\(prefix).upsample.filter"].to(torch.float).cpu().numpy()
    upsample.weight.copy(from: try! Tensor<Float>(numpy: upFilter))
    let downFilter = state_dict["\(prefix).downsample.lowpass.filter"].to(torch.float).cpu().numpy()
    downsample.weight.copy(from: try! Tensor<Float>(numpy: downFilter))
  }
  return (reader, Model([x], [out]))
}

func AMPResBlock1(
  prefix: String, channels: Int, width: Int, kernelSize: Int, dilations: [Int], name: String
) -> (
  (PythonObject) -> Void, Model
) {
  precondition(dilations.count == 3)
  let x = Input()
  var out: Model.IO = x
  var readers = [(PythonObject) -> Void]()
  for (i, dilation) in dilations.enumerated() {
    let residual = out
    let (act1Reader, act1) = Activation1d(
      prefix: "\(prefix).acts1.\(i)", channels: channels, width: width,
      name: name.isEmpty ? "amp_act1_\(i)" : "\(name)_amp_act1_\(i)")
    let conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize], dilation: [1, dilation],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(
          begin: [0, (kernelSize - 1) * dilation / 2], end: [0, (kernelSize - 1) * dilation / 2])),
      name: name.isEmpty ? "amp_resnet_conv1_\(i)" : "\(name)_amp_resnet_conv1_\(i)")
    let (act2Reader, act2) = Activation1d(
      prefix: "\(prefix).acts2.\(i)", channels: channels, width: width,
      name: name.isEmpty ? "amp_act2_\(i)" : "\(name)_amp_act2_\(i)")
    let conv2 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(begin: [0, (kernelSize - 1) / 2], end: [0, (kernelSize - 1) / 2])),
      name: name.isEmpty ? "amp_resnet_conv2_\(i)" : "\(name)_amp_resnet_conv2_\(i)")
    var xt = act1(out)
    xt = conv1(xt)
    xt = act2(xt)
    xt = conv2(xt)
    out = residual + xt
    let idx = i
    readers.append(act1Reader)
    readers.append(act2Reader)
    readers.append { state_dict in
      let conv1Weight = state_dict["\(prefix).convs1.\(idx).weight"].to(torch.float).cpu().numpy()
      let conv1Bias = state_dict["\(prefix).convs1.\(idx).bias"].to(torch.float).cpu().numpy()
      conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1Weight))
      conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1Bias))
      let conv2Weight = state_dict["\(prefix).convs2.\(idx).weight"].to(torch.float).cpu().numpy()
      let conv2Bias = state_dict["\(prefix).convs2.\(idx).bias"].to(torch.float).cpu().numpy()
      conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2Weight))
      conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2Bias))
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x], [out]))
}

enum LTX23VocoderVariant {
  case core
  case bwe
}

let ltx23VocoderResblockKernelSizes = [3, 7, 11]
let ltx23VocoderResblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
let ltx23CoreVocoderInitialChannels = 1536
let ltx23CoreVocoderLayers: [(channels: Int, kernelSize: Int, stride: Int, padding: Int)] = [
  (channels: 768, kernelSize: 11, stride: 5, padding: 3),
  (channels: 384, kernelSize: 4, stride: 2, padding: 1),
  (channels: 192, kernelSize: 4, stride: 2, padding: 1),
  (channels: 96, kernelSize: 4, stride: 2, padding: 1),
  (channels: 48, kernelSize: 4, stride: 2, padding: 1),
  (channels: 24, kernelSize: 4, stride: 2, padding: 1),
]
let ltx23BWEVocoderInitialChannels = 512
let ltx23BWEVocoderLayers: [(channels: Int, kernelSize: Int, stride: Int, padding: Int)] = [
  (channels: 256, kernelSize: 12, stride: 6, padding: 3),
  (channels: 128, kernelSize: 11, stride: 5, padding: 3),
  (channels: 64, kernelSize: 4, stride: 2, padding: 1),
  (channels: 32, kernelSize: 4, stride: 2, padding: 1),
  (channels: 16, kernelSize: 4, stride: 2, padding: 1),
]
let ltx23VocoderHopLength = 80
let ltx23VocoderNFFT = 512
let ltx23VocoderNMelChannels = 64
let ltx23BWEResampleRatio = 3
let ltx23BWEResampleKernelSize = 43
let ltx23BWEResamplePad = 7
let ltx23BWEResamplePadLeft = 42

func Vocoder(width: Int, variant: LTX23VocoderVariant) -> (
  (PythonObject) -> Void, Model
) {
  let prefix: String
  let name: String
  let initialChannels: Int
  let layers: [(channels: Int, kernelSize: Int, stride: Int, padding: Int)]
  let useBiasAtFinal: Bool
  let applyFinalActivation: Bool
  let useTanhAtFinal: Bool
  switch variant {
  case .core:
    prefix = "vocoder"
    name = ""
    initialChannels = ltx23CoreVocoderInitialChannels
    layers = ltx23CoreVocoderLayers
    useBiasAtFinal = false
    applyFinalActivation = true
    useTanhAtFinal = false
  case .bwe:
    prefix = "bwe_generator"
    name = "bwe"
    initialChannels = ltx23BWEVocoderInitialChannels
    layers = ltx23BWEVocoderLayers
    useBiasAtFinal = false
    applyFinalActivation = false
    useTanhAtFinal = false
  }
  let x = Input()
  let convPre = Convolution(
    groups: 1, filters: initialChannels, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])),
    name: name.isEmpty ? "conv_pre" : "\(name)_conv_pre")
  var out = convPre(x)
  var currentWidth = width
  var readers = [(PythonObject) -> Void]()
  for (i, layer) in layers.enumerated() {
    let up = ConvolutionTranspose(
      groups: 1, filters: layer.channels, filterSize: [1, layer.kernelSize],
      hint: Hint(
        stride: [1, layer.stride],
        border: Hint.Border(begin: [0, layer.padding], end: [0, layer.padding])),
      name: name.isEmpty ? "up" : "\(name)_up")
    out = up(out)
    currentWidth *= layer.stride
    let upIdx = i
    readers.append { state_dict in
      let upsWeight = state_dict["\(prefix).ups.\(upIdx).weight"].to(torch.float).cpu().numpy()
      let upsBias = state_dict["\(prefix).ups.\(upIdx).bias"].to(torch.float).cpu().numpy()
      up.weight.copy(from: try! Tensor<Float>(numpy: upsWeight))
      up.bias.copy(from: try! Tensor<Float>(numpy: upsBias))
    }
    var blockOutputs = [Model.IO]()
    for (j, kernelSize) in ltx23VocoderResblockKernelSizes.enumerated() {
      let blockIndex = i * ltx23VocoderResblockKernelSizes.count + j
      let (blockReader, block) = AMPResBlock1(
        prefix: "\(prefix).resblocks.\(blockIndex)", channels: layer.channels, width: currentWidth,
        kernelSize: kernelSize, dilations: ltx23VocoderResblockDilations[j], name: name)
      readers.append(blockReader)
      blockOutputs.append(block(out))
    }
    precondition(!blockOutputs.isEmpty)
    let blockSum = blockOutputs.dropFirst().reduce(blockOutputs[0]) { $0 + $1 }
    out = (1.0 / Float(blockOutputs.count)) * blockSum
  }
  let finalChannels = layers.last?.channels ?? initialChannels
  let (actPostReader, actPost) = Activation1d(
    prefix: "\(prefix).act_post", channels: finalChannels, width: currentWidth,
    name: name.isEmpty ? "act_post" : "\(name)_act_post")
  out = actPost(out)
  readers.append(actPostReader)
  let convPost = Convolution(
    groups: 1, filters: 2, filterSize: [1, 7], noBias: !useBiasAtFinal,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])),
    name: name.isEmpty ? "conv_post" : "\(name)_conv_post")
  out = convPost(out)
  if applyFinalActivation {
    if useTanhAtFinal {
      out = out.tanh()
    } else {
      out = out.clamped(-1...1)
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let convPreWeight = state_dict["\(prefix).conv_pre.weight"].to(torch.float).cpu().numpy()
    let convPreBias = state_dict["\(prefix).conv_pre.bias"].to(torch.float).cpu().numpy()
    convPre.weight.copy(from: try! Tensor<Float>(numpy: convPreWeight))
    convPre.bias.copy(from: try! Tensor<Float>(numpy: convPreBias))
    for reader in readers {
      reader(state_dict)
    }
    let convPostWeight = state_dict["\(prefix).conv_post.weight"].to(torch.float).cpu().numpy()
    convPost.weight.copy(from: try! Tensor<Float>(numpy: convPostWeight))
    if useBiasAtFinal {
      let convPostBias = state_dict["\(prefix).conv_post.bias"].to(torch.float).cpu().numpy()
      convPost.bias.copy(from: try! Tensor<Float>(numpy: convPostBias))
    }
  }
  return (reader, Model([x], [out]))
}

func VocoderWithBWE(inputMelWidth: Int, melBins: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()  // [1, 2, T, mel_bins]
  var coreInput = x.transposed(2, 3).contiguous()  // [1, 2, mel_bins, T]
  coreInput = coreInput.reshaped(
    [1, 2 * melBins, 1, inputMelWidth], offset: [0, 0, 0, 0],
    strides: [2 * melBins * inputMelWidth, inputMelWidth, inputMelWidth, 1]
  ).contiguous()  // [1, 128, 1, T]
  let (coreReader, coreVocoder) = Vocoder(width: inputMelWidth, variant: .core)
  let coreOut = coreVocoder(coreInput)  // [1, 2, 1, coreWidth]
  let coreWidth = inputMelWidth * ltx23CoreVocoderLayers.reduce(1) { $0 * $1.stride }
  let remainder = coreWidth % ltx23VocoderHopLength
  let corePad = remainder == 0 ? 0 : (ltx23VocoderHopLength - remainder)
  let paddedCoreWidth = coreWidth + corePad
  let bweInputWidth = paddedCoreWidth / ltx23VocoderHopLength
  var paddedCoreOut = coreOut
  if corePad > 0 {
    paddedCoreOut = coreOut.padded(.zero, begin: [0, 0, 0, 0], end: [0, 0, 0, corePad])
  }
  let nFreqs = ltx23VocoderNFFT / 2 + 1
  let leftPad = max(0, ltx23VocoderNFFT - ltx23VocoderHopLength)
  let stft = Convolution(
    groups: 1, filters: nFreqs * 2, filterSize: [1, ltx23VocoderNFFT], noBias: true,
    hint: Hint(stride: [1, ltx23VocoderHopLength]), name: "mel_stft_forward")
  var stftInput = paddedCoreOut.reshaped([2, 1, 1, paddedCoreWidth])
  stftInput = stftInput.padded(.zero, begin: [0, 0, 0, leftPad], end: [0, 0, 0, 0])
  let stftOut = stft(stftInput)  // [2, 2*nFreqs, 1, T_frames]
  let stftParts = stftOut.chunked(2, axis: 1)
  let magnitude = ((stftParts[0] .* stftParts[0]) + (stftParts[1] .* stftParts[1])).squareRoot()
  let melProjection = Convolution(
    groups: 1, filters: ltx23VocoderNMelChannels, filterSize: [1, 1], noBias: true,
    hint: Hint(stride: [1, 1]), name: "mel_projection")
  var mel = melProjection(magnitude).clamped(1e-5...).log()
  mel = mel.reshaped(
    [1, 2, ltx23VocoderNMelChannels, bweInputWidth], offset: [0, 0, 0, 0],
    strides: [
      2 * ltx23VocoderNMelChannels * bweInputWidth, ltx23VocoderNMelChannels * bweInputWidth,
      bweInputWidth, 1,
    ]
  ).contiguous()  // [1, 2, mel_bins, T_frames]
  let bweInput = mel.reshaped(
    [1, 2 * ltx23VocoderNMelChannels, 1, bweInputWidth], offset: [0, 0, 0, 0],
    strides: [
      2 * ltx23VocoderNMelChannels * bweInputWidth, bweInputWidth, bweInputWidth, 1,
    ]
  ).contiguous()  // [1, 128, 1, T_frames]
  let (bweReader, bweGenerator) = Vocoder(width: bweInputWidth, variant: .bwe)
  let residual = bweGenerator(bweInput)  // [1, 2, 1, coreWidth * ratio]
  let resampler = ConvolutionTranspose(
    groups: 1, filters: 1, filterSize: [1, ltx23BWEResampleKernelSize], noBias: true,
    hint: Hint(stride: [1, ltx23BWEResampleRatio]), name: "bwe_resampler")
  let resampleRawWidth =
    (paddedCoreWidth + 2 * ltx23BWEResamplePad - 1) * ltx23BWEResampleRatio
    + ltx23BWEResampleKernelSize
  var skip = paddedCoreOut.reshaped([2, 1, 1, paddedCoreWidth]).padded(
    .replicate, begin: [0, 0, 0, ltx23BWEResamplePad], end: [0, 0, 0, ltx23BWEResamplePad])
  skip = Float(ltx23BWEResampleRatio) * resampler(skip)
  skip = skip.reshaped(
    [2, 1, 1, paddedCoreWidth * ltx23BWEResampleRatio], offset: [0, 0, 0, ltx23BWEResamplePadLeft],
    strides: [resampleRawWidth, resampleRawWidth, resampleRawWidth, 1]
  ).contiguous()
  skip = skip.reshaped([1, 2, 1, paddedCoreWidth * ltx23BWEResampleRatio])
  var out = (residual + skip).clamped(-1...1)
  let outputLength = coreWidth * ltx23BWEResampleRatio
  if outputLength != paddedCoreWidth * ltx23BWEResampleRatio {
    out = out.reshaped(
      [1, 2, 1, outputLength], offset: [0, 0, 0, 0],
      strides: [
        2 * paddedCoreWidth * ltx23BWEResampleRatio, paddedCoreWidth * ltx23BWEResampleRatio,
        paddedCoreWidth * ltx23BWEResampleRatio, 1,
      ]
    ).contiguous()
  }
  let reader: (PythonObject) -> Void = { state_dict in
    coreReader(state_dict)
    bweReader(state_dict)
    let stftWeight = state_dict["mel_stft.stft_fn.forward_basis"].to(torch.float).unsqueeze(2).cpu()
      .numpy()
    stft.weight.copy(from: try! Tensor<Float>(numpy: stftWeight))
    let melWeight = state_dict["mel_stft.mel_basis"].to(torch.float).unsqueeze(2).unsqueeze(3).cpu()
      .numpy()
    melProjection.weight.copy(from: try! Tensor<Float>(numpy: melWeight))
    let resamplerWeight = hannUpSample1dFilterWeight(
      ratio: ltx23BWEResampleRatio, kernelSize: ltx23BWEResampleKernelSize)
    resampler.weight.copy(from: resamplerWeight)
  }
  return (reader, Model([x], [out]))
}

func intArray(from values: PythonObject) -> [Int] {
  Python.list(values).compactMap { Int($0) }
}

func intMatrix(from values: PythonObject) -> [[Int]] {
  Python.list(values).map { intArray(from: $0) }
}

func vocoderLayers(from config: PythonObject) -> [(
  channels: Int, kernelSize: Int, stride: Int, padding: Int
)] {
  let rates = intArray(from: config["upsample_rates"])
  let kernels = intArray(from: config["upsample_kernel_sizes"])
  precondition(rates.count == kernels.count)
  var channels = Int(config["upsample_initial_channel"])!
  var layers = [(channels: Int, kernelSize: Int, stride: Int, padding: Int)]()
  for i in 0..<rates.count {
    channels /= 2
    let stride = rates[i]
    let kernelSize = kernels[i]
    let padding = max(0, (kernelSize - stride) / 2)
    layers.append((channels: channels, kernelSize: kernelSize, stride: stride, padding: padding))
  }
  return layers
}

func boolConfig(_ dict: PythonObject, key: String, default defaultValue: Bool) -> Bool {
  if Bool(dict.__contains__(key)) ?? false {
    return Bool(dict[key]) ?? defaultValue
  }
  return defaultValue
}

func hannUpSample1dFilterWeight(ratio: Int, kernelSize: Int) -> Tensor<Float> {
  let rolloff: Float = 0.99
  let lowpassFilterWidth: Float = 6
  let width = Int(ceil(Double(lowpassFilterWidth / rolloff)))
  precondition(2 * width * ratio + 1 == kernelSize)
  var values = [Float]()
  values.reserveCapacity(kernelSize)
  for i in 0..<kernelSize {
    let t = (Float(i) / Float(ratio) - Float(width)) * rolloff
    let clamped = min(max(t, -lowpassFilterWidth), lowpassFilterWidth)
    let window = cos(clamped * Float.pi / lowpassFilterWidth / 2)
    let sinc = abs(t) < 1e-8 ? Float(1) : sin(Float.pi * t) / (Float.pi * t)
    values.append(sinc * window * window * rolloff / Float(ratio))
  }
  let filterNumpy = numpy.array(values, dtype: numpy.float32).reshape([1, 1, kernelSize])
  return try! Tensor<Float>(numpy: filterNumpy)
}

func maxAbsDiffVocoder(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>)
  -> Float
{
  precondition(swiftTensor.shape.count == 4)
  precondition(torchTensor.shape.count == 3)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == 1)
  precondition(swiftTensor.shape[3] == torchTensor.shape[2])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for t in 0..<torchTensor.shape[2] {
        let diff = abs(Float(swiftTensor[i, c, 0, t]) - Float(torchTensor[i, c, t]))
        if diff > maxDiff {
          maxDiff = diff
        }
      }
    }
  }
  return maxDiff
}

func maxAbsDiff4D(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float
{
  precondition(swiftTensor.shape.count == 4)
  precondition(torchTensor.shape.count == 4)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for h in 0..<torchTensor.shape[2] {
        for w in 0..<torchTensor.shape[3] {
          let diff = abs(Float(swiftTensor[i, c, h, w]) - Float(torchTensor[i, c, h, w]))
          if diff > maxDiff {
            maxDiff = diff
          }
        }
      }
    }
  }
  return maxDiff
}

func sampledMaxAbsDiff4D(
  _ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>, sampleCount: Int
) -> Float {
  precondition(swiftTensor.shape.count == 4)
  precondition(torchTensor.shape.count == 4)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  let c = swiftTensor.shape[0]
  let d = swiftTensor.shape[1]
  let h = swiftTensor.shape[2]
  let w = swiftTensor.shape[3]
  let total = c * d * h * w
  let n = min(sampleCount, total)
  var state: UInt64 = 0x9e37_79b9_7f4a_7c15
  var maxDiff: Float = 0
  for _ in 0..<n {
    state = state &* 2_862_933_555_777_941_757 &+ 3_037_000_493
    let linear = Int(state % UInt64(total))
    let wi = linear % w
    let hi = (linear / w) % h
    let di = (linear / (w * h)) % d
    let ci = linear / (w * h * d)
    let diff = abs(Float(swiftTensor[ci, di, hi, wi]) - Float(torchTensor[ci, di, hi, wi]))
    if diff > maxDiff {
      maxDiff = diff
    }
  }
  return maxDiff
}

func maxAbsDiff5D(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float
{
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  precondition(swiftTensor.shape[4] == torchTensor.shape[4])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - Float(torchTensor[i, c, d, h, w]))
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
  }
  return maxDiff
}

let runVocoderParityOnly =
  (ProcessInfo.processInfo.environment["LTX23_VOCODER_PARITY_ONLY"] ?? "") == "1"
if runVocoderParityOnly {
  print("vocoder parity: start")
  let audioVocoderStateDict = audio_vocoder.state_dict()
  precondition(Int(audio_vocoder.hop_length)! == ltx23VocoderHopLength)
  precondition(Int(audio_vocoder.resampler.ratio)! == ltx23BWEResampleRatio)
  precondition(Int(audio_vocoder.resampler.kernel_size)! == ltx23BWEResampleKernelSize)
  precondition(
    Int(audioVocoderStateDict["mel_stft.stft_fn.forward_basis"].shape[2])! == ltx23VocoderNFFT)
  precondition(
    Int(audioVocoderStateDict["mel_stft.mel_basis"].shape[0])! == ltx23VocoderNMelChannels)
  audio_vocoder.to(torch.float)
  audio_vocoder.eval()
  graph.withNoGrad {
    let pythonModelDevice = audio_vocoder.vocoder.conv_pre.weight.device
    let testCoreMel = torch.randn([1, 2, 481, 64]).to(torch.float).to(pythonModelDevice)
    let torchCoreOut = audio_vocoder.vocoder(testCoreMel)
    let coreInputWidth = Int(testCoreMel.shape[2])!
    let (coreReader, swiftCoreVocoder) = Vocoder(width: coreInputWidth, variant: .core)
    let coreSwiftInputTorch = testCoreMel.transpose(2, 3).reshape([1, 128, 1, coreInputWidth])
    let coreSwiftInputCPU = graph.variable(
      try! Tensor<Float>(numpy: coreSwiftInputTorch.to(torch.float).cpu().numpy()))
    let coreSwiftInput = coreSwiftInputCPU.toGPU(swiftDevice)
    swiftCoreVocoder.compile(inputs: coreSwiftInput)
    coreReader(audioVocoderStateDict)
    let swiftCoreOut = swiftCoreVocoder(inputs: coreSwiftInput)[0].as(of: Float.self).toCPU()
    let torchCoreTensor = try! Tensor<Float>(numpy: torchCoreOut.to(torch.float).cpu().numpy())
    print("core vocoder max abs diff:", maxAbsDiffVocoder(swiftCoreOut, torchCoreTensor))

    var coreForBWE = torchCoreOut
    let coreRemainder = Int(coreForBWE.shape[2])! % ltx23VocoderHopLength
    if coreRemainder != 0 {
      coreForBWE = torch.nn.functional.pad(coreForBWE, [0, ltx23VocoderHopLength - coreRemainder])
    }
    let melForBWE = audio_vocoder._compute_mel(coreForBWE).transpose(2, 3)
    let torchBWEResidual = audio_vocoder.bwe_generator(melForBWE)
    let bweInputWidth = Int(melForBWE.shape[2])!
    let (bweReader, swiftBWEGenerator) = Vocoder(width: bweInputWidth, variant: .bwe)
    let bweSwiftInputTorch = melForBWE.transpose(2, 3).reshape([1, 128, 1, bweInputWidth])
    let bweSwiftInputCPU = graph.variable(
      try! Tensor<Float>(numpy: bweSwiftInputTorch.to(torch.float).cpu().numpy()))
    let bweSwiftInput = bweSwiftInputCPU.toGPU(swiftDevice)
    swiftBWEGenerator.compile(inputs: bweSwiftInput)
    bweReader(audioVocoderStateDict)
    let swiftBWEOut = swiftBWEGenerator(inputs: bweSwiftInput)[0].as(of: Float.self).toCPU()
    let torchBWETensor = try! Tensor<Float>(numpy: torchBWEResidual.to(torch.float).cpu().numpy())
    print("bwe generator max abs diff:", maxAbsDiffVocoder(swiftBWEOut, torchBWETensor))

    let inputMelWidth = Int(testCoreMel.shape[2])!
    let melBins = Int(testCoreMel.shape[3])!
    let (vocoderWithBWEReader, swiftVocoderWithBWE) = VocoderWithBWE(
      inputMelWidth: inputMelWidth, melBins: melBins)
    let fullSwiftInputCPU = graph.variable(
      try! Tensor<Float>(numpy: testCoreMel.to(torch.float).cpu().numpy()))
    let fullSwiftInput = fullSwiftInputCPU.toGPU(swiftDevice)
    swiftVocoderWithBWE.compile(inputs: fullSwiftInput)
    vocoderWithBWEReader(audioVocoderStateDict)
    let swiftVocoderWithBWEOut = swiftVocoderWithBWE(inputs: fullSwiftInput)[0].as(of: Float.self)
      .toCPU()
    let torchVocoderWithBWEOut = audio_vocoder(testCoreMel)
    let torchVocoderWithBWETensor = try! Tensor<Float>(
      numpy: torchVocoderWithBWEOut.to(torch.float).cpu().numpy())
    print(
      "vocoder_with_bwe max abs diff:",
      maxAbsDiffVocoder(swiftVocoderWithBWEOut, torchVocoderWithBWETensor))
  }
  print("vocoder parity: done")
  exit(0)
}

let ltx_core_model_transformer_model_configurator = Python.import(
  "ltx_core.model.transformer.model_configurator")
let ltx_core_text_encoders_gemma_encoders_av_encoder = Python.None
let ltx_core_model_transformer_modality = Python.import("ltx_core.model.transformer.modality")
let ltx_core_model_video_vae_model_configurator = Python.import(
  "ltx_core.model.video_vae.model_configurator")
let ltx_core_tools = Python.import("ltx_core.tools")
let ltx_core_types = Python.import("ltx_core.types")
let ltx_core_components_patchifiers = Python.import(
  "ltx_core.components.patchifiers")
let ltx_core_loader_sd_ops = Python.import("ltx_core.loader.sd_ops")  // legacy section below exit(0)
let ltx_core_conditioning = Python.import("ltx_core.conditioning")  // legacy section below exit(0)

/* Temporarily disabled while iterating on main diffusion model.
  var exportedAudioDecoder: Model? = nil
  var exportedAudioEncoder: Model? = nil
  var exportedVocoderWithBWE: Model? = nil
  var exportedVideoDecoder: Model? = nil
  var exportedVideoEncoder: Model? = nil

  print("audio vae + vocoder parity: start")
  let audioVocoderStateDict = audio_vocoder.state_dict()

  audio_vocoder.to(torch.float)
  audio_vocoder.eval()

graph.withNoGrad {
  let audioModelDevice = audio_decoder.conv_in.conv.weight.device
  let aForAudio = a.to(audioModelDevice)
  let decodedAudio = audio_decoder(aForAudio).to(torch.float)
  let encodedAudio = audio_encoder(decodedAudio).to(torch.float)
  let audioDecoderStateDict = audio_decoder.state_dict()
  let decoderStd = graph.variable(
    try! Tensor<Float>(
      numpy: audioDecoderStateDict["per_channel_statistics.std-of-means"].to(torch.float)
        .view(1, 1, -1).cpu().numpy())
  ).toGPU(swiftDevice)
  let decoderMean = graph.variable(
    try! Tensor<Float>(
      numpy: audioDecoderStateDict["per_channel_statistics.mean-of-means"].to(torch.float)
        .view(1, 1, -1).cpu().numpy())
  ).toGPU(swiftDevice)
  let aTensor = graph.variable(
    try! Tensor<Float>(numpy: aForAudio.to(torch.float).cpu().numpy())
  ).toGPU(swiftDevice)
  var decoderInput = aTensor.reshaped(.CHW(8, 121, 16)).permuted(1, 0, 2).contiguous().reshaped(
    .HWC(1, 121, 128))
  decoderInput = decoderInput .* decoderStd + decoderMean
  decoderInput = decoderInput.reshaped(.CHW(121, 8, 16)).permuted(1, 0, 2).contiguous().reshaped(
    .NCHW(1, 8, 121, 16))
  let (audioDecoderReader, audioDecoderSwift) = DecoderCausal2D(
    channels: [128, 256, 512], numRepeat: 3, startWidth: 16, startHeight: 121)
  audioDecoderSwift.compile(inputs: decoderInput)
  exportedAudioDecoder = audioDecoderSwift
  audioDecoderReader(audioDecoderStateDict)
  let swiftMel = audioDecoderSwift(inputs: decoderInput)[0].as(of: Float.self).toCPU()
  let torchMel = try! Tensor<Float>(numpy: decodedAudio.to(torch.float).cpu().numpy())
  print("audio decoder max abs diff:", maxAbsDiff4D(swiftMel, torchMel))

  let audioEncoderStateDict = audio_encoder.state_dict()
  let encoderStd = graph.variable(
    try! Tensor<Float>(
      numpy: audioEncoderStateDict["per_channel_statistics.std-of-means"].to(torch.float)
        .view(1, 1, -1).cpu().numpy())
  ).toGPU(swiftDevice)
  let encoderMean = graph.variable(
    try! Tensor<Float>(
      numpy: audioEncoderStateDict["per_channel_statistics.mean-of-means"].to(torch.float)
        .view(1, 1, -1).cpu().numpy())
  ).toGPU(swiftDevice)
  let encoderInvStd = Functional.reciprocal(encoderStd)
  let encoderBias = (-encoderMean) .* encoderInvStd
  let (audioEncoderReader, audioEncoderSwift) = EncoderCausal2D(
    channels: [128, 256, 512], numRepeat: 2, startWidth: 16, startHeight: 121)
  let melTensor = graph.variable(torchMel).toGPU(swiftDevice)
  audioEncoderSwift.compile(inputs: melTensor)
  exportedAudioEncoder = audioEncoderSwift
  audioEncoderReader(audioEncoderStateDict)
  var swiftEncoded = audioEncoderSwift(inputs: melTensor)[0].as(of: Float.self)
  swiftEncoded = swiftEncoded.reshaped(.CHW(8, 121, 16)).permuted(1, 0, 2).contiguous().reshaped(
    .HWC(1, 121, 128))
  swiftEncoded = swiftEncoded .* encoderInvStd + encoderBias
  swiftEncoded = swiftEncoded.reshaped(.CHW(121, 8, 16)).permuted(1, 0, 2).contiguous().reshaped(
    .NCHW(1, 8, 121, 16))
  let swiftEncodedCPU = swiftEncoded.toCPU()
  let torchEncoded = try! Tensor<Float>(numpy: encodedAudio.to(torch.float).cpu().numpy())
  print("audio encoder max abs diff:", maxAbsDiff4D(swiftEncodedCPU, torchEncoded))

  let pythonModelDevice = audio_vocoder.vocoder.conv_pre.weight.device
  let testCoreMel = torch.randn([1, 2, 481, 64]).to(torch.float).to(pythonModelDevice)
  let torchCoreOut = audio_vocoder.vocoder(testCoreMel)
  let coreInputWidth = Int(testCoreMel.shape[2])!
  let (coreReader, swiftCoreVocoder) = Vocoder(width: coreInputWidth, variant: .core)
  let coreSwiftInputTorch = testCoreMel.transpose(2, 3).reshape([1, 128, 1, coreInputWidth])
  let coreSwiftInputCPU = graph.variable(
    try! Tensor<Float>(numpy: coreSwiftInputTorch.to(torch.float).cpu().numpy()))
  let coreSwiftInput = coreSwiftInputCPU.toGPU(swiftDevice)
  swiftCoreVocoder.compile(inputs: coreSwiftInput)
  coreReader(audioVocoderStateDict)
  let swiftCoreOut = swiftCoreVocoder(inputs: coreSwiftInput)[0].as(of: Float.self).toCPU()
  let torchCoreTensor = try! Tensor<Float>(numpy: torchCoreOut.to(torch.float).cpu().numpy())
  print("core vocoder max abs diff:", maxAbsDiffVocoder(swiftCoreOut, torchCoreTensor))

  var coreForBWE = torchCoreOut
  let coreRemainder = Int(coreForBWE.shape[2])! % ltx23VocoderHopLength
  if coreRemainder != 0 {
    coreForBWE = torch.nn.functional.pad(coreForBWE, [0, ltx23VocoderHopLength - coreRemainder])
  }
  let melForBWE = audio_vocoder._compute_mel(coreForBWE).transpose(2, 3)
  let torchBWEResidual = audio_vocoder.bwe_generator(melForBWE)
  let bweInputWidth = Int(melForBWE.shape[2])!
  let (bweReader, swiftBWEGenerator) = Vocoder(width: bweInputWidth, variant: .bwe)
  let bweSwiftInputTorch = melForBWE.transpose(2, 3).reshape([1, 128, 1, bweInputWidth])
  let bweSwiftInputCPU = graph.variable(
    try! Tensor<Float>(numpy: bweSwiftInputTorch.to(torch.float).cpu().numpy()))
  let bweSwiftInput = bweSwiftInputCPU.toGPU(swiftDevice)
  swiftBWEGenerator.compile(inputs: bweSwiftInput)
  bweReader(audioVocoderStateDict)
  let swiftBWEOut = swiftBWEGenerator(inputs: bweSwiftInput)[0].as(of: Float.self).toCPU()
  let torchBWETensor = try! Tensor<Float>(numpy: torchBWEResidual.to(torch.float).cpu().numpy())
  print("bwe generator max abs diff:", maxAbsDiffVocoder(swiftBWEOut, torchBWETensor))

  let inputMelWidth = Int(testCoreMel.shape[2])!
  let melBins = Int(testCoreMel.shape[3])!
  let (vocoderWithBWEReader, swiftVocoderWithBWE) = VocoderWithBWE(
    inputMelWidth: inputMelWidth, melBins: melBins)
  let fullSwiftInputCPU = graph.variable(
    try! Tensor<Float>(numpy: testCoreMel.to(torch.float).cpu().numpy()))
  let fullSwiftInput = fullSwiftInputCPU.toGPU(swiftDevice)
  swiftVocoderWithBWE.compile(inputs: fullSwiftInput)
  exportedVocoderWithBWE = swiftVocoderWithBWE
  vocoderWithBWEReader(audioVocoderStateDict)
  let swiftVocoderWithBWEOut = swiftVocoderWithBWE(inputs: fullSwiftInput)[0].as(of: Float.self).toCPU()
  let torchVocoderWithBWEOut = audio_vocoder(testCoreMel)
  let torchVocoderWithBWETensor = try! Tensor<Float>(numpy: torchVocoderWithBWEOut.to(torch.float).cpu().numpy())
  print("vocoder_with_bwe max abs diff:", maxAbsDiffVocoder(swiftVocoderWithBWEOut, torchVocoderWithBWETensor))
}
print("audio vae + vocoder parity: done")

let vae_decoder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: ltx23ModelPath,
  model_class_configurator: ltx_core_model_video_vae_model_configurator.VideoDecoderConfigurator,
  model_sd_ops: ltx_core_model_video_vae_model_configurator.VAE_DECODER_COMFY_KEYS_FILTER
).build(device: pythonDevice)
let vae_encoder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: ltx23ModelPath,
  model_class_configurator: ltx_core_model_video_vae_model_configurator.VideoEncoderConfigurator,
  model_sd_ops: ltx_core_model_video_vae_model_configurator.VAE_ENCODER_COMFY_KEYS_FILTER
).build(device: pythonDevice)
let z = torch.randn([1, 128, 16, 16, 24]).to(torch.float).to(torch_device)
vae_decoder.to(torch.float)
vae_encoder.to(torch.float)
vae_decoder.eval()
vae_encoder.eval()

func ResnetBlockCausal3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int, isCausal: Bool
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "resnet_norm1")
  var out = norm1(x.reshaped([channels, depth, height, width])).reshaped([
    1, channels, depth, height, width,
  ])
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "resnet_conv1")
  out = out.padded(
    .replicate, begin: [0, 0, isCausal ? 2 : 1, 0, 0], end: [0, 0, isCausal ? 0 : 1, 0, 0])
  out = conv1(out)
  let norm2 = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "resnet_norm2")
  out = norm2(out.reshaped([channels, depth, height, width])).reshaped([
    1, channels, depth, height, width,
  ])
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(
      stride: [1, 1, 1],
      border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "resnet_conv2")
  out = out.padded(
    .replicate, begin: [0, 0, isCausal ? 2 : 1, 0, 0], end: [0, 0, isCausal ? 0 : 1, 0, 0])
  out = conv2(out)
  out = x + out
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1_weight = state_dict["\(prefix).conv1.conv.weight"].to(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.conv.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let conv2_weight = state_dict["\(prefix).conv2.conv.weight"].to(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.conv.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
  }
  return (reader, Model([x], [out]))
}

func DecoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int,
  startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  precondition(!layers.isEmpty)

  let x = Input()
  var previousChannel = layers[0].channels
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_in")
  var out = convIn(
    x.padded(.replicate, begin: [0, 0, 1, 0, 0], end: [0, 0, 1, 0, 0]))
  var readers = [(PythonObject) -> Void]()
  var j = 0
  var depth = startDepth
  var height = startHeight
  var width = startWidth

  for layer in layers {
    let channels = layer.channels
    let strideT = layer.stride.0
    let strideH = layer.stride.1
    let strideW = layer.stride.2
    if strideT > 1 || strideH > 1 || strideW > 1 {
      precondition(previousChannel % channels == 0)
      let outChannelsReductionFactor = previousChannel / channels
      let upsampleOutChannels =
        previousChannel * strideT * strideH * strideW / outChannelsReductionFactor
      let conv = Convolution(
        groups: 1, filters: upsampleOutChannels, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
        name: "depth_to_space_upsample")
      out = conv(
        out.padded(.replicate, begin: [0, 0, 1, 0, 0], end: [0, 0, 1, 0, 0]))
      let upBlocks = j
      readers.append { state_dict in
        let up_blocks_conv_conv_weight = state_dict["up_blocks.\(upBlocks).conv.conv.weight"].to(
          torch.float
        ).cpu().numpy()
        let up_blocks_conv_conv_bias = state_dict["up_blocks.\(upBlocks).conv.conv.bias"].to(
          torch.float
        ).cpu().numpy()
        conv.weight.copy(from: try! Tensor<Float>(numpy: up_blocks_conv_conv_weight))
        conv.bias.copy(from: try! Tensor<Float>(numpy: up_blocks_conv_conv_bias))
      }
      out = out.reshaped([
        channels, strideT, strideH, strideW, depth, height, width,
      ]).permuted(0, 4, 1, 5, 2, 6, 3).contiguous().reshaped([
        1, channels, depth * strideT, height * strideH, width * strideW,
      ])
      if strideT == 2 {
        out = out.reshaped(
          [1, channels, depth * strideT - 1, height * strideH, width * strideW],
          offset: [0, 0, 1, 0, 0],
          strides: [
            channels * depth * strideT * height * strideH * width * strideW,
            depth * strideT * height * strideH * width * strideW, height * strideH * width * strideW,
            width * strideW, 1,
          ]
        ).contiguous()
      }
      depth = depth * strideT - (strideT == 2 ? 1 : 0)
      height = height * strideH
      width = width * strideW
      previousChannel = channels
      j += 1
    } else {
      precondition(channels == previousChannel)
    }

    for i in 0..<layer.numRepeat {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "up_blocks.\(j).res_blocks.\(i)", channels: channels, depth: depth,
        height: height, width: width, isCausal: false)
      out = block(out)
      readers.append(reader)
    }
    j += 1
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ]).swish()
  let convOut = Convolution(
    groups: 1, filters: 48, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_out")
  out = convOut(
    out.padded(.replicate, begin: [0, 0, 1, 0, 0], end: [0, 0, 1, 0, 0]))
  // LTXV weirdly, did "b (c p r q) f h w -> b c (f p) (h q) (w r)"
  out = out.reshaped([3, 4, 4, depth, height, width]).permuted(0, 3, 4, 2, 5, 1).contiguous()
    .reshaped([3, depth, height * 4, width * 4])

  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["conv_in.conv.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["conv_in.conv.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    let conv_out_weight = state_dict["conv_out.conv.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["conv_out.conv.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

func EncoderCausal3D(
  layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))], startWidth: Int,
  startHeight: Int,
  startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for layer in layers {
    depth = (depth - 1) * layer.stride.0 + 1
    height *= layer.stride.1
    width *= layer.stride.2
  }
  // LTXV weirdly, did "b (c p r q) f h w -> b c (f p) (h q) (w r)"
  var out = x.reshaped([3, depth, height, 4, width, 4]).permuted(0, 5, 3, 1, 2, 4).contiguous()
    .reshaped([1, 48, depth, height, width])
  var previousChannel = layers[0].channels
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_in")
  out = convIn(out.padded(.replicate, begin: [0, 0, 2, 0, 0], end: [0, 0, 0, 0, 0]))
  var j = 0
  var readers = [(PythonObject) -> Void]()
  for layer in layers {
    let channels = layer.channels
    if layer.stride.0 > 1 || layer.stride.1 > 1 || layer.stride.2 > 1 {
      // Convolution & reshape.
      let conv = Convolution(
        groups: 1, filters: channels / (layer.stride.0 * layer.stride.1 * layer.stride.2),
        filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
        name: "space_to_depth_downsample")
      if layer.stride.0 == 1 {
        var residual = out.reshaped([
          previousChannel, depth, height / layer.stride.1, layer.stride.1, width / layer.stride.2,
          layer.stride.2,
        ]).permuted(0, 3, 5, 1, 2, 4).contiguous().reshaped([
          1, channels, previousChannel * layer.stride.1 * layer.stride.2 / channels, depth,
          height / layer.stride.1, width / layer.stride.2,
        ])
        residual = residual.reduced(.mean, axis: [2]).reshaped([
          1, channels, depth, height / layer.stride.1, width / layer.stride.2,
        ])
        out = conv(out.padded(.replicate, begin: [0, 0, 2, 0, 0]))
        out = out.reshaped([
          channels / (layer.stride.1 * layer.stride.2), depth, height / layer.stride.1,
          layer.stride.1, width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 3, 5, 1, 2, 4).contiguous().reshaped([
          1, channels, depth, height / layer.stride.1, width / layer.stride.2,
        ])
        out = residual + out
        height = height / layer.stride.1
        width = width / layer.stride.2
      } else if layer.stride.1 == 1 && layer.stride.2 == 1 {
        var residual = out.padded(.replicate, begin: [0, 0, 1, 0, 0]).reshaped([
          previousChannel, (depth + 1) / layer.stride.0, layer.stride.0, height, width,
        ]).permuted(0, 2, 1, 3, 4).contiguous()
        if previousChannel * layer.stride.0 / channels > 1 {
          residual = residual.reshaped([
            1, channels, previousChannel * layer.stride.0 / channels, (depth + 1) / layer.stride.0,
            height, width,
          ])
          residual = residual.reduced(.mean, axis: [2]).reshaped([
            1, channels, (depth + 1) / layer.stride.0, height, width,
          ])
        } else {
          residual = residual.reshaped([1, channels, (depth + 1) / layer.stride.0, height, width])
        }
        out = conv(out.padded(.replicate, begin: [0, 0, 3, 0, 0]))
        out = out.reshaped([
          channels / layer.stride.0, (depth + 1) / layer.stride.0, layer.stride.0, height, width,
        ]).permuted(0, 2, 1, 3, 4).contiguous().reshaped([
          1, channels, (depth + 1) / layer.stride.0, height, width,
        ])
        out = residual + out
        depth = (depth + 1) / layer.stride.0
      } else {
        var residual = out.padded(.replicate, begin: [0, 0, 1, 0, 0]).reshaped([
          previousChannel, (depth + 1) / layer.stride.0, layer.stride.0, height / layer.stride.1,
          layer.stride.1, width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 2, 4, 6, 1, 3, 5).contiguous().reshaped([
          1, channels,
          previousChannel * layer.stride.0 * layer.stride.1 * layer.stride.2 / channels,
          (depth + 1) / layer.stride.0, height / layer.stride.1, width / layer.stride.2,
        ])
        residual = residual.reduced(.mean, axis: [2]).reshaped([
          1, channels, (depth + 1) / layer.stride.0, height / layer.stride.1,
          width / layer.stride.2,
        ])
        out = conv(out.padded(.replicate, begin: [0, 0, 3, 0, 0]))
        out = out.reshaped([
          channels / (layer.stride.0 * layer.stride.1 * layer.stride.2),
          (depth + 1) / layer.stride.0, layer.stride.0, height / layer.stride.1, layer.stride.1,
          width / layer.stride.2, layer.stride.2,
        ]).permuted(0, 2, 4, 6, 1, 3, 5).contiguous().reshaped([
          1, channels, (depth + 1) / layer.stride.0, height / layer.stride.1,
          width / layer.stride.2,
        ])
        out = residual + out
        depth = (depth + 1) / layer.stride.0
        height = height / layer.stride.1
        width = width / layer.stride.2
      }
      previousChannel = channels
      let downBlocks = j
      readers.append { state_dict in
        let down_blocks_conv_conv_weight = state_dict["down_blocks.\(downBlocks).conv.conv.weight"]
          .to(torch.float).cpu().numpy()
        let down_blocks_conv_conv_bias = state_dict["down_blocks.\(downBlocks).conv.conv.bias"].to(
          torch.float
        ).cpu().numpy()
        conv.weight.copy(from: try! Tensor<Float>(numpy: down_blocks_conv_conv_weight))
        conv.bias.copy(from: try! Tensor<Float>(numpy: down_blocks_conv_conv_bias))
      }
      j += 1
    }
    for i in 0..<layer.numRepeat {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "down_blocks.\(j).res_blocks.\(i)", channels: channels, depth: depth,
        height: height, width: width, isCausal: true)
      out = block(out)
      readers.append(reader)
    }
    j += 1
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [0], elementwiseAffine: false, name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ]).swish()
  let convOut = Convolution(
    groups: 1, filters: 129, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_out")
  out = convOut(out.padded(.replicate, begin: [0, 0, 2, 0, 0]))
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["conv_in.conv.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["conv_in.conv.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    let conv_out_weight = state_dict["conv_out.conv.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["conv_out.conv.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

graph.withNoGrad {
  print("video vae parity: start")
  let videoModelDevice = vae_decoder.conv_in.conv.weight.device
  let zForVideo = torch.randn([1, 128, 16, 16, 16]).to(torch.float).to(videoModelDevice)
  let decodedVideo = vae_decoder(zForVideo).to(torch.float)
  let encodedVideo = vae_encoder(decodedVideo).to(torch.float)
  let vaeDecoderStateDict = vae_decoder.state_dict()
  let std = graph.variable(
    try! Tensor<Float>(
      numpy: vaeDecoderStateDict["per_channel_statistics.std-of-means"].to(torch.float).view(
        1, -1, 1, 1, 1
      ).cpu().numpy())
  ).toGPU(swiftDevice)
  let mean = graph.variable(
    try! Tensor<Float>(
      numpy: vaeDecoderStateDict["per_channel_statistics.mean-of-means"].to(torch.float).view(
        1, -1, 1, 1, 1
      ).cpu().numpy())
  ).toGPU(swiftDevice)
  var zTensor = graph.variable(
    try! Tensor<Float>(numpy: zForVideo.to(torch.float).cpu().numpy())
  ).toGPU(swiftDevice)
  zTensor = zTensor .* std + mean
  let (decoderReader, decoder) = DecoderCausal3D(
    layers: [
      (channels: 1024, numRepeat: 2, stride: (1, 1, 1)),
      (channels: 512, numRepeat: 2, stride: (2, 2, 2)),
      (channels: 512, numRepeat: 4, stride: (2, 2, 2)),
      (channels: 256, numRepeat: 6, stride: (2, 1, 1)),
      (channels: 128, numRepeat: 4, stride: (1, 2, 2)),
    ], startWidth: 16, startHeight: 16, startDepth: 16)
  decoder.compile(inputs: zTensor)
  exportedVideoDecoder = decoder
  decoderReader(vaeDecoderStateDict)
  let image = decoder(inputs: zTensor)[0].as(of: Float.self)
  let swiftImage = image.toCPU()
  let torchImage = try! Tensor<Float>(numpy: decodedVideo[0].to(torch.float).cpu().numpy())
  print("video decoder sampled max abs diff:", sampledMaxAbsDiff4D(swiftImage, torchImage, sampleCount: 200_000))

  let (encoderReader, encoder) = EncoderCausal3D(
    layers: [
      (channels: 128, numRepeat: 4, stride: (1, 1, 1)),
      (channels: 256, numRepeat: 6, stride: (1, 2, 2)),
      (channels: 512, numRepeat: 4, stride: (2, 1, 1)),
      (channels: 1024, numRepeat: 2, stride: (2, 2, 2)),
      (channels: 1024, numRepeat: 2, stride: (2, 2, 2)),
    ], startWidth: 16, startHeight: 16, startDepth: 16)
  encoder.compile(inputs: image)
  exportedVideoEncoder = encoder
  let vaeEncoderStateDict = vae_encoder.state_dict()
  encoderReader(vaeEncoderStateDict)
  let x = encoder(inputs: image)[0].as(of: Float.self)
  let norm =
    (x[0..<1, 0..<128, 0..<16, 0..<16, 0..<16].contiguous() - mean) .* Functional.reciprocal(std)
  let swiftEncoded = norm.toCPU()
  let torchEncoded = try! Tensor<Float>(numpy: encodedVideo.to(torch.float).cpu().numpy())
  print("video encoder max abs diff:", maxAbsDiff5D(swiftEncoded, torchEncoded))
  print("video vae parity: done")

  guard let exportedAudioDecoder = exportedAudioDecoder,
    let exportedAudioEncoder = exportedAudioEncoder,
    let exportedVocoderWithBWE = exportedVocoderWithBWE,
    let exportedVideoDecoder = exportedVideoDecoder,
    let exportedVideoEncoder = exportedVideoEncoder
  else {
    fatalError("Missing one or more models for ckpt export.")
  }
  let ckptPath = "/home/liu/workspace/swift-diffusion/ltx_2.3_audio_video_vae_f32.ckpt"
  graph.openStore(ckptPath) {
    $0.write("audio_decoder", model: exportedAudioDecoder)
    $0.write("audio_encoder", model: exportedAudioEncoder)
    $0.write("vocoder", model: exportedVocoderWithBWE)
    $0.write("decoder", model: exportedVideoDecoder)
    $0.write("encoder", model: exportedVideoEncoder)
  }
  print("Wrote \(ckptPath)")
}
*/

print("main model: start")
let mainDType = torch.bfloat16
let mainFPS = 25.0
let mainModelPath = "/fast/Data/ltx-2.3-22b-distilled.safetensors"
let mainCkptPath = "/slow/Data/ltx_2.3_22b_distilled_f16.ckpt"
let mainVideoPixelShape = ltx_core_types.VideoPixelShape(
  batch: 1, frames: 121, height: 512, width: 768, fps: mainFPS)
let mainVideoLatentShape = ltx_core_types.VideoLatentShape.from_pixel_shape(mainVideoPixelShape)
let mainAudioLatentShape = ltx_core_types.AudioLatentShape.from_video_pixel_shape(
  mainVideoPixelShape)
let mainVideoPatchifier = ltx_core_components_patchifiers.VideoLatentPatchifier(patch_size: 1)
let mainAudioPatchifier = ltx_core_components_patchifiers.AudioPatchifier(patch_size: 1)
let mainVideoTools = ltx_core_tools.VideoLatentTools(
  patchifier: mainVideoPatchifier, target_shape: mainVideoLatentShape, fps: mainFPS)
let mainAudioTools = ltx_core_tools.AudioLatentTools(
  patchifier: mainAudioPatchifier, target_shape: mainAudioLatentShape)
let MainModality = ltx_core_model_transformer_modality.Modality
let mainVideoLatent = torch.randn(mainVideoLatentShape.to_torch_shape()).to(mainDType).to(
  torch_device)
let mainAudioLatent = torch.randn(mainAudioLatentShape.to_torch_shape()).to(mainDType).to(
  torch_device)
let mainVideoState = mainVideoTools.create_initial_state(
  device: torch_device, dtype: mainDType, initial_latent: mainVideoLatent)
let mainAudioState = mainAudioTools.create_initial_state(
  device: torch_device, dtype: mainDType, initial_latent: mainAudioLatent)
let mainSigma = torch.full([1], 1.0).to(torch.float32).to(torch_device)
let mainVideoTimesteps = mainVideoState.denoise_mask * mainSigma
let mainAudioTimesteps = mainAudioState.denoise_mask * mainSigma
print("main model path:", mainModelPath)
print("main ckpt path:", mainCkptPath)
let mainTransformerBuilder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: mainModelPath,
  model_class_configurator: ltx_core_model_transformer_model_configurator.LTXModelConfigurator,
  model_sd_ops: ltx_core_model_transformer_model_configurator.LTXV_MODEL_COMFY_RENAMING_MAP)
print(mainTransformerBuilder)
let mainTransformer = mainTransformerBuilder.build(device: pythonDevice)
print("main model built")
let mainVideoContext = torch.randn([1, 1024, 4096]).to(mainDType).to(torch_device)
let mainAudioContext = torch.randn([1, 1024, 2048]).to(mainDType).to(torch_device)
// Match Swift-side fixed RoPE (cos=1, sin=0): use zero positions in Python reference.
let mainVideoPositions = torch.zeros_like(mainVideoState.positions)
let mainAudioPositions = torch.zeros_like(mainAudioState.positions)
let mainVideo = MainModality(
  latent: mainVideoState.latent, sigma: mainSigma, timesteps: mainVideoTimesteps,
  positions: mainVideoPositions, context: mainVideoContext, enabled: true,
  context_mask: Python.None,
  attention_mask: mainVideoState.attention_mask)
let mainAudio = MainModality(
  latent: mainAudioState.latent, sigma: mainSigma, timesteps: mainAudioTimesteps,
  positions: mainAudioPositions, context: mainAudioContext, enabled: true,
  context_mask: Python.None,
  attention_mask: mainAudioState.attention_mask)
mainTransformer.to(torch_device)
let mainOutput = mainTransformer(video: mainVideo, audio: mainAudio, perturbations: Python.None)
print("main model output shapes:", mainOutput[0].shape, mainOutput[1].shape)
print(
  "main model sample:",
  mainOutput[0][0][0][0].to(torch.float).cpu().item(),
  mainOutput[1][0][0][0].to(torch.float).cpu().item())

print("main swift dit: start")
let (mainDITReader, mainDIT) = LTX2(b: 1, h: 64, w: 64, useContextProjection: false)
let verifyTextAndConnectorsOnly = false
let verifyTextAndConnectorParity = false
graph.withNoGrad {
  DynamicGraph.logLevel = .none
  if !verifyTextAndConnectorsOnly {
    let xTensor = graph.variable(
      Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: mainVideoState.latent.to(torch.float).cpu().numpy()))
    ).toGPU(1).reshaped(.HWC(1, 6144, 128))
    let txtTensor = graph.variable(
      Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: mainVideoContext.to(torch.float).cpu().numpy()))
    ).toGPU(1).reshaped(.HWC(1, 1024, 4096))
    let aTensor = graph.variable(
      Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: mainAudioState.latent.to(torch.float).cpu().numpy()))
    ).toGPU(1).reshaped(.HWC(1, 121, 128))
    let aTxtTensor = graph.variable(
      Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: mainAudioContext.to(torch.float).cpu().numpy()))
    ).toGPU(1).reshaped(.HWC(1, 1024, 2048))
    let timestepTensor = graph.variable(
      Tensor<Float16>(
        from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
      ).toGPU(1))
    let promptTimestepTensor = graph.variable(
      Tensor<Float16>(
        from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
      ).toGPU(1))
    let rotTensor = graph.variable(.CPU, .HWC(1, 1, 4096), of: Float.self)
    for i in 0..<2048 {
      rotTensor[0, 0, i * 2] = 1
      rotTensor[0, 0, i * 2 + 1] = 0
    }
    let rot1TensorGPU = DynamicGraph.Tensor<FloatType>(
      from: rotTensor.reshaped(.NHWC(1, 1, 32, 128))
    )
    .toGPU(1)
    let rot2TensorGPU = DynamicGraph.Tensor<FloatType>(
      from: rotTensor.reshaped(.NHWC(1, 1, 32, 128))
    )
    .toGPU(1)
    let aRotTensor = graph.variable(.CPU, .HWC(1, 1, 2048), of: Float.self)
    for i in 0..<1024 {
      aRotTensor[0, 0, i * 2] = 1
      aRotTensor[0, 0, i * 2 + 1] = 0
    }
    let rot3TensorGPU = DynamicGraph.Tensor<FloatType>(
      from: aRotTensor.reshaped(.NHWC(1, 1, 32, 64))
    )
    .toGPU(1)
    let rot4TensorGPU = DynamicGraph.Tensor<FloatType>(
      from: aRotTensor.reshaped(.NHWC(1, 1, 32, 64))
    )
    .toGPU(1)
    let rot5TensorGPU = DynamicGraph.Tensor<FloatType>(
      from: aRotTensor.reshaped(.NHWC(1, 1, 32, 64))
    )
    .toGPU(1)
    mainDIT.maxConcurrency = .limit(1)
    print("main swift dit: compile")
    mainDIT.compile(
      inputs: xTensor, txtTensor, aTensor, aTxtTensor, timestepTensor, promptTimestepTensor,
      rot1TensorGPU,
      rot2TensorGPU,
      rot3TensorGPU, rot4TensorGPU, rot5TensorGPU)
    let mainExpectedVideo = try! Tensor<Float>(numpy: mainOutput[0].to(torch.float).cpu().numpy())
    let mainExpectedAudio = try! Tensor<Float>(numpy: mainOutput[1].to(torch.float).cpu().numpy())

    print("main swift dit: load state_dict")
    mainDITReader(mainTransformer.state_dict())
    print("main swift dit: forward")
    let mainSwiftOutput = mainDIT(
      inputs: xTensor, txtTensor, aTensor, aTxtTensor, timestepTensor, promptTimestepTensor,
      rot1TensorGPU,
      rot2TensorGPU,
      rot3TensorGPU, rot4TensorGPU, rot5TensorGPU)
    print("main swift dit output shapes:", mainSwiftOutput[0].shape, mainSwiftOutput[1].shape)
    let mainSwiftVideo = mainSwiftOutput[0].as(of: Float16.self).toCPU()
    let mainSwiftAudio = mainSwiftOutput[1].as(of: Float16.self).toCPU()
    print("main swift dit sample:", Float(mainSwiftVideo[0, 0, 0]), Float(mainSwiftAudio[0, 0, 0]))
    var maxVideoDiff: Float = 0
    var maxExpectedVideoAbs: Float = 0
    for i in 0..<6144 {
      for j in 0..<128 {
        let expectedAbs = abs(mainExpectedVideo[0, i, j])
        if expectedAbs > maxExpectedVideoAbs {
          maxExpectedVideoAbs = expectedAbs
        }
        let diff = abs(Float(mainSwiftVideo[0, i, j]) - mainExpectedVideo[0, i, j])
        if diff > maxVideoDiff {
          maxVideoDiff = diff
        }
      }
    }
    var maxAudioDiff: Float = 0
    var maxExpectedAudioAbs: Float = 0
    for i in 0..<121 {
      for j in 0..<128 {
        let expectedAbs = abs(mainExpectedAudio[0, i, j])
        if expectedAbs > maxExpectedAudioAbs {
          maxExpectedAudioAbs = expectedAbs
        }
        let diff = abs(Float(mainSwiftAudio[0, i, j]) - mainExpectedAudio[0, i, j])
        if diff > maxAudioDiff {
          maxAudioDiff = diff
        }
      }
    }
    print(
      "main swift dit max-abs diff:",
      maxVideoDiff,
      maxAudioDiff,
      "relative:",
      maxVideoDiff / max(1e-6, maxExpectedVideoAbs),
      maxAudioDiff / max(1e-6, maxExpectedAudioAbs))
  } else {
    print("main swift dit parity: skipped")
  }

  let connectorStateDict = safetensors_torch.load_file(mainModelPath, device: "cpu")
  let textFeatureInput = Input()
  let textVideoAggregateEmbed = Dense(count: 4096, name: "video_aggregate_embed")
  let textAudioAggregateEmbed = Dense(count: 2048, name: "audio_aggregate_embed")
  let textFeatureExtractor = Model(
    [textFeatureInput],
    [
      Functional.concat(
        axis: 1, textVideoAggregateEmbed(textFeatureInput),
        textAudioAggregateEmbed(textFeatureInput),
        flags: [.disableOpt])
    ]
  )
  let textFeatureExtractorInput = graph.variable(.GPU(1), .WC(1, 188_160), of: BFloat16.self)
  textFeatureExtractor.compile(inputs: textFeatureExtractorInput)
  let textProjectionVideoWeight = connectorStateDict[
    "text_embedding_projection.video_aggregate_embed.weight"
  ]
  .type(torch.float)
  let textProjectionVideoBias = connectorStateDict[
    "text_embedding_projection.video_aggregate_embed.bias"
  ]
  .type(torch.float)
  let textProjectionAudioWeight = connectorStateDict[
    "text_embedding_projection.audio_aggregate_embed.weight"
  ]
  .type(torch.float)
  let textProjectionAudioBias = connectorStateDict[
    "text_embedding_projection.audio_aggregate_embed.bias"
  ]
  .type(torch.float)
  textVideoAggregateEmbed.weight.copy(
    from: Tensor<BFloat16>(
      from: try! Tensor<Float>(
        numpy: textProjectionVideoWeight.cpu().numpy())))
  textVideoAggregateEmbed.bias.copy(
    from: Tensor<BFloat16>(
      from: try! Tensor<Float>(
        numpy: textProjectionVideoBias.cpu().numpy())))
  textAudioAggregateEmbed.weight.copy(
    from: Tensor<BFloat16>(
      from: try! Tensor<Float>(
        numpy: textProjectionAudioWeight.cpu().numpy())))
  textAudioAggregateEmbed.bias.copy(
    from: Tensor<BFloat16>(
      from: try! Tensor<Float>(
        numpy: textProjectionAudioBias.cpu().numpy())))

  if verifyTextAndConnectorParity {
    let textFeatureInputTorch = torch.randn([1, 188_160], device: torch_device, dtype: torch.float)
    let textFeatureInputTorchBF16 = textFeatureInputTorch.to(torch.bfloat16)
    let textFeatureRefVideo = torch.nn.functional.linear(
      textFeatureInputTorchBF16,
      textProjectionVideoWeight.to(torch_device).to(torch.bfloat16),
      textProjectionVideoBias.to(torch_device).to(torch.bfloat16))
    let textFeatureRefAudio = torch.nn.functional.linear(
      textFeatureInputTorchBF16,
      textProjectionAudioWeight.to(torch_device).to(torch.bfloat16),
      textProjectionAudioBias.to(torch_device).to(torch.bfloat16))
    let textFeatureRef = torch.cat([textFeatureRefVideo, textFeatureRefAudio], dim: 1).to(
      torch.float
    ).cpu()
    let textFeatureInputSwift = graph.variable(
      Tensor<BFloat16>(
        from: try! Tensor<Float>(numpy: textFeatureInputTorch.to(torch.float).cpu().numpy()))
    ).reshaped(.WC(1, 188_160)).copied().toGPU(1)
    let textFeatureSwift = DynamicGraph.Tensor<Float>(
      from: textFeatureExtractor(inputs: textFeatureInputSwift)[0].as(of: BFloat16.self)
    ).toCPU()
    let textFeatureRefTensor = try! Tensor<Float>(numpy: textFeatureRef.numpy())
    var maxTextFeatureDiff: Float = 0
    var maxTextFeatureRefAbs: Float = 0
    for i in 0..<6144 {
      let refAbs = abs(textFeatureRefTensor[0, i])
      if refAbs > maxTextFeatureRefAbs {
        maxTextFeatureRefAbs = refAbs
      }
      let diff = abs(Float(textFeatureSwift[0, i]) - textFeatureRefTensor[0, i])
      if diff > maxTextFeatureDiff {
        maxTextFeatureDiff = diff
      }
    }
    print(
      "text feature extractor max-abs diff:",
      maxTextFeatureDiff,
      "relative:",
      maxTextFeatureDiff / max(1e-6, maxTextFeatureRefAbs))
  } else {
    print("text feature extractor parity: skipped")
  }

  let connectorTokenLength = 1024
  let embeddingsConnectorSourcePath =
    "\(freshLTX2CorePath)/ltx_core/text_encoders/gemma/embeddings_connector.py"
  let embeddingsConnectorPatchedPath = "/tmp/ltx2_embeddings_connector_no_rope.py"
  let embeddingsConnectorSource = try! String(
    contentsOfFile: embeddingsConnectorSourcePath, encoding: .utf8)
  let embeddingsConnectorNoRopeSource = embeddingsConnectorSource.replacingOccurrences(
    of: "pe=freqs_cis", with: "pe=None")
  try! embeddingsConnectorNoRopeSource.write(
    toFile: embeddingsConnectorPatchedPath, atomically: true, encoding: .utf8)
  let importlib_util = Python.import("importlib.util")
  let embeddingsConnectorSpec = importlib_util.spec_from_file_location(
    "ltx2_embeddings_connector_runtime",
    embeddingsConnectorPatchedPath)
  let embeddingsConnectorModule = importlib_util.module_from_spec(embeddingsConnectorSpec)
  _ = embeddingsConnectorSpec.loader.exec_module(embeddingsConnectorModule)
  let attentionModule = Python.import("ltx_core.model.transformer.attention")
  attentionModule.memory_efficient_attention = Python.None

  let videoConnectorLearnableRegisters = graph.variable(
    try! Tensor<Float>(
      numpy: connectorStateDict[
        "model.diffusion_model.video_embeddings_connector.learnable_registers"
      ].type(
        torch.float
      ).cpu().numpy()
    )
  ).toGPU(1)
  let videoConnectorInputTorch = torch.tile(
    connectorStateDict["model.diffusion_model.video_embeddings_connector.learnable_registers"]
      .type(torch.float16).unsqueeze(0), [1, 8, 1]
  ).contiguous()
  let videoConnectorInput = graph.variable(
    Tensor<Float16>(
      from: try! Tensor<Float>(
        numpy: videoConnectorInputTorch.flatten().to(torch.float).cpu().numpy()))
  ).toGPU(1).reshaped(.HWC(1, connectorTokenLength, 4096))
  let videoConnectorRot = graph.variable(.CPU, .HWC(1, connectorTokenLength, 4096), of: Float.self)
  for i in 0..<connectorTokenLength {
    for j in 0..<2048 {
      videoConnectorRot[0, i, j * 2] = 1
      videoConnectorRot[0, i, j * 2 + 1] = 0
    }
  }
  let videoConnectorRotGPU = DynamicGraph.Tensor<FloatType>(
    from: videoConnectorRot.reshaped(.NHWC(1, connectorTokenLength, 32, 128))
  ).toGPU(1)
  let (videoConnector, videoConnectorReader) = Embedding1DConnector(
    prefix: "model.diffusion_model.video_embeddings_connector", layers: 8,
    tokenLength: connectorTokenLength, k: 128, h: 32, useGatedAttention: true)
  videoConnector.compile(inputs: videoConnectorInput, videoConnectorRotGPU)
  videoConnectorReader(connectorStateDict)
  if verifyTextAndConnectorParity {
    let videoConnectorSwift = videoConnector(inputs: videoConnectorInput, videoConnectorRotGPU)[0]
      .as(of: Float16.self)
      .toCPU()
    let videoConnectorRefModule = embeddingsConnectorModule.Embeddings1DConnector(
      attention_head_dim: 128, num_attention_heads: 32, num_layers: 8,
      positional_embedding_max_pos: [4096], num_learnable_registers: Python.None,
      rope_type: embeddingsConnectorModule.LTXRopeType.SPLIT,
      double_precision_rope: true,
      apply_gated_attention: true
    ).to(torch_device).to(torch.float16).eval()
    let videoConnectorRefStateDict = Python.dict()
    let videoConnectorPrefix = "model.diffusion_model.video_embeddings_connector."
    for keyObj in connectorStateDict.keys() {
      let key = String(keyObj)!
      if key.hasPrefix(videoConnectorPrefix) {
        let stripped = String(key.dropFirst(videoConnectorPrefix.count))
        videoConnectorRefStateDict[PythonObject(stripped)] = connectorStateDict[keyObj].type(
          torch.float16
        ).to(
          torch_device)
      }
    }
    _ = videoConnectorRefModule.load_state_dict(videoConnectorRefStateDict, strict: false)
    let videoConnectorRef = videoConnectorRefModule(
      videoConnectorInputTorch.to(torch_device), Python.None)[0].to(
        torch.float
      ).cpu()
    let videoConnectorRefTensor = try! Tensor<Float>(numpy: videoConnectorRef.numpy())
    var maxVideoConnectorDiff: Float = 0
    var maxVideoConnectorRefAbs: Float = 0
    for i in 0..<connectorTokenLength {
      for j in 0..<4096 {
        let refAbs = abs(videoConnectorRefTensor[0, i, j])
        if refAbs > maxVideoConnectorRefAbs {
          maxVideoConnectorRefAbs = refAbs
        }
        let diff = abs(Float(videoConnectorSwift[0, i, j]) - videoConnectorRefTensor[0, i, j])
        if diff > maxVideoConnectorDiff {
          maxVideoConnectorDiff = diff
        }
      }
    }
    print(
      "video connector max-abs diff:",
      maxVideoConnectorDiff,
      "relative:",
      maxVideoConnectorDiff / max(1e-6, maxVideoConnectorRefAbs))
  } else {
    print("video connector parity: skipped")
  }

  let audioConnectorLearnableRegisters = graph.variable(
    try! Tensor<Float>(
      numpy: connectorStateDict[
        "model.diffusion_model.audio_embeddings_connector.learnable_registers"
      ].type(
        torch.float
      ).cpu().numpy()
    )
  ).toGPU(1)
  let audioConnectorInputTorch = torch.tile(
    connectorStateDict["model.diffusion_model.audio_embeddings_connector.learnable_registers"]
      .type(torch.float16).unsqueeze(0), [1, 8, 1]
  ).contiguous()
  let audioConnectorInput = graph.variable(
    Tensor<Float16>(
      from: try! Tensor<Float>(
        numpy: audioConnectorInputTorch.flatten().to(torch.float).cpu().numpy()))
  ).toGPU(1).reshaped(.HWC(1, connectorTokenLength, 2048))
  let audioConnectorRot = graph.variable(.CPU, .HWC(1, connectorTokenLength, 2048), of: Float.self)
  for i in 0..<connectorTokenLength {
    for j in 0..<1024 {
      audioConnectorRot[0, i, j * 2] = 1
      audioConnectorRot[0, i, j * 2 + 1] = 0
    }
  }
  let audioConnectorRotGPU = DynamicGraph.Tensor<FloatType>(
    from: audioConnectorRot.reshaped(.NHWC(1, connectorTokenLength, 32, 64))
  ).toGPU(1)
  let (audioConnector, audioConnectorReader) = Embedding1DConnector(
    prefix: "model.diffusion_model.audio_embeddings_connector", layers: 8,
    tokenLength: connectorTokenLength, k: 64, h: 32, useGatedAttention: true)
  audioConnector.compile(inputs: audioConnectorInput, audioConnectorRotGPU)
  audioConnectorReader(connectorStateDict)
  if verifyTextAndConnectorParity {
    let audioConnectorSwift = audioConnector(inputs: audioConnectorInput, audioConnectorRotGPU)[0]
      .as(of: Float16.self)
      .toCPU()
    let audioConnectorRefModule = embeddingsConnectorModule.Embeddings1DConnector(
      attention_head_dim: 64, num_attention_heads: 32, num_layers: 8,
      positional_embedding_max_pos: [4096], num_learnable_registers: Python.None,
      rope_type: embeddingsConnectorModule.LTXRopeType.SPLIT,
      double_precision_rope: true,
      apply_gated_attention: true
    ).to(torch_device).to(torch.float16).eval()
    let audioConnectorRefStateDict = Python.dict()
    let audioConnectorPrefix = "model.diffusion_model.audio_embeddings_connector."
    for keyObj in connectorStateDict.keys() {
      let key = String(keyObj)!
      if key.hasPrefix(audioConnectorPrefix) {
        let stripped = String(key.dropFirst(audioConnectorPrefix.count))
        audioConnectorRefStateDict[PythonObject(stripped)] = connectorStateDict[keyObj].type(
          torch.float16
        ).to(
          torch_device)
      }
    }
    _ = audioConnectorRefModule.load_state_dict(audioConnectorRefStateDict, strict: false)
    let audioConnectorRef = audioConnectorRefModule(
      audioConnectorInputTorch.to(torch_device), Python.None)[0].to(
        torch.float
      ).cpu()
    let audioConnectorRefTensor = try! Tensor<Float>(numpy: audioConnectorRef.numpy())
    var maxAudioConnectorDiff: Float = 0
    var maxAudioConnectorRefAbs: Float = 0
    for i in 0..<connectorTokenLength {
      for j in 0..<2048 {
        let refAbs = abs(audioConnectorRefTensor[0, i, j])
        if refAbs > maxAudioConnectorRefAbs {
          maxAudioConnectorRefAbs = refAbs
        }
        let diff = abs(Float(audioConnectorSwift[0, i, j]) - audioConnectorRefTensor[0, i, j])
        if diff > maxAudioConnectorDiff {
          maxAudioConnectorDiff = diff
        }
      }
    }
    print(
      "audio connector max-abs diff:",
      maxAudioConnectorDiff,
      "relative:",
      maxAudioConnectorDiff / max(1e-6, maxAudioConnectorRefAbs))
  } else {
    print("audio connector parity: skipped")
  }

  if verifyTextAndConnectorsOnly {
    print("verification done; skipping ckpt export")
    exit(0)
  }

  graph.openStore(mainCkptPath) {
    $0.write("text_feature_extractor", model: textFeatureExtractor)
    $0.write("text_video_connector_learnable_registers", variable: videoConnectorLearnableRegisters)
    $0.write("text_video_connector", model: videoConnector)
    $0.write("text_audio_connector_learnable_registers", variable: audioConnectorLearnableRegisters)
    $0.write("text_audio_connector", model: audioConnector)
    $0.write("dit", model: mainDIT)
  }
  print("Wrote \(mainCkptPath)")
}

exit(0)

/* Legacy ltx2.0 text-encoder experiment (kept for reference; unreachable).
let prompt = "a dance party"
// Text Encoder
let text_encoder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: "/fast/Data/ltx-2-19b-dev.safetensors",
  model_class_configurator: ltx_core_text_encoders_gemma_encoders_av_encoder
    .AVGemmaTextEncoderModelConfigurator.with_gemma_root_path(
      "/slow/Data/google/gemma-3-12b-it-qat-q4_0-unquantized"),
  model_sd_ops: ltx_core_text_encoders_gemma_encoders_av_encoder.AV_GEMMA_TEXT_ENCODER_KEY_OPS
).build(device: "cuda")
text_encoder.to(torch_device)
let token_pairs = text_encoder.tokenizer.tokenize_with_weights(prompt)["gemma"]
let inputIdsAndMask = text_encoder._process_token_pairs(token_pairs)
let inputIds = inputIdsAndMask[0]
let attentionMask = inputIdsAndMask[1]
print(text_encoder)

text_encoder(prompt)
// text_encoder.model(input_ids: inputIds, attention_mask: attentionMask, output_hidden_states: true)

let text_encoder_state_dict = text_encoder.cpu().state_dict()
let text_state_dict = text_encoder.model.language_model.state_dict()

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/swift-diffusion/examples/ltx2/tokenizer.model")
var positiveTokens = sentencePiece.encode(prompt).map { return $0.id }
positiveTokens.insert(2, at: 0)
*/

func SelfAttention(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: width, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    // The rotary in Qwen is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    let q_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).view(
      16, 2, 128, 3_840
    ).transpose(1, 2).cpu().numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let q_norm_weight =
      (state_dict["\(prefix).self_attn.q_norm.weight"].type(torch.float).view(
        2, 128
      ).transpose(0, 1).cpu() + 1).numpy()
    normQ.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_norm_weight)))
    let k_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).view(
      8, 2, 128, 3_840
    ).transpose(1, 2).cpu().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    let k_norm_weight =
      (state_dict["\(prefix).self_attn.k_norm.weight"].type(torch.float).view(
        2, 128
      ).transpose(0, 1).cpu() + 1).numpy()
    normK.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_norm_weight)))
    let v_weight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: v_weight)))
    let proj_weight = state_dict["\(prefix).self_attn.o_proj.weight"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_weight)))
  }
  return (Model([x, rot], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x) .* w1(x).GELU(approximate: .tanh)
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out]))
}

func TransformerBlock(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x).to(.Float16)
  let (attention, attnReader) = SelfAttention(
    prefix: prefix, width: width, k: k, h: h, hk: hk, b: b, t: t)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(attention(out, rot).to(of: x)) + x
  let residual = out
  let norm3 = RMSNorm(epsilon: 1e-6, axis: [1], name: "pre_feedforward_layernorm")
  out = norm3(out).to(.Float16)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: width, intermediateSize: MLP, name: "mlp")
  let norm4 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_feedforward_layernorm")
  out = residual + norm4(ffn(out).to(of: residual))
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight =
      (state_dict["\(prefix).input_layernorm.weight"].type(torch.float)
      .cpu() + 1).numpy()
    norm1.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm1_weight)))
    let norm2_weight =
      (state_dict["\(prefix).post_attention_layernorm.weight"].type(torch.float)
      .cpu() + 1).numpy()
    norm2.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm2_weight)))
    attnReader(state_dict)
    let norm3_weight =
      (state_dict["\(prefix).pre_feedforward_layernorm.weight"].type(torch.float)
      .cpu() + 1).numpy()
    norm3.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm3_weight)))
    let norm4_weight =
      (state_dict["\(prefix).post_feedforward_layernorm.weight"].type(torch.float)
      .cpu() + 1).numpy()
    norm4.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm4_weight)))
    let w1_weight = state_dict["\(prefix).mlp.gate_proj.weight"].type(torch.float).cpu()
      .numpy()
    w1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w1_weight)))
    let w2_weight = state_dict["\(prefix).mlp.down_proj.weight"].type(torch.float).cpu()
      .numpy()
    w2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w2_weight)))
    let w3_weight = state_dict["\(prefix).mlp.up_proj.weight"].type(torch.float).cpu()
      .numpy()
    w3.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w3_weight)))
  }
  return (Model([x, rot], [out]), reader)
}

func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["embed_tokens.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: Tensor<BFloat16>(from: try! Tensor<Float>(numpy: vocab)))
  }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rotLocal = Input()
  let rot = Input()
  let (embedding, embedReader) = TextEmbedding(
    BFloat16.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = 62 * embedding(tokens).to(.Float32)
  var readers = [(PythonObject) -> Void]()
  var hiddenStates = [Model.IO]()
  for i in 0..<layers {
    hiddenStates.append(out)
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", width: width, k: 256, h: heads, hk: 8, b: batchSize,
      t: tokenLength, MLP: MLP)
    out = layer(out, (i + 1) % 6 == 0 ? rot : rotLocal)
    readers.append(reader)
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  hiddenStates.append(norm(out))
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_weight = (state_dict["norm.weight"].type(torch.float).cpu() + 1).numpy()
    norm.weight.copy(from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm_weight)))
  }
  return (Model([tokens, rotLocal, rot], hiddenStates), reader)
}

func BasicTransformerBlock1D(
  prefix: String, k: Int, h: Int, b: Int, t: Int, useGatedAttention: Bool
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let normX = norm(x).to(.Float16)
  let rot = Input()
  let toKeys = Dense(count: k * h, name: "to_k")
  let toQueries = Dense(count: k * h, name: "to_q")
  let toValues = Dense(count: k * h, name: "to_v")
  let toGate = useGatedAttention ? Dense(count: h, name: "to_gate") : nil
  var keys = toKeys(normX)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "norm_k")
  keys = normK(keys).reshaped([b, t, h, k])
  var queries = toQueries(normX)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "norm_q")
  queries = normQ(queries).reshaped([b, t, h, k])
  let values = toValues(normX).reshaped([b, t, h, k])
  queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, h, k])
  if let toGate = toGate {
    let gates = (2 * toGate(normX).sigmoid()).reshaped([b, t, h, 1]).to(.Float16)
    out = out .* gates
  }
  out = out.reshaped([b, t, k * h])
  let unifyheads = Dense(count: k * h, name: "to_o")
  out = unifyheads(out).to(of: x) + x
  let residual = out
  let upProj = Dense(count: k * h * 4, name: "up_proj")
  out = (1.0 / 8.0) * upProj(norm(out).to(.Float16)).GELU(approximate: .tanh)
  let downProj = Dense(count: k * h, name: "down_proj")
  out = Add(leftScalar: 8, rightScalar: 1)(downProj(out).to(of: residual), residual)
  let reader: (PythonObject) -> Void = { state_dict in
    let to_q_weight = state_dict["\(prefix).attn1.to_q.weight"].type(torch.float).cpu()
    let to_q_bias = state_dict["\(prefix).attn1.to_q.bias"].type(torch.float).cpu()
    let q_weight = to_q_weight.view(
      h, 2, k / 2, k * h
    ).transpose(1, 2).cpu().numpy()
    let q_bias = to_q_bias.view(
      h, 2, k / 2
    ).transpose(1, 2).cpu().numpy()
    toQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    toQueries.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_bias)))
    let to_k_weight = state_dict["\(prefix).attn1.to_k.weight"].type(torch.float).cpu()
    let to_k_bias = state_dict["\(prefix).attn1.to_k.bias"].type(torch.float).cpu()
    let k_weight = to_k_weight.view(
      h, 2, k / 2, k * h
    ).transpose(1, 2).cpu().numpy()
    let k_bias = to_k_bias.view(
      h, 2, k / 2
    ).transpose(1, 2).cpu().numpy()
    toKeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    toKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_bias)))
    let to_v_weight = state_dict["\(prefix).attn1.to_v.weight"].type(torch.float).cpu().numpy()
    let to_v_bias = state_dict["\(prefix).attn1.to_v.bias"].type(torch.float).cpu().numpy()
    toValues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_v_weight)))
    toValues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_v_bias)))
    if let toGate = toGate {
      let to_gate_logits_weight = state_dict["\(prefix).attn1.to_gate_logits.weight"].type(
        torch.float
      ).cpu()
        .numpy()
      let to_gate_logits_bias = state_dict["\(prefix).attn1.to_gate_logits.bias"].type(torch.float)
        .cpu()
        .numpy()
      toGate.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_gate_logits_weight)))
      toGate.bias.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_gate_logits_bias)))
    }
    let to_out_0_weight = state_dict["\(prefix).attn1.to_out.0.weight"].type(torch.float).cpu()
      .numpy()
    let to_out_0_bias = state_dict["\(prefix).attn1.to_out.0.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_out_0_weight)))
    unifyheads.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_out_0_bias)))
    let norm_k_weight = state_dict["\(prefix).attn1.k_norm.weight"]
      .to(torch.float).cpu().view(h, 2, k / 2).transpose(1, 2).cpu().numpy()
    normK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_k_weight)))
    let norm_q_weight = state_dict["\(prefix).attn1.q_norm.weight"]
      .to(torch.float).cpu().view(h, 2, k / 2).transpose(1, 2).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_q_weight)))

    let ff_net_0_proj_weight = state_dict["\(prefix).ff.net.0.proj.weight"].type(torch.float).cpu()
      .numpy()
    let ff_net_0_proj_bias = state_dict["\(prefix).ff.net.0.proj.bias"].type(torch.float).cpu()
      .numpy()
    upProj.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_net_0_proj_weight)))
    upProj.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_net_0_proj_bias)))
    let ff_net_2_weight = state_dict["\(prefix).ff.net.2.weight"].type(torch.float).cpu().numpy()
    let ff_net_2_bias = ((1.0 / 8) * state_dict["\(prefix).ff.net.2.bias"].type(torch.float).cpu())
      .numpy()
    downProj.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_net_2_weight)))
    downProj.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_net_2_bias)))
  }
  return (reader, Model([x, rot], [out]))
}

func Embedding1DConnector(
  prefix: String, layers: Int, tokenLength: Int, k: Int, h: Int, useGatedAttention: Bool
) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  var readers = [(PythonObject) -> Void]()
  var out: Model.IO = x
  for i in 0..<layers {
    let (reader, block) = BasicTransformerBlock1D(
      prefix: "\(prefix).transformer_1d_blocks.\(i)", k: k, h: h, b: 1, t: tokenLength,
      useGatedAttention: useGatedAttention)
    out = block(out, rot)
    readers.append(reader)
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = norm(out).to(.Float16)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x, rot], [out]), reader)
}

/* Legacy ltx2.0 connector + export experiment (kept for reference; unreachable).
let _ = graph.withNoGrad {
  let positiveRotTensor = graph.variable(
    .CPU, .NHWC(1, positiveTokens.count, 1, 256), of: Float.self)
  for i in 0..<positiveTokens.count {
    for k in 0..<128 {
      let theta = Double(i) * 0.125 / pow(1_000_000, Double(k) * 2 / 256)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      positiveRotTensor[0, i, 0, k * 2] = Float(costheta)
      positiveRotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let positiveRotTensorLocal = graph.variable(
    .CPU, .NHWC(1, positiveTokens.count, 1, 256), of: Float.self)
  for i in 0..<positiveTokens.count {
    for k in 0..<128 {
      let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 256)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      positiveRotTensorLocal[0, i, 0, k * 2] = Float(costheta)
      positiveRotTensorLocal[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let (transformer, reader) = Transformer(
    Float16.self, vocabularySize: 262_208, maxLength: positiveTokens.count, width: 3_840,
    tokenLength: positiveTokens.count,
    layers: 48, MLP: 15_360, heads: 16, batchSize: 1
  )
  let positiveTokensTensor = graph.variable(
    .CPU, format: .NHWC, shape: [positiveTokens.count], of: Int32.self)
  for i in 0..<positiveTokens.count {
    positiveTokensTensor[i] = positiveTokens[i]
  }
  let positiveTokensTensorGPU = positiveTokensTensor.toGPU(1)
  let positiveRotTensorLocalGPU = DynamicGraph.Tensor<Float16>(from: positiveRotTensorLocal).toGPU(
    1)
  let positiveRotTensorGPU = DynamicGraph.Tensor<Float16>(from: positiveRotTensor).toGPU(1)
  transformer.compile(
    inputs: positiveTokensTensorGPU, positiveRotTensorLocalGPU, positiveRotTensorGPU)
  reader(text_state_dict)
  let positiveHiddenStates = transformer(
    inputs: positiveTokensTensorGPU, positiveRotTensorLocalGPU, positiveRotTensorGPU
  ).map {
    $0.as(of: Float.self)
  }
  graph.openStore("/home/liu/workspace/swift-diffusion/gemma_3_12b_it_qat_f16.ckpt") {
    $0.write("text_model", model: transformer)
  }
  let featureExtractorLinear = Dense(count: 3840, noBias: true)
  var hiddenStates = graph.variable(.GPU(1), .HWC(positiveTokens.count, 3840, 49), of: Float.self)
  for i in 0..<49 {
    hiddenStates[0..<positiveTokens.count, 0..<3840, i..<(i + 1)] = positiveHiddenStates[i]
      .reshaped(.HWC(positiveTokens.count, 3840, 1))
  }
  let mean = hiddenStates.reduced(.mean, axis: [0, 1])
  let min_ = hiddenStates.reduced(.min, axis: [0, 1])
  let max_ = hiddenStates.reduced(.max, axis: [0, 1])
  let range_ = 8.0 * Functional.reciprocal(max_ - min_)
  let normedHiddenStates = DynamicGraph.Tensor<BFloat16>(from: (hiddenStates - mean) .* range_)
    .reshaped(.WC(positiveTokens.count, 3840 * 49))
  featureExtractorLinear.compile(inputs: normedHiddenStates)
  let aggregate_embed_weight = text_encoder_state_dict[
    "feature_extractor_linear.aggregate_embed.weight"
  ].type(torch.float).cpu().numpy()
  featureExtractorLinear.weight.copy(
    from: Tensor<BFloat16>(from: try! Tensor<Float>(numpy: aggregate_embed_weight)))
  let features = DynamicGraph.Tensor<Float>(
    from: featureExtractorLinear(inputs: normedHiddenStates)[0].as(of: BFloat16.self))
  let connectorLearnableRegisters = graph.variable(
    try! Tensor<Float>(
      numpy: text_encoder_state_dict["embeddings_connector.learnable_registers"].type(torch.float)
        .cpu().numpy()
    ).toGPU(1))
  let audioConnectorLearnableRegisters = graph.variable(
    try! Tensor<Float>(
      numpy: text_encoder_state_dict["audio_embeddings_connector.learnable_registers"].type(
        torch.float
      ).cpu().numpy()
    ).toGPU(1))
  var videoHiddenStates = graph.variable(.GPU(1), .HWC(1, 1024, 3840), of: Float.self)
  for i in 0..<8 {
    videoHiddenStates[0..<1, (i * 128)..<((i + 1) * 128), 0..<3840] =
      connectorLearnableRegisters.reshaped(.HWC(1, 128, 3840))
  }
  videoHiddenStates[0..<1, 0..<positiveTokens.count, 0..<3840] = features.reshaped(
    .HWC(1, positiveTokens.count, 3840))
  let rotTensor = graph.variable(.CPU, .HWC(1, 1024, 3840), of: Float.self)
  for i in 0..<1024 {
    let fractionalPosition = Double(i) / 4096 * 2 - 1
    for j in 0..<1920 {
      let theta: Double = pow(10_000, Double(j) / 1919) * .pi * 0.5
      let freq = theta * fractionalPosition
      let cosFreq = cos(freq)
      let sinFreq = sin(freq)
      rotTensor[0, i, j * 2] = Float(cosFreq)
      rotTensor[0, i, j * 2 + 1] = Float(sinFreq)
    }
  }
  let rotTensorGPU = DynamicGraph.Tensor<FloatType>(
    from: rotTensor.reshaped(.NHWC(1, 1024, 30, 128))
  )
  .toGPU(1)
  let (connector, connectorReader) = Embedding1DConnector(
    prefix: "embeddings_connector", layers: 2, tokenLength: 1024, k: 128, h: 30,
    useGatedAttention: false)
  connector.compile(inputs: videoHiddenStates, rotTensorGPU)
  connectorReader(text_encoder_state_dict)
  debugPrint(connector(inputs: videoHiddenStates, rotTensorGPU))
  for i in 0..<8 {
    videoHiddenStates[0..<1, (i * 128)..<((i + 1) * 128), 0..<3840] =
      audioConnectorLearnableRegisters.reshaped(.HWC(1, 128, 3840))
  }
  videoHiddenStates[0..<1, 0..<positiveTokens.count, 0..<3840] = features.reshaped(
    .HWC(1, positiveTokens.count, 3840))
  let (audioConnector, audioConnectorReader) = Embedding1DConnector(
    prefix: "audio_embeddings_connector", layers: 2, tokenLength: 1024, k: 128, h: 30,
    useGatedAttention: false)
  audioConnector.compile(inputs: videoHiddenStates, rotTensorGPU)
  audioConnectorReader(text_encoder_state_dict)
  debugPrint(audioConnector(inputs: videoHiddenStates, rotTensorGPU))
  graph.openStore("/home/liu/workspace/swift-diffusion/ltx_2_19b_dev_f16.ckpt") {
    $0.write("text_feature_extractor", model: featureExtractorLinear)
    $0.write("text_video_connector_learnable_registers", variable: connectorLearnableRegisters)
    $0.write("text_video_connector", model: connector)
    $0.write("text_audio_connector_learnable_registers", variable: audioConnectorLearnableRegisters)
    $0.write("text_audio_connector", model: audioConnector)
  }
  return positiveHiddenStates
}

let audio_builder = ltx_core_conditioning.AudioConditioningBuilder(
  patchifier: ltx_core_components_patchifiers.AudioPatchifier(patch_size: 1), batch: 1,
  duration: 121.0 / 25.0)
let video_builder = ltx_core_conditioning.VideoConditioningBuilder(
  patchifier: ltx_core_components_patchifiers.VideoLatentPatchifier(patch_size: 1),
  batch: 1, width: 768, height: 512, num_frames: 121, fps: 25.0)
let generator = torch.Generator(device: torch_device)
generator.manual_seed(42)
let audio_input = audio_builder.build(
  device: torch_device, dtype: torch.bfloat16, generator: generator)
let video_input = video_builder.build(
  device: torch_device, dtype: torch.bfloat16, generator: generator)

let transformer_builder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: "/fast/Data/ltx-2-19b-dev.safetensors",
  model_class_configurator: ltx_core_model_transformer_model_configurator.LTXModelConfigurator,
  model_sd_ops: ltx_core_loader_sd_ops.LTXV_LORA_COMFY_RENAMING_MAP)
print(transformer_builder)
let Modality = ltx_core_model_transformer_modality.Modality
let transformer = transformer_builder.build(device: "cuda")
print(transformer)

let timesteps = torch.full([1, 6144], 1).to(torch.bfloat16).cuda()
let audio_timesteps = torch.full([1, 121], 1).to(torch.bfloat16).cuda()

let video = Modality(
  enabled: true, latent: torch.randn([1, 6144, 128]).to(torch.bfloat16).cuda(),
  timesteps: timesteps, positions: video_input.positions.to(torch.bfloat16).cuda(),
  context: torch.randn([1, 1024, 3840]).to(torch.bfloat16).cuda(), context_mask: Python.None)
let audio = Modality(
  enabled: true, latent: torch.randn([1, 121, 128]).to(torch.bfloat16).cuda(),
  timesteps: audio_timesteps, positions: audio_input.positions.to(torch.bfloat16).cuda(),
  context: torch.randn([1, 1024, 3840]).to(torch.bfloat16).cuda(), context_mask: Python.None)

transformer.to(torch_device)
let output = transformer(video: video, audio: audio, perturbations: Python.None)

let state_dict = transformer.state_dict()
*/

func GELUMLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func LTX2SelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int, name: String) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let rot = Input()
  let toKeys = Dense(count: k * h, name: "\(name)_k")
  let toQueries = Dense(count: k * h, name: "\(name)_q")
  let toValues = Dense(count: k * h, name: "\(name)_v")
  let toGate = Dense(count: h, name: "\(name)_gate")
  var keys = toKeys(x)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_k")
  keys = normK(keys).reshaped([b, t, h, k])
  var queries = toQueries(x)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_q")
  queries = normQ(queries).reshaped([b, t, h, k])
  let values = toValues(x).reshaped([b, t, h, k])
  queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, h, k])
  let gates = (2 * toGate(x).sigmoid()).reshaped([b, t, h, 1]).to(.Float16)
  out = out .* gates
  out = out.reshaped([b, t, k * h])
  let unifyheads = Dense(count: k * h, name: "\(name)_o")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let to_q_weight = state_dict["\(prefix).to_q.weight"].type(torch.float).cpu()
    let to_q_bias = state_dict["\(prefix).to_q.bias"].type(torch.float).cpu()
    let q_weight = to_q_weight.view(
      h, 2, k / 2, k * h
    ).transpose(1, 2).cpu().numpy()
    let q_bias = to_q_bias.view(
      h, 2, k / 2
    ).transpose(1, 2).cpu().numpy()
    toQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    toQueries.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_bias)))
    let to_k_weight = state_dict["\(prefix).to_k.weight"].type(torch.float).cpu()
    let to_k_bias = state_dict["\(prefix).to_k.bias"].type(torch.float).cpu()
    let k_weight = to_k_weight.view(
      h, 2, k / 2, k * h
    ).transpose(1, 2).cpu().numpy()
    let k_bias = to_k_bias.view(
      h, 2, k / 2
    ).transpose(1, 2).cpu().numpy()
    toKeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    toKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_bias)))
    let to_v_weight = state_dict["\(prefix).to_v.weight"].type(torch.float).cpu().numpy()
    let to_v_bias = state_dict["\(prefix).to_v.bias"].type(torch.float).cpu().numpy()
    toValues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_v_weight)))
    toValues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_v_bias)))
    let to_gate_logits_weight = state_dict["\(prefix).to_gate_logits.weight"].type(torch.float)
      .cpu()
      .numpy()
    let to_gate_logits_bias = state_dict["\(prefix).to_gate_logits.bias"].type(torch.float).cpu()
      .numpy()
    toGate.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_gate_logits_weight)))
    toGate.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_gate_logits_bias)))
    let to_out_0_weight = state_dict["\(prefix).to_out.0.weight"].type(torch.float).cpu()
      .numpy()
    let to_out_0_bias = state_dict["\(prefix).to_out.0.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_out_0_weight)))
    unifyheads.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_out_0_bias)))
    let norm_k_weight = state_dict["\(prefix).k_norm.weight"]
      .to(torch.float).cpu().view(h, 2, k / 2).transpose(1, 2).cpu().numpy()
    normK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_k_weight)))
    let norm_q_weight = state_dict["\(prefix).q_norm.weight"]
      .to(torch.float).cpu().view(h, 2, k / 2).transpose(1, 2).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_q_weight)))
  }
  return (reader, Model([x, rot], [out]))
}

func LTX2CrossAttention(
  prefix: String, k: (Int, Int, Int), h: Int, b: Int, t: (Int, Int), name: String,
  useRotary: Bool = true
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let context = Input()
  let rot = Input()
  let rotK = Input()
  let toKeys = Dense(count: k.1 * h, name: "\(name)_k")
  let toQueries = Dense(count: k.1 * h, name: "\(name)_q")
  let toValues = Dense(count: k.1 * h, name: "\(name)_v")
  let toGate = Dense(count: h, name: "\(name)_gate")
  var keys = toKeys(context)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_k")
  keys = normK(keys).reshaped([b, t.1, h, k.1])
  var queries = toQueries(x)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_q")
  queries = normQ(queries).reshaped([b, t.0, h, k.1])
  let values = toValues(context).reshaped([b, t.1, h, k.1])
  let attentionScale = 1 / Float(k.1).squareRoot().squareRoot()
  if useRotary {
    queries = attentionScale * Functional.cmul(left: queries, right: rot)
    keys = attentionScale * Functional.cmul(left: keys, right: rotK)
  } else {
    queries = attentionScale * queries
    keys = attentionScale * keys
  }
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t.0, h, k.1])
  let gates = (2 * toGate(x).sigmoid()).reshaped([b, t.0, h, 1]).to(.Float16)
  out = out .* gates
  out = out.reshaped([b, t.0, k.1 * h])
  let unifyheads = Dense(count: k.0 * h, name: "\(name)_o")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let to_q_weight = state_dict["\(prefix).to_q.weight"].type(torch.float).cpu()
    let to_q_bias = state_dict["\(prefix).to_q.bias"].type(torch.float).cpu()
    let q_weight = to_q_weight.view(
      h, 2, k.1 / 2, k.0 * h
    ).transpose(1, 2).cpu().numpy()
    let q_bias = to_q_bias.view(
      h, 2, k.1 / 2
    ).transpose(1, 2).cpu().numpy()
    toQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    toQueries.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_bias)))
    let to_k_weight = state_dict["\(prefix).to_k.weight"].type(torch.float).cpu()
    let to_k_bias = state_dict["\(prefix).to_k.bias"].type(torch.float).cpu()
    let k_weight = to_k_weight.view(
      h, 2, k.1 / 2, k.2 * h
    ).transpose(1, 2).cpu().numpy()
    let k_bias = to_k_bias.view(
      h, 2, k.1 / 2
    ).transpose(1, 2).cpu().numpy()
    toKeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    toKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_bias)))
    let to_v_weight = state_dict["\(prefix).to_v.weight"].type(torch.float).cpu().numpy()
    let to_v_bias = state_dict["\(prefix).to_v.bias"].type(torch.float).cpu().numpy()
    toValues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_v_weight)))
    toValues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_v_bias)))
    let to_gate_logits_weight = state_dict["\(prefix).to_gate_logits.weight"].type(torch.float)
      .cpu()
      .numpy()
    let to_gate_logits_bias = state_dict["\(prefix).to_gate_logits.bias"].type(torch.float).cpu()
      .numpy()
    toGate.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_gate_logits_weight)))
    toGate.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_gate_logits_bias)))
    let to_out_0_weight = state_dict["\(prefix).to_out.0.weight"].type(torch.float).cpu()
      .numpy()
    let to_out_0_bias = state_dict["\(prefix).to_out.0.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_out_0_weight)))
    unifyheads.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: to_out_0_bias)))
    let norm_k_weight = state_dict["\(prefix).k_norm.weight"]
      .to(torch.float).cpu().view(h, 2, k.1 / 2).transpose(1, 2).cpu().numpy()
    normK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_k_weight)))
    let norm_q_weight = state_dict["\(prefix).q_norm.weight"]
      .to(torch.float).cpu().view(h, 2, k.1 / 2).transpose(1, 2).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_q_weight)))
  }
  if useRotary {
    return (reader, Model([x, rot, context, rotK], [out]))
  } else {
    return (reader, Model([x, context], [out]))
  }
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  return (linear1, outProjection, Model([x], [out]))
}

func LTX2TransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, a: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let vx = Input()
  let ax = Input()
  let cv = Input()
  let ca = Input()
  let rot = Input()
  let rotC = Input()
  let rotA = Input()
  let rotAC = Input()
  let rotCX = Input()
  let timesteps = (0..<9).map { _ in Input() }
  let attn1Modulations = (0..<9).map {
    Parameter<Float>(.GPU(1), .HWC(1, 1, k * h), name: "attn1_ada_ln_\($0)")
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var out =
    norm(vx) .* (1 + (attn1Modulations[1] + timesteps[1])) + (attn1Modulations[0] + timesteps[0])
  let (attn1Reader, attn1) = LTX2SelfAttention(
    prefix: "\(prefix).attn1", k: k, h: h, b: b, t: hw, name: "x")
  out = vx + attn1(out.to(.Float16), rot).to(of: vx) .* (attn1Modulations[2] + timesteps[2])
  let (attn2Reader, attn2) = LTX2CrossAttention(
    prefix: "\(prefix).attn2", k: (k, k, k), h: h, b: b, t: (hw, t), name: "cv", useRotary: false)
  let promptTimesteps = (0..<2).map { _ in Input() }
  let promptScaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(1), .HWC(1, 1, k * h), name: "prompt_scale_shift_ada_ln_\($0)")
  }
  let normOut = norm(out)
  let normOutScaled =
    normOut .* (1 + (attn1Modulations[7] + timesteps[7]))
    + (attn1Modulations[6] + timesteps[6])
  let cvScale = (promptScaleShiftModulations[1] + promptTimesteps[1]).to(.Float16)
  let cvShift = (promptScaleShiftModulations[0] + promptTimesteps[0]).to(.Float16)
  let cvScaled = cv .* (1 + cvScale) + cvShift
  out = out + attn2(normOutScaled.to(.Float16), cvScaled.to(.Float16)).to(of: out)
    .* (attn1Modulations[8] + timesteps[8])
  let audioTimesteps = (0..<9).map { _ in Input() }
  let audioAttn1Modulations = (0..<9).map {
    Parameter<Float>(.GPU(1), .HWC(1, 1, k / 2 * h), name: "audio_attn1_ada_ln_\($0)")
  }
  let (audioAttn1Reader, audioAttn1) = LTX2SelfAttention(
    prefix: "\(prefix).audio_attn1", k: k / 2, h: h, b: b, t: a, name: "a")
  var aOut =
    norm(ax) .* (1 + (audioAttn1Modulations[1] + audioTimesteps[1]))
    + (audioAttn1Modulations[0] + audioTimesteps[0])
  aOut = ax + audioAttn1(aOut.to(.Float16), rotA).to(of: ax)
    .* (audioAttn1Modulations[2] + audioTimesteps[2])
  let (audioAttn2Reader, audioAttn2) = LTX2CrossAttention(
    prefix: "\(prefix).audio_attn2", k: (k / 2, k / 2, k / 2), h: h, b: b, t: (a, t), name: "ca",
    useRotary: false)
  let audioPromptTimesteps = (0..<2).map { _ in Input() }
  let audioPromptScaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(1), .HWC(1, 1, k / 2 * h), name: "audio_prompt_scale_shift_ada_ln_\($0)")
  }
  let normAOut = norm(aOut)
  let normAOutScaled =
    normAOut .* (1 + (audioAttn1Modulations[7] + audioTimesteps[7]))
    + (audioAttn1Modulations[6] + audioTimesteps[6])
  let caScale = (audioPromptScaleShiftModulations[1] + audioPromptTimesteps[1]).to(.Float16)
  let caShift = (audioPromptScaleShiftModulations[0] + audioPromptTimesteps[0]).to(.Float16)
  let caScaled = ca .* (1 + caScale) + caShift
  aOut =
    aOut + audioAttn2(normAOutScaled.to(.Float16), caScaled.to(.Float16)).to(of: aOut)
    .* (audioAttn1Modulations[8] + audioTimesteps[8])
  let vxNorm3 = norm(out)
  let axNorm3 = norm(aOut)
  let (audioToVideoAttnReader, audioToVideoAttn) = LTX2CrossAttention(
    prefix: "\(prefix).audio_to_video_attn", k: (k, k / 2, k / 2), h: h, b: b, t: (hw, a),
    name: "ax")
  let caScaleShiftTimesteps = (0..<4).map { _ in Input() }
  let caGateTimesteps = Input()
  let audioToVideoAttnModulations = (0..<5).map {
    if $0 < 2 {
      return Parameter<Float>(
        .GPU(1), .HWC(1, 1, k * h), name: "audio_to_video_attn_ada_ln_\($0)")
    } else if $0 < 4 {
      return Parameter<Float>(
        .GPU(1), .HWC(1, 1, k / 2 * h), name: "audio_to_video_attn_ada_ln_\($0)")
    } else {
      return Parameter<Float>(
        .GPU(1), .HWC(1, 1, k * h), name: "audio_to_video_attn_ada_ln_\($0)")
    }
  }
  let vxScaled =
    vxNorm3 .* (1 + (audioToVideoAttnModulations[1] + caScaleShiftTimesteps[1]))
    + (audioToVideoAttnModulations[0] + caScaleShiftTimesteps[0])
  let axScaled =
    axNorm3 .* (1 + (audioToVideoAttnModulations[3] + caScaleShiftTimesteps[3]))
    + (audioToVideoAttnModulations[2] + caScaleShiftTimesteps[2])
  out =
    out + audioToVideoAttn(vxScaled.to(.Float16), rotCX, axScaled.to(.Float16), rotA).to(of: out)
    .* (audioToVideoAttnModulations[4] + caGateTimesteps)
  let (videoToAudioAttnReader, videoToAudioAttn) = LTX2CrossAttention(
    prefix: "\(prefix).video_to_audio_attn", k: (k / 2, k / 2, k), h: h, b: b, t: (a, hw),
    name: "xa")
  let audioCaScaleShiftTimesteps = (0..<4).map { _ in Input() }
  let audioCaGateTimesteps = Input()
  let videoToAudioAttnModulations = (0..<5).map {
    if $0 < 2 {
      return Parameter<Float>(
        .GPU(1), .HWC(1, 1, k * h), name: "video_to_audio_attn_ada_ln_\($0)")
    } else {
      return Parameter<Float>(
        .GPU(1), .HWC(1, 1, k / 2 * h), name: "video_to_audio_attn_ada_ln_\($0)")
    }
  }
  let audioVxScaled =
    vxNorm3 .* (1 + (videoToAudioAttnModulations[1] + audioCaScaleShiftTimesteps[1]))
    + (videoToAudioAttnModulations[0] + audioCaScaleShiftTimesteps[0])
  let audioAxScaled =
    axNorm3 .* (1 + (videoToAudioAttnModulations[3] + audioCaScaleShiftTimesteps[3]))
    + (videoToAudioAttnModulations[2] + audioCaScaleShiftTimesteps[2])
  aOut =
    aOut
    + videoToAudioAttn(audioAxScaled.to(.Float16), rotA, audioVxScaled.to(.Float16), rotCX).to(
      of: aOut)
    .* (videoToAudioAttnModulations[4] + audioCaGateTimesteps)
  // Now attention done, do MLP.
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: 4096, intermediateSize: 4096 * 4, name: "x")
  let lastVxScaled =
    norm(out) .* (1 + (attn1Modulations[4] + timesteps[4])) + (attn1Modulations[3] + timesteps[3])
  out = out + xFF(lastVxScaled.to(.Float16)).to(of: out) .* (attn1Modulations[5] + timesteps[5])
  let lastAxScaled =
    norm(aOut) .* (1 + (audioAttn1Modulations[4] + audioTimesteps[4]))
    + (audioAttn1Modulations[3] + audioTimesteps[3])
  let (audioLinear1, audioOutProjection, audioFF) = FeedForward(
    hiddenSize: 2048, intermediateSize: 2048 * 4, name: "a")
  aOut = aOut + audioFF(lastAxScaled.to(.Float16)).to(of: aOut)
    .* (audioAttn1Modulations[5] + audioTimesteps[5])
  let reader: (PythonObject) -> Void = { state_dict in
    let scale_shift_table = state_dict["\(prefix).scale_shift_table"].to(torch.float).cpu().numpy()
    for i in 0..<9 {
      attn1Modulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: scale_shift_table[i..<(i + 1), ...])))
    }
    attn1Reader(state_dict)
    attn2Reader(state_dict)
    let audio_scale_shift_table = state_dict["\(prefix).audio_scale_shift_table"].to(torch.float)
      .cpu().numpy()
    for i in 0..<9 {
      audioAttn1Modulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: audio_scale_shift_table[i..<(i + 1), ...])))
    }
    let prompt_scale_shift_table = state_dict["\(prefix).prompt_scale_shift_table"].to(torch.float)
      .cpu()
      .numpy()
    for i in 0..<2 {
      promptScaleShiftModulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: prompt_scale_shift_table[i..<(i + 1), ...])))
    }
    let audio_prompt_scale_shift_table = state_dict["\(prefix).audio_prompt_scale_shift_table"].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<2 {
      audioPromptScaleShiftModulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: audio_prompt_scale_shift_table[i..<(i + 1), ...])))
    }
    audioAttn1Reader(state_dict)
    audioAttn2Reader(state_dict)
    let scale_shift_table_a2v_ca_audio = state_dict["\(prefix).scale_shift_table_a2v_ca_audio"].to(
      torch.float
    ).cpu().numpy()
    let scale_shift_table_a2v_ca_video = state_dict["\(prefix).scale_shift_table_a2v_ca_video"].to(
      torch.float
    ).cpu().numpy()
    // shift
    audioToVideoAttnModulations[0].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_video[1..<2, ...])))
    // scale
    audioToVideoAttnModulations[1].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_video[0..<1, ...])))
    // shift
    audioToVideoAttnModulations[2].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_audio[1..<2, ...])))
    // scale
    audioToVideoAttnModulations[3].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_audio[0..<1, ...])))
    // gate
    audioToVideoAttnModulations[4].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_video[4..<5, ...])))
    audioToVideoAttnReader(state_dict)
    // shift
    videoToAudioAttnModulations[0].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_video[3..<4, ...])))
    // scale
    videoToAudioAttnModulations[1].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_video[2..<3, ...])))
    // shift
    videoToAudioAttnModulations[2].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_audio[3..<4, ...])))
    // scale
    videoToAudioAttnModulations[3].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_audio[2..<3, ...])))
    // gate
    videoToAudioAttnModulations[4].weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(numpy: scale_shift_table_a2v_ca_audio[4..<5, ...])))
    videoToAudioAttnReader(state_dict)
    let ff_net_0_proj_weight = state_dict["\(prefix).ff.net.0.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_net_0_proj_weight)))
    let ff_net_0_proj_bias =
      state_dict["\(prefix).ff.net.0.proj.bias"].to(
        torch.float
      ).cpu().numpy()
    xLinear1.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_net_0_proj_bias)))
    let ff_net_2_weight =
      state_dict["\(prefix).ff.net.2.weight"].to(
        torch.float
      ).cpu().numpy()
    xOutProjection.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_net_2_weight)))
    let ff_net_2_bias =
      state_dict["\(prefix).ff.net.2.bias"].to(
        torch.float
      ).cpu().numpy()
    xOutProjection.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_net_2_bias)))
    let audio_ff_net_0_proj_weight = state_dict["\(prefix).audio_ff.net.0.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    audioLinear1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: audio_ff_net_0_proj_weight)))
    let audio_ff_net_0_proj_bias =
      state_dict["\(prefix).audio_ff.net.0.proj.bias"].to(
        torch.float
      ).cpu().numpy()
    audioLinear1.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: audio_ff_net_0_proj_bias)))
    let audio_ff_net_2_weight =
      state_dict["\(prefix).audio_ff.net.2.weight"].to(
        torch.float
      ).cpu().numpy()
    audioOutProjection.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: audio_ff_net_2_weight)))
    let audio_ff_net_2_bias =
      state_dict["\(prefix).audio_ff.net.2.bias"].to(
        torch.float
      ).cpu().numpy()
    audioOutProjection.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: audio_ff_net_2_bias)))
  }
  var inputs: [Input] = [vx, rot, cv, rotC, ax, rotA, ca, rotAC, rotCX]
  inputs.append(contentsOf: timesteps + audioTimesteps)
  inputs.append(contentsOf: promptTimesteps + audioPromptTimesteps)
  inputs.append(contentsOf: caScaleShiftTimesteps + [caGateTimesteps])
  inputs.append(contentsOf: audioCaScaleShiftTimesteps + [audioCaGateTimesteps])
  return (reader, Model(inputs, [out, aOut]))
}

func LTX2AdaLNSingle(
  prefix: String, channels: Int, count: Int, outputEmbedding: Bool, name: String, t: Input
) -> (
  (PythonObject) -> Void, Model.IO?, [Model.IO]
) {
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: name)
  let adaLNSingles = (0..<count).map { Dense(count: channels, name: "\(name)_adaln_single_\($0)") }
  var tOut = tEmbedder(t).reshaped([1, 1, channels])
  let tEmb: Model.IO?
  if outputEmbedding {
    tEmb = tOut.to(.Float32)
  } else {
    tEmb = nil
  }
  tOut = tOut.swish()
  let chunks = adaLNSingles.map { $0(tOut).to(.Float32) }
  let reader: (PythonObject) -> Void = { state_dict in
    let adaln_single_emb_timestep_embedder_linear_1_weight = state_dict[
      "\(prefix).emb.timestep_embedder.linear_1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    tMlp0.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: adaln_single_emb_timestep_embedder_linear_1_weight)))
    let adaln_single_emb_timestep_embedder_linear_1_bias = state_dict[
      "\(prefix).emb.timestep_embedder.linear_1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    tMlp0.bias.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: adaln_single_emb_timestep_embedder_linear_1_bias)))
    let adaln_single_emb_timestep_embedder_linear_2_weight = state_dict[
      "\(prefix).emb.timestep_embedder.linear_2.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    tMlp2.weight.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: adaln_single_emb_timestep_embedder_linear_2_weight)))
    let adaln_single_emb_timestep_embedder_linear_2_bias = state_dict[
      "\(prefix).emb.timestep_embedder.linear_2.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    tMlp2.bias.copy(
      from: Tensor<FloatType>(
        from: try! Tensor<Float>(numpy: adaln_single_emb_timestep_embedder_linear_2_bias)))
    let adaln_single_linear_weight = state_dict[
      "\(prefix).linear.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let adaln_single_linear_bias = state_dict[
      "\(prefix).linear.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<count {
      adaLNSingles[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: adaln_single_linear_weight[(channels * i)..<(channels * (i + 1)), ...])))
      adaLNSingles[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: adaln_single_linear_bias[(channels * i)..<(channels * (i + 1))])))
    }
  }
  return (reader, tEmb, chunks)
}

func LTX2(b: Int, h: Int, w: Int, useContextProjection: Bool = true) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let rot = Input()
  let rotC = Input()
  let rotA = Input()
  let rotAC = Input()
  let rotCX = Input()
  let xEmbedder = Dense(count: 4096, name: "x_embedder")
  var out = xEmbedder(x).to(.Float32)
  let txt = Input()
  let contextMlp0: Model?
  let contextMlp2: Model?
  let txtOut: Model.IO
  if useContextProjection {
    let (c0, c2, contextEmbedder) = GELUMLPEmbedder(channels: 4096, name: "context")
    contextMlp0 = c0
    contextMlp2 = c2
    txtOut = contextEmbedder(txt)
  } else {
    contextMlp0 = nil
    contextMlp2 = nil
    txtOut = txt.to(.Float16)
  }
  let a = Input()
  let aEmbedder = Dense(count: 2048, name: "a_embedder")
  var aOut = aEmbedder(a).to(.Float32)
  let aTxt = Input()
  let aContextMlp0: Model?
  let aContextMlp2: Model?
  let aTxtOut: Model.IO
  if useContextProjection {
    let (ac0, ac2, aContextEmbedder) = GELUMLPEmbedder(channels: 2048, name: "a_context")
    aContextMlp0 = ac0
    aContextMlp2 = ac2
    aTxtOut = aContextEmbedder(aTxt)
  } else {
    aContextMlp0 = nil
    aContextMlp2 = nil
    aTxtOut = aTxt.to(.Float16)
  }
  let t = Input()
  let p = Input()
  let (txReader, txEmb, txEmbChunks) = LTX2AdaLNSingle(
    prefix: "adaln_single", channels: 4096, count: 9, outputEmbedding: true, name: "tx", t: t)
  let (taReader, taEmb, taEmbChunks) = LTX2AdaLNSingle(
    prefix: "audio_adaln_single", channels: 2048, count: 9, outputEmbedding: true, name: "ta", t: t)
  let (ptxReader, _, ptxEmbChunks) = LTX2AdaLNSingle(
    prefix: "prompt_adaln_single", channels: 4096, count: 2, outputEmbedding: false, name: "ptx",
    t: p)
  let (ptaReader, _, ptaEmbChunks) = LTX2AdaLNSingle(
    prefix: "audio_prompt_adaln_single", channels: 2048, count: 2, outputEmbedding: false,
    name: "pta", t: p)
  let (caReader, _, tcxEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_video_scale_shift_adaln_single", channels: 4096, count: 4,
    outputEmbedding: false, name: "tcx", t: t)
  let (audioCaReader, _, tcaEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_audio_scale_shift_adaln_single", channels: 2048, count: 4,
    outputEmbedding: false, name: "tca", t: t)
  let (gateReader, _, a2vEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_a2v_gate_adaln_single", channels: 4096, count: 1, outputEmbedding: false,
    name: "a2v", t: t)
  let (audioGateReader, _, v2aEmbChunks) = LTX2AdaLNSingle(
    prefix: "av_ca_v2a_gate_adaln_single", channels: 2048, count: 1, outputEmbedding: false,
    name: "v2a", t: t)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<48 {
    let (reader, block) = LTX2TransformerBlock(
      prefix: "transformer_blocks.\(i)", k: 128, h: 32, b: 1, t: 1024, hw: 6144, a: 121,
      intermediateSize: 0)
    let blockOut = block(
      out, rot, txtOut, rotC, aOut, rotA, aTxtOut, rotAC, rotCX,
      txEmbChunks[0], txEmbChunks[1], txEmbChunks[2], txEmbChunks[3], txEmbChunks[4],
      txEmbChunks[5], txEmbChunks[6], txEmbChunks[7], txEmbChunks[8],
      taEmbChunks[0], taEmbChunks[1], taEmbChunks[2], taEmbChunks[3], taEmbChunks[4],
      taEmbChunks[5], taEmbChunks[6], taEmbChunks[7], taEmbChunks[8],
      ptxEmbChunks[0], ptxEmbChunks[1], ptaEmbChunks[0], ptaEmbChunks[1],
      tcxEmbChunks[1], tcxEmbChunks[0], tcaEmbChunks[1], tcaEmbChunks[0], a2vEmbChunks[0],
      tcxEmbChunks[3], tcxEmbChunks[2], tcaEmbChunks[3], tcaEmbChunks[2], v2aEmbChunks[0])
    readers.append(reader)
    out = blockOut[0]
    aOut = blockOut[1]
  }
  let scaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(1), .HWC(1, 1, 4096), name: "norm_out_ada_ln_\($0)")
  }
  let normOut = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if let txEmb = txEmb {
    out = normOut(out) .* (1 + (scaleShiftModulations[1] + txEmb))
      + (scaleShiftModulations[0] + txEmb)
  }
  let projOut = Dense(count: 128, name: "proj_out")
  out = projOut(out.to(.Float16))
  let audioScaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(1), .HWC(1, 1, 2048), name: "audio_norm_out_ada_ln_\($0)")
  }
  if let taEmb = taEmb {
    aOut = normOut(aOut) .* (1 + (audioScaleShiftModulations[1] + taEmb))
      + (audioScaleShiftModulations[0] + taEmb)
  }
  let audioProjOut = Dense(count: 128, name: "audio_proj_out")
  aOut = audioProjOut(aOut.to(.Float16))
  let reader: (PythonObject) -> Void = { state_dict in
    let patchify_proj_weight = state_dict["patchify_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    xEmbedder.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: patchify_proj_weight)))
    xEmbedder.weight.to(.unifiedMemory)
    let patchify_proj_bias = state_dict["patchify_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    xEmbedder.bias.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: patchify_proj_bias)))
    if useContextProjection,
      let contextMlp0 = contextMlp0,
      let contextMlp2 = contextMlp2,
      Bool(state_dict.__contains__("caption_projection.linear_1.weight")) ?? false
    {
      let caption_projection_linear_1_weight = state_dict["caption_projection.linear_1.weight"].to(
        torch.float
      ).cpu().numpy()
      contextMlp0.weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: caption_projection_linear_1_weight)))
      let caption_projection_linear_1_bias = state_dict["caption_projection.linear_1.bias"].to(
        torch.float
      ).cpu().numpy()
      contextMlp0.bias.copy(
        from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: caption_projection_linear_1_bias)))
      let caption_projection_linear_2_weight = state_dict["caption_projection.linear_2.weight"].to(
        torch.float
      ).cpu().numpy()
      contextMlp2.weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: caption_projection_linear_2_weight)))
      let caption_projection_linear_2_bias = state_dict["caption_projection.linear_2.bias"].to(
        torch.float
      ).cpu().numpy()
      contextMlp2.bias.copy(
        from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: caption_projection_linear_2_bias)))
    }
    let audio_patchify_proj_weight = state_dict["audio_patchify_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    aEmbedder.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: audio_patchify_proj_weight)))
    aEmbedder.weight.to(.unifiedMemory)
    let audio_patchify_proj_bias = state_dict["audio_patchify_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    aEmbedder.bias.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: audio_patchify_proj_bias)))
    if useContextProjection,
      let aContextMlp0 = aContextMlp0,
      let aContextMlp2 = aContextMlp2,
      Bool(state_dict.__contains__("audio_caption_projection.linear_1.weight")) ?? false
    {
      let audio_caption_projection_linear_1_weight = state_dict[
        "audio_caption_projection.linear_1.weight"
      ].to(
        torch.float
      ).cpu().numpy()
      aContextMlp0.weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: audio_caption_projection_linear_1_weight)))
      let audio_caption_projection_linear_1_bias = state_dict[
        "audio_caption_projection.linear_1.bias"
      ].to(
        torch.float
      ).cpu().numpy()
      aContextMlp0.bias.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: audio_caption_projection_linear_1_bias)))
      let audio_caption_projection_linear_2_weight = state_dict[
        "audio_caption_projection.linear_2.weight"
      ].to(
        torch.float
      ).cpu().numpy()
      aContextMlp2.weight.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: audio_caption_projection_linear_2_weight)))
      let audio_caption_projection_linear_2_bias = state_dict[
        "audio_caption_projection.linear_2.bias"
      ].to(
        torch.float
      ).cpu().numpy()
      aContextMlp2.bias.copy(
        from: Tensor<FloatType>(
          from: try! Tensor<Float>(numpy: audio_caption_projection_linear_2_bias)))
    }
    txReader(state_dict)
    taReader(state_dict)
    ptxReader(state_dict)
    ptaReader(state_dict)
    caReader(state_dict)
    audioCaReader(state_dict)
    gateReader(state_dict)
    audioGateReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let scale_shift_table = state_dict["scale_shift_table"].to(torch.float).cpu().numpy()
    for i in 0..<2 {
      scaleShiftModulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: scale_shift_table[i..<(i + 1), ...])))
    }
    let audio_scale_shift_table = state_dict["audio_scale_shift_table"].to(torch.float).cpu()
      .numpy()
    for i in 0..<2 {
      audioScaleShiftModulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: audio_scale_shift_table[i..<(i + 1), ...])))
    }
    let proj_out_weight = state_dict["proj_out.weight"].to(torch.float).cpu().numpy()
    projOut.weight.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: proj_out_weight)))
    let proj_out_bias = state_dict["proj_out.bias"].to(torch.float).cpu().numpy()
    projOut.bias.copy(from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: proj_out_bias)))
    let audio_proj_out_weight = state_dict["audio_proj_out.weight"].to(torch.float).cpu().numpy()
    audioProjOut.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: audio_proj_out_weight)))
    let audio_proj_out_bias = state_dict["audio_proj_out.bias"].to(torch.float).cpu().numpy()
    audioProjOut.bias.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: audio_proj_out_bias)))
  }
  return (reader, Model([x, txt, a, aTxt, t, p, rot, rotC, rotA, rotAC, rotCX], [out, aOut]))
}

// Legacy ltx2.0 one-off DiT export run kept disabled below.

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timesteps
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

/*
graph.withNoGrad {
  let xTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: video.latent.to(torch.float).cpu().numpy()))
      .toGPU(2)
  ).reshaped(.HWC(1, 6144, 128))
  let txtTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: video.context.to(torch.float).cpu().numpy()))
      .toGPU(2)
  ).reshaped(.HWC(1, 1024, 3840))
  let aTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: audio.latent.to(torch.float).cpu().numpy()))
      .toGPU(2)
  ).reshaped(.HWC(1, 121, 128))
  let aTxtTensor = graph.variable(
    Tensor<FloatType>(from: try! Tensor<Float>(numpy: audio.context.to(torch.float).cpu().numpy()))
      .toGPU(2)
  ).reshaped(.HWC(1, 1024, 3840))
  let timestepTensor = graph.variable(
    Tensor<Float16>(
      from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(2))
  let promptTimestepTensor = graph.variable(
    Tensor<Float16>(
      from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(2))
  /*
  let rotTensor = graph.variable(.CPU, .HWC(1, 6144, 4096), of: Float.self)
  for i in 0..<16 { // frame
    let frames: Double = (Double(max(0, i * 8 - 7)) + Double(i * 8 + 1)) / 50
    let fib = BFloat16(frames)
    let fi: Double = Double(fib.floatValue) / 20
    for y in 0..<16 { // height
      let fy: Double = (Double(y) + 0.5) / 64
      for x in 0..<24 { // width
        let idx = i * 16 * 24 + y * 24 + x
        rotTensor[0, idx, 0] = 1
        rotTensor[0, idx, 1] = 0
        rotTensor[0, idx, 2] = 1
        rotTensor[0, idx, 3] = 0
        for j in 0..<682 {
          let theta: Double = pow(10_000, Double(j) / 681) * .pi * 0.5
          let fx: Double = (Double(x) + 0.5) / 64
          let cosfi = cos(theta * (fi * 2 - 1))
          let sinfi = sin(theta * (fi * 2 - 1))
          rotTensor[0, idx, j * 6 + 4] = 1 // Float(cosfi)
          rotTensor[0, idx, j * 6 + 1 + 4] = 0 // Float(sinfi)
          let cosfy = cos(theta * (fy * 2 - 1))
          let sinfy = sin(theta * (fy * 2 - 1))
          rotTensor[0, idx, j * 6 + 2 + 4] = 1 // Float(cosfy)
          rotTensor[0, idx, j * 6 + 3 + 4] = 0 // Float(sinfy)
          let cosfx = cos(theta * (fx * 2 - 1))
          let sinfx = sin(theta * (fx * 2 - 1))
          rotTensor[0, idx, j * 6 + 4 + 4] = 1 // Float(cosfx)
          rotTensor[0, idx, j * 6 + 5 + 4] = 0 // Float(sinfx)
        }
      }
    }
  }
  let rotTensorGPU = DynamicGraph.Tensor<FloatType>(from: rotTensor.reshaped(.NHWC(1, 6144, 32, 128))).toGPU(2)
  */
  let rotTensor = graph.variable(.CPU, .HWC(1, 1, 4096), of: Float.self)
  for i in 0..<2048 {
    rotTensor[0, 0, i * 2] = 1
    rotTensor[0, 0, i * 2 + 1] = 0
  }
  let rot1TensorGPU = DynamicGraph.Tensor<FloatType>(from: rotTensor.reshaped(.NHWC(1, 1, 32, 128)))
    .toGPU(2)
  let rot2TensorGPU = DynamicGraph.Tensor<FloatType>(from: rotTensor.reshaped(.NHWC(1, 1, 32, 128)))
    .toGPU(2)
  let aRotTensor = graph.variable(.CPU, .HWC(1, 1, 2048), of: Float.self)
  for i in 0..<1024 {
    aRotTensor[0, 0, i * 2] = 1
    aRotTensor[0, 0, i * 2 + 1] = 0
  }
  let rot3TensorGPU = DynamicGraph.Tensor<FloatType>(from: aRotTensor.reshaped(.NHWC(1, 1, 32, 64)))
    .toGPU(2)
  let rot4TensorGPU = DynamicGraph.Tensor<FloatType>(from: aRotTensor.reshaped(.NHWC(1, 1, 32, 64)))
    .toGPU(2)
  let rot5TensorGPU = DynamicGraph.Tensor<FloatType>(from: aRotTensor.reshaped(.NHWC(1, 1, 32, 64)))
    .toGPU(2)
  dit.maxConcurrency = .limit(1)
  dit.compile(
    inputs: xTensor, txtTensor, aTensor, aTxtTensor, timestepTensor, promptTimestepTensor,
    rot1TensorGPU, rot2TensorGPU, rot3TensorGPU, rot4TensorGPU, rot5TensorGPU)
  reader(state_dict)
  debugPrint(
    dit(
      inputs: xTensor, txtTensor, aTensor, aTxtTensor, timestepTensor, promptTimestepTensor,
      rot1TensorGPU, rot2TensorGPU, rot3TensorGPU, rot4TensorGPU, rot5TensorGPU))
  graph.openStore("/home/liu/workspace/swift-diffusion/ltx_2_19b_dev_f16.ckpt") {
    $0.write("dit", model: dit)
  }
}
*/
