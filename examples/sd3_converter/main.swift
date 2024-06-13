import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit
import SentencePiece

typealias FloatType = Float

let torch = Python.import("torch")
let nodes = Python.import("nodes")
torch.set_grad_enabled(false)

nodes.init_custom_nodes()
let triplecliploader = nodes.NODE_CLASS_MAPPINGS["TripleCLIPLoader"]()
let triplecliploader_11 = triplecliploader.load_clip(
  clip_name1: "clip_g.safetensors",
  clip_name2: "clip_l.safetensors",
  clip_name3: "t5xxl_fp16.safetensors"
)

let cliptextencode = nodes.CLIPTextEncode()
let cliptextencode_71 = cliptextencode.encode(
  text: "", clip: triplecliploader_11[0]
)

let emptysd3latentimage = nodes.NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
let emptysd3latentimage_135 = emptysd3latentimage.generate(
  width: 1024, height: 1024, batch_size: 1
)

let checkpointloadersimple = nodes.CheckpointLoaderSimple()
let checkpointloadersimple_252 = checkpointloadersimple.load_checkpoint(
  ckpt_name: "sdv3/2b_1024/sd3_medium.safetensors"
)

let cliptextencodesd3 = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncodeSD3"]()
let cliptextencodesd3_273 = cliptextencodesd3.encode(
  clip_l:
    "photo of a young woman with long, wavy brown hair lying down in grass, top down shot, summer, warm, laughing, joy, fun",
  clip_g:
    "photo of a young woman with long, wavy brown hair lying down in grass, top down shot, summer, warm, laughing, joy, fun",
  t5xxl:
    "photo of a young woman with long, wavy brown hair lying down in grass, top down shot, summer, warm, laughing, joy, fun",
  empty_padding: "none",
  clip: triplecliploader_11[0]
)

let modelsamplingsd3 = nodes.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
let conditioningzeroout = nodes.ConditioningZeroOut()
let conditioningsettimesteprange = nodes.ConditioningSetTimestepRange()
let conditioningcombine = nodes.ConditioningCombine()
let ksampler = nodes.KSampler()
let vaedecode = nodes.VAEDecode()

let modelsamplingsd3_13 = modelsamplingsd3.patch(
  shift: 3, model: checkpointloadersimple_252[0]
)

let conditioningzeroout_67 = conditioningzeroout.zero_out(
  conditioning: cliptextencode_71[0]
)

let conditioningsettimesteprange_68 = conditioningsettimesteprange.set_range(
  start: 0.1,
  end: 1,
  conditioning: conditioningzeroout_67[0]
)

let conditioningsettimesteprange_70 = conditioningsettimesteprange.set_range(
  start: 0, end: 0.1, conditioning: cliptextencode_71[0]
)

let conditioningcombine_69 = conditioningcombine.combine(
  conditioning_1: conditioningsettimesteprange_68[0],
  conditioning_2: conditioningsettimesteprange_70[0]
)

let ksampler_271 = ksampler.sample(
  seed: 23,
  steps: 28,
  cfg: 4.5,
  sampler_name: "dpmpp_2m",
  scheduler: "sgm_uniform",
  denoise: 1,
  model: modelsamplingsd3_13[0],
  positive: cliptextencodesd3_273[0],
  negative: conditioningcombine_69[0],
  latent_image: emptysd3latentimage_135[0]
)

let vaedecode_231 = vaedecode.decode(
  samples: ksampler_271[0],
  vae: checkpointloadersimple_252[2]
)

print(vaedecode_231[0])

let vae_state_dict = checkpointloadersimple_252[2].first_stage_model.state_dict()

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
    let norm1_weight = state_dict["\(prefix).norm1.weight"].to(torch.float).cpu().numpy()
    let norm1_bias = state_dict["\(prefix).norm1.bias"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let conv1_weight = state_dict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2_bias = state_dict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).nin_shortcut.weight"].to(torch.float).cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).nin_shortcut.bias"].to(torch.float).cpu()
        .numpy()
      ninShortcut.weight.copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
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
    let norm_weight = state_dict["\(prefix).norm.weight"].to(torch.float).cpu().numpy()
    let norm_bias = state_dict["\(prefix).norm.bias"].to(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: norm_bias))
    let k_weight = state_dict["\(prefix).k.weight"].to(torch.float).cpu().numpy()
    let k_bias = state_dict["\(prefix).k.bias"].to(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["\(prefix).q.weight"].to(torch.float).cpu().numpy()
    let q_bias = state_dict["\(prefix).q.bias"].to(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["\(prefix).v.weight"].to(torch.float).cpu().numpy()
    let v_bias = state_dict["\(prefix).v.bias"].to(torch.float).cpu().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["\(prefix).proj_out.weight"].to(torch.float).cpu().numpy()
    let proj_out_bias = state_dict["\(prefix).proj_out.bias"].to(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
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
        prefix: "encoder.down.\(i).block.\(j)", outChannels: channel,
        shortcut: previousChannel != channel)
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
        let conv_weight = state_dict["encoder.down.\(downLayer).downsample.conv.weight"].to(
          torch.float
        ).cpu()
          .numpy()
        let conv_bias = state_dict["encoder.down.\(downLayer).downsample.conv.bias"].to(torch.float)
          .cpu().numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "encoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["encoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["encoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    let norm_out_weight = state_dict["encoder.norm_out.weight"].to(torch.float).cpu().numpy()
    let norm_out_bias = state_dict["encoder.norm_out.bias"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["encoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["encoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

func Decoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "decoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "decoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  var readers = [(PythonObject) -> Void]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", outChannels: channel,
        shortcut: previousChannel != channel)
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
        let conv_weight = state_dict["decoder.up.\(upLayer).upsample.conv.weight"].to(torch.float)
          .cpu().numpy()
        let conv_bias = state_dict["decoder.up.\(upLayer).upsample.conv.bias"].to(torch.float).cpu()
          .numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
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
    let conv_in_weight = state_dict["decoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["decoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["decoder.norm_out.weight"].to(torch.float).cpu().numpy()
    let norm_out_bias = state_dict["decoder.norm_out.bias"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["decoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["decoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

let z = ksampler_271[0]["samples"].to(torch.float).cpu()
let graph = DynamicGraph()
graph.withNoGrad {
  let zTensor = graph.variable(try! Tensor<Float>(numpy: z.numpy())).toGPU(0)
  // Already processed out.
  // zTensor = (1.0 / 1.5305) * zTensor + 0.0609
  debugPrint(zTensor)
  let (decoderReader, decoder) = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  decoder.compile(inputs: zTensor)
  decoderReader(vae_state_dict)
  let image = decoder(inputs: zTensor)[0].as(of: Float.self)
  let decodedImage = (image.permuted(0, 2, 3, 1) + 1) * 0.5
  debugPrint(decodedImage)
  let (encoderReader, encoder) = Encoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 128, startHeight: 128)
  encoder.compile(inputs: image)
  encoderReader(vae_state_dict)
  let _ = encoder(inputs: image)[0].as(of: Float.self)
  graph.openStore("/home/liu/workspace/swift-diffusion/sd3_vae_f32.ckpt") {
    $0.write("encoder", model: encoder)
    $0.write("decoder", model: decoder)
  }
}

func CLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    FloatType.self, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 {
      penultimate = out
    }
    let encoderLayer =
      CLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [penultimate!, out])
}

func OpenCLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    FloatType.self, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 {
      penultimate = out
    }
    let encoderLayer =
      OpenCLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [penultimate!, out])
}

func T5TextEmbedding(vocabularySize: Int, embeddingSize: Int, name: String) -> Model {
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: name)
  return tokenEmbed
}

func T5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let positionBias = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  // No scaling the queries.
  let queries = toqueries(x).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + positionBias
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: outFeatures, noBias: true, name: "o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, positionBias], [out]))
}

func T5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true, name: "w0")
  let wi_1 = Dense(count: intermediateSize, noBias: true, name: "w1")
  var out = wi_1(x) .* wi_0(x).GELU(approximate: .tanh)
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  out = wo(out)
  return (wi_0, wi_1, wo, Model([x], [out]))
}

func T5Block(
  prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let positionBias = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (tokeys, toqueries, tovalues, unifyheads, attention) = T5LayerSelfAttention(
    k: k, h: h, b: b, t: t, outFeatures: outFeatures)
  var out = x + attention(norm1(x), positionBias)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (wi_0, wi_1, wo, ff) = T5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize)
  out = out + ff(norm2(out))
  let reader: (PythonObject) -> Void = { state_dict in
    let layer_0_layer_norm_weight = state_dict["\(prefix).layer.0.layer_norm.weight"].cpu().float()
      .numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: layer_0_layer_norm_weight))
    let k_weight = state_dict["\(prefix).layer.0.SelfAttention.k.weight"].cpu().float().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    let q_weight = state_dict["\(prefix).layer.0.SelfAttention.q.weight"].cpu().float().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    let v_weight = state_dict["\(prefix).layer.0.SelfAttention.v.weight"].cpu().float().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    let o_weight = state_dict["\(prefix).layer.0.SelfAttention.o.weight"].cpu().float().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: o_weight))
    let layer_1_layer_norm_weight = state_dict["\(prefix).layer.1.layer_norm.weight"].cpu().float()
      .numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: layer_1_layer_norm_weight))
    let wi_0_weight = state_dict["\(prefix).layer.1.DenseReluDense.wi_0.weight"].cpu().float()
      .numpy()
    wi_0.weight.copy(from: try! Tensor<Float>(numpy: wi_0_weight))
    let wi_1_weight = state_dict["\(prefix).layer.1.DenseReluDense.wi_1.weight"].cpu().float()
      .numpy()
    wi_1.weight.copy(from: try! Tensor<Float>(numpy: wi_1_weight))
    let wo_weight = state_dict["\(prefix).layer.1.DenseReluDense.wo.weight"].cpu().float().numpy()
    wo.weight.copy(from: try! Tensor<Float>(numpy: wo_weight))
  }
  return (reader, Model([x, positionBias], [out]))
}

func T5ForConditionalGeneration(b: Int, t: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let relativePositionBuckets = Input()
  let textEmbed = T5TextEmbedding(vocabularySize: 32_128, embeddingSize: 4_096, name: "shared")
  var out = textEmbed(x)
  let relativePositionEmbedding = Embedding(
    Float.self, vocabularySize: 32, embeddingSize: 64, name: "relative_position_embedding")
  let positionBias = relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, 64])
    .permuted(0, 3, 1, 2)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = T5Block(
      prefix: "encoder.block.\(i)", k: 64, h: 64, b: b, t: t, outFeatures: 4_096,
      intermediateSize: 10_240)
    out = block(out, positionBias)
    readers.append(reader)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["shared.weight"].cpu().float().numpy()
    textEmbed.weight.copy(from: try! Tensor<Float>(numpy: vocab))
    let relative_attention_bias_weight = state_dict[
      "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ].cpu().float().numpy()
    relativePositionEmbedding.weight.copy(
      from: try! Tensor<Float>(numpy: relative_attention_bias_weight))
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_norm_weight = state_dict["encoder.final_layer_norm.weight"].cpu().float()
      .numpy()
    finalNorm.weight.copy(from: try! Tensor<Float>(numpy: final_layer_norm_weight))
  }
  return (reader, Model([x, relativePositionBuckets], [out]))
}

let tokenizer0 = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let tokenizer1 = CLIPTokenizer(
  vocabulary: "examples/open_clip/vocab_16e6.json",
  merges: "examples/open_clip/bpe_simple_vocab_16e6.txt")

let prompt =
  "photo of a young woman with long, wavy brown hair lying down in grass, top down shot, summer, warm, laughing, joy, fun"
let negativePrompt = ""

let tokens0 = tokenizer0.tokenize(text: prompt, truncation: true, maxLength: 77)
let tokens1 = tokenizer1.tokenize(text: prompt, truncation: true, maxLength: 77, paddingToken: 0)
let unconditionalTokens0 = tokenizer0.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77)
let unconditionalTokens1 = tokenizer1.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77, paddingToken: 0)
let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/swift-diffusion/examples/sd3/spiece.model")
var tokens2 = sentencePiece.encode(prompt).map { return $0.id }
tokens2.append(1)

let tokensTensor0 = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensTensor1 = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensTensor2 = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
for i in 0..<77 {
  tokensTensor0[i] = tokens0[i]
  tokensTensor1[i] = tokens1[i]
  tokensTensor2[i] = i < tokens2.count ? tokens2[i] : 0
  positionTensor[i] = Int32(i)
}

func relativePositionBuckets(sequenceLength: Int, numBuckets: Int, maxDistance: Int) -> Tensor<
  Int32
> {
  // isBidirectional = true.
  let numBuckets = numBuckets / 2
  let maxExact = numBuckets / 2
  var relativePositionBuckets = Tensor<Int32>(.CPU, .C(sequenceLength * sequenceLength))
  for i in 0..<sequenceLength {
    for j in 0..<sequenceLength {
      var relativePositionBucket = j > i ? numBuckets : 0
      let relativePosition = abs(i - j)
      let isSmall = relativePosition < maxExact
      if isSmall {
        relativePositionBucket += relativePosition
      } else {
        let relativePositionIfLarge = min(
          numBuckets - 1,
          maxExact
            + Int(
              (log(Double(relativePosition) / Double(maxExact))
                / log(Double(maxDistance) / Double(maxExact)) * Double(numBuckets - maxExact))
                .rounded(.down)))
        relativePositionBucket += relativePositionIfLarge
      }
      relativePositionBuckets[i * sequenceLength + j] = Int32(relativePositionBucket)
    }
  }
  return relativePositionBuckets
}

let casualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
  }
}

let (c0, c0Pooled) = graph.withNoGrad {
  let tokensTensorGPU = tokensTensor0.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  let textModel0 = CLIPTextModel(
    vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
    batchSize: 1, intermediateSize: 3072)
  textModel0.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/clip_vit_l14_f32.ckpt") {
    $0.read("text_model", model: textModel0)
  }
  let c = textModel0(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU).map {
    $0.as(of: FloatType.self)
  }
  var pooled = graph.variable(.GPU(0), .NC(1, 768), of: FloatType.self)
  let c1 = c[0].reshaped(.CHW(1, 77, 768))
  for (i, token) in tokens0.enumerated() {
    if token == tokenizer0.endToken {
      pooled[0..<1, 0..<768] = c[1][i..<(i + 1), 0..<768]
      break
    }
  }
  return (c1, pooled)
}

let (c1, c1Pooled) = graph.withNoGrad {
  let tokensTensorGPU = tokensTensor1.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  let textModel1 = OpenCLIPTextModel(
    vocabularySize: 49408, maxLength: 77, embeddingSize: 1280, numLayers: 32, numHeads: 20,
    batchSize: 1, intermediateSize: 5120)
  let textProjection = graph.variable(.GPU(0), .NC(1280, 1280), of: FloatType.self)
  textModel1.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_vit_bigg14_f16.ckpt") {
    $0.read("text_model", model: textModel1)
    $0.read("text_projection", variable: textProjection)
  }
  let c = textModel1(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU).map {
    $0.as(of: FloatType.self)
  }
  var pooled = graph.variable(.GPU(0), .NC(1, 1280), of: FloatType.self)
  let c1 = c[0].reshaped(.CHW(1, 77, 1280))
  for (i, token) in tokens1.enumerated() {
    if token == tokenizer1.endToken {
      pooled[0..<1, 0..<1280] = c[1][i..<(i + 1), 0..<1280] * textProjection
      break
    }
  }
  return (c1, pooled)
}

let c2 = graph.withNoGrad {
  let (_, textModel) = T5ForConditionalGeneration(b: 1, t: 77)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: 77, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor2.toGPU(0)
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, relativePositionBucketsGPU)
  graph.openStore("/home/liu/workspace/swift-llm/t5_xxl_encoder_f32.ckpt") {
    $0.read("text_model", model: textModel)
  }
  let output = textModel(inputs: tokensTensorGPU, relativePositionBucketsGPU)[0].as(of: Float.self)
  return output
}

let (c, pooled) = graph.withNoGrad {
  var pooled = graph.variable(.GPU(0), .NC(1, 2048), of: FloatType.self)
  pooled[0..<1, 0..<768] = c0Pooled
  pooled[0..<1, 768..<2048] = c1Pooled
  var c = graph.variable(.GPU(0), .CHW(1, 154, 4096), of: FloatType.self)
  c.full(0)
  c[0..<1, 0..<77, 0..<768] = c0
  c[0..<1, 0..<77, 768..<2048] = c1
  c[0..<1, 77..<154, 0..<4096] = c2
  return (c, pooled)
}

debugPrint(c)
debugPrint(pooled)

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([2, 16, 128, 128]).to(torch.float16).cuda()
let y = torch.randn([2, 2048]).to(torch.float16).cuda() * 0.01
let t = torch.full([2], 1000).cuda()
let ctx = torch.randn([2, 154, 4096]).to(torch.float16).cuda() * 0.01

let diffusion_model = modelsamplingsd3_13[0].model.diffusion_model.cuda()

let out = diffusion_model(x: x, timesteps: t, context: ctx, y: y)
print(out)

let state_dict = modelsamplingsd3_13[0].model.state_dict()

func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func VectorEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "y_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "y_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLP(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  let contextK = contextToKeys(contextOut)
  let contextQ = contextToQueries(contextOut)
  let contextV = contextToValues(contextOut)
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  let xK = xToKeys(xOut)
  let xQ = xToQueries(xOut)
  let xV = xToValues(xOut)
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  keys = keys.reshaped([b, t + hw, h, k]).permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
    .permuted(0, 2, 1, 3)
  values = values.reshaped([b, t + hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = Dense(count: k * h)
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h)
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = x + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextFc1: Model?
  let contextFc2: Model?
  if !contextBlockPreOnly {
    let contextMlp: Model
    (contextFc1, contextFc2, contextMlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4)
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut = contextOut + contextChunks[5]
      .* contextMlp(contextNorm2(contextOut) .* (1 + contextChunks[4]) + contextChunks[3])
  } else {
    contextFc1 = nil
    contextFc2 = nil
  }
  let (xFc1, xFc2, xMlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut = xOut + xChunks[5] .* xMlp(xNorm2(xOut) .* (1 + xChunks[4]) + xChunks[3])
  let reader: (PythonObject) -> Void = { state_dict in
    let context_block_attn_qkv_weight = state_dict["\(prefix).context_block.attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    let context_block_attn_qkv_bias = state_dict["\(prefix).context_block.attn.qkv.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_weight[..<(k * h), ...]))
    contextToQueries.bias.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_bias[..<(k * h)]))
    contextToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_weight[(k * h)..<(2 * k * h), ...]))
    contextToKeys.bias.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_bias[(k * h)..<(2 * k * h)]))
    contextToValues.weight.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...])
    )
    contextToValues.bias.copy(
      from: try! Tensor<Float>(numpy: context_block_attn_qkv_bias[(2 * k * h)..<(3 * k * h)]))
    let x_block_attn_qkv_weight = state_dict["\(prefix).x_block.attn.qkv.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_block_attn_qkv_bias = state_dict["\(prefix).x_block.attn.qkv.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_qkv_weight[..<(k * h), ...]))
    xToQueries.bias.copy(from: try! Tensor<Float>(numpy: x_block_attn_qkv_bias[..<(k * h)]))
    xToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_qkv_weight[(k * h)..<(2 * k * h), ...]))
    xToKeys.bias.copy(from: try! Tensor<Float>(numpy: x_block_attn_qkv_bias[(k * h)..<(2 * k * h)]))
    xToValues.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_qkv_weight[(2 * k * h)..<(3 * k * h), ...]))
    xToValues.bias.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_qkv_bias[(2 * k * h)..<(3 * k * h)]))
    if let contextUnifyheads = contextUnifyheads {
      let context_block_attn_proj_weight = state_dict["\(prefix).context_block.attn.proj.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      let context_block_attn_proj_bias = state_dict["\(prefix).context_block.attn.proj.bias"].to(
        torch.float
      ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: try! Tensor<Float>(numpy: context_block_attn_proj_weight))
      contextUnifyheads.bias.copy(
        from: try! Tensor<Float>(numpy: context_block_attn_proj_bias))
    }
    let x_block_attn_proj_weight = state_dict["\(prefix).x_block.attn.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_block_attn_proj_bias = state_dict["\(prefix).x_block.attn.proj.bias"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_proj_weight))
    xUnifyheads.bias.copy(
      from: try! Tensor<Float>(numpy: x_block_attn_proj_bias))
    if let contextFc1 = contextFc1, let contextFc2 = contextFc2 {
      let context_block_mlp_fc1_weight = state_dict["\(prefix).context_block.mlp.fc1.weight"].to(
        torch.float
      ).cpu().numpy()
      let context_block_mlp_fc1_bias = state_dict["\(prefix).context_block.mlp.fc1.bias"].to(
        torch.float
      ).cpu().numpy()
      contextFc1.weight.copy(
        from: try! Tensor<Float>(numpy: context_block_mlp_fc1_weight))
      contextFc1.bias.copy(
        from: try! Tensor<Float>(numpy: context_block_mlp_fc1_bias))
      let context_block_mlp_fc2_weight = state_dict["\(prefix).context_block.mlp.fc2.weight"].to(
        torch.float
      ).cpu().numpy()
      let context_block_mlp_fc2_bias = state_dict["\(prefix).context_block.mlp.fc2.bias"].to(
        torch.float
      ).cpu().numpy()
      contextFc2.weight.copy(
        from: try! Tensor<Float>(numpy: context_block_mlp_fc2_weight))
      contextFc2.bias.copy(
        from: try! Tensor<Float>(numpy: context_block_mlp_fc2_bias))
    }
    let x_block_mlp_fc1_weight = state_dict["\(prefix).x_block.mlp.fc1.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_block_mlp_fc1_bias = state_dict["\(prefix).x_block.mlp.fc1.bias"].to(
      torch.float
    ).cpu().numpy()
    xFc1.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_mlp_fc1_weight))
    xFc1.bias.copy(
      from: try! Tensor<Float>(numpy: x_block_mlp_fc1_bias))
    let x_block_mlp_fc2_weight = state_dict["\(prefix).x_block.mlp.fc2.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_block_mlp_fc2_bias = state_dict["\(prefix).x_block.mlp.fc2.bias"].to(
      torch.float
    ).cpu().numpy()
    xFc2.weight.copy(
      from: try! Tensor<Float>(numpy: x_block_mlp_fc2_weight))
    xFc2.bias.copy(
      from: try! Tensor<Float>(numpy: x_block_mlp_fc2_bias))
    let context_block_adaln_modulation_weight = state_dict[
      "\(prefix).context_block.adaLN_modulation.1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let context_block_adaln_modulation_bias = state_dict[
      "\(prefix).context_block.adaLN_modulation.1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<(contextBlockPreOnly ? 2 : 6) {
      contextAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: context_block_adaln_modulation_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
      contextAdaLNs[i].bias.copy(
        from: try! Tensor<Float>(
          numpy: context_block_adaln_modulation_bias[(k * h * i)..<(k * h * (i + 1))]))
    }
    let x_block_adaln_modulation_weight = state_dict["\(prefix).x_block.adaLN_modulation.1.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let x_block_adaln_modulation_bias = state_dict["\(prefix).x_block.adaLN_modulation.1.bias"].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: x_block_adaln_modulation_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
      xAdaLNs[i].bias.copy(
        from: try! Tensor<Float>(
          numpy: x_block_adaln_modulation_bias[(k * h * i)..<(k * h * (i + 1))]))
    }
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, c], [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, c], [xOut]))
  }
}

func MMDiT(b: Int, h: Int, w: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t = Input()
  let y = Input()
  let contextIn = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 1536, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 1536, h * w]).transposed(1, 2)
  let posEmbed = Parameter<Float>(.GPU(0), .NHWC(1, 192, 192, 1536))
  let spatialPosEmbed = posEmbed.reshaped(
    [1, h, w, 1536], offset: [0, (192 - h) / 2, (192 - w) / 2, 0],
    strides: [192 * 192 * 1536, 192 * 1536, 1536, 1]
  ).contiguous().reshaped([1, h * w, 1536])
  out = spatialPosEmbed + out
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 1536)
  let (yMlp0, yMlp2, yEmbedder) = VectorEmbedder(channels: 1536)
  let c = (tEmbedder(t) + yEmbedder(y)).reshaped([b, 1, 1536]).swish()
  let contextEmbedder = Dense(count: 1536, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = JointTransformerBlock(
      prefix: "diffusion_model.joint_blocks.\(i)", k: 64, h: 24, b: b, t: 154, hw: h * w,
      contextBlockPreOnly: i == 23)
    let blockOut = block(context, out, c)
    if i == 23 {
      out = blockOut
    } else {
      context = blockOut[0]
      out = blockOut[1]
    }
    readers.append(reader)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shift = Dense(count: 1536, name: "ada_ln_0")
  let scale = Dense(count: 1536, name: "ada_ln_1")
  out = (1 + scale(c)) .* normFinal(out) + shift(c)
  let linear = Dense(count: 2 * 2 * 16, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 16]).permuted(0, 5, 1, 3, 2, 4).contiguous().reshaped([
    b, 16, h * 2, w * 2,
  ])
  let reader: (PythonObject) -> Void = { state_dict in
    let x_embedder_proj_weight = state_dict["diffusion_model.x_embedder.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_embedder_proj_bias = state_dict["diffusion_model.x_embedder.proj.bias"].to(torch.float)
      .cpu().numpy()
    xEmbedder.weight.copy(from: try! Tensor<Float>(numpy: x_embedder_proj_weight))
    xEmbedder.bias.copy(from: try! Tensor<Float>(numpy: x_embedder_proj_bias))
    let pos_embed = state_dict["diffusion_model.pos_embed"].to(torch.float).cpu().numpy()
    posEmbed.weight.copy(from: try! Tensor<Float>(numpy: pos_embed))
    let t_embedder_mlp_0_weight = state_dict["diffusion_model.t_embedder.mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_0_bias = state_dict["diffusion_model.t_embedder.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    tMlp0.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight))
    tMlp0.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias))
    let t_embedder_mlp_2_weight = state_dict["diffusion_model.t_embedder.mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_2_bias = state_dict["diffusion_model.t_embedder.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    tMlp2.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight))
    tMlp2.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias))
    let y_embedder_mlp_0_weight = state_dict["diffusion_model.y_embedder.mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_mlp_0_bias = state_dict["diffusion_model.y_embedder.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    yMlp0.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_0_weight))
    yMlp0.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_0_bias))
    let y_embedder_mlp_2_weight = state_dict["diffusion_model.y_embedder.mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let y_embedder_mlp_2_bias = state_dict["diffusion_model.y_embedder.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    yMlp2.weight.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_2_weight))
    yMlp2.bias.copy(from: try! Tensor<Float>(numpy: y_embedder_mlp_2_bias))
    let context_embedder_weight = state_dict["diffusion_model.context_embedder.weight"].to(
      torch.float
    ).cpu().numpy()
    let context_embedder_bias = state_dict["diffusion_model.context_embedder.bias"].to(torch.float)
      .cpu().numpy()
    contextEmbedder.weight.copy(from: try! Tensor<Float>(numpy: context_embedder_weight))
    contextEmbedder.bias.copy(from: try! Tensor<Float>(numpy: context_embedder_bias))
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_adaln_modulation_weight = state_dict[
      "diffusion_model.final_layer.adaLN_modulation.1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let final_layer_adaln_modulation_bias = state_dict[
      "diffusion_model.final_layer.adaLN_modulation.1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    shift.weight.copy(
      from: try! Tensor<Float>(numpy: final_layer_adaln_modulation_weight[0..<1536, ...]))
    shift.bias.copy(from: try! Tensor<Float>(numpy: final_layer_adaln_modulation_bias[0..<1536]))
    scale.weight.copy(
      from: try! Tensor<Float>(numpy: final_layer_adaln_modulation_weight[1536..<(1536 * 2), ...]))
    scale.bias.copy(
      from: try! Tensor<Float>(numpy: final_layer_adaln_modulation_bias[1536..<(1536 * 2)]))
    let final_layer_linear_weight = state_dict["diffusion_model.final_layer.linear.weight"].to(
      torch.float
    ).cpu().numpy()
    let final_layer_linear_bias = state_dict["diffusion_model.final_layer.linear.bias"].to(
      torch.float
    ).cpu().numpy()
    linear.weight.copy(from: try! Tensor<Float>(numpy: final_layer_linear_weight))
    linear.bias.copy(from: try! Tensor<Float>(numpy: final_layer_linear_bias))
  }
  return (reader, Model([x, t, contextIn, y], [out]))
}

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
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

let (reader, dit) = MMDiT(b: 2, h: 64, w: 64)

graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy()).toGPU(0))
  let tTensor = graph.variable(
    timeEmbedding(timesteps: 1000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000).toGPU(0))
  let cTensor = graph.variable(
    try! Tensor<Float>(numpy: ctx.to(torch.float).cpu().numpy()).toGPU(0))
  let yTensor = graph.variable(try! Tensor<Float>(numpy: y.to(torch.float).cpu().numpy()).toGPU(0))
  dit.compile(inputs: xTensor, tTensor, cTensor, yTensor)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, tTensor, cTensor, yTensor))
  graph.openStore("/home/liu/workspace/swift-diffusion/sd3_medium_f32.ckpt") {
    $0.write("dit", model: dit)
  }
}
