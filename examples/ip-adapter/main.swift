import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let torch = Python.import("torch")
let diffusers = Python.import("diffusers")
let Image = Python.import("PIL.Image")
let ip_adapter = Python.import("ip_adapter")

// let base_model_path = "SG161222/RealVisXL_V1.0"
let base_model_path = "runwayml/stable-diffusion-v1-5"
let vae_model_path = "stabilityai/sd-vae-ft-mse"
let image_encoder_path = "/home/liu/workspace/IP-Adapter/models/image_encoder"
// let ip_ckpt = "/home/liu/workspace/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
// let ip_ckpt = "/home/liu/workspace/IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
let ip_ckpt = "/home/liu/workspace/IP-Adapter/models/ip-adapter-plus_sd15.bin"
let device = "cuda"

let noise_scheduler = diffusers.DDIMScheduler(
  num_train_timesteps: 1000,
  beta_start: 0.00085,
  beta_end: 0.012,
  beta_schedule: "scaled_linear",
  clip_sample: false,
  set_alpha_to_one: false,
  steps_offset: 1
)
let vae = diffusers.AutoencoderKL.from_pretrained(vae_model_path).to(dtype: torch.float16)

// load SD pipeline
let pipe = diffusers.StableDiffusionPipeline.from_pretrained(
  base_model_path,
  torch_dtype: torch.float16,
  scheduler: noise_scheduler,
  vae: vae,
  feature_extractor: Python.None,
  safety_checker: Python.None
)

// read image prompt (face, here we use a ai-generation face)
let image = Image.open("/home/liu/workspace/IP-Adapter/assets/images/ai_face.png")
image.resize(PythonObject(tupleOf: 256, 256))

// load ip-adapter
let ip_model = ip_adapter.IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens: 16)

let images = ip_model.generate(
  pil_image: image, num_samples: 1, num_inference_steps: 50, seed: 420,
  prompt: "photo of a beautiful girl wearing casual shirt in a garden")
images[0].save("/home/liu/workspace/IP-Adapter/grid.png")

/*
// load SDXL pipeline
let pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
  base_model_path,
  torch_dtype: torch.float16,
  add_watermarker: false
)

// load ip-adapter
let ip_model = ip_adapter.IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens: 16)

// read image prompt
let image = Image.open("/home/liu/workspace/IP-Adapter/assets/images/woman.png")
image.resize(PythonObject(tupleOf: 512, 512))

// generate image variations with only image prompt
let num_samples = 1
let images = ip_model.generate(
  pil_image: image, num_samples: num_samples, num_inference_steps: 30, seed: 42)
*/
let state_dict = ip_model.image_encoder.state_dict()
print(ip_model.image_encoder)
let proj_state_dict = ip_model.image_proj_model.state_dict()
print(ip_model.image_proj_model)
let ip_layers_state_dict = ip_model.pipe.unet.state_dict()

var clip_image = ip_model.clip_image_processor(images: image, return_tensors: "pt").pixel_values
clip_image = clip_image.to(ip_model.device, dtype: torch.float16)

// multimodal prompts
// images = ip_model.generate(pil_image: image, num_samples: num_samples, num_inference_steps: 30, seed: 42, prompt: "best quality, high quality, wearing sunglasses on the beach", scale: 0.5)
// print(images)

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
    let ln_1_weight = state_dict["\(prefix).layer_norm1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).layer_norm1.bias"].type(torch.float).cpu().numpy()
    ln1.weight.copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    ln1.bias.copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let q_proj_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).cpu()
      .numpy()
    let q_proj_bias = state_dict["\(prefix).self_attn.q_proj.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_proj_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_proj_bias))
    let k_proj_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).cpu()
      .numpy()
    let k_proj_bias = state_dict["\(prefix).self_attn.k_proj.bias"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_proj_weight))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_proj_bias))
    let v_proj_weight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu()
      .numpy()
    let v_proj_bias = state_dict["\(prefix).self_attn.v_proj.bias"].type(torch.float).cpu().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_proj_weight))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_proj_bias))
    let out_proj_weight = state_dict["\(prefix).self_attn.out_proj.weight"].type(torch.float).cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).self_attn.out_proj.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: out_proj_bias))
    let ln_2_weight = state_dict["\(prefix).layer_norm2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).layer_norm2.bias"].type(torch.float).cpu().numpy()
    ln2.weight.copy(from: try! Tensor<Float>(numpy: ln_2_weight))
    ln2.bias.copy(from: try! Tensor<Float>(numpy: ln_2_bias))
    let c_fc_weight = state_dict["\(prefix).mlp.fc1.weight"].type(torch.float).cpu().numpy()
    let c_fc_bias = state_dict["\(prefix).mlp.fc1.bias"].type(torch.float).cpu().numpy()
    fc.weight.copy(from: try! Tensor<Float>(numpy: c_fc_weight))
    fc.bias.copy(from: try! Tensor<Float>(numpy: c_fc_bias))
    let c_proj_weight = state_dict["\(prefix).mlp.fc2.weight"].type(torch.float).cpu().numpy()
    let c_proj_bias = state_dict["\(prefix).mlp.fc2.bias"].type(torch.float).cpu().numpy()
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
  let classEmbedding = Parameter<Float>(.GPU(0), .CHW(1, 1, width))
  let positionalEmbedding = Parameter<Float>(.GPU(0), .CHW(1, grid * grid + 1, width))
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
      prefix: "vision_model.encoder.layers.\(i)", k: width / heads, h: heads, b: batchSize,
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
    let conv1_weight = state_dict["vision_model.embeddings.patch_embedding.weight"].type(
      torch.float
    ).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let class_embedding_weight = state_dict["vision_model.embeddings.class_embedding"].type(
      torch.float
    ).cpu().numpy()
    classEmbedding.weight.copy(from: try! Tensor<Float>(numpy: class_embedding_weight))
    let positional_embedding_weight = state_dict[
      "vision_model.embeddings.position_embedding.weight"
    ].type(torch.float).cpu().numpy()
    positionalEmbedding.weight.copy(from: try! Tensor<Float>(numpy: positional_embedding_weight))
    let ln_pre_weight = state_dict["vision_model.pre_layrnorm.weight"].type(torch.float).cpu()
      .numpy()
    let ln_pre_bias = state_dict["vision_model.pre_layrnorm.bias"].type(torch.float).cpu().numpy()
    lnPre.weight.copy(from: try! Tensor<Float>(numpy: ln_pre_weight))
    lnPre.bias.copy(from: try! Tensor<Float>(numpy: ln_pre_bias))
    for reader in readers {
      reader(state_dict)
    }
    if let lnPost = finalLayerNorm {
      let ln_post_weight = state_dict["vision_model.post_layernorm.weight"].type(torch.float).cpu()
        .numpy()
      let ln_post_bias = state_dict["vision_model.post_layernorm.bias"].type(torch.float).cpu()
        .numpy()
      lnPost.weight.copy(from: try! Tensor<Float>(numpy: ln_post_weight))
      lnPost.bias.copy(from: try! Tensor<Float>(numpy: ln_post_bias))
    }
  }
  return (reader, Model([x], [out]))
}

func PerceiverAttention(prefix: String, k: Int, h: Int, b: Int, t: (Int, Int)) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let outX = norm1(x)
  let c = Input()
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  let outC = norm2(c)
  let outXC = Functional.concat(axis: 1, outX, outC)
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(outXC).reshaped([b, t.0 + t.1, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(outC)).reshaped([b, t.1, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(outXC).reshaped([b, t.0 + t.1, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t.1, t.0 + t.1])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t.1, t.0 + t.1])
  var out = dot * values
  out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: k * h, noBias: true)
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    let norm1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu().numpy()
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    let norm2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu().numpy()
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let to_q_weight = state_dict["\(prefix).to_q.weight"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: to_q_weight))
    let to_kv_weight = state_dict["\(prefix).to_kv.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: to_kv_weight[0..<(k * h), ...]))
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: to_kv_weight[(k * h)..., ...]))
    let to_out_weight = state_dict["\(prefix).to_out.weight"].type(torch.float).cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: to_out_weight))
  }
  return (reader, Model([x, c], [out]))
}

func ResamplerLayer(prefix: String, k: Int, h: Int, b: Int, t: (Int, Int)) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let c = Input()
  let (attentionReader, attention) = PerceiverAttention(
    prefix: prefix + ".0", k: k, h: h, b: b, t: t)
  var out = c + attention(x, c).reshaped([b, t.1, h * k])
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
  let fc1 = Dense(count: k * h * 4, noBias: true)
  let gelu = GELU()
  let fc2 = Dense(count: k * h, noBias: true)
  out = out + fc2(gelu(fc1(layerNorm(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    attentionReader(state_dict)
    let layerNorm_weight = state_dict["\(prefix).1.0.weight"].type(torch.float).cpu().numpy()
    layerNorm.weight.copy(from: try! Tensor<Float>(numpy: layerNorm_weight))
    let layerNorm_bias = state_dict["\(prefix).1.0.bias"].type(torch.float).cpu().numpy()
    layerNorm.bias.copy(from: try! Tensor<Float>(numpy: layerNorm_bias))
    let fc1_weight = state_dict["\(prefix).1.1.weight"].type(torch.float).cpu().numpy()
    fc1.weight.copy(from: try! Tensor<Float>(numpy: fc1_weight))
    let fc2_weight = state_dict["\(prefix).1.3.weight"].type(torch.float).cpu().numpy()
    fc2.weight.copy(from: try! Tensor<Float>(numpy: fc2_weight))
  }
  return (reader, Model([x, c], [out]))
}

func Resampler(
  width: Int, outputDim: Int, heads: Int, grid: Int, queries: Int, layers: Int, batchSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let latents = Parameter<Float>(.GPU(0), .CHW(1, queries, width))
  let projIn = Dense(count: width)
  let projX = projIn(x)
  var readers = [(PythonObject) -> Void]()
  let (firstReader, firstLayer) = ResamplerLayer(
    prefix: "layers.0", k: width / heads, h: heads, b: batchSize, t: (grid * grid + 1, queries))
  readers.append(firstReader)
  var out = firstLayer(projX, latents)
  for i in 1..<layers {
    let (reader, layer) = ResamplerLayer(
      prefix: "layers.\(i)", k: width / heads, h: heads, b: batchSize, t: (grid * grid + 1, queries)
    )
    readers.append(reader)
    out = layer(projX, out)
  }
  let projOut = Dense(count: outputDim)
  out = projOut(out)
  let normOut = LayerNorm(epsilon: 1e-5, axis: [2])
  out = normOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let proj_in_weight = state_dict["proj_in.weight"].type(torch.float).cpu().numpy()
    projIn.weight.copy(from: try! Tensor<Float>(numpy: proj_in_weight))
    let proj_in_bias = state_dict["proj_in.bias"].type(torch.float).cpu().numpy()
    projIn.bias.copy(from: try! Tensor<Float>(numpy: proj_in_bias))
    let latents_weight = state_dict["latents"].type(torch.float).cpu().numpy()
    latents.weight.copy(from: try! Tensor<Float>(numpy: latents_weight))
    for reader in readers {
      reader(state_dict)
    }
    let proj_out_weight = state_dict["proj_out.weight"].type(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    let proj_out_bias = state_dict["proj_out.bias"].type(torch.float).cpu().numpy()
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
    let norm_out_weight = state_dict["norm_out.weight"].type(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    let norm_out_bias = state_dict["norm_out.bias"].type(torch.float).cpu().numpy()
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
  }
  return (reader, Model([x], [out]))
}

func CrossAttentionFixed(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model) {
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
  return (tokeys, tovalues, Model([c], [keys, values]))
}

func BasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(k: k, h: h, b: b, hw: hw, t: t)
  let reader: (PythonObject) -> Void = { state_dict in
    let attn2_to_k_weight = state_dict[
      "\(prefix).attn2.processor.to_k_ip.weight"
    ].type(torch.float).cpu().numpy()
    tokeys2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_k_weight))
    // print("\"diffusion_model.\(prefix).attn2.to_k.weight\": [\"\(tokeys2.weight.name)\"],")
    let attn2_to_v_weight = state_dict[
      "\(prefix).attn2.processor.to_v_ip.weight"
    ].type(torch.float).cpu().numpy()
    tovalues2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
    // print("\"diffusion_model.\(prefix).attn2.to_v.weight\": [\"\(tovalues2.weight.name)\"],")
  }
  return (reader, attn2)
}

func SpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  let hw = height * width
  for i in 0..<depth {
    let (reader, block) = BasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(block(c))
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([c], outs))
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
    prefix: "\(prefix).attentions.\(layerStart)",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingSize,
    intermediateSize: channels * 4)
  return (transformerReader, transformer)
}

func MiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, c: Model.IO
) -> ((PythonObject) -> Void, Model.IO) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (
    transformerReader, transformer
  ) = SpatialTransformerFixed(
    prefix: "mid_block.attentions.0", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  let out = transformer(c)
  return (transformerReader, out)
}

func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO
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
    for j in 0..<numRepeat {
      if attentionBlock > 0 {
        let (reader, inputLayer) = BlockLayerFixed(
          prefix: "down_blocks.\(i)",
          layerStart: j, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        previousChannel = channel
        outs.append(inputLayer(c))
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
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO
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
    for j in 0..<(numRepeat + 1) {
      if attentionBlock > 0 {
        let (reader, outputLayer) = BlockLayerFixed(
          prefix: "up_blocks.\(channels.count - 1 - i)",
          layerStart: j, skipConnection: true,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        outs.append(outputLayer(c))
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
  let (inputReader, inputBlocks) = InputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    c: c)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (middleReader, middleBlock) = MiddleBlockFixed(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 77, attentionBlock: middleBlockAttentionBlock, c: c)
  out.append(middleBlock)
  let (outputReader, outputBlocks) = OutputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    c: c)
  out.append(contentsOf: outputBlocks)
  let reader: (PythonObject) -> Void = { state_dict in
    inputReader(state_dict)
    middleReader(state_dict)
    outputReader(state_dict)
  }
  return (reader, Model([c], out))
}

let graph = DynamicGraph()
graph.withNoGrad {
  let (reader, vit) = VisionTransformer(
    grid: 16, width: 1280, outputDim: 1024, layers: 31, heads: 16, batchSize: 1,
    noFinalLayerNorm: true)
  let clip_image = clip_image.type(torch.float).cpu().numpy()
  let clipImageTensor = graph.variable(try! Tensor<Float>(numpy: clip_image)).toGPU(0)
  vit.compile(inputs: clipImageTensor)
  reader(state_dict)
  let imageEmbeds = vit(inputs: clipImageTensor)[0].as(of: Float.self).reshaped(.CHW(1, 257, 1280))
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_vit_h14_vision_model_f32.ckpt") {
    $0.write("vision_model", model: vit)
  }
  */
  debugPrint(imageEmbeds)
  let (resamplerReader, resampler) = Resampler(
    width: 768, outputDim: 768, heads: 12, grid: 16, queries: 16, layers: 4, batchSize: 1)
  resampler.compile(inputs: imageEmbeds)
  resamplerReader(proj_state_dict)
  let imagePromptEmebeds = resampler(inputs: imageEmbeds)[0].as(of: Float.self)
  debugPrint(imagePromptEmebeds)
  let c = torch.randn([2, 77, 768])
  let cTensor = graph.variable(try! Tensor<Float>(numpy: c.numpy())).toGPU(0)
  let (readerFixed, unetFixed) = UNetXLFixed(
    batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280, 1280],
    attentionRes: [1: 1, 2: 1, 4: 1])
  unetFixed.compile(inputs: cTensor)
  readerFixed(ip_layers_state_dict)
  let kvs = unetFixed(inputs: cTensor).map { $0.as(of: Float.self) }
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/ip_adapter_plus_sd_v1.x_open_clip_h14_f32.ckpt"
  ) {
    $0.write("resampler", model: resampler)
    $0.write("unet_ip_fixed", model: unetFixed)
  }
}
