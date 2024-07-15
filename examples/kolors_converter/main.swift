import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit
import SentencePiece

typealias FloatType = Float

let torch = Python.import("torch")
let diffusers = Python.import("diffusers")
let kolors_models_modeling_chatglm = Python.import("kolors.models.modeling_chatglm")
let kolors_models_tokenization_chatglm = Python.import("kolors.models.tokenization_chatglm")
let kolors_pipelines_pipeline_stable_diffusion_xl_chatglm_256 = Python.import(
  "kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256")

torch.set_grad_enabled(false)

let prompt = "一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“可图”"
let ckpt_dir = "/home/liu/workspace/Kolors/weights/Kolors"
let text_encoder = kolors_models_modeling_chatglm.ChatGLMModel.from_pretrained(
  "\(ckpt_dir)/text_encoder",
  torch_dtype: torch.float
).float()
print(text_encoder)
let tokenizer = kolors_models_tokenization_chatglm.ChatGLMTokenizer.from_pretrained(
  "\(ckpt_dir)/text_encoder")
let vae = diffusers.AutoencoderKL.from_pretrained("\(ckpt_dir)/vae").float()
let scheduler = diffusers.EulerDiscreteScheduler.from_pretrained("\(ckpt_dir)/scheduler")
let unet = diffusers.UNet2DConditionModel.from_pretrained("\(ckpt_dir)/unet").float()
let pipe = kolors_pipelines_pipeline_stable_diffusion_xl_chatglm_256.StableDiffusionXLPipeline(
  vae: vae,
  text_encoder: text_encoder,
  tokenizer: tokenizer,
  unet: unet,
  scheduler: scheduler,
  force_zeros_for_empty_prompt: false
).to("cuda")
pipe.enable_model_cpu_offload()
let image = pipe(
  prompt: prompt,
  height: 1024,
  width: 1024,
  num_inference_steps: 50,
  guidance_scale: 5.0,
  num_images_per_prompt: 1,
  generator: torch.Generator(pipe.device).manual_seed(66)
).images[0]
image.save("/home/liu/workspace/Kolors/scripts/outputs/sample_test.jpg")

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/Kolors/weights/Kolors/tokenizer/tokenizer.model")
var tokens = sentencePiece.encode(prompt).map { return $0.id }
tokens.insert(64790, at: 0)
tokens.insert(64792, at: 1)

let state_dict = text_encoder.state_dict()
let unet_state_dict = unet.state_dict()
// print(unet_state_dict.keys())

let graph = DynamicGraph()

var rotTensor = graph.variable(.CPU, .NHWC(1, 27, 1, 128), of: Float.self)
for i in 0..<27 {
  for k in 0..<32 {
    let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 64)
    let sintheta = sin(theta)
    let costheta = cos(theta)
    rotTensor[0, i, 0, k * 2] = Float(costheta)
    rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
  }
  for k in 32..<64 {
    rotTensor[0, i, 0, k * 2] = 1
    rotTensor[0, i, 0, k * 2 + 1] = 0
  }
}

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int)) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t.1, hk, k])
  var queries = toqueries(x).reshaped([b, t.1, h, k])
  var values = tovalues(x).reshaped([b, t.1, hk, k])
  if h > hk {
    keys = Concat(axis: 3)(Array(repeating: keys, count: h / hk))
    values = Concat(axis: 3)(Array(repeating: values, count: h / hk))
  }
  keys = keys.reshaped([b, t.1, h, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  keys = keys.transposed(1, 2)
  queries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
  values = values.reshaped([b, t.1, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
  dot = dot.reshaped([b * h * t.1, t.1])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t.1, t.1])
  var out = dot * values
  out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "o")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let qkv_weight = state_dict["\(prefix).self_attention.query_key_value.weight"].type(torch.float)
      .cpu().numpy()
    let qkv_bias = state_dict["\(prefix).self_attention.query_key_value.bias"].type(torch.float)
      .cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: qkv_weight[0..<(h * k), ...]))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: qkv_bias[0..<(h * k)]))
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: qkv_weight[(h * k)..<((h + hk) * k), ...]))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: qkv_bias[(h * k)..<((h + hk) * k)]))
    tovalues.weight.copy(
      from: try! Tensor<Float>(numpy: qkv_weight[((h + hk) * k)..<((h + hk * 2) * k), ...]))
    tovalues.bias.copy(
      from: try! Tensor<Float>(numpy: qkv_bias[((h + hk) * k)..<((h + hk * 2) * k)]))
    let dense_weight = state_dict["\(prefix).self_attention.dense.weight"].type(torch.float)
      .cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: dense_weight))
  }
  return (Model([x, rot, causalAttentionMask], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true)
  let w3 = Dense(count: intermediateSize, noBias: true)
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true)
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

func GLMTransformerBlock(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int), MLP: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let (attention, attentionReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot, causalAttentionMask) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "ffn")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).input_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    attentionReader(state_dict)
    let norm2_weight = state_dict["\(prefix).post_attention_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    let dense_h_to_4h_weight = state_dict["\(prefix).mlp.dense_h_to_4h.weight"].type(torch.float)
      .cpu().numpy()
    w1.weight.copy(from: try! Tensor<Float>(numpy: dense_h_to_4h_weight[0..<MLP, ...]))
    w3.weight.copy(from: try! Tensor<Float>(numpy: dense_h_to_4h_weight[MLP..<(MLP * 2), ...]))
    let dense_4h_to_h_weight = state_dict["\(prefix).mlp.dense_4h_to_h.weight"].type(torch.float)
      .cpu().numpy()
    w2.weight.copy(from: try! Tensor<Float>(numpy: dense_4h_to_h_weight))
  }
  return (Model([x, rot, causalAttentionMask], [out]), reader)
}

public func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "word_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["embedding.word_embeddings.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab))
  }
  return (Model([tokens], [embedding]), reader)
}

func GLMTransformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: Int, cachedTokenLength: Int,
  layers: Int, MLP: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let (embedding, embeddingReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (layer, reader) = GLMTransformerBlock(
      prefix: "encoder.layers.\(i)", k: width / heads, h: heads, hk: heads / 16, b: batchSize,
      t: (cachedTokenLength + tokenLength, tokenLength),
      MLP: MLP)
    out = layer(out, rot, causalAttentionMask)
    readers.append(reader)
  }
  /*
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  */
  let reader: (PythonObject) -> Void = { state_dict in
    embeddingReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    /*
    let norm_weight = state_dict["encoder.final_layernorm.weight"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    */
  }
  return (Model([tokens, rot, causalAttentionMask], [out]), reader)
}
let (transformer, reader) = GLMTransformer(
  Float.self, vocabularySize: 65_024, width: 4_096, tokenLength: 256, cachedTokenLength: 0,
  layers: 28, MLP: 13696, heads: 32, batchSize: 1)
graph.withNoGrad {
  let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [256], of: Int32.self)
  for i in 0..<(256 - tokens.count) {
    tokensTensor[i] = 0
  }
  for i in (256 - tokens.count)..<256 {
    tokensTensor[i] = tokens[i - (256 - tokens.count)]
  }
  var paddedRotTensor = graph.variable(.CPU, .NHWC(1, 256, 1, 128), of: Float.self)
  for i in 0..<(256 - tokens.count) {
    paddedRotTensor[0..<1, i..<(i + 1), 0..<1, 0..<128] = rotTensor[0..<1, 0..<1, 0..<1, 0..<128]
  }
  paddedRotTensor[0..<1, (256 - tokens.count)..<256, 0..<1, 0..<128] =
    rotTensor[0..<1, 0..<tokens.count, 0..<1, 0..<128]
  let causalAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 256, 256)))
  causalAttentionMask.full(0)
  for i in (256 - tokens.count)..<255 {
    for j in (i + 1)..<256 {
      causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  for i in (256 - tokens.count)..<256 {
    for j in 0..<(256 - tokens.count) {
      causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  debugPrint(causalAttentionMask)
  let tokensTensorGPU = tokensTensor.toGPU(1)
  let rotTensorGPU = paddedRotTensor.toGPU(1)
  let causalAttentionMaskGPU = causalAttentionMask.toGPU(1)
  transformer.compile(inputs: tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU)
  reader(state_dict)
  let out = transformer(inputs: tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU).map {
    $0.as(of: Float.self)
  }
  debugPrint(out)
  let encoderHidProj = Dense(count: 2048)
  encoderHidProj.compile(inputs: out[0])
  let encoder_hid_proj_weight = unet_state_dict["encoder_hid_proj.weight"].type(torch.float)
    .cpu().numpy()
  let encoder_hid_proj_bias = unet_state_dict["encoder_hid_proj.bias"].type(torch.float)
    .cpu().numpy()
  encoderHidProj.weight.copy(from: try! Tensor<Float>(numpy: encoder_hid_proj_weight))
  encoderHidProj.bias.copy(from: try! Tensor<Float>(numpy: encoder_hid_proj_bias))
  let encoderOut = encoderHidProj(inputs: out[0]).map { $0.as(of: Float.self) }
  debugPrint(encoderOut)
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/chatglm3_6b_f32.ckpt") {
    $0.write("text_model", model: transformer)
  }
  graph.openStore("/home/liu/workspace/swift-diffusion/kolors_encoder_hid_proj_f32.ckpt") {
    $0.write("encoder_hid_proj", model: encoderHidProj)
  }
  */
}
