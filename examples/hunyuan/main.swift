import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let torch = Python.import("torch")

let numpy = Python.import("numpy")

let hyvideo_config = Python.import("hyvideo.config")
let hyvideo_inference = Python.import("hyvideo.inference")

let args = hyvideo_config.parse_args()
args.dit_weight = "/home/liu/workspace/HunyuanVideo/\(args.dit_weight)".pythonObject
print(args)
args.use_cpu_offload = false
args.video_size = PythonObject(tupleOf: 544, 960)
args.prompt = "A cat walks on the grass, realistic style."
/*
model='HYVideo-T/2-cfgdistill', latent_channels=16, precision='bf16', rope_theta=256, vae='884-16c-hy', vae_precision='fp16', vae_tiling=True, text_encoder='llm', text_encoder_precision='fp16', text_states_dim=4096, text_len=256, tokenizer='llm', prompt_template='dit-llm-encode', prompt_template_video='dit-llm-encode-video', hidden_state_skip_layer=2, apply_final_norm=False, text_encoder_2='clipL', text_encoder_precision_2='fp16', text_states_dim_2=768, tokenizer_2='clipL', text_len_2=77, denoise_type='flow', flow_shift=7.0, flow_reverse=False, flow_solver='euler', use_linear_quadratic_schedule=False, linear_schedule_end=25, model_base='ckpts', dit_weight='/home/liu/workspace/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt', model_resolution='540p', load_key='module', use_cpu_offload=False, batch_size=1, infer_steps=50, disable_autocast=False, save_path='./results', save_path_suffix='', name_suffix='', num_videos=1, video_size=(720, 1280), video_length=129, prompt=None, seed_type='auto', seed=None, neg_prompt=None, cfg_scale=1.0, embedded_cfg_scale=6.0, reproduce=False, ulysses_degree=1, ring_degree=1)

model='HYVideo-T/2-cfgdistill', latent_channels=16, precision='bf16', rope_theta=256, vae='884-16c-hy', vae_precision='fp16', vae_tiling=True, text_encoder='llm', text_encoder_precision='fp16', text_states_dim=4096, text_len=256, tokenizer='llm', prompt_template='dit-llm-encode', prompt_template_video='dit-llm-encode-video', hidden_state_skip_layer=2, apply_final_norm=False, text_encoder_2='clipL', text_encoder_precision_2='fp16', text_states_dim_2=768, tokenizer_2='clipL', text_len_2=77, denoise_type='flow', flow_shift=7.0, flow_reverse=True, flow_solver='euler', use_linear_quadratic_schedule=False, linear_schedule_end=25, model_base='ckpts', dit_weight='ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt', model_resolution='540p', load_key='module', use_cpu_offload=True, batch_size=1, infer_steps=50, disable_autocast=False, save_path='./results', save_path_suffix='', name_suffix='', num_videos=1, video_size=[544, 960], video_length=129, prompt='A cat walks on the grass, realistic style.', seed_type='auto', seed=None, neg_prompt=None, cfg_scale=1.0, embedded_cfg_scale=6.0, reproduce=False, ulysses_degree=1, ring_degree=1
*/

let tokenizer = GPT2Tokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/hunyuan/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/hunyuan/merges.txt",
  specialTokens: [
    "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
    "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
  ])
let prompt = "A cat walks on the grass, realistic style."
let promptWithTemplate =
  "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\(prompt)<|eot_id|>"
print(promptWithTemplate)
let result = tokenizer.tokenize(text: promptWithTemplate, addSpecialTokens: true)
print(result)

let hunyuan_video_sampler = hyvideo_inference.HunyuanVideoSampler.from_pretrained(
  "/home/liu/workspace/HunyuanVideo/ckpts", args)

print(hunyuan_video_sampler.pipeline.text_encoder.model)

let text_inputs = hunyuan_video_sampler.pipeline.text_encoder.text2tokens(
  [prompt], data_type: "video")
let prompt_outputs = hunyuan_video_sampler.pipeline.text_encoder.encode(
  text_inputs, data_type: "video", device: torch.device("cuda:0"))
print(prompt_outputs)

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  // The rotary in Llama is first half and second half, so we need to do the extra transpose to use with cmul.
  var keys = tokeys(x).reshaped([b, t, hk, 2, k / 2]).transposed(3, 4).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, 2, k / 2]).transposed(3, 4).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let q_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).cpu()
      .numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let k_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
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
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

func TransformerBlock(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
    let norm1_weight = state_dict["\(prefix).input_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm1_weight)))
    let norm2_weight = state_dict["\(prefix).post_attention_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm2_weight)))
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

public func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["embed_tokens.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vocab)))
  }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Int?, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embedReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  var hiddenStates: Model.IO? = nil
  for i in 0..<layers {
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: heads / 4, b: batchSize,
      t: tokenLength,
      MLP: MLP)
    out = layer(out, rot)
    readers.append(reader)
    if let outputHiddenStates = outputHiddenStates, outputHiddenStates == i {
      hiddenStates = out
    }
  }
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_weight = state_dict["norm.weight"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_weight)))
  }
  return (Model([tokens, rot], (hiddenStates.map { [$0] } ?? []) + [out]), reader)
}

let graph = DynamicGraph()
graph.withNoGrad {
  let text_encoder_state_dict = hunyuan_video_sampler.pipeline.text_encoder.model.state_dict()
  let (transformer, reader) = Transformer(
    Float16.self, vocabularySize: 128_320, maxLength: 351, width: 4_096, tokenLength: 351,
    layers: 32, MLP: 14336, heads: 32, outputHiddenStates: 29, batchSize: 1)
  let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [351], of: Int32.self)
  for i in 0..<result.count {
    tokensTensor[i] = result[i]
  }
  for i in result.count..<351 {
    tokensTensor[i] = 128258
  }
  let rotTensor = graph.variable(.CPU, .NHWC(1, 351, 1, 128), of: Float.self)
  for i in 0..<351 {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(500_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let tokensTensorGPU = tokensTensor.toGPU(1)
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(1)
  transformer.compile(inputs: tokensTensorGPU, rotTensorGPU)
  reader(text_encoder_state_dict)
  let lastHiddenStates = transformer(inputs: tokensTensorGPU, rotTensorGPU)[0].as(of: Float16.self)[
    95..<351, 0..<4096
  ].copied()
  debugPrint(lastHiddenStates)
}
