import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let torch = Python.import("torch")

torch.set_grad_enabled(false)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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

/*
print(hunyuan_video_sampler.pipeline.text_encoder.model)
print(hunyuan_video_sampler.pipeline.transformer)
*/
print(hunyuan_video_sampler.pipeline.vae)

let text_inputs = hunyuan_video_sampler.pipeline.text_encoder.text2tokens(
  [prompt], data_type: "video")
let prompt_outputs = hunyuan_video_sampler.pipeline.text_encoder.encode(
  text_inputs, data_type: "video", device: torch.device("cuda:0"))
let text_inputs_2 = hunyuan_video_sampler.pipeline.text_encoder_2.text2tokens(
  [prompt], data_type: "video")
let prompt_outputs_2 = hunyuan_video_sampler.pipeline.text_encoder_2.encode(
  text_inputs_2, data_type: "video", device: torch.device("cuda:0"))
let x = torch.randn([1, 16, 33, 68, 120], dtype: torch.float16, device: torch.device("cuda:0"))
let t = torch.tensor([900], dtype: torch.float, device: torch.device("cuda:0"))
let text_states_2 = prompt_outputs_2.hidden_state.to(torch.float16)
let guidance = torch.tensor([3500], dtype: torch.float, device: torch.device("cuda:0"))
let hyvideo_modules_posemb_layers = Python.import("hyvideo.modules.posemb_layers")
let (freqs_cos, freqs_sin) = hyvideo_modules_posemb_layers.get_nd_rotary_pos_embed(
  [16, 56, 56], [33, 34, 60], use_real: true, theta: 256, theta_rescale_factor: 1
).tuple2
print("freqs_cos \(freqs_cos), freqs_sin \(freqs_sin)")

// Offload to CPU.
hunyuan_video_sampler.pipeline.text_encoder.model.to(torch.device("cpu"))
torch.set_autocast_enabled("cuda", true)
torch.set_autocast_dtype("cuda", torch.bfloat16)
let outputs = hunyuan_video_sampler.pipeline.transformer(
  x, t, text_states: prompt_outputs.hidden_state.to(torch.float16),
  text_mask: prompt_outputs.attention_mask, text_states_2: text_states_2, freqs_cos: freqs_cos,
  freqs_sin: freqs_sin, guidance: guidance, return_dict: true)
print(outputs)
torch.set_autocast_enabled("cuda", false)

// Offload to CPU.
/*
hunyuan_video_sampler.pipeline.text_encoder.model.to(torch.device("cpu"))
hunyuan_video_sampler.pipeline.transformer.to(torch.device("cpu"))
let vae = hunyuan_video_sampler.pipeline.vae
vae.to(torch.float)
let z = torch.randn([1, 16, 17, 32, 32]).to(torch.float).cuda()
let sample = vae.decode(z).sample
vae.encode(sample)
*/
let graph = DynamicGraph()

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    // The rotary in Llama is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    let q_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).view(
      32, 2, 64, 4096
    ).transpose(1, 2).cpu().numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let k_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).view(
      8, 2, 64, 4096
    ).transpose(1, 2).cpu().numpy()
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

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func RefinerSelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let tokeys = Dense(count: k * hk, name: "refiner_k_proj")
  let toqueries = Dense(count: k * h, name: "refiner_q_proj")
  let tovalues = Dense(count: k * hk, name: "refiner_v_proj")
  let keys = tokeys(x).reshaped([b, t, hk, k])
  let queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
    queries, keys, values
  ).reshaped([b, t, h * k])
  let unifyheads = Dense(count: k * h, name: "refiner_out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    // The rotary in Llama is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    let qkv_weight = state_dict["\(prefix).self_attn_qkv.weight"].type(torch.float).cpu().numpy()
    let qkv_bias = state_dict["\(prefix).self_attn_qkv.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: qkv_weight[..<(k * h), ...])))
    tokeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: qkv_weight[(k * h)..<(2 * k * h), ...]))
    )
    tovalues.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: qkv_weight[(2 * k * h)..<(3 * k * h), ...])))
    toqueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: qkv_bias[..<(k * h)])))
    tokeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: qkv_bias[(k * h)..<(2 * k * h)])))
    tovalues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: qkv_bias[(2 * k * h)..<(3 * k * h)])))
    let proj_weight = state_dict["\(prefix).self_attn_proj.weight"].type(torch.float).cpu()
      .numpy()
    let proj_bias = state_dict["\(prefix).self_attn_proj.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_weight)))
    unifyheads.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_bias)))
  }
  return (Model([x], [out]), reader)
}

func IndividualRefinerBlock(prefix: String, t: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let c = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], name: "refiner_norm1")
  let gateMsa = Dense(count: 3_072, name: "refiner_ada_ln_msa")
  let (attention, attentionReader) = RefinerSelfAttention(
    prefix: prefix, k: 128, h: 24, hk: 24, b: 1, t: t)
  var out = x + attention(norm1(x)) .* gateMsa(c)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], name: "refiner_norm2")
  let mlp0 = Dense(count: 3_072 * 4, name: "refiner_mlp_0")
  let mlp1 = Dense(count: 3_072, name: "refiner_mlp_1")
  let gateMlp = Dense(count: 3_072, name: "refiner_ada_ln_mlp")
  out = out + mlp1(mlp0(norm2(out)).swish()) .* gateMlp(c)
  let reader: (PythonObject) -> Void = { state_dict in
    attentionReader(state_dict)
    let norm1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu()
      .numpy()
    let norm1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu()
      .numpy()
    norm1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm1_weight)))
    norm1.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm1_bias)))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu()
      .numpy()
    let norm2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu()
      .numpy()
    norm2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm2_weight)))
    norm2.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm2_bias)))
    let fc1_weight = state_dict["\(prefix).mlp.fc1.weight"].type(torch.float).cpu()
      .numpy()
    let fc1_bias = state_dict["\(prefix).mlp.fc1.bias"].type(torch.float).cpu()
      .numpy()
    mlp0.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: fc1_weight)))
    mlp0.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: fc1_bias)))
    let fc2_weight = state_dict["\(prefix).mlp.fc2.weight"].type(torch.float).cpu()
      .numpy()
    let fc2_bias = state_dict["\(prefix).mlp.fc2.bias"].type(torch.float).cpu()
      .numpy()
    mlp1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: fc2_weight)))
    mlp1.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: fc2_bias)))
    let adaLN_weight = state_dict["\(prefix).adaLN_modulation.1.weight"].type(torch.float).cpu()
      .numpy()
    gateMsa.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: adaLN_weight[..<3072, ...])))
    gateMlp.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: adaLN_weight[3072..<6144, ...])))
    let adaLN_bias = state_dict["\(prefix).adaLN_modulation.1.bias"].type(torch.float).cpu()
      .numpy()
    gateMsa.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: adaLN_bias[..<3072])))
    gateMlp.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: adaLN_bias[3072..<6144])))
  }
  return (Model([x, c], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool, upcast: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let rot = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context).to(.Float16) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  xQ = Functional.cmul(left: xQ, right: rot)
  xK = Functional.cmul(left: xK, right: rot)
  let keys = Functional.concat(axis: 1, xK, contextK)
  let values = Functional.concat(axis: 1, xV, contextV)
  let queries = Functional.concat(axis: 1, xQ, contextQ)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    queries, keys, values
  ).reshaped([b, t + hw, h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], offset: [0, hw, 0], strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + ((upcast ? contextChunks[5].to(of: contextOut) : contextChunks[5])
      .* contextFF(
        contextNorm2(contextOut).to(.Float16) .* (1 + contextChunks[4]) + contextChunks[3])).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + ((upcast ? xChunks[5].to(of: xOut) : xChunks[5])
    .* xFF(xNorm2(xOut).to(.Float16) .* (1 + xChunks[4]) + xChunks[3])).to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
    let txt_attn_q_weight = state_dict["\(prefix).txt_attn_qkv.weight"][..<(k * h), ...].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_q_bias = state_dict["\(prefix).txt_attn_qkv.bias"][..<(k * h)].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_q_weight)))
    contextToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_q_bias)))
    let txt_attn_k_weight = state_dict["\(prefix).txt_attn_qkv.weight"][(k * h)..<(2 * k * h), ...]
      .to(
        torch.float
      ).cpu().numpy()
    let txt_attn_k_bias = state_dict["\(prefix).txt_attn_qkv.bias"][(k * h)..<(2 * k * h)].to(
      torch.float
    ).cpu().numpy()
    contextToKeys.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: txt_attn_k_weight)))
    contextToKeys.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: txt_attn_k_bias)))
    let txt_attn_v_weight = state_dict["\(prefix).txt_attn_qkv.weight"][
      (2 * k * h)..<(3 * k * h), ...
    ].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_v_bias = state_dict["\(prefix).txt_attn_qkv.bias"][(2 * k * h)..<(3 * k * h)].to(
      torch.float
    ).cpu().numpy()
    contextToValues.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: txt_attn_v_weight))
    )
    contextToValues.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: txt_attn_v_bias)))
    let txt_attn_key_norm_scale = state_dict["\(prefix).txt_attn_k_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    normAddedK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_key_norm_scale)))
    let txt_attn_query_norm_scale = state_dict["\(prefix).txt_attn_q_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    normAddedQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_query_norm_scale)))
    let img_attn_q_weight = state_dict["\(prefix).img_attn_qkv.weight"][..<(k * h), ...].to(
      torch.float
    ).cpu().numpy()
    let img_attn_q_bias = state_dict["\(prefix).img_attn_qkv.bias"][..<(k * h)].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_q_weight)))
    xToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_q_bias)))
    let img_attn_k_weight = state_dict["\(prefix).img_attn_qkv.weight"][(k * h)..<(2 * k * h), ...]
      .to(
        torch.float
      ).cpu().numpy()
    let img_attn_k_bias = state_dict["\(prefix).img_attn_qkv.bias"][(k * h)..<(2 * k * h)].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_k_weight)))
    xToKeys.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_k_bias)))
    let img_attn_v_weight = state_dict["\(prefix).img_attn_qkv.weight"][
      (2 * k * h)..<(3 * k * h), ...
    ].to(
      torch.float
    ).cpu().numpy()
    let img_attn_v_bias = state_dict["\(prefix).img_attn_qkv.bias"][(2 * k * h)..<(3 * k * h)].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_v_weight)))
    xToValues.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_v_bias)))
    let img_attn_key_norm_scale = state_dict["\(prefix).img_attn_k_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    normK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_key_norm_scale)))
    let img_attn_query_norm_scale = state_dict["\(prefix).img_attn_q_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_query_norm_scale)))
    if let contextUnifyheads = contextUnifyheads {
      let attn_to_add_out_weight = state_dict["\(prefix).txt_attn_proj.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_add_out_weight)))
      let attn_to_add_out_bias = state_dict["\(prefix).txt_attn_proj.bias"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.bias.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_add_out_bias)))
    }
    let attn_to_out_0_weight = state_dict["\(prefix).img_attn_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_out_0_weight)))
    let attn_to_out_0_bias = state_dict["\(prefix).img_attn_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_out_0_bias)))
    let scaleFactor: Float = upcast ? 8 : 1
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      let ff_context_linear_1_weight = state_dict["\(prefix).txt_mlp.fc1.weight"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_context_linear_1_weight)))
      let ff_context_linear_1_bias = state_dict["\(prefix).txt_mlp.fc1.bias"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.bias.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_context_linear_1_bias)))
      let ff_context_out_projection_weight =
        state_dict[
          "\(prefix).txt_mlp.fc2.weight"
        ].to(
          torch.float
        ).cpu().numpy()
      contextOutProjection.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_context_out_projection_weight)))
      let ff_context_out_projection_bias =
        ((1 / scaleFactor).pythonObject
        * state_dict[
          "\(prefix).txt_mlp.fc2.bias"
        ].to(
          torch.float
        ).cpu()).numpy()
      contextOutProjection.bias.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_context_out_projection_bias)))
    }
    let ff_linear_1_weight = state_dict["\(prefix).img_mlp.fc1.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_linear_1_weight)))
    let ff_linear_1_bias = state_dict["\(prefix).img_mlp.fc1.bias"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_linear_1_bias)))
    let ff_out_projection_weight =
      state_dict["\(prefix).img_mlp.fc2.weight"].to(
        torch.float
      ).cpu().numpy()
    xOutProjection.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_out_projection_weight)))
    let ff_out_projection_bias =
      ((1 / scaleFactor).pythonObject
      * state_dict["\(prefix).img_mlp.fc2.bias"].to(
        torch.float
      ).cpu()).numpy()
    xOutProjection.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_out_projection_bias)))
    let norm1_context_linear_weight = state_dict[
      "\(prefix).txt_mod.linear.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let norm1_context_linear_bias = state_dict[
      "\(prefix).txt_mod.linear.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<(contextBlockPreOnly ? 2 : 6) {
      contextAdaLNs[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_context_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...])))
      contextAdaLNs[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_context_linear_bias[(k * h * i)..<(k * h * (i + 1))])))
    }
    let norm1_linear_weight = state_dict["\(prefix).img_mod.linear.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let norm1_linear_bias = state_dict["\(prefix).img_mod.linear.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...])))
      xAdaLNs[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_linear_bias[(k * h * i)..<(k * h * (i + 1))])))
    }
  }
  if !contextBlockPreOnly {
    return (reader, Model([x, context, c, rot], [xOut, contextOut]))
  } else {
    return (reader, Model([x, context, c, rot], [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let c = Input()
  let rot = Input()
  let xAdaLNs = (0..<3).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  let queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xQ, right: rot)
  let keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xK, right: rot)
  let values = xV
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xLinear1 = Dense(count: k * h * 4, name: "x_linear1")
  let xOutProjection = Dense(count: k * h, name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + xChunks[2].to(of: xIn) .* out.to(of: xIn)
  let reader: (PythonObject) -> Void = { state_dict in
    let q_weight = state_dict["\(prefix).linear1.weight"][..<(k * h), ...].to(
      torch.float
    ).cpu().numpy()
    let q_bias = state_dict["\(prefix).linear1.bias"][..<(k * h)].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    xToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_bias)))
    let k_weight = state_dict["\(prefix).linear1.weight"][(k * h)..<(2 * k * h), ...].to(
      torch.float
    ).cpu().numpy()
    let k_bias = state_dict["\(prefix).linear1.bias"][(k * h)..<(2 * k * h)].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: k_weight)))
    xToKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_bias)))
    let v_weight = state_dict["\(prefix).linear1.weight"][(2 * k * h)..<(3 * k * h), ...].to(
      torch.float
    ).cpu().numpy()
    let v_bias = state_dict["\(prefix).linear1.bias"][(2 * k * h)..<(3 * k * h)].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: v_weight)))
    xToValues.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: v_bias)))
    let linear1_weight = state_dict["\(prefix).linear1.weight"].to(
      torch.float
    ).cpu().numpy()
    let linear1_bias = state_dict["\(prefix).linear1.bias"].to(
      torch.float
    ).cpu().numpy()
    let key_norm_scale = state_dict["\(prefix).k_norm.weight"].to(torch.float).cpu().numpy()
    normK.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: key_norm_scale)))
    let query_norm_scale = state_dict["\(prefix).q_norm.weight"].to(torch.float).cpu()
      .numpy()
    normQ.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: query_norm_scale)))
    xLinear1.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: linear1_weight[(3 * k * h)..<(7 * k * h), ...])))
    xLinear1.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: linear1_bias[(3 * k * h)..<(7 * k * h)])))
    let linear2_weight = state_dict["\(prefix).linear2.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: linear2_weight[..., 0..<(k * h)])))
    xOutProjection.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: linear2_weight[..., (k * h)..<(k * h * 5)])))
    let linear2_bias = state_dict["\(prefix).linear2.bias"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: linear2_bias)))
    let norm1_linear_weight = state_dict["\(prefix).modulation.linear.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let norm1_linear_bias = state_dict["\(prefix).modulation.linear.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    for i in 0..<3 {
      xAdaLNs[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...])))
      xAdaLNs[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_linear_bias[(k * h * i)..<(k * h * (i + 1))])))
    }
  }
  return (reader, Model([x, c, rot], [out]))
}

func Hunyuan(time: Int, height: Int, width: Int, textLength: Int) -> (Model, (PythonObject) -> Void)
{
  let x = Input()
  let rot = Input()
  let imgIn = Dense(count: 3072, name: "x_embedder")
  let txt = Input()
  let t = Input()
  let vector = Input()
  let guidanceEmbed = Input()
  let (tMlp0, tMlp2, timeEmbedder) = MLPEmbedder(channels: 3_072, name: "txt_in_t")
  var c = txt.reduced(.mean, axis: [1])
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: 3_072, name: "c")
  c = timeEmbedder(t) + contextEmbedder(c)
  c = c.reshaped([1, 1, 3072]).swish()
  let inputEmbedder = Dense(count: 3_072, name: "input_embedder")
  var context = inputEmbedder(txt)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<2 {
    let (block, reader) = IndividualRefinerBlock(
      prefix: "txt_in.individual_token_refiner.blocks.\(i)", t: textLength)
    context = block(context, c)
    readers.append(reader)
  }
  context = context.to(.Float32)
  var out = imgIn(x).to(.Float32)
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(channels: 3_072, name: "t")
  let (vMlp0, vMlp2, vectorIn) = MLPEmbedder(channels: 3_072, name: "vector")
  let (gMlp0, gMlp2, guidanceIn) = MLPEmbedder(channels: 3_072, name: "guidance")
  var vec = timeIn(t) + vectorIn(vector) + guidanceIn(guidanceEmbed)
  vec = vec.reshaped([1, 1, 3072]).swish()
  let h = height / 2
  let w = width / 2
  for i in 0..<20 {
    let (reader, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: time * h * w,
      contextBlockPreOnly: false, upcast: true)
    let blockOut = block(out, context, vec, rot)
    out = blockOut[0]
    context = blockOut[1]
    readers.append(reader)
  }
  let rot2 = Input()
  out = Functional.concat(axis: 1, out, context)
  for i in 0..<40 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: time * h * w,
      contextBlockPreOnly: i == 39)
    out = block(out, vec, rot2)
    readers.append(reader)
  }
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_proj_weight = state_dict["img_in.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_in_proj_bias = state_dict["img_in.proj.bias"].to(torch.float)
      .cpu().numpy()
    imgIn.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_in_proj_weight)))
    imgIn.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_in_proj_bias)))
    let input_embedder_weight = state_dict["txt_in.input_embedder.weight"].to(
      torch.float
    ).cpu().numpy()
    let input_embedder_bias = state_dict["txt_in.input_embedder.bias"].to(torch.float)
      .cpu().numpy()
    inputEmbedder.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: input_embedder_weight)))
    inputEmbedder.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: input_embedder_bias)))
    let t_embedder_mlp_0_weight = state_dict["txt_in.t_embedder.mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_0_bias = state_dict["txt_in.t_embedder.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    tMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight)))
    tMlp0.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias)))
    let t_embedder_mlp_2_weight = state_dict["txt_in.t_embedder.mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_2_bias = state_dict["txt_in.t_embedder.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    tMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight)))
    tMlp2.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias)))
    let c_embedder_linear_1_weight = state_dict["txt_in.c_embedder.linear_1.weight"].to(
      torch.float
    ).cpu().numpy()
    let c_embedder_linear_1_bias = state_dict["txt_in.c_embedder.linear_1.bias"].to(torch.float)
      .cpu().numpy()
    cLinear1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: c_embedder_linear_1_weight)))
    cLinear1.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: c_embedder_linear_1_bias)))
    let c_embedder_linear_2_weight = state_dict["txt_in.c_embedder.linear_2.weight"].to(
      torch.float
    ).cpu().numpy()
    let c_embedder_linear_2_bias = state_dict["txt_in.c_embedder.linear_2.bias"].to(torch.float)
      .cpu().numpy()
    cLinear2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: c_embedder_linear_2_weight)))
    cLinear2.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: c_embedder_linear_2_bias)))
    let time_in_mlp_0_weight = state_dict["time_in.mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let time_in_mlp_0_bias = state_dict["time_in.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    timeInMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: time_in_mlp_0_weight)))
    timeInMlp0.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: time_in_mlp_0_bias)))
    let time_in_mlp_2_weight = state_dict["time_in.mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let time_in_mlp_2_bias = state_dict["time_in.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    timeInMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: time_in_mlp_2_weight)))
    timeInMlp2.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: time_in_mlp_2_bias)))
    let vector_in_in_layer_weight = state_dict["vector_in.in_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    let vector_in_in_layer_bias = state_dict["vector_in.in_layer.bias"].to(torch.float)
      .cpu().numpy()
    vMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vector_in_in_layer_weight)))
    vMlp0.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vector_in_in_layer_bias)))
    let vector_in_out_layer_weight = state_dict["vector_in.out_layer.weight"].to(
      torch.float
    ).cpu().numpy()
    let vector_in_out_layer_bias = state_dict["vector_in.out_layer.bias"].to(torch.float)
      .cpu().numpy()
    vMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vector_in_out_layer_weight)))
    vMlp2.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vector_in_out_layer_bias)))
    let guidance_in_mlp_0_weight = state_dict["guidance_in.mlp.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let guidance_in_mlp_0_bias = state_dict["guidance_in.mlp.0.bias"].to(torch.float)
      .cpu().numpy()
    gMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: guidance_in_mlp_0_weight)))
    gMlp0.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: guidance_in_mlp_0_bias)))
    let guidance_in_mlp_2_weight = state_dict["guidance_in.mlp.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let guidance_in_mlp_2_bias = state_dict["guidance_in.mlp.2.bias"].to(torch.float)
      .cpu().numpy()
    gMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: guidance_in_mlp_2_weight)))
    gMlp2.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: guidance_in_mlp_2_bias)))
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_linear_weight = state_dict["final_layer.adaLN_modulation.1.weight"].to(torch.float)
      .cpu().numpy()
    let norm_out_linear_bias = state_dict["final_layer.adaLN_modulation.1.bias"].to(torch.float)
      .cpu().numpy()
    shift.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_out_linear_weight[0..<3072, ...])))
    scale.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: norm_out_linear_weight[3072..<(3072 * 2), ...])))
    shift.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_out_linear_bias[0..<3072])))
    scale.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: norm_out_linear_bias[3072..<(3072 * 2)])))
    let proj_out_weight = state_dict["final_layer.linear.weight"].to(
      torch.float
    ).cpu().numpy()
    projOut.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_out_weight)))
    let proj_out_bias = state_dict["final_layer.linear.bias"].to(
      torch.float
    ).cpu().numpy()
    projOut.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_out_bias)))
  }
  return (Model([x, rot, rot2, txt, t, vector, guidanceEmbed], [out]), reader)
}

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
    95..<106, 0..<4096
  ].reshaped(.HWC(1, 11, 4096)).toGPU(2)  // We don't need attention mask, just reduce the hidden states.
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/llava_llama_3_8b_v1.1_f16.ckpt") {
    $0.write("llava", model: transformer)
  }
  */
  debugPrint(lastHiddenStates)
  let timestep = timeEmbedding(timesteps: 900, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  debugPrint(timestep)
  let transformer_state_dict = hunyuan_video_sampler.pipeline.transformer.state_dict()
  var rotNdTensor = graph.variable(.CPU, .NHWC(1, 33 * 34 * 60, 1, 128), of: Float.self)
  var rotNdTensor2 = graph.variable(.CPU, .NHWC(1, 33 * 34 * 60 + 11, 1, 128), of: Float.self)
  for t in 0..<33 {
    for y in 0..<34 {
      for x in 0..<60 {
        let i = t * 34 * 60 + y * 60 + x
        for k in 0..<8 {
          let theta = Double(t) * 1.0 / pow(256, Double(k) / 8)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, k * 2] = Float(costheta)
          rotNdTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
          rotNdTensor2[0, i, 0, k * 2] = Float(costheta)
          rotNdTensor2[0, i, 0, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<28 {
          let theta = Double(y) * 1.0 / pow(256, Double(k) / 28)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, (k + 8) * 2] = Float(costheta)
          rotNdTensor[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
          rotNdTensor2[0, i, 0, (k + 8) * 2] = Float(costheta)
          rotNdTensor2[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<28 {
          let theta = Double(x) * 1.0 / pow(256, Double(k) / 28)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
          rotNdTensor[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
          rotNdTensor2[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
          rotNdTensor2[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  for i in 33 * 34 * 60..<(33 * 34 * 60 + 11) {
    for k in 0..<64 {
      rotNdTensor2[0, i, 0, k * 2] = 1
      rotNdTensor2[0, i, 0, k * 2 + 1] = 0
    }
  }
  let (hunyuan, hunyuanReader) = Hunyuan(time: 33, height: 68, width: 120, textLength: 11)
  let tGPU = graph.variable(Tensor<Float16>(from: timestep)).toGPU(2)
  let xTensor = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy())).toGPU(2)
  ).reshaped(format: .NHWC, shape: [1, 16, 33 * 34, 2, 60, 2]).permuted(0, 2, 4, 1, 3, 5).copied()
    .reshaped(format: .NHWC, shape: [1, 33 * 34 * 60, 16 * 2 * 2])
  let guidanceEmbed = timeEmbedding(
    timesteps: 3500, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let gGPU = graph.variable(Tensor<Float16>(from: guidanceEmbed)).toGPU(2)
  let vector = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: text_states_2.to(torch.float).cpu().numpy()))
      .reshaped(.WC(1, 768)).toGPU(2))
  let rotNdTensorGPU = DynamicGraph.Tensor<Float16>(from: rotNdTensor).toGPU(2)
  let rotNdTensor2GPU = DynamicGraph.Tensor<Float16>(from: rotNdTensor2).toGPU(2)
  hunyuan.compile(
    inputs: xTensor, rotNdTensorGPU, rotNdTensor2GPU, lastHiddenStates, tGPU, vector, gGPU)
  hunyuanReader(transformer_state_dict)
  debugPrint(
    hunyuan(inputs: xTensor, rotNdTensorGPU, rotNdTensor2GPU, lastHiddenStates, tGPU, vector, gGPU))
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/hunyuan_video_t2v_720p_f16.ckpt") {
    $0.write("dit", model: hunyuan)
  }
  */
}
/*

func ResnetBlockCausal3D(
  prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool, depth: Int, height: Int,
  width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  var out = norm1(x.reshaped([inChannels, depth, height, width])).reshaped([
    1, inChannels, depth, height, width,
  ])
  out = Swish()(out)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  out = conv1(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let norm2 = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  out = norm2(out.reshaped([outChannels, depth, height, width])).reshaped([
    1, outChannels, depth, height, width,
  ])
  out = Swish()(out)
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  out = conv2(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
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
    let conv1_weight = state_dict["\(prefix).conv1.conv.weight"].to(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.conv.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2_bias = state_dict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["\(prefix).conv2.conv.weight"].to(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.conv.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).conv_shortcut.conv.weight"].to(torch.float)
        .cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).conv_shortcut.conv.bias"].to(torch.float).cpu()
        .numpy()
      ninShortcut.weight.copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
  }
  return (reader, Model([x], [out]))
}

func AttnBlockCausal3D(
  prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let causalAttentionMask = Input()
  let norm = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  var out = norm(x.reshaped([inChannels, depth, height, width])).reshaped([
    1, inChannels, depth, height, width,
  ])
  let hw = width * height * depth
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
  let k = tokeys(out).reshaped([1, inChannels, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    1, inChannels, hw,
  ])
  var dot =
    Matmul(transposeA: (1, 2))(q, k).reshaped([
      depth, height * width, depth, height * width,
    ]) + causalAttentionMask
  dot = dot.reshaped([hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([1, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
  let v = tovalues(out).reshaped([1, inChannels, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]))
  out = x + projOut(out.reshaped([1, inChannels, depth, height, width]))
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_weight = state_dict["\(prefix).group_norm.weight"].to(torch.float).cpu().numpy()
    let norm_bias = state_dict["\(prefix).group_norm.bias"].to(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: norm_bias))
    let k_weight = state_dict["\(prefix).to_k.weight"].to(torch.float).cpu().numpy()
    let k_bias = state_dict["\(prefix).to_k.bias"].to(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["\(prefix).to_q.weight"].to(torch.float).cpu().numpy()
    let q_bias = state_dict["\(prefix).to_q.bias"].to(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["\(prefix).to_v.weight"].to(torch.float).cpu().numpy()
    let v_bias = state_dict["\(prefix).to_v.bias"].to(torch.float).cpu().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["\(prefix).to_out.0.weight"].to(torch.float).cpu().numpy()
    let proj_out_bias = state_dict["\(prefix).to_out.0.bias"].to(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x, causalAttentionMask], [out]))
}

func EncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  let causalAttentionMask = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  var out = convIn(x.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  var readers = [(PythonObject) -> Void]()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "encoder.down_blocks.\(i).resnets.\(j)", inChannels: previousChannel,
        outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
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
      let strideZ: Int
      if i > 0 {
        depth = (depth - 1) / 2 + 1
        strideZ = 2
      } else {
        strideZ = 1
      }
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3, 3],
        hint: Hint(
          stride: [strideZ, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
      out = conv2d(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
      let downLayer = i
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict[
          "encoder.down_blocks.\(downLayer).downsamplers.0.conv.conv.weight"
        ].to(
          torch.float
        ).cpu()
          .numpy()
        let conv_bias = state_dict["encoder.down_blocks.\(downLayer).downsamplers.0.conv.conv.bias"]
          .to(torch.float)
          .cpu().numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlockCausal3D(
    prefix: "encoder.mid_block.attentions.0", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out, causalAttentionMask)
  let (midBlockReader2, midBlock2) = ResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3])
  out = convOut(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let quantConv = Convolution(groups: 1, filters: 32, filterSize: [1, 1, 1])
  out = quantConv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["encoder.conv_in.conv.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["encoder.conv_in.conv.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    let norm_out_weight = state_dict["encoder.conv_norm_out.weight"].to(torch.float).cpu().numpy()
    let norm_out_bias = state_dict["encoder.conv_norm_out.bias"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["encoder.conv_out.conv.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["encoder.conv_out.conv.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
    let quant_conv_weight = state_dict["quant_conv.weight"].to(torch.float).cpu().numpy()
    let quant_conv_bias = state_dict["quant_conv.bias"].to(torch.float).cpu().numpy()
    quantConv.weight.copy(from: try! Tensor<Float>(numpy: quant_conv_weight))
    quantConv.bias.copy(from: try! Tensor<Float>(numpy: quant_conv_bias))
  }
  return (reader, Model([x, causalAttentionMask], [out]))
}

func DecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  let causalAttentionMask = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(groups: 1, filters: 16, filterSize: [1, 1, 1])
  var out = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  out = convIn(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let (midBlockReader1, midBlock1) = ResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlockCausal3D(
    prefix: "decoder.mid_block.attentions.0", inChannels: previousChannel, depth: startDepth,
    height: startHeight, width: startWidth)
  out = midAttn1(out, causalAttentionMask)
  let (midBlockReader2, midBlock2) = ResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock2(out)
  var width = startWidth
  var height = startHeight
  var depth = startDepth
  var readers = [(PythonObject) -> Void]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)",
        inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
        out.reshaped([channel, depth, height, width])
      ).reshaped([1, channel, depth, height * 2, width * 2])
      width *= 2
      height *= 2
      if i < channels.count - 1 {  // Scale time too.
        let first = out.reshaped(
          [channel, 1, height * width], strides: [depth * height * width, height * width, 1]
        ).contiguous()
        let more = out.reshaped(
          [channel, (depth - 1), 1, height * width], offset: [0, 1, 0, 0],
          strides: [depth * height * width, height * width, height * width, 1]
        ).contiguous()
        out = Functional.concat(
          axis: 1, first,
          Functional.concat(axis: 2, more, more).reshaped([
            channel, (depth - 1) * 2, height * width,
          ]))
        depth = 1 + (depth - 1) * 2
        out = out.reshaped([1, channel, depth, height, width])
      }
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
      out = conv2d(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
      let upLayer = channels.count - 1 - i
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict["decoder.up_blocks.\(upLayer).upsamplers.0.conv.conv.weight"]
          .to(torch.float)
          .cpu().numpy()
        let conv_bias = state_dict["decoder.up_blocks.\(upLayer).upsamplers.0.conv.conv.bias"].to(
          torch.float
        ).cpu()
          .numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let normOut = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3])
  out = normOut(out.reshaped([channels[0], depth, height, width])).reshaped([
    1, channels[0], depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])))
  out = convOut(out.padded(.replication, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let reader: (PythonObject) -> Void = { state_dict in
    let post_quant_conv_weight = state_dict["post_quant_conv.weight"].to(torch.float).cpu().numpy()
    let post_quant_conv_bias = state_dict["post_quant_conv.bias"].to(torch.float).cpu().numpy()
    postQuantConv.weight.copy(from: try! Tensor<Float>(numpy: post_quant_conv_weight))
    postQuantConv.bias.copy(from: try! Tensor<Float>(numpy: post_quant_conv_bias))
    let conv_in_weight = state_dict["decoder.conv_in.conv.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["decoder.conv_in.conv.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["decoder.conv_norm_out.weight"].to(torch.float).cpu().numpy()
    let norm_out_bias = state_dict["decoder.conv_norm_out.bias"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["decoder.conv_out.conv.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["decoder.conv_out.conv.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x, causalAttentionMask], [out]))
}

graph.withNoGrad {
  let vae_state_dict = vae.state_dict()
  print(vae_state_dict.keys())
  var zTensor = graph.variable(try! Tensor<Float>(numpy: z.to(torch.float).cpu().numpy())).toGPU(1)
  let (decoderReader, decoder) = DecoderCausal3D(
    channels: [128, 256, 512, 512], numRepeat: 2, startWidth: 32, startHeight: 32, startDepth: 17)
  let causalAttentionMask = graph.variable(Tensor<Float>(.CPU, .NCHW(17, 1, 17, 1)))
  causalAttentionMask.full(0)
  for i in 0..<16 {
    for j in (i + 1)..<17 {
      causalAttentionMask[i, 0, j, 0] = -Float.greatestFiniteMagnitude
    }
  }
  let causalAttentionMaskGPU = causalAttentionMask.toGPU(1)
  decoder.compile(inputs: zTensor, causalAttentionMaskGPU)
  decoderReader(vae_state_dict)
  let image = decoder(inputs: zTensor, causalAttentionMaskGPU)[0].as(of: Float.self)
  let (encoderReader, encoder) = EncoderCausal3D(
    channels: [128, 256, 512, 512], numRepeat: 2, startWidth: 32, startHeight: 32, startDepth: 17)
  encoder.compile(inputs: image, causalAttentionMaskGPU)
  encoderReader(vae_state_dict)
  let x = encoder(inputs: image, causalAttentionMaskGPU)[0].as(of: Float.self)
  debugPrint(x)
  graph.openStore("/home/liu/workspace/swift-diffusion/hunyuan_video_vae_f32.ckpt") {
    $0.write("decoder", model: decoder)
    $0.write("encoder", model: encoder)
  }
}
*/
