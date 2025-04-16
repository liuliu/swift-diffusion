import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let graph = DynamicGraph()

let torch = Python.import("torch")

torch.set_grad_enabled(false)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let hi_diffusers = Python.import("hi_diffusers")
let transformers = Python.import("transformers")
let hi_diffusers_schedulers_fm_solvers_unipc = Python.import(
  "hi_diffusers.schedulers.fm_solvers_unipc")

let pretrained_model_name_or_path = "HiDream-ai/HiDream-I1-Full"
let scheduler = hi_diffusers_schedulers_fm_solvers_unipc.FlowUniPCMultistepScheduler(
  num_train_timesteps: 1000, shift: 3.0, use_dynamic_shifting: false)

let tokenizer_4 = transformers.PreTrainedTokenizerFast.from_pretrained(
  "meta-llama/Meta-Llama-3.1-8B-Instruct",
  use_fast: false)
tokenizer_4.pad_token = tokenizer_4.eos_token

let text_encoder_4 = transformers.LlamaForCausalLM.from_pretrained(
  "meta-llama/Meta-Llama-3.1-8B-Instruct",
  output_hidden_states: true,
  output_attentions: true,
  torch_dtype: torch.float16
).to("cuda")  // torch.bfloat16).to("cpu")

let generator = torch.Generator("cuda").manual_seed(42)
let prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"."

let text_inputs = tokenizer_4(
  prompt, padding: "max_length", max_length: 128, truncation: true, add_special_tokens: true,
  return_tensors: "pt")
print(text_inputs.input_ids)
print(text_inputs.attention_mask)

let outputs = text_encoder_4(
  text_inputs.input_ids.to("cuda"), text_inputs.attention_mask.to("cuda"),
  output_hidden_states: true, output_attentions: true)
print(outputs.hidden_states)
/*
let transformer = hi_diffusers.HiDreamImageTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder: "transformer",
    torch_dtype: torch.bfloat16).to("cuda")

let pipe = hi_diffusers.HiDreamImagePipeline.from_pretrained(
    pretrained_model_name_or_path,
    scheduler: scheduler,
    tokenizer_4: tokenizer_4,
    text_encoder_4: text_encoder_4,
    torch_dtype: torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.transformer = transformer

print(transformer)

let image = pipe(
    prompt,
    height: 1024,
    width: 1024,
    guidance_scale: 5,
    num_inference_steps: 5,
    num_images_per_prompt: 1,
    generator: generator
).images[0]

image.save("/home/liu/workspace/swift-diffusion/output.png")
*/
let tokenizer = GPT2Tokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/hunyuan/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/hunyuan/merges.txt",
  specialTokens: [
    "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
    "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
  ])

let result = tokenizer.tokenize(text: prompt, addSpecialTokens: true)

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k]).transposed(1, 2)
  queries = ((1.0 / Float(k).squareRoot()) * Functional.cmul(left: queries, right: rot)).transposed(
    1, 2)
  keys = Functional.cmul(left: keys, right: rot).transposed(1, 2)
  var outs = [Model.IO]()
  for i in 0..<hk {
    let query = queries.reshaped(
      [b, h / hk, t, k], offset: [0, i * (h / hk), 0, 0], strides: [h * t * k, t * k, k, 1])
    let key = keys.reshaped(
      [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1])
    let value = values.reshaped(
      [b, 1, t, k], offset: [0, i, 0, 0], strides: [hk * t * k, t * k, k, 1])
    var dot = Matmul(transposeB: (2, 3))(query, key) + causalAttentionMask
    if let last = outs.last {
      dot.add(dependencies: [last])
    }
    dot = dot.reshaped([b * (h / hk) * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h / hk, t, t])
    let out = dot * value
    outs.append(out)
  }
  var out = Concat(axis: 1)(outs).reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
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
  return (Model([x, rot, causalAttentionMask], [out]), reader)
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
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot, causalAttentionMask) + x
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
  return (Model([x, rot, causalAttentionMask], [out]), reader)
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
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Bool, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let (embedding, embedReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  var hiddenStates = [Model.IO]()
  for i in 0..<layers {
    if outputHiddenStates {
      hiddenStates.append(out)
    }
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: heads / 4, b: batchSize,
      t: tokenLength,
      MLP: MLP)
    out = layer(out, rot, causalAttentionMask)
    readers.append(reader)
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
  return (Model([tokens, rot, causalAttentionMask], hiddenStates + [out]), reader)
}

graph.withNoGrad {
  let text_encoder_state_dict = text_encoder_4.model.state_dict()
  print(text_encoder_state_dict.keys())
  let (transformer, reader) = Transformer(
    Float16.self, vocabularySize: 128_256, maxLength: 128, width: 4_096, tokenLength: 128,
    layers: 32, MLP: 14336, heads: 32, outputHiddenStates: true, batchSize: 1)
  let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [128], of: Int32.self)
  for i in 0..<result.count {
    tokensTensor[i] = result[i]
  }
  for i in result.count..<128 {
    tokensTensor[i] = 128009
  }
  let rotTensor = graph.variable(.CPU, .NHWC(1, 128, 1, 128), of: Float.self)
  let invFreqLlama = (0..<64).map { k in
    let lowFreqWavelen = Double(8_192) / 1.0
    let highFreqWavelen = Double(8_192) / 4.0
    let invFreq = 1.0 / pow(500_000, Double(k) * 2 / 128)
    let wavelen = 2.0 * .pi / invFreq
    var invFreqLlama: Double
    if wavelen > lowFreqWavelen {
      invFreqLlama = invFreq / 8.0
    } else {
      invFreqLlama = invFreq
    }
    let smoothFactor = (Double(8_192) / wavelen - 1.0) / (4.0 - 1.0)
    let smoothInvFreq = (1 - smoothFactor) * invFreqLlama / 8.0 + smoothFactor * invFreqLlama
    if wavelen >= highFreqWavelen && wavelen <= lowFreqWavelen {
      invFreqLlama = smoothInvFreq
    }
    return invFreqLlama
  }
  for i in 0..<128 {
    for k in 0..<64 {
      let theta = Double(i) * invFreqLlama[k]
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let causalAttentionMask = graph.variable(Tensor<Float16>(.CPU, .NHWC(1, 1, 128, 128)))
  causalAttentionMask.full(0)
  for i in 0..<128 {
    for j in min(i + 1, result.count)..<128 {
      causalAttentionMask[0, 0, i, j] = -Float16.greatestFiniteMagnitude
    }
  }
  let tokensTensorGPU = tokensTensor.toGPU(1)
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(1)
  let causalAttentionMaskGPU = causalAttentionMask.toGPU(1)
  transformer.compile(inputs: tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU)
  reader(text_encoder_state_dict)
  let outputHiddenStates = transformer(
    inputs: tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU)
  debugPrint(outputHiddenStates)
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/llama_3.1_8b_instruct_f16.ckpt") {
    $0.write("text_model", model: transformer)
  }
  */
}
