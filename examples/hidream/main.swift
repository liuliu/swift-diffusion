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
  torch_dtype: torch.bfloat16
).to("cpu")  // torch.float16).to("cuda")

let generator = torch.Generator("cuda").manual_seed(42)
let prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"."
/*
let text_inputs = tokenizer_4(
  prompt, padding: "max_length", max_length: 128, truncation: true, add_special_tokens: true,
  return_tensors: "pt")
print(text_inputs.input_ids)
print(text_inputs.attention_mask)

let outputs = text_encoder_4(
  text_inputs.input_ids.to("cuda"), text_inputs.attention_mask.to("cuda"),
  output_hidden_states: true, output_attentions: true)
print(outputs.hidden_states)
*/
let transformer = hi_diffusers.HiDreamImageTransformer2DModel.from_pretrained(
  pretrained_model_name_or_path,
  subfolder: "transformer",
  torch_dtype: torch.float
).to("cpu")

let x = torch.randn([1, 16, 128, 128], dtype: torch.bfloat16).cuda()
let prompt_embed_3 = torch.randn([1, 128, 4096], dtype: torch.bfloat16).cuda()
let prompt_embed_4 = torch.randn([32, 1, 128, 4096], dtype: torch.bfloat16).cuda()
let pooled_prompt_embed = torch.randn([1, 2048], dtype: torch.bfloat16).cuda()
let t = torch.full([1], 1000).to(torch.bfloat16).cuda()

let output = transformer(
  hidden_states: x, timesteps: t, encoder_hidden_states: [prompt_embed_3, prompt_embed_4],
  pooled_embeds: pooled_prompt_embed, return_dict: false)

/*
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
  /*
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
  graph.openStore("/home/liu/workspace/swift-diffusion/llama_3.1_8b_instruct_f16.ckpt") {
    $0.write("text_model", model: transformer)
  }
  */
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

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w1")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_w3")
  var out = w1(x).swish() .* w3(x)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (w1, w2, w3, Model([x], [out]))
}

func MoEFeedForward(
  segments: Int, tokenLength: Int, hiddenSize: Int, intermediateSize: Int, upcast: Bool,
  name: String
) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: segments, noBias: true, name: "\(name)_gate")
  let route = gate(x).partitioned(kth: 2, axis: 2, descending: true)
  var weights = route[0].reshaped([tokenLength, 2]).softmax().reshaped([tokenLength * 2])
  let experts = route[1].reshaped([tokenLength * 2])  // This is to select into experts.
  let sort = experts.sorted(axis: 0, descending: false)
  weights = IndexSelect()(weights, sort[1])  // Reorder the weights by the sorting order.
  let expertIds = sort[0].uniqueConsecutive(bincount: segments)
  let indices = 0.5 * sort[1]  // Scale it to 0..<tokenLength.
  let gathered = IndexSelect()(x.reshaped([tokenLength, hiddenSize]), indices)
  let w1 = SegmentedDense(
    segments: segments, count: intermediateSize, noBias: true, name: "\(name)_w1")
  let w3 = SegmentedDense(
    segments: segments, count: intermediateSize, noBias: true, name: "\(name)_w3")
  var out = w1(gathered, expertIds).swish() .* w3(gathered, expertIds)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let w2 = SegmentedDense(segments: segments, count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out, expertIds)
  // Out is tokenLength * 2, now multiply weights and scale back.
  out = out .* weights.reshaped([tokenLength * 2, 1])
  out = Functional.scatterAdd(bincount: tokenLength, out, index: indices)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (gate, w1, w2, w3, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool
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
  var contextK = contextToKeys(contextOut)
  let normAddedK = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_k")
  contextK = normAddedK(contextK).reshaped([b, t.1, h, k])
  var contextQ = contextToQueries(contextOut)
  let normAddedQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_q")
  contextQ = normAddedQ(contextQ).reshaped([b, t.1, h, k])
  let contextV = contextToValues(contextOut).reshaped([b, t.1, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  let values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    queries, keys, values
  ).reshaped([b, t.1 + hw, h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t.0, h * k], offset: [0, hw, 0], strides: [(t.1 + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut =
      context.reshaped([b, t.0, h * k], strides: [t.1 * h * k, h * k, 1]).contiguous()
      + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: 6912, upcast: upcast, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + ((upcast ? contextChunks[5].to(of: contextOut) : contextChunks[5])
      .* contextFF(
        contextNorm2(contextOut).to(.Float16) .* (1 + contextChunks[4]) + contextChunks[3])).to(
        of: contextOut)
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: 3584, upcast: upcast, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = MoEFeedForward(
    segments: 4, tokenLength: hw, hiddenSize: k * h, intermediateSize: 6912, upcast: upcast,
    name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xIn = xNorm2(xOut).to(.Float16) .* (1 + xChunks[4]) + xChunks[3]
  xOut =
    xOut
    + ((upcast ? xChunks[5].to(of: xOut) : xChunks[5])
    .* (xSharedFF(xIn) + xMoEFF(xIn))).to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
    let txt_attn_q_weight = state_dict["\(prefix).attn1.to_q_t.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_q_bias = state_dict["\(prefix).attn1.to_q_t.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_q_weight)))
    contextToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_q_bias)))
    let txt_attn_k_weight = state_dict["\(prefix).attn1.to_k_t.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let txt_attn_k_bias = state_dict["\(prefix).attn1.to_k_t.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToKeys.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: txt_attn_k_weight)))
    contextToKeys.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: txt_attn_k_bias)))
    let txt_attn_v_weight = state_dict["\(prefix).attn1.to_v_t.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_v_bias = state_dict["\(prefix).attn1.to_v_t.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToValues.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: txt_attn_v_weight))
    )
    contextToValues.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: txt_attn_v_bias)))
    let txt_attn_key_norm_scale = state_dict["\(prefix).attn1.k_rms_norm_t.weight"].to(
      torch.float
    ).cpu().numpy()
    normAddedK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_key_norm_scale)))
    let txt_attn_query_norm_scale = state_dict["\(prefix).attn1.q_rms_norm_t.weight"].to(
      torch.float
    ).cpu().numpy()
    normAddedQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_query_norm_scale)))
    let img_attn_q_weight = state_dict["\(prefix).attn1.to_q.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_q_bias = state_dict["\(prefix).attn1.to_q.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_q_weight)))
    xToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_q_bias)))
    let img_attn_k_weight = state_dict["\(prefix).attn1.to_k.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let img_attn_k_bias = state_dict["\(prefix).attn1.to_k.bias"].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_k_weight)))
    xToKeys.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_k_bias)))
    let img_attn_v_weight = state_dict["\(prefix).attn1.to_v.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_v_bias = state_dict["\(prefix).attn1.to_v.bias"].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_v_weight)))
    xToValues.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_v_bias)))
    let img_attn_key_norm_scale = state_dict["\(prefix).attn1.k_rms_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    normK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_key_norm_scale)))
    let img_attn_query_norm_scale = state_dict["\(prefix).attn1.q_rms_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_query_norm_scale)))
    if let contextUnifyheads = contextUnifyheads {
      let attn_to_add_out_weight = state_dict["\(prefix).attn1.to_out_t.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_add_out_weight)))
      let attn_to_add_out_bias = state_dict["\(prefix).attn1.to_out_t.bias"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.bias.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_add_out_bias)))
    }
    let attn_to_out_0_weight = state_dict["\(prefix).attn1.to_out.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_out_0_weight)))
    let attn_to_out_0_bias = state_dict["\(prefix).attn1.to_out.bias"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_out_0_bias)))
    let ada_ln_weight = state_dict[
      "\(prefix).adaLN_modulation.1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let ada_ln_bias = state_dict[
      "\(prefix).adaLN_modulation.1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: ada_ln_weight[(k * h * i)..<(k * h * (i + 1)), ...])))
      xAdaLNs[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: ada_ln_bias[(k * h * i)..<(k * h * (i + 1))])))
    }
    for i in 0..<(contextBlockPreOnly ? 2 : 6) {
      contextAdaLNs[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: ada_ln_weight[(k * h * (i + 6))..<(k * h * (i + 7)), ...])))
      contextAdaLNs[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: ada_ln_bias[(k * h * (i + 6))..<(k * h * (i + 7))])))
    }
    if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
      let ff_t_w1_weight = state_dict["\(prefix).ff_t.w1.weight"].to(
        torch.float
      ).cpu().numpy()
      contextW1.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_t_w1_weight)))
      let ff_t_w2_weight = state_dict["\(prefix).ff_t.w2.weight"].to(
        torch.float
      ).cpu().numpy()
      contextW2.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_t_w2_weight)))
      let ff_t_w3_weight = state_dict["\(prefix).ff_t.w3.weight"].to(
        torch.float
      ).cpu().numpy()
      contextW3.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_t_w3_weight)))
    }
    let ff_i_shared_w1_weight = state_dict["\(prefix).ff_i.shared_experts.w1.weight"].to(
      torch.float
    ).cpu().numpy()
    xSharedW1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_shared_w1_weight)))
    let ff_i_shared_w2_weight = state_dict["\(prefix).ff_i.shared_experts.w2.weight"].to(
      torch.float
    ).cpu().numpy()
    xSharedW2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_shared_w2_weight)))
    let ff_i_shared_w3_weight = state_dict["\(prefix).ff_i.shared_experts.w3.weight"].to(
      torch.float
    ).cpu().numpy()
    xSharedW3.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_shared_w3_weight)))
    let ff_i_gate_weight = state_dict["\(prefix).ff_i.gate.weight"].to(
      torch.float
    ).cpu().numpy()
    xMoEGate.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_gate_weight)))
    let ff_i_experts_0_w1_weight = state_dict["\(prefix).ff_i.experts.0.w1.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_1_w1_weight = state_dict["\(prefix).ff_i.experts.1.w1.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_2_w1_weight = state_dict["\(prefix).ff_i.experts.2.w1.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_3_w1_weight = state_dict["\(prefix).ff_i.experts.3.w1.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_w1_weight = torch.concat(
      [
        ff_i_experts_0_w1_weight, ff_i_experts_1_w1_weight, ff_i_experts_2_w1_weight,
        ff_i_experts_3_w1_weight,
      ], dim: 0
    ).numpy()
    xMoEW1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_experts_w1_weight)))
    let ff_i_experts_0_w2_weight = state_dict["\(prefix).ff_i.experts.0.w2.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_1_w2_weight = state_dict["\(prefix).ff_i.experts.1.w2.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_2_w2_weight = state_dict["\(prefix).ff_i.experts.2.w2.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_3_w2_weight = state_dict["\(prefix).ff_i.experts.3.w2.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_w2_weight = torch.concat(
      [
        ff_i_experts_0_w2_weight, ff_i_experts_1_w2_weight, ff_i_experts_2_w2_weight,
        ff_i_experts_3_w2_weight,
      ], dim: 0
    ).numpy()
    xMoEW2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_experts_w2_weight)))
    let ff_i_experts_0_w3_weight = state_dict["\(prefix).ff_i.experts.0.w3.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_1_w3_weight = state_dict["\(prefix).ff_i.experts.1.w3.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_2_w3_weight = state_dict["\(prefix).ff_i.experts.2.w3.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_3_w3_weight = state_dict["\(prefix).ff_i.experts.3.w3.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_w3_weight = torch.concat(
      [
        ff_i_experts_0_w3_weight, ff_i_experts_1_w3_weight, ff_i_experts_2_w3_weight,
        ff_i_experts_3_w3_weight,
      ], dim: 0
    ).numpy()
    xMoEW3.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_experts_w3_weight)))
  }
  if !contextBlockPreOnly {
    return (reader, Model([x, context, c, rot], [xOut, contextOut]))
  } else {
    return (reader, Model([x, context, c, rot], [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let c = Input()
  let rot = Input()
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x).to(.Float16) + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw + t.1, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw + t.1, h, k])
  let xV = xToValues(xOut).reshaped([b, hw + t.1, h, k])
  xQ = Functional.cmul(left: xQ, right: rot)
  xK = Functional.cmul(left: xK, right: rot)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    xQ, xK, xV
  ).reshaped([b, t.1 + hw, h * k])
  var xIn: Model.IO = x
  let xLength: Int
  if contextBlockPreOnly {
    xOut = out.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xIn = xIn.reshaped([b, hw, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xLength = hw
  } else {
    xOut = out.reshaped([b, hw + t.0, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xIn = xIn.reshaped([b, hw + t.0, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
    xLength = hw + t.0
  }
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  xOut = xIn + (xChunks[2] .* xOut).to(of: xIn)
  // Attentions are now. Now run MLP.
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: 3584, upcast: upcast, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = MoEFeedForward(
    segments: 4, tokenLength: xLength, hiddenSize: k * h, intermediateSize: 6912, upcast: upcast,
    name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xFFIn = xNorm2(xOut).to(.Float16) .* (1 + xChunks[4]) + xChunks[3]
  xOut =
    xOut
    + ((upcast ? xChunks[5].to(of: xOut) : xChunks[5])
    .* (xSharedFF(xFFIn) + xMoEFF(xFFIn))).to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_attn_q_weight = state_dict["\(prefix).attn1.to_q.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_q_bias = state_dict["\(prefix).attn1.to_q.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_q_weight)))
    xToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_q_bias)))
    let img_attn_k_weight = state_dict["\(prefix).attn1.to_k.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let img_attn_k_bias = state_dict["\(prefix).attn1.to_k.bias"].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_k_weight)))
    xToKeys.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_k_bias)))
    let img_attn_v_weight = state_dict["\(prefix).attn1.to_v.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_v_bias = state_dict["\(prefix).attn1.to_v.bias"].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_v_weight)))
    xToValues.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: img_attn_v_bias)))
    let img_attn_key_norm_scale = state_dict["\(prefix).attn1.k_rms_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    normK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_key_norm_scale)))
    let img_attn_query_norm_scale = state_dict["\(prefix).attn1.q_rms_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_query_norm_scale)))
    let attn_to_out_0_weight = state_dict["\(prefix).attn1.to_out.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_out_0_weight)))
    let attn_to_out_0_bias = state_dict["\(prefix).attn1.to_out.bias"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_out_0_bias)))
    let ada_ln_weight = state_dict[
      "\(prefix).adaLN_modulation.1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let ada_ln_bias = state_dict[
      "\(prefix).adaLN_modulation.1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: ada_ln_weight[(k * h * i)..<(k * h * (i + 1)), ...])))
      xAdaLNs[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: ada_ln_bias[(k * h * i)..<(k * h * (i + 1))])))
    }
    let ff_i_shared_w1_weight = state_dict["\(prefix).ff_i.shared_experts.w1.weight"].to(
      torch.float
    ).cpu().numpy()
    xSharedW1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_shared_w1_weight)))
    let ff_i_shared_w2_weight = state_dict["\(prefix).ff_i.shared_experts.w2.weight"].to(
      torch.float
    ).cpu().numpy()
    xSharedW2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_shared_w2_weight)))
    let ff_i_shared_w3_weight = state_dict["\(prefix).ff_i.shared_experts.w3.weight"].to(
      torch.float
    ).cpu().numpy()
    xSharedW3.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_shared_w3_weight)))
    let ff_i_gate_weight = state_dict["\(prefix).ff_i.gate.weight"].to(
      torch.float
    ).cpu().numpy()
    xMoEGate.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_gate_weight)))
    let ff_i_experts_0_w1_weight = state_dict["\(prefix).ff_i.experts.0.w1.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_1_w1_weight = state_dict["\(prefix).ff_i.experts.1.w1.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_2_w1_weight = state_dict["\(prefix).ff_i.experts.2.w1.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_3_w1_weight = state_dict["\(prefix).ff_i.experts.3.w1.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_w1_weight = torch.concat(
      [
        ff_i_experts_0_w1_weight, ff_i_experts_1_w1_weight, ff_i_experts_2_w1_weight,
        ff_i_experts_3_w1_weight,
      ], dim: 0
    ).numpy()
    xMoEW1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_experts_w1_weight)))
    let ff_i_experts_0_w2_weight = state_dict["\(prefix).ff_i.experts.0.w2.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_1_w2_weight = state_dict["\(prefix).ff_i.experts.1.w2.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_2_w2_weight = state_dict["\(prefix).ff_i.experts.2.w2.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_3_w2_weight = state_dict["\(prefix).ff_i.experts.3.w2.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_w2_weight = torch.concat(
      [
        ff_i_experts_0_w2_weight, ff_i_experts_1_w2_weight, ff_i_experts_2_w2_weight,
        ff_i_experts_3_w2_weight,
      ], dim: 0
    ).numpy()
    xMoEW2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_experts_w2_weight)))
    let ff_i_experts_0_w3_weight = state_dict["\(prefix).ff_i.experts.0.w3.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_1_w3_weight = state_dict["\(prefix).ff_i.experts.1.w3.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_2_w3_weight = state_dict["\(prefix).ff_i.experts.2.w3.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_3_w3_weight = state_dict["\(prefix).ff_i.experts.3.w3.weight"].to(
      torch.float
    ).cpu()
    let ff_i_experts_w3_weight = torch.concat(
      [
        ff_i_experts_0_w3_weight, ff_i_experts_1_w3_weight, ff_i_experts_2_w3_weight,
        ff_i_experts_3_w3_weight,
      ], dim: 0
    ).numpy()
    xMoEW3.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_i_experts_w3_weight)))
  }
  return (reader, Model([x, c, rot], [xOut]))
}

func HiDream(height: Int, width: Int, textLength: (Int, Int), layers: (Int, Int)) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let imgIn = Dense(count: 2_560, name: "x_embedder")
  var out = imgIn(x).to(.Float32)
  let t = Input()
  let vector = Input()
  let (tMlp0, tMlp2, timeEmbedder) = MLPEmbedder(channels: 2_560, name: "t")
  let (pMlp0, pMlp2, pooledEmbedder) = MLPEmbedder(channels: 2_560, name: "p")
  var vec = timeEmbedder(t) + pooledEmbedder(vector)
  let t5EncoderHiddenStates = Input()
  let llamaEncoderHiddenStates = (0..<32).map { _ in Input() }
  let captionProjections = (0..<49).map { _ in
    Dense(count: 2_560, noBias: true, name: "caption_projection")
  }
  var encoderHiddenStates = [Model.IO]()
  for i in 0..<48 {
    encoderHiddenStates.append(
      captionProjections[i](llamaEncoderHiddenStates[min(i, llamaEncoderHiddenStates.count - 1)]))
  }
  encoderHiddenStates.append(captionProjections[48](t5EncoderHiddenStates))
  var context = Functional.concat(
    axis: 1, encoderHiddenStates[encoderHiddenStates.count - 1],
    encoderHiddenStates[encoderHiddenStates.count - 2]
  ).to(.Float32)
  let h = height / 2
  let w = width / 2
  vec = vec.reshaped([1, 1, 2_560]).swish()
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers.0 {
    let contextIn = Functional.concat(axis: 1, context, encoderHiddenStates[i].to(.Float32))
    let (reader, block) = JointTransformerBlock(
      prefix: "double_stream_blocks.\(i).block", k: 128, h: 20, b: 1,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: false, upcast: false)
    readers.append(reader)
    let blockOut = block(out, contextIn, vec, rot)
    out = blockOut[0]
    context = blockOut[1]
  }
  out = Functional.concat(axis: 1, out, context)
  for i in 0..<layers.1 {
    let xIn = Functional.concat(axis: 1, out, encoderHiddenStates[layers.0 + i].to(.Float32))
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_stream_blocks.\(i).block", k: 128, h: 20, b: 1,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: i == layers.1 - 1, upcast: false)
    readers.append(reader)
    out = block(xIn, vec, rot)
  }
  let scale = Dense(count: 2_560, name: "ada_ln_0")
  let shift = Dense(count: 2_560, name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_proj_weight = state_dict["x_embedder.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_in_proj_bias = state_dict["x_embedder.proj.bias"].to(torch.float)
      .cpu().numpy()
    imgIn.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_in_proj_weight)))
    imgIn.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_in_proj_bias)))
    let t_embedder_mlp_0_weight = state_dict["t_embedder.timestep_embedder.linear_1.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_0_bias = state_dict["t_embedder.timestep_embedder.linear_1.bias"].to(
      torch.float
    )
    .cpu().numpy()
    tMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight)))
    tMlp0.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias)))
    let t_embedder_mlp_2_weight = state_dict["t_embedder.timestep_embedder.linear_2.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_2_bias = state_dict["t_embedder.timestep_embedder.linear_2.bias"].to(
      torch.float
    )
    .cpu().numpy()
    tMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight)))
    tMlp2.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias)))
    let p_embedder_mlp_0_weight = state_dict["p_embedder.pooled_embedder.linear_1.weight"].to(
      torch.float
    ).cpu().numpy()
    let p_embedder_mlp_0_bias = state_dict["p_embedder.pooled_embedder.linear_1.bias"].to(
      torch.float
    )
    .cpu().numpy()
    pMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: p_embedder_mlp_0_weight)))
    pMlp0.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: p_embedder_mlp_0_bias)))
    let p_embedder_mlp_2_weight = state_dict["p_embedder.pooled_embedder.linear_2.weight"].to(
      torch.float
    ).cpu().numpy()
    let p_embedder_mlp_2_bias = state_dict["p_embedder.pooled_embedder.linear_2.bias"].to(
      torch.float
    )
    .cpu().numpy()
    pMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: p_embedder_mlp_2_weight)))
    pMlp2.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: p_embedder_mlp_2_bias)))
    for i in 0..<49 {
      let caption_projection_i_linear_weight = state_dict["caption_projection.\(i).linear.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      captionProjections[i].weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: caption_projection_i_linear_weight)))
    }
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_linear_weight = state_dict["final_layer.adaLN_modulation.1.weight"].to(torch.float)
      .cpu().numpy()
    let norm_out_linear_bias = state_dict["final_layer.adaLN_modulation.1.bias"].to(torch.float)
      .cpu().numpy()
    shift.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_out_linear_weight[0..<2560, ...])))
    scale.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: norm_out_linear_weight[2560..<(2560 * 2), ...])))
    shift.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_out_linear_bias[0..<2560])))
    scale.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: norm_out_linear_bias[2560..<(2560 * 2)])))
    let proj_out_weight = state_dict["final_layer.linear.weight"].to(
      torch.float
    ).cpu().numpy()
    projOut.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_out_weight)))
    let proj_out_bias = state_dict["final_layer.linear.bias"].to(
      torch.float
    ).cpu().numpy()
    projOut.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_out_bias)))
  }
  return (
    Model([x, rot, t, vector, t5EncoderHiddenStates] + llamaEncoderHiddenStates, [out]), reader
  )
}

print(transformer)
let transformer_state_dict = transformer.state_dict()
print(transformer_state_dict.keys())

graph.withNoGrad {
  let xTensor = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: x.cpu().float().numpy())).toGPU(1)
  ).reshaped(format: .NHWC, shape: [1, 16, 64, 2, 64, 2]).permuted(0, 2, 4, 3, 5, 1).copied()
    .reshaped(format: .NHWC, shape: [1, 64 * 64, 2 * 2 * 16])
  let promptEmbed3Tensor = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: prompt_embed_3.cpu().float().numpy())).toGPU(1))
  let promptEmbed4Tensor = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: prompt_embed_4.cpu().float().numpy())).toGPU(1))
  let promptEmbed4Tensors = (0..<32).map {
    promptEmbed4Tensor[$0..<($0 + 1), 0..<1, 0..<128, 0..<4096].copied().reshaped(
      .HWC(1, 128, 4096))
  }
  let pooledPromptEmbedTensor = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: pooled_prompt_embed.cpu().float().numpy()))
      .toGPU(1))
  let timestep = graph.variable(
    Tensor<Float16>(
      from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(1))
  let rotTensor = graph.variable(.CPU, .NHWC(1, 4096 + 128 * 3, 1, 128), of: Float.self)
  for y in 0..<64 {
    for x in 0..<64 {
      let i = y * 64 + x
      for k in 0..<32 {
        let theta = 0 * 1.0 / pow(10_000, Double(k) / 32)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(y) * 1.0 / pow(10_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 32) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 32) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<16 {
        let theta = Double(x) * 1.0 / pow(10_000, Double(k) / 16)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 32 + 16) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 32 + 16) * 2 + 1] = Float(sintheta)
      }
    }
  }
  for i in 0..<(128 * 3) {
    for k in 0..<32 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 32)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, 4096 + i, 0, k * 2] = Float(costheta)
      rotTensor[0, 4096 + i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, 4096 + i, 0, (k + 32) * 2] = Float(costheta)
      rotTensor[0, 4096 + i, 0, (k + 32) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<16 {
      let theta = 0 * 1.0 / pow(10_000, Double(k) / 16)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, 4096 + i, 0, (k + 32 + 16) * 2] = Float(costheta)
      rotTensor[0, 4096 + i, 0, (k + 32 + 16) * 2 + 1] = Float(sintheta)
    }
  }
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(1)
  let (hiDream, reader) = HiDream(height: 128, width: 128, textLength: (128, 128), layers: (16, 32))
  hiDream.compile(
    inputs: [xTensor, rotTensorGPU, timestep, pooledPromptEmbedTensor, promptEmbed3Tensor]
      + promptEmbed4Tensors)
  reader(transformer_state_dict)
  debugPrint(
    hiDream(
      inputs: xTensor,
      [rotTensorGPU, timestep, pooledPromptEmbedTensor, promptEmbed3Tensor] + promptEmbed4Tensors))
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/hidream_i1_full_f16.ckpt") {
    $0.write("dit", model: hiDream)
  }
  */
}
