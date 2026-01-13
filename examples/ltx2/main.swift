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

let torch = Python.import("torch")
let PIL = Python.import("PIL")

torch.set_grad_enabled(false)

let torch_device = torch.device("cuda")

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let ltx_core_loader_single_gpu_model_builder = Python.import(
  "ltx_core.loader.single_gpu_model_builder")
let ltx_core_model_transformer_model_configurator = Python.import(
  "ltx_core.model.transformer.model_configurator")
let ltx_core_model_clip_gemma_encoders_av_encoder = Python.import(
  "ltx_core.model.clip.gemma.encoders.av_encoder")
let ltx_core_model_transformer_modality = Python.import("ltx_core.model.transformer.modality")
let ltx_core_loader_sd_keys_ops = Python.import("ltx_core.loader.sd_keys_ops")
let ltx_core_pipeline_components_patchifiers = Python.import(
  "ltx_core.pipeline.components.patchifiers")
let ltx_core_pipeline_conditioning = Python.import("ltx_core.pipeline.conditioning")

let prompt = "a dance party"
// Text Encoder
let text_encoder = ltx_core_loader_single_gpu_model_builder.SingleGPUModelBuilder(
  model_path: "/fast/Data/ltx-2-19b-dev.safetensors",
  model_class_configurator: ltx_core_model_clip_gemma_encoders_av_encoder
    .AVGemmaTextEncoderModelConfigurator.with_gemma_root_path(
      "/slow/Data/google/gemma-3-12b-it-qat-q4_0-unquantized"),
  model_sd_key_ops: ltx_core_model_clip_gemma_encoders_av_encoder.AV_GEMMA_TEXT_ENCODER_KEY_OPS
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

func BasicTransformerBlock1D(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let normX = norm(x).to(.Float16)
  let rot = Input()
  let toKeys = Dense(count: k * h, name: "to_k")
  let toQueries = Dense(count: k * h, name: "to_q")
  let toValues = Dense(count: k * h, name: "to_v")
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
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
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

func Embedding1DConnector(prefix: String, layers: Int, tokenLength: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  var readers = [(PythonObject) -> Void]()
  var out: Model.IO = x
  for i in 0..<layers {
    let (reader, block) = BasicTransformerBlock1D(
      prefix: "\(prefix).transformer_1d_blocks.\(i)", k: 128, h: 30, b: 1, t: tokenLength)
    out = block(out, rot)
    readers.append(reader)
  }
  let norm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = norm(out).to(.Float16)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x, rot], [out]), reader)
}

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
    prefix: "embeddings_connector", layers: 2, tokenLength: 1024)
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
    prefix: "audio_embeddings_connector", layers: 2, tokenLength: 1024)
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

let audio_builder = ltx_core_pipeline_conditioning.AudioConditioningBuilder(
  patchifier: ltx_core_pipeline_components_patchifiers.AudioPatchifier(patch_size: 1), batch: 1,
  duration: 121.0 / 25.0)
let video_builder = ltx_core_pipeline_conditioning.VideoConditioningBuilder(
  patchifier: ltx_core_pipeline_components_patchifiers.VideoLatentPatchifier(patch_size: 1),
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
  model_sd_key_ops: ltx_core_loader_sd_keys_ops.LTXV_MODEL_COMFY_RENAMING_MAP)
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
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
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
  prefix: String, k: (Int, Int, Int), h: Int, b: Int, t: (Int, Int), name: String
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let context = Input()
  let rot = Input()
  let rotK = Input()
  let toKeys = Dense(count: k.1 * h, name: "\(name)_k")
  let toQueries = Dense(count: k.1 * h, name: "\(name)_q")
  let toValues = Dense(count: k.1 * h, name: "\(name)_v")
  var keys = toKeys(context)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_k")
  keys = normK(keys).reshaped([b, t.1, h, k.1])
  var queries = toQueries(x)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(name)_norm_q")
  queries = normQ(queries).reshaped([b, t.0, h, k.1])
  let values = toValues(context).reshaped([b, t.1, h, k.1])
  queries = (1 / Float(k.1).squareRoot().squareRoot()) * Functional.cmul(left: queries, right: rot)
  keys = (1 / Float(k.1).squareRoot().squareRoot()) * Functional.cmul(left: keys, right: rotK)
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t.0, k.1 * h])
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
  return (reader, Model([x, rot, context, rotK], [out]))
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
  let timesteps = (0..<6).map { _ in Input() }
  let attn1Modulations = (0..<6).map {
    Parameter<Float>(.GPU(2), .HWC(1, 1, k * h), name: "attn1_ada_ln_\($0)")
  }
  let norm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var out =
    norm(vx) .* (1 + (attn1Modulations[1] + timesteps[1])) + (attn1Modulations[0] + timesteps[0])
  let (attn1Reader, attn1) = LTX2SelfAttention(
    prefix: "\(prefix).attn1", k: k, h: h, b: b, t: hw, name: "x")
  out = vx + attn1(out.to(.Float16), rot).to(of: vx) .* (attn1Modulations[2] + timesteps[2])
  let (attn2Reader, attn2) = LTX2CrossAttention(
    prefix: "\(prefix).attn2", k: (k, k, k), h: h, b: b, t: (hw, t), name: "cv")
  let normOut = norm(out).to(.Float16)
  out = out + attn2(normOut, rot, cv, rotC).to(of: out)
  let audioTimesteps = (0..<6).map { _ in Input() }
  let audioAttn1Modulations = (0..<6).map {
    Parameter<Float>(.GPU(2), .HWC(1, 1, k / 2 * h), name: "audio_attn1_ada_ln_\($0)")
  }
  let (audioAttn1Reader, audioAttn1) = LTX2SelfAttention(
    prefix: "\(prefix).audio_attn1", k: k / 2, h: h, b: b, t: a, name: "a")
  var aOut =
    norm(ax) .* (1 + (audioAttn1Modulations[1] + audioTimesteps[1]))
    + (audioAttn1Modulations[0] + audioTimesteps[0])
  aOut = ax + audioAttn1(aOut.to(.Float16), rotA).to(of: ax)
    .* (audioAttn1Modulations[2] + audioTimesteps[2])
  let (audioAttn2Reader, audioAttn2) = LTX2CrossAttention(
    prefix: "\(prefix).audio_attn2", k: (k / 2, k / 2, k / 2), h: h, b: b, t: (a, t), name: "ca")
  let normAOut = norm(aOut).to(.Float16)
  aOut = aOut + audioAttn2(normAOut, rotA, ca, rotAC).to(of: aOut)
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
        .GPU(2), .HWC(1, 1, k * h), name: "audio_to_video_attn_ada_ln_\($0)")
    } else if $0 < 4 {
      return Parameter<Float>(
        .GPU(2), .HWC(1, 1, k / 2 * h), name: "audio_to_video_attn_ada_ln_\($0)")
    } else {
      return Parameter<Float>(
        .GPU(2), .HWC(1, 1, k * h), name: "audio_to_video_attn_ada_ln_\($0)")
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
        .GPU(2), .HWC(1, 1, k * h), name: "video_to_audio_attn_ada_ln_\($0)")
    } else {
      return Parameter<Float>(
        .GPU(2), .HWC(1, 1, k / 2 * h), name: "video_to_audio_attn_ada_ln_\($0)")
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
    for i in 0..<6 {
      attn1Modulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: scale_shift_table[i..<(i + 1), ...])))
    }
    attn1Reader(state_dict)
    attn2Reader(state_dict)
    let audio_scale_shift_table = state_dict["\(prefix).audio_scale_shift_table"].to(torch.float)
      .cpu().numpy()
    for i in 0..<6 {
      audioAttn1Modulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: audio_scale_shift_table[i..<(i + 1), ...])))
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

func LTX2(b: Int, h: Int, w: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let rot = Input()
  let rotC = Input()
  let rotA = Input()
  let rotAC = Input()
  let rotCX = Input()
  let xEmbedder = Dense(count: 4096, name: "x_embedder")
  let (contextMlp0, contextMlp2, contextEmbedder) = GELUMLPEmbedder(channels: 4096, name: "context")
  var out = xEmbedder(x).to(.Float32)
  let txt = Input()
  let txtOut = contextEmbedder(txt)
  let a = Input()
  let aEmbedder = Dense(count: 2048, name: "a_embedder")
  let (aContextMlp0, aContextMlp2, aContextEmbedder) = GELUMLPEmbedder(
    channels: 2048, name: "a_context")
  var aOut = aEmbedder(a).to(.Float32)
  let aTxt = Input()
  let aTxtOut = aContextEmbedder(aTxt)
  let t = Input()
  let (txReader, txEmb, txEmbChunks) = LTX2AdaLNSingle(
    prefix: "adaln_single", channels: 4096, count: 6, outputEmbedding: true, name: "tx", t: t)
  let (taReader, taEmb, taEmbChunks) = LTX2AdaLNSingle(
    prefix: "audio_adaln_single", channels: 2048, count: 6, outputEmbedding: true, name: "ta", t: t)
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
      txEmbChunks[5],
      taEmbChunks[0], taEmbChunks[1], taEmbChunks[2], taEmbChunks[3], taEmbChunks[4],
      taEmbChunks[5],
      tcxEmbChunks[1], tcxEmbChunks[0], tcaEmbChunks[1], tcaEmbChunks[0], a2vEmbChunks[0],
      tcxEmbChunks[3], tcxEmbChunks[2], tcaEmbChunks[3], tcaEmbChunks[2], v2aEmbChunks[0])
    readers.append(reader)
    out = blockOut[0]
    aOut = blockOut[1]
  }
  let scaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(2), .HWC(1, 1, 4096), name: "norm_out_ada_ln_\($0)")
  }
  let normOut = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if let txEmb = txEmb {
    out = normOut(out) .* (1 + (scaleShiftModulations[1] + txEmb))
      + (scaleShiftModulations[0] + txEmb)
  }
  let projOut = Dense(count: 128, name: "proj_out")
  out = projOut(out.to(.Float16))
  let audioScaleShiftModulations = (0..<2).map {
    Parameter<Float>(.GPU(2), .HWC(1, 1, 2048), name: "audio_norm_out_ada_ln_\($0)")
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
    let caption_projection_linear_1_weight = state_dict["caption_projection.linear_1.weight"].to(
      torch.float
    ).cpu().numpy()
    contextMlp0.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: caption_projection_linear_1_weight)))
    let caption_projection_linear_1_bias = state_dict["caption_projection.linear_1.bias"].to(
      torch.float
    ).cpu().numpy()
    contextMlp0.bias.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: caption_projection_linear_1_bias)))
    let caption_projection_linear_2_weight = state_dict["caption_projection.linear_2.weight"].to(
      torch.float
    ).cpu().numpy()
    contextMlp2.weight.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: caption_projection_linear_2_weight)))
    let caption_projection_linear_2_bias = state_dict["caption_projection.linear_2.bias"].to(
      torch.float
    ).cpu().numpy()
    contextMlp2.bias.copy(
      from: Tensor<FloatType>(from: try! Tensor<Float>(numpy: caption_projection_linear_2_bias)))
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
    txReader(state_dict)
    taReader(state_dict)
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
  return (reader, Model([x, txt, a, aTxt, t, rot, rotC, rotA, rotAC, rotCX], [out, aOut]))
}

let (reader, dit) = LTX2(b: 1, h: 64, w: 64)

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
    inputs: xTensor, txtTensor, aTensor, aTxtTensor, timestepTensor, rot1TensorGPU, rot2TensorGPU,
    rot3TensorGPU, rot4TensorGPU, rot5TensorGPU)
  reader(state_dict)
  debugPrint(
    dit(
      inputs: xTensor, txtTensor, aTensor, aTxtTensor, timestepTensor, rot1TensorGPU, rot2TensorGPU,
      rot3TensorGPU, rot4TensorGPU, rot5TensorGPU))
  graph.openStore("/home/liu/workspace/swift-diffusion/ltx_2_19b_dev_f16.ckpt") {
    $0.write("dit", model: dit)
  }
}
