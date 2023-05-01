import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let kandinsky2 = Python.import("kandinsky2")
let torch = Python.import("torch")
let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let prompt = "red cat, 4k photo"

let model = kandinsky2.get_kandinsky2(
  "cuda", task_type: "text2img", model_version: "2.1", use_flash_attention: false)
let state_dict = model.text_encoder.state_dict()
print(model.clip_model.text_projection.shape)
let prior_state_dict = model.prior.state_dict()
print(prior_state_dict.keys())

func XLMRobertaTextEmbedding(
  prefix: String, vocabularySize: Int, maxLength: Int, tokenTypes: Int, embeddingSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let tokens = Input()
  let tokenType = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let tokenTypeEmbed = Embedding(
    Float.self, vocabularySize: tokenTypes, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(Float.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + tokenTypeEmbed(tokenType) + positionEmbed(positions)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  let out = layerNorm(embedding)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["\(prefix).word_embeddings.weight"].type(torch.float).cpu().numpy()
    let token_type = state_dict["\(prefix).token_type_embeddings.weight"].type(torch.float).cpu()
      .numpy()
    let pos = state_dict["\(prefix).position_embeddings.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab))
    tokenTypeEmbed.parameters.copy(from: try! Tensor<Float>(numpy: token_type))
    positionEmbed.parameters.copy(from: try! Tensor<Float>(numpy: pos))
    let layer_norm_weight = state_dict["\(prefix).LayerNorm.weight"].type(torch.float).cpu().numpy()
    let layer_norm_bias = state_dict["\(prefix).LayerNorm.bias"].type(torch.float).cpu().numpy()
    layerNorm.weight.copy(from: try! Tensor<Float>(numpy: layer_norm_weight))
    layerNorm.bias.copy(from: try! Tensor<Float>(numpy: layer_norm_bias))
  }
  return (reader, Model([tokens, positions, tokenType], [out], name: "embeddings"))
}

func XLMRobertaSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + casualAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let self_key_weight = state_dict["\(prefix).self.key.weight"].type(torch.float).cpu().numpy()
    let self_key_bias = state_dict["\(prefix).self.key.bias"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: self_key_weight))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: self_key_bias))
    let self_query_weight = state_dict["\(prefix).self.query.weight"].type(torch.float).cpu()
      .numpy()
    let self_query_bias = state_dict["\(prefix).self.query.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: self_query_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: self_query_bias))
    let self_value_weight = state_dict["\(prefix).self.value.weight"].type(torch.float).cpu()
      .numpy()
    let self_value_bias = state_dict["\(prefix).self.value.bias"].type(torch.float).cpu().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: self_value_weight))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: self_value_bias))
    let output_dense_weight = state_dict["\(prefix).output.dense.weight"].type(torch.float).cpu()
      .numpy()
    let output_dense_bias = state_dict["\(prefix).output.dense.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: output_dense_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: output_dense_bias))
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

func XLMRobertaLayer(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let casualAttentionMask = Input()
  let (selfAttentionReader, selfAttention) = XLMRobertaSelfAttention(
    prefix: "\(prefix).attention", k: k, h: h, b: b, t: t)
  var out = selfAttention(x, casualAttentionMask)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm(out + x)
  let intermediate = Dense(count: k * h * 4)
  let ff = out
  out = intermediate(out).GELU()
  let output = Dense(count: k * h)
  out = output(out)
  let layerNormFinal = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNormFinal(out + ff)
  let reader: (PythonObject) -> Void = { state_dict in
    selfAttentionReader(state_dict)
    let attention_output_layerNorm_weight = state_dict[
      "\(prefix).attention.output.LayerNorm.weight"
    ].type(torch.float).cpu().numpy()
    let attention_output_layerNorm_bias = state_dict["\(prefix).attention.output.LayerNorm.bias"]
      .type(torch.float).cpu().numpy()
    layerNorm.weight.copy(
      from: try! Tensor<Float>(numpy: attention_output_layerNorm_weight))
    layerNorm.bias.copy(
      from: try! Tensor<Float>(numpy: attention_output_layerNorm_bias))
    let intermediate_dense_weight = state_dict["\(prefix).intermediate.dense.weight"].type(
      torch.float
    ).cpu().numpy()
    let intermediate_dense_bias = state_dict["\(prefix).intermediate.dense.bias"].type(torch.float)
      .cpu().numpy()
    intermediate.weight.copy(
      from: try! Tensor<Float>(numpy: intermediate_dense_weight))
    intermediate.bias.copy(
      from: try! Tensor<Float>(numpy: intermediate_dense_bias))
    let output_dense_weight = state_dict["\(prefix).output.dense.weight"].type(torch.float).cpu()
      .numpy()
    let output_dense_bias = state_dict["\(prefix).output.dense.bias"].type(torch.float).cpu()
      .numpy()
    output.weight.copy(from: try! Tensor<Float>(numpy: output_dense_weight))
    output.bias.copy(from: try! Tensor<Float>(numpy: output_dense_bias))
    let output_layerNorm_weight = state_dict["\(prefix).output.LayerNorm.weight"].type(torch.float)
      .cpu().numpy()
    let output_layerNorm_bias = state_dict["\(prefix).output.LayerNorm.bias"].type(torch.float)
      .cpu().numpy()
    layerNormFinal.weight.copy(
      from: try! Tensor<Float>(numpy: output_layerNorm_weight))
    layerNormFinal.bias.copy(
      from: try! Tensor<Float>(numpy: output_layerNorm_bias))
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

func XLMRobertaModel(numberOfLayers: Int, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  var readers = [(PythonObject) -> Void]()
  let x = Input()
  let casualAttentionMask = Input()
  var out: Model.IO = x
  for i in 0..<numberOfLayers {
    let (reader, layer) = XLMRobertaLayer(
      prefix: "model.transformer.encoder.layer.\(i)", k: k, h: h, b: b, t: t)
    out = layer(out, casualAttentionMask)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

func SelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + casualAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let c_qkv_weight = state_dict["\(prefix).c_qkv.weight"].type(torch.float).cpu()
    let c_qkv_weight_view = c_qkv_weight.view(32, 64 * 3, 2048)
    let (c_q_weight, c_k_weight, c_v_weight) = torch[dynamicMember: "split"](
      c_qkv_weight_view, 64, dim: 1
    ).tuple3
    let c_qkv_bias = state_dict["\(prefix).c_qkv.bias"].type(torch.float).cpu()
    let c_qkv_bias_view = c_qkv_bias.view(32, 64 * 3)
    let (c_q_bias, c_k_bias, c_v_bias) = torch[dynamicMember: "split"](c_qkv_bias_view, 64, dim: 1)
      .tuple3
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: c_q_weight.numpy()))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: c_q_bias.numpy()))
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: c_k_weight.numpy()))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: c_k_bias.numpy()))
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: c_v_weight.numpy()))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: c_v_bias.numpy()))
    let c_proj_weight = state_dict["\(prefix).c_proj.weight"].type(torch.float).cpu()
      .numpy()
    let c_proj_bias = state_dict["\(prefix).c_proj.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: c_proj_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: c_proj_bias))
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

func ResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  let (selfAttentionReader, selfAttention) = SelfAttention(
    prefix: "\(prefix).attn", k: k, h: h, b: b, t: t)
  var out = x + selfAttention(layerNorm1(x), casualAttentionMask)
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let intermediate = Dense(count: k * h * 4)
  let output = Dense(count: k * h)
  out = out + output(intermediate(layerNorm2(out)).GELU())
  let reader: (PythonObject) -> Void = { state_dict in
    selfAttentionReader(state_dict)
    let ln_1_weight = state_dict["\(prefix).ln_1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).ln_1.bias"].type(torch.float).cpu().numpy()
    layerNorm1.weight.copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    layerNorm1.bias.copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let ln_2_weight = state_dict["\(prefix).ln_2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).ln_2.bias"].type(torch.float).cpu().numpy()
    layerNorm2.weight.copy(from: try! Tensor<Float>(numpy: ln_2_weight))
    layerNorm2.bias.copy(from: try! Tensor<Float>(numpy: ln_2_bias))
    let mlp_c_fc_weight = state_dict["\(prefix).mlp.c_fc.weight"].type(torch.float).cpu().numpy()
    let mlp_c_fc_bias = state_dict["\(prefix).mlp.c_fc.bias"].type(torch.float).cpu().numpy()
    intermediate.weight.copy(from: try! Tensor<Float>(numpy: mlp_c_fc_weight))
    intermediate.bias.copy(from: try! Tensor<Float>(numpy: mlp_c_fc_bias))
    let mlp_c_proj_weight = state_dict["\(prefix).mlp.c_proj.weight"].type(torch.float).cpu()
      .numpy()
    let mlp_c_proj_bias = state_dict["\(prefix).mlp.c_proj.bias"].type(torch.float).cpu()
      .numpy()
    output.weight.copy(from: try! Tensor<Float>(numpy: mlp_c_proj_weight))
    output.bias.copy(from: try! Tensor<Float>(numpy: mlp_c_proj_bias))
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

func timeEmbedding(timestep: Int, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * Float(timestep)
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

func DiffusionMapping(numberOfLayers: Int, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  var readers = [(PythonObject) -> Void]()
  let x = Input()
  let casualAttentionMask = Input()
  var out: Model.IO = x
  for i in 0..<numberOfLayers {
    let (reader, layer) = ResidualAttentionBlock(
      prefix: "model.transformer.resblocks.\(i)", k: k, h: h, b: b, t: t)
    out = layer(out, casualAttentionMask)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

let dmInput = torch.randn([2, 81, 2048])
var dmMask: PythonObject? = nil
let graph = DynamicGraph()
graph.withNoGrad {
  let (reader, textEncoder) = XLMRobertaTextEmbedding(
    prefix: "model.transformer.embeddings", vocabularySize: 250_002, maxLength: 514, tokenTypes: 1,
    embeddingSize: 1_024)
  let tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  for i in 0..<154 {
    tokensTensor[i] = 1
  }
  tokensTensor[0] = 0
  tokensTensor[1] = 4842
  tokensTensor[2] = 7515
  tokensTensor[3] = 4
  tokensTensor[4] = 201
  tokensTensor[5] = 92
  tokensTensor[6] = 16186
  tokensTensor[7] = 2
  tokensTensor[77] = 0
  tokensTensor[78] = 2
  let tokenTypesTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  for i in 0..<154 {
    tokenTypesTensor[i] = 0
    positionTensor[i] = 1
  }
  for i in 0..<8 {
    positionTensor[i] = Int32(i + 2)
  }
  positionTensor[77] = 2
  positionTensor[78] = 3
  let tokensTensorGPU = tokensTensor.toGPU(1)
  let positionTensorGPU = positionTensor.toGPU(1)
  let tokenTypesTensorGPU = tokenTypesTensor.toGPU(1)
  textEncoder.compile(inputs: tokensTensorGPU, positionTensorGPU, tokenTypesTensorGPU)
  reader(state_dict)
  let embeddings = textEncoder(inputs: tokensTensorGPU, positionTensorGPU, tokenTypesTensorGPU)[0]
    .as(
      of: Float.self)
  let (layerReader, layer) = XLMRobertaModel(numberOfLayers: 24, k: 64, h: 16, b: 2, t: 77)
  let attentionMask = graph.variable(.CPU, .NCHW(2, 1, 1, 77), of: Float.self)
  attentionMask.full(0)
  for i in 8..<77 {
    attentionMask[0, 0, 0, i] = -Float.greatestFiniteMagnitude
  }
  for i in 2..<77 {
    attentionMask[1, 0, 0, i] = -Float.greatestFiniteMagnitude
  }
  let attentionMaskGPU = attentionMask.toGPU(1)
  layer.compile(inputs: embeddings, attentionMaskGPU)
  layerReader(state_dict)
  let output = layer(inputs: embeddings, attentionMaskGPU)[0].as(of: Float.self)
  debugPrint(output.reshaped(.CHW(2, 77, 1024)))

  let tokenizer = CLIPTokenizer(
    vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

  let unconditionalTokens = tokenizer.tokenize(text: "", truncation: true, maxLength: 77)
  let tokens = tokenizer.tokenize(text: prompt, truncation: true, maxLength: 77)

  let textModel = CLIPTextModel(
    Float.self,
    vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
    batchSize: 2, intermediateSize: 3072)

  let tokensTensorCLIP = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  let positionTensorCLIP = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  for i in 0..<77 {
    // Kandinsky implementation error, need to replicate that.
    tokensTensorCLIP[i] = i >= 8 ? 0 : tokens[i]
    tokensTensorCLIP[i + 77] = i >= 2 ? 0 : unconditionalTokens[i]
    positionTensorCLIP[i] = Int32(i)
    positionTensorCLIP[i + 77] = Int32(i)
  }

  let casualAttentionMask = graph.variable(Tensor<Float>(.CPU, .NHWC(1, 1, 77, 77)))
  casualAttentionMask.full(0)
  for i in 0..<76 {
    for j in (i + 1)..<77 {
      casualAttentionMask[0, 0, i, j] = -Float.greatestFiniteMagnitude
    }
  }
  let tokensTensorCLIPGPU = tokensTensorCLIP.toGPU(1)
  let positionTensorCLIPGPU = positionTensorCLIP.toGPU(1)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(1)
  textModel.compile(inputs: tokensTensorCLIPGPU, positionTensorCLIPGPU, casualAttentionMaskGPU)
  graph.openStore("/fast/Data/SD/swift-diffusion/clip_vit_l14_f16.ckpt") { store in
    store.read("text_model", model: textModel)
  }
  let textProjection = try! Tensor<Float>(
    numpy: model.clip_model.text_projection.detach().type(torch.float).cpu().numpy())
  let textProjectionGPU = graph.variable(textProjection.toGPU(1))
  let c = textModel(
    inputs: tokensTensorCLIPGPU, positionTensorCLIPGPU, casualAttentionMaskGPU)[0].as(
      of: Float.self
    )
  var indexGP = Tensor<Int32>(.CPU, .C(2))
  indexGP[0] = 7
  indexGP[1] = 78
  let textEmb =
    Functional.indexSelect(
      input: c.reshaped(.NC(2 * 77, 768)), index: graph.variable(indexGP.toGPU(1)))
    * textProjectionGPU
  let textEnc = c.reshaped(.CHW(2, 77, 768))
  debugPrint(textEmb)
  debugPrint(textEnc)
  let dmInputTensorGPU = graph.variable(try! Tensor<Float>(numpy: dmInput.numpy())).reshaped(
    .NC(2 * 81, 2048)
  ).toGPU(1)
  let (diffusionMappingReader, diffusionMapping) = DiffusionMapping(
    numberOfLayers: 20, k: 64, h: 32, b: 2, t: 81)
  let dmCasualAttentionMask = graph.variable(Tensor<Float>(.CPU, .NHWC(1, 1, 81, 81)))
  dmCasualAttentionMask.full(0)
  for i in 0..<80 {
    for j in (i + 1)..<81 {
      dmCasualAttentionMask[0, 0, i, j] = -Float.greatestFiniteMagnitude
    }
  }
  dmMask = torch.from_numpy(dmCasualAttentionMask.reshaped(.CHW(1, 81, 81)).rawValue)
  let dmCasualAttentionMaskGPU = dmCasualAttentionMask.toGPU(1)
  debugPrint(dmInputTensorGPU)
  diffusionMapping.compile(inputs: dmInputTensorGPU, dmCasualAttentionMaskGPU)
  diffusionMappingReader(prior_state_dict)
  let result = diffusionMapping(inputs: dmInputTensorGPU, dmCasualAttentionMaskGPU)[0].as(
    of: Float.self)
  debugPrint(result.reshaped(.CHW(2, 81, 2048)))
}

torch.cuda.set_device(0)
let dmMask2 = torch.cat([dmMask!, dmMask!])
let result = model.prior.model.transformer(
  dmInput.to(torch.float16).cuda(), mask: dmMask2.to(torch.float16).cuda())
print(result)
/*
let images = model.generate_text2img(
  prompt, num_steps: 100, batch_size: 1, guidance_scale: 4, h: 768, w: 768,
  sampler: "p_sampler", prior_cf_scale: 4, prior_steps: "5")
images[0].save("/home/liu/workspace/swift-diffusion/kandinsky.png")
*/
