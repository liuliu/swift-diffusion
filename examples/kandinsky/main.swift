import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

public struct DiffusionModel {
  public var linearStart: Float
  public var linearEnd: Float
  public var timesteps: Int
  public var steps: Int
}

extension DiffusionModel {
  public var betas: [Float] {  // Linear for now.
    var betas = [Float]()
    let start = linearStart
    let length = linearEnd - start
    for i in 0..<timesteps {
      let beta = start + Float(i) * length / Float(timesteps - 1)
      betas.append(beta)
    }
    return betas
  }
  public var alphasCumprod: [Float] {
    var cumprod: Float = 1
    return betas.map {
      cumprod *= 1 - $0
      return cumprod
    }
  }
}

public struct CLIPDiffusionModel {
  public var timesteps: Int
  public var steps: Int
}

extension CLIPDiffusionModel {
  public var betas: [Float] {  // Cosine based.
    var betas = [Float]()
    for i in 0..<timesteps {
      let t1 = Double(i) / Double(timesteps)
      let t2 = Double(i + 1) / Double(timesteps)
      let cos1 = cos((t1 + 0.008) / 1.008 * Double.pi / 2)
      let cos2 = cos((t2 + 0.008) / 1.008 * Double.pi / 2)
      let beta = Float(min(1 - (cos2 * cos2) / (cos1 * cos1), 0.999))
      betas.append(beta)
    }
    return betas
  }
  public var alphasCumprod: [Float] {
    var cumprod: Float = 1
    return betas.map {
      cumprod *= 1 - $0
      return cumprod
    }
  }
}

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
let prior_state_dict = model.prior.state_dict()
let model_state_dict = model.model.state_dict()
let movq_state_dict = model.image_encoder.state_dict()

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

func timestepEmbedding(prefix: String, channels: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let dense1 = Dense(count: channels)
  var out = dense1(x).swish()
  let dense2 = Dense(count: channels)
  out = dense2(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let time_embed_0_weight = state_dict["\(prefix).0.weight"].type(torch.float).cpu()
      .numpy()
    let time_embed_0_bias = state_dict["\(prefix).0.bias"].type(torch.float).cpu().numpy()
    dense1.weight.copy(from: try! Tensor<Float>(numpy: time_embed_0_weight))
    dense1.bias.copy(from: try! Tensor<Float>(numpy: time_embed_0_bias))
    let time_embed_2_weight = state_dict["\(prefix).2.weight"].type(torch.float).cpu()
      .numpy()
    let time_embed_2_bias = state_dict["\(prefix).2.bias"].type(torch.float).cpu().numpy()
    dense2.weight.copy(from: try! Tensor<Float>(numpy: time_embed_2_weight))
    dense2.bias.copy(from: try! Tensor<Float>(numpy: time_embed_2_bias))
  }
  return (reader, Model([x], [out]))
}

func DiffusionMapping(numberOfLayers: Int, k: Int, h: Int, b: Int, t: Int, outChannels: Int) -> (
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
  let finalLn = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLn(out)
  let outProj = Dense(count: outChannels)
  out = outProj(
    out.reshaped([b, 1, k * h], offset: [0, t - 1, 0], strides: [t * k * h, k * h, 1]))
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
    let final_ln_weight = state_dict["model.final_ln.weight"].type(torch.float).cpu().numpy()
    let final_ln_bias = state_dict["model.final_ln.bias"].type(torch.float).cpu().numpy()
    finalLn.weight.copy(from: try! Tensor<Float>(numpy: final_ln_weight))
    finalLn.bias.copy(from: try! Tensor<Float>(numpy: final_ln_bias))
    let out_proj_weight = state_dict["model.out_proj.weight"].type(torch.float).cpu().numpy()
    let out_proj_bias = state_dict["model.out_proj.bias"].type(torch.float).cpu().numpy()
    outProj.weight.copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    outProj.bias.copy(from: try! Tensor<Float>(numpy: out_proj_bias))
  }
  return (reader, Model([x, casualAttentionMask], [out]))
}

func ResBlock(
  prefix: String, batchSize: Int, outChannels: Int, up: Bool, down: Bool, skipConnection: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let emb = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = norm1(x).swish()
  var xhd: Model.IO = x
  if up {
    let hup = Upsample(.nearest, widthScale: 2, heightScale: 2)
    out = hup(out)
    let xup = Upsample(.nearest, widthScale: 2, heightScale: 2)
    xhd = xup(x)
  } else if down {
    let hdown = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
    out = hdown(out)
    let xdown = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
    xhd = xdown(x)
  }
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let embLayer = Dense(count: 2 * outChannels)
  let embOut = embLayer(emb.swish())
  let embScale = embOut.reshaped(
    [batchSize, outChannels, 1, 1], offset: [0, 0, 0, 0], strides: [outChannels * 2, 1, 1, 1])
  let embShift = embOut.reshaped(
    [batchSize, outChannels, 1, 1], offset: [0, outChannels, 0, 0],
    strides: [outChannels * 2, 1, 1, 1])
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = norm2(out) .* (1 + embScale) + embShift
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out.swish())
  var skipConv: Model?
  if skipConnection {
    let conv = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1])
    xhd = conv(xhd)
    skipConv = conv
  } else {
    skipConv = nil
  }
  out = xhd + out
  let reader: (PythonObject) -> Void = { state_dict in
    let in_layers_0_weight = state_dict["\(prefix).in_layers.0.weight"].type(torch.float).cpu()
      .numpy()
    let in_layers_0_bias = state_dict["\(prefix).in_layers.0.bias"].type(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: in_layers_0_weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: in_layers_0_bias))
    let in_layers_2_weight = state_dict["\(prefix).in_layers.2.weight"].type(torch.float).cpu()
      .numpy()
    let in_layers_2_bias = state_dict["\(prefix).in_layers.2.bias"].type(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: in_layers_2_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: in_layers_2_bias))
    let emb_layers_1_weight = state_dict["\(prefix).emb_layers.1.weight"].type(torch.float).cpu()
      .numpy()
    let emb_layers_1_bias = state_dict["\(prefix).emb_layers.1.bias"].type(torch.float).cpu()
      .numpy()
    embLayer.weight.copy(from: try! Tensor<Float>(numpy: emb_layers_1_weight))
    embLayer.bias.copy(from: try! Tensor<Float>(numpy: emb_layers_1_bias))
    let out_layers_0_weight = state_dict["\(prefix).out_layers.0.weight"].type(torch.float).cpu()
      .numpy()
    let out_layers_0_bias = state_dict["\(prefix).out_layers.0.bias"].type(torch.float).cpu()
      .numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: out_layers_0_weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: out_layers_0_bias))
    let out_layers_3_weight = state_dict["\(prefix).out_layers.3.weight"].type(torch.float).cpu()
      .numpy()
    let out_layers_3_bias = state_dict["\(prefix).out_layers.3.bias"].type(torch.float).cpu()
      .numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: out_layers_3_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: out_layers_3_bias))
    if let skipConv = skipConv {
      let skip_connection_weight = state_dict["\(prefix).skip_connection.weight"].type(torch.float)
        .cpu().numpy()
      let skip_connection_bias = state_dict["\(prefix).skip_connection.bias"].type(torch.float)
        .cpu().numpy()
      skipConv.weight.copy(from: try! Tensor<Float>(numpy: skip_connection_weight))
      skipConv.bias.copy(from: try! Tensor<Float>(numpy: skip_connection_bias))
    }
  }
  return (reader, Model([x, emb], [out]))
}

func AttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let hw = height * width
  let x = Input()
  let encoderOut = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = norm(x).reshaped([b, k * h, hw]).transposed(1, 2)
  let toencoderkeys = Dense(count: k * h)
  let toencodervalues = Dense(count: k * h)
  let encoderIn = encoderOut
  let encoderkeys = toencoderkeys(encoderIn).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let encodervalues = toencodervalues(encoderIn).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(out).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(out)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(out).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, Functional.concat(axis: 2, encoderkeys, keys))
  dot = dot.reshaped([b * h * hw, t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t + hw])
  out = dot * Functional.concat(axis: 2, encodervalues, values)
  out = out.reshaped([b, h, hw, k]).transposed(2, 3).reshaped([b, k * h, height, width])
  let projOut = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  out = projOut(out) + x
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_weight = state_dict["\(prefix).norm.weight"].type(torch.float).cpu().numpy()
    let norm_bias = state_dict["\(prefix).norm.bias"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: norm_bias))
    let encoder_kv_weight = state_dict["\(prefix).encoder_kv.weight"].type(torch.float).cpu()
    let encoder_kv_weight_view = encoder_kv_weight.view(h, k * 2, -1)
    let (encoder_k_weight, encoder_v_weight) = torch[dynamicMember: "split"](
      encoder_kv_weight_view, k, dim: 1
    ).tuple2
    let encoder_kv_bias = state_dict["\(prefix).encoder_kv.bias"].type(torch.float).cpu()
    let encoder_kv_bias_view = encoder_kv_bias.view(h, k * 2)
    let (encoder_k_bias, encoder_v_bias) = torch[dynamicMember: "split"](
      encoder_kv_bias_view, k, dim: 1
    )
    .tuple2
    toencoderkeys.weight.copy(from: try! Tensor<Float>(numpy: encoder_k_weight.numpy()))
    toencoderkeys.bias.copy(from: try! Tensor<Float>(numpy: encoder_k_bias.numpy()))
    toencodervalues.weight.copy(from: try! Tensor<Float>(numpy: encoder_v_weight.numpy()))
    toencodervalues.bias.copy(from: try! Tensor<Float>(numpy: encoder_v_bias.numpy()))
    let qkv_weight = state_dict["\(prefix).qkv.weight"].type(torch.float).cpu()
    let qkv_weight_view = qkv_weight.view(h, k * 3, -1)
    let (q_weight, k_weight, v_weight) = torch[dynamicMember: "split"](
      qkv_weight_view, k, dim: 1
    ).tuple3
    let qkv_bias = state_dict["\(prefix).qkv.bias"].type(torch.float).cpu()
    let qkv_bias_view = qkv_bias.view(h, k * 3)
    let (q_bias, k_bias, v_bias) = torch[dynamicMember: "split"](qkv_bias_view, k, dim: 1)
      .tuple3
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight.numpy()))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_bias.numpy()))
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight.numpy()))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_bias.numpy()))
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight.numpy()))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_bias.numpy()))
    let proj_out_weight = state_dict["\(prefix).proj_out.weight"].type(torch.float).cpu()
      .numpy()
    let proj_out_bias = state_dict["\(prefix).proj_out.bias"].type(torch.float).cpu()
      .numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x, encoderOut], [out]))
}

func InputBlocks(
  prefix: String, batchSize: Int, channels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>,
  x: Model.IO, emb: Model.IO, xfOut: Model.IO
) -> ((PythonObject) -> Void, Model.IO, [Model.IO]) {
  var readers = [(PythonObject) -> Void]()
  let convIn = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var i = 1
  var lastCh = channels
  var ds = 1
  var height = startHeight
  var width = startWidth
  var hs = [Model.IO]()
  hs.append(out)
  for (level, mult) in channelMult.enumerated() {
    let ch = channels * mult
    for _ in 0..<numResBlocks {
      let (reader, resBlock) = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: false,
        skipConnection: ch != lastCh)
      out = resBlock(out, emb)
      readers.append(reader)
      lastCh = ch
      if attentionResolutions.contains(ds) {
        let (reader, attentionBlock) = AttentionBlock(
          prefix: "\(prefix).\(i).1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize,
          t: t, height: height, width: width)
        out = attentionBlock(out, xfOut)
        readers.append(reader)
      }
      hs.append(out)
      i += 1
    }
    if level != channelMult.count - 1 {
      let (reader, resBlock) = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: true,
        skipConnection: false)
      out = resBlock(out, emb)
      readers.append(reader)
      hs.append(out)
      i += 1
      ds *= 2
      height /= 2
      width /= 2
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let input_blocks_0_0_weight = state_dict["\(prefix).0.0.weight"].type(torch.float).cpu().numpy()
    let input_blocks_0_0_bias = state_dict["\(prefix).0.0.bias"].type(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_bias))
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out, hs)
}

func OutputBlocks(
  prefix: String, batchSize: Int, channels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>,
  x: Model.IO, emb: Model.IO, xfOut: Model.IO, hs: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO) {
  var readers = [(PythonObject) -> Void]()
  var out: Model.IO = x
  var i = 0
  var ds = 1
  var height = startHeight
  var width = startWidth
  for _ in 1..<channelMult.count {
    ds *= 2
    height /= 2
    width /= 2
  }
  for (level, mult) in channelMult.enumerated().reversed() {
    let ch = channels * mult
    for j in 0..<(numResBlocks + 1) {
      out = Functional.concat(axis: 1, out, hs[hs.count - 1 - i])
      let (reader, resBlock) = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: false,
        skipConnection: true)
      out = resBlock(out, emb)
      readers.append(reader)
      if attentionResolutions.contains(ds) {
        let (reader, attentionBlock) = AttentionBlock(
          prefix: "\(prefix).\(i).1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize,
          t: t, height: height, width: width)
        out = attentionBlock(out, xfOut)
        readers.append(reader)
      }
      if level > 0 && j == numResBlocks {
        let (reader, resBlock) = ResBlock(
          prefix: "\(prefix).\(i).2", batchSize: batchSize, outChannels: ch, up: true, down: false,
          skipConnection: false)
        out = resBlock(out, emb)
        readers.append(reader)
        ds /= 2
        height *= 2
        width *= 2
      }
      i += 1
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out)
}

func ImageAndTextEmbedding(batchSize: Int) -> ((PythonObject) -> Void, Model) {
  let imageEmb = Input()
  let poolEmb = Input()
  let fullEmb = Input()
  let clipToSeq = Dense(count: 10 * 768)
  let projN = Dense(count: 384 * 4)
  let lnModelN = LayerNorm(epsilon: 1e-5, axis: [2])
  let imgLayer = Dense(count: 384 * 4)
  let toModelDimN = Dense(count: 768)
  let clipSeq = clipToSeq(imageEmb).reshaped([batchSize, 10, 768])
  let xfProj = lnModelN(projN(poolEmb)) + imgLayer(imageEmb)
  let textEmb = toModelDimN(fullEmb)
  let xfOut = Functional.concat(axis: 1, clipSeq, textEmb)
  let reader: (PythonObject) -> Void = { state_dict in
    let clip_to_seq_weight = state_dict["clip_to_seq.weight"].type(torch.float).cpu()
      .numpy()
    clipToSeq.weight.copy(from: try! Tensor<Float>(numpy: clip_to_seq_weight))
    let clip_to_seq_bias = state_dict["clip_to_seq.bias"].type(torch.float).cpu()
      .numpy()
    clipToSeq.bias.copy(from: try! Tensor<Float>(numpy: clip_to_seq_bias))
    let proj_n_weight = state_dict["proj_n.weight"].type(torch.float).cpu()
      .numpy()
    projN.weight.copy(from: try! Tensor<Float>(numpy: proj_n_weight))
    let proj_n_bias = state_dict["proj_n.bias"].type(torch.float).cpu()
      .numpy()
    projN.bias.copy(from: try! Tensor<Float>(numpy: proj_n_bias))
    let ln_model_n_weight = state_dict["ln_model_n.weight"].type(torch.float).cpu()
      .numpy()
    lnModelN.weight.copy(from: try! Tensor<Float>(numpy: ln_model_n_weight))
    let ln_model_n_bias = state_dict["ln_model_n.bias"].type(torch.float).cpu()
      .numpy()
    lnModelN.bias.copy(from: try! Tensor<Float>(numpy: ln_model_n_bias))
    let img_layer_weight = state_dict["img_layer.weight"].type(torch.float).cpu()
      .numpy()
    imgLayer.weight.copy(from: try! Tensor<Float>(numpy: img_layer_weight))
    let img_layer_bias = state_dict["img_layer.bias"].type(torch.float).cpu()
      .numpy()
    imgLayer.bias.copy(from: try! Tensor<Float>(numpy: img_layer_bias))
    let to_model_dim_n_weight = state_dict["to_model_dim_n.weight"].type(torch.float).cpu()
      .numpy()
    toModelDimN.weight.copy(from: try! Tensor<Float>(numpy: to_model_dim_n_weight))
    let to_model_dim_n_bias = state_dict["to_model_dim_n.bias"].type(torch.float).cpu()
      .numpy()
    toModelDimN.bias.copy(from: try! Tensor<Float>(numpy: to_model_dim_n_bias))
  }
  return (reader, Model([poolEmb, fullEmb, imageEmb], [xfProj, xfOut]))
}

func UNet(
  batchSize: Int, channels: Int, outChannels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let emb = Input()
  let xfOut = Input()
  let (inputBlocksReader, inputBlocksOut, hs) = InputBlocks(
    prefix: "input_blocks", batchSize: batchSize, channels: channels, channelMult: channelMult,
    numResBlocks: numResBlocks, numHeadChannels: numHeadChannels, t: t, startHeight: startHeight,
    startWidth: startWidth, attentionResolutions: attentionResolutions, x: x, emb: emb, xfOut: xfOut
  )
  let ch = channelMult[channelMult.count - 1] * channels
  var out = inputBlocksOut
  let (middleBlockReader1, middleResBlock1) = ResBlock(
    prefix: "middle_block.0", batchSize: batchSize, outChannels: ch, up: false, down: false,
    skipConnection: false)
  out = middleResBlock1(out, emb)
  var height = startHeight
  var width = startWidth
  for _ in 1..<channelMult.count {
    height /= 2
    width /= 2
  }
  let (middleBlockReader2, middleAttentionBlock2) = AttentionBlock(
    prefix: "middle_block.1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize, t: t,
    height: height, width: width)
  out = middleAttentionBlock2(out, xfOut)
  let (middleBlockReader3, middleResBlock3) = ResBlock(
    prefix: "middle_block.2", batchSize: batchSize, outChannels: ch, up: false, down: false,
    skipConnection: false)
  out = middleResBlock3(out, emb)
  let (outputBlocksReader, outputBlocksOut) = OutputBlocks(
    prefix: "output_blocks", batchSize: batchSize, channels: channels, channelMult: channelMult,
    numResBlocks: numResBlocks, numHeadChannels: numHeadChannels, t: t, startHeight: startHeight,
    startWidth: startWidth, attentionResolutions: attentionResolutions, x: out, emb: emb,
    xfOut: xfOut, hs: hs)
  out = outputBlocksOut
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    inputBlocksReader(state_dict)
    middleBlockReader1(state_dict)
    middleBlockReader2(state_dict)
    middleBlockReader3(state_dict)
    outputBlocksReader(state_dict)
    let out_0_weight = state_dict["out.0.weight"].type(torch.float).cpu().numpy()
    let out_0_bias = state_dict["out.0.bias"].type(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: out_0_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: out_0_bias))
    let out_2_weight = state_dict["out.2.weight"].type(torch.float).cpu().numpy()
    let out_2_bias = state_dict["out.2.bias"].type(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: out_2_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: out_2_bias))
  }
  return (reader, Model([x, emb, xfOut], [out]))
}

func ResnetBlock(prefix: String, inChannels: Int, outChannels: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x).swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = norm2(out).swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  let ninShortcut: Model?
  if inChannels != outChannels {
    let shortcut = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1])
    out = shortcut(x) + out
    ninShortcut = shortcut
  } else {
    out = x + out
    ninShortcut = nil
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu()
      .numpy()
    let norm1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let conv1_weight = state_dict["\(prefix).conv1.weight"].type(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.bias"].type(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu()
      .numpy()
    let norm2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["\(prefix).conv2.weight"].type(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.bias"].type(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).nin_shortcut.weight"].type(torch.float).cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).nin_shortcut.bias"].type(torch.float).cpu()
        .numpy()
      ninShortcut.weight.copy(from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
  }
  return (reader, Model([x], [out]))
}

func AttnBlock(
  prefix: String, inChannels: Int, batchSize: Int, height: Int, width: Int
) -> ((PythonObject) -> Void, Model) {
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
    let norm_weight = state_dict["\(prefix).norm.weight"].type(torch.float).cpu()
      .numpy()
    let norm_bias = state_dict["\(prefix).norm.bias"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: norm_bias))
    let k_weight = state_dict["\(prefix).k.weight"].type(torch.float).cpu().numpy()
    let k_bias = state_dict["\(prefix).k.bias"].type(torch.float).cpu().numpy()
    tokeys.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["\(prefix).q.weight"].type(torch.float).cpu().numpy()
    let q_bias = state_dict["\(prefix).q.bias"].type(torch.float).cpu().numpy()
    toqueries.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["\(prefix).v.weight"].type(torch.float).cpu().numpy()
    let v_bias = state_dict["\(prefix).v.bias"].type(torch.float).cpu().numpy()
    tovalues.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["\(prefix).proj_out.weight"].type(torch.float).cpu().numpy()
    let proj_out_bias = state_dict["\(prefix).proj_out.bias"].type(torch.float).cpu().numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x], [out]))
}

func Encoder(
  zChannels: Int, channels: Int, channelMult: [Int], numResBlocks: Int, startHeight: Int,
  startWidth: Int, attnResolutions: Set<Int>
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var lastCh = channels
  var readers = [(PythonObject) -> Void]()
  var currentRes = 256
  var height = startHeight
  var width = startWidth
  for (i, mult) in channelMult.enumerated() {
    let ch = channels * mult
    for j in 0..<numResBlocks {
      let (resnetReader, resnetBlock) = ResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", inChannels: lastCh, outChannels: ch)
      out = resnetBlock(out)
      readers.append(resnetReader)
      lastCh = ch
      if attnResolutions.contains(currentRes) {
        let (attnReader, attnBlock) = AttnBlock(
          prefix: "encoder.down.\(i).attn.\(j)", inChannels: ch, batchSize: 1, height: height,
          width: width)
        out = attnBlock(out)
        readers.append(attnReader)
      }
    }
    if i != channelMult.count - 1 {
      currentRes /= 2
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: ch, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])))
      out = conv2d(out).reshaped(
        [1, ch, height, width], offset: [0, 0, 1, 1],
        strides: [ch * (height + 1) * (width + 1), (height + 1) * (width + 1), width + 1, 1])
      let downLayer = i
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict["encoder.down.\(downLayer).downsample.conv.weight"].type(
          torch.float
        ).cpu().numpy()
        let conv_bias = state_dict["encoder.down.\(downLayer).downsample.conv.bias"].type(
          torch.float
        ).cpu().numpy()
        conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let (midReader1, midResnetBlock1) = ResnetBlock(
    prefix: "encoder.mid.block_1", inChannels: lastCh, outChannels: lastCh)
  out = midResnetBlock1(out)
  readers.append(midReader1)
  let (midAttnReader1, midAttnBlock1) = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: lastCh, batchSize: 1, height: height, width: width)
  out = midAttnBlock1(out)
  readers.append(midAttnReader1)
  let (midReader2, midResnetBlock2) = ResnetBlock(
    prefix: "encoder.mid.block_2", inChannels: lastCh, outChannels: lastCh)
  out = midResnetBlock2(out)
  readers.append(midReader2)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: zChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let quantConv = Convolution(groups: 1, filters: zChannels, filterSize: [1, 1])
  out = quantConv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["encoder.conv_in.weight"].type(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["encoder.conv_in.bias"].type(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["encoder.norm_out.weight"].type(torch.float).cpu()
      .numpy()
    let norm_out_bias = state_dict["encoder.norm_out.bias"].type(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["encoder.conv_out.weight"].type(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["encoder.conv_out.bias"].type(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
    let quant_conv_weight = state_dict["quant_conv.weight"].type(torch.float).cpu().numpy()
    let quant_conv_bias = state_dict["quant_conv.bias"].type(torch.float).cpu().numpy()
    quantConv.weight.copy(from: try! Tensor<Float>(numpy: quant_conv_weight))
    quantConv.bias.copy(from: try! Tensor<Float>(numpy: quant_conv_bias))
  }
  return (reader, Model([x], [out]))
}

func SpatialNorm(prefix: String, channels: Int, heightScale: Float, widthScale: Float) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let zq = Input()
  let normLayer = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = normLayer(x)
  let zqOut = Upsample(.nearest, widthScale: widthScale, heightScale: heightScale)(zq)
  zqOut.add(dependencies: [out])
  let convY = Convolution(groups: 1, filters: channels, filterSize: [1, 1])
  out = out .* convY(zqOut)
  let convB = Convolution(groups: 1, filters: channels, filterSize: [1, 1])
  let bias = convB(zqOut)
  bias.add(dependencies: [out])
  out = out + bias
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_layer_weight = state_dict["\(prefix).norm_layer.weight"].type(torch.float).cpu()
      .numpy()
    let norm_layer_bias = state_dict["\(prefix).norm_layer.bias"].type(torch.float).cpu().numpy()
    normLayer.weight.copy(from: try! Tensor<Float>(numpy: norm_layer_weight))
    normLayer.bias.copy(from: try! Tensor<Float>(numpy: norm_layer_bias))
    let conv_y_weight = state_dict["\(prefix).conv_y.weight"].type(torch.float).cpu().numpy()
    let conv_y_bias = state_dict["\(prefix).conv_y.bias"].type(torch.float).cpu().numpy()
    convY.weight.copy(from: try! Tensor<Float>(numpy: conv_y_weight))
    convY.bias.copy(from: try! Tensor<Float>(numpy: conv_y_bias))
    let conv_b_weight = state_dict["\(prefix).conv_b.weight"].type(torch.float).cpu().numpy()
    let conv_b_bias = state_dict["\(prefix).conv_b.bias"].type(torch.float).cpu().numpy()
    convB.weight.copy(from: try! Tensor<Float>(numpy: conv_b_weight))
    convB.bias.copy(from: try! Tensor<Float>(numpy: conv_b_bias))
  }
  return (reader, Model([x, zq], [out]))
}

func MOVQResnetBlock(prefix: String, inChannels: Int, outChannels: Int, scale: Float) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let zq = Input()
  let (norm1Reader, norm1) = SpatialNorm(
    prefix: "\(prefix).norm1", channels: inChannels, heightScale: scale, widthScale: scale)
  var out = norm1(x, zq).swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let (norm2Reader, norm2) = SpatialNorm(
    prefix: "\(prefix).norm2", channels: outChannels, heightScale: scale, widthScale: scale)
  out = norm2(out, zq).swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  let ninShortcut: Model?
  if inChannels != outChannels {
    let shortcut = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1])
    out = shortcut(x) + out
    ninShortcut = shortcut
  } else {
    out = x + out
    ninShortcut = nil
  }
  let reader: (PythonObject) -> Void = { state_dict in
    norm1Reader(state_dict)
    let conv1_weight = state_dict["\(prefix).conv1.weight"].type(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.bias"].type(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    norm2Reader(state_dict)
    let conv2_weight = state_dict["\(prefix).conv2.weight"].type(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.bias"].type(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).nin_shortcut.weight"].type(torch.float).cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).nin_shortcut.bias"].type(torch.float).cpu()
        .numpy()
      ninShortcut.weight.copy(from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
  }
  return (reader, Model([x, zq], [out]))
}

func MOVQAttnBlock(
  prefix: String, inChannels: Int, batchSize: Int, height: Int, width: Int, scale: Float
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let zq = Input()
  let (normReader, norm) = SpatialNorm(
    prefix: "\(prefix).norm", channels: inChannels, heightScale: scale, widthScale: scale)
  var out = norm(x, zq)
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
    normReader(state_dict)
    let k_weight = state_dict["\(prefix).k.weight"].type(torch.float).cpu().numpy()
    let k_bias = state_dict["\(prefix).k.bias"].type(torch.float).cpu().numpy()
    tokeys.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["\(prefix).q.weight"].type(torch.float).cpu().numpy()
    let q_bias = state_dict["\(prefix).q.bias"].type(torch.float).cpu().numpy()
    toqueries.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["\(prefix).v.weight"].type(torch.float).cpu().numpy()
    let v_bias = state_dict["\(prefix).v.bias"].type(torch.float).cpu().numpy()
    tovalues.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["\(prefix).proj_out.weight"].type(torch.float).cpu().numpy()
    let proj_out_bias = state_dict["\(prefix).proj_out.bias"].type(torch.float).cpu().numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x, zq], [out]))
}

func MOVQDecoder(
  zChannels: Int, channels: Int, channelMult: [Int], numResBlocks: Int, startHeight: Int,
  startWidth: Int, attnResolutions: Set<Int>
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let postQuantConv = Convolution(groups: 1, filters: zChannels, filterSize: [1, 1])
  let z = postQuantConv(x)
  var blockIn = channels * channelMult[channelMult.count - 1]
  let convIn = Convolution(
    groups: 1, filters: blockIn, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(z)
  let (midBlockReader1, midBlock1) = MOVQResnetBlock(
    prefix: "decoder.mid.block_1", inChannels: blockIn, outChannels: blockIn, scale: 1)
  out = midBlock1(out, x)
  let (midAttnReader1, midAttn1) = MOVQAttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: blockIn, batchSize: 1, height: startHeight,
    width: startWidth, scale: 1)
  out = midAttn1(out, x)
  let (midBlockReader2, midBlock2) = MOVQResnetBlock(
    prefix: "decoder.mid.block_2", inChannels: blockIn, outChannels: blockIn, scale: 1)
  out = midBlock2(out, x)
  var readers = [(PythonObject) -> Void]()
  var ds = 1
  var currentRes = 32
  var height = startHeight
  var width = startWidth
  for (i, mult) in channelMult.enumerated().reversed() {
    let blockOut = channels * mult
    for j in 0..<(numResBlocks + 1) {
      let (reader, resnetBlock) = MOVQResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", inChannels: blockIn, outChannels: blockOut,
        scale: Float(ds))
      out = resnetBlock(out, x)
      readers.append(reader)
      blockIn = blockOut
      if attnResolutions.contains(currentRes) {
        let (reader, attn) = MOVQAttnBlock(
          prefix: "decoder.up.\(i).attn.\(j)", inChannels: blockIn, batchSize: 1, height: height,
          width: width, scale: Float(ds))
        out = attn(out, x)
        readers.append(reader)
      }
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv = Convolution(
        groups: 1, filters: blockIn, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv(out)
      readers.append({ state_dict in
        let upsample_conv_weight = state_dict["decoder.up.\(i).upsample.conv.weight"].type(
          torch.float
        ).cpu().numpy()
        let upsample_conv_bias = state_dict["decoder.up.\(i).upsample.conv.bias"].type(torch.float)
          .cpu().numpy()
        conv.weight.copy(from: try! Tensor<Float>(numpy: upsample_conv_weight))
        conv.bias.copy(from: try! Tensor<Float>(numpy: upsample_conv_bias))
      })
      ds *= 2
      currentRes *= 2
      height *= 2
      width *= 2
    }
  }
  let (normOutReader, normOut) = SpatialNorm(
    prefix: "decoder.norm_out", channels: blockIn, heightScale: Float(ds), widthScale: Float(ds))
  out = normOut(out, x).swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let post_quant_conv_weight = state_dict["post_quant_conv.weight"].type(torch.float).cpu()
      .numpy()
    let post_quant_conv_bias = state_dict["post_quant_conv.bias"].type(torch.float).cpu().numpy()
    postQuantConv.weight.copy(from: try! Tensor<Float>(numpy: post_quant_conv_weight))
    postQuantConv.bias.copy(from: try! Tensor<Float>(numpy: post_quant_conv_bias))
    let conv_in_weight = state_dict["decoder.conv_in.weight"].type(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["decoder.conv_in.bias"].type(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    normOutReader(state_dict)
    let conv_out_weight = state_dict["decoder.conv_out.weight"].type(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["decoder.conv_out.bias"].type(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

let dmInput = torch.randn([2, 81, 2048])
var dmMask: PythonObject? = nil
let graph = DynamicGraph()
let diffusion = CLIPDiffusionModel(timesteps: 1_000, steps: 30)
let alphasCumprod = diffusion.alphasCumprod
var newBetas = [Double]()
var lastAlphasCumprod: Float = 1.0
for i in [0, 250, 500, 749, 999] {
  newBetas.append(1 - Double(alphasCumprod[i] / lastAlphasCumprod))
  lastAlphasCumprod = alphasCumprod[i]
}
var cumprod: Double = 1
let newAlphasCumprod = newBetas.map {
  cumprod *= 1 - $0
  return cumprod
}
var posteriorVariance = [Double]()
var posteriorLogVarianceClipped = [Double]()
var posteriorMeanCoef1 = [Double]()
var posteriorMeanCoef2 = [Double]()
DynamicGraph.setSeed(0)
for i in 0..<newAlphasCumprod.count {
  let alphasCumProdPrev = i > 0 ? newAlphasCumprod[i - 1] : 1
  posteriorVariance.append(newBetas[i] * (1 - alphasCumProdPrev) / (1 - newAlphasCumprod[i]))
  if i == 0 {
    posteriorLogVarianceClipped.append(
      log(newBetas[i + 1] * (1 - newAlphasCumprod[i]) / (1 - newAlphasCumprod[i + 1])))
  } else {
    posteriorLogVarianceClipped.append(
      log(newBetas[i] * (1 - newAlphasCumprod[i - 1]) / (1 - newAlphasCumprod[i])))
  }
  posteriorMeanCoef1.append(
    newBetas[i] * alphasCumProdPrev.squareRoot() / (1 - newAlphasCumprod[i]))
  posteriorMeanCoef2.append(
    (1 - alphasCumProdPrev) * (1 - newBetas[i]).squareRoot() / (1 - newAlphasCumprod[i]))
}
torch.cuda.set_device(0)
let noise = torch.randn([2, 768]).type(torch.float16).cuda()
let zeroImgEmb = model.create_zero_img_emb(1).detach().type(torch.float).cpu().numpy()
model.prior.noise = noise
var fullEmb1: DynamicGraph.Tensor<Float>? = nil
var poolEmb1: DynamicGraph.Tensor<Float>? = nil
var imageEmb1: DynamicGraph.Tensor<Float>? = nil
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
  let textEncoderEmb = layer(inputs: embeddings, attentionMaskGPU)[0].as(of: Float.self).reshaped(
    .CHW(2, 77, 1024))
  fullEmb1 = textEncoderEmb
  let poolingMask = graph.variable(.CPU, .CHW(2, 1, 77), of: Float.self)
  let weightPoolingMask = graph.variable(.CPU, .CHW(2, 1, 1), of: Float.self)
  poolingMask.full(0)
  for i in 0..<8 {
    poolingMask[0, 0, i] = 1
  }
  weightPoolingMask[0, 0, 0] = 1 / 8
  for i in 0..<2 {
    poolingMask[1, 0, i] = 1
  }
  weightPoolingMask[1, 0, 0] = 1 / 2
  let poolEmb = weightPoolingMask.toGPU(1) .* (poolingMask.toGPU(1) * textEncoderEmb)
  let linearTransformation = Dense(count: 768)
  linearTransformation.compile(inputs: poolEmb)
  let LinearTransformation_weight = state_dict["model.LinearTransformation.weight"].type(
    torch.float
  ).cpu()
    .numpy()
  linearTransformation.weight.copy(from: try! Tensor<Float>(numpy: LinearTransformation_weight))
  let LinearTransformation_bias = state_dict["model.LinearTransformation.bias"].type(torch.float)
    .cpu()
    .numpy()
  linearTransformation.bias.copy(from: try! Tensor<Float>(numpy: LinearTransformation_bias))
  let poolEmbOut = linearTransformation(inputs: poolEmb)[0].as(of: Float.self)
  debugPrint(poolEmbOut)
  poolEmb1 = poolEmbOut
  graph.openStore("/home/liu/workspace/swift-diffusion/xlm_roberta_f32.ckpt") {
    $0.write("embedding", model: textEncoder)
    $0.write("roberta", model: layer)
    $0.write("linear_transformation", model: linearTransformation)
  }

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
  let textEncProj = Dense(count: 2048)
  let textEmbProj = Dense(count: 2048)
  let clipImgProj = Dense(count: 2048)
  let (timeEmbedReader, timeEmbed) = timestepEmbedding(prefix: "model.time_embed", channels: 2048)
  var xIn = graph.variable(.GPU(1), .NC(2, 768), of: Float.self)
  textEncProj.compile(inputs: textEnc)
  let text_enc_proj_weight = prior_state_dict["model.text_enc_proj.weight"].type(torch.float).cpu()
    .numpy()
  textEncProj.weight.copy(from: try! Tensor<Float>(numpy: text_enc_proj_weight))
  let text_enc_proj_bias = prior_state_dict["model.text_enc_proj.bias"].type(torch.float).cpu()
    .numpy()
  textEncProj.bias.copy(from: try! Tensor<Float>(numpy: text_enc_proj_bias))
  let textEncOut = textEncProj(inputs: textEnc)[0].as(of: Float.self)
  textEmbProj.compile(inputs: textEmb)
  let text_emb_proj_weight = prior_state_dict["model.text_emb_proj.weight"].type(torch.float).cpu()
    .numpy()
  textEmbProj.weight.copy(from: try! Tensor<Float>(numpy: text_emb_proj_weight))
  let text_emb_proj_bias = prior_state_dict["model.text_emb_proj.bias"].type(torch.float).cpu()
    .numpy()
  textEmbProj.bias.copy(from: try! Tensor<Float>(numpy: text_emb_proj_bias))
  let textEmbOut = textEmbProj(inputs: textEmb)[0].as(of: Float.self)
  clipImgProj.compile(inputs: xIn)
  let clip_img_proj_weight = prior_state_dict["model.clip_img_proj.weight"].type(torch.float).cpu()
    .numpy()
  clipImgProj.weight.copy(from: try! Tensor<Float>(numpy: clip_img_proj_weight))
  let clip_img_proj_bias = prior_state_dict["model.clip_img_proj.bias"].type(torch.float).cpu()
    .numpy()
  clipImgProj.bias.copy(from: try! Tensor<Float>(numpy: clip_img_proj_bias))
  let timesteps = graph.variable(
    timeEmbedding(timestep: 999, batchSize: 2, embeddingSize: 2048, maxPeriod: 10_000).toGPU(1))
  timeEmbed.compile(inputs: timesteps)
  timeEmbedReader(prior_state_dict)
  var dmInputTensorGPU = graph.variable(try! Tensor<Float>(numpy: dmInput.numpy())).reshaped(
    .NC(2 * 81, 2048)
  ).toGPU(1)
  dmInputTensorGPU[0..<77, 0..<2048] = textEncOut[0..<1, 0..<77, 0..<2048].reshaped(.NC(77, 2048))
  dmInputTensorGPU[81..<(81 + 77), 0..<2048] = textEncOut[1..<2, 0..<77, 0..<2048].reshaped(
    .NC(77, 2048))
  dmInputTensorGPU[77..<78, 0..<2048] = textEmbOut[0..<1, 0..<2048]
  dmInputTensorGPU[(81 + 77)..<(81 + 78), 0..<2048] = textEmbOut[1..<2, 0..<2048]
  let prd_emb = prior_state_dict["model.prd_emb"].type(torch.float).cpu().numpy()
  let prdEmb = graph.variable(try! Tensor<Float>(numpy: prd_emb)).toGPU(1)
  dmInputTensorGPU[80..<81, 0..<2048] = prdEmb.reshaped(.NC(1, 2048))
  dmInputTensorGPU[(81 + 80)..<(81 + 81), 0..<2048] = prdEmb.reshaped(.NC(1, 2048))
  let positional_embedding = prior_state_dict["model.positional_embedding"].type(torch.float).cpu()
    .numpy()
  let positionalEmbedding = graph.variable(try! Tensor<Float>(numpy: positional_embedding)).toGPU(1)
    .reshaped(.NC(81, 2048))
  let clip_std = model.prior.clip_std.type(torch.float).cpu().numpy()
  let clipStd = graph.variable(try! Tensor<Float>(numpy: clip_std)).toGPU(1)
  let clip_mean = model.prior.clip_mean.type(torch.float).cpu().numpy()
  let clipMean = graph.variable(try! Tensor<Float>(numpy: clip_mean)).toGPU(1)
  var positionalEmbeddingGPU = graph.variable(.GPU(1), .NC(2 * 81, 2048), of: Float.self)
  positionalEmbeddingGPU[0..<81, 0..<2048] = positionalEmbedding
  positionalEmbeddingGPU[81..<(81 * 2), 0..<2048] = positionalEmbedding
  let (diffusionMappingReader, diffusionMapping) = DiffusionMapping(
    numberOfLayers: 20, k: 64, h: 32, b: 2, t: 81, outChannels: 768)
  let dmCasualAttentionMask = graph.variable(Tensor<Float>(.CPU, .NHWC(2, 1, 81, 81)))
  dmCasualAttentionMask.full(0)
  for i in 0..<80 {
    for j in (i + 1)..<81 {
      dmCasualAttentionMask[0, 0, i, j] = -Float.greatestFiniteMagnitude
      dmCasualAttentionMask[1, 0, i, j] = -Float.greatestFiniteMagnitude
    }
  }
  for i in 0..<81 {
    for j in 8..<77 {
      dmCasualAttentionMask[0, 0, i, j] = -Float.greatestFiniteMagnitude
    }
    for j in 2..<77 {
      dmCasualAttentionMask[1, 0, i, j] = -Float.greatestFiniteMagnitude
    }
  }
  dmMask = torch.from_numpy(dmCasualAttentionMask.reshaped(.CHW(2, 81, 81)).rawValue)
  let dmCasualAttentionMaskGPU = dmCasualAttentionMask.toGPU(1)
  diffusionMapping.compile(inputs: dmInputTensorGPU, dmCasualAttentionMaskGPU)
  diffusionMappingReader(prior_state_dict)
  let noiseGPU = graph.variable(try! Tensor<Float>(numpy: noise.type(torch.float).cpu().numpy()))
    .toGPU(1)
  var x = noiseGPU[0..<1, 0..<768]
  let zeroImgEmb = graph.variable(try! Tensor<Float>(numpy: zeroImgEmb))
  let zeroImgEmbGPU = zeroImgEmb.toGPU(1)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_diffusion_mapping_f32.ckpt") {
    $0.write("diffusion_mapping", model: diffusionMapping)
    $0.write("time_embed", model: timeEmbed)
    $0.write("clip_img_proj", model: clipImgProj)
    $0.write("text_enc_proj", model: textEncProj)
    $0.write("text_emb_proj", model: textEmbProj)
    $0.write("positional_embedding", variable: positionalEmbedding)
    $0.write("clip_std", variable: clipStd)
    $0.write("clip_mean", variable: clipMean)
    $0.write("zero_img_emb", variable: zeroImgEmb)
    $0.write("text_projection", variable: textProjectionGPU)
    $0.write("prd_emb", variable: prdEmb)
  }
  for (i, timestep) in [0, 250, 500, 749, 999].enumerated().reversed() {
    xIn[0..<1, 0..<768] = x
    xIn[1..<2, 0..<768] = x
    let timesteps = graph.variable(
      timeEmbedding(timestep: timestep, batchSize: 1, embeddingSize: 2048, maxPeriod: 10_000).toGPU(
        1))
    let tEmb = timeEmbed(inputs: timesteps)[0].as(of: Float.self)
    let xProj = clipImgProj(inputs: xIn)[0].as(of: Float.self)
    dmInputTensorGPU[78..<79, 0..<2048] = tEmb
    dmInputTensorGPU[(81 + 78)..<(81 + 79), 0..<2048] = tEmb
    dmInputTensorGPU[79..<80, 0..<2048] = xProj[0..<1, 0..<2048]
    dmInputTensorGPU[(81 + 79)..<(81 + 80), 0..<2048] = xProj[1..<2, 0..<2048]
    let input = dmInputTensorGPU + positionalEmbeddingGPU
    let result = diffusionMapping(inputs: input, dmCasualAttentionMaskGPU)[0].as(
      of: Float.self)
    let condEps = result[0..<1, 0..<1, 0..<768].reshaped(.NC(1, 768))
    let uncondEps = result[1..<2, 0..<1, 0..<768].reshaped(.NC(1, 768))
    let eps = (uncondEps + 4 * (condEps - uncondEps)).clamped(-10...10)
    let posteriorMean = Functional.add(
      left: eps, right: x, leftScalar: Float(posteriorMeanCoef1[i]),
      rightScalar: Float(posteriorMeanCoef2[i]))
    noiseGPU.randn()
    if i > 0 {
      x = Functional.add(
        left: posteriorMean, right: noiseGPU[0..<1, 0..<768],
        rightScalar: Float(exp(0.5 * posteriorLogVarianceClipped[i])))
    } else {
      x = posteriorMean
    }
  }
  let imageEmbGPU = x .* clipStd + clipMean
  var imageEmb = graph.variable(.GPU(1), .NC(2, 768), of: Float.self)
  imageEmb[0..<1, 0..<768] = imageEmbGPU
  imageEmb[1..<2, 0..<768] = zeroImgEmbGPU
  imageEmb1 = imageEmb.reshaped(.CHW(2, 1, 768))
}

/*
let dmMask2 = torch.cat([dmMask!, dmMask!])
let result = model.prior.model.transformer(
  dmInput.to(torch.float16).cuda(), mask: dmMask2.to(torch.float16).cuda())
print(result)
*/

torch.cuda.set_device(0)
let hInput = torch.randn([1, 4, 96, 96]).type(torch.float16).cuda()
let emb = torch.randn([2, 768 * 2]).type(torch.float).cuda()
let xfOut = torch.randn([2, 768, 87]).type(torch.float16).cuda()

let diffusionModel = DiffusionModel(
  linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 50)
let mAlphasCumprod = diffusionModel.alphasCumprod
var mNewBetas = [Double]()
var mLastAlphasCumprod: Float = 1.0
let mTimesteps: [Int] = (0..<100).map {
  return (999 * $0 + 45) / 99
}
for i in mTimesteps {
  mNewBetas.append(1 - Double(mAlphasCumprod[i] / mLastAlphasCumprod))
  mLastAlphasCumprod = mAlphasCumprod[i]
}
var mCumprod: Double = 1
let mNewAlphasCumprod = mNewBetas.map {
  mCumprod *= 1 - $0
  return mCumprod
}
var mPosteriorVariance = [Double]()
var mPosteriorLogVarianceClipped = [Double]()
var mPosteriorMeanCoef1 = [Double]()
var mPosteriorMeanCoef2 = [Double]()
for i in 0..<mNewAlphasCumprod.count {
  let alphasCumProdPrev = i > 0 ? mNewAlphasCumprod[i - 1] : 1
  mPosteriorVariance.append(mNewBetas[i] * (1 - alphasCumProdPrev) / (1 - mNewAlphasCumprod[i]))
  if i == 0 {
    mPosteriorLogVarianceClipped.append(
      log(mNewBetas[i + 1] * (1 - mNewAlphasCumprod[i]) / (1 - mNewAlphasCumprod[i + 1])))
  } else {
    mPosteriorLogVarianceClipped.append(
      log(mNewBetas[i] * (1 - mNewAlphasCumprod[i - 1]) / (1 - mNewAlphasCumprod[i])))
  }
  mPosteriorMeanCoef1.append(
    mNewBetas[i] * alphasCumProdPrev.squareRoot() / (1 - mNewAlphasCumprod[i]))
  mPosteriorMeanCoef2.append(
    (1 - alphasCumProdPrev) * (1 - mNewBetas[i]).squareRoot() / (1 - mNewAlphasCumprod[i]))
}

func percentile(_ tensor: DynamicGraph.Tensor<Float>) -> Float {
  let tensor = tensor.toCPU()
  var value = [Float]()
  for i in 0..<1 {
    for j in 0..<4 {
      for x in 0..<96 {
        for y in 0..<96 {
          value.append(abs(tensor[i, j, x, y]))
        }
      }
    }
  }
  value = value.sorted()
  return value[Int(floor((Float(1 * 4 * 96 * 96 - 1) * 0.995)))]
}

var image: DynamicGraph.Tensor<Float>? = nil
graph.withNoGrad {
  guard let fullEmb1 = fullEmb1, let poolEmb1 = poolEmb1, let imageEmb1 = imageEmb1 else { return }
  let (imageAndTextEmbeddingReader, imageAndTextEmbedding) = ImageAndTextEmbedding(batchSize: 2)
  imageAndTextEmbedding.compile(inputs: poolEmb1, fullEmb1, imageEmb1)
  imageAndTextEmbeddingReader(model_state_dict)
  let outputs = imageAndTextEmbedding(inputs: poolEmb1, fullEmb1, imageEmb1).map {
    $0.as(of: Float.self)
  }
  let xfProj = outputs[0]
  let xfOutGPU = outputs[1]
  debugPrint(xfOutGPU)
  let timesteps = graph.variable(
    timeEmbedding(timestep: 999, batchSize: 2, embeddingSize: 384, maxPeriod: 10_000).toGPU(1))
  let (timeEmbedReader, timeEmbed) = timestepEmbedding(prefix: "time_embed", channels: 384 * 4)
  timeEmbed.compile(inputs: timesteps)
  timeEmbedReader(model_state_dict)
  var embGPU = timeEmbed(inputs: timesteps)[0].as(of: Float.self)
  embGPU = embGPU + xfProj.reshaped(.NC(2, 384 * 4))
  let (reader, unet) = UNet(
    batchSize: 2, channels: 384, outChannels: 8, channelMult: [1, 2, 3, 4], numResBlocks: 3,
    numHeadChannels: 64, t: 87, startHeight: 96, startWidth: 96,
    attentionResolutions: Set([2, 4, 8]))
  let hInputGPU = graph.variable(
    try! Tensor<Float>(numpy: hInput.detach().type(torch.float).cpu().numpy())
  ).toGPU(1)
  /*
  let embGPU = graph.variable(
    try! Tensor<Float>(numpy: emb.detach().type(torch.float).cpu().numpy())
  ).toGPU(1)
  let xfOutGPU = graph.variable(
    try! Tensor<Float>(numpy: xfOut.detach().type(torch.float).cpu().numpy())
  ).toGPU(1).permuted(0, 2, 1).copied()
  */
  var x = hInputGPU
  var xIn = graph.variable(.GPU(1), .NCHW(2, 4, 96, 96), of: Float.self)
  unet.compile(inputs: xIn, embGPU, xfOutGPU)
  reader(model_state_dict)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_f32.ckpt") {
    $0.write("unet", model: unet)
    $0.write("time_embed", model: timeEmbed)
    $0.write("image_and_text_embed", model: imageAndTextEmbedding)
  }
  for (i, timestep) in mTimesteps.enumerated().reversed() {
    let timesteps = graph.variable(
      timeEmbedding(timestep: timestep, batchSize: 2, embeddingSize: 384, maxPeriod: 10_000).toGPU(
        1))
    embGPU = timeEmbed(inputs: timesteps)[0].as(of: Float.self)
    embGPU = embGPU + xfProj.reshaped(.NC(2, 384 * 4))
    xIn[0..<1, 0..<4, 0..<96, 0..<96] = x
    xIn[1..<2, 0..<4, 0..<96, 0..<96] = x
    let result = unet(inputs: xIn, embGPU, xfOutGPU)[0].as(of: Float.self)
    let modelVar = result[0..<1, 4..<8, 0..<96, 0..<96].copied().clamped(-1...1)
    let minLog = Float(mPosteriorLogVarianceClipped[i])
    let maxLog = log(Float(mNewBetas[i]))
    let frac = 0.5 * (modelVar + 1)
    let modelLogVar = frac * maxLog + (1 - frac) * minLog
    let condEps = result[0..<1, 0..<4, 0..<96, 0..<96].copied()
    let uncondEps = result[1..<2, 0..<4, 0..<96, 0..<96].copied()
    let eps = uncondEps + 4 * (condEps - uncondEps)
    var predXStart = Functional.add(
      left: x, right: eps, leftScalar: Float((1.0 / mNewAlphasCumprod[i]).squareRoot()),
      rightScalar: -Float((1.0 / mNewAlphasCumprod[i] - 1).squareRoot())
    ).clamped(-2...2)
    let s = max(percentile(predXStart), 1)
    predXStart = (1.0 / s) * predXStart.clamped(-s...s)
    x = Functional.add(
      left: predXStart, right: x, leftScalar: Float(mPosteriorMeanCoef1[i]),
      rightScalar: Float(mPosteriorMeanCoef2[i]))
    if i > 0 {
      let noise = graph.variable(like: x)
      noise.randn()
      x = x + Functional.exp(0.5 * modelLogVar) .* noise
    }
  }
  image = x
}
print(model.scale)
debugPrint(image!)

/*
model.model.eval()
torch.set_grad_enabled(false)
var h = hInput
var hs = [PythonObject]()
for module in model.model.input_blocks {
  h = module(h, emb, xfOut)
  hs.append(h)
}
h = model.model.middle_block(h, emb, xfOut)
for module in model.model.output_blocks {
  h = torch.cat([h, hs.popLast()!], dim: 1)
  h = module(h, emb, xfOut)
}
h = model.model.out(h.type(torch.float))
torch.set_grad_enabled(true)
print("h \(h), h.shape \(h.shape)")
*/

/*
torch.cuda.set_device(0)
let h = torch.randn([1, 3, 768, 768]).type(torch.float16).cuda()
torch.set_grad_enabled(false)
model.image_encoder.eval()
let result = model.image_encoder.encode(h * model.scale)
torch.set_grad_enabled(true)
print(result.cpu())
graph.withNoGrad {
  let (reader, encoder) = Encoder(
    zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2, startHeight: 768,
    startWidth: 768, attnResolutions: Set([32]))
  let hGPU = graph.variable(try! Tensor<Float>(numpy: h.detach().type(torch.float).cpu().numpy()))
    .toGPU(1)
  encoder.compile(inputs: hGPU)
  reader(movq_state_dict)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_movq_f32.ckpt") {
    $0.write("encoder", model: encoder)
  }
  let result = encoder(inputs: hGPU)[0].as(of: Float.self)
  debugPrint(result)
}
*/

graph.withNoGrad {
  guard let image = image else { return }
  let (reader, movq) = MOVQDecoder(
    zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2, startHeight: 96,
    startWidth: 96, attnResolutions: Set([32]))
  /*
  let hGPU = graph.variable(try! Tensor<Float>(numpy: h.detach().type(torch.float).cpu().numpy()))
    .toGPU(1)
  */
  movq.compile(inputs: image)
  reader(movq_state_dict)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_movq_f32.ckpt") {
    $0.write("movq", model: movq)
  }
  let result = movq(inputs: image)[0].as(of: Float.self).toCPU()
  debugPrint(result)
  let startWidth = 96
  let startHeight = 96
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: startWidth * 8 * startHeight * 8)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let (r, g, b) = (result[0, 0, y, x], result[0, 1, y, x], result[0, 2, y, x])
      rgba[y * startWidth * 8 + x].r = UInt8(
        min(max(Int(Float((r + 1) / 2) * 255), 0), 255))
      rgba[y * startWidth * 8 + x].g = UInt8(
        min(max(Int(Float((g + 1) / 2) * 255), 0), 255))
      rgba[y * startWidth * 8 + x].b = UInt8(
        min(max(Int(Float((b + 1) / 2) * 255), 0), 255))
    }
  }
  let png = PNG.Data.Rectangular(
    packing: rgba, size: (startWidth * 8, startHeight * 8),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! png.compress(path: "/home/liu/workspace/swift-diffusion/kandinsky.png", level: 4)
}

/*
torch.cuda.set_device(0)
let images = model.generate_text2img(
  prompt, num_steps: 100, batch_size: 1, guidance_scale: 4, h: 768, w: 768,
  sampler: "p_sampler", prior_cf_scale: 4, prior_steps: "5")
images[0].save("/home/liu/workspace/swift-diffusion/kandinsky.png")
*/
