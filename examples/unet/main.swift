import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let ldm_util = Python.import("ldm.util")
let torch = Python.import("torch")
let omegaconf = Python.import("omegaconf")
let random = Python.import("random")
let numpy = Python.import("numpy")

func timeEmbedding(timesteps: Int, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * Float(timesteps)
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

func TimeEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func ResBlock(b: Int, outChannels: Int, skipConnection: Bool) -> (
  Model, Model, Model, Model, Model, Model?, Model
) {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: outChannels)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([b, outChannels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var skipModel: Model? = nil
  if skipConnection {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]))
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = skip
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
  }
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel,
    Model([x, emb], [out])
  )
}

func SelfAttention(k: Int, h: Int, b: Int, hw: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, hw])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func CrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model, Model, Model)
{
  let x = Input()
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(c).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, c], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let fc10 = Dense(count: intermediateSize)
  let fc11 = Dense(count: intermediateSize)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc10, fc11, fc2, Model([x], [out]))
}

func BasicTransformerBlock(k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model,
  Model
) {
  let x = Input()
  let c = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (tokeys2, toqueries2, tovalues2, unifyheads2, attn2) = CrossAttention(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, c) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  return (
    layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2, toqueries2,
    tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, Model([x, c], [out])
  )
}

func SpatialTransformer(
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, t: Int, intermediateSize: Int
) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model,
  Model, Model, Model, Model
) {
  let x = Input()
  let c = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).permuted(0, 2, 1)
  let (
    layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2, toqueries2,
    tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, block
  ) = BasicTransformerBlock(k: k, h: h, b: b, hw: hw, t: t, intermediateSize: intermediateSize)
  out = block(out, c).reshaped([b, height, width, k * h]).permuted(0, 3, 1, 2)
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  return (
    norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
    toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, projOut, Model([x, c], [out])
  )
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Bool, channels: Int, numHeads: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let emb = Input()
  let c = Input()
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  var norm: Model? = nil
  var projIn: Model? = nil
  var layerNorm1: Model? = nil
  var tokeys1: Model? = nil
  var toqueries1: Model? = nil
  var tovalues1: Model? = nil
  var unifyheads1: Model? = nil
  var layerNorm2: Model? = nil
  var tokeys2: Model? = nil
  var toqueries2: Model? = nil
  var tovalues2: Model? = nil
  var unifyheads2: Model? = nil
  var layerNorm3: Model? = nil
  var fc10: Model? = nil
  var fc11: Model? = nil
  var tfc2: Model? = nil
  var projOut: Model? = nil
  if attentionBlock {
    let transformer: Model
    (
      norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
      toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, tfc2, projOut, transformer
    ) = SpatialTransformer(
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer(out, c)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let in_layers_0_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.0.weight"
    ].numpy()
    let in_layers_0_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.0.bias"
    ].numpy()
    inLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_0_weight))
    inLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_bias))
    let in_layers_2_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.2.weight"
    ].numpy()
    let in_layers_2_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.in_layers.2.bias"
    ].numpy()
    inLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_2_weight))
    inLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_bias))
    let emb_layers_1_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.weight"
    ].numpy()
    let emb_layers_1_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.bias"
    ].numpy()
    embLayer.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_1_weight))
    embLayer.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_1_bias))
    let out_layers_0_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.0.weight"
    ].numpy()
    let out_layers_0_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.0.bias"
    ].numpy()
    outLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_layers_0_weight))
    outLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_bias))
    let out_layers_3_weight = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.3.weight"
    ].numpy()
    let out_layers_3_bias = state_dict[
      "diffusion_model.\(prefix).\(layerStart).0.out_layers.3.bias"
    ].numpy()
    outLayerConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_3_weight))
    outLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_3_bias))
    if let skipModel = skipModel {
      let skip_connection_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).0.skip_connection.weight"
      ].numpy()
      let skip_connection_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).0.skip_connection.bias"
      ].numpy()
      skipModel.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: skip_connection_weight))
      skipModel.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: skip_connection_bias))
    }
    if let norm = norm, let projIn = projIn, let layerNorm1 = layerNorm1, let tokeys1 = tokeys1,
      let toqueries1 = toqueries1, let tovalues1 = tovalues1, let unifyheads1 = unifyheads1,
      let layerNorm2 = layerNorm2, let tokeys2 = tokeys2, let toqueries2 = toqueries2,
      let tovalues2 = tovalues2, let unifyheads2 = unifyheads2, let layerNorm3 = layerNorm3,
      let fc10 = fc10, let fc11 = fc11, let tfc2 = tfc2, let projOut = projOut
    {
      let norm_weight = state_dict["diffusion_model.\(prefix).\(layerStart).1.norm.weight"]
        .numpy()
      let norm_bias = state_dict["diffusion_model.\(prefix).\(layerStart).1.norm.bias"].numpy()
      norm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_weight))
      norm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_bias))
      let proj_in_weight = state_dict["diffusion_model.\(prefix).\(layerStart).1.proj_in.weight"]
        .numpy()
      let proj_in_bias = state_dict["diffusion_model.\(prefix).\(layerStart).1.proj_in.bias"]
        .numpy()
      projIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_in_weight))
      projIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_in_bias))
      let attn1_to_k_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_k.weight"
      ].numpy()
      tokeys1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_k_weight))
      let attn1_to_q_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_q.weight"
      ].numpy()
      toqueries1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_q_weight))
      let attn1_to_v_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_v.weight"
      ].numpy()
      tovalues1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_v_weight))
      let attn1_to_out_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_out.0.weight"
      ].numpy()
      let attn1_to_out_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_out.0.bias"
      ].numpy()
      unifyheads1.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: attn1_to_out_weight))
      unifyheads1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn1_to_out_bias))
      let ff_net_0_proj_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.0.proj.weight"
      ].numpy()
      let ff_net_0_proj_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.0.proj.bias"
      ].numpy()
      fc10.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[..<intermediateSize, ...]))
      fc10.parameters(for: .bias).copy(
        from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[..<intermediateSize]))
      fc11.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[intermediateSize..., ...]))
      fc11.parameters(for: .bias).copy(
        from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[intermediateSize...]))
      let ff_net_2_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.2.weight"
      ].numpy()
      let ff_net_2_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.2.bias"
      ].numpy()
      tfc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ff_net_2_weight))
      tfc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_2_bias))
      let attn2_to_k_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_k.weight"
      ].numpy()
      tokeys2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_k_weight))
      let attn2_to_q_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_q.weight"
      ].numpy()
      toqueries2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_q_weight))
      let attn2_to_v_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_v.weight"
      ].numpy()
      tovalues2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
      let attn2_to_out_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_out.0.weight"
      ].numpy()
      let attn2_to_out_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_out.0.bias"
      ].numpy()
      unifyheads2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: attn2_to_out_weight))
      unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
      let norm1_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm1.weight"
      ]
      .numpy()
      let norm1_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm1.bias"
      ]
      .numpy()
      layerNorm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
      layerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
      let norm2_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm2.weight"
      ]
      .numpy()
      let norm2_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm2.bias"
      ]
      .numpy()
      layerNorm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
      layerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
      let norm3_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm3.weight"
      ]
      .numpy()
      let norm3_bias = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm3.bias"
      ]
      .numpy()
      layerNorm3.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm3_weight))
      layerNorm3.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm3_bias))
      let proj_out_weight = state_dict[
        "diffusion_model.\(prefix).\(layerStart).1.proj_out.weight"
      ].numpy()
      let proj_out_bias = state_dict["diffusion_model.\(prefix).\(layerStart).1.proj_out.bias"]
        .numpy()
      projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
      projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
    }
  }
  if attentionBlock {
    return (reader, Model([x, emb, c], [out]))
  } else {
    return (reader, Model([x, emb], [out]))
  }
}

func MiddleBlock(
  channels: Int, numHeads: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  x: Model.IO, emb: Model.IO, c: Model.IO
) -> ((PythonObject) -> Void, Model.IO) {
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let (
    norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
    toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, tfc2, projOut, transformer
  ) = SpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
    intermediateSize: channels * 4)
  out = transformer(out, c)
  let (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  let reader: (PythonObject) -> Void = { state_dict in
    let intermediateSize = channels * 4
    let in_layers_0_0_weight = state_dict["diffusion_model.middle_block.0.in_layers.0.weight"]
      .numpy()
    let in_layers_0_0_bias = state_dict["diffusion_model.middle_block.0.in_layers.0.bias"].numpy()
    inLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_0_0_weight))
    inLayerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_0_bias))
    let in_layers_0_2_weight = state_dict["diffusion_model.middle_block.0.in_layers.2.weight"]
      .numpy()
    let in_layers_0_2_bias = state_dict["diffusion_model.middle_block.0.in_layers.2.bias"].numpy()
    inLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_0_2_weight))
    inLayerConv2d1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_2_bias))
    let emb_layers_0_1_weight = state_dict["diffusion_model.middle_block.0.emb_layers.1.weight"]
      .numpy()
    let emb_layers_0_1_bias = state_dict["diffusion_model.middle_block.0.emb_layers.1.bias"].numpy()
    embLayer1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_0_1_weight))
    embLayer1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_0_1_bias))
    let out_layers_0_0_weight = state_dict["diffusion_model.middle_block.0.out_layers.0.weight"]
      .numpy()
    let out_layers_0_0_bias = state_dict[
      "diffusion_model.middle_block.0.out_layers.0.bias"
    ].numpy()
    outLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_0_weight))
    outLayerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_0_bias))
    let out_layers_0_3_weight = state_dict["diffusion_model.middle_block.0.out_layers.3.weight"]
      .numpy()
    let out_layers_0_3_bias = state_dict["diffusion_model.middle_block.0.out_layers.3.bias"].numpy()
    outLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_3_weight))
    outLayerConv2d1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_3_bias))
    let norm_weight = state_dict["diffusion_model.middle_block.1.norm.weight"]
      .numpy()
    let norm_bias = state_dict["diffusion_model.middle_block.1.norm.bias"].numpy()
    norm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_bias))
    let proj_in_weight = state_dict["diffusion_model.middle_block.1.proj_in.weight"]
      .numpy()
    let proj_in_bias = state_dict["diffusion_model.middle_block.1.proj_in.bias"]
      .numpy()
    projIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_in_weight))
    projIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_in_bias))
    let attn1_to_k_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight"
    ].numpy()
    tokeys1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_k_weight))
    let attn1_to_q_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"
    ].numpy()
    toqueries1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_q_weight))
    let attn1_to_v_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight"
    ].numpy()
    tovalues1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_v_weight))
    let attn1_to_out_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight"
    ].numpy()
    let attn1_to_out_bias = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias"
    ].numpy()
    unifyheads1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attn1_to_out_weight))
    unifyheads1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn1_to_out_bias))
    let ff_net_0_proj_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight"
    ].numpy()
    let ff_net_0_proj_bias = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias"
    ].numpy()
    fc10.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[..<intermediateSize, ...]))
    fc10.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[..<intermediateSize]))
    fc11.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[intermediateSize..., ...]))
    fc11.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[intermediateSize...]))
    let ff_net_2_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight"
    ].numpy()
    let ff_net_2_bias = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias"
    ].numpy()
    tfc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ff_net_2_weight))
    tfc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_2_bias))
    let attn2_to_k_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight"
    ].numpy()
    tokeys2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_k_weight))
    let attn2_to_q_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight"
    ].numpy()
    toqueries2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_q_weight))
    let attn2_to_v_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight"
    ].numpy()
    tovalues2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
    let attn2_to_out_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight"
    ].numpy()
    let attn2_to_out_bias = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias"
    ].numpy()
    unifyheads2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attn2_to_out_weight))
    unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
    let norm1_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight"
    ]
    .numpy()
    let norm1_bias = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias"
    ]
    .numpy()
    layerNorm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    layerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let norm2_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight"
    ]
    .numpy()
    let norm2_bias = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias"
    ]
    .numpy()
    layerNorm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
    layerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let norm3_weight = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight"
    ]
    .numpy()
    let norm3_bias = state_dict[
      "diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias"
    ]
    .numpy()
    layerNorm3.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm3_weight))
    layerNorm3.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm3_bias))
    let proj_out_weight = state_dict[
      "diffusion_model.middle_block.1.proj_out.weight"
    ].numpy()
    let proj_out_bias = state_dict["diffusion_model.middle_block.1.proj_out.bias"]
      .numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
    let in_layers_2_0_weight = state_dict["diffusion_model.middle_block.2.in_layers.0.weight"]
      .numpy()
    let in_layers_2_0_bias = state_dict["diffusion_model.middle_block.2.in_layers.0.bias"].numpy()
    inLayerNorm2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_2_0_weight))
    inLayerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_0_bias))
    let in_layers_2_2_weight = state_dict["diffusion_model.middle_block.2.in_layers.2.weight"]
      .numpy()
    let in_layers_2_2_bias = state_dict["diffusion_model.middle_block.2.in_layers.2.bias"].numpy()
    inLayerConv2d2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_2_2_weight))
    inLayerConv2d2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_2_bias))
    let emb_layers_2_1_weight = state_dict["diffusion_model.middle_block.2.emb_layers.1.weight"]
      .numpy()
    let emb_layers_2_1_bias = state_dict["diffusion_model.middle_block.2.emb_layers.1.bias"].numpy()
    embLayer2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_2_1_weight))
    embLayer2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_2_1_bias))
    let out_layers_2_0_weight = state_dict["diffusion_model.middle_block.2.out_layers.0.weight"]
      .numpy()
    let out_layers_2_0_bias = state_dict[
      "diffusion_model.middle_block.2.out_layers.0.bias"
    ].numpy()
    outLayerNorm2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_0_weight))
    outLayerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_2_0_bias))
    let out_layers_2_3_weight = state_dict["diffusion_model.middle_block.2.out_layers.3.weight"]
      .numpy()
    let out_layers_2_3_bias = state_dict["diffusion_model.middle_block.2.out_layers.3.bias"].numpy()
    outLayerConv2d2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_3_weight))
    outLayerConv2d2.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: out_layers_2_3_bias))
  }
  return (reader, out)
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO
) -> ((PythonObject) -> Void, [Model.IO], Model.IO) {
  let conv2d = Convolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for _ in 0..<numRepeat {
      let (reader, inputLayer) = BlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      if attentionBlock {
        out = inputLayer(out, emb, c)
      } else {
        out = inputLayer(out, emb)
      }
      passLayers.append(out)
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let reader: (PythonObject) -> Void = { state_dict in
        let op_weight = state_dict["diffusion_model.input_blocks.\(downLayer).0.op.weight"].numpy()
        let op_bias = state_dict["diffusion_model.input_blocks.\(downLayer).0.op.bias"].numpy()
        downsample.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: op_weight))
        downsample.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: op_bias))
      }
      readers.append(reader)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let input_blocks_0_0_weight = state_dict["diffusion_model.input_blocks.0.0.weight"].numpy()
    let input_blocks_0_0_bias = state_dict["diffusion_model.input_blocks.0.0.bias"].numpy()
    conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_weight))
    conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_bias))
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, passLayers, out)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO,
  inputs: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO) {
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
  var out = x
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let (reader, outputLayer) = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      if attentionBlock {
        out = outputLayer(out, emb, c)
      } else {
        out = outputLayer(out, emb)
      }
      readers.append(reader)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock ? 2 : 1
        let reader: (PythonObject) -> Void = { state_dict in
          let op_weight = state_dict[
            "diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight"
          ].numpy()
          let op_bias = state_dict["diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias"]
            .numpy()
          conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: op_weight))
          conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: op_bias))
        }
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
  return (reader, out)
}

func UNet(batchSize: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  let (inputReader, inputs, inputBlocks) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: 64,
    startWidth: 64, embeddingSize: 77, attentionRes: attentionRes, x: x, emb: emb, c: c)
  var out = inputBlocks
  let (middleReader, middleBlock) = MiddleBlock(
    channels: 1280, numHeads: 8, batchSize: batchSize, height: 8, width: 8, embeddingSize: 77,
    x: out,
    emb: emb, c: c)
  out = middleBlock
  let (outputReader, outputBlocks) = OutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: 64,
    startWidth: 64, embeddingSize: 77, attentionRes: attentionRes, x: out, emb: emb, c: c,
    inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let time_embed_0_weight = state_dict["diffusion_model.time_embed.0.weight"].numpy()
    let time_embed_0_bias = state_dict["diffusion_model.time_embed.0.bias"].numpy()
    let time_embed_2_weight = state_dict["diffusion_model.time_embed.2.weight"].numpy()
    let time_embed_2_bias = state_dict["diffusion_model.time_embed.2.bias"].numpy()
    fc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_0_weight))
    fc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_0_bias))
    fc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_2_weight))
    fc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_2_bias))
    inputReader(state_dict)
    middleReader(state_dict)
    outputReader(state_dict)
    let out_0_weight = state_dict["diffusion_model.out.0.weight"].numpy()
    let out_0_bias = state_dict["diffusion_model.out.0.bias"].numpy()
    outNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_0_weight))
    outNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_0_bias))
    let out_2_weight = state_dict["diffusion_model.out.2.weight"].numpy()
    let out_2_bias = state_dict["diffusion_model.out.2.bias"].numpy()
    outConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_2_weight))
    outConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_2_bias))
  }
  return (reader, Model([x, t_emb, c], [out]))
}

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([2, 4, 64, 64])
let t = torch.full([1], 981)
let c = torch.randn([2, 77, 768])

let config = omegaconf.OmegaConf.load(
  "/home/liu/workspace/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
let pl_sd = torch.load(
  "/home/liu/workspace/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
  map_location: "cpu")
let sd = pl_sd["state_dict"]
let model = ldm_util.instantiate_from_config(config.model)
model.load_state_dict(sd, strict: false)
model.eval()
let state_dict = model.model.state_dict()
let ret = model.apply_model(x, t, c)
print(ret)

let graph = DynamicGraph()

let t_emb = graph.variable(
  timeEmbedding(timesteps: 981, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000)
).toGPU(0)
let xTensor = graph.variable(try! Tensor<Float>(numpy: x.numpy())).toGPU(0)
let cTensor = graph.variable(try! Tensor<Float>(numpy: c.numpy())).toGPU(0)
let (reader, unet) = UNet(batchSize: 2)
graph.workspaceSize = 1_024 * 1_024 * 1_024
graph.withNoGrad {
  let _ = unet(inputs: xTensor, t_emb, cTensor)
  reader(state_dict)
  let attnOut = unet(inputs: xTensor, t_emb, cTensor)[0].as(of: Float.self)
  let attnOutCPU = attnOut.toCPU()
  print(attnOutCPU)
  for i in 0..<4 {
    let x = i
    for j in 0..<6 {
      let y = j < 3 ? j : 58 + j
      for k in 0..<6 {
        let z = k < 3 ? k : 58 + k
        print("0 \(x) \(y) \(z) \(attnOutCPU[0, x, y, z])")
      }
    }
  }
  for i in 0..<4 {
    let x = i
    for j in 0..<6 {
      let y = j < 3 ? j : 58 + j
      for k in 0..<6 {
        let z = k < 3 ? k : 58 + k
        print("1 \(x) \(y) \(z) \(attnOutCPU[1, x, y, z])")
      }
    }
  }
  graph.openStore("/home/liu/workspace/swift-diffusion/unet.ckpt") {
    $0.write("unet", model: unet)
  }
}
