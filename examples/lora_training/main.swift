import Diffusion
import Foundation
import NNC
import PNG

public let LowRank = 16

private func LoRAConvolution(
  groups: Int, filters: Int, filterSize: [Int], noBias: Bool = false, hint: Hint = Hint(),
  format: Convolution.Format? = nil, name: String = ""
) -> Model {
  let x = Input()
  let conv2d = Convolution(
    groups: groups, filters: filters, filterSize: filterSize, noBias: noBias, hint: hint,
    format: format, name: name)
  let conv2dDown = Convolution(
    groups: groups, filters: LowRank, filterSize: filterSize, noBias: true, hint: hint,
    format: format, trainable: true, name: "lora_down")
  let conv2dUp = Convolution(
    groups: groups, filters: filters, filterSize: [1, 1], noBias: true, hint: Hint(stride: [1, 1]),
    format: format, trainable: true, name: "lora_up")
  let out = conv2d(x) + conv2dUp(conv2dDown(x))
  return Model([x], [out])
}

private func LoRADense(count: Int, noBias: Bool = false, name: String = "") -> Model {
  let x = Input()
  let dense = Dense(count: count, noBias: noBias, name: name)
  let denseDown = Dense(count: LowRank, noBias: true, trainable: true, name: "lora_down")
  let denseUp = Dense(count: count, noBias: true, trainable: true, name: "lora_up")
  let out = dense(x) + denseUp(denseDown(x))
  return Model([x], [out])
}

/// Text Model

func CLIPTextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
)
  -> Model
{
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(T.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + positionEmbed(positions)
  return Model([tokens, positions], [embedding], name: "embeddings")
}

func CLIPAttention(k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = LoRADense(count: k * h)
  let toqueries = LoRADense(count: k * h)
  let tovalues = LoRADense(count: k * h)
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
  let unifyheads = LoRADense(count: k * h)
  out = unifyheads(out)
  return Model([x, casualAttentionMask], [out])
}

func QuickGELU() -> Model {
  let x = Input()
  let y = x .* Sigmoid()(1.702 * x)
  return Model([x], [y])
}

func CLIPMLP(hiddenSize: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let fc1 = LoRADense(count: intermediateSize)
  var out = fc1(x)
  out = QuickGELU()(out)
  let fc2 = LoRADense(count: hiddenSize)
  out = fc2(out)
  return Model([x], [out])
}

func CLIPEncoderLayer(k: Int, h: Int, b: Int, t: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let attention = CLIPAttention(k: k, h: h, b: b, t: t)
  out = attention(out, casualAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let mlp = CLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return Model([x, casualAttentionMask], [out])
}

public func LoRACLIPTextModel<T: TensorNumeric>(
  _ dataType: T.Type,
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    T.self, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  let k = embeddingSize / numHeads
  for _ in 0..<numLayers {
    let encoderLayer = CLIPEncoderLayer(
      k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [out], trainable: false)
}

/// UNet

func TimeEmbed(modelChannels: Int) -> Model {
  let x = Input()
  let fc0 = LoRADense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: modelChannels * 4)
  out = fc2(out)
  return Model([x], [out])
}

func ResBlock(b: Int, outChannels: Int, skipConnection: Bool) -> Model {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = out.swish()
  let inLayerConv2d = LoRAConvolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let embLayer = LoRADense(count: outChannels)
  var embOut = emb.swish()
  embOut = embLayer(embOut).reshaped([b, outChannels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = out.swish()
  // Dropout if needed in the future (for training).
  let outLayerConv2d = LoRAConvolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  if skipConnection {
    let skip = LoRAConvolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]))
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
  }
  return Model([x, emb], [out])
}

func SelfAttention(k: Int, h: Int, b: Int, hw: Int) -> Model {
  let x = Input()
  let tokeys = LoRADense(count: k * h, noBias: true)
  let toqueries = LoRADense(count: k * h, noBias: true)
  let tovalues = LoRADense(count: k * h, noBias: true)
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
  let unifyheads = LoRADense(count: k * h)
  out = unifyheads(out)
  return Model([x], [out])
}

func CrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
  let x = Input()
  let c = Input()
  let tokeys = LoRADense(count: k * h, noBias: true)
  let toqueries = LoRADense(count: k * h, noBias: true)
  let tovalues = LoRADense(count: k * h, noBias: true)
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
  let unifyheads = LoRADense(count: k * h)
  out = unifyheads(out)
  return Model([x, c], [out])
}

func FeedForward(hiddenSize: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let fc10 = LoRADense(count: intermediateSize)
  let fc11 = LoRADense(count: intermediateSize)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = LoRADense(count: hiddenSize)
  out = fc2(out)
  return Model([x], [out])
}

func BasicTransformerBlock(k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int) -> Model
{
  let x = Input()
  let c = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let attn1 = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let attn2 = CrossAttention(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, c) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let ff = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  return Model([x, c], [out])
}

func SpatialTransformer(
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, t: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let c = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = LoRAConvolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).permuted(0, 2, 1)
  let block = BasicTransformerBlock(
    k: k, h: h, b: b, hw: hw, t: t, intermediateSize: intermediateSize)
  out = block(out, c).transposed(1, 2).reshaped([b, k * h, height, width])
  let projOut = LoRAConvolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  return Model([x, c], [out])
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Bool, channels: Int, numHeads: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let emb = Input()
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let resBlock = ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  if attentionBlock {
    let c = Input()
    let transformer = SpatialTransformer(
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer(out, c)
    return Model([x, emb, c], [out])
  } else {
    return Model([x, emb], [out])
  }
}

func MiddleBlock(
  channels: Int, numHeads: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  x: Model.IO, emb: Model.IO, c: Model.IO
) -> Model.IO {
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let resBlock1 = ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let transformer = SpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
    intermediateSize: channels * 4)
  out = transformer(out, c)
  let resBlock2 = ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  return out
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO,
  adapters: [Model.IO]
) -> ([Model.IO], Model.IO) {
  let conv2d = LoRAConvolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<numRepeat {
      let inputLayer = BlockLayer(
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
      if j == numRepeat - 1 && adapters.count == channels.count {
        out = out + adapters[i]
      }
      passLayers.append(out)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = LoRAConvolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append(out)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  return (passLayers, out)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO,
  inputs: [Model.IO]
) -> Model.IO {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
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
      let outputLayer = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      if attentionBlock {
        out = outputLayer(out, emb, c)
      } else {
        out = outputLayer(out, emb)
      }
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = LoRAConvolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
      }
      layerStart += 1
    }
  }
  return out
}

func MiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  x: Model.IO, emb: Model.IO, c: Model.IO
) -> Model.IO {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let resBlock1 = ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let transformer = SpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
    intermediateSize: channels * 4)
  out = transformer(out, c)
  let resBlock2 = ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  return out
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO,
  adapters: [Model.IO]
) -> ([Model.IO], Model.IO) {
  let conv2d = LoRAConvolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<numRepeat {
      let inputLayer = BlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeads: channel / numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      if attentionBlock {
        out = inputLayer(out, emb, c)
      } else {
        out = inputLayer(out, emb)
      }
      if j == numRepeat - 1 && adapters.count == channels.count {
        out = out + adapters[i]
      }
      passLayers.append(out)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = LoRAConvolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append(out)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  return (passLayers, out)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO,
  inputs: [Model.IO]
) -> Model.IO {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
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
      let outputLayer = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeads: channel / numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      if attentionBlock {
        out = outputLayer(out, emb, c)
      } else {
        out = outputLayer(out, emb)
      }
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = LoRAConvolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
      }
      layerStart += 1
    }
  }
  return out
}

func LoRAUNet(
  batchSize: Int, startWidth: Int, startHeight: Int, control: Bool = false, adapter: Bool = false
) -> Model {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  var controls = [Model.IO]()
  if control {
    controls = (0..<13).map { _ in Input() }
  }
  var adapters = [Model.IO]()
  if adapter {
    adapters = (0..<4).map { _ in Input() }
  }
  let timeEmbed = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  var (inputs, inputBlocks) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes, x: x, emb: emb, c: c,
    adapters: adapters)
  var out = inputBlocks
  let middleBlock = MiddleBlock(
    channels: 1280, numHeads: 8, batchSize: batchSize, height: startHeight / 8,
    width: startWidth / 8, embeddingSize: 77,
    x: out,
    emb: emb, c: c)
  out = middleBlock
  if control {
    out = out + controls[12]
    for i in 0..<inputs.count {
      inputs[i] = inputs[i] + controls[i]
    }
  }
  let outputBlocks = OutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes, x: out, emb: emb, c: c,
    inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = out.swish()
  let outConv2d = LoRAConvolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  controls.insert(contentsOf: [x, t_emb, c], at: 0)
  controls.append(contentsOf: adapters)
  return Model(controls, [out], trainable: false)
}

public typealias FloatType = Float16

public struct DiffusionModel {
  public var linearStart: Float
  public var linearEnd: Float
  public var timesteps: Int
  public var steps: Int
}

extension DiffusionModel {
  public var betas: [Float] {  // Linear for now.
    var betas = [Float]()
    let start = linearStart.squareRoot()
    let length = linearEnd.squareRoot() - start
    for i in 0..<timesteps {
      let beta = start + Float(i) * length / Float(timesteps - 1)
      betas.append(beta * beta)
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
  // This is Karras scheduler sigmas.
  public func karrasSigmas(_ range: ClosedRange<Float>, rho: Float = 7.0) -> [Float] {
    let minInvRho = pow(range.lowerBound, 1.0 / rho)
    let maxInvRho = pow(range.upperBound, 1.0 / rho)
    var sigmas = [Float]()
    for i in 0..<steps {
      sigmas.append(pow(maxInvRho + Float(i) * (minInvRho - maxInvRho) / Float(steps - 1), rho))
    }
    sigmas.append(0)
    return sigmas
  }

  public func fixedStepSigmas(_ range: ClosedRange<Float>, sigmas sigmasForTimesteps: [Float])
    -> [Float]
  {
    var sigmas = [Float]()
    for i in 0..<steps {
      let timestep = Float(steps - 1 - i) / Float(steps - 1) * Float(timesteps - 1)
      let lowIdx = Int(floor(timestep))
      let highIdx = min(lowIdx + 1, timesteps - 1)
      let w = timestep - Float(lowIdx)
      let logSigma =
        (1 - w) * log(sigmasForTimesteps[lowIdx]) + w * log(sigmasForTimesteps[highIdx])
      sigmas.append(exp(logSigma))
    }
    sigmas.append(0)
    return sigmas
  }

  public static func sigmas(from alphasCumprod: [Float]) -> [Float] {
    return alphasCumprod.map { ((1 - $0) / $0).squareRoot() }
  }

  public static func timestep(from sigma: Float, sigmas: [Float]) -> Float {
    guard sigma > sigmas[0] else {
      return 0
    }
    guard sigma < sigmas[sigmas.count - 1] else {
      return Float(sigmas.count - 1)
    }
    // Find in between which sigma resides.
    var highIdx: Int = sigmas.count - 1
    var lowIdx: Int = 0
    while lowIdx < highIdx - 1 {
      let midIdx = lowIdx + (highIdx - lowIdx) / 2
      if sigma < sigmas[midIdx] {
        highIdx = midIdx
      } else {
        lowIdx = midIdx
      }
    }
    assert(sigma >= sigmas[highIdx - 1] && sigma <= sigmas[highIdx])
    let low = log(sigmas[highIdx - 1])
    let high = log(sigmas[highIdx])
    let logSigma = log(sigma)
    let w = min(max((low - logSigma) / (low - high), 0), 1)
    return (1.0 - w) * Float(highIdx - 1) + w * Float(highIdx)
  }
}

DynamicGraph.setSeed(40)
DynamicGraph.memoryEfficient = true

let unconditionalGuidanceScale: Float = 7.5
let scaleFactor: Float = 0.18215
let strength: Float = 0.75
var startWidth: Int = 64
var startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 30)
let alphasCumprod = model.alphasCumprod
let sigmasForTimesteps = DiffusionModel.sigmas(from: alphasCumprod)
let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let alphas = alphasCumprod.map { $0.squareRoot() }
let sigmas = alphasCumprod.map { (1 - $0).squareRoot() }
let lambdas = zip(alphas, sigmas).map { log($0) - log($1) }

let workDir = CommandLine.arguments[1]
let text =
  CommandLine.arguments.count > 2
  ? CommandLine.arguments.suffix(from: 2).joined(separator: " ") : ""

let tokens = tokenizer.tokenize(text: text, truncation: true, maxLength: 77)

var initImg = Tensor<FloatType>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
if let image = try PNG.Data.Rectangular.decompress(path: workDir + "/init_img.png") {
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  // print(rgba)

  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let pixel = rgba[y * startWidth * 8 + x]
      initImg[0, 0, y, x] = FloatType(Float(pixel.r) / 255 * 2 - 1)
      initImg[0, 1, y, x] = FloatType(Float(pixel.g) / 255 * 2 - 1)
      initImg[0, 2, y, x] = FloatType(Float(pixel.b) / 255 * 2 - 1)
    }
  }
}

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

let textModel = LoRACLIPTextModel(
  FloatType.self,
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
  batchSize: 1, intermediateSize: 3072)

textModel.maxConcurrency = .limit(1)

let tokensTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
for i in 0..<77 {
  tokensTensor[i] = tokens[i]
  positionTensor[i] = Int32(i)
}

let casualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
  }
}

let unet = ModelBuilder {
  let startWidth = $0[0].shape[3]
  let startHeight = $0[0].shape[2]
  return LoRAUNet(batchSize: 1, startWidth: startWidth, startHeight: startHeight)
}

unet.maxConcurrency = .limit(1)

let decoder = ModelBuilder {
  let startWidth = $0[0].shape[3]
  let startHeight = $0[0].shape[2]
  return Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
    startHeight: startHeight)
}

graph.workspaceSize = 1_024 * 1_024 * 1_024

let latents = graph.withNoGrad {
  let encoder = ModelBuilder {
    let startWidth = $0[0].shape[3] / 8
    let startHeight = $0[0].shape[2] / 8
    return Encoder(
      channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
      startHeight: startHeight)
  }
  let initImgGPU = graph.variable(initImg.toGPU(0))
  encoder.compile(inputs: initImgGPU)
  graph.openStore("/fast/Data/SD/swift-diffusion/sd-v1.5.ckpt") {
    $0.read("encoder", model: encoder)
  }
  let encoded = encoder(inputs: initImgGPU)[0].as(of: FloatType.self)
  return scaleFactor * encoded[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
}

let x_T = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
x_T.randn(std: 1, mean: 0)

let tokensTensorGPU = tokensTensor.toGPU(0)
let positionTensorGPU = positionTensor.toGPU(0)
let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
textModel.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
graph.openStore("/fast/Data/SD/swift-diffusion/sd-v1.5.ckpt") { store in
  store.read("text_model", model: textModel) { name, dataType, format, shape in
    if name.contains("lora_up") {
      precondition(dataType == .Float16)
      var tensor = Tensor<FloatType>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<FloatType>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return .final(tensor)
    }
    return .continue(name)
  }
}
let c: DynamicGraph.AnyTensor = textModel(
  inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
    of: FloatType.self
  ).reshaped(.CHW(1, 77, 768))

var x = x_T
var xIn = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
let ts = timeEmbedding(timestep: 0, batchSize: 1, embeddingSize: 320, maxPeriod: 10_000).toGPU(0)
unet.compile(inputs: xIn, graph.variable(Tensor<FloatType>(from: ts)), c)
graph.openStore("/fast/Data/SD/swift-diffusion/sd-v1.5.ckpt") { store in
  store.read("unet", model: unet) { name, dataType, format, shape in
    if name.contains("lora_up") {
      precondition(dataType == .Float16)
      var tensor = Tensor<FloatType>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<FloatType>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return .final(tensor)
    }
    return .continue(name)
  }
}

var adamWOptimizer = AdamWOptimizer(graph, rate: 0.0000001, betas: (0.9, 0.98), decay: 0)
adamWOptimizer.parameters = [unet.parameters]
var timestepList = [Int]()
var outputList = [DynamicGraph.Tensor<FloatType>]()
let startTime = Date()
let tEnc = Int(strength * Float(model.steps))
for epoch in 0..<100 {
  let c = textModel(
    inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
      of: FloatType.self
    ).reshaped(.CHW(1, 77, 768))
  let noise = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
  noise.randn(std: 1, mean: 0)
  let i = Int.random(in: Int(model.steps - tEnc)..<Int(model.steps))
  let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
  let sqrtAlphasCumprod = alphasCumprod[timestep].squareRoot()
  let sqrtOneMinusAlphasCumprod = (1 - alphasCumprod[timestep]).squareRoot()
  let noisyLatents = sqrtAlphasCumprod * latents + sqrtOneMinusAlphasCumprod * noise
  let ts = timeEmbedding(timestep: timestep, batchSize: 1, embeddingSize: 320, maxPeriod: 10_000)
    .toGPU(0)
  let t = graph.variable(Tensor<FloatType>(from: ts))
  let et = unet(inputs: noisyLatents, t, c)[0].as(of: FloatType.self).reshaped(
    .NC(1, 4 * startWidth * startHeight))
  let loss = MSELoss()(et, target: noise.reshaped(.NC(1, 4 * startWidth * startHeight)))[0].as(
    of: FloatType.self)
  loss.backward(to: [latents, tokensTensorGPU])
  let value = loss.toCPU()[0, 0]
  adamWOptimizer.step()
  print("epoch: \(epoch), \(i), loss: \(value)")
}
print("Total time \(Date().timeIntervalSince(startTime))")
