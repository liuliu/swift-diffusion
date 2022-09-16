import C_ccv
import Foundation
import NNC
import NNCPythonConversion

/// Text Model

func CLIPTextEmbedding(batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int)
  -> Model
{
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(Float.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + positionEmbed(positions)
  return Model([tokens, positions], [embedding], name: "embeddings")
}

func CLIPAttention(k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2).reshaped([b * h, t, k])
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .transposed(1, 2).reshaped([b * h, t, k])
  let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2).reshaped([b * h, t, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys) + casualAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b * h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
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
  let fc1 = Dense(count: intermediateSize)
  var out = fc1(x)
  out = QuickGELU()(out)
  let fc2 = Dense(count: hiddenSize)
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

func CLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    batchSize: batchSize,
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
  return Model([tokens, positions, casualAttentionMask], [out])
}

/// UNet

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

func TimeEmbed(modelChannels: Int) -> Model {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return Model([x], [out])
}

func ResBlock(b: Int, outChannels: Int, skipConnection: Bool) -> Model {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = out.swish()
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: outChannels)
  var embOut = emb.swish()
  embOut = embLayer(embOut).reshaped([b, outChannels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = out.swish()
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  if skipConnection {
    let skip = Convolution(
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
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2).reshaped([b * h, hw, k])
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2).reshaped([b * h, hw, k])
  let values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2).reshaped([b * h, hw, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([b * h, hw, hw])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return Model([x], [out])
}

func CrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
  let x = Input()
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2).reshaped([b * h, t, k])
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2).reshaped([b * h, hw, k])
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2).reshaped([b * h, t, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b * h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return Model([x, c], [out])
}

func FeedForward(hiddenSize: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let fc10 = Dense(count: intermediateSize)
  let fc11 = Dense(count: intermediateSize)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = Dense(count: hiddenSize)
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
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).transposed(1, 2)
  let block = BasicTransformerBlock(
    k: k, h: h, b: b, hw: hw, t: t, intermediateSize: intermediateSize)
  out = block(out, c).reshaped([b, hw, k * h]).transposed(1, 2).reshaped([b, k * h, height, width])
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
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
  let c = Input()
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let resBlock = ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  if attentionBlock {
    let transformer = SpatialTransformer(
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer(out, c)
  }
  if attentionBlock {
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
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO
) -> ([Model.IO], Model.IO) {
  let conv2d = Convolution(
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
    for _ in 0..<numRepeat {
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
      passLayers.append(out)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
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
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
      }
      layerStart += 1
    }
  }
  return out
}

func UNet(batchSize: Int, startWidth: Int, startHeight: Int) -> Model {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  let timeEmbed = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  let (inputs, inputBlocks) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes, x: x, emb: emb, c: c)
  var out = inputBlocks
  let middleBlock = MiddleBlock(
    channels: 1280, numHeads: 8, batchSize: batchSize, height: 8, width: 8, embeddingSize: 77,
    x: out,
    emb: emb, c: c)
  out = middleBlock
  let outputBlocks = OutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes, x: out, emb: emb, c: c,
    inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = out.swish()
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  return Model([x, t_emb, c], [out])
}

/// Autoencoder

func ResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> Model {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = norm2(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = nin(x) + out
  } else {
    out = x + out
  }
  return Model([x], [out])
}

func AttnBlock(prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int) -> Model {
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
  return Model([x], [out])
}

func Decoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> Model
{
  let x = Input()
  let postQuantConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = postQuantConv2d(x)
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convIn(out)
  let midBlock1 = ResnetBlock(
    prefix: "mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let midAttn1 = AttnBlock(
    prefix: "mid.attn_1", inChannels: previousChannel, batchSize: batchSize, width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let midBlock2 = ResnetBlock(
    prefix: "mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let block = ResnetBlock(
        prefix: "up.\(i).block.\(j)", outChannels: channel, shortcut: previousChannel != channel)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
    }
  }
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  return Model([x], [out])
}

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
}

public struct CLIPTokenizer {
  public struct Pair: Hashable, Equatable {
    public var first: String
    public var second: String
    public init(first: String, second: String) {
      self.first = first
      self.second = second
    }
  }
  let vocabulary: [String: Int32]
  let bpeRanks: [Pair: Int]
  let unknownToken: Int32
  let startToken: Int32
  let endToken: Int32
  public init(vocabulary: String, merges: String) {
    let vocabJSONData = try! Data(contentsOf: URL(fileURLWithPath: vocabulary))
    let decoder = JSONDecoder()
    self.vocabulary = try! decoder.decode([String: Int32].self, from: vocabJSONData)
    let bpeMerges = (try! String(contentsOf: URL(fileURLWithPath: merges), encoding: .utf8))
      .trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "\n")[
        1..<(49152 - 256 - 2 + 1)]
    var bpeRanks = [Pair: Int]()
    for (i, merge) in bpeMerges.enumerated() {
      let splits = merge.split(separator: " ", maxSplits: 2)
      bpeRanks[Pair(first: String(splits[0]), second: String(splits[1]))] = i
    }
    self.bpeRanks = bpeRanks
    self.unknownToken = self.vocabulary["<|endoftext|>"]!
    self.startToken = self.vocabulary["<|startoftext|>"]!
    self.endToken = self.vocabulary["<|endoftext|>"]!
  }

  public func tokenize(text: String, truncation: Bool, maxLength: Int) -> [Int32] {
    let fixText = text.split(separator: " ").joined(separator: " ").lowercased()
    // Logic for r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
    // Implement this with for loop rather than regex so it is applicable with Swift 5.6.x
    var tokens = [Substring]()
    var lastIndex = fixText.startIndex
    for (i, character) in fixText.enumerated() {
      let index = fixText.index(fixText.startIndex, offsetBy: i)
      if character.isNumber {
        if lastIndex < index {
          tokens.append(fixText[lastIndex..<index])
        }
        lastIndex = fixText.index(index, offsetBy: 1)  // Skip this one.
        tokens.append(fixText[index..<lastIndex])
        continue
      }
      let pat = fixText[lastIndex...index]
      if pat.hasSuffix("'s") || pat.hasSuffix("'t") || pat.hasSuffix("'m") || pat.hasSuffix("'d") {
        let splitIndex = fixText.index(index, offsetBy: -1)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if pat.hasSuffix("'re") || pat.hasSuffix("'ve") || pat.hasSuffix("'ll") {
        let splitIndex = fixText.index(index, offsetBy: -2)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if pat.hasSuffix("<|startoftext|>") {
        let splitIndex = fixText.index(index, offsetBy: -14)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if pat.hasSuffix("<|endoftext|>") {
        let splitIndex = fixText.index(index, offsetBy: -12)
        if lastIndex < splitIndex {
          tokens.append(fixText[lastIndex..<splitIndex])
        }
        lastIndex = fixText.index(index, offsetBy: 1)
        tokens.append(fixText[splitIndex..<lastIndex])
        continue
      }
      if character.isWhitespace {
        if lastIndex < index {
          tokens.append(fixText[lastIndex..<index])
        }
        lastIndex = fixText.index(index, offsetBy: 1)  // Skip this one.
        continue
      }
    }
    if lastIndex < fixText.endIndex {
      tokens.append(fixText[lastIndex...])
    }
    // Now filter token further by split if it not a number nor a letter.
    tokens = tokens.flatMap { token -> [Substring] in
      // Remove special tokens (start and end)
      guard token != "<|startoftext|>" && token != "<|endoftext|>" else {
        return []
      }
      // Skip these tokens
      guard
        token != "'s" && token != "'t" && token != "'m" && token != "'d" && token != "'re"
          && token != "'ve" && token != "'ll"
      else {
        return [token]
      }
      var tokens = [Substring]()
      var lastIndex = token.startIndex
      for (i, character) in token.enumerated() {
        let index = token.index(token.startIndex, offsetBy: i)
        // Split further if it is not a letter nor a number.
        if !character.isLetter && !character.isNumber {
          if lastIndex < index {
            tokens.append(token[lastIndex..<index])
          }
          tokens.append(token[index..<token.index(after: index)])  // Add this character.
          lastIndex = fixText.index(index, offsetBy: 1)
        }
      }
      if lastIndex < token.endIndex {
        tokens.append(token[lastIndex...])
      }
      return tokens
    }
    // token should match the token before sending to bpe mapping. Now do bpe merge.
    let bpeTokens = tokens.flatMap { token -> [String] in
      bpe(token: String(token))
    }
    // With bpeTokens, we can query vocabulary and return index now.
    var ids = [startToken]
    if truncation {
      for bpeToken in bpeTokens.prefix(maxLength - 2) {
        ids.append(vocabulary[bpeToken, default: unknownToken])
      }
    } else {
      for bpeToken in bpeTokens {
        ids.append(vocabulary[bpeToken, default: unknownToken])
      }
    }
    if ids.count < maxLength {
      for _ in ids.count..<maxLength {
        ids.append(endToken)
      }
    } else {
      ids.append(endToken)
    }
    return ids
  }

  func getPairs(word: [String]) -> Set<Pair>? {
    guard word.count > 1 else {
      return nil
    }
    var pairs = Set<Pair>()
    var previousCharacter = word[0]
    for character in word.suffix(from: 1) {
      pairs.insert(Pair(first: previousCharacter, second: character))
      previousCharacter = character
    }
    return pairs
  }

  func bpe(token: String) -> [String] {
    var word = [String]()
    for (i, character) in token.enumerated() {
      guard i < token.count - 1 else {
        word.append(String(character) + "</w>")
        break
      }
      word.append(String(character))
    }
    guard var pairs = getPairs(word: word) else {
      return word
    }
    while true {
      var bigram: Pair? = nil
      var minRank: Int? = nil
      for pair in pairs {
        if let rank = bpeRanks[pair] {
          guard let currentMinRank = minRank else {
            bigram = pair
            minRank = rank
            continue
          }
          if rank < currentMinRank {
            bigram = pair
            minRank = rank
          }
        }
      }
      guard let bigram = bigram else {
        break
      }
      var newWord = [String]()
      var i = 0
      while i < word.count {
        guard let j = word[i...].firstIndex(of: bigram.first) else {
          newWord.append(contentsOf: word[i...])
          break
        }
        if i < j {
          newWord.append(contentsOf: word[i..<j])
        }
        i = j
        if word[i] == bigram.first && i < word.count - 1 && word[i + 1] == bigram.second {
          newWord.append(bigram.first + bigram.second)
          i += 2
        } else {
          newWord.append(word[i])
          i += 1
        }
      }
      word = newWord
      if word.count == 1 {
        break
      }
      pairs = getPairs(word: word)!  // word.count > 1, should be able to get pair.
    }
    return word
  }
}

DynamicGraph.setSeed(40)

let unconditionalGuidanceScale: Float = 7.5
let scaleFactor: Float = 0.18215
let startWidth: Int = 64
let startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 50)
let tokenizer = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let workDir = CommandLine.arguments[1]
let text = CommandLine.arguments.suffix(2).joined(separator: " ")

let unconditionalTokens = tokenizer.tokenize(text: "", truncation: true, maxLength: 77)
let tokens = tokenizer.tokenize(text: text, truncation: true, maxLength: 77)

let graph = DynamicGraph()

let textModel = CLIPTextModel(
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
  batchSize: 2, intermediateSize: 3072)

let tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
for i in 0..<77 {
  tokensTensor[i] = unconditionalTokens[i]
  tokensTensor[i + 77] = tokens[i]
  positionTensor[i] = Int32(i)
  positionTensor[i + 77] = Int32(i)
}

let casualAttentionMask = graph.variable(Tensor<Float>(.CPU, .HWC(1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, i, j] = -Float.greatestFiniteMagnitude
  }
}

var ts = [Tensor<Float>]()
for i in 0..<model.steps {
  let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
  ts.append(
    timeEmbedding(timestep: timestep, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000).toGPU(0))
}
let unet = UNet(batchSize: 2, startWidth: startWidth, startHeight: startHeight)
let decoder = Decoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
  startHeight: startHeight)

func xPrevAndPredX0(
  x: DynamicGraph.Tensor<Float>, et: DynamicGraph.Tensor<Float>, alpha: Float, alphaPrev: Float
) -> (DynamicGraph.Tensor<Float>, DynamicGraph.Tensor<Float>) {
  let predX0 = (1 / alpha.squareRoot()) * (x - (1 - alpha).squareRoot() * et)
  let dirXt = (1 - alphaPrev).squareRoot() * et
  let xPrev = alphaPrev.squareRoot() * predX0 + dirXt
  return (xPrev, predX0)
}

graph.workspaceSize = 1_024 * 1_024 * 1_024

graph.withNoGrad {
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  let _ = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore(workDir + "/sd-v1.4.ckpt") {
    $0.read("text_model", model: textModel)
  }
  let c = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
    of: Float.self
  ).reshaped(.CHW(2, 77, 768))
  let x_T = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
  x_T.randn(std: 1, mean: 0)
  var x = x_T
  var xIn = graph.variable(.GPU(0), .NCHW(2, 4, startHeight, startWidth), of: Float.self)
  let _ = unet(inputs: xIn, graph.variable(ts[0]), c)
  let _ = decoder(inputs: x)
  graph.openStore(workDir + "/sd-v1.4.ckpt") {
    $0.read("unet", model: unet)
    $0.read("decoder", model: decoder)
  }
  let alphasCumprod = model.alphasCumprod
  var oldEps = [DynamicGraph.Tensor<Float>]()
  let startTime = Date()
  // Now do PLMS sampling.
  for i in 0..<model.steps {
    let timestep = model.timesteps - model.timesteps / model.steps * (i + 1) + 1
    let t = graph.variable(ts[i])
    let tNext = ts[min(i + 1, ts.count - 1)]
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = x
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = x
    var et = unet(inputs: xIn, t, c)[0].as(of: Float.self)
    var etUncond = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
    var etCond = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
    etUncond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
    etCond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[1..<2, 0..<4, 0..<startHeight, 0..<startWidth]
    et = etUncond + unconditionalGuidanceScale * (etCond - etUncond)
    let alpha = alphasCumprod[timestep]
    let alphaPrev = alphasCumprod[max(timestep - model.timesteps / model.steps, 0)]
    let etPrime: DynamicGraph.Tensor<Float>
    switch oldEps.count {
    case 0:
      let (xPrev, _) = xPrevAndPredX0(x: x, et: et, alpha: alpha, alphaPrev: alphaPrev)
      // Compute etNext.
      xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = xPrev
      xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = xPrev
      var etNext = unet(inputs: xIn, graph.variable(tNext), c)[0].as(of: Float.self)
      var etNextUncond = graph.variable(
        .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
      var etNextCond = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: Float.self)
      etNextUncond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
        etNext[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
      etNextCond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
        etNext[1..<2, 0..<4, 0..<startHeight, 0..<startWidth]
      etNext = etNextUncond + unconditionalGuidanceScale * (etNextCond - etNextUncond)
      etPrime = 0.5 * (et + etNext)
    case 1:
      etPrime = 0.5 * (3 * et - oldEps[0])
    case 2:
      etPrime =
        Float(1) / Float(12) * (Float(23) * et - Float(16) * oldEps[1] + Float(5) * oldEps[0])
    case 3:
      etPrime =
        Float(1) / Float(24)
        * (Float(55) * et - Float(59) * oldEps[2] + Float(37) * oldEps[1] - Float(9) * oldEps[0])
    default:
      fatalError()
    }
    let (xPrev, _) = xPrevAndPredX0(x: x, et: etPrime, alpha: alpha, alphaPrev: alphaPrev)
    x = xPrev
    oldEps.append(et)
    if oldEps.count > 3 {
      oldEps.removeFirst()
    }
  }
  let z = 1.0 / scaleFactor * x
  let img = decoder(inputs: z)[0].as(of: Float.self).toCPU()
  print("Total time \(Date().timeIntervalSince(startTime))")
  let image = ccv_dense_matrix_new(
    Int32(startHeight * 8), Int32(startWidth * 8), Int32(CCV_8U | CCV_C3), nil, 0)
  // I have better way to copy this out (basically, transpose and then ccv_shift). Doing this just for fun.
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let (r, g, b) = (img[0, 0, y, x], img[0, 1, y, x], img[0, 2, y, x])
      image!.pointee.data.u8[y * startWidth * 8 * 3 + x * 3] = UInt8(
        min(max(Int(Float((r + 1) / 2) * 255), 0), 255))
      image!.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 1] = UInt8(
        min(max(Int(Float((g + 1) / 2) * 255), 0), 255))
      image!.pointee.data.u8[y * startWidth * 8 * 3 + x * 3 + 2] = UInt8(
        min(max(Int(Float((b + 1) / 2) * 255), 0), 255))
    }
  }
  let _ = (workDir + "/txt2img.png").withCString {
    ccv_write(image, UnsafeMutablePointer(mutating: $0), nil, Int32(CCV_IO_PNG_FILE), nil)
  }
}
