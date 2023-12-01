import C_ccv
import Diffusion
import Foundation
import NNC
import PNG

public typealias FloatType = Float

struct PythonObject {}

func CLIPSelfAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, tokeys, tovalues, unifyheads, Model([x], [out]))
}

func CLIPResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let (toqueries, tokeys, tovalues, unifyheads, attention) = CLIPSelfAttention(
    k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x], [out]))
}

func VisionTransformer(
  grid: Int, width: Int, outputDim: Int, layers: Int, heads: Int, batchSize: Int,
  noFinalLayerNorm: Bool = false
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let classEmbedding = Parameter<FloatType>(.GPU(0), .CHW(1, 1, width))
  let positionalEmbedding = Parameter<FloatType>(.GPU(0), .CHW(1, grid * grid + 1, width))
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], noBias: true,
    hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPre(out)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (reader, block) = CLIPResidualAttentionBlock(
      prefix: "open_clip.model.visual.transformer.resblocks.\(i)", k: width / heads, h: heads,
      b: batchSize,
      t: grid * grid + 1)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
    readers.append(reader)
  }
  let finalLayerNorm: Model?
  if !noFinalLayerNorm {
    let lnPost = LayerNorm(epsilon: 1e-5, axis: [1], name: "post_layernorm")
    out = lnPost(out.reshaped([batchSize, width], strides: [width, 1]))
    finalLayerNorm = lnPost
  } else {
    finalLayerNorm = nil
  }
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x], [out]))
}

func ResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  out = Swish()(out)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = norm2(out)
  out = Swish()(out)
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = nin(x) + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x], [out]))
}

func AttnBlock(prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int) -> (
  (PythonObject) -> Void, Model
) {
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
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x], [out]))
}

func Encoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var readers = [(PythonObject) -> Void]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (reader, block) = ResnetBlock(
        prefix: "down.\(i).block.\(j)", outChannels: channel, shortcut: previousChannel != channel)
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
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])))
      out = conv2d(out).reshaped(
        [batchSize, channel, height, width], offset: [0, 0, 1, 1],
        strides: [channel * (height + 1) * (width + 1), (height + 1) * (width + 1), width + 1, 1])
      let downLayer = i
      let reader: (PythonObject) -> Void = { _ in
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "mid.attn_1", inChannels: previousChannel, batchSize: batchSize, width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 8, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let quantConv2d = Convolution(
    groups: 1, filters: 8, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = quantConv2d(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x], [out]))
}

public func timeEmbedding(timestep: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int)
  -> Tensor<
    Float
  >
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timestep
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

func LabelEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func TimePosEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out], name: "time_pos_embed"))
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
  embOut = embLayer(embOut).reshaped([1, outChannels, 1, 1])
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

func TimeResBlock(b: Int, h: Int, w: Int, channels: Int) -> (
  Model, Model, Model, Model, Model, Model
) {
  let x = Input()
  let emb = Input()
  let y = x.transposed(0, 1).reshaped([1, channels, b, h * w])  // [b, c, h, w] -> [c, b, h, w] -> [1, c, b, h * w]
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(y)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: channels, filterSize: [3, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 0], end: [1, 0])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: channels)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([1, channels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: channels, filterSize: [3, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 0], end: [1, 0])))
  out = y + outLayerConv2d(out)  // This layer should be zero init if training.
  out = out.reshaped([channels, b, h, w]).transposed(0, 1)
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d,
    Model([x, emb], [out], name: "time_stack")
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

func CrossAttentionKeysAndValues(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = Dense(count: k * h, noBias: true)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, unifyheads, Model([x, keys, values], [out]))
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

func BasicTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (toqueries2, unifyheads2, attn2) = CrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, keys, values) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x, keys, values], [out]))
}

func BasicTimeTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let timeEmb = Input()
  let keys = Input()
  let values = Input()
  var out = x.transposed(0, 1) + timeEmb.reshaped([1, b, k * h])
  let normIn = LayerNorm(epsilon: 1e-5, axis: [2])
  let (ffIn10, ffIn11, ffIn2, ffIn) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ffIn(normIn(out)) + out
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: hw, hw: b)
  out = attn1(layerNorm1(out)) + out
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (toqueries2, unifyheads2, attn2) = CrossAttentionKeysAndValues(
    k: k, h: h, b: hw, hw: b, t: t)
  out = attn2(out, keys, values) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  out = out.transposed(0, 1)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x, timeEmb, keys, values], [out], name: "time_stack"))
}

func SpatialTransformer(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let hw = height * width
  let projIn = Dense(count: k * h)
  out = projIn(out.reshaped([b, k * h, hw]).transposed(1, 2))
  var readers = [(PythonObject) -> Void]()
  let timeEmb: Input?
  let mixFactor: Parameter<FloatType>?
  if depth > 0 {
    let emb = Input()
    kvs.append(emb)
    timeEmb = emb
    mixFactor = Parameter<FloatType>(.GPU(0), .C(1), name: "time_mixer")
  } else {
    timeEmb = nil
    mixFactor = nil
  }
  for i in 0..<depth {
    let keys = Input()
    kvs.append(keys)
    let values = Input()
    kvs.append(values)
    let (reader, block) = BasicTransformerBlock(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    out = block(out, keys, values)
    readers.append(reader)
    if let timeEmb = timeEmb, let mixFactor = mixFactor {
      let keys = Input()
      kvs.append(keys)
      let values = Input()
      kvs.append(values)
      let (reader, block) = BasicTimeTransformerBlock(
        prefix: "\(prefix).time_stack.\(i)", k: k, h: h, b: b, hw: hw, t: t,
        intermediateSize: intermediateSize)
      out = mixFactor .* out + (1 - mixFactor) .* block(out, timeEmb, keys, values)
      readers.append(reader)
    }
  }
  out = out.reshaped([b, height, width, k * h]).permuted(0, 3, 1, 2)
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x] + kvs, [out]))
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let emb = Input()
  var kvs = [Model.IO]()
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  let (
    timeInLayerNorm, timeInLayerConv2d, timeEmbLayer, timeOutLayerNorm, timeOutLayerConv2d,
    timeResBlock
  ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels)
  let mixFactor = Parameter<FloatType>(.GPU(0), .C(1), name: "time_mixer")
  out = mixFactor .* out + (1 - mixFactor) .* timeResBlock(out, emb)
  var transformerReader: ((PythonObject) -> Void)? = nil
  if attentionBlock > 0 {
    let c = (0..<(attentionBlock * 4 + 1)).map { _ in Input() }
    let transformer: Model
    (
      transformerReader, transformer
    ) = SpatialTransformer(
      prefix: "\(prefix).\(layerStart).1",
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x, emb] + kvs, [out]))
}

func MiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, x: Model.IO, emb: Model.IO
) -> ((PythonObject) -> Void, Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let (
    timeInLayerNorm1, timeInLayerConv2d1, timeEmbLayer1, timeOutLayerNorm1, timeOutLayerConv2d1,
    timeResBlock1
  ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels)
  let mixFactor1 = Parameter<FloatType>(.GPU(0), .C(1), name: "time_mixer")
  out = mixFactor1 .* out + (1 - mixFactor1) .* timeResBlock1(out, emb)
  let kvs = (0..<(attentionBlock * 4 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
  let (
    transformerReader, transformer
  ) = SpatialTransformer(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  out = transformer([out] + kvs)
  let (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  let (
    timeInLayerNorm2, timeInLayerConv2d2, timeEmbLayer2, timeOutLayerNorm2, timeOutLayerConv2d2,
    timeResBlock2
  ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels)
  let mixFactor2 = Parameter<FloatType>(.GPU(0), .C(1), name: "time_mixer")
  out = mixFactor2 .* out + (1 - mixFactor2) .* timeResBlock2(out, emb)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, out, kvs)
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO
) -> ((PythonObject) -> Void, [Model.IO], Model.IO, [Input]) {
  let conv2d = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      let (reader, inputLayer) = BlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      let c = (0..<(attentionBlock * 4 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
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
      let reader: (PythonObject) -> Void = { _ in
      }
      readers.append(reader)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, passLayers, out, kvs)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO,
  inputs: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO, [Input]) {
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
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let (reader, outputLayer) = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      let c = (0..<(attentionBlock * 4 + (attentionBlock > 0 ? 1 : 0))).map { _ in Input() }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      readers.append(reader)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock > 0 ? 2 : 1
        let reader: (PythonObject) -> Void = { _ in
        }
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, out, kvs)
}

func UNetXL(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  attentionRes: KeyValuePairs<Int, Int>
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let (timeFc0, timeFc2, timeEmbed) = TimeEmbed(modelChannels: channels[0])
  let (labelFc0, labelFc2, labelEmbed) = LabelEmbed(modelChannels: channels[0])
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (inputReader, inputs, inputBlocks, inputKVs) = InputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 1, attentionRes: attentionRes,
    x: x, emb: emb)
  var out = inputBlocks
  let (middleReader, middleBlock, middleKVs) = MiddleBlock(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 1, attentionBlock: middleBlockAttentionBlock, x: out, emb: emb)
  out = middleBlock
  let (outputReader, outputBlocks, outputKVs) = OutputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 1, attentionRes: attentionRes,
    x: out, emb: emb, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x, t_emb, y] + inputKVs + middleKVs + outputKVs, [out]))
}

func CrossAttentionFixed(k: Int, h: Int, b: Int, t: Int, name: String = "") -> (Model, Model, Model)
{
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
  return (tokeys, tovalues, Model([c], [keys, values], name: name))
}

func BasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(k: k, h: h, b: b, t: t)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, attn2)
}

func TimePosEmbedTransformerBlockFixed(
  prefix: String, k: Int, h: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let (timePosFc0, timePosFc2, timePosEmbed) = TimePosEmbed(modelChannels: k * h)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, timePosEmbed)
}

func BasicTimeTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(k: k, h: h, b: b, t: t, name: "time_stack")
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, attn2)
}

func SpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  let numFrames = Input()
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  let hw = height * width
  let (timePosEmbedReader, timePosEmbed) = TimePosEmbedTransformerBlockFixed(
    prefix: prefix, k: k, h: h)
  outs.append(timePosEmbed(numFrames))
  readers.append(timePosEmbedReader)
  for i in 0..<depth {
    let (reader, block) = BasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(block(c))
    readers.append(reader)
    let (timeReader, timeBlock) = BasicTimeTransformerBlockFixed(
      prefix: "\(prefix).time_stack.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(timeBlock(c))
    readers.append(timeReader)
  }
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([c, numFrames], outs))
}

func BlockLayerFixed(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (transformerReader, transformer) = SpatialTransformerFixed(
    prefix: "\(prefix).\(layerStart).1",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingSize,
    intermediateSize: channels * 4)
  return (transformerReader, transformer)
}

func MiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, c: Model.IO, numFrames: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (
    transformerReader, transformer
  ) = SpatialTransformerFixed(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  let out = transformer(c, numFrames[numFrames.count - 1])
  return (transformerReader, out)
}

func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO, numFrames: [Model.IO]
) -> ((PythonObject) -> Void, [Model.IO]) {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      if attentionBlock > 0 {
        let (reader, inputLayer) = BlockLayerFixed(
          prefix: "input_blocks",
          layerStart: layerStart, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        previousChannel = channel
        outs.append(inputLayer(c, numFrames[i]))
        readers.append(reader)
      }
      layerStart += 1
    }
    if i != channels.count - 1 {
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, outs)
}

func OutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO, numFrames: [Model.IO]
) -> ((PythonObject) -> Void, [Model.IO]) {
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
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<(numRepeat + 1) {
      if attentionBlock > 0 {
        let (reader, outputLayer) = BlockLayerFixed(
          prefix: "output_blocks",
          layerStart: layerStart, skipConnection: true,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        outs.append(outputLayer(c, numFrames[i]))
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, outs)
}

func UNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  attentionRes: KeyValuePairs<Int, Int>
) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let numFrames = (0..<channels.count).map { _ in Input() }
  let (inputReader, inputBlocks) = InputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 1, attentionRes: attentionRes,
    c: c, numFrames: numFrames)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (middleReader, middleBlock) = MiddleBlockFixed(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 1, attentionBlock: middleBlockAttentionBlock, c: c, numFrames: numFrames)
  out.append(middleBlock)
  let (outputReader, outputBlocks) = OutputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 1, attentionRes: attentionRes,
    c: c, numFrames: numFrames)
  out.append(contentsOf: outputBlocks)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([c] + numFrames, out))
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

DynamicGraph.setSeed(42)

let unconditionalGuidanceScale: Float = 5
let scaleFactor: Float = 0.18215
let strength: Float = 0.75
let startWidth: Int = 64
let startHeight: Int = 64
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 50)

var initImg = Tensor<FloatType>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
let u8Img = ccv_dense_matrix_new(512, 512, Int32(CCV_8U | CCV_C3), nil, 0)!
if let image = try PNG.Data.Rectangular.decompress(
  path: "/home/liu/workspace/swift-diffusion/kandinsky-512.png")
{
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let pixel = rgba[y * startWidth * 8 + x]
      initImg[0, 0, y, x] = FloatType(Float(pixel.r) / 255 * 2 - 1)
      initImg[0, 1, y, x] = FloatType(Float(pixel.g) / 255 * 2 - 1)
      initImg[0, 2, y, x] = FloatType(Float(pixel.b) / 255 * 2 - 1)
      u8Img.pointee.data.u8[y * 512 * 3 + x * 3] = pixel.r
      u8Img.pointee.data.u8[y * 512 * 3 + x * 3 + 1] = pixel.g
      u8Img.pointee.data.u8[y * 512 * 3 + x * 3 + 2] = pixel.b
    }
  }
}

var clipImg = Tensor<FloatType>(.CPU, .NCHW(1, 3, 224, 224))
var smallerImg: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
ccv_resample(u8Img, &smallerImg, 0, 224 / 512, 224 / 512, Int32(CCV_INTER_AREA))
ccv_matrix_free(u8Img)
for y in 0..<224 {
  for x in 0..<224 {
    let (r, g, b) = (
      smallerImg!.pointee.data.u8[y * 224 * 3 + x * 3],
      smallerImg!.pointee.data.u8[y * 224 * 3 + x * 3 + 1],
      smallerImg!.pointee.data.u8[y * 224 * 3 + x * 3 + 2]
    )
    clipImg[0, 0, y, x] = FloatType((Float(r) / 255 - 0.48145466) / 0.26862954)
    clipImg[0, 1, y, x] = FloatType((Float(g) / 255 - 0.4578275) / 0.26130258)
    clipImg[0, 2, y, x] = FloatType((Float(b) / 255 - 0.40821073) / 0.27577711)
  }
}
ccv_matrix_free(smallerImg)

let graph = DynamicGraph()

graph.withNoGrad {
  let (_, vit) = VisionTransformer(
    grid: 16, width: 1280, outputDim: 1024, layers: 32, heads: 16, batchSize: 1)
  let clipImageTensor = graph.variable(clipImg).toGPU(0)
  vit.compile(inputs: clipImageTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_vit_h14_vision_model_f32.ckpt") {
    $0.read("vision_model", model: vit)
  }
  let imageEmbeds = vit(inputs: clipImageTensor)[0].as(of: FloatType.self).reshaped(
    .CHW(1, 1, 1280))
  let visualProj = Dense(count: 1024, noBias: true)
  visualProj.compile(inputs: imageEmbeds)
  graph.openStore("/home/liu/workspace/swift-diffusion/svd_i2v_1.0_f32.ckpt") {
    $0.read("visual_proj", model: visualProj)
  }
  let imageProj = visualProj(inputs: imageEmbeds)[0].as(of: FloatType.self)
  debugPrint(imageProj)
  let (_, encoder) = Encoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64, startHeight: 64)
  let imageTensor = graph.variable(initImg).toGPU(0)
  encoder.compile(inputs: imageTensor)
  graph.openStore("/home/liu/workspace/swift-diffusion/vae_ft_mse_840000_f16.ckpt") {
    $0.read("encoder", model: encoder)
  }
  let parameters = encoder(inputs: imageTensor)[0].as(of: FloatType.self)
  let mean = parameters[0..<1, 0..<4, 0..<64, 0..<64].copied()
  debugPrint(mean)
  let fpsId = timeEmbedding(timestep: 5, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let motionBucketId = timeEmbedding(
    timestep: 127, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let condAug = timeEmbedding(timestep: 0.02, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let vector = DynamicGraph.Tensor<FloatType>(
    from: Concat(axis: 1)(
      inputs: graph.variable(fpsId), graph.variable(motionBucketId), graph.variable(condAug))[0].as(
        of: Float.self
      )
  ).toGPU(0)
  var xIn = graph.variable(.GPU(0), .NCHW(14, 8, 64, 64), of: FloatType.self)
  xIn.full(0)
  let t_emb = DynamicGraph.Tensor<FloatType>(
    from: graph.variable(
      timeEmbedding(timestep: 0.9771, batchSize: 1, embeddingSize: 320, maxPeriod: 10_000)
    )
  ).toGPU(0)
  let numFramesEmb = [320, 640, 1280, 1280].map { embeddingSize in
    let tensors = (0..<14).map {
      graph.variable(
        timeEmbedding(
          timestep: Float($0), batchSize: 1, embeddingSize: embeddingSize, maxPeriod: 10_000)
      ).toGPU(0)
    }
    return DynamicGraph.Tensor<FloatType>(
      from: Concat(axis: 0)(inputs: tensors[0], Array(tensors[1...]))[0].as(of: Float.self))
  }
  let (_, unetFixed) = UNetXLFixed(
    batchSize: 1, startHeight: 64, startWidth: 64, channels: [320, 640, 1280, 1280],
    attentionRes: [1: 1, 2: 1, 4: 1])
  unetFixed.compile(inputs: [imageProj] + numFramesEmb)
  graph.openStore("/home/liu/workspace/swift-diffusion/svd_i2v_1.0_f32.ckpt") {
    $0.read("unet_fixed", model: unetFixed)
  }
  let kvs = unetFixed(inputs: imageProj, numFramesEmb).map { $0.as(of: FloatType.self) }
  let zeroProj = graph.variable(like: imageProj)
  zeroProj.full(0)
  let kvs0 = unetFixed(inputs: zeroProj, numFramesEmb).map { $0.as(of: FloatType.self) }
  let (_, unet) = UNetXL(
    batchSize: 14, startHeight: 64, startWidth: 64, channels: [320, 640, 1280, 1280],
    attentionRes: [1: 1, 2: 1, 4: 1])
  unet.compile(inputs: [xIn, t_emb, vector] + kvs)
  graph.openStore("/home/liu/workspace/swift-diffusion/svd_i2v_1.0_f32.ckpt") {
    $0.read("unet", model: unet)
  }
  // 0.002 - 700
  let minInvRho: Double = pow(0.002, 1 / 7)
  let maxInvRho: Double = pow(700, 1 / 7)
  let sigmas: [Double] = (0..<25).map {
    pow(maxInvRho + Double($0) / 24 * (minInvRho - maxInvRho), 7)
  }
  var x = graph.variable(.GPU(0), .NCHW(14, 4, 64, 64), of: FloatType.self)
  x.randn(std: Float((sigmas[0] * sigmas[0] + 1).squareRoot()))
  let scaleCPU = graph.variable(.CPU, .NCHW(14, 1, 1, 1), of: FloatType.self)
  for i in 0..<14 {
    scaleCPU[i, 0, 0, 0] = FloatType(Float(i) * 1.5 / 13 + 1)
  }
  let scale = scaleCPU.toGPU(0)
  for i in 0..<25 {
    print("\(i)")
    let sigma: Double = sigmas[i]
    let cSkip: Double = 1.0 / (sigma * sigma + 1.0)
    let cOut: Double = -sigma / (sigma * sigma + 1.0).squareRoot()
    let cIn: Double = 1 / (sigma * sigma + 1.0).squareRoot()
    let cNoise: Double = 0.25 * log(sigma)
    let t_emb = DynamicGraph.Tensor<FloatType>(
      from: graph.variable(
        timeEmbedding(timestep: Float(cNoise), batchSize: 1, embeddingSize: 320, maxPeriod: 10_000)
      )
    ).toGPU(0)
    xIn[0..<14, 0..<4, 0..<64, 0..<64] = Float(cIn) * x
    for j in 0..<14 {
      xIn[j..<(j + 1), 4..<8, 0..<64, 0..<64] = mean
    }
    var etCond = unet(inputs: xIn, [t_emb, vector] + kvs)[0].as(of: FloatType.self)
    xIn.full(0)
    xIn[0..<14, 0..<4, 0..<64, 0..<64] = Float(cIn) * x
    var etUncond = unet(inputs: xIn, [t_emb, vector] + kvs0)[0].as(of: FloatType.self)
    print("cSkip \(cSkip) cOut \(cOut) cIn \(cIn) cNoise \(cNoise) sigma \(sigma)")
    var et = etUncond + scale .* (etCond - etUncond)
    et = Float(cOut) * et + Float(cSkip) * x
    let d = Float(1.0 / sigma) * (x - et)
    if i < 25 - 1 {
      x = x + Float(sigmas[i + 1] - sigma) * d
    } else {
      x = x - Float(sigma) * d
    }
    debugPrint(x)
  }
  let z = 1.0 / scaleFactor * x
  let decoder = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64,
    startHeight: 64)
  decoder.compile(inputs: z[0..<1, 0..<4, 0..<64, 0..<64])
  graph.openStore("/home/liu/workspace/swift-diffusion/vae_ft_mse_840000_f16.ckpt") {
    $0.read("decoder", model: decoder)
  }
  let imgs = (0..<14).map {
    DynamicGraph.Tensor<Float>(
      from: decoder(inputs: z[$0..<($0 + 1), 0..<4, 0..<64, 0..<64].copied())[0].as(
        of: FloatType.self)
    )
    .toCPU()
  }
  for (i, img) in imgs.enumerated() {
    var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: startWidth * 8 * startHeight * 8)
    for y in 0..<startHeight * 8 {
      for x in 0..<startWidth * 8 {
        let (r, g, b) = (img[0, 0, y, x], img[0, 1, y, x], img[0, 2, y, x])
        rgba[y * startWidth * 8 + x].r = UInt8(
          min(max(Int(Float((r + 1) / 2) * 255), 0), 255))
        rgba[y * startWidth * 8 + x].g = UInt8(
          min(max(Int(Float((g + 1) / 2) * 255), 0), 255))
        rgba[y * startWidth * 8 + x].b = UInt8(
          min(max(Int(Float((b + 1) / 2) * 255), 0), 255))
      }
    }
    let image = PNG.Data.Rectangular(
      packing: rgba, size: (startWidth * 8, startHeight * 8),
      layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
    try! image.compress(path: "/home/liu/workspace/swift-diffusion/outputs/\(i).png", level: 4)
  }
}
