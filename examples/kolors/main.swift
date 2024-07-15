import Diffusion
import Foundation
import NNC
import PNG
import SentencePiece

typealias FloatType = Float16
struct PythonObject {}

func LabelEmbed(modelChannels: Int) -> Model {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return Model([x], [out])
}

func CrossAttentionKeysAndValues(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
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
  return Model([x, keys, values], [out])
}

private func BasicTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let keys = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let attn1 = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let attn2 = CrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, keys, values) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let ff = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  return Model([x, keys, values], [out])
}

private func SpatialTransformer(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> Model {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).permuted(0, 2, 1)
  for i in 0..<depth {
    let keys = Input()
    kvs.append(keys)
    let values = Input()
    kvs.append(values)
    let block = BasicTransformerBlock(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    out = block(out, keys, values)
  }
  out = out.reshaped([b, height, width, k * h]).permuted(0, 3, 1, 2)
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  return Model([x] + kvs, [out])
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let emb = Input()
  var kvs = [Model.IO]()
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let resBlock = ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  if attentionBlock > 0 {
    let c = (0..<(attentionBlock * 2)).map { _ in Input() }
    let transformer = SpatialTransformer(
      prefix: "\(prefix).\(layerStart).1",
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  return Model([x, emb] + kvs, [out])
}

func MiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, x: Model.IO, emb: Model.IO
) -> (Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let resBlock1 =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let kvs = (0..<(attentionBlock * 2)).map { _ in Input() }
  if attentionBlock > 0 {
    let transformer = SpatialTransformer(
      prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
      width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
    out = transformer([out] + kvs)
    let resBlock2 =
      ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
    out = resBlock2(out, emb)
  }
  return (out, kvs)
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingSize: Int, attentionRes: [Int: [Int]], x: Model.IO, emb: Model.IO
) -> ([Model.IO], Model.IO, [Input]) {
  let conv2d = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    for j in 0..<numRepeat {
      let inputLayer = BlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      let c = (0..<(attentionBlock[j] * 2)).map { _ in Input() }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
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
  return (passLayers, out, kvs)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: [Int]], x: Model.IO, emb: Model.IO,
  inputs: [Model.IO]
) -> (Model.IO, [Input]) {
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
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat + 1)]
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let outputLayer = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      let c = (0..<(attentionBlock[j] * 2)).map { _ in Input() }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
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
  return (out, kvs)
}

func UNetXL(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlock: Int,
  outputAttentionRes: KeyValuePairs<Int, [Int]>
) -> Model {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let timeEmbed = TimeEmbed(modelChannels: channels[0])
  let labelEmbed = LabelEmbed(modelChannels: channels[0])
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (inputs, inputBlocks, inputKVs) = InputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 256,
    attentionRes: inputAttentionRes,
    x: x, emb: emb)
  var out = inputBlocks
  let (middleBlock, middleKVs) = MiddleBlock(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 256, attentionBlock: middleAttentionBlock, x: out, emb: emb)
  out = middleBlock
  let (outputBlocks, outputKVs) = OutputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 256,
    attentionRes: outputAttentionRes,
    x: out, emb: emb, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  return Model([x, t_emb, y] + inputKVs + middleKVs + outputKVs, [out])
}

func CrossAttentionFixed(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
  return Model([c], [keys, values])
}

func BasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> Model {
  let attn2 = CrossAttentionFixed(k: k, h: h, b: b, hw: hw, t: t)
  return attn2
}

func SpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> Model {
  let c = Input()
  var outs = [Model.IO]()
  let hw = height * width
  for i in 0..<depth {
    let block = BasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(block(c))
  }
  return Model([c], outs)
}

func BlockLayerFixed(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> Model {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let transformer = SpatialTransformerFixed(
    prefix: "\(prefix).\(layerStart).1",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingSize,
    intermediateSize: channels * 4)
  return transformer
}

func MiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, c: Model.IO
) -> Model.IO {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let transformer = SpatialTransformerFixed(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  let out = transformer(c)
  return out
}

func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: [Int]], c: Model.IO
) -> [Model.IO] {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    for j in 0..<numRepeat {
      if attentionBlock[j] > 0 {
        let inputLayer = BlockLayerFixed(
          prefix: "input_blocks",
          layerStart: layerStart, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        previousChannel = channel
        outs.append(inputLayer(c))
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
  return outs
}

func OutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: [Int]], c: Model.IO
) -> [Model.IO] {
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
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat + 1)]
    for j in 0..<(numRepeat + 1) {
      if attentionBlock[j] > 0 {
        let outputLayer = BlockLayerFixed(
          prefix: "output_blocks",
          layerStart: layerStart, skipConnection: true,
          attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        outs.append(outputLayer(c))
      }
      layerStart += 1
    }
  }
  return outs
}

func UNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlock: Int,
  outputAttentionRes: KeyValuePairs<Int, [Int]>
) -> Model {
  let c = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let inputBlocks = InputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 256,
    attentionRes: inputAttentionRes,
    c: c)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  if middleAttentionBlock > 0 {
    let middleBlock = MiddleBlockFixed(
      channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
      height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
      embeddingSize: 256, attentionBlock: middleAttentionBlock, c: c)
    out.append(middleBlock)
  }
  let outputBlocks = OutputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 256,
    attentionRes: outputAttentionRes,
    c: c)
  out.append(contentsOf: outputBlocks)
  return Model([c], out)
}

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/Kolors/weights/Kolors/tokenizer/tokenizer.model")
let prompt = "一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“可图”"
var tokens = sentencePiece.encode(prompt).map { return $0.id }
tokens.insert(64790, at: 0)
tokens.insert(64792, at: 1)

let graph = DynamicGraph()

var rotTensor = graph.variable(.CPU, .NCHW(1, tokens.count, 1, 128), of: FloatType.self)
for i in 0..<tokens.count {
  for k in 0..<32 {
    let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 64)
    let sintheta = sin(theta)
    let costheta = cos(theta)
    rotTensor[0, i, 0, k * 2] = FloatType(costheta)
    rotTensor[0, i, 0, k * 2 + 1] = FloatType(sintheta)
  }
  for k in 32..<64 {
    rotTensor[0, i, 0, k * 2] = 1
    rotTensor[0, i, 0, k * 2 + 1] = 0
  }
}

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int)) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t.1, hk, k])
  var queries = toqueries(x).reshaped([b, t.1, h, k])
  var values = tovalues(x).reshaped([b, t.1, hk, k])
  if h > hk {
    keys = Concat(axis: 3)(Array(repeating: keys, count: h / hk))
    values = Concat(axis: 3)(Array(repeating: values, count: h / hk))
  }
  keys = keys.reshaped([b, t.1, h, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  keys = keys.transposed(1, 2)
  queries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
  values = values.reshaped([b, t.1, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
  dot = dot.reshaped([b * h * t.1, t.1])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t.1, t.1])
  var out = dot * values
  out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "o")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (Model([x, rot, causalAttentionMask], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true)
  let w3 = Dense(count: intermediateSize, noBias: true)
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true)
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

func GLMTransformerBlock(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int), MLP: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let (attention, attentionReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot, causalAttentionMask) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "ffn")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (Model([x, rot, causalAttentionMask], [out]), reader)
}

func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "word_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (Model([tokens], [embedding]), reader)
}

func GLMTransformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: Int, cachedTokenLength: Int,
  layers: Int, MLP: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let (embedding, embeddingReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  var penultimate: Model.IO? = nil
  for i in 0..<layers {
    if i == layers - 1 {
      penultimate = out
    }
    let (layer, reader) = GLMTransformerBlock(
      prefix: "encoder.layers.\(i)", k: width / heads, h: heads, hk: heads / 16, b: batchSize,
      t: (cachedTokenLength + tokenLength, tokenLength),
      MLP: MLP)
    out = layer(out, rot, causalAttentionMask)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    embeddingReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
  }
  if let penultimate = penultimate {
    return (Model([tokens, rot, causalAttentionMask], [penultimate, out]), reader)
  } else {
    return (Model([tokens, rot, causalAttentionMask], [out]), reader)
  }
}

let (crossattn, pooled) = graph.withNoGrad {
  let (transformer, reader) = GLMTransformer(
    FloatType.self, vocabularySize: 65_024, width: 4_096, tokenLength: 256, cachedTokenLength: 0,
    layers: 28, MLP: 13696, heads: 32, batchSize: 2)
  let tokensTensor = graph.variable(.CPU, format: .NCHW, shape: [256 * 2], of: Int32.self)
  for i in 0..<254 {
    tokensTensor[i] = 0
  }
  tokensTensor[254] = 64790
  tokensTensor[255] = 64792
  for i in 0..<(256 - tokens.count) {
    tokensTensor[256 + i] = 0
  }
  for i in (256 - tokens.count)..<256 {
    tokensTensor[256 + i] = tokens[i - (256 - tokens.count)]
  }
  var paddedRotTensor = graph.variable(.CPU, .NCHW(2, 256, 1, 128), of: FloatType.self)
  for i in 0..<(256 - 2) {
    paddedRotTensor[0..<1, i..<(i + 1), 0..<1, 0..<128] = rotTensor[0..<1, 0..<1, 0..<1, 0..<128]
  }
  paddedRotTensor[0..<1, (256 - 2)..<256, 0..<1, 0..<128] =
    rotTensor[0..<1, 0..<2, 0..<1, 0..<128]
  for i in 0..<(256 - tokens.count) {
    paddedRotTensor[1..<2, i..<(i + 1), 0..<1, 0..<128] = rotTensor[0..<1, 0..<1, 0..<1, 0..<128]
  }
  paddedRotTensor[1..<2, (256 - tokens.count)..<256, 0..<1, 0..<128] =
    rotTensor[0..<1, 0..<tokens.count, 0..<1, 0..<128]
  let causalAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NCHW(2, 1, 256, 256)))
  causalAttentionMask.full(0)
  for i in (256 - 2)..<255 {
    for j in (i + 1)..<256 {
      causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  for i in (256 - 2)..<256 {
    for j in 0..<(256 - 2) {
      causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  for i in (256 - tokens.count)..<255 {
    for j in (i + 1)..<256 {
      causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  for i in (256 - tokens.count)..<256 {
    for j in 0..<(256 - tokens.count) {
      causalAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let rotTensorGPU = paddedRotTensor.toGPU(0)
  let causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
  transformer.compile(inputs: tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/chatglm3_6b_q6p_q8p.ckpt") {
    $0.read("text_model", model: transformer, codec: [.q6p, .q8p, .ezm7])
  }
  let out = transformer(inputs: tokensTensorGPU, rotTensorGPU, causalAttentionMaskGPU).map {
    $0.as(of: FloatType.self)
  }
  /*
  let truth = graph.variable(like: out[0])
  graph.openStore("/home/liu/workspace/swift-diffusion/chatglm3_6b_f16_truth.ckpt") {
    $0.read("fp16", variable: truth)
  }
  let error = DynamicGraph.Tensor<Float>(from: out[0]) - DynamicGraph.Tensor<Float>(from: truth)
  let mse = (error .* error).reduced(.mean, axis: [0, 1])
  debugPrint(mse)
  */
  let encoderHidProj = Dense(count: 2048)
  encoderHidProj.compile(inputs: out[0])
  graph.openStore("/home/liu/workspace/swift-diffusion/kolors_f16.ckpt") {
    $0.read("encoder_hid_proj", model: encoderHidProj)
  }
  let encoderOut = encoderHidProj(inputs: out[0]).map { $0.as(of: FloatType.self) }
  return (encoderOut[0].reshaped(.CHW(2, 256, 2048)), out[1])
}

debugPrint(crossattn)

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

let kvs0 = graph.withNoGrad {
  let unetBaseFixed = UNetXLFixed(
    batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
    inputAttentionRes: [2: [2, 2], 4: [10, 10]], middleAttentionBlock: 10,
    outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]])
  unetBaseFixed.maxConcurrency = .limit(1)
  unetBaseFixed.compile(inputs: crossattn)
  graph.openStore("/home/liu/workspace/swift-diffusion/kolors_f16.ckpt") {
    $0.read("unet_fixed", model: unetBaseFixed)
  }
  let result = unetBaseFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
  return result
}

DynamicGraph.setSeed(120)

let unconditionalGuidanceScale: Float = 5
let scaleFactor: Float = 0.13025
let startHeight = 128
let startWidth = 128
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 30)
let alphasCumprod = model.alphasCumprod
let sigmasForTimesteps = DiffusionModel.sigmas(from: alphasCumprod)
// This is for Karras scheduler (used in DPM++ 2M Karras)
let sigmas = model.karrasSigmas(sigmasForTimesteps[0]...sigmasForTimesteps[999])

let startTime = Date()
let originalWidth = Tensor<FloatType>(
  from: timeEmbedding(
    timestep: 1024, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
let originalHeight = Tensor<FloatType>(
  from: timeEmbedding(
    timestep: 1024, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
var originalSize = Tensor<FloatType>(.CPU, .C(512))
originalSize[0..<256] = originalHeight
originalSize[256..<512] = originalWidth
let cropX = Tensor<FloatType>(
  from: timeEmbedding(timestep: 0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
let cropY = Tensor<FloatType>(
  from: timeEmbedding(timestep: 0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
var cropCoord = Tensor<FloatType>(.CPU, .C(512))
cropCoord[0..<256] = cropY
cropCoord[256..<512] = cropX
let targetWidth = Tensor<FloatType>(
  from: timeEmbedding(timestep: 1024, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
let targetHeight = Tensor<FloatType>(
  from: timeEmbedding(
    timestep: 1024, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
var targetSize = Tensor<FloatType>(.CPU, .C(512))
targetSize[0..<256] = targetHeight
targetSize[256..<512] = targetWidth
let z = graph.withNoGrad {
  var vector0 = graph.variable(.GPU(0), .NC(2, 1536 + 4096), of: FloatType.self)
  vector0[0..<1, 0..<4096] = pooled[255..<256, 0..<4096]
  vector0[1..<2, 0..<4096] = pooled[511..<512, 0..<4096]
  vector0[0..<1, 4096..<4608] = graph.variable(originalSize.toGPU(0))
  vector0[1..<2, 4096..<4608] = graph.variable(originalSize.toGPU(0))
  vector0[0..<1, 4608..<5120] = graph.variable(cropCoord.toGPU(0))
  vector0[1..<2, 4608..<5120] = graph.variable(cropCoord.toGPU(0))
  vector0[0..<1, 5120..<5632] = graph.variable(targetSize.toGPU(0))
  vector0[1..<2, 5120..<5632] = graph.variable(targetSize.toGPU(0))
  let x_T = graph.variable(.GPU(0), .NCHW(1, 4, 128, 128), of: FloatType.self)
  x_T.randn(std: 1, mean: 0)
  var x = x_T
  var xIn = graph.variable(.GPU(0), .NCHW(2, 4, 128, 128), of: FloatType.self)
  let ts = timeEmbedding(timestep: 0, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000).toGPU(0)
  let unet = UNetXL(
    batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
    inputAttentionRes: [2: [2, 2], 4: [10, 10]], middleAttentionBlock: 10,
    outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]])
  unet.maxConcurrency = .limit(1)
  unet.compile(inputs: [xIn, graph.variable(Tensor<FloatType>(from: ts)), vector0] + kvs0)
  graph.openStore("/home/liu/workspace/swift-diffusion/kolors_f16.ckpt") {
    $0.read("unet", model: unet)
  }
  var oldDenoised: DynamicGraph.Tensor<FloatType>? = nil
  // Now do DPM++ 2M Karras sampling. (DPM++ 2S a Karras requires two denoising per step, not ideal for my use case).
  x = sigmas[0] * x
  for i in 0..<model.steps {
    let sigma = sigmas[i]
    let timestep = DiffusionModel.timestep(from: sigma, sigmas: sigmasForTimesteps)
    let ts = timeEmbedding(
      timestep: timestep, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000
    ).toGPU(0)
    let t = graph.variable(Tensor<FloatType>(from: ts))
    let cIn = 1.0 / (sigma * sigma + 1).squareRoot()
    let cOut = -sigma
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = cIn * x
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = cIn * x
    var et = unet(inputs: xIn, [t, vector0] + kvs0)[0].as(
      of: FloatType.self)
    var etUncond = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
    var etCond = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
    etUncond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
    etCond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[1..<2, 0..<4, 0..<startHeight, 0..<startWidth]
    et = etUncond + unconditionalGuidanceScale * (etCond - etUncond)
    let denoised = x + cOut * et
    let h = log(sigmas[i]) - log(sigmas[i + 1])
    if let oldDenoised = oldDenoised, i < model.steps - 1 {
      let hLast = log(sigmas[i - 1]) - log(sigmas[i])
      let r = (h / hLast) / 2
      let denoisedD = (1 + r) * denoised - r * oldDenoised
      let w = sigmas[i + 1] / sigmas[i]
      x = w * x - (w - 1) * denoisedD
    } else if i == model.steps - 1 {
      x = denoised
    } else {
      let w = sigmas[i + 1] / sigmas[i]
      x = w * x - (w - 1) * denoised
    }
    oldDenoised = denoised
  }
  return 1.0 / scaleFactor * x
}

graph.withNoGrad {
  let decoder = ModelBuilder {
    let startWidth = $0[0].shape[3]
    let startHeight = $0[0].shape[2]
    return Decoder(
      channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
      startHeight: startHeight)
  }
  let z32 = DynamicGraph.Tensor<Float>(from: z)
  decoder.compile(inputs: z32)
  graph.openStore("/home/liu/workspace/swift-diffusion/sdxl_vae_v1.0_f16.ckpt") {
    $0.read("decoder", model: decoder)
  }
  let img = decoder(inputs: z32)[0].as(of: Float.self)
    .toCPU()
  print("Total time \(Date().timeIntervalSince(startTime))")
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
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/txt2img.png", level: 4)
}
