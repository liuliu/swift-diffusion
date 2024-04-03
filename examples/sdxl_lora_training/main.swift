import Diffusion
import Foundation
import NNC
import PNG

public typealias FloatType = Float16

private func LoRAOpenCLIPMLP(hiddenSize: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let fc1 = LoRADense(count: intermediateSize)
  var out = fc1(x)
  out = GELU()(out)
  let fc2 = LoRADense(count: hiddenSize)
  out = fc2(out)
  return Model([x], [out])
}

public func LoRAOpenCLIPEncoderLayer(k: Int, h: Int, b: Int, t: Int, intermediateSize: Int) -> Model
{
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let attention = LoRACLIPAttention(k: k, h: h, b: b, t: t)
  out = attention(out, casualAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let mlp = LoRAOpenCLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return Model([x, casualAttentionMask], [out])
}

func LoRAOpenCLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    FloatType.self, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 {
      penultimate = out
    }
    let encoderLayer =
      LoRAOpenCLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [penultimate!, out], trainable: false)
}

func LoRALabelEmbed(modelChannels: Int) -> Model {
  let x = Input()
  let fc0 = LoRADense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: modelChannels * 4)
  out = fc2(out)
  return Model([x], [out])
}

func LoRACrossAttentionKeysAndValues(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = LoRADense(count: k * h, noBias: true)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = LoRADense(count: k * h)
  out = unifyheads(out)
  return Model([x, keys, values], [out])
}

private func LoRABasicTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let keys = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let attn1 = LoRASelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let attn2 = LoRACrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, keys, values) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let ff = LoRAFeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  ff.gradientCheckpointing = true
  out = ff(out) + residual
  return Model([x, keys, values], [out])
}

private func LoRASpatialTransformer(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> Model {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = LoRAConvolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).permuted(0, 2, 1)
  for i in 0..<depth {
    let keys = Input()
    kvs.append(keys)
    let values = Input()
    kvs.append(values)
    let block = LoRABasicTransformerBlock(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    out = block(out, keys, values)
  }
  out = out.transposed(1, 2).reshaped([b, k * h, height, width])
  let projOut = LoRAConvolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  return Model([x] + kvs, [out])
}

private func LoRABlockLayer(
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
  let resBlock = LoRAResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  if attentionBlock > 0 {
    let c = (0..<(attentionBlock * 2)).map { _ in Input() }
    let transformer = LoRASpatialTransformer(
      prefix: "\(prefix).\(layerStart).1",
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  return Model([x, emb] + kvs, [out])
}

func LoRAMiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, x: Model.IO, emb: Model.IO
) -> (Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let resBlock1 =
    LoRAResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let kvs = (0..<(attentionBlock * 2)).map { _ in Input() }
  let transformer = LoRASpatialTransformer(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  out = transformer([out] + kvs)
  let resBlock2 =
    LoRAResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  return (out, kvs)
}

func LoRAInputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO
) -> ([Model.IO], Model.IO, [Input]) {
  let conv2d = LoRAConvolution(
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
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      let inputLayer = LoRABlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      let c = (0..<(attentionBlock * 2)).map { _ in Input() }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
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
  return (passLayers, out, kvs)
}

func LoRAOutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO,
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
    let attentionBlock = attentionRes[ds, default: 0]
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let outputLayer = LoRABlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      let c = (0..<(attentionBlock * 2)).map { _ in Input() }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
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
  return (out, kvs)
}

func LoRAUNetXL(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  attentionRes: KeyValuePairs<Int, Int>
) -> Model {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let timeEmbed = LoRATimeEmbed(modelChannels: channels[0])
  let labelEmbed = LoRALabelEmbed(modelChannels: channels[0])
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (inputs, inputBlocks, inputKVs) = LoRAInputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    x: x, emb: emb)
  var out = inputBlocks
  let (middleBlock, middleKVs) = LoRAMiddleBlock(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 77, attentionBlock: middleBlockAttentionBlock, x: out, emb: emb)
  out = middleBlock
  let (outputBlocks, outputKVs) = LoRAOutputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    x: out, emb: emb, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = LoRAConvolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  return Model([x, t_emb, y] + inputKVs + middleKVs + outputKVs, [out], trainable: false)
}

func LoRACrossAttentionFixed(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
  let c = Input()
  let tokeys = LoRADense(count: k * h, noBias: true)
  let tovalues = LoRADense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
  return Model([c], [keys, values])
}

func LoRABasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> Model {
  let attn2 = LoRACrossAttentionFixed(k: k, h: h, b: b, hw: hw, t: t)
  return attn2
}

func LoRASpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> Model {
  let c = Input()
  var outs = [Model.IO]()
  let hw = height * width
  for i in 0..<depth {
    let block = LoRABasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(block(c))
  }
  return Model([c], outs)
}

func LoRABlockLayerFixed(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> Model {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let transformer = LoRASpatialTransformerFixed(
    prefix: "\(prefix).\(layerStart).1",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingSize,
    intermediateSize: channels * 4)
  return transformer
}

func LoRAMiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, c: Model.IO
) -> Model.IO {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let transformer = LoRASpatialTransformerFixed(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  let out = transformer(c)
  return out
}

func LoRAInputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO
) -> [Model.IO] {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      if attentionBlock > 0 {
        let inputLayer = LoRABlockLayerFixed(
          prefix: "input_blocks",
          layerStart: layerStart, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
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

func LoRAOutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO
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
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<(numRepeat + 1) {
      if attentionBlock > 0 {
        let outputLayer = LoRABlockLayerFixed(
          prefix: "output_blocks",
          layerStart: layerStart, skipConnection: true,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        outs.append(outputLayer(c))
      }
      layerStart += 1
    }
  }
  return outs
}

func LoRAUNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  attentionRes: KeyValuePairs<Int, Int>
) -> Model {
  let c = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let inputBlocks = LoRAInputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    c: c)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let middleBlock = LoRAMiddleBlockFixed(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 77, attentionBlock: middleBlockAttentionBlock, c: c)
  out.append(middleBlock)
  let outputBlocks = LoRAOutputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    c: c)
  out.append(contentsOf: outputBlocks)
  return Model([c], out, trainable: false)
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

let tokenizer0 = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let tokenizer1 = CLIPTokenizer(
  vocabulary: "examples/open_clip/vocab_16e6.json",
  merges: "examples/open_clip/bpe_simple_vocab_16e6.txt")

let prompt =
  //  "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
  //  "a professional photograph of an astronaut riding a horse, detailed, 8k"
  "a smiling indian man with a google t-shirt next to a frowning asian man with a shirt saying nexus at a meeting table facing each other, photograph, detailed, 8k"
let negativePrompt = ""

let tokens0 = tokenizer0.tokenize(text: prompt, truncation: true, maxLength: 77)
let tokens1 = tokenizer1.tokenize(text: prompt, truncation: true, maxLength: 77, paddingToken: 0)
let unconditionalTokens0 = tokenizer0.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77)
let unconditionalTokens1 = tokenizer1.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77, paddingToken: 0)

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

let tokensTensor0 = graph.variable(.CPU, .C(1 * 77), of: Int32.self)
let tokensTensor1 = graph.variable(.CPU, .C(1 * 77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(1 * 77), of: Int32.self)
for i in 0..<77 {
  // tokensTensor0[i] = unconditionalTokens0[i]
  // tokensTensor0[i + 77] = tokens0[i]
  // tokensTensor1[i] = unconditionalTokens1[i]
  // tokensTensor1[i + 77] = tokens1[i]
  tokensTensor0[i] = tokens0[i]
  tokensTensor1[i] = tokens1[i]
  positionTensor[i] = Int32(i)
  // positionTensor[i + 77] = Int32(i)
}

let casualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
  }
}

let tokensTensor0GPU = tokensTensor0.toGPU(0)
let positionTensorGPU = positionTensor.toGPU(0)
let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
let textModel0 = LoRACLIPTextModel(
  FloatType.self,
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 11, numHeads: 12,
  batchSize: 1, intermediateSize: 3072, noFinalLayerNorm: true)
textModel0.compile(inputs: tokensTensor0GPU, positionTensorGPU, casualAttentionMaskGPU)
graph.openStore("/home/liu/workspace/swift-diffusion/clip_vit_l14_f32.ckpt") {
  $0.read("text_model", model: textModel0) { name, dataType, format, shape in
    if name.contains("lora_up") {
      precondition(dataType == .Float32)
      var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<Float32>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return .final(tensor)
    }
    return .continue(name)
  }
}

let tokensTensor1GPU = tokensTensor1.toGPU(0)
let textModel1 = LoRAOpenCLIPTextModel(
  vocabularySize: 49408, maxLength: 77, embeddingSize: 1280, numLayers: 32, numHeads: 20,
  batchSize: 1, intermediateSize: 5120)
let textProjection = graph.variable(.GPU(0), .NC(1280, 1280), of: FloatType.self)
textModel1.compile(inputs: tokensTensor1GPU, positionTensorGPU, casualAttentionMaskGPU)
graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_vit_bigg14_f16.ckpt") {
  $0.read("text_model", model: textModel1) { name, dataType, format, shape in
    if name.contains("lora_up") {
      precondition(dataType == .Float32)
      var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<Float32>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return .final(tensor)
    }
    return .continue(name)
  }
  $0.read("text_projection", variable: textProjection)
}

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
let aestheticScore = Tensor<FloatType>(
  from: timeEmbedding(timestep: 6.0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))

DynamicGraph.setSeed(120)

let unconditionalGuidanceScale: Float = 5
let scaleFactor: Float = 0.13025
let startHeight = 128
let startWidth = 128
let refinerTimestep: Float = 300
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 30)
let alphasCumprod = model.alphasCumprod

var initImg = Tensor<FloatType>(.CPU, .NCHW(1, 3, startHeight * 8, startWidth * 8))
if let image = try PNG.Data.Rectangular.decompress(
  path: "/home/liu/workspace/swift-diffusion/init_img.png")
{
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let pixel = rgba[y * startWidth * 8 + x]
      initImg[0, 0, y, x] = FloatType(Float(pixel.r) / 255 * 2 - 1)
      initImg[0, 1, y, x] = FloatType(Float(pixel.g) / 255 * 2 - 1)
      initImg[0, 2, y, x] = FloatType(Float(pixel.b) / 255 * 2 - 1)
    }
  }
}

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
  graph.openStore("/home/liu/workspace/swift-diffusion/sdxl_vae_v1.0_f16.ckpt") {
    $0.read("encoder", model: encoder)
  }
  let encoded = encoder(inputs: initImgGPU)[0].as(of: FloatType.self)
  return scaleFactor * encoded[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
}

let c0 = textModel0(inputs: tokensTensor0GPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
  of: FloatType.self
).reshaped(.CHW(1, 77, 768))
let c = textModel1(inputs: tokensTensor1GPU, positionTensorGPU, casualAttentionMaskGPU).map {
  $0.as(of: FloatType.self)
}
var pooled: DynamicGraph.Tensor<FloatType>!
let c1 = c[0].reshaped(.CHW(1, 77, 1280))
for (i, token) in tokens1.enumerated() {
  if token == tokenizer1.endToken {
    pooled = c[1][i..<(i + 1), 0..<1280] * textProjection
    break
  }
}

var crossattn = graph.variable(.GPU(0), .CHW(1, 77, 2048), of: FloatType.self)
crossattn[0..<1, 0..<77, 0..<768] = c0
crossattn[0..<1, 0..<77, 768..<2048] = c1
let unetBaseFixed = LoRAUNetXLFixed(
  batchSize: 1, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
  attentionRes: [2: 2, 4: 10])
unetBaseFixed.maxConcurrency = .limit(1)
unetBaseFixed.compile(inputs: crossattn)
graph.openStore("/home/liu/workspace/swift-diffusion/sd_xl_base_1.0_f16.ckpt") {
  $0.read("unet_fixed", model: unetBaseFixed) { name, dataType, format, shape in
    if name.contains("lora_up") {
      precondition(dataType == .Float32)
      var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
      tensor.withUnsafeMutableBytes {
        let size = shape.reduce(MemoryLayout<Float32>.size, *)
        memset($0.baseAddress!, 0, size)
      }
      return .final(tensor)
    }
    return .continue(name)
  }
}
let kvs0 = unetBaseFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }

var vector0 = graph.variable(.GPU(0), .NC(1, 2816), of: FloatType.self)
vector0[0..<1, 0..<1280] = pooled
vector0[0..<1, 1280..<1792] = graph.variable(originalSize.toGPU(0))
vector0[0..<1, 1792..<2304] = graph.variable(cropCoord.toGPU(0))
vector0[0..<1, 2304..<2816] = graph.variable(targetSize.toGPU(0))
let x_T = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
x_T.randn(std: 1, mean: 0)
var x = x_T
let ts = timeEmbedding(timestep: 0, batchSize: 1, embeddingSize: 320, maxPeriod: 10_000).toGPU(0)
let unet = LoRAUNetXL(
  batchSize: 1, startHeight: startHeight, startWidth: startWidth, channels: [320, 640, 1280],
  attentionRes: [2: 2, 4: 10])
unet.maxConcurrency = .limit(1)
unet.memoryReduction = true
unet.compile(inputs: [x, graph.variable(Tensor<FloatType>(from: ts)), vector0] + kvs0)
graph.openStore("/home/liu/workspace/swift-diffusion/sd_xl_base_1.0_f16.ckpt") {
  $0.read("unet", model: unet) { name, dataType, format, shape in
    if name.contains("lora_up") {
      if dataType == .Float32 {
        var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
        tensor.withUnsafeMutableBytes {
          let size = shape.reduce(MemoryLayout<Float32>.size, *)
          memset($0.baseAddress!, 0, size)
        }
        return .final(tensor)
      } else {
        var tensor = Tensor<Float16>(.CPU, format: format, shape: shape)
        tensor.withUnsafeMutableBytes {
          let size = shape.reduce(MemoryLayout<Float16>.size, *)
          memset($0.baseAddress!, 0, size)
        }
        return .final(tensor)
      }
    }
    return .continue(name)
  }
}

var adamWOptimizer = AdamWOptimizer(
  graph, rate: 0.0001, betas: (0.9, 0.999), decay: 0.001, epsilon: 1e-8)
adamWOptimizer.parameters = [unet.parameters, unetBaseFixed.parameters]
let startTime = Date()
var accumulateGradSteps = 0
let minSNRGamma: Float = 1
var scaler = GradScaler()
for epoch in 0..<1000 {
  let noise = graph.variable(.GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
  noise.randn(std: 1, mean: 0)
  let timestep = Int.random(in: 0...999)
  let c0 = textModel0(inputs: tokensTensor0GPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
    of: FloatType.self
  ).reshaped(.CHW(1, 77, 768))
  let c = textModel1(inputs: tokensTensor1GPU, positionTensorGPU, casualAttentionMaskGPU).map {
    $0.as(of: FloatType.self)
  }
  var pooled: DynamicGraph.Tensor<FloatType>!
  let c1 = c[0].reshaped(.CHW(1, 77, 1280))
  for (i, token) in tokens1.enumerated() {
    if token == tokenizer1.endToken {
      pooled = c[1][i..<(i + 1), 0..<1280] * textProjection
      break
    }
  }

  var crossattn = graph.variable(.GPU(0), .CHW(1, 77, 2048), of: FloatType.self)
  crossattn[0..<1, 0..<77, 0..<768] = c0
  crossattn[0..<1, 0..<77, 768..<2048] = c1
  let kvs0 = unetBaseFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
  var vector0 = graph.variable(.GPU(0), .NC(1, 2816), of: FloatType.self)
  vector0[0..<1, 0..<1280] = pooled
  vector0[0..<1, 1280..<1792] = graph.variable(originalSize.toGPU(0))
  vector0[0..<1, 1792..<2304] = graph.variable(cropCoord.toGPU(0))
  vector0[0..<1, 2304..<2816] = graph.variable(targetSize.toGPU(0))
  let sqrtAlphasCumprod = alphasCumprod[timestep].squareRoot()
  let sqrtOneMinusAlphasCumprod = (1 - alphasCumprod[timestep]).squareRoot()
  let snr = alphasCumprod[timestep] / (1 - alphasCumprod[timestep])
  let gammaOverSNR = minSNRGamma / snr
  let snrWeight = min(gammaOverSNR, 1)
  let noisyLatents = sqrtAlphasCumprod * latents + sqrtOneMinusAlphasCumprod * noise
  let ts = timeEmbedding(timestep: timestep, batchSize: 1, embeddingSize: 320, maxPeriod: 10_000)
    .toGPU(0)
  let t = graph.variable(Tensor<FloatType>(from: ts))
  let et = unet(inputs: noisyLatents, [t, vector0] + kvs0)[0].as(of: FloatType.self)
  let d = et - noise
  let loss = snrWeight * (d .* d).reduced(.mean, axis: [1, 2, 3])
  scaler.scale(loss).backward(to: [latents, tokensTensor0GPU, tokensTensor1GPU])
  let value = loss.toCPU()[0, 0, 0, 0]
  if accumulateGradSteps == 5 {
    scaler.step(&adamWOptimizer)
    accumulateGradSteps = 0
  } else {
    accumulateGradSteps += 1
  }
  print(
    "epoch: \(epoch), \(timestep), loss: \(value), step \(adamWOptimizer.step), scale \(scaler.scale)"
  )
  if value.isNaN {
    fatalError()
  }
}

graph.openStore("/home/liu/workspace/swift-diffusion/sdxl_lora_training.ckpt") {
  $0.write("lora_unet", model: unet)
  $0.write("lora_unet_fixed", model: unetBaseFixed)
  $0.write("lora_text_model_0", model: textModel0)
  $0.write("lora_text_model_1", model: textModel1)
}
