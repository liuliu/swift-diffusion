import Diffusion
import Foundation
import NNC
import PNG
import SentencePiece

typealias FloatType = Float16

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

let prompt =
  CommandLine.arguments.count > 1
  ? CommandLine.arguments.suffix(from: 1).joined(separator: " ") : ""

func XLMRobertaTextEmbedding(
  prefix: String, vocabularySize: Int, maxLength: Int, tokenTypes: Int, embeddingSize: Int
) -> Model {
  let tokens = Input()
  let tokenType = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    FloatType.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let tokenTypeEmbed = Embedding(
    FloatType.self, vocabularySize: tokenTypes, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(
    FloatType.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + tokenTypeEmbed(tokenType) + positionEmbed(positions)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  let out = layerNorm(embedding)
  return Model([tokens, positions, tokenType], [out], name: "embeddings")
}

func XLMRobertaSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> Model {
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
  return Model([x, casualAttentionMask], [out])
}

func XLMRobertaLayer(prefix: String, k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let selfAttention = XLMRobertaSelfAttention(
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
  return Model([x, casualAttentionMask], [out])
}

func XLMRobertaModel(numberOfLayers: Int, k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  var out: Model.IO = x
  for i in 0..<numberOfLayers {
    let layer = XLMRobertaLayer(
      prefix: "model.transformer.encoder.layer.\(i)", k: k, h: h, b: b, t: t)
    out = layer(out, casualAttentionMask)
  }
  return Model([x, casualAttentionMask], [out])
}

func SelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> Model {
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
  return Model([x, casualAttentionMask], [out])
}

func ResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  let selfAttention = SelfAttention(
    prefix: "\(prefix).attn", k: k, h: h, b: b, t: t)
  var out = x + selfAttention(layerNorm1(x), casualAttentionMask)
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let intermediate = Dense(count: k * h * 4)
  let output = Dense(count: k * h)
  out = out + output(intermediate(layerNorm2(out)).GELU())
  return Model([x, casualAttentionMask], [out])
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

func timestepEmbedding(prefix: String, channels: Int) -> Model {
  let x = Input()
  let dense1 = Dense(count: channels)
  var out = dense1(x).swish()
  let dense2 = Dense(count: channels)
  out = dense2(out)
  return Model([x], [out])
}

func DiffusionMapping(numberOfLayers: Int, k: Int, h: Int, b: Int, t: Int, outChannels: Int)
  -> Model
{
  let x = Input()
  let casualAttentionMask = Input()
  var out: Model.IO = x
  for i in 0..<numberOfLayers {
    let layer = ResidualAttentionBlock(
      prefix: "model.transformer.resblocks.\(i)", k: k, h: h, b: b, t: t)
    out = layer(out, casualAttentionMask)
  }
  let finalLn = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLn(out)
  let outProj = Dense(count: outChannels)
  out = outProj(
    out.reshaped([b, 1, k * h], offset: [0, t - 1, 0], strides: [t * k * h, k * h, 1]))
  return Model([x, casualAttentionMask], [out])
}

func ResBlock(
  prefix: String, batchSize: Int, outChannels: Int, up: Bool, down: Bool, skipConnection: Bool
) -> Model {
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
  if skipConnection {
    let conv = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1])
    xhd = conv(xhd)
  }
  out = xhd + out
  return Model([x, emb], [out])
}

func AttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int, height: Int, width: Int)
  -> Model
{
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
  return Model([x, encoderOut], [out])
}

func InputBlocks(
  prefix: String, batchSize: Int, channels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>,
  x: Model.IO, emb: Model.IO, xfOut: Model.IO
) -> (Model.IO, [Model.IO]) {
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
      let resBlock = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: false,
        skipConnection: ch != lastCh)
      out = resBlock(out, emb)
      lastCh = ch
      if attentionResolutions.contains(ds) {
        let attentionBlock = AttentionBlock(
          prefix: "\(prefix).\(i).1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize,
          t: t, height: height, width: width)
        out = attentionBlock(out, xfOut)
      }
      hs.append(out)
      i += 1
    }
    if level != channelMult.count - 1 {
      let resBlock = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: true,
        skipConnection: false)
      out = resBlock(out, emb)
      hs.append(out)
      i += 1
      ds *= 2
      height /= 2
      width /= 2
    }
  }
  return (out, hs)
}

func OutputBlocks(
  prefix: String, batchSize: Int, channels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>,
  x: Model.IO, emb: Model.IO, xfOut: Model.IO, hs: [Model.IO]
) -> Model.IO {
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
      let resBlock = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: false,
        skipConnection: true)
      out = resBlock(out, emb)
      if attentionResolutions.contains(ds) {
        let attentionBlock = AttentionBlock(
          prefix: "\(prefix).\(i).1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize,
          t: t, height: height, width: width)
        out = attentionBlock(out, xfOut)
      }
      if level > 0 && j == numResBlocks {
        let resBlock = ResBlock(
          prefix: "\(prefix).\(i).2", batchSize: batchSize, outChannels: ch, up: true, down: false,
          skipConnection: false)
        out = resBlock(out, emb)
        ds /= 2
        height *= 2
        width *= 2
      }
      i += 1
    }
  }
  return out
}

func ImageAndTextEmbedding(batchSize: Int) -> Model {
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
  return Model([poolEmb, fullEmb, imageEmb], [xfProj, xfOut])
}

func UNet(
  batchSize: Int, channels: Int, outChannels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>
) -> Model {
  let x = Input()
  let emb = Input()
  let xfOut = Input()
  let (inputBlocksOut, hs) = InputBlocks(
    prefix: "input_blocks", batchSize: batchSize, channels: channels, channelMult: channelMult,
    numResBlocks: numResBlocks, numHeadChannels: numHeadChannels, t: t, startHeight: startHeight,
    startWidth: startWidth, attentionResolutions: attentionResolutions, x: x, emb: emb, xfOut: xfOut
  )
  let ch = channelMult[channelMult.count - 1] * channels
  var out = inputBlocksOut
  let middleResBlock1 = ResBlock(
    prefix: "middle_block.0", batchSize: batchSize, outChannels: ch, up: false, down: false,
    skipConnection: false)
  out = middleResBlock1(out, emb)
  var height = startHeight
  var width = startWidth
  for _ in 1..<channelMult.count {
    height /= 2
    width /= 2
  }
  let middleAttentionBlock2 = AttentionBlock(
    prefix: "middle_block.1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize, t: t,
    height: height, width: width)
  out = middleAttentionBlock2(out, xfOut)
  let middleResBlock3 = ResBlock(
    prefix: "middle_block.2", batchSize: batchSize, outChannels: ch, up: false, down: false,
    skipConnection: false)
  out = middleResBlock3(out, emb)
  let outputBlocksOut = OutputBlocks(
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
  return Model([x, emb, xfOut], [out])
}

func ResnetBlock(prefix: String, inChannels: Int, outChannels: Int) -> Model {
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
  if inChannels != outChannels {
    let shortcut = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1])
    out = shortcut(x) + out
  } else {
    out = x + out
  }
  return Model([x], [out])
}

func AttnBlock(
  prefix: String, inChannels: Int, batchSize: Int, height: Int, width: Int
) -> Model {
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

func Encoder(
  zChannels: Int, channels: Int, channelMult: [Int], numResBlocks: Int, startHeight: Int,
  startWidth: Int, attnResolutions: Set<Int>
) -> Model {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var lastCh = channels
  var currentRes = 256
  var height = startHeight
  var width = startWidth
  for (i, mult) in channelMult.enumerated() {
    let ch = channels * mult
    for j in 0..<numResBlocks {
      let resnetBlock = ResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", inChannels: lastCh, outChannels: ch)
      out = resnetBlock(out)
      lastCh = ch
      if attnResolutions.contains(currentRes) {
        let attnBlock = AttnBlock(
          prefix: "encoder.down.\(i).attn.\(j)", inChannels: ch, batchSize: 1, height: height,
          width: width)
        out = attnBlock(out)
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
    }
  }
  let midResnetBlock1 = ResnetBlock(
    prefix: "encoder.mid.block_1", inChannels: lastCh, outChannels: lastCh)
  out = midResnetBlock1(out)
  let midAttnBlock1 = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: lastCh, batchSize: 1, height: height, width: width)
  out = midAttnBlock1(out)
  let midResnetBlock2 = ResnetBlock(
    prefix: "encoder.mid.block_2", inChannels: lastCh, outChannels: lastCh)
  out = midResnetBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: zChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let quantConv = Convolution(groups: 1, filters: zChannels, filterSize: [1, 1])
  out = quantConv(out)
  return Model([x], [out])
}

func SpatialNorm(prefix: String, channels: Int, heightScale: Float, widthScale: Float) -> Model {
  let x = Input()
  let zq = Input()
  let normLayer = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = normLayer(x)
  let zqOut = Upsample(.nearest, widthScale: widthScale, heightScale: heightScale)(zq)
  let convY = Convolution(groups: 1, filters: channels, filterSize: [1, 1])
  let convB = Convolution(groups: 1, filters: channels, filterSize: [1, 1])
  out = out .* convY(zqOut) + convB(zqOut)
  return Model([x, zq], [out])
}

func MOVQResnetBlock(prefix: String, inChannels: Int, outChannels: Int, scale: Float) -> Model {
  let x = Input()
  let zq = Input()
  let norm1 = SpatialNorm(
    prefix: "\(prefix).norm1", channels: inChannels, heightScale: scale, widthScale: scale)
  var out = norm1(x, zq).swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = SpatialNorm(
    prefix: "\(prefix).norm2", channels: outChannels, heightScale: scale, widthScale: scale)
  out = norm2(out, zq).swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  if inChannels != outChannels {
    let shortcut = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1])
    out = shortcut(x) + out
  } else {
    out = x + out
  }
  return Model([x, zq], [out])
}

func MOVQAttnBlock(
  prefix: String, inChannels: Int, batchSize: Int, height: Int, width: Int, scale: Float
) -> Model {
  let x = Input()
  let zq = Input()
  let norm = SpatialNorm(
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
  return Model([x, zq], [out])
}

func MOVQDecoder(
  zChannels: Int, channels: Int, channelMult: [Int], numResBlocks: Int, startHeight: Int,
  startWidth: Int, attnResolutions: Set<Int>
) -> Model {
  let x = Input()
  let postQuantConv = Convolution(groups: 1, filters: zChannels, filterSize: [1, 1])
  let z = postQuantConv(x)
  var blockIn = channels * channelMult[channelMult.count - 1]
  let convIn = Convolution(
    groups: 1, filters: blockIn, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(z)
  let midBlock1 = MOVQResnetBlock(
    prefix: "decoder.mid.block_1", inChannels: blockIn, outChannels: blockIn, scale: 1)
  out = midBlock1(out, x)
  let midAttn1 = MOVQAttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: blockIn, batchSize: 1, height: startHeight,
    width: startWidth, scale: 1)
  out = midAttn1(out, x)
  let midBlock2 = MOVQResnetBlock(
    prefix: "decoder.mid.block_2", inChannels: blockIn, outChannels: blockIn, scale: 1)
  out = midBlock2(out, x)
  var ds = 1
  var currentRes = 32
  var height = startHeight
  var width = startWidth
  for (i, mult) in channelMult.enumerated().reversed() {
    let blockOut = channels * mult
    for j in 0..<(numResBlocks + 1) {
      let resnetBlock = MOVQResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", inChannels: blockIn, outChannels: blockOut,
        scale: Float(ds))
      out = resnetBlock(out, x)
      blockIn = blockOut
      if attnResolutions.contains(currentRes) {
        let attn = MOVQAttnBlock(
          prefix: "decoder.up.\(i).attn.\(j)", inChannels: blockIn, batchSize: 1, height: height,
          width: width, scale: Float(ds))
        out = attn(out, x)
      }
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv = Convolution(
        groups: 1, filters: blockIn, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv(out)
      ds *= 2
      currentRes *= 2
      height *= 2
      width *= 2
    }
  }
  let normOut = SpatialNorm(
    prefix: "decoder.norm_out", channels: blockIn, heightScale: Float(ds), widthScale: Float(ds))
  out = normOut(out, x).swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  return Model([x], [out])
}

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
var fullEmb1: DynamicGraph.Tensor<FloatType>? = nil
var poolEmb1: DynamicGraph.Tensor<FloatType>? = nil
var imageEmb1: DynamicGraph.Tensor<FloatType>? = nil
graph.withNoGrad {
  let textEncoder = XLMRobertaTextEmbedding(
    prefix: "model.transformer.embeddings", vocabularySize: 250_002, maxLength: 514, tokenTypes: 1,
    embeddingSize: 1_024)
  let tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  let sentencePiece = SentencePiece(
    file: "/home/liu/workspace/swift-diffusion/examples/kandinsky2/sentencepiece.bpe.model")
  let ids = sentencePiece.encode(prompt)
  for i in 0..<154 {
    tokensTensor[i] = 1
  }
  tokensTensor[0] = 0
  for i in 0..<ids.count {
    tokensTensor[i + 1] = Int32(ids[i] + 1)
  }
  tokensTensor[ids.count + 1] = 2
  tokensTensor[77] = 0
  tokensTensor[78] = 2
  let tokenTypesTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  for i in 0..<154 {
    tokenTypesTensor[i] = 0
    positionTensor[i] = 1
  }
  for i in 0..<ids.count + 2 {
    positionTensor[i] = Int32(i + 2)
  }
  positionTensor[77] = 2
  positionTensor[78] = 3
  let tokensTensorGPU = tokensTensor.toGPU(1)
  let positionTensorGPU = positionTensor.toGPU(1)
  let tokenTypesTensorGPU = tokenTypesTensor.toGPU(1)
  textEncoder.compile(inputs: tokensTensorGPU, positionTensorGPU, tokenTypesTensorGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/xlm_roberta_f16.ckpt") {
    $0.read("embedding", model: textEncoder)
  }
  let embeddings = textEncoder(inputs: tokensTensorGPU, positionTensorGPU, tokenTypesTensorGPU)[0]
    .as(
      of: FloatType.self)
  let layer = XLMRobertaModel(numberOfLayers: 24, k: 64, h: 16, b: 2, t: 77)
  let attentionMask = graph.variable(.CPU, .NCHW(2, 1, 1, 77), of: FloatType.self)
  attentionMask.full(0)
  for i in (ids.count + 2)..<77 {
    attentionMask[0, 0, 0, i] = -FloatType.greatestFiniteMagnitude
  }
  for i in 2..<77 {
    attentionMask[1, 0, 0, i] = -FloatType.greatestFiniteMagnitude
  }
  let attentionMaskGPU = attentionMask.toGPU(1)
  layer.compile(inputs: embeddings, attentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/xlm_roberta_f16.ckpt") {
    $0.read("roberta", model: layer)
  }
  let textEncoderEmb = layer(inputs: embeddings, attentionMaskGPU)[0].as(of: FloatType.self)
    .reshaped(
      .CHW(2, 77, 1024))
  fullEmb1 = textEncoderEmb
  let poolingMask = graph.variable(.CPU, .CHW(2, 1, 77), of: FloatType.self)
  let weightPoolingMask = graph.variable(.CPU, .CHW(2, 1, 1), of: FloatType.self)
  poolingMask.full(0)
  for i in 0..<(ids.count + 2) {
    poolingMask[0, 0, i] = 1
  }
  weightPoolingMask[0, 0, 0] = FloatType(1 / Float(ids.count + 2))
  for i in 0..<2 {
    poolingMask[1, 0, i] = 1
  }
  weightPoolingMask[1, 0, 0] = 1 / 2
  let poolEmb = weightPoolingMask.toGPU(1) .* (poolingMask.toGPU(1) * textEncoderEmb)
  let linearTransformation = Dense(count: 768)
  linearTransformation.compile(inputs: poolEmb)
  graph.openStore("/home/liu/workspace/swift-diffusion/xlm_roberta_f16.ckpt") {
    $0.read("linear_transformation", model: linearTransformation)
  }
  let poolEmbOut = linearTransformation(inputs: poolEmb)[0].as(of: FloatType.self)
  debugPrint(poolEmbOut)
  poolEmb1 = poolEmbOut
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/xlm_roberta_f16.ckpt") {
    $0.write("embedding", model: textEncoder)
    $0.write("roberta", model: layer)
    $0.write("linear_transformation", model: linearTransformation)
  }
  */

  let tokenizer = CLIPTokenizer(
    vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

  let unconditionalTokens = tokenizer.tokenize(text: "", truncation: true, maxLength: 77)
  let tokens = tokenizer.tokenize(text: prompt, truncation: true, maxLength: 77)
  var tokenLength = 0
  for i in 0..<tokens.count {
    if tokens[i] == tokenizer.endToken {
      tokenLength = i + 1
      break
    }
  }

  let textModel = CLIPTextModel(
    FloatType.self,
    vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
    batchSize: 2, intermediateSize: 3072)

  let tokensTensorCLIP = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  let positionTensorCLIP = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
  for i in 0..<77 {
    // Kandinsky implementation error, need to replicate that.
    tokensTensorCLIP[i] = i >= tokenLength ? 0 : tokens[i]
    tokensTensorCLIP[i + 77] = i >= 2 ? 0 : unconditionalTokens[i]
    positionTensorCLIP[i] = Int32(i)
    positionTensorCLIP[i + 77] = Int32(i)
  }

  let casualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 77, 77)))
  casualAttentionMask.full(0)
  for i in 0..<76 {
    for j in (i + 1)..<77 {
      casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  let tokensTensorCLIPGPU = tokensTensorCLIP.toGPU(1)
  let positionTensorCLIPGPU = positionTensorCLIP.toGPU(1)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(1)
  textModel.compile(inputs: tokensTensorCLIPGPU, positionTensorCLIPGPU, casualAttentionMaskGPU)
  graph.openStore("/fast/Data/SD/swift-diffusion/clip_vit_l14_f16.ckpt") { store in
    store.read("text_model", model: textModel)
  }
  let textProjectionGPU = graph.variable(.GPU(1), .NC(768, 768), of: FloatType.self)
  let c = textModel(
    inputs: tokensTensorCLIPGPU, positionTensorCLIPGPU, casualAttentionMaskGPU)[0].as(
      of: FloatType.self
    )
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_diffusion_mapping_f16.ckpt") {
    $0.read("text_projection", variable: textProjectionGPU)
  }
  var indexGP = Tensor<Int32>(.CPU, .C(2))
  indexGP[0] = Int32(tokenLength) - 1
  indexGP[1] = 78
  let textEmb =
    Functional.indexSelect(
      input: c.reshaped(.NC(2 * 77, 768)), index: graph.variable(indexGP.toGPU(1)))
    * textProjectionGPU
  let textEnc = c.reshaped(.CHW(2, 77, 768))
  let textEncProj = Dense(count: 2048)
  let textEmbProj = Dense(count: 2048)
  let clipImgProj = Dense(count: 2048)
  let timeEmbed = timestepEmbedding(prefix: "model.time_embed", channels: 2048)
  var xIn = graph.variable(.GPU(1), .NC(2, 768), of: FloatType.self)
  textEncProj.compile(inputs: textEnc)
  textEmbProj.compile(inputs: textEmb)
  clipImgProj.compile(inputs: xIn)
  let timesteps = graph.variable(
    Tensor<FloatType>(
      from: timeEmbedding(timestep: 999, batchSize: 2, embeddingSize: 2048, maxPeriod: 10_000)
        .toGPU(1)))
  timeEmbed.compile(inputs: timesteps)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_diffusion_mapping_f16.ckpt") {
    $0.read("time_embed", model: timeEmbed)
    $0.read("clip_img_proj", model: clipImgProj)
    $0.read("text_enc_proj", model: textEncProj)
    $0.read("text_emb_proj", model: textEmbProj)
  }
  let textEncOut = textEncProj(inputs: textEnc)[0].as(of: FloatType.self)
  let textEmbOut = textEmbProj(inputs: textEmb)[0].as(of: FloatType.self)
  var dmInputTensorGPU = graph.variable(.GPU(1), .NC(2 * 81, 2048), of: FloatType.self)
  dmInputTensorGPU[0..<77, 0..<2048] = textEncOut[0..<1, 0..<77, 0..<2048].reshaped(.NC(77, 2048))
  dmInputTensorGPU[81..<(81 + 77), 0..<2048] = textEncOut[1..<2, 0..<77, 0..<2048].reshaped(
    .NC(77, 2048))
  dmInputTensorGPU[77..<78, 0..<2048] = textEmbOut[0..<1, 0..<2048]
  dmInputTensorGPU[(81 + 77)..<(81 + 78), 0..<2048] = textEmbOut[1..<2, 0..<2048]
  let prdEmb = graph.variable(.GPU(1), .NC(1, 2048), of: FloatType.self)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_diffusion_mapping_f16.ckpt") {
    $0.read("prd_emb", variable: prdEmb)
  }
  dmInputTensorGPU[80..<81, 0..<2048] = prdEmb
  dmInputTensorGPU[(81 + 80)..<(81 + 81), 0..<2048] = prdEmb
  let positionalEmbedding = graph.variable(.GPU(1), .NC(81, 2048), of: FloatType.self)
  let clipStd = graph.variable(.GPU(1), .NC(1, 768), of: FloatType.self)
  let clipMean = graph.variable(.GPU(1), .NC(1, 768), of: FloatType.self)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_diffusion_mapping_f16.ckpt") {
    $0.read("positional_embedding", variable: positionalEmbedding)
    $0.read("clip_std", variable: clipStd)
    $0.read("clip_mean", variable: clipMean)
  }
  var positionalEmbeddingGPU = graph.variable(.GPU(1), .NC(2 * 81, 2048), of: FloatType.self)
  positionalEmbeddingGPU[0..<81, 0..<2048] = positionalEmbedding
  positionalEmbeddingGPU[81..<(81 * 2), 0..<2048] = positionalEmbedding
  let diffusionMapping = DiffusionMapping(
    numberOfLayers: 20, k: 64, h: 32, b: 2, t: 81, outChannels: 768)
  let dmCasualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(2, 1, 81, 81)))
  dmCasualAttentionMask.full(0)
  for i in 0..<80 {
    for j in (i + 1)..<81 {
      dmCasualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      dmCasualAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  for i in 0..<81 {
    for j in tokenLength..<77 {
      dmCasualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
    for j in 2..<77 {
      dmCasualAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  let dmCasualAttentionMaskGPU = dmCasualAttentionMask.toGPU(1)
  diffusionMapping.compile(inputs: dmInputTensorGPU, dmCasualAttentionMaskGPU)
  let noiseGPU = graph.variable(.GPU(1), .NC(1, 768), of: FloatType.self)
  noiseGPU.randn()
  var x = noiseGPU
  let zeroImgEmbGPU = graph.variable(.GPU(1), .NC(1, 768), of: FloatType.self)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_diffusion_mapping_f16.ckpt") {
    $0.read("diffusion_mapping", model: diffusionMapping)
    $0.read("zero_img_emb", variable: zeroImgEmbGPU)
  }
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_diffusion_mapping_f16.ckpt") {
    $0.write("diffusion_mapping", model: diffusionMapping)
    $0.write("time_embed", model: timeEmbed)
    $0.write("clip_img_proj", model: clipImgProj)
    $0.write("text_enc_proj", model: textEncProj)
    $0.write("text_emb_proj", model: textEmbProj)
    $0.write("positional_embedding", variable: positionalEmbedding)
    $0.write("clip_std", variable: clipStd)
    $0.write("clip_mean", variable: clipMean)
    $0.write("zero_img_emb", variable: zeroImgEmbGPU)
    $0.write("text_projection", variable: textProjectionGPU)
    $0.write("prd_emb", variable: prdEmb)
  }
  */
  for (i, timestep) in [0, 250, 500, 749, 999].enumerated().reversed() {
    xIn[0..<1, 0..<768] = x
    xIn[1..<2, 0..<768] = x
    let timesteps = graph.variable(
      Tensor<FloatType>(
        from: timeEmbedding(
          timestep: timestep, batchSize: 1, embeddingSize: 2048, maxPeriod: 10_000
        ).toGPU(
          1)))
    let tEmb = timeEmbed(inputs: timesteps)[0].as(of: FloatType.self)
    let xProj = clipImgProj(inputs: xIn)[0].as(of: FloatType.self)
    dmInputTensorGPU[78..<79, 0..<2048] = tEmb
    dmInputTensorGPU[(81 + 78)..<(81 + 79), 0..<2048] = tEmb
    dmInputTensorGPU[79..<80, 0..<2048] = xProj[0..<1, 0..<2048]
    dmInputTensorGPU[(81 + 79)..<(81 + 80), 0..<2048] = xProj[1..<2, 0..<2048]
    let input = dmInputTensorGPU + positionalEmbeddingGPU
    let result = diffusionMapping(inputs: input, dmCasualAttentionMaskGPU)[0].as(
      of: FloatType.self)
    let condEps = result[0..<1, 0..<1, 0..<768].reshaped(.NC(1, 768))
    let uncondEps = result[1..<2, 0..<1, 0..<768].reshaped(.NC(1, 768))
    let eps = (uncondEps + 4 * (condEps - uncondEps)).clamped(-10...10)
    let posteriorMean = Functional.add(
      left: eps, right: x, leftScalar: Float(posteriorMeanCoef1[i]),
      rightScalar: Float(posteriorMeanCoef2[i]))
    noiseGPU.randn()
    if i > 0 {
      x = Functional.add(
        left: posteriorMean, right: noiseGPU,
        rightScalar: Float(exp(0.5 * posteriorLogVarianceClipped[i])))
    } else {
      x = posteriorMean
    }
  }
  let imageEmbGPU = x .* clipStd + clipMean
  var imageEmb = graph.variable(.GPU(1), .NC(2, 768), of: FloatType.self)
  imageEmb[0..<1, 0..<768] = Functional.add(
    left: zeroImgEmbGPU, right: imageEmbGPU, leftScalar: 1, rightScalar: 0)
  imageEmb[1..<2, 0..<768] = zeroImgEmbGPU
  imageEmb1 = imageEmb.reshaped(.CHW(2, 1, 768))
}

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

func percentile(_ tensor: DynamicGraph.Tensor<FloatType>) -> Float {
  let tensor = tensor.toCPU()
  var value = [Float]()
  for i in 0..<1 {
    for j in 0..<4 {
      for x in 0..<96 {
        for y in 0..<96 {
          value.append(abs(Float(tensor[i, j, x, y])))
        }
      }
    }
  }
  value = value.sorted()
  return value[Int(floor((Float(1 * 4 * 96 * 96 - 1) * 0.995)))]
}

var image: DynamicGraph.Tensor<FloatType>? = nil
graph.withNoGrad {
  guard let fullEmb1 = fullEmb1, let poolEmb1 = poolEmb1, let imageEmb1 = imageEmb1 else { return }
  let imageAndTextEmbedding = ImageAndTextEmbedding(batchSize: 2)
  imageAndTextEmbedding.compile(inputs: poolEmb1, fullEmb1, imageEmb1)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_f16.ckpt") {
    $0.read("image_and_text_embed", model: imageAndTextEmbedding)
  }
  let outputs = imageAndTextEmbedding(inputs: poolEmb1, fullEmb1, imageEmb1).map {
    $0.as(of: FloatType.self)
  }
  let xfProj = outputs[0]
  let xfOutGPU = outputs[1]
  let timesteps = graph.variable(
    Tensor<FloatType>(
      from: timeEmbedding(timestep: 999, batchSize: 2, embeddingSize: 384, maxPeriod: 10_000).toGPU(
        1)))
  let timeEmbed = timestepEmbedding(prefix: "time_embed", channels: 384 * 4)
  timeEmbed.compile(inputs: timesteps)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_f16.ckpt") {
    $0.read("time_embed", model: timeEmbed)
  }
  var embGPU = timeEmbed(inputs: timesteps)[0].as(of: FloatType.self)
  embGPU = embGPU + xfProj.reshaped(.NC(2, 384 * 4))
  let unet = UNet(
    batchSize: 2, channels: 384, outChannels: 8, channelMult: [1, 2, 3, 4], numResBlocks: 3,
    numHeadChannels: 64, t: 87, startHeight: 96, startWidth: 96,
    attentionResolutions: Set([2, 4, 8]))
  let hInputGPU = graph.variable(.GPU(1), .NCHW(1, 4, 96, 96), of: FloatType.self)
  hInputGPU.randn()
  var x = hInputGPU
  var xIn = graph.variable(.GPU(1), .NCHW(2, 4, 96, 96), of: FloatType.self)
  unet.compile(inputs: xIn, embGPU, xfOutGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_f16.ckpt") {
    $0.read("unet", model: unet)
  }
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_f16.ckpt") {
    $0.write("unet", model: unet)
    $0.write("time_embed", model: timeEmbed)
    $0.write("image_and_text_embed", model: imageAndTextEmbedding)
  }
  */
  for (i, timestep) in mTimesteps.enumerated().reversed() {
    let timesteps = graph.variable(
      Tensor<FloatType>(
        from: timeEmbedding(timestep: timestep, batchSize: 2, embeddingSize: 384, maxPeriod: 10_000)
          .toGPU(
            1)))
    embGPU = timeEmbed(inputs: timesteps)[0].as(of: FloatType.self)
    embGPU = embGPU + xfProj.reshaped(.NC(2, 384 * 4))
    xIn[0..<1, 0..<4, 0..<96, 0..<96] = x
    xIn[1..<2, 0..<4, 0..<96, 0..<96] = x
    let result = unet(inputs: xIn, embGPU, xfOutGPU)[0].as(of: FloatType.self)
    let modelVar = result[0..<1, 4..<8, 0..<96, 0..<96].copied().clamped(-1...1)
    let minLog = Float(mPosteriorLogVarianceClipped[i])
    let maxLog = Float(log(mNewBetas[i]))
    let frac = 0.5 * (modelVar + 1)
    let modelLogVar = frac * maxLog + (1 - frac) * minLog
    let condEps = result[0..<1, 0..<4, 0..<96, 0..<96].copied()
    let uncondEps = result[1..<2, 0..<4, 0..<96, 0..<96].copied()
    let eps = uncondEps + 10 * (condEps - uncondEps)
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

graph.withNoGrad {
  guard let image = image else { return }
  let movq = MOVQDecoder(
    zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2, startHeight: 96,
    startWidth: 96, attnResolutions: Set([32]))
  movq.compile(inputs: image)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_movq_f16.ckpt") {
    $0.read("movq", model: movq)
  }
  var result = movq(inputs: image)[0].as(of: FloatType.self)
  let encoder = Encoder(
    zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2, startHeight: 768,
    startWidth: 768, attnResolutions: Set([32]))
  encoder.compile(inputs: result)
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_movq_f16.ckpt") {
    $0.read("encoder", model: encoder)
  }
  let encodedImage = encoder(inputs: result)[0].as(of: FloatType.self)
  debugPrint(image)
  debugPrint(encodedImage)
  result = result.toCPU()
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/kandinsky_movq_f16.ckpt") {
    $0.write("movq", model: movq)
    $0.write("encoder", model: encoder)
  }
  */
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
  try! png.compress(path: "/home/liu/workspace/swift-diffusion/kandinsky2.png", level: 4)
}
