import Diffusion
import Foundation
import NNC
import PNG

public enum PythonObject {}
public typealias FloatType = Float16

func ResBlock(prefix: String, batchSize: Int, channels: Int, skip: Bool) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let depthwise = Convolution(
    groups: channels, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), name: "resblock")
  var out = depthwise(x)
  let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = norm(out)
  let xSkip: Input?
  if skip {
    let xSkipIn = Input()
    out = Functional.concat(axis: 1, out, xSkipIn)
    xSkip = xSkipIn
  } else {
    xSkip = nil
  }
  let convIn = Convolution(
    groups: 1, filters: channels * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    name: "resblock")
  out = convIn(out).GELU()
  let Gx = out.reduced(.norm2, axis: [2, 3])
  let Nx = Gx .* (1 / Gx.reduced(.mean, axis: [1])) + 1e-6
  let gamma = Parameter<FloatType>(
    .GPU(0), .NCHW(1, channels * 4, 1, 1), initBound: 1, name: "resblock")
  let beta = Parameter<FloatType>(
    .GPU(0), .NCHW(1, channels * 4, 1, 1), initBound: 1, name: "resblock")
  out = gamma .* (out .* Nx) + beta + out
  let convOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out) + x
  let reader: (PythonObject) -> Void = { state_dict in
  }
  if let xSkip = xSkip {
    return (Model([x, xSkip], [out]), reader)
  } else {
    return (Model([x], [out]), reader)
  }
}

func TimestepBlock(
  prefix: String, batchSize: Int, timeEmbedSize: Int, channels: Int, tConds: [String]
) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rEmbed = Input()
  let mapper = Dense(count: channels * 2, name: "timestepblock")
  var gate = mapper(
    rEmbed.reshaped(
      [batchSize, timeEmbedSize], offset: [0, 0], strides: [timeEmbedSize * (tConds.count + 1), 1]))
  var otherMappers = [Model]()
  for i in 0..<tConds.count {
    let otherMapper = Dense(count: channels * 2, name: "timestepblock")
    gate =
      gate
      + otherMapper(
        rEmbed.reshaped(
          [batchSize, timeEmbedSize], offset: [0, timeEmbedSize * (i + 1)],
          strides: [timeEmbedSize * (tConds.count + 1), 1]))
    otherMappers.append(otherMapper)
  }
  var out: Model.IO = x
  out =
    out
    .* (1
      + gate.reshaped(
        [batchSize, channels, 1, 1], offset: [0, 0, 0, 0], strides: [channels * 2, 1, 1, 1]))
    + gate.reshaped(
      [batchSize, channels, 1, 1], offset: [0, channels, 0, 0], strides: [channels * 2, 1, 1, 1])
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([x, rEmbed], [out]), reader)
}

func MultiHeadAttention(prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let key = Input()
  let value = Input()
  let tokeys = Dense(count: k * h, name: "\(prefix).keys")
  let toqueries = Dense(count: k * h, name: "queries")
  let tovalues = Dense(count: k * h, name: "\(prefix).values")
  var keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2)
  keys = Functional.concat(axis: 2, keys, key)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2)
  var values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2)
  values = Functional.concat(axis: 2, values, value)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw + t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, hw + t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h, name: "unifyheads")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([x, key, value], [out]), reader)
}

func AttnBlock(
  prefix: String, batchSize: Int, channels: Int, nHead: Int, height: Int, width: Int, t: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let key = Input()
  let value = Input()
  let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  var out = norm(x).reshaped([batchSize, channels, height * width]).transposed(1, 2)
  let k = channels / nHead
  let (multiHeadAttention, multiHeadAttentionReader) = MultiHeadAttention(
    prefix: prefix, k: k, h: nHead, b: batchSize, hw: height * width, t: t)
  out =
    x
    + multiHeadAttention(out, key, value).transposed(1, 2).reshaped([
      batchSize, channels, height, width,
    ])
  return (Model([x, key, value], [out]), multiHeadAttentionReader)
}

func AttnBlockFixed(prefix: String, batchSize: Int, channels: Int, nHead: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let kv = Input()
  let kvMapper = Dense(count: channels, name: "kv_mapper")
  let kvOut = kvMapper(kv.swish())
  let tokeys = Dense(count: channels, name: "\(prefix).keys")
  let tovalues = Dense(count: channels, name: "\(prefix).values")
  let k = channels / nHead
  let keys = tokeys(kvOut).reshaped([batchSize, t, nHead, k]).transposed(1, 2)
  let values = tovalues(kvOut).reshaped([batchSize, t, nHead, k]).transposed(1, 2)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([kv], [keys, values]), reader)
}

func StageCFixed(batchSize: Int, t: Int) -> (Model, (PythonObject) -> Void) {
  let clipText = Input()
  let clipTextPooled = Input()
  let clipImg = Input()
  let clipTextMapper = Dense(count: 2048, name: "clip_text_mapper")
  let clipTextMapped = clipTextMapper(clipText)
  let clipTextPooledMapper = Dense(count: 2048 * 4, name: "clip_text_pool_mapper")
  let clipTextPooledMapped = clipTextPooledMapper(clipTextPooled).reshaped([batchSize, 4, 2048])
  let clipImgMapper = Dense(count: 2048 * 4, name: "clip_image_mapper")
  let clipImgMapped = clipImgMapper(clipImg).reshaped([batchSize, 4, 2048])
  let clipNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let clip = clipNorm(
    Functional.concat(axis: 1, clipTextMapped, clipTextPooledMapped, clipImgMapped))
  let blocks: [[Int]] = [[8, 24], [24, 8]]
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  for i in 0..<2 {
    for j in 0..<blocks[0][i] {
      let (attnBlockFixed, attnBlockFixedReader) = AttnBlockFixed(
        prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        t: t)
      readers.append(attnBlockFixedReader)
      let out = attnBlockFixed(clip)
      outs.append(out)
    }
  }
  for i in 0..<2 {
    for j in 0..<blocks[1][i] {
      let (attnBlockFixed, attnBlockFixedReader) = AttnBlockFixed(
        prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        t: t)
      readers.append(attnBlockFixedReader)
      let out = attnBlockFixed(clip)
      outs.append(out)
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([clipText, clipTextPooled, clipImg], outs), reader)
}

func StageC(batchSize: Int, height: Int, width: Int, t: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rEmbed = Input()
  let conv2d = Convolution(
    groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = conv2d(x)
  let normIn = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normIn(out)

  let blocks: [[Int]] = [[8, 24], [24, 8]]
  var readers: [(PythonObject) -> Void] = []
  var levelOutputs = [Model.IO]()
  var kvs = [Input]()
  for i in 0..<2 {
    if i > 0 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
      out = norm(out)
      let downscaler = Convolution(
        groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
      out = downscaler(out)
    }
    for j in 0..<blocks[0][i] {
      let (resBlock, resBlockReader) = ResBlock(
        prefix: "down_blocks.\(i).\(j * 3)", batchSize: batchSize, channels: 2048, skip: false)
      readers.append(resBlockReader)
      out = resBlock(out)
      let (timestepBlock, timestepBlockReader) = TimestepBlock(
        prefix: "down_blocks.\(i).\(j * 3 + 1)", batchSize: batchSize, timeEmbedSize: 64,
        channels: 2048, tConds: ["sca", "crp"])
      readers.append(timestepBlockReader)
      out = timestepBlock(out, rEmbed)
      let (attnBlock, attnBlockReader) = AttnBlock(
        prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        height: height, width: width, t: t)
      readers.append(attnBlockReader)
      let key = Input()
      let value = Input()
      out = attnBlock(out, key, value)
      kvs.append(key)
      kvs.append(value)
    }
    if i < 2 - 1 {
      levelOutputs.append(out)
    }
  }

  var skip: Model.IO? = nil
  for i in 0..<2 {
    for j in 0..<blocks[1][i] {
      let (resBlock, resBlockReader) = ResBlock(
        prefix: "up_blocks.\(i).\(j * 3)", batchSize: batchSize, channels: 2048, skip: skip != nil)
      readers.append(resBlockReader)
      if let skip = skip {
        out = resBlock(out, skip)
      } else {
        out = resBlock(out)
      }
      skip = nil
      let (timestepBlock, timestepBlockReader) = TimestepBlock(
        prefix: "up_blocks.\(i).\(j * 3 + 1)", batchSize: batchSize, timeEmbedSize: 64,
        channels: 2048, tConds: ["sca", "crp"])
      readers.append(timestepBlockReader)
      out = timestepBlock(out, rEmbed)
      let (attnBlock, attnBlockReader) = AttnBlock(
        prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: 2048, nHead: 32,
        height: height, width: width, t: t)
      readers.append(attnBlockReader)
      let key = Input()
      let value = Input()
      out = attnBlock(out, key, value)
      kvs.append(key)
      kvs.append(value)
    }
    if i < 2 - 1 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
      out = norm(out)
      let upscaler = Convolution(
        groups: 1, filters: 2048, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
      out = upscaler(out)
      skip = levelOutputs.removeLast()
    }
  }

  let normOut = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normOut(out)
  let convOut = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out)

  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }

  }
  return (Model([x, rEmbed] + kvs, [out]), reader)
}

func SpatialMapper(prefix: String, cHidden: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: cHidden * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = convIn(x)
  let convOut = Convolution(
    groups: 1, filters: cHidden, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out.GELU())
  let normOut = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([x], [out]), reader)
}

func StageBFixed(batchSize: Int, height: Int, width: Int, effnetHeight: Int, effnetWidth: Int) -> (
  Model, (PythonObject) -> Void
) {
  let effnet = Input()
  let pixels = Input()
  let clip = Input()
  let cHidden: [Int] = [320, 640, 1280, 1280]
  let (effnetMapper, effnetMapperReader) = SpatialMapper(
    prefix: "effnet_mapper", cHidden: cHidden[0])
  var out = effnetMapper(
    Upsample(
      .bilinear, widthScale: Float(width / 2) / Float(effnetWidth),
      heightScale: Float(height / 2) / Float(effnetHeight), alignCorners: true)(effnet))
  let (pixelsMapper, pixelsMapperReader) = SpatialMapper(
    prefix: "pixels_mapper", cHidden: cHidden[0])
  out =
    out
    + Upsample(
      .bilinear, widthScale: Float(width / 2) / Float(8), heightScale: Float(height / 2) / Float(8),
      alignCorners: true)(pixelsMapper(pixels))
  var outs = [out]
  let clipMapper = Dense(count: 1280 * 4)
  let clipMapped = clipMapper(clip).reshaped([batchSize, 4, 1280])
  let clipNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let clipNormed = clipNorm(clipMapped)
  let blocks: [[Int]] = [[2, 6, 28, 6], [6, 28, 6, 2]]
  var readers: [(PythonObject) -> Void] = []
  let attentions: [[Bool]] = [[false, false, true, true], [true, true, false, false]]
  for i in 0..<4 {
    let attention = attentions[0][i]
    for j in 0..<blocks[0][i] {
      if attention {
        let (attnBlockFixed, attnBlockFixedReader) = AttnBlockFixed(
          prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[i],
          nHead: 20, t: 4)
        readers.append(attnBlockFixedReader)
        let out = attnBlockFixed(clipNormed)
        outs.append(out)
      }
    }
  }
  for i in 0..<4 {
    let attention = attentions[1][i]
    for j in 0..<blocks[1][i] {
      if attention {
        let (attnBlockFixed, attnBlockFixedReader) = AttnBlockFixed(
          prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[3 - i],
          nHead: 20, t: 4)
        readers.append(attnBlockFixedReader)
        let out = attnBlockFixed(clipNormed)
        outs.append(out)
      }
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    effnetMapperReader(state_dict)
    pixelsMapperReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([effnet, pixels, clip], outs), reader)
}

func StageB(batchSize: Int, cIn: Int, height: Int, width: Int, effnetHeight: Int, effnetWidth: Int)
  -> (Model, (PythonObject) -> Void)
{
  let x = Input()
  let rEmbed = Input()
  let effnetAndPixels = Input()
  let cHidden: [Int] = [320, 640, 1280, 1280]
  let conv2d = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = conv2d(x)
  let normIn = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normIn(out) + effnetAndPixels
  let blocks: [[Int]] = [[2, 6, 28, 6], [6, 28, 6, 2]]
  var readers: [(PythonObject) -> Void] = []
  let attentions: [[Bool]] = [[false, false, true, true], [true, true, false, false]]
  var levelOutputs = [Model.IO]()
  var height = height / 2
  var width = width / 2
  var kvs = [Input]()
  for i in 0..<4 {
    if i > 0 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
      out = norm(out)
      let downscaler = Convolution(
        groups: 1, filters: cHidden[i], filterSize: [2, 2], hint: Hint(stride: [2, 2]))
      out = downscaler(out)
      height = height / 2
      width = width / 2
    }
    let attention = attentions[0][i]
    for j in 0..<blocks[0][i] {
      let (resBlock, resBlockReader) = ResBlock(
        prefix: "down_blocks.\(i).\(j * (attention ? 3 : 2))", batchSize: batchSize,
        channels: cHidden[i], skip: false)
      readers.append(resBlockReader)
      out = resBlock(out)
      let (timestepBlock, timestepBlockReader) = TimestepBlock(
        prefix: "down_blocks.\(i).\(j * (attention ? 3 : 2) + 1)", batchSize: batchSize,
        timeEmbedSize: 64,
        channels: cHidden[i], tConds: ["sca"])
      readers.append(timestepBlockReader)
      out = timestepBlock(out, rEmbed)
      if attention {
        let (attnBlock, attnBlockReader) = AttnBlock(
          prefix: "down_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[i],
          nHead: 20, height: height, width: width, t: 4)
        readers.append(attnBlockReader)
        let key = Input()
        let value = Input()
        out = attnBlock(out, key, value)
        kvs.append(key)
        kvs.append(value)
      }
    }
    if i < 4 - 1 {
      levelOutputs.append(out)
    }
  }
  var skip: Model.IO? = nil
  let blockRepeat: [Int] = [3, 3, 2, 2]
  for i in 0..<4 {
    let cSkip = skip
    skip = nil
    let attention = attentions[1][i]
    var resBlocks = [Model]()
    var timestepBlocks = [Model]()
    var attnBlocks = [Model]()
    var keyAndValue = [(Input, Input)]()
    for j in 0..<blocks[1][i] {
      let (resBlock, resBlockReader) = ResBlock(
        prefix: "up_blocks.\(i).\(j * (attention ? 3 : 2))", batchSize: batchSize,
        channels: cHidden[3 - i], skip: j == 0 && cSkip != nil)
      readers.append(resBlockReader)
      resBlocks.append(resBlock)
      let (timestepBlock, timestepBlockReader) = TimestepBlock(
        prefix: "up_blocks.\(i).\(j * (attention ? 3 : 2) + 1)", batchSize: batchSize,
        timeEmbedSize: 64,
        channels: cHidden[3 - i], tConds: ["sca"])
      readers.append(timestepBlockReader)
      timestepBlocks.append(timestepBlock)
      if attention {
        let (attnBlock, attnBlockReader) = AttnBlock(
          prefix: "up_blocks.\(i).\(j * 3 + 2)", batchSize: batchSize, channels: cHidden[3 - i],
          nHead: 20,
          height: height, width: width, t: 4)
        readers.append(attnBlockReader)
        attnBlocks.append(attnBlock)
        keyAndValue.append((Input(), Input()))
      }
    }
    kvs.append(contentsOf: keyAndValue.flatMap { [$0.0, $0.1] })
    for j in 0..<blockRepeat[i] {
      for k in 0..<blocks[1][i] {
        if k == 0, let cSkip = cSkip {
          out = resBlocks[k](out, cSkip)
        } else {
          out = resBlocks[k](out)
        }
        out = timestepBlocks[k](out, rEmbed)
        if attention {
          out = attnBlocks[k](out, keyAndValue[k].0, keyAndValue[k].1)
        }
      }
      // repmap.
      if j < blockRepeat[i] - 1 {
        let repmap = Convolution(
          groups: 1, filters: cHidden[3 - i], filterSize: [1, 1], hint: Hint(stride: [1, 1]))
        out = repmap(out)
      }
    }
    if i < 4 - 1 {
      let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
      out = norm(out)
      let upscaler = ConvolutionTranspose(
        groups: 1, filters: cHidden[2 - i], filterSize: [2, 2], hint: Hint(stride: [2, 2]))
      out = upscaler(out)
      skip = levelOutputs.removeLast()
      height = height * 2
      width = width * 2
    }
  }
  let normOut = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  out = normOut(out)
  let convOut = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out).reshaped([batchSize, 4, 2, 2, height, width]).transposed(3, 4).transposed(4, 5)
    .transposed(2, 3).reshaped([batchSize, 4, height * 2, width * 2])  // This is the same as .permuted(0, 1, 4, 2, 5, 3).

  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x, rEmbed, effnetAndPixels] + kvs, [out]), reader)
}

func StageAResBlock(prefix: String, channels: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let gammas = Parameter<FloatType>(.GPU(0), .NCHW(1, 1, 1, 6), initBound: 1)
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  var out =
    norm1(x) .* (1 + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 0], strides: [6, 6, 6, 1]))
    + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 1], strides: [6, 6, 6, 1])
  let depthwise = Convolution(
    groups: channels, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1]))
  out = x + depthwise(out.padded(.replication, begin: [0, 0, 1, 1], end: [0, 0, 1, 1]))
    .* gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 2], strides: [6, 6, 6, 1])
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let xTemp =
    norm2(out)
    .* (1 + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 3], strides: [6, 6, 6, 1]))
    + gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 4], strides: [6, 6, 6, 1])
  let convIn = Convolution(
    groups: 1, filters: channels * 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let convOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = out + convOut(convIn(xTemp).GELU())
    .* gammas.reshaped([1, 1, 1, 1], offset: [0, 0, 0, 5], strides: [6, 6, 6, 1])
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([x], [out]), reader)
}

func StageAEncoder(batchSize: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let cHidden = [192, 384]
  let convIn = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out = convIn(x)
  var readers = [(PythonObject) -> Void]()
  var j = 0
  for i in 0..<cHidden.count {
    if i > 0 {
      let conv2d = Convolution(
        groups: 1, filters: cHidden[i], filterSize: [4, 4],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
      j += 1
    }
    let (resBlock, resBlockReader) = StageAResBlock(
      prefix: "down_blocks.\(j)", channels: cHidden[i])
    out = resBlock(out)
    readers.append(resBlockReader)
    j += 1
  }
  let conv2d = Convolution(groups: 1, filters: 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = conv2d(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x], [out]), reader)
}

func StageADecoder(batchSize: Int, height: Int, width: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let cHidden = [384, 192]
  let convIn = Convolution(
    groups: 1, filters: cHidden[0], filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = convIn(x)
  var readers = [(PythonObject) -> Void]()
  var j = 1
  for i in 0..<cHidden.count {
    for _ in 0..<(i == 0 ? 12 : 1) {
      let (resBlock, resBlockReader) = StageAResBlock(
        prefix: "up_blocks.\(j)", channels: cHidden[i])
      out = resBlock(out)
      readers.append(resBlockReader)
      j += 1
    }
    if i < cHidden.count - 1 {
      let conv2d = ConvolutionTranspose(
        groups: 1, filters: cHidden[i + 1], filterSize: [4, 4],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
      j += 1
    }
  }
  let convOut = Convolution(
    groups: 1, filters: 12, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out).reshaped([batchSize, 3, 2, 2, height, width]).transposed(3, 4).transposed(4, 5)
    .transposed(2, 3).reshaped([batchSize, 3, height * 2, width * 2])  // This is the same as .permuted(0, 1, 4, 2, 5, 3).
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x], [out]), reader)
}

func rEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let r = timesteps * Float(maxPeriod)
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half - 1)) * r
    let sinFreq = sin(freq)
    let cosFreq = cos(freq)
    for j in 0..<batchSize {
      embedding[j, i] = sinFreq
      embedding[j, i + half] = cosFreq
    }
  }
  return embedding
}

func FusedMBConv(
  prefix: String, outChannels: Int, stride: Int, filterSize: Int, skip: Bool,
  expandChannels: Int? = nil
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  var out: Model.IO = x
  let convOut: Model
  if let expandChannels = expandChannels {
    let conv = Convolution(
      groups: 1, filters: expandChannels, filterSize: [filterSize, filterSize],
      hint: Hint(
        stride: [stride, stride],
        border: Hint.Border(
          begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
          end: [(filterSize - 1) / 2, (filterSize - 1) / 2])))
    out = conv(out).swish()
    convOut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = convOut(out)
  } else {
    convOut = Convolution(
      groups: 1, filters: outChannels, filterSize: [filterSize, filterSize],
      hint: Hint(
        stride: [stride, stride],
        border: Hint.Border(
          begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
          end: [(filterSize - 1) / 2, (filterSize - 1) / 2])))
    out = convOut(out).swish()
  }
  if skip {
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([x], [out]), reader)
}

func MBConv(
  prefix: String, stride: Int, filterSize: Int, inChannels: Int, expandChannels: Int,
  outChannels: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  var out: Model.IO = x
  if expandChannels != inChannels {
    let conv = Convolution(
      groups: 1, filters: expandChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = conv(out).swish()
  }

  let depthwise = Convolution(
    groups: expandChannels, filters: expandChannels, filterSize: [filterSize, filterSize],
    hint: Hint(
      stride: [stride, stride],
      border: Hint.Border(
        begin: [(filterSize - 1) / 2, (filterSize - 1) / 2],
        end: [(filterSize - 1) / 2, (filterSize - 1) / 2])))
  out = depthwise(out).swish()

  // Squeeze and Excitation
  var scale = out.reduced(.mean, axis: [2, 3])
  let fc1 = Convolution(
    groups: 1, filters: inChannels / 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  scale = fc1(scale).swish()
  let fc2 = Convolution(
    groups: 1, filters: expandChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  scale = fc2(scale).sigmoid()
  out = scale .* out

  let convOut = Convolution(
    groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out)

  if inChannels == outChannels && stride == 1 {
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (Model([x], [out]), reader)
}

func EfficientNetEncoder() -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let conv = Convolution(
    groups: 1, filters: 24, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv(x).swish()
  var readers = [(PythonObject) -> Void]()
  // 1.
  let (backbone_1_0, backbone_1_0_reader) = FusedMBConv(
    prefix: "backbone.1.0", outChannels: 24, stride: 1, filterSize: 3, skip: true)
  out = backbone_1_0(out)
  readers.append(backbone_1_0_reader)
  let (backbone_1_1, backbone_1_1_reader) = FusedMBConv(
    prefix: "backbone.1.1", outChannels: 24, stride: 1, filterSize: 3, skip: true)
  out = backbone_1_1(out)
  readers.append(backbone_1_1_reader)
  // 2.
  let (backbone_2_0, backbone_2_0_reader) = FusedMBConv(
    prefix: "backbone.2.0", outChannels: 48, stride: 2, filterSize: 3, skip: false,
    expandChannels: 96)
  out = backbone_2_0(out)
  readers.append(backbone_2_0_reader)
  for i in 1..<4 {
    let (backbone_2_x, backbone_2_x_reader) = FusedMBConv(
      prefix: "backbone.2.\(i)", outChannels: 48, stride: 1, filterSize: 3, skip: true,
      expandChannels: 192)
    out = backbone_2_x(out)
    readers.append(backbone_2_x_reader)
  }
  // 3.
  let (backbone_3_0, backbone_3_0_reader) = FusedMBConv(
    prefix: "backbone.3.0", outChannels: 64, stride: 2, filterSize: 3, skip: false,
    expandChannels: 192)
  out = backbone_3_0(out)
  readers.append(backbone_3_0_reader)
  for i in 1..<4 {
    let (backbone_3_x, backbone_3_x_reader) = FusedMBConv(
      prefix: "backbone.3.\(i)", outChannels: 64, stride: 1, filterSize: 3, skip: true,
      expandChannels: 256)
    out = backbone_3_x(out)
    readers.append(backbone_3_x_reader)
  }
  // 4.
  let (backbone_4_0, backbone_4_0_reader) = MBConv(
    prefix: "backbone.4.0", stride: 2, filterSize: 3, inChannels: 64, expandChannels: 256,
    outChannels: 128)
  out = backbone_4_0(out)
  readers.append(backbone_4_0_reader)
  for i in 1..<6 {
    let (backbone_4_x, backbone_4_x_reader) = MBConv(
      prefix: "backbone.4.\(i)", stride: 1, filterSize: 3, inChannels: 128, expandChannels: 512,
      outChannels: 128)
    out = backbone_4_x(out)
    readers.append(backbone_4_x_reader)
  }
  // 5.
  let (backbone_5_0, backbone_5_0_reader) = MBConv(
    prefix: "backbone.5.0", stride: 1, filterSize: 3, inChannels: 128, expandChannels: 768,
    outChannels: 160)
  out = backbone_5_0(out)
  readers.append(backbone_5_0_reader)
  for i in 1..<9 {
    let (backbone_5_x, backbone_5_x_reader) = MBConv(
      prefix: "backbone.5.\(i)", stride: 1, filterSize: 3, inChannels: 160, expandChannels: 960,
      outChannels: 160)
    out = backbone_5_x(out)
    readers.append(backbone_5_x_reader)
  }
  // 6.
  let (backbone_6_0, backbone_6_0_reader) = MBConv(
    prefix: "backbone.6.0", stride: 2, filterSize: 3, inChannels: 160, expandChannels: 960,
    outChannels: 256)
  out = backbone_6_0(out)
  readers.append(backbone_6_0_reader)
  for i in 1..<15 {
    let (backbone_6_x, backbone_6_x_reader) = MBConv(
      prefix: "backbone.6.\(i)", stride: 1, filterSize: 3, inChannels: 256, expandChannels: 1536,
      outChannels: 256)
    out = backbone_6_x(out)
    readers.append(backbone_6_x_reader)
  }
  let convOut = Convolution(
    groups: 1, filters: 1280, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = convOut(out).swish()
  let mapper = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = mapper(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x], [out]), reader)
}

func OpenCLIPTextModel(
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
  var penultimate: Model.IO = out
  for i in 0..<numLayers {
    if i == numLayers - 1 {
      penultimate = out
    }
    let encoderLayer =
      OpenCLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let hiddenState = out
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [penultimate, out, hiddenState])
}

let tokenizer = CLIPTokenizer(
  vocabulary: "examples/open_clip/vocab_16e6.json",
  merges: "examples/open_clip/bpe_simple_vocab_16e6.txt")

let prompt =
  "cinematic photo of an anthropomorphic polar bear sitting in a cafe reading a book and having a coffee"
let negativePrompt = ""

let tokens = tokenizer.tokenize(text: prompt, truncation: true, maxLength: 77)
let unconditionalTokens = tokenizer.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77)

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

let tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
for i in 0..<77 {
  tokensTensor[i] = unconditionalTokens[i]
  tokensTensor[i + 77] = tokens[i]
  positionTensor[i] = Int32(i)
  positionTensor[i + 77] = Int32(i)
}

let casualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(2, 1, 77, 77)))
casualAttentionMask.full(0)
var lastToken: Int? = nil
var lastUnconditionalToken: Int? = nil
for i in 0..<77 {
  if i < 76 {
    for j in (i + 1)..<77 {
      casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
      casualAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  if let lastUnconditionalToken = lastUnconditionalToken {
    for j in (lastUnconditionalToken + 1)..<(i + 1) {
      casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  if lastUnconditionalToken == nil, unconditionalTokens[i] == tokenizer.endToken {
    lastUnconditionalToken = i
  }
  if let lastToken = lastToken {
    for j in (lastToken + 1)..<(i + 1) {
      casualAttentionMask[1, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  if lastToken == nil, tokens[i] == tokenizer.endToken {
    lastToken = i
  }
}

let (c, pooled) = graph.withNoGrad {
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  let textModel = OpenCLIPTextModel(
    vocabularySize: 49408, maxLength: 77, embeddingSize: 1280, numLayers: 32, numHeads: 20,
    batchSize: 2, intermediateSize: 5120)
  let textProjection = graph.variable(.GPU(0), .NC(1280, 1280), of: FloatType.self)
  textModel.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_vit_bigg14_f16.ckpt") {
    $0.read("text_model", model: textModel)
    $0.read("text_projection", variable: textProjection)
  }
  let c = textModel(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU).map {
    $0.as(of: FloatType.self)
  }
  var pooled = graph.variable(.GPU(0), .NC(2, 1280), of: FloatType.self)
  let c0 = c[2].reshaped(.CHW(2, 77, 1280))
  for (i, token) in tokens.enumerated() {
    if token == tokenizer.endToken {
      pooled[1..<2, 0..<1280] = c[1][(77 + i)..<(77 + i + 1), 0..<1280] * textProjection
      break
    }
  }
  for (i, token) in unconditionalTokens.enumerated() {
    if token == tokenizer.endToken {
      pooled[0..<1, 0..<1280] = c[1][i..<(i + 1), 0..<1280] * textProjection
      break
    }
  }
  return (c0, pooled)
}

DynamicGraph.setSeed(42)

struct CosineSchedule {
  private let s: Double
  private let range: ClosedRange<Double>
  private let minVar: Double
  init(s: Double = 0.008, range: ClosedRange<Double> = 0.0001...0.9999) {
    self.s = s
    self.range = range
    let minStd = cos(s / (1 + s) * Double.pi * 0.5)
    self.minVar = minStd * minStd
  }
  public func schedule(steps: Int, shift: Double = 1.0) -> [Double] {
    var alphasCumprod = [Double]()
    for i in 0..<steps {
      let t = Double(steps - i) / Double(steps)
      let std = min(max(cos((s + t) / (1 + s) * Double.pi * 0.5), 0), 1)
      let v = min(max((std * std) / minVar, range.lowerBound), range.upperBound)
      if shift != 1 {
        // Simplify Sigmoid[Log[x / (1 - x)] + 2 * Log[1 / shift]]
        let shiftedV = 1.0 / (1.0 + shift * shift * (1.0 - v) / v)
        alphasCumprod.append(min(max(shiftedV, range.lowerBound), range.upperBound))
      } else {
        alphasCumprod.append(v)
      }
    }
    return alphasCumprod
  }
  public func noise(alphaCumprod: Double) -> Double {
    let t = acos((alphaCumprod * minVar).squareRoot()) / (Double.pi * 0.5) * (1 + s) - s
    return t
  }
}

let schedule = CosineSchedule()
let stageCSteps = 20
var stageCAlphasCumprod = schedule.schedule(steps: stageCSteps, shift: 2)
stageCAlphasCumprod.append(1.0)
let stageBSteps = 10
var stageBAlphasCumprod = schedule.schedule(steps: stageBSteps)
stageBAlphasCumprod.append(1.0)

graph.withNoGrad {
  var x = graph.variable(.GPU(0), .NCHW(1, 16, 24, 24), of: FloatType.self)
  x.randn(std: 1, mean: 0)
  let rZeros = rEmbedding(timesteps: 0, batchSize: 2, embeddingSize: 64, maxPeriod: 10_000)
  var rEmbed = Tensor<Float>(.CPU, .NC(2, 192))
  rEmbed[0..<2, 64..<128] = rZeros
  rEmbed[0..<2, 128..<192] = rZeros
  var input = graph.variable(.GPU(0), .NCHW(2, 16, 24, 24), of: FloatType.self)
  let clipText = c
  let clipTextPooled = pooled.reshaped(.CHW(2, 1, 1280))
  let clipImg = graph.variable(.GPU(0), .CHW(2, 1, 1280), of: FloatType.self)
  clipImg.full(0)
  let (stageCFixed, _) = StageCFixed(batchSize: 2, t: 77 + 8)
  stageCFixed.compile(inputs: clipText, clipTextPooled, clipImg)
  graph.openStore("/home/liu/workspace/swift-diffusion/wurstchen_3.0_stage_c_f16_f32.ckpt") {
    $0.read("stage_c_fixed", model: stageCFixed)
  }
  let stageCKvs = stageCFixed(inputs: clipText, clipTextPooled, clipImg).map {
    $0.as(of: FloatType.self)
  }
  let (stageC, _) = StageC(batchSize: 2, height: 24, width: 24, t: 77 + 8)
  var rEmbedVariable = graph.variable(Tensor<FloatType>(from: rEmbed)).toGPU(0)
  stageC.compile(inputs: [input, rEmbedVariable] + stageCKvs)
  graph.openStore("/home/liu/workspace/swift-diffusion/wurstchen_3.0_stage_c_f16_f32.ckpt") {
    $0.read("stage_c", model: stageC)
  }
  for i in 0..<stageCSteps {
    let rTimeEmbed = rEmbedding(
      timesteps: Float(schedule.noise(alphaCumprod: stageCAlphasCumprod[i])), batchSize: 2,
      embeddingSize: 64, maxPeriod: 10_000)
    rEmbed[0..<2, 0..<64] = rTimeEmbed
    let rEmbedVariable = graph.variable(Tensor<FloatType>(from: rEmbed)).toGPU(0)
    input[0..<1, 0..<16, 0..<24, 0..<24] = x
    input[1..<2, 0..<16, 0..<24, 0..<24] = x
    let out = stageC(inputs: x, [rEmbedVariable] + stageCKvs)[0].as(
      of: FloatType.self)
    let etUncond = out[0..<1, 0..<16, 0..<24, 0..<24]
    let etCond = out[1..<2, 0..<16, 0..<24, 0..<24]
    let et = etUncond + 3.0 * (etCond - etUncond)
    let a = Float(stageCAlphasCumprod[i].squareRoot())
    let b = Float((1 - stageCAlphasCumprod[i]).squareRoot())
    let a_prev = Float(stageCAlphasCumprod[i + 1].squareRoot())
    let b_prev = Float((1 - stageCAlphasCumprod[i + 1]).squareRoot())
    x = (a_prev / a) * x + (b_prev - a_prev * b / a) * et
  }

  rEmbed = Tensor<Float>(.CPU, .NC(2, 128))
  rEmbed[0..<2, 64..<128] = rZeros
  var effnet = graph.variable(.GPU(0), .NCHW(2, 16, 24, 24), of: FloatType.self)
  effnet.full(0)
  effnet[1..<2, 0..<16, 0..<24, 0..<24] = x
  x = graph.variable(.GPU(0), .NCHW(1, 4, 256, 256), of: FloatType.self)
  x.randn(std: 1, mean: 0)
  let (stageBFixed, _) = StageBFixed(
    batchSize: 2, height: 256, width: 256, effnetHeight: 24, effnetWidth: 24)
  let pixels = graph.variable(.GPU(0), .NCHW(2, 3, 8, 8), of: FloatType.self)
  pixels.full(0)
  stageBFixed.compile(inputs: effnet, pixels, clipTextPooled)
  graph.openStore("/home/liu/workspace/swift-diffusion/wurstchen_3.0_stage_b_f32.ckpt") {
    $0.read("stage_b_fixed", model: stageBFixed)
  }
  let stageBKvs = stageBFixed(inputs: effnet, pixels, clipTextPooled).map {
    $0.as(of: FloatType.self)
  }
  let (stageB, _) = StageB(
    batchSize: 2, cIn: 4, height: 256, width: 256, effnetHeight: 24, effnetWidth: 24)
  rEmbedVariable = graph.variable(Tensor<FloatType>(from: rEmbed)).toGPU(0)
  input = graph.variable(.GPU(0), .NCHW(2, 4, 256, 256), of: FloatType.self)
  stageB.compile(inputs: [input, rEmbedVariable] + stageBKvs)
  graph.openStore("/home/liu/workspace/swift-diffusion/wurstchen_3.0_stage_b_f32.ckpt") {
    $0.read("stage_b", model: stageB)
  }
  for i in 0..<stageBSteps {
    let rTimeEmbed = rEmbedding(
      timesteps: Float(schedule.noise(alphaCumprod: stageBAlphasCumprod[i])), batchSize: 2,
      embeddingSize: 64, maxPeriod: 10_000)
    rEmbed[0..<2, 0..<64] = rTimeEmbed
    let rEmbedVariable = graph.variable(Tensor<FloatType>(from: rEmbed)).toGPU(0)
    input[0..<1, 0..<4, 0..<256, 0..<256] = x
    input[1..<2, 0..<4, 0..<256, 0..<256] = x
    let out = stageB(inputs: input, [rEmbedVariable] + stageBKvs)[0].as(
      of: FloatType.self)
    let etUncond = out[0..<1, 0..<4, 0..<256, 0..<256]
    let etCond = out[1..<2, 0..<4, 0..<256, 0..<256]
    let et = etUncond + 1.1 * (etCond - etUncond)
    let a = Float(stageBAlphasCumprod[i].squareRoot())
    let b = Float((1.0 - stageBAlphasCumprod[i]).squareRoot())
    let a_prev = Float(stageBAlphasCumprod[i + 1].squareRoot())
    let b_prev = Float((1.0 - stageBAlphasCumprod[i + 1]).squareRoot())
    x = (a_prev / a) * x + (b_prev - a_prev * b / a) * et
  }

  let (stageADecoder, _) = StageADecoder(batchSize: 1, height: 512, width: 512)
  x = 0.43 * x
  stageADecoder.compile(inputs: x)
  graph.openStore("/home/liu/workspace/swift-diffusion/wurstchen_3.0_stage_a_f32.ckpt") {
    $0.read("decoder", model: stageADecoder)
  }
  let img = stageADecoder(inputs: x)[0].as(of: FloatType.self).toCPU()
  debugPrint(img)
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: 1024 * 1024)
  for y in 0..<1024 {
    for x in 0..<1024 {
      let (r, g, b) = (img[0, 0, y, x], img[0, 1, y, x], img[0, 2, y, x])
      rgba[y * 1024 + x].r = UInt8(
        min(max(Int(Float(r) * 255), 0), 255))
      rgba[y * 1024 + x].g = UInt8(
        min(max(Int(Float(g) * 255), 0), 255))
      rgba[y * 1024 + x].b = UInt8(
        min(max(Int(Float(b) * 255), 0), 255))
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (1024, 1024),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/wurstchen.png", level: 4)
}
