import NNC

private func fiboVAEBindings(_ model: Model, _ prefix: String) -> [FiboBinding] {
  [
    .init(model: model, key: prefix + ".weight"),
    .init(model: model, key: prefix + ".bias", isBias: true),
  ]
}

// FLUX.2 VAE graph and store layout. Latent statistics stay outside these models.
func FiboVAEResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> (
  [FiboBinding], Model
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
  var reader =
    fiboVAEBindings(norm1, "\(prefix).norm1")
    + fiboVAEBindings(conv1, "\(prefix).conv1")
    + fiboVAEBindings(norm2, "\(prefix).norm2")
    + fiboVAEBindings(conv2, "\(prefix).conv2")
  if let ninShortcut { reader += fiboVAEBindings(ninShortcut, "\(prefix).nin_shortcut") }
  return (reader, Model([x], [out]))
}

func FiboVAEAttnBlock(prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int)
  -> (
    [FiboBinding], Model
  )
{
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
  let reader =
    fiboVAEBindings(norm, "\(prefix).norm")
    + fiboVAEBindings(tokeys, "\(prefix).k")
    + fiboVAEBindings(toqueries, "\(prefix).q")
    + fiboVAEBindings(tovalues, "\(prefix).v")
    + fiboVAEBindings(projOut, "\(prefix).proj_out")
  return (reader, Model([x], [out]))
}

func FiboVAEEncoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int
)
  -> ([FiboBinding], Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var readers = [FiboBinding]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (reader, block) = FiboVAEResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", outChannels: channel,
        shortcut: previousChannel != channel)
      readers += reader
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
      let reader = fiboVAEBindings(conv2d, "encoder.down.\(downLayer).downsample.conv")
      readers += reader
    }
  }
  let (midBlockReader1, midBlock1) = FiboVAEResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = FiboVAEAttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth, height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = FiboVAEResnetBlock(
    prefix: "encoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let quantConv = Convolution(
    groups: 1, filters: 64, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
  out = quantConv(out)
  let reader =
    fiboVAEBindings(convIn, "encoder.conv_in") + readers
    + midBlockReader1 + midAttnReader1 + midBlockReader2
    + fiboVAEBindings(normOut, "encoder.norm_out")
    + fiboVAEBindings(convOut, "encoder.conv_out")
    + fiboVAEBindings(quantConv, "encoder.quant_conv")
  return (reader, Model([x], [out]))
}

func FiboVAEDecoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int
)
  -> ([FiboBinding], Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 32, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])))
  var out = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convIn(out)
  let (midBlockReader1, midBlock1) = FiboVAEResnetBlock(
    prefix: "decoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = FiboVAEAttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = FiboVAEResnetBlock(
    prefix: "decoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  var readers = [FiboBinding]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = FiboVAEResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", outChannels: channel,
        shortcut: previousChannel != channel)
      readers += reader
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
      let upLayer = i
      let reader = fiboVAEBindings(conv2d, "decoder.up.\(upLayer).upsample.conv")
      readers += reader
    }
  }
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = Swish()(out)
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader =
    fiboVAEBindings(postQuantConv, "decoder.post_quant_conv")
    + fiboVAEBindings(convIn, "decoder.conv_in")
    + midBlockReader1 + midAttnReader1 + midBlockReader2 + readers
    + fiboVAEBindings(normOut, "decoder.norm_out")
    + fiboVAEBindings(convOut, "decoder.conv_out")
  return (reader, Model([x], [out]))
}
