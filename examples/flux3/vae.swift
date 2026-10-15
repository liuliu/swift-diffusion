import NNC

private func flux3VAEBindings(_ model: Model, _ prefix: String) -> [Flux3Binding] {
  [
    .init(model: model, key: prefix + ".weight"),
    .init(model: model, key: prefix + ".bias", isBias: true),
  ]
}

// FLUX.2 VAE graph and store layout. Latent statistics stay outside these models.
func Flux3VAEResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> (
  [Flux3Binding], Model
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
    flux3VAEBindings(norm1, "\(prefix).norm1")
    + flux3VAEBindings(conv1, "\(prefix).conv1")
    + flux3VAEBindings(norm2, "\(prefix).norm2")
    + flux3VAEBindings(conv2, "\(prefix).conv2")
  if let ninShortcut { reader += flux3VAEBindings(ninShortcut, "\(prefix).nin_shortcut") }
  return (reader, Model([x], [out]))
}

func Flux3VAEAttnBlock(prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int)
  -> (
    [Flux3Binding], Model
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
    flux3VAEBindings(norm, "\(prefix).norm")
    + flux3VAEBindings(tokeys, "\(prefix).k")
    + flux3VAEBindings(toqueries, "\(prefix).q")
    + flux3VAEBindings(tovalues, "\(prefix).v")
    + flux3VAEBindings(projOut, "\(prefix).proj_out")
  return (reader, Model([x], [out]))
}

func Flux3VAEEncoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int
)
  -> ([Flux3Binding], Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var readers = [Flux3Binding]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (reader, block) = Flux3VAEResnetBlock(
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
      let reader = flux3VAEBindings(conv2d, "encoder.down.\(downLayer).downsample.conv")
      readers += reader
    }
  }
  let (midBlockReader1, midBlock1) = Flux3VAEResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = Flux3VAEAttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth, height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = Flux3VAEResnetBlock(
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
    flux3VAEBindings(convIn, "encoder.conv_in") + readers
    + midBlockReader1 + midAttnReader1 + midBlockReader2
    + flux3VAEBindings(normOut, "encoder.norm_out")
    + flux3VAEBindings(convOut, "encoder.conv_out")
    + flux3VAEBindings(quantConv, "encoder.quant_conv")
  return (reader, Model([x], [out]))
}

func Flux3VAEDecoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int
)
  -> ([Flux3Binding], Model)
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
  let (midBlockReader1, midBlock1) = Flux3VAEResnetBlock(
    prefix: "decoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = Flux3VAEAttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize,
    width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = Flux3VAEResnetBlock(
    prefix: "decoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  var readers = [Flux3Binding]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = Flux3VAEResnetBlock(
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
      let reader = flux3VAEBindings(conv2d, "decoder.up.\(upLayer).upsample.conv")
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
    flux3VAEBindings(postQuantConv, "decoder.post_quant_conv")
    + flux3VAEBindings(convIn, "decoder.conv_in")
    + midBlockReader1 + midAttnReader1 + midBlockReader2 + readers
    + flux3VAEBindings(normOut, "decoder.norm_out")
    + flux3VAEBindings(convOut, "decoder.conv_out")
  return (reader, Model([x], [out]))
}
