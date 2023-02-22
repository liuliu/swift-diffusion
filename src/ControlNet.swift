import NNC

func InputHintBlocks(modelChannel: Int, hint: Model.IO) -> Model.IO {
  let conv2d0 = Convolution(
    groups: 1, filters: 16, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d0(hint)
  out = Swish()(out)
  let conv2d1 = Convolution(
    groups: 1, filters: 16, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2d1(out)
  out = Swish()(out)
  let conv2d2 = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2d2(out)
  out = Swish()(out)
  let conv2d3 = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2d3(out)
  out = Swish()(out)
  let conv2d4 = Convolution(
    groups: 1, filters: 96, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2d4(out)
  out = Swish()(out)
  let conv2d5 = Convolution(
    groups: 1, filters: 96, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2d5(out)
  out = Swish()(out)
  let conv2d6 = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2d6(out)
  out = Swish()(out)
  let conv2d7 = Convolution(
    groups: 1, filters: modelChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2d7(out)
  return out
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, hint: Model.IO, emb: Model.IO, c: Model.IO
) -> ([(Model.IO, Int)], Model.IO) {
  let conv2d = Convolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x) + hint
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [(out, 320)]
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
      passLayers.append((out, channel))
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append((out, channel))
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  return (passLayers, out)
}

public func HintNet() -> Model {
  let hint = Input()
  let out = InputHintBlocks(modelChannel: 320, hint: hint)
  return Model([hint], [out])
}

public func ControlNet(batchSize: Int) -> Model {
  let x = Input()
  let hint = Input()
  let t_emb = Input()
  let c = Input()
  let timeEmbed = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  let (inputs, inputBlocks) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: 64,
    startWidth: 64, embeddingSize: 77, attentionRes: attentionRes, x: x, hint: hint, emb: emb, c: c)
  var out = inputBlocks
  let middleBlock = MiddleBlock(
    channels: 1280, numHeads: 8, batchSize: batchSize, height: 8, width: 8, embeddingSize: 77,
    x: out,
    emb: emb, c: c)
  out = middleBlock
  var zeroConvs = [Model]()
  var outputs = [Model.IO]()
  for i in 0..<inputs.count {
    let channel = inputs[i].1
    let zeroConv = Convolution(
        groups: 1, filters: channel, filterSize: [1, 1],
        hint: Hint(stride: [1, 1]))
    outputs.append(zeroConv(inputs[i].0))
    zeroConvs.append(zeroConv)
  }
  let middleBlockOut = Convolution(
        groups: 1, filters: 1280, filterSize: [1, 1],
        hint: Hint(stride: [1, 1]))
  outputs.append(middleBlockOut(out))
  return Model([x, hint, t_emb, c], outputs)
}
