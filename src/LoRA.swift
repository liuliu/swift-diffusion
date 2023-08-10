import NNC

public let LowRank = 16

public func LoRAConvolution(
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
  let out = conv2d(x) + conv2dUp(conv2dDown(x))  // .to(.Float32))).to(of: x)
  return Model([x], [out])
}

public func LoRADense(count: Int, noBias: Bool = false, name: String = "") -> Model {
  let x = Input()
  let dense = Dense(count: count, noBias: noBias, name: name)
  let denseDown = Dense(count: LowRank, noBias: true, trainable: true, name: "lora_down")
  let denseUp = Dense(count: count, noBias: true, trainable: true, name: "lora_up")
  let out = dense(x) + denseUp(denseDown(x))  // .to(.Float32))).to(of: x)
  return Model([x], [out])
}

/// Text Model

public func LoRACLIPAttention(k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = LoRADense(count: k * h)
  let toqueries = LoRADense(count: k * h)
  let tovalues = LoRADense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .transposed(1, 2)
  let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
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

private func LoRACLIPMLP(hiddenSize: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let fc1 = LoRADense(count: intermediateSize)
  var out = fc1(x)
  out = QuickGELU()(out)
  let fc2 = LoRADense(count: hiddenSize)
  out = fc2(out)
  return Model([x], [out])
}

private func LoRACLIPEncoderLayer(k: Int, h: Int, b: Int, t: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let attention = LoRACLIPAttention(k: k, h: h, b: b, t: t)
  out = attention(out, casualAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let mlp = LoRACLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return Model([x, casualAttentionMask], [out])
}

public func LoRACLIPTextModel<T: TensorNumeric>(
  _ dataType: T.Type,
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int, noFinalLayerNorm: Bool = false
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
    let encoderLayer = LoRACLIPEncoderLayer(
      k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  if !noFinalLayerNorm {
    let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
    out = finalLayerNorm(out)
  }
  return Model([tokens, positions, casualAttentionMask], [out], trainable: false)
}

/// UNet

public func LoRATimeEmbed(modelChannels: Int) -> Model {
  let x = Input()
  let fc0 = LoRADense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: modelChannels * 4)
  out = fc2(out)
  return Model([x], [out])
}

public func LoRAResBlock(b: Int, outChannels: Int, skipConnection: Bool) -> Model {
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

public func LoRASelfAttention(k: Int, h: Int, b: Int, hw: Int) -> Model {
  let x = Input()
  let tokeys = LoRADense(count: k * h, noBias: true)
  let toqueries = LoRADense(count: k * h, noBias: true)
  let tovalues = LoRADense(count: k * h, noBias: true)
  let keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2)
  let values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2)
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

private func LoRACrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
  let x = Input()
  let c = Input()
  let tokeys = LoRADense(count: k * h, noBias: true)
  let toqueries = LoRADense(count: k * h, noBias: true)
  let tovalues = LoRADense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
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

public func LoRAFeedForward(hiddenSize: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let fc10 = LoRADense(count: intermediateSize)
  let fc11 = LoRADense(count: intermediateSize)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = LoRADense(count: hiddenSize)
  out = fc2(out)
  return Model([x], [out])
}

private func LoRABasicTransformerBlock(
  k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let c = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let attn1 = LoRASelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let attn2 = LoRACrossAttention(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, c) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let ff = LoRAFeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  return Model([x, c], [out])
}

private func LoRASpatialTransformer(
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, t: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let c = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = LoRAConvolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).transposed(1, 2)
  let block = LoRABasicTransformerBlock(
    k: k, h: h, b: b, hw: hw, t: t, intermediateSize: intermediateSize)
  out = block(out, c).transposed(1, 2).reshaped([b, k * h, height, width])
  let projOut = LoRAConvolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  return Model([x, c], [out])
}

private func LoRABlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Bool, channels: Int, numHeads: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let emb = Input()
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let resBlock = LoRAResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  if attentionBlock {
    let c = Input()
    let transformer = LoRASpatialTransformer(
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer(out, c)
    return Model([x, emb, c], [out])
  } else {
    return Model([x, emb], [out])
  }
}

private func LoRAMiddleBlock(
  channels: Int, numHeads: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  x: Model.IO, emb: Model.IO, c: Model.IO
) -> Model.IO {
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let resBlock1 = LoRAResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let transformer = LoRASpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
    intermediateSize: channels * 4)
  out = transformer(out, c)
  let resBlock2 = LoRAResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  return out
}

private func LoRAInputBlocks(
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
      let inputLayer = LoRABlockLayer(
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

private func LoRAOutputBlocks(
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
      let outputLayer = LoRABlockLayer(
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

public func LoRAUNet(
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
  let timeEmbed = LoRATimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  var (inputs, inputBlocks) = LoRAInputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: startHeight,
    startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes, x: x, emb: emb, c: c,
    adapters: adapters)
  var out = inputBlocks
  let middleBlock = LoRAMiddleBlock(
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
  let outputBlocks = LoRAOutputBlocks(
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
