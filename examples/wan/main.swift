import Diffusion
import Foundation
import NNC
import PNG
import SentencePiece

struct PythonObject {}

DynamicGraph.setSeed(42)

let graph = DynamicGraph()

let prompt =
  "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

let negativePrompt =
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/Wan2.1/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model")
var tokens = sentencePiece.encode(prompt).map { return $0.id }
tokens.append(1)
print(tokens)
var negativeTokens = sentencePiece.encode(negativePrompt).map { return $0.id }
negativeTokens.append(1)
print(negativeTokens)

func UMT5TextEmbedding(vocabularySize: Int, embeddingSize: Int, name: String) -> Model {
  let tokenEmbed = Embedding(
    Float16.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: name)
  return tokenEmbed
}

func UMT5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int, upcast: Bool) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let positionBias = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  // No scaling the queries.
  let queries = toqueries(x).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + positionBias
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: outFeatures, noBias: true, name: "o")
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, positionBias], [out]))
}

func UMT5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int, upcast: Bool) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true, name: "w0")
  let wi_1 = Dense(count: intermediateSize, noBias: true, name: "w1")
  var out = wi_1(x) .* wi_0(x).GELU(approximate: .tanh)
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  out = wo(out)
  return (wi_0, wi_1, wo, Model([x], [out]))
}

func UMT5Block(
  prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int,
  upcast: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let relativePositionEmbedding = Embedding(
    Float16.self, vocabularySize: 32, embeddingSize: 64, name: "relative_position_embedding")
  let positionBias =
    relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, 64])
    .permuted(0, 3, 1, 2) + attentionMask
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (tokeys, toqueries, tovalues, unifyheads, attention) = UMT5LayerSelfAttention(
    k: k, h: h, b: b, t: t, outFeatures: outFeatures, upcast: upcast)
  let scaleFactor: Float = 8
  var out: Model.IO
  if upcast {
    out = x + scaleFactor * attention(norm1(x).to(.Float16), positionBias).to(of: x)
  } else {
    out = x + attention(norm1(x).to(.Float16), positionBias).to(of: x)
  }
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (wi_0, wi_1, wo, ff) = UMT5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize, upcast: upcast)
  if upcast {
    out = out + scaleFactor * ff(norm2(out).to(.Float16)).to(of: out)
  } else {
    out = out + ff(norm2(out).to(.Float16)).to(of: out)
  }
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, attentionMask, relativePositionBuckets], [out]))
}

func UMT5ForConditionalGeneration(b: Int, t: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let textEmbed = UMT5TextEmbedding(vocabularySize: 256_384, embeddingSize: 4_096, name: "shared")
  var out = textEmbed(x).to(.Float32)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = UMT5Block(
      prefix: "blocks.\(i)", k: 64, h: 64, b: b, t: t, outFeatures: 4_096,
      intermediateSize: 10_240, upcast: i >= 12)
    out = block(out, attentionMask, relativePositionBuckets)
    readers.append(reader)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out).to(.Float16)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x, attentionMask, relativePositionBuckets], [out]))
}

func relativePositionBuckets(sequenceLength: Int, numBuckets: Int, maxDistance: Int) -> Tensor<
  Int32
> {
  // isBidirectional = true.
  let numBuckets = numBuckets / 2
  let maxExact = numBuckets / 2
  var relativePositionBuckets = Tensor<Int32>(.CPU, .C(sequenceLength * sequenceLength))
  for i in 0..<sequenceLength {
    for j in 0..<sequenceLength {
      var relativePositionBucket = j > i ? numBuckets : 0
      let relativePosition = abs(i - j)
      let isSmall = relativePosition < maxExact
      if isSmall {
        relativePositionBucket += relativePosition
      } else {
        let relativePositionIfLarge = min(
          numBuckets - 1,
          maxExact
            + Int(
              (log(Double(relativePosition) / Double(maxExact))
                / log(Double(maxDistance) / Double(maxExact)) * Double(numBuckets - maxExact))
                .rounded(.down)))
        relativePositionBucket += relativePositionIfLarge
      }
      relativePositionBuckets[i * sequenceLength + j] = Int32(relativePositionBucket)
    }
  }
  return relativePositionBuckets
}

let context = graph.withNoGrad {
  let tokensTensor = graph.variable(.CPU, .C(256), of: Int32.self)
  for i in 0..<256 {
    tokensTensor[i] = i < tokens.count ? tokens[i] : 0
  }
  let negativeTokensTensor = graph.variable(.CPU, .C(256), of: Int32.self)
  for i in 0..<256 {
    negativeTokensTensor[i] = i < negativeTokens.count ? negativeTokens[i] : 0
  }
  let (reader, textModel) = UMT5ForConditionalGeneration(b: 1, t: 256)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: 256, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor.toGPU(0)
  let negativeTokensTensorGPU = negativeTokensTensor.toGPU(0)
  var attentionMask = Tensor<Float16>(.CPU, .NCHW(1, 1, 1, 256))
  for i in 0..<256 {
    attentionMask[0, 0, 0, i] = i < tokens.count ? 0 : -Float16.greatestFiniteMagnitude
  }
  let attentionMaskGPU = graph.variable(attentionMask.toGPU(0))
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/umt5_xxl_encoder_q8p.ckpt", flags: [.readOnly]
  ) {
    $0.read("text_model", model: textModel, codec: [.jit, .q8p, .ezm7])
  }
  var output = textModel(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[0]
    .as(of: Float16.self)
  let context = output[0..<tokens.count, 0..<4096].copied()
  output = textModel(inputs: negativeTokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[
    0
  ]
  .as(of: Float16.self)
  let negativeContext = output[0..<negativeTokens.count, 0..<4096].copied()
  return (context, negativeContext)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

func WanAttentionBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let context = Input()
  let c = (0..<6).map { _ in Input() }
  let rot = Input()
  let modulations = (0..<6).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, k * h), name: "attn_ada_ln_\($0)")
  }
  let chunks = zip(c, modulations).map { $0 + $1 }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = ((1 + chunks[1]) .* xNorm1(x) + chunks[0]).to(.Float16)
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  let queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xQ, right: rot)
  let keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xK, right: rot)
  let values = xV
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  out = xUnifyheads(out)
  out = x + chunks[2] .* out.to(of: x)
  let xNorm3 = LayerNorm(epsilon: 1e-6, axis: [2], name: "x_norm_3")
  xOut = xNorm3(out).to(.Float16)
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let xToContextQueries = Dense(count: k * h, name: "x_c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var cK = contextToKeys(context)
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "c_norm_k")
  cK = contextNormK(cK).reshaped([b, t, h, k])
  var cQ = xToContextQueries(xOut)
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_c_norm_q")
  cQ = contextNormQ(cQ).reshaped([b, hw, h, k])
  let cV = contextToValues(context).reshaped([b, t, h, k])
  let crossAttention = ScaledDotProductAttention(
    scale: 1 / Float(k).squareRoot(), flags: [.Float16])
  let crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  let contextUnifyheads = Dense(count: k * h, name: "c_o")
  out = out + contextUnifyheads(crossOut).to(of: out)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, upcast: false, name: "x")
  out =
    out + xFF(((1 + chunks[4]) .* xNorm2(out) + chunks[3]).to(.Float16)).to(of: out) .* chunks[5]
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x, context, rot] + c, [out]))
}

func TimeEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func Wan(
  channels: Int, layers: Int, intermediateSize: Int, time: Int, height: Int, width: Int,
  textLength: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let imgIn = Dense(count: channels, name: "x_embedder")
  var out = imgIn(x)
  let txt = Input()
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: channels, name: "c")
  let context = contextEmbedder(txt)
  let t = Input()
  let rot = Input()
  let (timeInMlp0, timeInMlp2, timeIn) = TimeEmbedder(channels: channels, name: "t")
  let vector = timeIn(t).reshaped([1, 1, channels])
  let vectorIn = vector.swish()
  let timeProjections = (0..<6).map { Dense(count: channels, name: "ada_ln_\($0)") }
  let tOut = timeProjections.map { $0(vectorIn) }
  let h = height / 2
  let w = width / 2
  var readers = [(PythonObject) -> Void]()
  out = out.to(.Float32)
  for i in 0..<layers {
    let (reader, block) = WanAttentionBlock(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: 1, t: textLength, hw: time * h * w,
      intermediateSize: intermediateSize)
    out = block([out, context, rot] + tOut)
    readers.append(reader)
  }
  let scale = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_0")
  let shift = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = ((1 + scale + vector) .* normFinal(out) + (vector + shift)).to(.Float16)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (Model([x, txt, t, rot], [out]), reader)
}

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timesteps
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

graph.maxConcurrency = .limit(1)
let z = graph.withNoGrad {
  var rotNdTensor = graph.variable(.CPU, .NHWC(1, 21 * 30 * 52, 1, 128), of: Float.self)
  for t in 0..<21 {
    for y in 0..<30 {
      for x in 0..<52 {
        let i = t * 30 * 52 + y * 52 + x
        for k in 0..<22 {
          let theta = Double(t) * 1.0 / pow(10_000, Double(k) / 22)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, k * 2] = Float(costheta)
          rotNdTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<21 {
          let theta = Double(y) * 1.0 / pow(10_000, Double(k) / 21)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, (k + 22) * 2] = Float(costheta)
          rotNdTensor[0, i, 0, (k + 22) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<21 {
          let theta = Double(x) * 1.0 / pow(10_000, Double(k) / 21)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, (k + 22 + 21) * 2] = Float(costheta)
          rotNdTensor[0, i, 0, (k + 22 + 21) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  let (wan, reader) = Wan(
    channels: 1536, layers: 30, intermediateSize: 8960, time: 21, height: 60, width: 104,
    // channels: 5120, layers: 40, intermediateSize: 13824, time: 21, height: 60, width: 104,
    textLength: 512)
  var xTensor = graph.variable(.GPU(0), .HWC(1, 21 * 30 * 52, 16 * 2 * 2), of: Float16.self)
  xTensor.randn()
  let txt = context.0.reshaped(.WC(tokens.count, 4096)).toGPU(0)
  var txtIn = graph.variable(.GPU(0), .WC(512, 4096), of: Float16.self)
  txtIn.full(0)
  txtIn[0..<tokens.count, 0..<4096] = txt
  txtIn = txtIn.reshaped(.HWC(1, 512, 4096))
  let negTxt = context.1.reshaped(.WC(negativeTokens.count, 4096)).toGPU(0)
  var negTxtIn = graph.variable(.GPU(0), .WC(512, 4096), of: Float16.self)
  negTxtIn.full(0)
  negTxtIn[0..<negativeTokens.count, 0..<4096] = negTxt
  negTxtIn = negTxtIn.reshaped(.HWC(1, 512, 4096))
  let timestep = timeEmbedding(timesteps: 900, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let tGPU = graph.variable(Tensor<Float>(from: timestep)).toGPU(0)
  let rotNdTensorGPU = DynamicGraph.Tensor<Float16>(from: rotNdTensor).toGPU(0)
  wan.compile(inputs: xTensor, txtIn, tGPU, rotNdTensorGPU)
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/wan_v2.1_1.3b_480p_q8p.ckpt", flags: [.readOnly]
  ) {
    $0.read("dit", model: wan, codec: [.jit, .q8p, .ezm7])
  }
  let samplingSteps = 30
  for i in (1...samplingSteps).reversed() {
    let t = Float(i) / Float(samplingSteps) * 1_000
    let tGPU = graph.variable(
      Tensor<Float>(
        from: timeEmbedding(timesteps: t, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
          .toGPU(0)))
    var vc = wan(
      inputs: xTensor, txtIn, tGPU, rotNdTensorGPU)[0].as(
        of: Float16.self)
    vc = vc.reshaped(format: .NCHW, shape: [1, 21 * 30 * 52, 2, 2, 16]).permuted(0, 1, 4, 2, 3)
      .contiguous().reshaped(.CHW(1, 21 * 30 * 52, 16 * 2 * 2))
    var vu = wan(
      inputs: xTensor, negTxtIn, tGPU, rotNdTensorGPU)[0].as(
        of: Float16.self)
    vu = vu.reshaped(format: .NCHW, shape: [1, 21 * 30 * 52, 2, 2, 16]).permuted(0, 1, 4, 2, 3)
      .contiguous().reshaped(.CHW(1, 21 * 30 * 52, 16 * 2 * 2))
    let v = vu + 6 * (vc - vu)
    xTensor = xTensor - (1 / Float(samplingSteps)) * v
    debugPrint(xTensor)
  }
  let z = xTensor.reshaped(format: .NCHW, shape: [1, 21, 30, 52, 16, 2, 2]).permuted(
    0, 4, 1, 2, 5, 3, 6
  )
  .contiguous().reshaped(format: .NCHW, shape: [1, 16, 21, 60, 104])
  return z
}

struct ResnetBlockCausal3D {
  private let norm1: Model
  private let conv1: Model
  private let norm2: Model
  private let conv2: Model
  private let ninShortcut: Model?
  init(outChannels: Int, shortcut: Bool) {
    norm1 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm1")
    conv1 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      name: "resnet_conv1")
    norm2 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm2")
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      name: "resnet_conv2")
    if shortcut {
      ninShortcut = Convolution(
        groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
        name: "resnet_shortcut")
    } else {
      ninShortcut = nil
    }
  }
  private var conv1Inputs: Model.IO? = nil
  private var conv2Inputs: Model.IO? = nil
  mutating func callAsFunction(
    input x: Model.IO, prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool,
    depth: Int, height: Int, width: Int, inputsOnly: Bool
  ) -> (
    (PythonObject) -> Void, Model.IO
  ) {
    var out = norm1(x.reshaped([inChannels, depth, height, width]))
    var pre = out.swish()
    if let conv1Inputs = conv1Inputs {
      out = conv1(
        Functional.concat(axis: 1, conv1Inputs, pre, flags: [.disableOpt]).reshaped([
          1, inChannels, depth + 2, height, width,
        ]).padded(.zero, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1]))
    } else {
      out = conv1(
        pre.reshaped([
          1, inChannels, depth, height, width,
        ]).padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
    }
    if !inputsOnly {
      conv1Inputs = pre.reshaped(
        [inChannels, 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).contiguous()
    } else {
      conv1Inputs = nil
    }
    out = norm2(out.reshaped([outChannels, depth, height, width]))
    pre = out.swish()
    if let conv2Inputs = conv2Inputs {
      out = conv2(
        Functional.concat(axis: 1, conv2Inputs, pre, flags: [.disableOpt]).reshaped([
          1, outChannels, depth + 2, height, width,
        ]).padded(.zero, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1]))
    } else {
      out = conv2(
        pre.reshaped([
          1, outChannels, depth, height, width,
        ]).padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
    }
    if !inputsOnly {
      conv2Inputs = pre.reshaped(
        [outChannels, 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).contiguous()
    } else {
      conv2Inputs = nil
    }
    if let ninShortcut = ninShortcut {
      out = ninShortcut(x) + out
    } else {
      out = x + out
    }
    let reader: (PythonObject) -> Void = { state_dict in
    }
    return (reader, out)
  }
}

struct AttnBlockCausal3D {
  let norm: Model
  let toqueries: Model
  let tokeys: Model
  let tovalues: Model
  let projOut: Model
  init(inChannels: Int) {
    norm = RMSNorm(epsilon: 1e-6, axis: [0], name: "attn_norm")
    tokeys = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "to_k")
    toqueries = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "to_q")
    tovalues = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "to_v")
    projOut = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "proj_out")
  }
  func callAsFunction(
    prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
  ) -> (
    (PythonObject) -> Void, Model
  ) {
    let x = Input()
    var out = norm(x.reshaped([inChannels, depth, height, width])).reshaped([
      1, inChannels, depth, height, width,
    ])
    let hw = width * height
    let k = tokeys(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
    let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
      inChannels, depth, hw,
    ]).transposed(0, 1)
    var dot =
      Matmul(transposeA: (1, 2))(q, k)
    dot = dot.reshaped([depth * hw, hw])
    dot = dot.softmax()
    dot = dot.reshaped([depth, hw, hw])
    let v = tovalues(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
    out = Matmul(transposeB: (1, 2))(v, dot)
    out = x + projOut(out.transposed(0, 1).reshaped([1, inChannels, depth, height, width]))
    let reader: (PythonObject) -> Void = { state_dict in
    }
    return (reader, Model([x], [out]))
  }
}

func DecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "post_quant_conv")
  let postQuantX = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_in")
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_out")
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  var finalOut: Model.IO? = nil
  var midBlock1Builder = ResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let midAttn1Builder = AttnBlockCausal3D(inChannels: previousChannel)
  var midBlock2Builder = ResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  var upBlockBuilders = [ResnetBlockCausal3D]()
  for (i, channel) in channels.enumerated().reversed() {
    for _ in 0..<numRepeat + 1 {
      upBlockBuilders.append(
        ResnetBlockCausal3D(outChannels: channel, shortcut: previousChannel != channel))
      previousChannel = channel
    }
    if i > 0 {
      previousChannel = channel / 2
    }
  }
  let timeConvs = (0..<(channels.count - 2)).map { i in
    return Convolution(
      groups: 1, filters: channels[channels.count - i - 1] * 2, filterSize: [3, 1, 1],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
      name: "time_conv")
  }
  let upsampleConv2d = (0..<(channels.count - 1)).map { i in
    return Convolution(
      groups: 1, filters: channels[channels.count - i - 1] / 2, filterSize: [1, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      name: "upsample")
  }
  var timeInputs: [Model.IO?] = Array(repeating: nil, count: channels.count - 2)
  var convOutInputs: Model.IO? = nil
  for d in stride(from: 0, to: startDepth - 1, by: 2) {
    previousChannel = channels[channels.count - 1]
    var out: Model.IO
    if d == 0 {
      out = postQuantX.reshaped(
        [1, 16, min(startDepth - d, 3), startHeight, startWidth], offset: [0, 0, d, 0, 0],
        strides: [
          16 * startDepth * startHeight * startWidth, startDepth * startHeight * startWidth,
          startHeight * startWidth, startWidth, 1,
        ]
      ).contiguous()
      out = convIn(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
    } else {
      out = postQuantX.reshaped(
        [1, 16, min(startDepth - (d - 1), 4), startHeight, startWidth],
        offset: [0, 0, d - 1, 0, 0],
        strides: [
          16 * startDepth * startHeight * startWidth, startDepth * startHeight * startWidth,
          startHeight * startWidth, startWidth, 1,
        ]
      ).contiguous()
      if let last = finalOut {
        out.add(dependencies: [last])
      }
      out = convIn(out.padded(.zero, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1]))
    }
    let inputsOnly = startDepth - 1 - d <= 2  // This is the last one.
    var width = startWidth
    var height = startHeight
    var depth = d > 0 ? min(startDepth - 1 - d, 2) : 3
    let (midBlockReader1, midBlock1Out) = midBlock1Builder(
      input: out,
      prefix: "decoder.middle.0", inChannels: previousChannel,
      outChannels: previousChannel, shortcut: false, depth: depth, height: height,
      width: width, inputsOnly: inputsOnly)
    out = midBlock1Out
    let (midAttnReader1, midAttn1) = midAttn1Builder(
      prefix: "decoder.middle.1", inChannels: previousChannel, depth: depth,
      height: height, width: width)
    out = midAttn1(out)
    let (midBlockReader2, midBlock2Out) = midBlock2Builder(
      input: out,
      prefix: "decoder.middle.2", inChannels: previousChannel,
      outChannels: previousChannel, shortcut: false, depth: depth, height: height,
      width: width, inputsOnly: inputsOnly)
    out = midBlock2Out
    var readers = [(PythonObject) -> Void]()
    var j = 0
    var k = 0
    for (i, channel) in channels.enumerated().reversed() {
      for _ in 0..<numRepeat + 1 {
        let (reader, blockOut) = upBlockBuilders[j](
          input: out,
          prefix: "decoder.upsamples.\(k)",
          inChannels: previousChannel, outChannels: channel,
          shortcut: previousChannel != channel, depth: depth, height: height, width: width,
          inputsOnly: inputsOnly)
        readers.append(reader)
        out = blockOut
        previousChannel = channel
        j += 1
        k += 1
      }
      if i > 0 {
        if i > 1 && startDepth > 1 {  // Need to bump up on the depth axis.
          if d == 0 {  // Special case for first frame.
            let first = out.reshaped(
              [channel, 1, height, width],
              strides: [depth * height * width, height * width, width, 1]
            ).contiguous()
            let more = out.reshaped(
              [channel, (depth - 1), height, width], offset: [0, 1, 0, 0],
              strides: [depth * height * width, height * width, width, 1]
            ).contiguous()
            var expanded = timeConvs[channels.count - i - 1](
              more.reshaped([1, channel, depth - 1, height, width]).padded(
                .zero, begin: [0, 0, 2, 0, 0], end: [0, 0, 0, 0, 0]))
            if !inputsOnly {
              timeInputs[channels.count - i - 1] = out.reshaped(
                [channel, 2, height, width], offset: [0, depth - 2, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).contiguous()
            }
            let upLayer = k
            let reader: (PythonObject) -> Void = { state_dict in
            }
            readers.append(reader)
            expanded = expanded.reshaped([2, channel, depth - 1, height, width]).permuted(
              1, 2, 0, 3, 4
            )
            .contiguous().reshaped([channel, 2 * (depth - 1), height, width])
            out = Functional.concat(axis: 1, first, expanded)
            depth = 1 + (depth - 1) * 2
            out = out.reshaped([1, channel, depth, height, width])
          } else if let timeInput = timeInputs[channels.count - i - 1] {
            let more = out.reshaped([channel, depth, height, width])
            let expanded = timeConvs[channels.count - i - 1](
              Functional.concat(axis: 1, timeInput, more, flags: [.disableOpt]).reshaped([
                1, channel, depth + 2, height, width,
              ]))
            if !inputsOnly {
              timeInputs[channels.count - i - 1] = out.reshaped(
                [channel, 2, height, width], offset: [0, depth - 2, 0, 0],
                strides: [depth * height * width, height * width, width, 1]
              ).contiguous()
            }
            let upLayer = k
            let reader: (PythonObject) -> Void = { state_dict in
            }
            readers.append(reader)
            out = expanded.reshaped([2, channel, depth, height, width]).permuted(1, 2, 0, 3, 4)
              .contiguous().reshaped([1, channel, 2 * depth, height, width])
            depth = depth * 2
          }
        }
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
          out.reshaped([channel, depth, height, width])
        ).reshaped([1, channel, depth, height * 2, width * 2])
        width *= 2
        height *= 2
        out = upsampleConv2d[channels.count - i - 1](out)
        previousChannel = channel / 2
        let upLayer = k
        let reader: (PythonObject) -> Void = { state_dict in
        }
        readers.append(reader)
        k += 1
      }
    }
    out = normOut(out.reshaped([channels[0], depth, height, width]))
    let pre = out.swish()
    if let convOutInputs = convOutInputs {
      out = convOut(
        Functional.concat(axis: 1, convOutInputs, pre, flags: [.disableOpt]).reshaped([
          1, channels[0], depth + 2, height, width,
        ]).padded(.zero, begin: [0, 0, 0, 1, 1], end: [0, 0, 0, 1, 1]))
    } else {
      out = convOut(
        pre.reshaped([
          1, channels[0], depth, height, width,
        ]).padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
    }
    if !inputsOnly {
      convOutInputs = pre.reshaped(
        [channels[0], 2, height, width], offset: [0, depth - 2, 0, 0],
        strides: [depth * height * width, height * width, width, 1]
      ).contiguous()
    }
    if let otherOut = finalOut {
      finalOut = Functional.concat(axis: 2, otherOut, out, flags: [.disableOpt])
    } else {
      finalOut = out
    }
  }
  let out = finalOut!
  let reader: (PythonObject) -> Void = { state_dict in
  }
  return (reader, Model([x], [out]))
}

func EncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_in")
  var out = convIn(x.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  var readers = [(PythonObject) -> Void]()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  var k = 0
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      var builder = ResnetBlockCausal3D(outChannels: channel, shortcut: previousChannel != channel)
      let (reader, blockOut) = builder(
        input: out,
        prefix: "encoder.downsamples.\(k)", inChannels: previousChannel,
        outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width,
        inputsOnly: true)
      readers.append(reader)
      out = blockOut
      previousChannel = channel
      k += 1
    }
    if i < channels.count - 1 {
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [1, 3, 3],
        hint: Hint(
          stride: [1, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        name: "downsample")
      out = conv2d(out.padded(.zero, begin: [0, 0, 0, 0, 0], end: [0, 0, 0, 1, 1]))
      let downLayer = k
      let reader: (PythonObject) -> Void = { state_dict in
      }
      readers.append(reader)
      if i > 0 && depth > 1 {
        let first = out.reshaped(
          [1, channel, 1, height, width],
          strides: [depth * height * width, depth * height * width, height * width, width, 1]
        ).contiguous()
        let timeConv = Convolution(
          groups: 1, filters: channel, filterSize: [3, 1, 1],
          hint: Hint(
            stride: [2, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
          name: "time_conv")
        let shrunk = timeConv(out)
        let upLayer = k
        let reader: (PythonObject) -> Void = { state_dict in
        }
        readers.append(reader)
        depth = (depth - 1) / 2 + 1
        out = Functional.concat(axis: 2, first, shrunk)
      }
      k += 1
    }
  }
  var midBlock1Builder = ResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let (midBlockReader1, midBlock1Out) = midBlock1Builder(
    input: out,
    prefix: "encoder.middle.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width, inputsOnly: true)
  out = midBlock1Out
  var midAttn1Builder = AttnBlockCausal3D(inChannels: previousChannel)
  let (midAttnReader1, midAttn1) = midAttn1Builder(
    prefix: "encoder.middle.1", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out)
  var midBlock2Builder = ResnetBlockCausal3D(outChannels: previousChannel, shortcut: false)
  let (midBlockReader2, midBlock2Out) = midBlock2Builder(
    input: out,
    prefix: "encoder.middle.2", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width, inputsOnly: true)
  out = midBlock2Out
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let quantConv = Convolution(groups: 1, filters: 32, filterSize: [1, 1, 1], name: "quant_conv")
  out = quantConv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
  }
  return (reader, Model([x], [out]))
}

graph.withNoGrad {
  let mean = graph.variable(
    Tensor<Float>(
      [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
      ], kind: .GPU(0), format: .NCHW, shape: [1, 16, 1, 1, 1]))
  let std = graph.variable(
    Tensor<Float>(
      [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
      ], kind: .GPU(0), format: .NCHW, shape: [1, 16, 1, 1, 1]))
  let zTensor = DynamicGraph.Tensor<Float16>(
    from: (DynamicGraph.Tensor<Float>(from: z) .* std + mean)[
      0..<1, 0..<16, 0..<21, 0..<60, 0..<104
    ].copied())
  debugPrint(zTensor)
  let (decoderReader, decoder) = DecoderCausal3D(
    channels: [96, 192, 384, 384], numRepeat: 2, startWidth: 104, startHeight: 60, startDepth: 21)
  decoder.compile(inputs: zTensor)
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/wan_v2.1_video_vae_f32.ckpt", flags: [.readOnly]
  ) {
    $0.read("decoder", model: decoder)
  }
  // DynamicGraph.logLevel = .verbose
  let image = decoder(inputs: zTensor)[0].as(of: Float16.self).toCPU()
  debugPrint(image)
  for k in 0..<81 {
    var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: 104 * 8 * 60 * 8)
    for y in 0..<(60 * 8) {
      for x in 0..<(104 * 8) {
        let (r, g, b) = (image[0, 0, k, y, x], image[0, 1, k, y, x], image[0, 2, k, y, x])
        rgba[y * 104 * 8 + x].r = UInt8(
          min(max(Int(Float(r.isFinite ? (r + 1) / 2 : 0) * 255), 0), 255))
        rgba[y * 104 * 8 + x].g = UInt8(
          min(max(Int(Float(g.isFinite ? (g + 1) / 2 : 0) * 255), 0), 255))
        rgba[y * 104 * 8 + x].b = UInt8(
          min(max(Int(Float(b.isFinite ? (b + 1) / 2 : 0) * 255), 0), 255))
      }
    }
    let image = PNG.Data.Rectangular(
      packing: rgba, size: (104 * 8, 60 * 8),
      layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
    try! image.compress(path: "/home/liu/workspace/swift-diffusion/frame-\(k).png", level: 4)
  }
  debugPrint(image)
}
