import Foundation
import NNC

enum Flux3Config {
  static let width = 3_072
  static let heads = 24
  static let headDim = 128
  static let intermediate = 9_216
  static let refinerLayers = 5
  static let layers = 28
  static let latentChannels = 128
  static let contextWidth = 20_480
  static let ropeAxes = [32, 32, 32, 32]
  static let ropeTheta: Double = 10_000
  static let epsilon: Float = 1e-6
  // Power-of-two compensation keeps the FP16 projection outputs in range.
  static let fp16TextScale: Float = 4
  static let fp16FFNScale: Float = 8
}

struct Flux3Binding {
  let model: Model
  let key: String
  var transform = "identity"
  var heads = 0
  var isBias = false
}

func flux3SiLU<T: TensorNumeric>(_ x: Model.IO, _ type: T.Type) -> Model.IO {
  // The CUDA swish kernel lacks BF16; compute in FP32, then round at the
  // same operator boundary as PyTorch's BF16 SiLU.
  if T.dataType == .BFloat16 { return x.to(.Float32).swish().to(.BFloat16) }
  return x.swish()
}

func flux3OnePlus<T: TensorNumeric>(_ x: Model.IO, _ type: T.Type) -> Model.IO {
  // Avoid the unsupported CUDA BF16 scalar-constant initializer.
  if T.dataType == .BFloat16 { return (1 + x.to(.Float32)).to(.BFloat16) }
  return 1 + x
}

func flux3ModulatedNorm<T: TensorNumeric>(
  _ x: Model.IO, scale: Model.IO, shift: Model.IO, type: T.Type
) -> Model.IO {
  let norm = LayerNorm(epsilon: Flux3Config.epsilon, axis: [2], elementwiseAffine: false)
  if T.dataType == .Float16 {
    // Residuals, normalization, and AdaLN stay in FP32 until the projections.
    return ((1 + scale.to(.Float32)) .* norm(x.to(.Float32)) + shift.to(.Float32))
      .to(.Float16)
  }
  if T.dataType == .BFloat16 {
    // The native sampler's autocast promotes layer_norm and modulation to FP32.
    // 1 + scale is still rounded in BF16; the following linear rounds its input.
    return
      (flux3OnePlus(scale, type).to(.Float32) .* norm(x.to(.Float32))
      + shift.to(.Float32)).to(.BFloat16)
  }
  return flux3OnePlus(scale, type) .* norm(x) + shift
}

// FP32 controls use explicit attention; reduced precision uses the same SDPA path as FLUX.2.
func flux3Attention<T: TensorNumeric>(
  _ type: T.Type, q: Model.IO, k: Model.IO, v: Model.IO, length: Int, heads: Int
) -> Model.IO {
  if T.dataType != .Float32 {
    return ScaledDotProductAttention(scale: 1 / Float(128).squareRoot())(q, k, v)
      .reshaped([1, length, heads * 128])
  }
  let queries = q.transposed(1, 2).reshaped([heads, length, 128])
  let keys = k.transposed(1, 2).reshaped([heads, length, 128])
  let values = v.transposed(1, 2).reshaped([heads, length, 128])
  let logits = Matmul(transposeB: (1, 2))(queries, keys) * (1 / Float(128).squareRoot())
  let probabilities = logits.reshaped([heads * length, length]).softmax()
    .reshaped([heads, length, length])
  return Matmul()(probabilities, values).reshaped([1, heads, length, 128])
    .transposed(1, 2).reshaped([1, length, heads * 128])
}

func Flux3Block<T: TensorNumeric>(
  _ type: T.Type, length: Int, prefix: String, name: String
) -> (Model, [Flux3Binding]) {
  let x = Input()
  let rotary = Input()
  let shift = Input()
  let scale = Input()
  let gate = Input()
  let u = flux3ModulatedNorm(x, scale: scale, shift: shift, type: type)
  let q = Dense(count: 3_072, noBias: true, name: "\(name)_q")
  let k = Dense(count: 3_072, noBias: true, name: "\(name)_k")
  let v = Dense(count: 3_072, noBias: true, name: "\(name)_v")
  let qNorm = RMSNorm(epsilon: Flux3Config.epsilon, axis: [3], name: "\(name)_norm_q")
  let kNorm = RMSNorm(epsilon: Flux3Config.epsilon, axis: [3], name: "\(name)_norm_k")
  // BFL keeps normalization, learned norm scale, and RoPE in FP32 until attention.
  let queries = Functional.cmul(
    left: qNorm(q(u).reshaped(.NHWC(1, length, 24, 128)).to(.Float32)), right: rotary
  ).to(T.dataType)
  let keys = Functional.cmul(
    left: kNorm(k(u).reshaped(.NHWC(1, length, 24, 128)).to(.Float32)), right: rotary
  ).to(T.dataType)
  let values = v(u).reshaped(.NHWC(1, length, 24, 128))
  let attended = flux3Attention(type, q: queries, k: keys, v: values, length: length, heads: 24)
  let output = Dense(count: 3_072, noBias: true, name: "\(name)_o")
  let w1 = Dense(count: 9_216, noBias: true, name: "\(name)_w1")
  let w3 = Dense(count: 9_216, noBias: true, name: "\(name)_w3")
  let w2 = Dense(count: 3_072, noBias: true, name: "\(name)_w2")
  let result: Model.IO
  if T.dataType == .Float16 {
    // Both FFN projections are bias-free. Scale the gated product before w2,
    // then restore its magnitude only after converting the output to FP32.
    let product = Functional.swishMul(
      value: w3(u), gate: w1(u), scale: 1 / Flux3Config.fp16FFNScale)
    let ffn = w2(product).to(.Float32) * Flux3Config.fp16FFNScale
    let update = gate.to(.Float32) .* (output(attended).to(.Float32) + ffn)
    result = x.to(.Float32) + update
  } else {
    result = x + gate .* (output(attended) + w2(w3(u) .* flux3SiLU(w1(u), type)))
  }
  return (
    // Like FLUX.2 and Z Image, names belong to the parameterized leaf models.
    // The store supplies occurrence indices for repeated block projections.
    Model([x, rotary, shift, scale, gate], [result]),
    [
      .init(model: q, key: prefix + ".q_proj.weight"),
      .init(model: k, key: prefix + ".k_proj.weight"),
      .init(model: v, key: prefix + ".v_proj.weight"),
      .init(model: qNorm, key: prefix + ".norm.query_norm.scale", transform: "float32"),
      .init(model: kNorm, key: prefix + ".norm.key_norm.scale", transform: "float32"),
      .init(model: output, key: prefix + ".attn_out.weight"),
      .init(model: w1, key: prefix + ".mlp_in.weight", transform: "first_half"),
      .init(model: w3, key: prefix + ".mlp_in.weight", transform: "second_half"),
      .init(model: w2, key: prefix + ".mlp_out.weight"),
    ]
  )
}

// Sequence order follows our FLUX.2 implementation: context, target image, references.
// All rotary tensors and per-token modulations use that same order.
func Flux3DiT<T: TensorNumeric>(
  _ type: T.Type, textLength: Int, imageLength: Int, referenceLength: Int,
  flowMap: Bool = true, debug: Bool = false
) -> (Model, [Flux3Binding], [String]) {
  let x = Input()
  let context = Input()
  let reference = Input()
  let rotary = Input()
  let time = Input()
  let nextTime = Input()
  var bindings = [Flux3Binding]()
  func dense(_ count: Int, _ key: String, _ name: String) -> Model {
    let layer = Dense(count: count, noBias: true, name: name)
    let keepFP32 = T.dataType == .Float16 && ["t_2", "next_t_2", "final_modulation"].contains(name)
    bindings.append(
      .init(model: layer, key: key + ".weight", transform: keepFP32 ? "float32" : "identity"))
    return layer
  }
  let time0 = dense(3_072, "time_in.in_layer", "t_0")
  let time2 = dense(3_072, "time_in.out_layer", "t_2")
  let timeHidden = flux3SiLU(time0(time), type)
  let embeddedTime = time2(T.dataType == .Float16 ? timeHidden.to(.Float32) : timeHidden)
  func timeRow(_ i: Int) -> Model.IO {
    embeddedTime.reshaped([1, 3_072], offset: [i, 0], strides: [3_072, 1]).contiguous()
  }
  var xVector = timeRow(0)
  var cVector = timeRow(1)
  if flowMap {
    let next0 = dense(3_072, "ref_time_in.in_layer", "next_t_0")
    let next2 = dense(3_072, "ref_time_in.out_layer", "next_t_2")
    let hidden = flux3SiLU(next0(nextTime), type)
    let vector = next2(T.dataType == .Float16 ? hidden.to(.Float32) : hidden)
    xVector = xVector + vector
    cVector = cVector + vector
  }
  func modulation(_ vector: Model.IO, _ key: String, _ name: String) -> [Model.IO] {
    let projected = dense(9_216, key + ".lin", name)(flux3SiLU(vector, type).to(T.dataType))
    return (0..<3).map {
      projected.reshaped(
        [1, 1, 3_072], offset: [0, 0, $0 * 3_072],
        strides: [9_216, 9_216, 1]
      ).contiguous()
    }
  }
  let cm = modulation(cVector, "early_stream_modulations.txt", "context_refiner_modulation")
  let xm = modulation(xVector, "early_stream_modulations.image", "x_refiner_modulation")
  let cs = modulation(cVector, "single_stream_modulations.txt", "context_modulation")
  let xs = modulation(xVector, "single_stream_modulations.image", "x_modulation")
  let length = textLength + imageLength + referenceLength
  func rotarySlice(_ offset: Int, _ count: Int) -> Model.IO {
    rotary.reshaped(
      [1, count, 1, 128], offset: [0, offset, 0, 0],
      strides: [length * 128, 128, 128, 1]
    ).contiguous()
  }
  let contextEmbedder = dense(3_072, "txt_in", "context_embedder")
  var c: Model.IO
  if T.dataType == .Float16 {
    c =
      contextEmbedder(context * (1 / Flux3Config.fp16TextScale)).to(.Float32)
      * Flux3Config.fp16TextScale
  } else {
    c = contextEmbedder(context)
  }
  var image = dense(3_072, "emb_in.image", "x_embedder")(x)
  let cr = rotarySlice(0, textLength)
  let xr = rotarySlice(textLength, imageLength)
  var outputs = [Model.IO]()
  var names = [String]()
  func capture(_ name: String, _ value: Model.IO) {
    if debug {
      names.append(name)
      outputs.append(value)
    }
  }
  for i in 0..<Flux3Config.refinerLayers {
    let (block, weights) = Flux3Block(
      type, length: textLength,
      prefix: "txt_mode_blocks.\(i)", name: "context_refiner")
    bindings += weights
    c = block(c, cr, cm[0], cm[1], cm[2])
    let (imageBlock, imageWeights) = Flux3Block(
      type, length: imageLength,
      prefix: "content_mode_blocks.image.\(i)", name: "x_refiner")
    bindings += imageWeights
    image = imageBlock(image, xr, xm[0], xm[1], xm[2])
  }
  capture("context_refiner", c)
  capture("x_refiner", image)
  var h = Functional.concat(axis: 1, c, image)
  // Broadcast the modality modulation before joining the sequence.
  let residualType = T.dataType == .Float16 ? DataType.Float32 : T.dataType
  var mods = (0..<3).map { i in
    Functional.concat(
      axis: 1, cs[i].to(residualType) + c * 0, xs[i].to(residualType) + image * 0)
  }
  if referenceLength > 0 {
    let rm = modulation(cVector, "early_stream_modulations.image_cond", "ref_refiner_modulation")
    let rs = modulation(cVector, "single_stream_modulations.image_cond", "ref_modulation")
    var r = dense(3_072, "emb_in.image_cond", "ref_embedder")(reference)
    let rr = rotarySlice(textLength + imageLength, referenceLength)
    for i in 0..<Flux3Config.refinerLayers {
      let (block, weights) = Flux3Block(
        type, length: referenceLength,
        prefix: "content_mode_blocks.image_cond.\(i)", name: "ref_refiner")
      bindings += weights
      r = block(r, rr, rm[0], rm[1], rm[2])
    }
    capture("ref_refiner", r)
    h = Functional.concat(axis: 1, h, r)
    mods = (0..<3).map {
      Functional.concat(axis: 1, mods[$0], rs[$0].to(residualType) + r * 0)
    }
  }
  for i in 0..<Flux3Config.layers {
    let (block, weights) = Flux3Block(
      type, length: length,
      prefix: "single_blocks.\(i)", name: "x")
    bindings += weights
    h = block(h, rotary, mods[0], mods[1], mods[2])
    if [0, 7, 27].contains(i) { capture("single_\(i)", h) }
  }
  let imageOut = h.reshaped(
    [1, imageLength, 3_072], offset: [0, textLength, 0],
    strides: [length * 3_072, 3_072, 1]
  ).contiguous()
  let finalMod = dense(6_144, "final_layer.image.adaLN_modulation.1", "final_modulation")(
    flux3SiLU(xVector, type))
  let shift = finalMod.reshaped([1, 1, 3_072], strides: [6_144, 6_144, 1]).contiguous()
  let scale = finalMod.reshaped(
    [1, 1, 3_072], offset: [0, 0, 3_072],
    strides: [6_144, 6_144, 1]
  ).contiguous()
  let out = dense(128, "final_layer.image.linear", "linear")(
    flux3ModulatedNorm(imageOut, scale: scale, shift: shift, type: type))
  outputs.append(out)
  names.append("output")
  return (
    Model(
      [x, context] + (referenceLength > 0 ? [reference] : [])
        + [rotary, time] + (flowMap ? [nextTime] : []), outputs), bindings, names
  )
}

func flux3Rotary(_ ids: Tensor<Float>) -> Tensor<Float> {
  let length = ids.shape[0]
  var out = Tensor<Float>(.CPU, .NHWC(1, length, 1, 128))
  for token in 0..<length {
    for axis in 0..<4 {
      for pair in 0..<16 {
        let phase = Double(ids[token, axis]) / pow(Flux3Config.ropeTheta, Double(2 * pair) / 32)
        out[0, token, 0, axis * 32 + pair * 2] = Float(cos(phase))
        out[0, token, 0, axis * 32 + pair * 2 + 1] = Float(sin(phase))
      }
    }
  }
  return out
}

func flux3Time(_ values: [Float], precision: String = "float32") -> Tensor<Float> {
  var out = Tensor<Float>(.CPU, .NC(values.count, 256))
  for (row, value) in values.enumerated() {
    // Upstream multiplies in the timestep dtype BEFORE converting to FP32.
    let scaled: Float
    if precision == "bfloat16" {
      scaled = BFloat16(BFloat16(value).floatValue * 1_000).floatValue
    } else if precision == "float16" {
      scaled = Float(Float16(Float(Float16(value)) * 1_000))
    } else {
      scaled = value * 1_000
    }
    for i in 0..<128 {
      let angle = scaled * exp(-log(Float(10_000)) * Float(i) / 128)
      out[row, i] = cos(angle)
      out[row, i + 128] = sin(angle)
    }
  }
  return out
}
