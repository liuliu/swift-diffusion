import Foundation
import NNC

enum FiboDiTConfig {
  static let width = 3_584
  static let heads = 28
  static let headDim = 128
  static let intermediate = 9_536
  static let layers = 44
  static let refinerLayers = 2
  static let perceiverLayers = 2
  static let contextWidth = 7_680
  static let packedChannels = 128
  static let injectionLayers = [6, 14, 22, 30, 38]
  static let maxGist = 512
  static let ropeAxes = [32, 48, 48]
  static let ropeTheta: Double = 256
}

// Inputs: packed image+references, six text bundles, FP32 image RoPE,
// FP32 gist+image+reference RoPE, time sinusoids [noisy, clean], and two
// Int32 row selectors (0 = noisy image, 1 = clean reference/gist).
func FiboDiT<T: TensorNumeric>(
  _ type: T.Type, imageLength: Int, referenceLength: Int, textLength: Int,
  gistCount: Int, device: Int, diagnostic: Bool = true, mainLayers: Int = 44
) -> (Model, [FiboBinding], [String]) {
  let d = FiboDiTConfig.width
  let visualLength = imageLength + referenceLength
  let length = gistCount + visualLength
  let x = Input()
  let contexts = (0..<6).map { _ in Input() }
  let imageRotary = Input()
  let rotary = Input()
  let time = Input()
  let refinerSelector = Input()
  let selector = Input()
  var bindings = [FiboBinding]()
  var outputs = [Model.IO]()
  var names = [String]()
  func tap(_ value: Model.IO, _ name: String) {
    if diagnostic {
      outputs.append(value)
      names.append(name)
    }
  }
  func dense(_ key: String, _ count: Int, bias: Bool = false) -> Model {
    let model = Dense(
      count: count, noBias: !bias, name: key.replacingOccurrences(of: ".", with: "_"))
    bindings.append(.init(model: model, key: key + ".weight"))
    if bias { bindings.append(.init(model: model, key: key + ".bias", isBias: true)) }
    return model
  }
  func norm(_ x: Model.IO, _ key: String, axis: Int = 1, channels: Int = 3584) -> Model.IO {
    let shape: TensorShapeFormat = axis == 3 ? .NHWC(1, 1, 1, channels) : .WC(1, channels)
    let gain = Parameter<T>(.GPU(device), shape, name: key.replacingOccurrences(of: ".", with: "_"))
    bindings.append(.init(model: gain, key: key + ".weight", parameter: true))
    // Diffusers rounds the normalized value before the learned RMS gain.
    return RMSNorm(epsilon: 1e-5, axis: [axis], elementwiseAffine: false)(x.to(.Float32))
      .to(T.dataType) .* gain()
  }
  func attention(
    _ query: Model.IO, _ context: Model.IO, _ key: String,
    queries: Int, keys: Int, rot: Model.IO? = nil
  ) -> Model.IO {
    let q = dense(key + ".to_q", d)(query).reshaped(.NHWC(1, queries, 28, 128))
    let k = dense(key + ".to_k", d)(context).reshaped(.NHWC(1, keys, 28, 128))
    let v = dense(key + ".to_v", d)(context).reshaped(.NHWC(1, keys, 28, 128))
    var normalizedQ = norm(q, key + ".norm_q", axis: 3, channels: 128)
    var normalizedK = norm(k, key + ".norm_k", axis: 3, channels: 128)
    if let rot {
      normalizedQ = Functional.cmul(left: normalizedQ.to(.Float32), right: rot).to(T.dataType)
      normalizedK = Functional.cmul(left: normalizedK.to(.Float32), right: rot).to(T.dataType)
    }
    return dense(key + ".to_out.0", d)(
      fiboAttention(type, q: normalizedQ, k: normalizedK, v: v, queries: queries, keys: keys))
  }
  func feedForward(_ x: Model.IO, _ key: String) -> Model.IO {
    let w1 = dense(key + ".w1", FiboDiTConfig.intermediate)
    let w3 = dense(key + ".w3", FiboDiTConfig.intermediate)
    let w2 = dense(key + ".w2", d)
    return w2(fiboSiLU(w1(x), type) .* w3(x))
  }
  let time0 = dense("t_embedder.mlp.0", 1024, bias: true)
  let time2 = dense("t_embedder.mlp.2", 256, bias: true)
  let embeddedTime = time2(fiboSiLU(time0(time), type))
  tap(embeddedTime, "time")
  func block(
    _ x: Model.IO, _ key: String, count: Int, rot: Model.IO,
    selected: Model.IO, context: Model.IO? = nil
  ) -> Model.IO {
    let modulation = dense(key + ".adaLN_modulation.0", d * 4, bias: true)(embeddedTime)
    func mod(_ i: Int, gate: Bool) -> Model.IO {
      let raw = fiboColumns(modulation, start: i * d, count: d, rows: 2, width: 4 * d)
      let rounded = (gate ? raw.to(.Float32).tanh() : 1 + raw.to(.Float32)).to(T.dataType)
      return Functional.indexSelect(input: rounded.to(.Float32), index: selected).to(T.dataType)
    }
    let u = norm(x, key + ".attention_norm1") .* mod(0, gate: false)
    let attended = attention(u, u, key + ".attention", queries: count, keys: count, rot: rot)
    var out = x + mod(1, gate: true) .* norm(attended, key + ".attention_norm2")
    if let context {
      let cross = attention(
        norm(out, key + ".cross_attn_norm1"), context,
        key + ".cross_attention", queries: count, keys: textLength)
      let gate = Parameter<T>(
        .GPU(device), .C(1), name: key.replacingOccurrences(of: ".", with: "_") + "_cross_gate")
      bindings.append(.init(model: gate, key: key + ".cross_attn_gate", parameter: true))
      out = out + gate().to(.Float32).tanh().to(T.dataType)
        .* norm(cross, key + ".cross_attn_norm2")
    }
    return out + mod(3, gate: true)
      .* norm(
        feedForward(norm(out, key + ".ffn_norm1") .* mod(2, gate: false), key + ".feed_forward"),
        key + ".ffn_norm2")
  }
  var image = dense("x_embedder", d, bias: true)(x)
  tap(image, "patch_embedding")
  for i in 0..<FiboDiTConfig.refinerLayers {
    image = block(
      image, "noise_refiner.\(i)", count: visualLength, rot: imageRotary, selected: refinerSelector)
    tap(image, "refiner_\(i)")
  }
  let queries = Parameter<T>(.GPU(device), .WC(FiboDiTConfig.maxGist, d), name: "gist_queries")
  bindings.append(.init(model: queries, key: "perceiver.latent_queries", parameter: true))
  var gist = fiboRows(queries(), start: 0, count: gistCount, width: d)
  for i in 0..<FiboDiTConfig.perceiverLayers {
    let key = "perceiver.layers.\(i)"
    gist =
      gist
      + attention(
        norm(gist, key + ".cross_attn_norm"), contexts[0], key + ".cross_attention",
        queries: gistCount, keys: textLength)
    let u = norm(gist, key + ".self_attn_norm")
    gist = gist + attention(u, u, key + ".self_attention", queries: gistCount, keys: gistCount)
    gist = gist + feedForward(norm(gist, key + ".ffn_norm"), key + ".ffn")
    tap(gist, "perceiver_\(i)")
  }
  gist = norm(gist, "perceiver.norm_out")
  tap(gist, "gist")
  var out = Functional.concat(axis: 0, gist, image)
  for i in 0..<mainLayers {
    let context = FiboDiTConfig.injectionLayers.firstIndex(of: i).map { contexts[$0 + 1] }
    out = block(
      out, "layers.\(i)", count: length, rot: rotary, selected: selector, context: context)
    tap(out, "layer_\(i)")
  }
  let finalTime = fiboRows(embeddedTime, start: 0, count: 1, width: 256)
  let finalScale = dense("final_layer.adaLN_modulation.1", d, bias: true)(fiboSiLU(finalTime, type))
  let finalNorm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let selected = fiboRows(out, start: gistCount, count: imageLength, width: d)
  let normalized = finalNorm(selected.to(.Float32)).to(T.dataType)
  let modulated = normalized .* (1 + finalScale.to(.Float32)).to(T.dataType)
  let output = dense("final_layer.linear", 128, bias: true)(modulated)
  outputs.append(output)
  names.append("output")
  return (
    Model([x] + contexts + [imageRotary, rotary, time, refinerSelector, selector], outputs),
    bindings, names
  )
}

func fiboTime(_ timestep: Float) -> Tensor<Float> {
  var result = Tensor<Float>(.CPU, .WC(2, 256))
  for row in 0..<2 {
    for band in 0..<128 {
      let phase = (row == 0 ? timestep * 1000 : 0) * exp(-log(Float(10000)) * Float(band) / 128)
      result[row, band] = cos(phase)
      result[row, band + 128] = sin(phase)
    }
  }
  return result
}

func fiboGistCount(_ textLength: Int) -> Int {
  let ratio = min(1.0, max(0.0, Double(textLength - 256) / Double(2048 - 256)))
  return min(512, max(256, Int(ceil((256 + ratio * 256) / 64)) * 64))
}

func fiboRotary(_ coordinates: [[Int]]) -> Tensor<Float> {
  var result = Tensor<Float>(.CPU, .NHWC(1, coordinates.count, 1, 128))
  for (token, coordinate) in coordinates.enumerated() {
    var offset = 0
    for (axis, dim) in FiboDiTConfig.ropeAxes.enumerated() {
      for band in 0..<(dim / 2) {
        // Upstream computes FP64 frequencies, then FP32 complex polar values.
        let angle = Float(
          Double(coordinate[axis]) / pow(FiboDiTConfig.ropeTheta, Double(2 * band) / Double(dim)))
        result[0, token, 0, offset + band * 2] = cos(angle)
        result[0, token, 0, offset + band * 2 + 1] = sin(angle)
      }
      offset += dim
    }
  }
  return result
}
