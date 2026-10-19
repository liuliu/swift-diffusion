import Foundation
import NNC

enum MOSSConfig {
  static let width = 3_072
  static let intermediate = 8_192
  static let layers = 42
  static let textLayers = 2
  static let heads = 24
  static let headDim = 128
  static let latentChannels = 128
  static let textWidth = 4_096
  static let timeWidth = 256
  static let ropeAxes = [32, 32, 32, 32]
  static let ropeBase: Float = 10_000
  static let epsilon: Float = 1e-6
}

func mossSiTU<T: TensorNumeric>(_ hidden: Model.IO, _ gate: Model.IO, _ type: T.Type) -> Model.IO {
  // Preserve each released PyTorch rounding boundary, including both smooth caps.
  let capGate =
    ((gate.to(.Float32) * 0.25).to(T.dataType).to(.Float32).tanh()
    .to(T.dataType).to(.Float32) * 4).to(T.dataType)
  let sigmoid = gate.to(.Float32).sigmoid().to(T.dataType)
  let capHidden =
    ((hidden.to(.Float32) * (1.0 / 25)).to(T.dataType).to(.Float32).tanh()
    .to(T.dataType).to(.Float32) * 25).to(T.dataType)
  return (capGate .* sigmoid) .* capHidden
}

func MOSSDiT<T: TensorNumeric>(
  _ type: T.Type, textLength: Int, height: Int, width: Int, device: Int,
  skipLayers: Set<Int> = [], diagnostic: Bool = false
) -> (Model, [MOSSBinding], [String]) {
  let image = Input()
  let context = Input()
  let rotary = Input()
  let time = Input()
  let sineRotary = Input()
  let pixels = height * width
  let length = pixels + textLength
  var bindings = [MOSSBinding]()
  var outputs = [Model.IO]()
  var names = [String]()
  func tap(_ x: Model.IO, _ name: String) {
    if diagnostic {
      outputs.append(x)
      names.append(name)
    }
  }
  func dense(
    _ count: Int, _ key: String, bias: Bool = false, name: String,
    transform: String = "identity", heads: Int = 0, start: Int = 0, rows: Int = 0,
    keepFloat32: Bool = false
  ) -> Model {
    let model = Dense(count: count, noBias: !bias, name: name)
    let fp32Bias = bias && T.dataType == .BFloat16
    bindings.append(
      .init(
        model: model, key: key + ".weight", transform: transform,
        heads: heads, rowStart: start, rowCount: rows, float32: fp32Bias || keepFloat32))
    if bias {
      bindings.append(
        .init(
          model: model, key: key + ".bias", transform: transform,
          heads: heads, rowStart: start, rowCount: rows, bias: true,
          float32: fp32Bias || keepFloat32))
    }
    if keepFloat32 {
      let x = Input()
      return Model([x], [model(x.to(.Float32))])
    }
    if fp32Bias {
      // Match PyTorch's fused BF16 GEMM+bias epilogue with one output rounding.
      let x = Input()
      return Model([x], [model(x.to(.Float32)).to(T.dataType)])
    }
    return model
  }
  func norm(_ x: Model.IO, _ key: String, axis: Int, channels: Int, name: String) -> Model.IO {
    let shape: TensorShapeFormat = axis == 3 ? .NHWC(1, 1, 1, channels) : .WC(1, channels)
    let gain = Parameter<T>(.GPU(device), shape, name: name)
    bindings.append(
      .init(
        model: gain, key: key + ".weight",
        transform: axis == 3 ? "qk_norm" : "identity", parameter: true))
    let normalized = RMSNorm(epsilon: 1e-6, axis: [axis], elementwiseAffine: false)(x.to(.Float32))
      .to(T.dataType)
    return normalized .* gain()
  }
  func onePlus(_ x: Model.IO) -> Model.IO { (1 + x.to(.Float32)).to(T.dataType) }
  func attention(
    _ x: Model.IO, _ rot: Model.IO, _ sine: Model.IO?, _ prefix: String,
    count: Int, packed: Bool, name: String
  )
    -> Model.IO
  {
    let q: Model
    let k: Model
    let v: Model
    if packed {
      q = dense(
        3_072, prefix + ".qkv", bias: true, name: name + "_q",
        transform: "qk", heads: 24, start: 0, rows: 3_072)
      k = dense(
        3_072, prefix + ".qkv", bias: true, name: name + "_k",
        transform: "qk", heads: 24, start: 3_072, rows: 3_072)
      v = dense(
        3_072, prefix + ".qkv", bias: true, name: name + "_v",
        start: 6_144, rows: 3_072)
    } else {
      q = dense(3_072, prefix + ".proj.q", name: name + "_q", transform: "qk", heads: 24)
      k = dense(3_072, prefix + ".proj.k", name: name + "_k", transform: "qk", heads: 24)
      v = dense(3_072, prefix + ".proj.v", name: name + "_v")
    }
    let queries = mossApplyRotary(
      norm(
        q(x).reshaped([1, count, 24, 128]), prefix + ".q_norm", axis: 3, channels: 128,
        name: name + "_norm_q"),
      cosine: rot, sine: sine)
    let keys = mossApplyRotary(
      norm(
        k(x).reshaped([1, count, 24, 128]), prefix + ".k_norm", axis: 3, channels: 128,
        name: name + "_norm_k"),
      cosine: rot, sine: sine)
    let attended = mossAttention(
      type, q: queries, k: keys, v: v(x).reshaped([1, count, 24, 128]),
      length: count, heads: 24, kvHeads: 24, headDim: 128)
    return dense(
      3_072, prefix + (packed ? ".proj" : ".proj.o"), bias: packed,
      name: name + "_o")(attended)
  }

  let encodedImage = dense(3_072, "video_embedder", bias: true, name: "x_embedder")(image)
  tap(encodedImage, "image_embedding")
  var text = dense(3_072, "text_embedder.proj", bias: true, name: "context_embedder")(context)
  tap(text, "text_projection")
  let textRotary = rotary.reshaped(
    [1, textLength, 1, 128], offset: [0, pixels, 0, 0],
    strides: [length * 128, 128, 128, 1]
  ).contiguous()
  let textSineRotary =
    T.dataType == .Float32
    ? nil
    : sineRotary.reshaped(
      [1, textLength, 1, 128], offset: [0, pixels, 0, 0],
      strides: [length * 128, 128, 128, 1]
    ).contiguous()
  for index in 0..<2 {
    let p = "text_embedder.blocks.\(index)"
    text =
      text
      + attention(
        norm(
          text, p + ".norm1", axis: 1, channels: 3_072,
          name: "context_norm1"), textRotary, textSineRotary, p, count: textLength,
        packed: true, name: "context")
    let h = norm(text, p + ".norm2", axis: 1, channels: 3_072, name: "context_norm2")
    let gate = dense(8_192, p + ".ffn_gate", bias: true, name: "context_ffn_gate_proj")
    let up = dense(8_192, p + ".ffn_up", bias: true, name: "context_ffn_up_proj")
    text =
      text
      + dense(3_072, p + ".ffn_down", bias: true, name: "context_ffn_down_proj")(
        mossSiTU(up(h), gate(h), type))
    tap(text, "text_block_\(index)")
  }
  let time0 = dense(3_072, "time_embedding.0", bias: true, name: "time_embedder_0")
  let time2 = dense(
    3_072, "time_embedding.2", bias: true, name: "time_embedder_2",
    keepFloat32: true)
  let timeVector = time2(mossSiLU(time0(time), type))
  tap(timeVector, "time_embedding")
  let modulationInput = timeVector.to(T.dataType)
  // These six projections are shared by all 42 blocks. Source row order is
  // attention shift / scale / gate, followed by FFN shift / scale / gate.
  let modulation = (0..<6).map { index in
    dense(
      3_072, "shared_block_modulation", bias: true, name: "x_ada_ln_\(index)",
      start: index * 3_072, rows: 3_072)(modulationInput)
  }
  let headShift = dense(
    3_072, "video_head_modulation", bias: true, name: "ada_ln_1",
    start: 0, rows: 3_072)(modulationInput)
  let headScale = dense(
    3_072, "video_head_modulation", bias: true, name: "ada_ln_0",
    start: 3_072, rows: 3_072)(modulationInput)
  if diagnostic {
    tap(
      modulation.dropFirst().reduce(modulation[0]) { Functional.concat(axis: 0, $0, $1) },
      "modulation")
    tap(Functional.concat(axis: 0, headShift, headScale), "head_modulation")
  }
  let attentionScale = onePlus(modulation[1])
  let ffnScale = onePlus(modulation[4])
  var x = Functional.concat(axis: 0, encodedImage, text)
  for layer in 0..<42 {
    if skipLayers.contains(layer) { continue }
    let p = "layers.\(layer)"
    let h =
      norm(x, p + ".norm1", axis: 1, channels: 3_072, name: "x_norm1")
      .* attentionScale + modulation[0]
    x =
      x
      + attention(
        h, rotary, T.dataType == .Float32 ? nil : sineRotary,
        p, count: length, packed: false, name: "x") .* modulation[2]
    let f =
      norm(x, p + ".norm2", axis: 1, channels: 3_072, name: "x_norm2")
      .* ffnScale + modulation[3]
    let up = dense(
      8_192, p + ".proj.ffn_up_gate", name: "ffn_up_proj",
      start: 0, rows: 8_192)
    let gate = dense(
      8_192, p + ".proj.ffn_up_gate", name: "ffn_gate_proj",
      start: 8_192, rows: 8_192)
    x =
      x
      + dense(3_072, p + ".proj.ffn_down", name: "ffn_down_proj")(
        mossSiTU(up(f), gate(f), type))
      .* modulation[5]
    if [0, 21, 22, 41].contains(layer) { tap(x, "layer_\(layer)") }
  }
  let video = x.reshaped([pixels, 3_072], strides: [3_072, 1]).contiguous()
  let final =
    norm(video, "final_norm_video", axis: 1, channels: 3_072, name: "final_norm")
    .* onePlus(headScale) + headShift
  outputs.append(dense(128, "final_linear_video", name: "final_linear")(final))
  names.append("output")
  return (
    Model(
      [image, context, rotary] + (T.dataType == .Float32 ? [] : [sineRotary]) + [time], outputs),
    bindings, names
  )
}

func mossDiTRotary<T: TensorNumeric>(
  _ type: T.Type, height: Int, width: Int, textLength: Int
) -> Tensor<T> {
  let pixels = height * width
  let length = pixels + textLength
  var tensor = Tensor<Float>(.CPU, .NHWC(1, length, 1, 128))
  for token in 0..<length {
    let coordinates: [Float] =
      token < pixels
      ? [
        0, Float(textLength), Float(token / width) - Float(height - 1) / 2,
        Float(token % width) - Float(width - 1) / 2,
      ]
      : [0, Float(token - pixels), 0, 0]
    for axis in 0..<4 {
      for band in 0..<16 {
        let frequency: Float = 1 / Float(pow(10_000, Double(band) / 16))
        let angle = coordinates[axis] * frequency
        let pair = axis * 16 + band
        tensor[0, token, 0, 2 * pair] = cos(angle)
        tensor[0, token, 0, 2 * pair + 1] = sin(angle)
      }
    }
  }
  return Tensor<T>(from: tensor)
}

func mossTimeEmbedding(_ timestep: Float) -> Tensor<Float> {
  var tensor = Tensor<Float>(.CPU, .WC(1, 256))
  for band in 0..<128 {
    let angle = Double(timestep) * pow(10_000, -Double(band) / 128)
    tensor[0, band] = Float(cos(angle))
    tensor[0, band + 128] = Float(sin(angle))
  }
  return tensor
}
