import Foundation
import NNC

enum Q21DiTConfig {
  static let width = 4_096
  static let heads = 32
  static let headDim = 128
  static let layers = 32
  static let intermediate = 12_288
  static let latentChannels = 64
  static let contextWidth = 4_096
  static let epsilon: Float = 1e-6
  static let ropeAxes = [16, 56, 56]
}

struct Q21Binding {
  let model: Model
  let key: String
  var bias = false
  var zeroCentered = false
  var parameter = false
  var chunk: Int? = nil
}

struct Q21Segment {
  let image: Bool
  let length: Int
  let sourceOffset: Int
  var height = 0
  var width = 0
}

func q21LayoutMask(_ segments: [Q21Segment]) -> Tensor<Float> {
  let count = segments.reduce(0) { $0 + $1.length }
  var mask = Tensor<Float>(.CPU, .NCHW(1, 1, count, count))
  _ = mask.withUnsafeMutableBytes { $0.initializeMemory(as: UInt8.self, repeating: 0) }
  var start = 0
  for segment in segments {
    let end = start + segment.length
    for row in start..<end {
      let allowed = segment.image ? end : row + 1
      if allowed < count { for column in allowed..<count { mask[0, 0, row, column] = -.infinity } }
    }
    start = end
  }
  return mask
}

func q21LayoutRotary(_ segments: [Q21Segment]) -> Tensor<Float> {
  let count = segments.reduce(0) { $0 + $1.length }
  var result = Tensor<Float>(.CPU, .NHWC(1, count, 1, 128))
  var token = 0
  var position = 0
  for segment in segments {
    for local in 0..<segment.length {
      let positions =
        segment.image
        ? [
          position, local / segment.width - (segment.height - segment.height / 2),
          local % segment.width - (segment.width - segment.width / 2),
        ]
        : [position + local, position + local, position + local]
      var offset = 0
      for (axis, dimension) in Q21DiTConfig.ropeAxes.enumerated() {
        for pair in 0..<(dimension / 2) {
          let frequency = pow(Float(10_000), -Float(2 * pair) / Float(dimension))
          let phase = Float(positions[axis]) * frequency
          result[0, token, 0, offset + 2 * pair] = cos(phase)
          result[0, token, 0, offset + 2 * pair + 1] = sin(phase)
        }
        offset += dimension
      }
      token += 1
    }
    position += segment.image ? max(segment.height, segment.width) : segment.length
  }
  return result
}

func q21AttentionMask(text: Int, pixels: Int) -> Tensor<Float> {
  let count = text + pixels
  var mask = Tensor<Float>(.CPU, .NCHW(1, 1, count, count))
  _ = mask.withUnsafeMutableBytes { $0.initializeMemory(as: UInt8.self, repeating: 0) }
  for row in 0..<text {
    for column in (row + 1)..<count { mask[0, 0, row, column] = -.infinity }
  }
  return mask
}

// Generation layout: causal text followed by one bidirectional target image.
// RoPE stays real/interleaved, with multiplication performed in FP32.
func q21Rotary(text: Int, height: Int, width: Int) -> Tensor<Float> {
  let count = text + height * width
  var result = Tensor<Float>(.CPU, .NHWC(1, count, 1, 128))
  for token in 0..<count {
    let positions: [Int]
    if token < text {
      positions = [token, token, token]
    } else {
      let pixel = token - text
      positions = [
        text, pixel / width - (height - height / 2), pixel % width - (width - width / 2),
      ]
    }
    var offset = 0
    for (axis, dimension) in Q21DiTConfig.ropeAxes.enumerated() {
      for pair in 0..<(dimension / 2) {
        let frequency = pow(Float(10_000), -Float(2 * pair) / Float(dimension))
        let phase = Float(positions[axis]) * frequency
        result[0, token, 0, offset + 2 * pair] = cos(phase)
        result[0, token, 0, offset + 2 * pair + 1] = sin(phase)
      }
      offset += dimension
    }
  }
  return result
}

func q21TimeEmbedding(_ timestep: Float) -> Tensor<Float> {
  var result = Tensor<Float>(.CPU, .NC(2, 256))
  for row in 0..<2 {
    for i in 0..<128 {
      let frequency = exp(-log(Float(10_000)) * Float(i) / 128)
      let phase = (row == 0 ? timestep * 1_000 : 0) * frequency
      result[row, i] = cos(phase)
      result[row, i + 128] = sin(phase)
    }
  }
  return result
}

func Q21DiT<T: TensorNumeric>(
  _ type: T.Type, text: Int, height: Int, width: Int, layers: Int = Q21DiTConfig.layers,
  device: Int = 1, layout: [Q21Segment]? = nil,
  mixedPrecision: Bool = false
) -> (Model, [Q21Binding]) {
  let image = Input()
  let context = Input()
  let time = Input()
  let rotary = Input()
  let mask = Input()
  let pixels = height * width
  let count = layout?.reduce(0) { $0 + $1.length } ?? (text + pixels)
  let prefixLength = count - pixels
  precondition(!mixedPrecision || T.dataType == .Float16)
  precondition(
    layout == nil || T.dataType == .Float32, "Edit DiT currently validates FP32 execution")
  let d = Q21DiTConfig.width
  let heads = Q21DiTConfig.heads
  let k = Q21DiTConfig.headDim
  var bindings = [Q21Binding]()
  func dense(_ key: String, _ output: Int, name: String, chunk: Int? = nil) -> Model {
    let model = Dense(
      count: output, noBias: true, name: name)
    bindings.append(.init(model: model, key: key, chunk: chunk))
    return model
  }
  func rms(_ key: String, axis: Int, name: String, zeroCentered: Bool = false) -> Model {
    let model = RMSNorm(
      epsilon: Q21DiTConfig.epsilon, axis: [axis],
      name: name)
    bindings.append(.init(model: model, key: key, zeroCentered: zeroCentered))
    return model
  }
  func norm(_ x: Model.IO) -> Model.IO {
    LayerNorm(epsilon: Q21DiTConfig.epsilon, axis: [1], elementwiseAffine: false)(x)
  }
  func swish(_ x: Model.IO) -> Model.IO { x.to(.Float32).swish().to(T.dataType) }
  func attentionNorm(_ x: Model.IO, key: String, name: String) -> Model.IO {
    if T.dataType == .Float32 { return rms(key, axis: 3, name: name)(x) }
    // Diffusers rounds the normalized vector BEFORE applying the learned gain.
    let gain = Parameter<T>(
      .GPU(device), .NHWC(1, 1, 1, k), name: name)
    bindings.append(.init(model: gain, key: key, parameter: true))
    let normalized = RMSNorm(epsilon: Q21DiTConfig.epsilon, axis: [3], elementwiseAffine: false)(
      x.to(.Float32)
    ).to(T.dataType)
    return normalized .* gain()
  }
  func slice(_ x: Model.IO, start: Int, length: Int, columns: Int = Q21DiTConfig.width) -> Model.IO
  {
    x.reshaped([length, columns], offset: [start, 0], strides: [columns, 1]).contiguous()
  }
  // Modulation has two rows: noisy target at row 0, invariant prefix at row 1.
  func modulate(_ x: Model.IO, _ table: Model.IO, columns: Int = Q21DiTConfig.width) -> Model.IO {
    let prefix =
      slice(x, start: 0, length: prefixLength, columns: columns)
      .* slice(table, start: 1, length: 1, columns: columns)
    let target =
      slice(x, start: prefixLength, length: pixels, columns: columns)
      .* slice(table, start: 0, length: 1, columns: columns)
    return Functional.concat(axis: 0, prefix, target)
  }
  let textNorm = rms("txt_in.text_norm", axis: 1, name: "context_norm", zeroCentered: true)
  // Effective zero-centered weights are kept in FP32, as in the reference.
  let projectionType = T.dataType
  let normalizedText = textNorm(context.to(.Float32)).to(projectionType)
  let textIn = dense("txt_in.in_layer", d, name: "context_embedder_0")
  let textOut = dense("txt_in.out_layer", d, name: "context_embedder_1")
  let encodedText = textOut(GELU(approximate: .tanh)(textIn(normalizedText)))
  let imageIn = dense("img_in", d, name: "x_embedder")
  let encodedImage = imageIn(image.to(projectionType))
  var x: Model.IO
  if let layout {
    let pieces = layout.map { part in
      slice(part.image ? encodedImage : encodedText, start: part.sourceOffset, length: part.length)
    }
    x = pieces[0]
    for piece in pieces.dropFirst() { x = Functional.concat(axis: 0, x, piece) }
  } else {
    x = Functional.concat(axis: 0, encodedText, encodedImage)
  }
  if mixedPrecision { x = x.to(.Float32) }
  let time0 = dense("time_text_embed.timestep_embedder.linear_1", d, name: "t_embedder_0")
  let time2 = dense("time_text_embed.timestep_embedder.linear_2", d, name: "t_embedder_1")
  let temb = time2(time0(time.to(projectionType)).to(.Float32).swish().to(projectionType))
  let modulationInput = temb.to(.Float32).swish().to(projectionType)
  // Shared across all 32 blocks; each Dense loads its own rows of modulation.1.weight.
  let mods = (0..<4).map { (chunk: Int) in
    dense("modulation.1", d, name: "x_ada_ln_\(chunk)", chunk: chunk)(modulationInput)
  }
  let residualType: DataType = mixedPrecision ? .Float32 : T.dataType
  let scale1 = (1 + mods[0].to(.Float32)).to(residualType)
  let gate1 = mods[1].to(.Float32).tanh().to(residualType)
  let scale2 = (1 + mods[2].to(.Float32)).to(residualType)
  let gate2 = mods[3].to(.Float32).tanh().to(residualType)
  for layer in 0..<layers {
    let prefix = "transformer_blocks.\(layer)"
    let attentionInput = modulate(norm(x), scale1).to(T.dataType)
    let qProj = dense("\(prefix).attn.to_q", d, name: "x_q")
    let kProj = dense("\(prefix).attn.to_k", d, name: "x_k")
    let vProj = dense("\(prefix).attn.to_v", d, name: "x_v")
    let q = Functional.cmul(
      left: attentionNorm(
        qProj(attentionInput).reshaped(.NHWC(1, count, heads, k)),
        key: "\(prefix).attn.norm_q", name: "x_norm_q"
      ).to(.Float32),
      right: rotary
    ).to(T.dataType)
    let keys = Functional.cmul(
      left: attentionNorm(
        kProj(attentionInput).reshaped(.NHWC(1, count, heads, k)),
        key: "\(prefix).attn.norm_k", name: "x_norm_k"
      ).to(.Float32),
      right: rotary
    ).to(T.dataType)
    let values = vProj(attentionInput).reshaped(
      .NHWC(1, count, heads, k))
    let attended: Model.IO
    if T.dataType == .Float32 {
      let scores =
        Matmul(transposeB: (2, 3))(q.transposed(1, 2), keys.transposed(1, 2))
        * (1 / Float(k).squareRoot())
      let probabilities = (scores + mask).reshaped([heads * count, count]).softmax().reshaped([
        1, heads, count, count,
      ])
      attended = Matmul()(probabilities, values.transposed(1, 2)).transposed(1, 2).reshaped([
        count, d,
      ])
    } else {
      func segment(_ tensor: Model.IO, start: Int, length: Int) -> Model.IO {
        tensor.reshaped(
          .NHWC(1, length, heads, k), offset: [0, start, 0, 0], strides: [count * d, d, k, 1]
        ).contiguous()
      }
      let prefixOut = ScaledDotProductAttention(scale: 1 / Float(k).squareRoot(), isCausal: true)(
        segment(q, start: 0, length: text), segment(keys, start: 0, length: text),
        segment(values, start: 0, length: text))
      let targetOut = ScaledDotProductAttention(scale: 1 / Float(k).squareRoot())(
        segment(q, start: text, length: pixels), keys, values)
      attended = Functional.concat(
        axis: 0, prefixOut.reshaped([text, d]), targetOut.reshaped([pixels, d]))
    }
    let attentionOut = dense("\(prefix).attn.to_out.0", d, name: "x_o")(attended).to(
      residualType)
    x = x + modulate(attentionOut, gate1)
    let ffnInput = modulate(norm(x), scale2).to(T.dataType)
    let up = dense("\(prefix).img_mlp.proj", Q21DiTConfig.intermediate, name: "ffn_up_proj")
    let gate = dense(
      "\(prefix).img_mlp.gate_layer", Q21DiTConfig.intermediate, name: "ffn_gate_proj")
    let down = dense("\(prefix).img_mlp.out", d, name: "ffn_down_proj")
    let upValue = up(ffnInput)
    let gateValue = gate(ffnInput)
    // Full 40-step profiles at 512px and 1024px require no FFN scaling.
    let product = swish(gateValue) .* upValue
    let branch = down(product).to(residualType)
    x = x + modulate(branch, gate2)

  }
  let finalScale =
    (1
    + dense("norm_out.linear", d, name: "ada_ln_0")(temb.to(.Float32).swish().to(projectionType))
    .to(.Float32)).to(
      residualType)
  let out = dense("proj_out", 64, name: "linear")(modulate(norm(x), finalScale).to(projectionType))
  let target = slice(out, start: prefixLength, length: pixels, columns: 64)
  return (
    Model(
      [image, context, time, rotary] + (T.dataType == .Float32 ? [mask] : []),
      [mixedPrecision ? target.to(.Float32) : target]), bindings
  )
}
