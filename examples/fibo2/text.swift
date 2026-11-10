import Foundation
import NNC

enum FiboTextConfig {
  static let width = 2_560
  static let intermediate = 9_728
  static let layers = 36
  static let heads = 32
  static let kvHeads = 8
  static let headDim = 128
  static let vocabulary = 151_936
  static let ropeTheta: Double = 5_000_000
  static let bundles = [
    [9, 20, 31], [5, 16, 27], [7, 18, 29], [9, 20, 31], [11, 22, 33], [13, 24, 35],
  ]
  // HF hidden_states indices; zero-based decoder indices are one smaller.
  static let hiddenStates = [5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35]
}

// Qwen3-VL-4B graph, adapted from FLUX.3 with Fibo's intermediate feature bundles.
func FiboText<T: TensorNumeric>(
  _ type: T.Type, length: Int, device: Int, validLength: Int? = nil,
  completeEncoder: Bool = false, visual: [FiboVisualSpan] = []
) -> (Model, [FiboBinding], [String]) {
  let valid = validLength ?? length
  precondition(valid > 0 && valid <= length)
  let tokens = Input()
  let rotary = Input()
  let mask = Input()
  let embedding = Embedding(
    T.self, vocabularySize: 151_936, embeddingSize: 2_560,
    name: "tok_embeddings")
  var bindings: [FiboBinding] = [
    .init(
      model: embedding,
      key: "model.language_model.embed_tokens.weight", transform: "embedding")
  ]
  func norm(_ key: String, channels: Int, axis: Int = 1, transform: String = "identity") -> Model {
    let x = Input()
    let shape: TensorShapeFormat = axis == 3 ? .NHWC(1, 1, 1, channels) : .WC(1, channels)
    let gain = Parameter<T>(.GPU(device), shape, name: key.replacingOccurrences(of: ".", with: "_"))
    bindings.append(.init(model: gain, key: key + ".weight", transform: transform, parameter: true))
    return Model(
      [x],
      [
        RMSNorm(epsilon: 1e-6, axis: [axis], elementwiseAffine: false)(x.to(.Float32))
          .copied().to(T.dataType) .* gain()
      ])
  }
  func rotate(_ x: Model.IO, _ rot: Model.IO, heads: Int) -> Model.IO {
    if T.dataType == .Float32 { return Functional.cmul(left: x, right: rot) }
    func split(_ value: Model.IO, heads: Int, imaginary: Bool) -> Model.IO {
      value.reshaped(
        [length, heads, 64, 1], offset: [0, 0, 0, imaginary ? 1 : 0],
        strides: [heads * 128, 128, 2, 1]
      ).contiguous()
    }
    let real = split(x, heads: heads, imaginary: false)
    let imaginary = split(x, heads: heads, imaginary: true)
    let cosine = split(rot, heads: 1, imaginary: false)
    let sine = split(rot, heads: 1, imaginary: true)
    // Qwen's language RoPE rounds each BF16 product before the sum.
    return Functional.concat(
      axis: 3, real .* cosine - imaginary .* sine,
      real .* sine + imaginary .* cosine
    ).reshaped([1, length, heads, 128])
  }
  var out = embedding(tokens)
  let vision = (0..<4).map { _ in Input() }
  func inject(_ text: Model.IO, _ features: Model.IO, add: Bool) -> Model.IO {
    var pieces = [Model.IO]()
    var cursor = 0
    var offset = 0
    func slice(_ value: Model.IO, _ start: Int, _ count: Int) -> Model.IO {
      value.reshaped([count, 2560], offset: [start, 0], strides: [2560, 1]).contiguous()
    }
    for span in visual {
      if span.start > cursor { pieces.append(slice(text, cursor, span.start - cursor)) }
      let value = slice(features, offset, span.length)
      pieces.append(add ? slice(text, span.start, span.length) + value : value)
      cursor = span.start + span.length
      offset += span.length
    }
    if cursor < length { pieces.append(slice(text, cursor, length - cursor)) }
    return pieces.dropFirst().reduce(pieces[0]) { Functional.concat(axis: 0, $0, $1) }
  }
  if !visual.isEmpty { out = inject(out, vision[0], add: false) }
  var outputs = [Model.IO]()
  var taps = [Int: Model.IO]()
  var names = [String]()
  let layers = completeEncoder ? FiboTextConfig.layers : FiboTextConfig.hiddenStates.last!
  for layer in 0..<layers {
    let prefix = "model.language_model.layers.\(layer)"
    let x = Input()
    let rot = Input()
    let attentionMask = Input()
    let norm1 = norm(prefix + ".input_layernorm", channels: 2560)
    let u = norm1(x).to(T.dataType)
    let q = Dense(count: 4_096, noBias: true, name: "q_proj")
    let k = Dense(count: 1_024, noBias: true, name: "k_proj")
    let v = Dense(count: 1_024, noBias: true, name: "v_proj")
    let qNorm = norm(prefix + ".self_attn.q_norm", channels: 128, axis: 3, transform: "qk_norm")
    let kNorm = norm(prefix + ".self_attn.k_norm", channels: 128, axis: 3, transform: "qk_norm")
    let queries = rotate(qNorm(q(u).reshaped([1, length, 32, 128])), rot, heads: 32)
    let keys = rotate(kNorm(k(u).reshaped([1, length, 8, 128])), rot, heads: 8)
    let values = v(u).reshaped([1, length, 8, 128])
    // Keep padded query outputs, while excluding padding from keys and values.
    let attended: Model.IO
    if T.dataType == .Float32 {
      let groupedQ = queries.reshaped([length, 8, 4, 128]).transposed(0, 1)
        .reshaped([8, length * 4, 128])
      let groupedK = keys.reshaped([length, 8, 128]).transposed(0, 1)
      let logits =
        Matmul(transposeB: (1, 2))(groupedQ, groupedK)
        .reshaped([8, length, 4, length]).transposed(1, 2) * (1 / Float(128).squareRoot())
      let probabilities = (logits + attentionMask)
        .reshaped([32 * length, length]).softmax()
        .reshaped([8, 4, length, length]).transposed(1, 2)
        .reshaped([8, length * 4, length])
      attended = Matmul()(probabilities, values.reshaped([length, 8, 128]).transposed(0, 1))
        .reshaped([8, length, 4, 128]).transposed(0, 1).reshaped([length, 4_096])
    } else {
      // CUDA FlashAttention currently ignores a supplied arbitrary mask.
      // Causal real-token queries + noncausal padded queries implement the
      // right-padding mask exactly, using Krea 2's causal SDPA path.
      func slice(_ x: Model.IO, _ offset: Int, _ count: Int, _ heads: Int) -> Model.IO {
        x.reshaped(
          [1, count, heads, 128], offset: [0, offset, 0, 0],
          strides: [length * heads * 128, heads * 128, 128, 1]
        ).contiguous()
      }
      let realKeys = valid == length ? keys : slice(keys, 0, valid, 8)
      let realValues = valid == length ? values : slice(values, 0, valid, 8)
      let realQueries = valid == length ? queries : slice(queries, 0, valid, 32)
      let real = ScaledDotProductAttention(
        scale: 1 / Float(128).squareRoot(), isCausal: true)(realQueries, realKeys, realValues)
      if valid == length {
        attended = real.reshaped([length, 4_096])
      } else {
        let padding = ScaledDotProductAttention(scale: 1 / Float(128).squareRoot())(
          slice(queries, valid, length - valid, 32), realKeys, realValues)
        attended = Functional.concat(axis: 1, real, padding).reshaped([length, 4_096])
      }
    }
    let projection = Dense(count: 2_560, noBias: true, name: "out_proj")
    let residual = x + projection(attended).to(of: x)
    let norm2 = norm(prefix + ".post_attention_layernorm", channels: 2560)
    let gate = Dense(count: 9_728, noBias: true, name: "mlp_gate_proj")
    let up = Dense(count: 9_728, noBias: true, name: "mlp_up_proj")
    let down = Dense(count: 2_560, noBias: true, name: "mlp_down_proj")
    let ffInput = Input()
    let ff = Model([ffInput], [down(up(ffInput) .* fiboSiLU(gate(ffInput), type))], name: "mlp")
    let block = Model(
      [x, rot] + (T.dataType == .Float32 ? [attentionMask] : []),
      [residual + ff(norm2(residual).to(T.dataType)).to(of: residual)])
    out = T.dataType == .Float32 ? block(out, rotary, mask) : block(out, rotary)
    if !visual.isEmpty && layer < 3 { out = inject(out, vision[layer + 1], add: true) }
    bindings += [
      .init(model: q, key: prefix + ".self_attn.q_proj.weight", transform: "qk", heads: 32),
      .init(model: k, key: prefix + ".self_attn.k_proj.weight", transform: "qk", heads: 8),
      .init(model: v, key: prefix + ".self_attn.v_proj.weight"),
      .init(model: projection, key: prefix + ".self_attn.o_proj.weight"),
      .init(model: gate, key: prefix + ".mlp.gate_proj.weight"),
      .init(model: up, key: prefix + ".mlp.up_proj.weight"),
      .init(model: down, key: prefix + ".mlp.down_proj.weight"),
    ]
    if FiboTextConfig.hiddenStates.contains(layer + 1) {
      outputs.append(out)
      // Keep feature normalization distinct from the next block's RMSNorm:
      // concatenation may otherwise alias that shared FP32 buffer as a
      // strided view, which CUDA's BF16 conversion cannot read.
      taps[layer + 1] = RMSNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)(
        out.to(.Float32).reshaped([length, 1, 2560])
      ).reshaped([length, 2560])
      names.append("hidden_\(layer + 1)")
    }
  }
  // Retain all 36 layers + final norm when building a reusable full encoder.
  if completeEncoder {
    let finalNorm = norm("model.language_model.norm", channels: 2560)
    outputs.append(finalNorm(out))
    names.append("final_norm")
  }
  for (i, bundle) in FiboTextConfig.bundles.enumerated() {
    // Casting after concatenation is equivalent to casting each independent
    // feature first, and keeps CUDA's conversion output contiguous.
    outputs.append(
      Functional.concat(axis: 1, taps[bundle[0]]!, taps[bundle[1]]!, taps[bundle[2]]!).to(
        T.dataType))
    names.append("bundle_\(i)")
  }
  return (
    Model(
      [tokens, rotary] + (T.dataType == .Float32 ? [mask] : []) + (visual.isEmpty ? [] : vision),
      outputs), bindings, names
  )
}

func fiboTextRotary<T: TensorNumeric>(_ type: T.Type, positions: Tensor<Int32>) -> Tensor<T> {
  let length = positions.shape[0]
  var out = Tensor<Float>(.CPU, .NHWC(1, length, 1, 128))
  for token in 0..<length {
    for pair in 0..<64 {
      let angle = Double(positions[token]) / pow(FiboTextConfig.ropeTheta, Double(pair * 2) / 128)
      out[0, token, 0, pair * 2] = Float(cos(angle))
      out[0, token, 0, pair * 2 + 1] = Float(sin(angle))
    }
  }
  return fiboRoundedInput(type, out)
}

struct FiboVisualSpan {
  let start: Int
  let height: Int
  let width: Int
  var length: Int { height * width }
}

func fiboMultimodalPositions(length: Int, visual: [FiboVisualSpan]) -> Tensor<Int32> {
  var result = Tensor<Int32>(.CPU, .NC(3, length))
  var cursor = 0
  var position = 0
  for span in visual {
    for token in cursor..<span.start {
      for axis in 0..<3 { result[axis, token] = Int32(position + token - cursor) }
    }
    position += span.start - cursor
    for local in 0..<span.length {
      result[0, span.start + local] = Int32(position)
      result[1, span.start + local] = Int32(position + local / span.width)
      result[2, span.start + local] = Int32(position + local % span.width)
    }
    cursor = span.start + span.length
    position += max(span.height, span.width)
  }
  for token in cursor..<length {
    for axis in 0..<3 { result[axis, token] = Int32(position + token - cursor) }
  }
  return result
}

func fiboMultimodalRotary<T: TensorNumeric>(_ type: T.Type, _ positions: Tensor<Int32>) -> Tensor<T>
{
  let length = positions.shape[1]
  var result = Tensor<Float>(.CPU, .NHWC(1, length, 1, 128))
  for token in 0..<length {
    for pair in 0..<64 {
      let axis = pair < 60 ? pair % 3 : 0
      let phase =
        Float(positions[axis, token]) / pow(Float(FiboTextConfig.ropeTheta), Float(2 * pair) / 128)
      result[0, token, 0, 2 * pair] = cos(phase)
      result[0, token, 0, 2 * pair + 1] = sin(phase)
    }
  }
  return fiboRoundedInput(type, result)
}
