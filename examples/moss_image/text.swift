import Foundation
import NNC

// Qwen3.5-9B architecture and parameter names follow Draw Things SwiftLLM's
// Models/Qwen3_5.swift. This encoder computes all tokens without generation caches.
enum MOSSTextConfig {
  static let width = 4_096
  static let intermediate = 12_288
  static let vocabulary = 248_320
  static let layers = 32
  static let fullAttentionInterval = 4
  static let heads = 16
  static let kvHeads = 4
  static let headDim = 256
  static let rotaryDim = 64
  static let ropeTheta: Double = 10_000_000
  static let linearKeyHeads = 16
  static let linearValueHeads = 32
  static let linearHeadDim = 128
  static let convDim = 8_192
  static let convKernel = 4
  static let epsilon: Float = 1e-6
  static let diagnosticLayers = [0, 1, 2, 3, 7, 15, 23, 31]
}

// CUDA has no GatedDelta backend yet. A 64-token chunk uses the upstream delta
// rule's unit-lower-triangular inverse. Blockwise inversion takes log2(chunkSize)
// stages with ordinary CUDA GEMMs and FP32 state; matrix powers are unstable
// when strongly correlated keys make the polynomial terms cancel.
func mossGatedDelta(
  q: Model.IO, k: Model.IO, v: Model.IO, logDecay: Model.IO, beta: Model.IO,
  initialState: Model.IO, lowerTriangle: Model.IO, length: Int
) -> Model.IO {
  var current = initialState.transposed(1, 2)
  var outputs = [Model.IO]()
  for start in stride(from: 0, to: length, by: 64) {
    let count = min(64, length - start)
    func chunk(_ x: Model.IO, width: Int) -> Model.IO {
      x.reshaped(
        [32, count, width], offset: [0, start, 0],
        strides: [width, 32 * width, 1]
      ).contiguous()
    }
    let query = chunk(q, width: 128)
    let key = chunk(k, width: 128)
    let value = chunk(v, width: 128)
    let b = chunk(beta, width: 1)
    let lower = lowerTriangle.reshaped([32, count, count], strides: [4_096, 64, 1]).contiguous()
    let identity = lower .* lower.transposed(1, 2)
    let cumulative = Matmul()(lower, chunk(logDecay, width: 1))
    let differences = (cumulative - cumulative.transposed(1, 2)) .* lower
    let decay = differences.exp() .* lower
    let keyBeta = key .* b
    var inverse = identity
    if count > 1 {
      let a = (Matmul(transposeB: (1, 2))(keyBeta, key) .* decay) .* (lower - identity) * -1
      var span = 2
      var maskIndex = 1
      while span / 2 < count {
        let blockMask = lowerTriangle.reshaped(
          [32, count, count], offset: [maskIndex * 32, 0, 0],
          strides: [4_096, 64, 1]
        ).contiguous()
        let cross = a .* blockMask
        inverse = inverse + Matmul()(Matmul()(inverse, cross), inverse)
        span *= 2
        maskIndex += 1
      }
    }
    let values = Matmul()(inverse, value .* b)
    let keyDecay = Matmul()(inverse, keyBeta .* cumulative.exp())
    let correctedValues = values - Matmul()(keyDecay, current)
    let attention = Matmul(transposeB: (1, 2))(query, key) .* decay
    let output = Matmul()(query .* cumulative.exp(), current) + Matmul()(attention, correctedValues)
    outputs.append(output.transposed(0, 1).contiguous())
    if start + count < length {
      let last = cumulative.reshaped(
        [32, 1, 1], offset: [0, count - 1, 0],
        strides: [count, 1, 1]
      ).contiguous()
      current = current .* last.exp()
        + Matmul(transposeA: (1, 2))(key .* (last - cumulative).exp(), correctedValues)
    }
  }
  var result = outputs[0]
  for output in outputs.dropFirst() { result = Functional.concat(axis: 0, result, output) }
  return result
}

func MOSSTextEncoder<T: TensorNumeric>(
  _ type: T.Type, length: Int, device: Int, diagnostic: Bool = false
) -> (Model, [MOSSBinding], [String]) {
  let tokens = Input()
  let rotary = Input()
  let mask = Input()
  let zeroState = Input()
  let convPadding = Input()
  let sineRotary = Input()
  let lowerTriangle = Input()
  var bindings = [MOSSBinding]()
  func dense(_ count: Int, _ name: String, transform: String = "identity", heads: Int = 0)
    -> Model
  {
    let model = Dense(count: count, noBias: true, name: name)
    bindings.append(.init(model: model, key: name, transform: transform, heads: heads))
    return model
  }
  func norm(_ x: Model.IO, _ name: String, axis: Int) -> Model.IO {
    // Qwen applies (1 + weight) to the FP32 normalized vector before rounding.
    let model = RMSNorm(epsilon: 1e-6, axis: [axis], name: name)
    bindings.append(.init(model: model, key: name, transform: "qwen_norm", float32: true))
    return model(x.to(.Float32)).to(T.dataType)
  }
  let embedding = Embedding(
    T.self, vocabularySize: MOSSTextConfig.vocabulary, embeddingSize: 4_096,
    name: "model.language_model.embed_tokens")
  bindings.append(
    .init(model: embedding, key: "model.language_model.embed_tokens", parameter: true))
  var x = embedding(tokens)
  var outputs = [Model.IO]()
  var names = [String]()
  func tap(_ value: Model.IO, _ name: String) {
    if diagnostic {
      outputs.append(value)
      names.append(name)
    }
  }
  tap(x, "embedding")
  for layer in 0..<32 {
    let p = "model.language_model.layers.\(layer)"
    let u = norm(x, p + ".input_layernorm", axis: 1)
    if layer == 0 { tap(u, "input_norm_0") }
    let attention: Model.IO
    if (layer + 1) % 4 == 0 {
      let q = dense(4_096, p + ".self_attn.q_proj", transform: "qwen_q", heads: 16)
      let gate = dense(4_096, p + ".self_attn.q_gate_proj", transform: "qwen_gate", heads: 16)
      let k = dense(1_024, p + ".self_attn.k_proj", transform: "qwen_k", heads: 4)
      let v = dense(1_024, p + ".self_attn.v_proj")
      let queries = mossApplyRotary(
        norm(q(u).reshaped([1, length, 16, 256]), p + ".self_attn.q_norm", axis: 3),
        cosine: rotary, sine: T.dataType == .Float32 ? nil : sineRotary)
      let keys = mossApplyRotary(
        norm(k(u).reshaped([1, length, 4, 256]), p + ".self_attn.k_norm", axis: 3),
        cosine: rotary, sine: T.dataType == .Float32 ? nil : sineRotary)
      let attended = mossAttention(
        type, q: queries, k: keys, v: v(u).reshaped([1, length, 4, 256]),
        length: length, heads: 16, kvHeads: 4, headDim: 256, causalMask: mask)
      attention = dense(4_096, p + ".self_attn.o_proj")(
        attended .* gate(u).to(.Float32).sigmoid().to(T.dataType))
    } else {
      let lp = p + ".linear_attn"
      let qkv = dense(8_192, lp + ".in_proj_qkv")
      let z = dense(4_096, lp + ".in_proj_z")
      let a = dense(32, lp + ".in_proj_a")
      let b = dense(32, lp + ".in_proj_b")
      let conv = Convolution(
        groups: 8_192, filters: 8_192, filterSize: [4, 1], noBias: true,
        hint: Hint(stride: [1, 1]),
        format: .OIHW, name: lp + ".conv1d.weight")
      bindings.append(
        .init(
          model: conv, key: lp + ".conv1d.weight", transform: "conv1d",
          float32: T.dataType == .BFloat16))
      let projected = qkv(u)
      if layer == 0 { tap(projected, "qkv_0") }
      let padded = Functional.concat(
        axis: 1, convPadding, projected.reshaped(.NHWC(1, length, 1, 8_192)))
      // cuDNN's NNC convolution backend has no BF16 registration. Accumulate
      // the exact BF16 operands in FP32, then round at the convolution boundary.
      let convolution = conv(T.dataType == .BFloat16 ? padded.to(.Float32) : padded).to(T.dataType)
      let mixed = mossSiLU(convolution, type)
        .reshaped([length, 8_192])
      if layer == 0 { tap(mixed, "conv_0") }
      func columns(_ start: Int, _ count: Int) -> Model.IO {
        mixed.reshaped([length, count], offset: [0, start], strides: [8_192, 1]).contiguous()
      }
      func linearNorm(_ source: Model.IO, query: Bool) -> Model.IO {
        if T.dataType == .Float32 {
          // CUDA's RMSNorm backend rejects scale != 1. Apply the scale explicitly.
          let normalized = RMSNorm(epsilon: 1e-6 / 128, axis: [2], elementwiseAffine: false)(source)
          return normalized * (query ? 1.0 / 128 : 1 / Float(128).squareRoot())
        }
        // Match Transformers' fallback l2norm rounding before FP32 recurrence.
        let squares = source .* source
        let sum = ReduceSum(axis: [2])(squares.to(.Float32)).to(T.dataType)
        let denominator = (sum.to(.Float32) + 1e-6).to(T.dataType)
        let inverse = denominator.to(.Float32).pow(-0.5).to(T.dataType)
        let unit = (source .* inverse).to(.Float32)
        return query ? unit * (1 / Float(128).squareRoot()) : unit
      }
      let q = linearNorm(columns(0, 2_048).reshaped([length, 16, 128]), query: true)
        .reshaped([length, 16, 1, 128])
      let k = linearNorm(columns(2_048, 2_048).reshaped([length, 16, 128]), query: false)
        .reshaped([length, 16, 1, 128])
      let query = Functional.concat(axis: 2, q, q).reshaped([length, 32, 128])
      let key = Functional.concat(axis: 2, k, k).reshaped([length, 32, 128])
      let alpha = Parameter<Float>(.GPU(device), .C(32), name: lp + ".A_log")
      let dtBias = Parameter<T>(.GPU(device), .C(32), name: lp + ".dt_bias")
      bindings += [
        .init(model: alpha, key: lp + ".A_log", transform: "exp", parameter: true, float32: true),
        .init(model: dtBias, key: lp + ".dt_bias", parameter: true),
      ]
      let decay =
        (alpha().reshaped([1, 32])
          .* (a(u).to(.Float32) + dtBias().to(.Float32).reshaped([1, 32])).softplus()) * -1
      let beta = b(u).to(.Float32).sigmoid().to(T.dataType).to(.Float32)
      let core = mossGatedDelta(
        q: query, k: key, v: columns(4_096, 4_096).to(.Float32).reshaped([length, 32, 128]),
        logDecay: decay, beta: beta, initialState: zeroState, lowerTriangle: lowerTriangle,
        length: length
      )
      .to(T.dataType).reshaped([length * 32, 128])
      if layer == 0 { tap(core, "core_0") }
      let gain = Parameter<T>(.GPU(device), .WC(1, 128), name: lp + ".norm")
      bindings.append(.init(model: gain, key: lp + ".norm", parameter: true))
      let normalized =
        RMSNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)(core.to(.Float32))
        .to(T.dataType) .* gain()
      let gated =
        (normalized.to(.Float32) .* z(u).reshaped([length * 32, 128]).to(.Float32).swish())
        .to(T.dataType).reshaped([length, 4_096])
      if layer == 0 { tap(gated, "gated_0") }
      attention = dense(4_096, lp + ".out_proj")(gated)
    }
    if layer == 0 { tap(attention, "attention_0") }
    x = x + attention
    let h = norm(x, p + ".post_attention_layernorm", axis: 1)
    let gate = dense(12_288, p + ".mlp.gate_proj")
    let up = dense(12_288, p + ".mlp.up_proj")
    let down = dense(4_096, p + ".mlp.down_proj")
    x = x + down(mossSiLU(gate(h), type) .* up(h))
    if diagnostic && MOSSTextConfig.diagnosticLayers.contains(layer) {
      outputs.append(x)
      names.append("layer_\(layer)")
    }
  }
  outputs.append(norm(x, "model.language_model.norm", axis: 1))
  names.append("output")
  return (
    Model(
      [tokens, rotary] + (T.dataType == .Float32 ? [mask] : [sineRotary])
        + [zeroState, convPadding, lowerTriangle], outputs), bindings, names
  )
}

func mossTextRotary<T: TensorNumeric>(_ type: T.Type, length: Int) -> Tensor<T> {
  var tensor = Tensor<Float>(.CPU, .NHWC(1, length, 1, 256))
  for token in 0..<length {
    for pair in 0..<128 {
      let angle = pair < 32 ? Double(token) / pow(10_000_000, Double(2 * pair) / 64) : 0
      tensor[0, token, 0, pair * 2] = Float(cos(angle))
      tensor[0, token, 0, pair * 2 + 1] = Float(sin(angle))
    }
  }
  return Tensor<T>(from: tensor)
}
