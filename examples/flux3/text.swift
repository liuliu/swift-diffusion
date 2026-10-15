import Foundation
import NNC

enum Flux3TextConfig {
  static let width = 2_560
  static let intermediate = 9_728
  static let layers = 36
  static let heads = 32
  static let kvHeads = 8
  static let headDim = 128
  static let vocabulary = 151_936
  static let ropeTheta: Double = 5_000_000
  // HF hidden_states indices; zero-based decoder indices are one smaller.
  static let hiddenStates = [4, 8, 12, 16, 20, 24, 28, 32]
}

// Krea 2's Qwen3-VL 4B graph and instruct store names, with FLUX.3 feature taps.
func Flux3Text<T: TensorNumeric>(
  _ type: T.Type, length: Int, validLength: Int? = nil, completeEncoder: Bool = false
) -> (Model, [Flux3Binding], [String]) {
  let valid = validLength ?? length
  precondition(valid > 0 && valid <= length)
  let tokens = Input()
  let rotary = Input()
  let mask = Input()
  let embedding = Embedding(
    T.self, vocabularySize: 151_936, embeddingSize: 2_560,
    name: "tok_embeddings")
  var bindings: [Flux3Binding] = [
    .init(
      model: embedding,
      key: "model.language_model.embed_tokens.weight", transform: "embedding")
  ]
  var out = embedding(tokens)
  var outputs = [Model.IO]()
  var names = [String]()
  let layers = completeEncoder ? Flux3TextConfig.layers : Flux3TextConfig.hiddenStates.last!
  for layer in 0..<layers {
    let prefix = "model.language_model.layers.\(layer)"
    let x = Input()
    let rot = Input()
    let attentionMask = Input()
    let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
    let u = norm1(x).to(T.dataType)
    let q = Dense(count: 4_096, noBias: true, name: "q_proj")
    let k = Dense(count: 1_024, noBias: true, name: "k_proj")
    let v = Dense(count: 1_024, noBias: true, name: "v_proj")
    let qNorm = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
    let kNorm = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
    let queries = Functional.cmul(left: qNorm(q(u).reshaped([1, length, 32, 128])), right: rot)
    let keys = Functional.cmul(left: kNorm(k(u).reshaped([1, length, 8, 128])), right: rot)
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
    let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
    let gate = Dense(count: 9_728, noBias: true, name: "mlp_gate_proj")
    let up = Dense(count: 9_728, noBias: true, name: "mlp_up_proj")
    let down = Dense(count: 2_560, noBias: true, name: "mlp_down_proj")
    let ffInput = Input()
    let ff = Model([ffInput], [down(up(ffInput) .* flux3SiLU(gate(ffInput), type))], name: "mlp")
    let block = Model(
      [x, rot] + (T.dataType == .Float32 ? [attentionMask] : []),
      [residual + ff(norm2(residual).to(T.dataType)).to(of: residual)])
    out = T.dataType == .Float32 ? block(out, rotary, mask) : block(out, rotary)
    bindings += [
      .init(model: norm1, key: prefix + ".input_layernorm.weight"),
      .init(model: norm2, key: prefix + ".post_attention_layernorm.weight"),
      .init(model: q, key: prefix + ".self_attn.q_proj.weight", transform: "qk", heads: 32),
      .init(model: k, key: prefix + ".self_attn.k_proj.weight", transform: "qk", heads: 8),
      .init(model: v, key: prefix + ".self_attn.v_proj.weight"),
      .init(model: qNorm, key: prefix + ".self_attn.q_norm.weight", transform: "qk_norm"),
      .init(model: kNorm, key: prefix + ".self_attn.k_norm.weight", transform: "qk_norm"),
      .init(model: projection, key: prefix + ".self_attn.o_proj.weight"),
      .init(model: gate, key: prefix + ".mlp.gate_proj.weight"),
      .init(model: up, key: prefix + ".mlp.up_proj.weight"),
      .init(model: down, key: prefix + ".mlp.down_proj.weight"),
    ]
    if Flux3TextConfig.hiddenStates.contains(layer + 1) {
      outputs.append(out)
      names.append("hidden_\(layer + 1)")
    }
  }
  // Retain all 36 layers + final norm when building a reusable full encoder.
  if completeEncoder {
    let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
    bindings.append(.init(model: norm, key: "model.language_model.norm.weight"))
    outputs.append(norm(out))
    names.append("final_norm")
  }
  return (
    Model([tokens, rotary] + (T.dataType == .Float32 ? [mask] : []), outputs), bindings, names
  )
}

func flux3TextRotary<T: TensorNumeric>(_ type: T.Type, positions: Tensor<Int32>) -> Tensor<T> {
  let length = positions.shape[0]
  var out = Tensor<Float>(.CPU, .NHWC(1, length, 1, 128))
  for token in 0..<length {
    for pair in 0..<64 {
      let angle = Double(positions[token]) / pow(Flux3TextConfig.ropeTheta, Double(pair * 2) / 128)
      out[0, token, 0, pair * 2] = Float(cos(angle))
      out[0, token, 0, pair * 2 + 1] = Float(sin(angle))
    }
  }
  return Tensor<T>(from: out)
}
