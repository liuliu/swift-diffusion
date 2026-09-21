import Foundation
import NNC

enum Q21TextConfig {
  static let width = 4_096
  static let intermediate = 12_288
  static let layers = 36
  static let vocabulary = 151_936
  static let heads = 32
  static let kvHeads = 8
  static let headDim = 128
  static let ropeTheta: Double = 5_000_000
}

struct Q21TextBinding {
  let model: Model
  let key: String
  var kind: String = "weight"
}

// Keep the established qwen_3_vl_8b store names and Q/K interleaving exactly.
func Q21TextAttention<T: TensorNumeric>(_ type: T.Type, length: Int, prefix: String) -> (
  Model, [Q21TextBinding]
) {
  let x = Input()
  let rot = Input()
  let k = Dense(count: 1_024, noBias: true, name: "k_proj")
  let q = Dense(count: 4_096, noBias: true, name: "q_proj")
  let v = Dense(count: 1_024, noBias: true, name: "v_proj")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  let keys = Functional.cmul(left: normK(k(x).reshaped(.NHWC(1, length, 8, 128))), right: rot)
  let queries = Functional.cmul(left: normQ(q(x).reshaped(.NHWC(1, length, 32, 128))), right: rot)
  let values = v(x).reshaped(.NHWC(1, length, 8, 128))
  let attended: Model.IO
  let causal = Input()
  if T.dataType == .Float32 {
    // Grouped-query attention broadcasts four query groups per KV head.
    // Explicit per-head grouping avoids relying on GQA support in FP32 matmul.
    let groupedQ = queries.reshaped([length, 8, 4, 128]).transposed(0, 1).reshaped([
      8, length * 4, 128,
    ])
    let groupedK = keys.reshaped([length, 8, 128]).transposed(0, 1)
    let logits =
      Matmul(transposeB: (1, 2))(groupedQ, groupedK).reshaped([8, length, 4, length]).transposed(
        1, 2) * (1 / Float(128).squareRoot())
    let probabilities = (logits + causal).reshaped([32 * length, length]).softmax().reshaped([
      8, 4, length, length,
    ]).transposed(1, 2).reshaped([8, length * 4, length])
    let result = Matmul()(probabilities, values.reshaped([length, 8, 128]).transposed(0, 1))
      .reshaped([8, length, 4, 128]).transposed(0, 1).reshaped([length, 4096])
    attended = result
  } else {
    attended = ScaledDotProductAttention(scale: 1 / Float(128).squareRoot(), isCausal: true)(
      queries, keys, values
    ).reshaped([length, 4096])
  }
  let output = Dense(count: 4_096, noBias: true, name: "out_proj")
  return (
    Model(T.dataType == .Float32 ? [x, rot, causal] : [x, rot], [output(attended)]),
    [
      .init(model: k, key: prefix + ".self_attn.k_proj.weight", kind: "k"),
      .init(model: q, key: prefix + ".self_attn.q_proj.weight", kind: "q"),
      .init(model: v, key: prefix + ".self_attn.v_proj.weight"),
      .init(model: normK, key: prefix + ".self_attn.k_norm.weight", kind: "norm"),
      .init(model: normQ, key: prefix + ".self_attn.q_norm.weight", kind: "norm"),
      .init(model: output, key: prefix + ".self_attn.o_proj.weight"),
    ]
  )
}

func Q21TextBlock<T: TensorNumeric>(_ type: T.Type, length: Int, prefix: String) -> (
  Model, [Q21TextBinding]
) {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  let (attention, attentionBindings) = Q21TextAttention(type, length: length, prefix: prefix)
  let mask = Input()
  let residual =
    x + (T.dataType == .Float32 ? attention(norm1(x), rot, mask) : attention(norm1(x), rot))
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  let mlpInput = Input()
  let gate = Dense(count: 12_288, noBias: true, name: "mlp_gate_proj")
  let up = Dense(count: 12_288, noBias: true, name: "mlp_up_proj")
  let down = Dense(count: 4_096, noBias: true, name: "mlp_down_proj")
  let mlp = Model([mlpInput], [down(up(mlpInput) .* gate(mlpInput).swish())], name: "mlp")
  return (
    Model(T.dataType == .Float32 ? [x, rot, mask] : [x, rot], [residual + mlp(norm2(residual))]),
    attentionBindings + [
      .init(model: norm1, key: prefix + ".input_layernorm.weight"),
      .init(model: norm2, key: prefix + ".post_attention_layernorm.weight"),
      .init(model: gate, key: prefix + ".mlp.gate_proj.weight"),
      .init(model: up, key: prefix + ".mlp.up_proj.weight"),
      .init(model: down, key: prefix + ".mlp.down_proj.weight"),
    ]
  )
}

struct Q21VisualSpan {
  let start: Int
  let height: Int
  let width: Int
  var length: Int { height * width }
}

func Q21Text<T: TensorNumeric>(_ type: T.Type, length: Int, visual: [Q21VisualSpan] = []) -> (
  Model, [Q21TextBinding]
) {
  let tokens = Input()
  let rotary = Input()
  let mask = Input()
  let embedding = Embedding(
    T.self, vocabularySize: Q21TextConfig.vocabulary,
    embeddingSize: Q21TextConfig.width, name: "tok_embeddings")
  var out = embedding(tokens)
  let vision = (0..<4).map { _ in Input() }
  func inject(_ text: Model.IO, _ features: Model.IO, add: Bool) -> Model.IO {
    var pieces = [Model.IO]()
    var cursor = 0
    var offset = 0
    func slice(_ value: Model.IO, _ start: Int, _ count: Int) -> Model.IO {
      value.reshaped([count, 4096], offset: [start, 0], strides: [4096, 1]).contiguous()
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
  var bindings = [
    Q21TextBinding(
      model: embedding, key: "model.language_model.embed_tokens.weight", kind: "embedding")
  ]
  for layer in 0..<Q21TextConfig.layers {
    let (block, blockBindings) = Q21TextBlock(
      type, length: length, prefix: "model.language_model.layers.\(layer)")
    out = T.dataType == .Float32 ? block(out, rotary, mask) : block(out, rotary)
    if !visual.isEmpty && layer < 3 { out = inject(out, vision[layer + 1], add: true) }
    bindings += blockBindings
  }
  // Qwen-Image-2.1 consumes the complete decoder output BEFORE final RMSNorm.
  return (
    Model(
      [tokens, rotary] + (visual.isEmpty ? [] : vision) + (T.dataType == .Float32 ? [mask] : []),
      [out]), bindings
  )
}

func q21MultimodalPositions(length: Int, visual: [Q21VisualSpan]) -> Tensor<Int32> {
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

func q21MultimodalRotary<T: TensorNumeric>(_ type: T.Type, _ positions: Tensor<Int32>) -> Tensor<T>
{
  let length = positions.shape[1]
  var result = Tensor<Float>(.CPU, .NHWC(1, length, 1, 128))
  for token in 0..<length {
    for pair in 0..<64 {
      let axis = pair < 60 ? pair % 3 : 0
      let phase =
        Float(positions[axis, token]) / pow(Float(Q21TextConfig.ropeTheta), Float(2 * pair) / 128)
      result[0, token, 0, 2 * pair] = cos(phase)
      result[0, token, 0, 2 * pair + 1] = sin(phase)
    }
  }
  return Tensor<T>(from: result)
}

func q21TextRotary(length: Int) -> Tensor<Float16> {
  var rot = Tensor<Float16>(.CPU, .NHWC(1, length, 1, 128))
  for token in 0..<length {
    for pair in 0..<64 {
      let theta = Double(token) / pow(Q21TextConfig.ropeTheta, Double(2 * pair) / 128)
      rot[0, token, 0, 2 * pair] = Float16(cos(theta))
      rot[0, token, 0, 2 * pair + 1] = Float16(sin(theta))
    }
  }
  return rot
}
