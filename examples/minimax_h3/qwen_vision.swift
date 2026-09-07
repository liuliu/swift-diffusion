import Foundation
import NNC

// Qwen3-VL vision architecture and positional layout, following the existing
// HiDream-O1 Qwen3-VL converter. No Python dependency in this model definition.
enum H3QwenVisionConfig {
  static let hiddenSize = 1_152
  static let intermediateSize = 4_304
  static let heads = 16
  static let headDim = 72
  static let layers = 27
  static let positionEmbeddings = 2_304
  static let positionSide = 48
  static let patchSize = 16
  static let temporalPatchSize = 2
  static let patchDimension = 3 * temporalPatchSize * patchSize * patchSize
  static let mergeSize = 2
  static let mergedSize = hiddenSize * mergeSize * mergeSize
  static let outputSize = 5_120
  static let deepStackLayers = [8, 16, 24]
}

typealias H3QwenVisionGrid = (t: Int, h: Int, w: Int)

func h3QwenVisionTokenCount(_ grids: [H3QwenVisionGrid]) -> Int {
  precondition(!grids.isEmpty)
  for grid in grids {
    precondition(grid.t > 0 && grid.h > 0 && grid.w > 0 && grid.h % 2 == 0 && grid.w % 2 == 0)
  }
  return grids.reduce(0) { $0 + $1.t * $1.h * $1.w }
}

func h3QwenVisionPositions(_ grids: [H3QwenVisionGrid]) -> (
  ids: Tensor<Int32>, weights: Tensor<Float>, rotary: Tensor<Float>
) {
  let count = h3QwenVisionTokenCount(grids)
  let side = H3QwenVisionConfig.positionSide
  let half = H3QwenVisionConfig.headDim / 2
  var ids = Tensor<Int32>(.CPU, .C(4 * count))
  var weights = Tensor<Float>(.CPU, .C(4 * count))
  var rotary = Tensor<Float>(.CPU, .NHWC(1, count, 1, H3QwenVisionConfig.headDim))
  var token = 0
  for grid in grids {
    for _ in 0..<grid.t {
      for blockY in 0..<(grid.h / 2) {
        for blockX in 0..<(grid.w / 2) {
          for intraY in 0..<2 {
            for intraX in 0..<2 {
              let y = blockY * 2 + intraY
              let x = blockX * 2 + intraX
              let yp = Float(y) * Float(side - 1) / Float(grid.h - 1)
              let xp = Float(x) * Float(side - 1) / Float(grid.w - 1)
              let yf = Int(yp)
              let xf = Int(xp)
              let yc = min(yf + 1, side - 1)
              let xc = min(xf + 1, side - 1)
              let dy = yp - Float(yf)
              let dx = xp - Float(xf)
              let corners = [yf * side + xf, yf * side + xc, yc * side + xf, yc * side + xc]
              let coefficients = [(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx]
              for i in 0..<4 {
                ids[i * count + token] = Int32(corners[i])
                weights[i * count + token] = coefficients[i]
              }
              for i in 0..<half {
                let position = i < half / 2 ? y : x
                let frequency = i % (half / 2)
                let angle = Double(position) / pow(10_000, Double(frequency * 2) / Double(half))
                rotary[0, token, 0, i * 2] = Float(cos(angle))
                rotary[0, token, 0, i * 2 + 1] = Float(sin(angle))
              }
              token += 1
            }
          }
        }
      }
    }
  }
  return (ids, weights, rotary)
}

// Bindings are also useful to audit coverage against the source checkpoint.
struct H3QwenVisionBinding {
  enum Kind { case affine, embedding, patch, query, key, value }
  let model: Model
  let prefix: String
  let kind: Kind
}

func H3QwenVisionAttention<T: TensorNumeric>(
  _ type: T.Type, prefix: String, grids: [H3QwenVisionGrid]
) -> (Model, [H3QwenVisionBinding]) {
  let x = Input()
  let rotary = Input()
  let count = h3QwenVisionTokenCount(grids)
  let width = H3QwenVisionConfig.hiddenSize
  let heads = H3QwenVisionConfig.heads
  let dim = H3QwenVisionConfig.headDim
  let q = Dense(count: width, name: "q_proj")
  let k = Dense(count: width, name: "k_proj")
  let v = Dense(count: width, name: "v_proj")
  // Reference RoPE multiplies in FP32, then returns to the attention dtype.
  let queries = Functional.cmul(
    left: q(x).reshaped(.NHWC(1, count, heads, dim)).to(.Float32), right: rotary
  ).to(T.dataType)
  let keys = Functional.cmul(
    left: k(x).reshaped(.NHWC(1, count, heads, dim)).to(.Float32), right: rotary
  ).to(T.dataType)
  let values = v(x).reshaped(.NHWC(1, count, heads, dim))
  var segments = [Model.IO]()
  var offset = 0
  // cu_seqlens in the reference isolates each temporal slice, including when
  // multiple differently sized images / videos are packed in one call.
  for grid in grids {
    for _ in 0..<grid.t {
      let length = grid.h * grid.w
      func segment(_ tensor: Model.IO) -> Model.IO {
        tensor.reshaped(
          .NHWC(1, length, heads, dim), offset: [0, offset, 0, 0],
          strides: [count * width, width, dim, 1]
        ).contiguous()
      }
      let query = segment(queries)
      let key = segment(keys)
      let value = segment(values)
      let attended: Model.IO
      if T.dataType == .Float32 {
        let scores =
          Matmul(transposeB: (2, 3))(
            query.transposed(1, 2), key.transposed(1, 2)) * (1 / Float(dim).squareRoot())
        let probabilities = scores.reshaped([heads * length, length]).softmax()
          .reshaped([1, heads, length, length])
        attended = Matmul()(probabilities, value.transposed(1, 2)).transposed(1, 2)
      } else {
        attended = ScaledDotProductAttention(scale: 1 / Float(dim).squareRoot())(query, key, value)
      }
      segments.append(attended.reshaped([length, width]))
      offset += length
    }
  }
  var attended = segments[0]
  for segment in segments.dropFirst() { attended = Functional.concat(axis: 0, attended, segment) }
  let output = Dense(count: width, name: "out_proj")
  return (
    Model([x, rotary], [output(attended)]),
    [
      .init(model: q, prefix: "\(prefix).qkv", kind: .query),
      .init(model: k, prefix: "\(prefix).qkv", kind: .key),
      .init(model: v, prefix: "\(prefix).qkv", kind: .value),
      .init(model: output, prefix: "\(prefix).proj", kind: .affine),
    ]
  )
}

func H3QwenVisionBlock<T: TensorNumeric>(
  _ type: T.Type, layer: Int, grids: [H3QwenVisionGrid]
) -> (Model, [H3QwenVisionBinding]) {
  let prefix = "model.visual.blocks.\(layer)"
  let x = Input()
  let rotary = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], name: "norm2")
  let (attention, bindings) = H3QwenVisionAttention(type, prefix: "\(prefix).attn", grids: grids)
  var out = x + attention(norm1(x), rotary)
  let fc1 = Dense(count: H3QwenVisionConfig.intermediateSize, name: "mlp_fc1")
  let fc2 = Dense(count: H3QwenVisionConfig.hiddenSize, name: "mlp_fc2")
  out = out + fc2(fc1(norm2(out)).GELU(approximate: .tanh))
  return (
    Model([x, rotary], [out]),
    bindings + [
      .init(model: norm1, prefix: "\(prefix).norm1", kind: .affine),
      .init(model: norm2, prefix: "\(prefix).norm2", kind: .affine),
      .init(model: fc1, prefix: "\(prefix).mlp.linear_fc1", kind: .affine),
      .init(model: fc2, prefix: "\(prefix).mlp.linear_fc2", kind: .affine),
    ]
  )
}

func H3QwenVisionMerger(prefix: String, name: String, count: Int, postShuffleNorm: Bool) -> (
  Model, [H3QwenVisionBinding]
) {
  let x = Input()
  let norm = LayerNorm(
    epsilon: 1e-6, axis: [1], name: postShuffleNorm ? "\(name)_norm" : "norm_out")
  let shape = TensorShapeFormat.NC(count / 4, H3QwenVisionConfig.mergedSize)
  // DeepStack normalizes AFTER merging 2x2 patches; the final merger BEFORE.
  let normalized = norm(postShuffleNorm ? x.reshaped(shape) : x).reshaped(shape)
  let fc1 = Dense(count: H3QwenVisionConfig.mergedSize, name: "\(name)_mlp_0")
  let fc2 = Dense(count: H3QwenVisionConfig.outputSize, name: "\(name)_mlp_1")
  return (
    Model([x], [fc2(fc1(normalized).GELU())]),
    [
      .init(model: norm, prefix: "\(prefix).norm", kind: .affine),
      .init(model: fc1, prefix: "\(prefix).linear_fc1", kind: .affine),
      .init(model: fc2, prefix: "\(prefix).linear_fc2", kind: .affine),
    ]
  )
}

// Inputs: flattened processor patches, four-corner position IDs and FP32
// interpolation weights, FP32 interleaved RoPE. Outputs: final merger followed
// by DeepStack mergers 0/1/2 (to add after language blocks 0/1/2 respectively).
func H3QwenVisionModel<T: TensorNumeric>(
  _ type: T.Type, grids: [H3QwenVisionGrid], debugOutputs: Bool = false
) -> (Model, [H3QwenVisionBinding]) {
  let count = h3QwenVisionTokenCount(grids)
  let width = H3QwenVisionConfig.hiddenSize
  let patches = Input()
  let positionIDs = Input()
  let positionWeights = Input()
  let rotary = Input()
  let patch = Dense(count: width, name: "conv_in")
  let position = Embedding(
    T.self, vocabularySize: H3QwenVisionConfig.positionEmbeddings,
    embeddingSize: width, name: "pos_embed")
  let learnedPosition =
    (position(positionIDs).to(.Float32).reshaped([4, count, width])
    .* positionWeights.reshaped([4, count, 1])).reduced(.sum, axis: [0])
    .reshaped([count, width]).to(T.dataType)
  var out = patch(patches) + learnedPosition
  var debug = [out]
  var bindings: [H3QwenVisionBinding] = [
    .init(model: patch, prefix: "model.visual.patch_embed.proj", kind: .patch),
    .init(model: position, prefix: "model.visual.pos_embed", kind: .embedding),
  ]
  var deepStack = [Model.IO]()
  for layer in 0..<H3QwenVisionConfig.layers {
    let (block, blockBindings) = H3QwenVisionBlock(type, layer: layer, grids: grids)
    out = block(out, rotary)
    bindings += blockBindings
    if let index = H3QwenVisionConfig.deepStackLayers.firstIndex(of: layer) {
      let (merger, mergerBindings) = H3QwenVisionMerger(
        prefix: "model.visual.deepstack_merger_list.\(index)", name: "deepstack_\(index)",
        count: count, postShuffleNorm: true)
      deepStack.append(merger(out))
      debug.append(out)
      bindings += mergerBindings
    }
  }
  debug.append(out)
  let (merger, mergerBindings) = H3QwenVisionMerger(
    prefix: "model.visual.merger", name: "merger", count: count, postShuffleNorm: false)
  bindings += mergerBindings
  return (
    Model(
      [patches, positionIDs, positionWeights, rotary],
      [merger(out)] + deepStack + (debugOutputs ? debug : [])), bindings
  )
}
