import Foundation
import NNC

struct FiboBinding {
  let model: Model
  let key: String
  var transform = "identity"
  var heads = 0
  var isBias = false
  var parameter = false
  // Optional output-channel slice, applied to checkpoint weights before loading.
  var rowStart = 0
  var rowCount = 0
}

// ccv's CPU Float -> BF16/FP16 tensor conversions truncate. Generated inputs
// need round-to-nearest-even; representable checkpoint BF16 values are exact.
func fiboRoundedInput<T: TensorNumeric>(_ type: T.Type, _ input: Tensor<Float>) -> Tensor<T> {
  guard T.dataType == .BFloat16 || T.dataType == .Float16 else { return Tensor<T>(from: input) }
  let count = input.shape.reduce(1, *)
  let flat = input.reshaped(.C(count))
  if T.dataType == .Float16 {
    var rounded = Tensor<Float16>(.CPU, .C(count))
    for i in 0..<count { rounded[i] = Float16(flat[i]) }
    return Tensor<T>(from: rounded.reshaped(format: input.format, shape: input.shape))
  }
  var rounded = Tensor<BFloat16>(.CPU, .C(count))
  for i in 0..<count { rounded[i] = BFloat16(flat[i]) }
  return Tensor<T>(from: rounded.reshaped(format: input.format, shape: input.shape))
}

func fiboSiLU<T: TensorNumeric>(_ x: Model.IO, _ type: T.Type) -> Model.IO {
  x.to(.Float32).swish().to(T.dataType)
}

func fiboAttention<T: TensorNumeric>(
  _ type: T.Type, q: Model.IO, k: Model.IO, v: Model.IO,
  queries: Int, keys: Int, heads: Int = 28, dim: Int = 128
) -> Model.IO {
  if T.dataType != .Float32 {
    return ScaledDotProductAttention(scale: 1 / Float(dim).squareRoot())(q, k, v)
      .reshaped([queries, heads * dim])
  }
  let logits =
    Matmul(transposeB: (2, 3))(q.transposed(1, 2), k.transposed(1, 2))
    * (1 / Float(dim).squareRoot())
  let probabilities = logits.reshaped([heads * queries, keys]).softmax()
    .reshaped([1, heads, queries, keys])
  return Matmul()(probabilities, v.transposed(1, 2)).transposed(1, 2)
    .reshaped([queries, heads * dim])
}

func fiboRows(_ x: Model.IO, start: Int, count: Int, width: Int) -> Model.IO {
  x.reshaped([count, width], offset: [start, 0], strides: [width, 1]).contiguous()
}
