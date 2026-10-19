import Foundation
import NNC

let mossDiTFP32StoreNames: Set<String> = [
  "__dit__[t-time_embedder_2-0-0]", "__dit__[t-time_embedder_2-0-1]",
]

struct MOSSBinding {
  let model: Model
  let key: String
  var transform = "identity"
  var heads = 0
  var rowStart = 0
  var rowCount = 0
  var bias = false
  var parameter = false
  var float32 = false
}

func mossSiLU<T: TensorNumeric>(_ x: Model.IO, _ type: T.Type) -> Model.IO {
  x.to(.Float32).swish().to(T.dataType)
}

func mossApplyRotary(_ x: Model.IO, cosine: Model.IO, sine: Model.IO?) -> Model.IO {
  guard let sine = sine else { return Functional.cmul(left: x, right: cosine) }
  // PyTorch rounds the cosine and sine products separately before adding them.
  return Functional.cmul(left: x, right: cosine) + Functional.cmul(left: x, right: sine)
}

func mossSplitRotary<T: TensorNumeric>(_ rotary: Tensor<T>) -> (Tensor<T>, Tensor<T>) {
  let length = rotary.shape[1]
  let dimension = rotary.shape[3]
  var cosine = Tensor<T>(.CPU, .NHWC(1, length, 1, dimension))
  var sine = Tensor<T>(.CPU, .NHWC(1, length, 1, dimension))
  for token in 0..<length {
    for pair in 0..<(dimension / 2) {
      cosine[0, token, 0, pair * 2] = rotary[0, token, 0, pair * 2]
      cosine[0, token, 0, pair * 2 + 1] = 0
      sine[0, token, 0, pair * 2] = 0
      sine[0, token, 0, pair * 2 + 1] = rotary[0, token, 0, pair * 2 + 1]
    }
  }
  return (cosine, sine)
}

func mossAttention<T: TensorNumeric>(
  _ type: T.Type, q: Model.IO, k: Model.IO, v: Model.IO,
  length: Int, heads: Int, kvHeads: Int, headDim: Int, causalMask: Model.IO? = nil
) -> Model.IO {
  if T.dataType != .Float32 {
    return ScaledDotProductAttention(
      scale: 1 / Float(headDim).squareRoot(), isCausal: causalMask != nil)(q, k, v)
      .reshaped([length, heads * headDim])
  }
  let groups = heads / kvHeads
  let queries = q.reshaped([length, kvHeads, groups, headDim]).transposed(0, 1)
    .reshaped([kvHeads, length * groups, headDim])
  let keys = k.reshaped([length, kvHeads, headDim]).transposed(0, 1)
  var logits =
    Matmul(transposeB: (1, 2))(queries, keys)
    .reshaped([kvHeads, length, groups, length]).transposed(1, 2)
    * (1 / Float(headDim).squareRoot())
  if let mask = causalMask { logits = logits + mask }
  let probabilities = logits.reshaped([heads * length, length]).softmax()
    .reshaped([kvHeads, groups, length, length]).transposed(1, 2)
    .reshaped([kvHeads, length * groups, length])
  return Matmul()(probabilities, v.reshaped([length, kvHeads, headDim]).transposed(0, 1))
    .reshaped([kvHeads, length, groups, headDim]).transposed(0, 1)
    .reshaped([length, heads * headDim])
}
