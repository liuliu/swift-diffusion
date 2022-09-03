import NNC
import NNCPythonConversion
import PythonKit

func CLIPTextEmbedding(vocabularySize: Int, maxLength: Int, embeddingSize: Int) -> Model {
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(Float.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + positionEmbed(positions)
  return Model([tokens, positions], [embedding], name: "embeddings")
}

func CLIPAttention(k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([t, b, h, k]).transposed(0, 2).reshaped([b * h, t, k])
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([t, b, h, k])
    .transposed(0, 2).reshaped([b * h, t, k])
  let values = tovalues(x).reshaped([t, b, h, k]).transposed(0, 2).reshaped([b * h, t, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys) + casualAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b * h, t, t])
  var out = dot * values
  out = out.reshaped([h, b, t, k]).transposed(0, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return Model([x, casualAttentionMask], [out])
}

func QuickGELU() -> Model {
  let x = Input()
  let y = x .* Sigmoid()(1.702 * x)
  return Model([x], [y])
}

func CLIPMLP(hiddenSize: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = fc1(x)
  out = QuickGELU()(out)
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return Model([x], [out])
}

func CLIPEncoderLayer(k: Int, h: Int, b: Int, t: Int, intermediateSize: Int) -> Model {
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let attention = CLIPAttention(k: k, h: h, b: b, t: t)
  out = attention(out, casualAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let mlp = CLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return Model([x, casualAttentionMask], [out])
}

func CLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  let k = embeddingSize / numHeads
  for _ in 0..<numLayers {
    let encoderLayer = CLIPEncoderLayer(
      k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [out])
}

let transformers = Python.import("transformers")

let tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

let batch_encoding = tokenizer(
  ["a photograph of an astronaut riding a horse"], truncation: true, max_length: 77,
  return_length: true, return_overflowing_tokens: false, padding: "max_length", return_tensors: "pt"
)
let tokens = batch_encoding["input_ids"]

let textModel = CLIPTextModel(
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
  batchSize: 1, intermediateSize: 3072)

let graph = DynamicGraph()
let tokensTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensNumpy = tokens.numpy()
for i in 0..<77 {
  tokensTensor[i] = Int32(tokensNumpy[0, i])!
  positionTensor[i] = Int32(i)
}
let casualAttentionMask = graph.variable(Tensor<Float>(.CPU, .HWC(1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, i, j] = -Float.greatestFiniteMagnitude
  }
}

let _ = textModel(inputs: tokensTensor, positionTensor, casualAttentionMask)

graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
  $0.read("text_model", model: textModel)
}

let c = textModel(inputs: tokensTensor, positionTensor, casualAttentionMask)[0].as(of: Float.self)
for i in 0..<6 {
  let x = i < 3 ? i : 71 + i
  for j in 0..<6 {
    let y = j < 3 ? j : 762 + j
    print("\(x) \(y) \(c[x, y])")
  }
}
