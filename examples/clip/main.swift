import NNC
import NNCPythonConversion
import PythonKit

func CLIPTextEmbedding() -> (Model, Model, Model) {
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(Float.self, vocabularySize: 49408, embeddingSize: 768)
  let positionEmbed = Embedding(Float.self, vocabularySize: 77, embeddingSize: 768)
  let embedding = tokenEmbed(tokens) + positionEmbed(positions)
  return (tokenEmbed, positionEmbed, Model([tokens, positions], [embedding], name: "embeddings"))
}

let transformers = Python.import("transformers")

let tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
let transformer = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

let batch_encoding = tokenizer(
  ["a photograph of an astronaut riding a horse"], truncation: true, max_length: 77,
  return_length: true, return_overflowing_tokens: false, padding: "max_length", return_tensors: "pt"
)
let tokens = batch_encoding["input_ids"]
let outputs = transformer(input_ids: tokens)
let state_dict = transformer.state_dict()

let vocab = state_dict["text_model.embeddings.token_embedding.weight"]
let pos = state_dict["text_model.embeddings.position_embedding.weight"]
let (tokenEmbed, positionEmbed, textEmbedding) = CLIPTextEmbedding()

let graph = DynamicGraph()
let tokensTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(77), of: Int32.self)
let tokensNumpy = tokens.numpy()
for i in 0..<77 {
  tokensTensor[i] = Int32(tokensNumpy[0, i])!
  positionTensor[i] = Int32(i)
}

let _ = textEmbedding(inputs: tokensTensor, positionTensor)

// tokenEmbed.parameters.copy(from: try! Tensor<Float>(numpy: vocab.numpy()))
// positionEmbed.parameters.copy(from: try! Tensor<Float>(numpy: pos.numpy()))

graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
  $0.read("text_model", model: textEmbedding)
}

let outputTensor = textEmbedding(inputs: tokensTensor, positionTensor)[0].as(of: Float.self)
for i in 0..<6 {
  let x = i < 3 ? i : 71 + i
  for j in 0..<6 {
    let y = j < 3 ? j : 762 + j
    print("\(x) \(y) \(outputTensor[x, y])")
  }
}

// graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
//   $0.write("text_model", model: textEmbedding)
// }
// print(state_dict["text_model.embeddings.position_ids"])
