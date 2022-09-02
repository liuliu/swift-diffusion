import PythonKit

let transformers = Python.import("transformers")

let tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
let transformer = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

let batch_encoding = tokenizer(
  ["a photograph of an astronaut riding a horse"], truncation: true, max_length: 77,
  return_length: true, return_overflowing_tokens: false, padding: "max_length", return_tensors: "pt"
)
/*
let tokens = batch_encoding["input_ids"]
let outputs = transformer(input_ids: tokens)
print(tokens)
print(outputs.last_hidden_state)
*/
let state_dict = transformer.state_dict()
print(state_dict["text_model.embeddings.position_ids"])
