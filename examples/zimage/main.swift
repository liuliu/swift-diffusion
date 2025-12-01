import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

typealias FloatType = Float

let torch = Python.import("torch")
let PIL = Python.import("PIL")

torch.set_grad_enabled(false)

let torch_device = torch.device("cuda")

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let diffusers = Python.import("diffusers")

let pipe = diffusers.ZImagePipeline.from_pretrained(
  "Tongyi-MAI/Z-Image-Turbo",
  torch_dtype: torch.bfloat16,
  low_cpu_mem_usage: false
)

print(pipe.text_encoder)
print(pipe.transformer)

pipe.to("cuda")

let graph = DynamicGraph()

let prompt =
  "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

let tokenizer = TiktokenTokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/zimage/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/zimage/merges.txt",
  specialTokens: [
    "<|im_end|>": 151645, "<|im_start|>": 151644, "<|endoftext|>": 151643,
    "<|file_sep|>": 151664, "</tool_call>": 151658,
  ])

let promptWithTemplate = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
let positiveTokens = tokenizer.tokenize(text: promptWithTemplate, addSpecialTokens: false)
let text_state_dict = pipe.text_encoder.state_dict()

func SelfAttention(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: width, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    // The rotary in Qwen is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    let q_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).view(
      32, 2, 64, 2_560
    ).transpose(1, 2).cpu().numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let q_norm_weight = state_dict["\(prefix).self_attn.q_norm.weight"].type(torch.float).view(
      2, 64
    ).transpose(0, 1).cpu()
      .numpy()
    normQ.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_norm_weight)))
    let k_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).view(
      8, 2, 64, 2_560
    ).transpose(1, 2).cpu().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    let k_norm_weight = state_dict["\(prefix).self_attn.k_norm.weight"].type(torch.float).view(
      2, 64
    ).transpose(0, 1).cpu()
      .numpy()
    normK.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_norm_weight)))
    let v_weight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: v_weight)))
    let proj_weight = state_dict["\(prefix).self_attn.o_proj.weight"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_weight)))
  }
  return (Model([x, rot], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

func TransformerBlock(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(
    prefix: prefix, width: width, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: width, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { state_dict in
    attnReader(state_dict)
    let norm1_weight = state_dict["\(prefix).input_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm1_weight)))
    let norm2_weight = state_dict["\(prefix).post_attention_layernorm.weight"].type(torch.float)
      .cpu().numpy()
    norm2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm2_weight)))
    let w1_weight = state_dict["\(prefix).mlp.gate_proj.weight"].type(torch.float).cpu()
      .numpy()
    w1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w1_weight)))
    let w2_weight = state_dict["\(prefix).mlp.down_proj.weight"].type(torch.float).cpu()
      .numpy()
    w2.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w2_weight)))
    let w3_weight = state_dict["\(prefix).mlp.up_proj.weight"].type(torch.float).cpu()
      .numpy()
    w3.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: w3_weight)))
  }
  return (Model([x, rot], [out]), reader)
}

func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["embed_tokens.weight"].type(torch.float).cpu().numpy()
    tokenEmbed.parameters.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vocab)))
  }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Int?, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embedReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  var hiddenStates: Model.IO? = nil
  for i in 0..<layers {
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", width: width, k: 128, h: heads, hk: 8, b: batchSize,
      t: tokenLength, MLP: MLP)
    out = layer(out, rot)
    readers.append(reader)
    if let outputHiddenStates = outputHiddenStates, outputHiddenStates == i {
      hiddenStates = out
    }
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    embedReader(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_weight = state_dict["norm.weight"].type(torch.float).cpu().numpy()
    norm.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_weight)))
  }
  return (Model([tokens, rot], (hiddenStates.map { [$0] } ?? []) + [out]), reader)
}

let txt = graph.withNoGrad {
  let positiveRotTensor = graph.variable(
    .CPU, .NHWC(1, positiveTokens.0.count, 1, 128), of: Float.self)
  for i in 0..<positiveTokens.0.count {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      positiveRotTensor[0, i, 0, k * 2] = Float(costheta)
      positiveRotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let (transformer, reader) = Transformer(
    Float16.self, vocabularySize: 151_936, maxLength: positiveTokens.0.count, width: 2_560,
    tokenLength: positiveTokens.0.count,
    layers: 36, MLP: 9_728, heads: 32, outputHiddenStates: 34, batchSize: 1
  )
  let positiveTokensTensor = graph.variable(
    .CPU, format: .NHWC, shape: [positiveTokens.0.count], of: Int32.self)
  for i in 0..<positiveTokens.0.count {
    positiveTokensTensor[i] = positiveTokens.0[i]
  }
  let positiveTokensTensorGPU = positiveTokensTensor.toGPU(0)
  let positiveRotTensorGPU = DynamicGraph.Tensor<Float16>(from: positiveRotTensor).toGPU(0)
  transformer.compile(inputs: positiveTokensTensorGPU, positiveRotTensorGPU)
  reader(text_state_dict)
  let positiveLastHiddenStates = transformer(inputs: positiveTokensTensorGPU, positiveRotTensorGPU)[
    0
  ].as(of: Float16.self)
  debugPrint(positiveLastHiddenStates)
  return positiveLastHiddenStates
}

let image = pipe(
  prompt: prompt,
  height: 1024,
  width: 1024,
  num_inference_steps: 9,  // This actually results in 8 DiT forwards
  guidance_scale: 0.0,  // Guidance should be 0 for the Turbo models
  generator: torch.Generator("cuda").manual_seed(42)
).images[0]

image.save("example.png")
