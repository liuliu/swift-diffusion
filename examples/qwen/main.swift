import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit
import TensorBoard

let graph = DynamicGraph()

graph.maxConcurrency = .limit(4)

let torch = Python.import("torch")
let diffusers = Python.import("diffusers")

torch.set_grad_enabled(false)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let pipe = diffusers.DiffusionPipeline.from_pretrained(
  "Qwen/Qwen-Image", torch_dtype: torch.bfloat16)
pipe.enable_model_cpu_offload()

let prompt = "A coffee shop entrance features a chalkboard sign reading Qwen Coffee"  // \"Qwen Coffee 😊 $2 per cup,\" with a neon light beside it displaying \"通义千问\". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written \"π≈3.1415926-53589793-23846264-33832795-02384197\"."

let tokenizer = GPT2Tokenizer(
  vocabulary: "/home/liu/workspace/swift-diffusion/examples/qwen/vocab.json",
  merges: "/home/liu/workspace/swift-diffusion/examples/qwen/merges.txt",
  specialTokens: [
    "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
    "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
  ])

let result = pipe.tokenizer(prompt, padding: true, truncation: true, return_tensors: "pt")
let result2 = tokenizer.tokenize(text: prompt, addSpecialTokens: false)

let negative_prompt = " "

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    // The rotary in Qwen is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    let q_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).view(
      28, 2, 64, 3_584
    ).transpose(1, 2).cpu().numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let q_bias = state_dict["\(prefix).self_attn.q_proj.bias"].type(torch.float).view(
      28, 2, 64
    ).transpose(1, 2).cpu().numpy()
    toqueries.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_bias)))
    let k_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).view(
      4, 2, 64, 3_584
    ).transpose(1, 2).cpu().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    let k_bias = state_dict["\(prefix).self_attn.k_proj.bias"].type(torch.float).view(
      4, 2, 64
    ).transpose(1, 2).cpu().numpy()
    tokeys.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_bias)))
    let v_weight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu()
      .numpy()
    tovalues.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: v_weight)))
    let v_bias = state_dict["\(prefix).self_attn.v_proj.bias"].type(torch.float).cpu()
      .numpy()
    tovalues.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: v_bias)))
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

func TransformerBlock(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(prefix: prefix, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "mlp")
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
      prefix: "layers.\(i)", k: width / heads, h: heads, hk: 4, b: batchSize,
      t: tokenLength,
      MLP: MLP)
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

// print(pipe.text_encoder)

graph.withNoGrad {
  /*
  let text_encoder_state_dict = pipe.text_encoder.model.language_model.state_dict()
  let rotTensor = graph.variable(.CPU, .NHWC(1, 13, 1, 128), of: Float.self)
  for i in 0..<13 {
    for k in 0..<64 {
      let theta = Double(i) * 1.0 / pow(1_000_000, Double(k) * 2 / 128)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
  }
  let (transformer, reader) = Transformer(
    Float16.self, vocabularySize: 152_064, maxLength: 13, width: 3_584, tokenLength: 13,
    layers: 28, MLP: 18_944, heads: 28, outputHiddenStates: 28, batchSize: 1)
  let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [13], of: Int32.self)
  for i in 0..<result2.count {
    tokensTensor[i] = result2[i]
  }
  let tokensTensorGPU = tokensTensor.toGPU(1)
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(1)
  transformer.compile(inputs: tokensTensorGPU, rotTensorGPU)
  // reader(text_encoder_state_dict)
  graph.openStore("/home/liu/workspace/swift-diffusion/qwen_2.5_vl_7b_f16.ckpt") {
    $0.read("text_model", model: transformer)
  }
  let lastHiddenStates = transformer(inputs: tokensTensorGPU, rotTensorGPU)[0].as(of: Float16.self)
  debugPrint(lastHiddenStates)
  */
}

var savedTensor: Tensor<Float>? = nil

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, scaleFactor: Float?, name: String)
  -> (
    Model, Model, Model
  )
{
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if let scaleFactor = scaleFactor {
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out).to(.Float32)
  if let scaleFactor = scaleFactor {
    out = out * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  scaleFactor: (Float, Float)
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let rot = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(
    epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut =
    ((1 + contextChunks[1].to(of: context)) .* contextNorm1(context)
    + contextChunks[0].to(of: context)).to(.Float16)
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x)
  xOut = ((1 + xChunks[1].to(of: x)) .* xOut + xChunks[0].to(of: x)).to(.Float16)
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  let out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])(
    queries, keys, values
  ).reshaped([b, t + hw, h * k])
  /*
  keys = keys.transposed(1, 2)
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .transposed(1, 2)
  values = values.transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  */
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = scaleFactor.0 * unifyheads((1.0 / scaleFactor.0) * contextOut).to(of: context)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = scaleFactor.0 * xUnifyheads((1.0 / scaleFactor.0) * xOut).to(of: x)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2]).to(of: context) .* contextOut
  }
  xOut = x + (xChunks[2]).to(of: x) .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1, name: "c")
    let contextNorm2 = LayerNorm(
      epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + contextChunks[5].to(of: contextOut)
      .* contextFF(
        (contextNorm2(contextOut) .* (1 + contextChunks[4].to(of: contextOut))
          + contextChunks[3].to(of: contextOut)).to(.Float16)
      ).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, scaleFactor: scaleFactor.1, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + xChunks[5].to(of: xOut)
    .* xFF((xNorm2(xOut) .* (1 + xChunks[4].to(of: xOut)) + xChunks[3].to(of: xOut)).to(.Float16))
    .to(of: xOut)
  let reader: (PythonObject) -> Void = { state_dict in
    let txt_attn_q_weight = state_dict["\(prefix).attn.add_q_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_q_bias = state_dict["\(prefix).attn.add_q_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_q_weight)))
    contextToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_q_bias)))
    let txt_attn_k_weight = state_dict["\(prefix).attn.add_k_proj.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let txt_attn_k_bias = state_dict["\(prefix).attn.add_k_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToKeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_k_weight)))
    contextToKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_k_bias)))
    let txt_attn_v_weight = state_dict["\(prefix).attn.add_v_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_attn_v_bias = state_dict["\(prefix).attn.add_v_proj.bias"].to(
      torch.float
    ).cpu().numpy()
    contextToValues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_v_weight)))
    contextToValues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_v_bias)))
    let txt_attn_key_norm_scale = state_dict["\(prefix).attn.norm_added_k.weight"].to(
      torch.float
    ).cpu().numpy()
    normAddedK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_key_norm_scale)))
    let txt_attn_query_norm_scale = state_dict["\(prefix).attn.norm_added_q.weight"].to(
      torch.float
    ).cpu().numpy()
    normAddedQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_attn_query_norm_scale)))
    let img_attn_q_weight = state_dict["\(prefix).attn.to_q.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_q_bias = state_dict["\(prefix).attn.to_q.bias"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_q_weight)))
    xToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_q_bias)))
    let img_attn_k_weight = state_dict["\(prefix).attn.to_k.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let img_attn_k_bias = state_dict["\(prefix).attn.to_k.bias"].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_k_weight)))
    xToKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_k_bias)))
    let img_attn_v_weight = state_dict["\(prefix).attn.to_v.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_attn_v_bias = state_dict["\(prefix).attn.to_v.bias"].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_v_weight)))
    xToValues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_v_bias)))
    let img_attn_key_norm_scale = state_dict["\(prefix).attn.norm_k.weight"].to(
      torch.float
    ).cpu().numpy()
    normK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_key_norm_scale)))
    let img_attn_query_norm_scale = state_dict["\(prefix).attn.norm_q.weight"].to(
      torch.float
    ).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_attn_query_norm_scale)))
    if let contextUnifyheads = contextUnifyheads {
      let attn_to_add_out_weight = state_dict["\(prefix).attn.to_add_out.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_add_out_weight)))
      let attn_to_add_out_bias =
        ((1 / scaleFactor.0).pythonObject
        * state_dict["\(prefix).attn.to_add_out.bias"]
        .to(
          torch.float
        ).cpu()).numpy()
      contextUnifyheads.bias.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_add_out_bias)))
    }
    let attn_to_out_0_weight = state_dict["\(prefix).attn.to_out.0.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_out_0_weight)))
    let attn_to_out_0_bias =
      ((1 / scaleFactor.0).pythonObject
      * state_dict["\(prefix).attn.to_out.0.bias"].to(
        torch.float
      ).cpu()).numpy()
    xUnifyheads.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: attn_to_out_0_bias)))
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      let ff_context_linear_1_weight = state_dict["\(prefix).txt_mlp.net.0.proj.weight"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_context_linear_1_weight)))
      let ff_context_linear_1_bias = state_dict["\(prefix).txt_mlp.net.0.proj.bias"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.bias.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_context_linear_1_bias)))
      let ff_context_out_projection_weight =
        state_dict[
          "\(prefix).txt_mlp.net.2.weight"
        ].to(
          torch.float
        ).cpu().numpy()
      contextOutProjection.weight.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_context_out_projection_weight)))
      let ff_context_out_projection_bias =
        ((1 / scaleFactor.1).pythonObject
        * state_dict[
          "\(prefix).txt_mlp.net.2.bias"
        ].to(
          torch.float
        ).cpu()).numpy()
      contextOutProjection.bias.copy(
        from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_context_out_projection_bias)))
    }
    let ff_linear_1_weight = state_dict["\(prefix).img_mlp.net.0.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_linear_1_weight)))
    let ff_linear_1_bias = state_dict["\(prefix).img_mlp.net.0.proj.bias"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_linear_1_bias)))
    let ff_out_projection_weight =
      state_dict["\(prefix).img_mlp.net.2.weight"].to(
        torch.float
      ).cpu().numpy()
    xOutProjection.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_out_projection_weight)))
    let ff_out_projection_bias =
      ((1 / scaleFactor.1).pythonObject
      * state_dict["\(prefix).img_mlp.net.2.bias"].to(
        torch.float
      ).cpu()).numpy()
    xOutProjection.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ff_out_projection_bias)))
    let norm1_context_linear_weight = state_dict[
      "\(prefix).txt_mod.1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let norm1_context_linear_bias = state_dict[
      "\(prefix).txt_mod.1.bias"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<(contextBlockPreOnly ? 2 : 6) {
      contextAdaLNs[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_context_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...])))
      contextAdaLNs[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_context_linear_bias[(k * h * i)..<(k * h * (i + 1))])))
    }
    let norm1_linear_weight = state_dict["\(prefix).img_mod.1.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let norm1_linear_bias = state_dict["\(prefix).img_mod.1.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...])))
      xAdaLNs[i].bias.copy(
        from: Tensor<Float16>(
          from: try! Tensor<Float>(
            numpy: norm1_linear_bias[(k * h * i)..<(k * h * (i + 1))])))
    }
  }
  if !contextBlockPreOnly {
    return (reader, Model([x, context, c, rot], [xOut, contextOut]))
  } else {
    return (reader, Model([x, context, c, rot], [xOut]))
  }
}

// let summaryWriter = SummaryWriter(logDirectory: "/tmp/qwen/")

func QwenImage(height: Int, width: Int, textLength: Int, layers: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let txt = Input()
  let t = Input()
  let imgIn = Dense(count: 3072, name: "x_embedder")
  let txtNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "context_norm")
  let txtIn = Dense(count: 3_072, name: "context_embedder")
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(channels: 3_072, name: "t")
  var vec = timeIn(t)
  vec = vec.reshaped([1, 1, 3072]).swish()
  var context = txtIn(txtNorm(txt))
  var readers = [(PythonObject) -> Void]()
  context = context.to(.Float32)
  var out = imgIn(x).to(.Float32)
  let h = height / 2
  let w = width / 2
  for i in 0..<layers {
    let (reader, block) = JointTransformerBlock(
      prefix: "transformer_blocks.\(i)", k: 128, h: 24, b: 1, t: textLength, hw: h * w,
      contextBlockPreOnly: i == layers - 1, scaleFactor: (4, i >= layers - 1 ? 64 : 2))
    let blockOut = block(out, context, vec, rot)
    if i == layers - 1 {
      out = blockOut
    } else {
      out = blockOut[0]
      context = blockOut[1]
    }
    /*
    out = out.debug { tensors, _ in
      let q = Tensor<Float32>(from: tensors[0]!).toCPU()
      summaryWriter.addHistogram("layer_\(i)", q, step: 0)
      let savedTensor = savedTensor!
      let graph = DynamicGraph()
      graph.withNoGrad {
        let val0 = graph.variable(q)
        let val1 = graph.variable(savedTensor)
        let diff = Functional.abs(val0 - val1)
        debugPrint(diff.reduced(.mean, axis: [0, 1, 2]))
      }
    }
    */
    readers.append(reader)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  out = (1 + scale(vec)) .* normFinal(out).to(.Float16) + shift(vec)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_weight = state_dict["img_in.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_in_bias = state_dict["img_in.bias"].to(torch.float)
      .cpu().numpy()
    imgIn.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_in_weight)))
    imgIn.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_in_bias)))
    let txt_norm_scale = state_dict["txt_norm.weight"].to(
      torch.float
    ).cpu().numpy()
    txtNorm.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_norm_scale)))
    let txt_in_weight = state_dict["txt_in.weight"].to(
      torch.float
    ).cpu().numpy()
    let txt_in_bias = state_dict["txt_in.bias"].to(torch.float)
      .cpu().numpy()
    txtIn.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_in_weight)))
    txtIn.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: txt_in_bias)))
    let timestep_embedder_linear_1_weight = state_dict[
      "time_text_embed.timestep_embedder.linear_1.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let timestep_embedder_linear_1_bias = state_dict[
      "time_text_embed.timestep_embedder.linear_1.bias"
    ].to(torch.float)
      .cpu().numpy()
    timeInMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: timestep_embedder_linear_1_weight)))
    timeInMlp0.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: timestep_embedder_linear_1_bias)))
    let timestep_embedder_linear_2_weight = state_dict[
      "time_text_embed.timestep_embedder.linear_2.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    let timestep_embedder_linear_2_bias = state_dict[
      "time_text_embed.timestep_embedder.linear_2.bias"
    ].to(torch.float)
      .cpu().numpy()
    timeInMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: timestep_embedder_linear_2_weight)))
    timeInMlp2.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: timestep_embedder_linear_2_bias)))
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_linear_weight = state_dict["norm_out.linear.weight"].to(torch.float)
      .cpu().numpy()
    let norm_out_linear_bias = state_dict["norm_out.linear.bias"].to(torch.float)
      .cpu().numpy()
    scale.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_out_linear_weight[0..<3072, ...])))
    shift.weight.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: norm_out_linear_weight[3072..<(3072 * 2), ...])))
    scale.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: norm_out_linear_bias[0..<3072])))
    shift.bias.copy(
      from: Tensor<Float16>(
        from: try! Tensor<Float>(numpy: norm_out_linear_bias[3072..<(3072 * 2)])))
    let proj_out_weight = state_dict["proj_out.weight"].to(
      torch.float
    ).cpu().numpy()
    projOut.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_out_weight)))
    let proj_out_bias = state_dict["proj_out.bias"].to(
      torch.float
    ).cpu().numpy()
    projOut.bias.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: proj_out_bias)))
  }
  return (Model([x, rot, t, txt], [out]), reader)
}

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timesteps
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

let x = torch.randn([1, 4096, 64]).to(torch.bfloat16).cuda()
let txt = torch.randn([1, 18, 3584]).to(torch.bfloat16).cuda()
let txt_mask = torch.full([1, 18], 1).to(torch.bfloat16).cuda()
let t = torch.full([1], 1).to(torch.bfloat16).cuda()

/*
let output = pipe.transformer(
  x, txt, txt_mask, t, img_shapes: [PythonObject(tupleOf: 1, 64, 64)], txt_seq_lens: [18],
  attention_kwargs: [PythonObject: PythonObject](), return_dict: false)

print(output)
print(pipe.transformer)
*/

// savedTensor = try! Tensor<Float>(numpy: pipe.transformer.saved_val.to(torch.float).cpu().numpy())

graph.withNoGrad {
  let state_dict = pipe.transformer.state_dict()
  /*
  let keys = state_dict.keys()
  for key in keys {
    let numpyTensor = state_dict[key].to(torch.float).cpu().numpy()
    let tensor = try! Tensor<Float>(numpy: numpyTensor)
    summaryWriter.addHistogram(String(key)!, tensor, step: 0)
  }
  */
  /*
  let xTensor = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy())).toGPU(1))
  let tTensor = graph.variable(
    Tensor<Float16>(
      from: timeEmbedding(timesteps: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
    ).toGPU(1))
  let cTensor = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: txt.to(torch.float).cpu().numpy())).toGPU(1))
  let rotTensor = graph.variable(.CPU, .NHWC(1, 4096 + 18, 1, 128), of: Float.self)
  for i in 0..<(4096 + 18) {
    for j in 0..<128 {
      rotTensor[0, i, 0, j] = -100000
    }
  }
  let maxImgIdx = max(64 / 2, 64 / 2)
  for i in 0..<18 {
    for k in 0..<8 {
      let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) / 8)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, k * 2] = Float(costheta)
      rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
    }
    for k in 0..<28 {
      let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 8) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
    }
    for k in 0..<28 {
      let theta = Double(i + maxImgIdx) * 1.0 / pow(10_000, Double(k) / 28)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotTensor[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
      rotTensor[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
    }
  }
  for y in 0..<64 {
    for x in 0..<64 {
      let i = y * 64 + x + 18
      for k in 0..<8 {
        let theta = 0 * 1.0 / pow(10_000, Double(k) / 8)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, k * 2] = Float(costheta)
        rotTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<28 {
        let theta = Double(y - (64 - 64 / 2)) * 1.0 / pow(10_000, Double(k) / 28)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 8) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 8) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<28 {
        let theta = Double(x - (64 - 64 / 2)) * 1.0 / pow(10_000, Double(k) / 28)
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, 0, (k + 8 + 28) * 2] = Float(costheta)
        rotTensor[0, i, 0, (k + 8 + 28) * 2 + 1] = Float(sintheta)
      }
    }
  }
  debugPrint(rotTensor)
  let rotTensorGPU = DynamicGraph.Tensor<Float16>(from: rotTensor).toGPU(1)
  let (dit, reader) = QwenImage(height: 128, width: 128, textLength: 18, layers: 60)
  dit.maxConcurrency = .limit(4)
  dit.compile(inputs: xTensor, rotTensorGPU, tTensor, cTensor)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, rotTensorGPU, tTensor, cTensor))
  */
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/qwen_image_1.0_f16.ckpt") {
    $0.write("dit", model: dit)
  }
  */
}

func ResnetBlockCausal3D(
  prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool, depth: Int, height: Int,
  width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm1")
  var out = norm1(x.reshaped([inChannels, depth, height, width])).reshaped([
    1, inChannels, depth, height, width,
  ])
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "resnet_conv1")
  out = conv1(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [0], name: "resnet_norm2")
  out = norm2(out.reshaped([outChannels, depth, height, width])).reshaped([
    1, outChannels, depth, height, width,
  ])
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "resnet_conv2")
  out = conv2(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "resnet_shortcut")
    out = nin(x) + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).norm1.gamma"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    let conv1_weight = state_dict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.gamma"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    let conv2_weight = state_dict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).conv_shortcut.weight"].to(torch.float)
        .cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).conv_shortcut.bias"].to(torch.float).cpu()
        .numpy()
      ninShortcut.weight.copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.bias.copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
  }
  return (reader, Model([x], [out]))
}

func AttnBlockCausal3D(
  prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm = RMSNorm(epsilon: 1e-6, axis: [0], name: "attn_norm")
  var out = norm(x.reshaped([inChannels, depth, height, width])).reshaped([
    1, inChannels, depth, height, width,
  ])
  let hw = width * height
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "to_k")
  let k = tokeys(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "to_q")
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    inChannels, depth, hw,
  ]).transposed(0, 1)
  var dot =
    Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([depth * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([depth, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "to_v")
  let v = tovalues(out).reshaped([inChannels, depth, hw]).transposed(0, 1)
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    name: "proj_out")
  out = x + projOut(out.transposed(0, 1).reshaped([1, inChannels, depth, height, width]))
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_weight = state_dict["\(prefix).norm.gamma"].to(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: norm_weight))
    let qkv_weight = state_dict["\(prefix).to_qkv.weight"].to(torch.float).cpu().numpy()
    let qkv_bias = state_dict["\(prefix).to_qkv.bias"].to(torch.float).cpu().numpy()
    toqueries.weight.copy(
      from: try! Tensor<Float>(numpy: qkv_weight[0..<inChannels, 0..<inChannels]))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: qkv_bias[0..<inChannels]))
    tokeys.weight.copy(
      from: try! Tensor<Float>(numpy: qkv_weight[inChannels..<(inChannels * 2), 0..<inChannels]))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: qkv_bias[inChannels..<(inChannels * 2)]))
    tovalues.weight.copy(
      from: try! Tensor<Float>(
        numpy: qkv_weight[(inChannels * 2)..<(inChannels * 3), 0..<inChannels]))
    tovalues.bias.copy(
      from: try! Tensor<Float>(numpy: qkv_bias[(inChannels * 2)..<(inChannels * 3)]))
    let proj_out_weight = state_dict["\(prefix).proj.weight"].to(torch.float).cpu().numpy()
    let proj_out_bias = state_dict["\(prefix).proj.bias"].to(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  return (reader, Model([x], [out]))
}

func DecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1, 1], name: "post_quant_conv")
  var out = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_in")
  out = convIn(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let (midBlockReader1, midBlock1) = ResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlockCausal3D(
    prefix: "decoder.mid_block.attentions.0", inChannels: previousChannel, depth: startDepth,
    height: startHeight, width: startWidth)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock2(out)
  var width = startWidth
  var height = startHeight
  var depth = startDepth
  var readers = [(PythonObject) -> Void]()
  var k = 0
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)",
        inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
      k += 1
    }
    if i > 0 {
      if i > 1 && startDepth > 1 {  // Need to bump up on the depth axis.
        let first = out.reshaped(
          [channel, 1, height, width], strides: [depth * height * width, height * width, width, 1]
        ).contiguous()
        let more = out.reshaped(
          [channel, (depth - 1), height, width], offset: [0, 1, 0, 0],
          strides: [depth * height * width, height * width, width, 1]
        ).contiguous().reshaped([1, channel, depth - 1, height, width])
        let timeConv = Convolution(
          groups: 1, filters: channel * 2, filterSize: [3, 1, 1],
          hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
          name: "time_conv")
        var expanded = timeConv(more.padded(.zero, begin: [0, 0, 2, 0, 0], end: [0, 0, 0, 0, 0]))
        let upLayer = channels.count - 1 - i
        let reader: (PythonObject) -> Void = { state_dict in
          let time_conv_weight = state_dict[
            "decoder.up_blocks.\(upLayer).upsamplers.0.time_conv.weight"
          ]
          .to(torch.float)
          .cpu().numpy()
          let time_conv_bias = state_dict[
            "decoder.up_blocks.\(upLayer).upsamplers.0.time_conv.bias"
          ].to(
            torch.float
          ).cpu()
            .numpy()
          timeConv.weight.copy(from: try! Tensor<Float>(numpy: time_conv_weight))
          timeConv.bias.copy(from: try! Tensor<Float>(numpy: time_conv_bias))
        }
        readers.append(reader)
        expanded = expanded.reshaped([2, channel, depth - 1, height, width]).permuted(1, 2, 0, 3, 4)
          .contiguous().reshaped([channel, 2 * (depth - 1), height, width])
        out = Functional.concat(axis: 1, first, expanded)
        depth = 1 + (depth - 1) * 2
        out = out.reshaped([1, channel, depth, height, width])
      }
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(
        out.reshaped([channel, depth, height, width])
      ).reshaped([1, channel, depth, height * 2, width * 2])
      width *= 2
      height *= 2
      let conv2d = Convolution(
        groups: 1, filters: channel / 2, filterSize: [1, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
        name: "upsample")
      out = conv2d(out)
      previousChannel = channel / 2
      let upLayer = channels.count - 1 - i
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict["decoder.up_blocks.\(upLayer).upsamplers.0.resample.1.weight"]
          .to(torch.float)
          .cpu().numpy()
        let conv_bias = state_dict["decoder.up_blocks.\(upLayer).upsamplers.0.resample.1.bias"].to(
          torch.float
        ).cpu()
          .numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
      k += 1
    }
  }
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  out = normOut(out.reshaped([channels[0], depth, height, width])).reshaped([
    1, channels[0], depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let reader: (PythonObject) -> Void = { state_dict in
    let post_quant_conv_weight = state_dict["post_quant_conv.weight"].to(torch.float).cpu().numpy()
    let post_quant_conv_bias = state_dict["post_quant_conv.bias"].to(torch.float).cpu().numpy()
    postQuantConv.weight.copy(from: try! Tensor<Float>(numpy: post_quant_conv_weight))
    postQuantConv.bias.copy(from: try! Tensor<Float>(numpy: post_quant_conv_bias))
    let conv_in_weight = state_dict["decoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["decoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["decoder.norm_out.gamma"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    let conv_out_weight = state_dict["decoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["decoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, Model([x], [out]))
}

func EncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int
)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_in")
  var out = convIn(x.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  var readers = [(PythonObject) -> Void]()
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  var k = 0
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "encoder.down_blocks.\(k)", inChannels: previousChannel,
        outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
      k += 1
    }
    if i < channels.count - 1 {
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [1, 3, 3],
        hint: Hint(
          stride: [1, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        name: "downsample")
      out = conv2d(out.padded(.zero, begin: [0, 0, 0, 0, 0], end: [0, 0, 0, 1, 1]))
      let downLayer = k
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict[
          "encoder.down_blocks.\(downLayer).resample.1.weight"
        ].to(
          torch.float
        ).cpu()
          .numpy()
        let conv_bias = state_dict["encoder.down_blocks.\(downLayer).resample.1.bias"]
          .to(torch.float)
          .cpu().numpy()
        conv2d.weight.copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv2d.bias.copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
      if i > 0 && depth > 1 {
        let first = out.reshaped(
          [1, channel, 1, height, width],
          strides: [depth * height * width, depth * height * width, height * width, width, 1]
        ).contiguous()
        let timeConv = Convolution(
          groups: 1, filters: channel, filterSize: [3, 1, 1],
          hint: Hint(
            stride: [2, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
          name: "time_conv")
        let shrunk = timeConv(out)
        let upLayer = k
        let reader: (PythonObject) -> Void = { state_dict in
          let time_conv_weight = state_dict["encoder.down_blocks.\(upLayer).time_conv.weight"]
            .to(torch.float)
            .cpu().numpy()
          let time_conv_bias = state_dict["encoder.down_blocks.\(upLayer).time_conv.bias"].to(
            torch.float
          ).cpu()
            .numpy()
          timeConv.weight.copy(from: try! Tensor<Float>(numpy: time_conv_weight))
          timeConv.bias.copy(from: try! Tensor<Float>(numpy: time_conv_bias))
        }
        readers.append(reader)
        depth = (depth - 1) / 2 + 1
        out = Functional.concat(axis: 2, first, shrunk)
      }
      k += 1
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlockCausal3D(
    prefix: "encoder.mid_block.attentions.0", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock2(out)
  let normOut = RMSNorm(epsilon: 1e-6, axis: [0], name: "norm_out")
  out = normOut(out.reshaped([previousChannel, depth, height, width])).reshaped([
    1, previousChannel, depth, height, width,
  ])
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1, 1], end: [0, 0, 0, 1, 1]))
  let quantConv = Convolution(groups: 1, filters: 32, filterSize: [1, 1, 1], name: "quant_conv")
  out = quantConv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["encoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["encoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    let norm_out_weight = state_dict["encoder.norm_out.gamma"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    let conv_out_weight = state_dict["encoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["encoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
    let quant_conv_weight = state_dict["quant_conv.weight"].to(torch.float).cpu().numpy()
    let quant_conv_bias = state_dict["quant_conv.bias"].to(torch.float).cpu().numpy()
    quantConv.weight.copy(from: try! Tensor<Float>(numpy: quant_conv_weight))
    quantConv.bias.copy(from: try! Tensor<Float>(numpy: quant_conv_bias))
  }
  return (reader, Model([x], [out]))
}

let z = torch.randn([1, 16, 3, 64, 64]).to(torch.bfloat16).cuda()
let vae_out = pipe.vae.decode(z, return_dict: false)[0]
print(pipe.vae.encode(vae_out, return_dict: false)[0].mode())
let vae_state_dict = pipe.vae.state_dict()

graph.withNoGrad {
  /*
  let mean = graph.variable(Tensor<Float>([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
  ], format: .NCHW, shape: [1, 16, 1, 1, 1])).toGPU(1)
  let std = graph.variable(Tensor<Float>([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
  ], format: .NCHW, shape: [1, 16, 1, 1, 1])).toGPU(1)
  */
  var zTensor = graph.variable(try! Tensor<Float>(numpy: z.to(torch.float).cpu().numpy())).reshaped(
    format: .NCHW, shape: [1, 16, 3, 64, 64]
  ).toGPU(1)
  // zTensor = zTensor / std + mean
  let (decoderReader, decoder) = DecoderCausal3D(
    channels: [96, 192, 384, 384], numRepeat: 2, startWidth: 64, startHeight: 64, startDepth: 3)
  decoder.compile(inputs: zTensor)
  decoderReader(vae_state_dict)
  let image = decoder(inputs: zTensor)[0].as(of: Float.self)
  debugPrint(image)
  let (encoderReader, encoder) = EncoderCausal3D(
    channels: [96, 192, 384, 384], numRepeat: 2, startWidth: 64, startHeight: 64, startDepth: 3)
  encoder.compile(inputs: image)
  encoderReader(vae_state_dict)
  debugPrint(encoder(inputs: image)[0].as(of: Float.self)[0..<1, 0..<16, 0..<3, 0..<64, 0..<64])
  graph.openStore("/home/liu/workspace/swift-diffusion/qwen_image_vae_f32.ckpt") {
    $0.write("decoder", model: decoder)
    $0.write("encoder", model: encoder)
  }
}

/*
let image = pipe(
    prompt: prompt,
    negative_prompt: negative_prompt,
    width: 1024,
    height: 1024,
    num_inference_steps: 50,
    true_cfg_scale: 4.0
).images[0]
*/
