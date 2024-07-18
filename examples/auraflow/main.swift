import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit
import SentencePiece

typealias FloatType = Float

let torch = Python.import("torch")
let diffusers = Python.import("diffusers")

torch.set_grad_enabled(false)

let pipeline = diffusers.AuraFlowPipeline.from_pretrained(
  "fal/AuraFlow",
  torch_dtype: torch.float16
).to("cuda")

print(pipeline.transformer)

let text_encoder_state_dict = pipeline.text_encoder.state_dict()
/*
let image = pipeline(
  prompt: "an astronaut.",
  height: 1024,
  width: 1024,
  num_inference_steps: 2,
  generator: torch.Generator().manual_seed(666),
  guidance_scale: 3.5
).images[0]
*/

func UMT5TextEmbedding(vocabularySize: Int, embeddingSize: Int, name: String) -> Model {
  let tokenEmbed = Embedding(
    Float.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: name)
  return tokenEmbed
}

func UMT5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let positionBias = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  // No scaling the queries.
  let queries = toqueries(x).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + positionBias
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: outFeatures, noBias: true, name: "o")
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, positionBias], [out]))
}

func UMT5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model)
{
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true, name: "w0")
  let wi_1 = Dense(count: intermediateSize, noBias: true, name: "w1")
  var out = wi_1(x) .* wi_0(x).GELU(approximate: .tanh)
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  out = wo(out)
  return (wi_0, wi_1, wo, Model([x], [out]))
}

func UMT5Block(
  prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let relativePositionEmbedding = Embedding(
    Float.self, vocabularySize: 32, embeddingSize: 32, name: "relative_position_embedding")
  let positionBias =
    relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, 32])
    .permuted(0, 3, 1, 2) + attentionMask
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (tokeys, toqueries, tovalues, unifyheads, attention) = UMT5LayerSelfAttention(
    k: k, h: h, b: b, t: t, outFeatures: outFeatures)
  var out = x + attention(norm1(x), positionBias)
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (wi_0, wi_1, wo, ff) = UMT5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize)
  out = out + ff(norm2(out))
  let reader: (PythonObject) -> Void = { state_dict in
    let relative_attention_bias_weight = state_dict[
      "\(prefix).layer.0.SelfAttention.relative_attention_bias.weight"
    ].cpu().float().numpy()
    relativePositionEmbedding.weight.copy(
      from: try! Tensor<Float>(numpy: relative_attention_bias_weight))
    let layer_0_layer_norm_weight = state_dict["\(prefix).layer.0.layer_norm.weight"].cpu().float()
      .numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: layer_0_layer_norm_weight))
    let k_weight = state_dict["\(prefix).layer.0.SelfAttention.k.weight"].cpu().float().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_weight))
    let q_weight = state_dict["\(prefix).layer.0.SelfAttention.q.weight"].cpu().float().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_weight))
    let v_weight = state_dict["\(prefix).layer.0.SelfAttention.v.weight"].cpu().float().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_weight))
    let o_weight = state_dict["\(prefix).layer.0.SelfAttention.o.weight"].cpu().float().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: o_weight))
    let layer_1_layer_norm_weight = state_dict["\(prefix).layer.1.layer_norm.weight"].cpu().float()
      .numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: layer_1_layer_norm_weight))
    let wi_0_weight = state_dict["\(prefix).layer.1.DenseReluDense.wi_0.weight"].cpu().float()
      .numpy()
    wi_0.weight.copy(from: try! Tensor<Float>(numpy: wi_0_weight))
    let wi_1_weight = state_dict["\(prefix).layer.1.DenseReluDense.wi_1.weight"].cpu().float()
      .numpy()
    wi_1.weight.copy(from: try! Tensor<Float>(numpy: wi_1_weight))
    let wo_weight = state_dict["\(prefix).layer.1.DenseReluDense.wo.weight"].cpu().float().numpy()
    wo.weight.copy(from: try! Tensor<Float>(numpy: wo_weight))
  }
  return (reader, Model([x, attentionMask, relativePositionBuckets], [out]))
}

func UMT5ForConditionalGeneration(b: Int, t: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let textEmbed = UMT5TextEmbedding(vocabularySize: 32_128, embeddingSize: 2_048, name: "shared")
  var out = textEmbed(x)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = UMT5Block(
      prefix: "encoder.block.\(i)", k: 64, h: 32, b: b, t: t, outFeatures: 2_048,
      intermediateSize: 5_120)
    out = block(out, attentionMask, relativePositionBuckets)
    readers.append(reader)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["shared.weight"].cpu().float().numpy()
    textEmbed.weight.copy(from: try! Tensor<Float>(numpy: vocab))
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_norm_weight = state_dict["encoder.final_layer_norm.weight"].cpu().float()
      .numpy()
    finalNorm.weight.copy(from: try! Tensor<Float>(numpy: final_layer_norm_weight))
  }
  return (reader, Model([x, attentionMask, relativePositionBuckets], [out]))
}

let graph = DynamicGraph()

let prompt =
  "an astronaut."
// "photo of a young woman with long, wavy brown hair lying down in grass, top down shot, summer, warm, laughing, joy, fun"
let negativePrompt = ""

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/swift-diffusion/tokenizer.model")
var tokens2 = sentencePiece.encode(prompt).map { return $0.id }
tokens2.append(2)

let tokensTensor2 = graph.variable(.CPU, .C(256), of: Int32.self)
for i in 0..<256 {
  tokensTensor2[i] = i < tokens2.count ? tokens2[i] : 1
}

func relativePositionBuckets(sequenceLength: Int, numBuckets: Int, maxDistance: Int) -> Tensor<
  Int32
> {
  // isBidirectional = true.
  let numBuckets = numBuckets / 2
  let maxExact = numBuckets / 2
  var relativePositionBuckets = Tensor<Int32>(.CPU, .C(sequenceLength * sequenceLength))
  for i in 0..<sequenceLength {
    for j in 0..<sequenceLength {
      var relativePositionBucket = j > i ? numBuckets : 0
      let relativePosition = abs(i - j)
      let isSmall = relativePosition < maxExact
      if isSmall {
        relativePositionBucket += relativePosition
      } else {
        let relativePositionIfLarge = min(
          numBuckets - 1,
          maxExact
            + Int(
              (log(Double(relativePosition) / Double(maxExact))
                / log(Double(maxDistance) / Double(maxExact)) * Double(numBuckets - maxExact))
                .rounded(.down)))
        relativePositionBucket += relativePositionIfLarge
      }
      relativePositionBuckets[i * sequenceLength + j] = Int32(relativePositionBucket)
    }
  }
  return relativePositionBuckets
}

graph.withNoGrad {
  let (reader, textModel) = UMT5ForConditionalGeneration(b: 1, t: 256)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: 256, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor2.toGPU(0)
  var attentionMask = Tensor<FloatType>(.CPU, .NCHW(1, 1, 1, 256))
  for i in 0..<256 {
    attentionMask[0, 0, 0, i] = i < tokens2.count ? 0 : -FloatType.greatestFiniteMagnitude
  }
  let attentionMaskGPU = graph.variable(attentionMask.toGPU(0))
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)
  reader(text_encoder_state_dict)
  /*
  graph.openStore("/home/liu/workspace/swift-diffusion/pile_t5_xl_encoder_f32.ckpt") {
    $0.write("text_model", model: textModel)
  }
  */
  let output = textModel(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[0]
    .as(of: Float.self)
  debugPrint(output)
}

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([2, 4, 128, 128]).to(torch.float16).cuda()
let y = torch.randn([2, 256, 2048]).to(torch.float16).cuda() * 0.01
let t = torch.full([2], 1).cuda()

print(pipeline.transformer(x, y, t))

let state_dict = pipeline.transformer.state_dict()

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
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
// debugPrint(timeEmbedding(timesteps: 1_000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000))

func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_linear1")
  let linear2 = Dense(count: intermediateSize, noBias: true, name: "\(name)_linear2")
  var out = linear1(x).swish() .* linear2(x)
  let outProjection = Dense(count: hiddenSize, noBias: true, name: "\(name)_out_proj")
  out = outProjection(out)
  return (linear1, linear2, outProjection, Model([x], [out]))
}

func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let context = Input()
  let x = Input()
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, noBias: true, name: "context_ada_ln_\($0)")
  }
  let contextChunks = contextAdaLNs.map { $0(c) }
  let contextNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  var contextOut = (1 + contextChunks[1]) .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, noBias: true, name: "c_k")
  let contextToQueries = Dense(count: k * h, noBias: true, name: "c_q")
  let contextToValues = Dense(count: k * h, noBias: true, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xAdaLNs = (0..<6).map { Dense(count: k * h, noBias: true, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  var xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  keys = keys.permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .permuted(0, 2, 1, 3)
  values = values.permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
    let unifyheads = Dense(count: k * h, noBias: true, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = x + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextLinear2: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextLinear2, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
    contextOut = context + contextChunks[5]
      .* contextFF(contextNorm2(contextOut) .* (1 + contextChunks[4]) + contextChunks[3])
  } else {
    contextLinear1 = nil
    contextLinear2 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xLinear2, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  xOut = x + xChunks[5] .* xFF(xNorm2(xOut) .* (1 + xChunks[4]) + xChunks[3])
  let reader: (PythonObject) -> Void = { state_dict in
    let attn_add_q_proj_weight = state_dict["\(prefix).attn.add_q_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    contextToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: attn_add_q_proj_weight))
    let attn_add_k_proj_weight = state_dict["\(prefix).attn.add_k_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    contextToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: attn_add_k_proj_weight))
    let attn_add_v_proj_weight = state_dict["\(prefix).attn.add_v_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    contextToValues.weight.copy(
      from: try! Tensor<Float>(numpy: attn_add_v_proj_weight)
    )
    let attn_to_q_weight = state_dict["\(prefix).attn.to_q.weight"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_q_weight))
    let attn_to_k_weight = state_dict["\(prefix).attn.to_k.weight"].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_k_weight))
    let attn_to_v_weight = state_dict["\(prefix).attn.to_v.weight"].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_v_weight))
    if let contextUnifyheads = contextUnifyheads {
      let attn_to_add_out_weight = state_dict["\(prefix).attn.to_add_out.weight"]
        .to(
          torch.float
        ).cpu().numpy()
      contextUnifyheads.weight.copy(
        from: try! Tensor<Float>(numpy: attn_to_add_out_weight))
    }
    let attn_to_out_0_weight = state_dict["\(prefix).attn.to_out.0.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_out_0_weight))
    if let contextLinear1 = contextLinear1, let contextLinear2 = contextLinear2,
      let contextOutProjection = contextOutProjection
    {
      let ff_context_linear_1_weight = state_dict["\(prefix).ff_context.linear_1.weight"].to(
        torch.float
      ).cpu().numpy()
      contextLinear1.weight.copy(
        from: try! Tensor<Float>(numpy: ff_context_linear_1_weight))
      let ff_context_linear_2_weight = state_dict["\(prefix).ff_context.linear_2.weight"].to(
        torch.float
      ).cpu().numpy()
      contextLinear2.weight.copy(
        from: try! Tensor<Float>(numpy: ff_context_linear_2_weight))
      let ff_context_out_projection_weight = state_dict[
        "\(prefix).ff_context.out_projection.weight"
      ].to(
        torch.float
      ).cpu().numpy()
      contextOutProjection.weight.copy(
        from: try! Tensor<Float>(numpy: ff_context_out_projection_weight))
    }
    let ff_linear_1_weight = state_dict["\(prefix).ff.linear_1.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.weight.copy(
      from: try! Tensor<Float>(numpy: ff_linear_1_weight))
    let ff_linear_2_weight = state_dict["\(prefix).ff.linear_2.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear2.weight.copy(
      from: try! Tensor<Float>(numpy: ff_linear_2_weight))
    let ff_out_projection_weight = state_dict["\(prefix).ff.out_projection.weight"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.weight.copy(
      from: try! Tensor<Float>(numpy: ff_out_projection_weight))
    let norm1_context_linear_weight = state_dict[
      "\(prefix).norm1_context.linear.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    for i in 0..<(contextBlockPreOnly ? 2 : 6) {
      contextAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: norm1_context_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
    }
    let norm1_linear_weight = state_dict["\(prefix).norm1.linear.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: norm1_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
    }
  }
  if !contextBlockPreOnly {
    return (reader, Model([context, x, c], [contextOut, xOut]))
  } else {
    return (reader, Model([context, x, c], [xOut]))
  }
}

func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let c = Input()
  let xAdaLNs = (0..<6).map { Dense(count: k * h, noBias: true, name: "x_ada_ln_\($0)") }
  let xChunks = xAdaLNs.map { $0(c) }
  let xNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  let xOut = (1 + xChunks[1]) .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  // Now run attention.
  keys = keys.permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(k).squareRoot()) * queries)
    .permuted(0, 2, 1, 3)
  values = values.permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * (t + hw), t + hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, (t + hw), t + hw])
  var out = dot * values
  out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  out = xUnifyheads(out)
  out = xIn + xChunks[2] .* out
  // Attentions are now. Now run MLP.
  let (xLinear1, xLinear2, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  out = xIn + xChunks[5] .* xFF(xNorm2(out) .* (1 + xChunks[4]) + xChunks[3])
  let reader: (PythonObject) -> Void = { state_dict in
    let attn_to_q_weight = state_dict["\(prefix).attn.to_q.weight"].to(
      torch.float
    ).cpu().numpy()
    xToQueries.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_q_weight))
    let attn_to_k_weight = state_dict["\(prefix).attn.to_k.weight"].to(
      torch.float
    ).cpu().numpy()
    xToKeys.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_k_weight))
    let attn_to_v_weight = state_dict["\(prefix).attn.to_v.weight"].to(
      torch.float
    ).cpu().numpy()
    xToValues.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_v_weight))
    let attn_to_out_0_weight = state_dict["\(prefix).attn.to_out.0.weight"].to(
      torch.float
    ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: try! Tensor<Float>(numpy: attn_to_out_0_weight))
    let ff_linear_1_weight = state_dict["\(prefix).ff.linear_1.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear1.weight.copy(
      from: try! Tensor<Float>(numpy: ff_linear_1_weight))
    let ff_linear_2_weight = state_dict["\(prefix).ff.linear_2.weight"].to(
      torch.float
    ).cpu().numpy()
    xLinear2.weight.copy(
      from: try! Tensor<Float>(numpy: ff_linear_2_weight))
    let ff_out_projection_weight = state_dict["\(prefix).ff.out_projection.weight"].to(
      torch.float
    ).cpu().numpy()
    xOutProjection.weight.copy(
      from: try! Tensor<Float>(numpy: ff_out_projection_weight))
    let norm1_linear_weight = state_dict["\(prefix).norm1.linear.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    for i in 0..<6 {
      xAdaLNs[i].weight.copy(
        from: try! Tensor<Float>(
          numpy: norm1_linear_weight[(k * h * i)..<(k * h * (i + 1)), ...]))
    }
  }
  return (reader, Model([x, c], [out]))
}

func MMDiT(b: Int, h: Int, w: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t = Input()
  let contextIn = Input()
  let xEmbedder = Convolution(
    groups: 1, filters: 3072, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), name: "x_embedder")
  var out = xEmbedder(x).reshaped([b, 3072, h * w]).transposed(1, 2)
  let posEmbed = Parameter<Float>(.GPU(1), .NHWC(1, 64, 64, 3072), name: "pos_embed")
  let spatialPosEmbed = posEmbed.reshaped(
    [1, h, w, 3072], offset: [0, (64 - h) / 2, (64 - w) / 2, 0],
    strides: [64 * 64 * 3072, 64 * 3072, 3072, 1]
  ).contiguous().reshaped([1, h * w, 3072])
  out = spatialPosEmbed + out
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: 3072)
  let c = tEmbedder(t).reshaped([b, 1, 3072]).swish()
  let contextEmbedder = Dense(count: 3072, noBias: true, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  let registerTokens = Parameter<Float>(.GPU(1), .HWC(1, 8, 3072), name: "register_tokens")
  context = Functional.concat(
    axis: 1, Concat(axis: 0)(Array(repeating: registerTokens, count: b)), context)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<4 {
    let (reader, block) = JointTransformerBlock(
      prefix: "joint_transformer_blocks.\(i)", k: 256, h: 12, b: b, t: 264, hw: h * w,
      contextBlockPreOnly: false)
    let blockOut = block(context, out, c)
    context = blockOut[0]
    out = blockOut[1]
    readers.append(reader)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<32 {
    let (reader, block) = SingleTransformerBlock(
      prefix: "single_transformer_blocks.\(i)", k: 256, h: 12, b: b, t: 264, hw: h * w,
      contextBlockPreOnly: i == 31)
    out = block(out, c)
    readers.append(reader)
  }
  let scale = Dense(count: 3072, name: "ada_ln_0")
  let shift = Dense(count: 3072, name: "ada_ln_1")
  out = (1 + scale(c)) .* out + shift(c)
  let projOut = Dense(count: 2 * 2 * 4, name: "linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([b, h, w, 2, 2, 4]).permuted(0, 5, 1, 3, 2, 4).contiguous().reshaped([
    b, 4, h * 2, w * 2,
  ])
  let reader: (PythonObject) -> Void = { state_dict in
    let x_embedder_proj_weight = state_dict["pos_embed.proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let x_embedder_proj_bias = state_dict["pos_embed.proj.bias"].to(torch.float)
      .cpu().numpy()
    xEmbedder.weight.copy(from: try! Tensor<Float>(numpy: x_embedder_proj_weight))
    xEmbedder.bias.copy(from: try! Tensor<Float>(numpy: x_embedder_proj_bias))
    let pos_embed = state_dict["pos_embed.pos_embed"].to(torch.float).cpu().numpy()
    posEmbed.weight.copy(from: try! Tensor<Float>(numpy: pos_embed))
    let t_embedder_mlp_0_weight = state_dict["time_step_proj.linear_1.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_0_bias = state_dict["time_step_proj.linear_1.bias"].to(torch.float)
      .cpu().numpy()
    tMlp0.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_weight))
    tMlp0.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_0_bias))
    let t_embedder_mlp_2_weight = state_dict["time_step_proj.linear_2.weight"].to(
      torch.float
    ).cpu().numpy()
    let t_embedder_mlp_2_bias = state_dict["time_step_proj.linear_2.bias"].to(torch.float)
      .cpu().numpy()
    tMlp2.weight.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_weight))
    tMlp2.bias.copy(from: try! Tensor<Float>(numpy: t_embedder_mlp_2_bias))
    let context_embedder_weight = state_dict["context_embedder.weight"].to(
      torch.float
    ).cpu().numpy()
    contextEmbedder.weight.copy(from: try! Tensor<Float>(numpy: context_embedder_weight))
    let register_tokens_weight = state_dict["register_tokens"].to(
      torch.float
    ).cpu().numpy()
    registerTokens.weight.copy(from: try! Tensor<Float>(numpy: register_tokens_weight))
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_linear_weight = state_dict[
      "norm_out.linear.weight"
    ].to(
      torch.float
    ).cpu().numpy()
    scale.weight.copy(
      from: try! Tensor<Float>(numpy: norm_out_linear_weight[0..<3072, ...]))
    shift.weight.copy(
      from: try! Tensor<Float>(numpy: norm_out_linear_weight[3072..<(3072 * 2), ...]))
    let proj_out_weight = state_dict["proj_out.weight"].to(
      torch.float
    ).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
  }
  return (reader, Model([x, t, contextIn], [out]))
}

let (reader, dit) = MMDiT(b: 2, h: 64, w: 64)

graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy()).toGPU(1))
  let tTensor = graph.variable(
    timeEmbedding(timesteps: 1000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000).toGPU(1))
  let cTensor = graph.variable(
    try! Tensor<Float>(numpy: y.to(torch.float).cpu().numpy()).toGPU(1))
  dit.compile(inputs: xTensor, tTensor, cTensor)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, tTensor, cTensor))
  graph.openStore("/home/liu/workspace/swift-diffusion/auraflow_v0.1_f32.ckpt") {
    $0.write("dit", model: dit)
  }
}
