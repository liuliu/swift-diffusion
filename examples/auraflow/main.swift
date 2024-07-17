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

print(pipeline.text_encoder)
print(pipeline.transformer)

let text_encoder_state_dict = pipeline.text_encoder.state_dict()
let image = pipeline(
  prompt: "an astronaut.",
  height: 1024,
  width: 1024,
  num_inference_steps: 2,
  generator: torch.Generator().manual_seed(666),
  guidance_scale: 3.5
).images[0]

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
  graph.openStore("/home/liu/workspace/swift-diffusion/flan_t5_xl_encoder_f32.ckpt") {
    $0.write("text_model", model: textModel)
  }
  let output = textModel(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[0]
    .as(of: Float.self)
  debugPrint(output)
}
