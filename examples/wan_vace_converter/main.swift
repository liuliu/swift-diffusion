import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit
import SentencePiece

let graph = DynamicGraph()

let torch = Python.import("torch")

torch.set_grad_enabled(false)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let wan = Python.import("wan")
let wan_utils_utils = Python.import("wan.utils.utils")
let Image = Python.import("PIL.Image")
let torch_nn_functional = Python.import("torch.nn.functional")
let torchvision_transforms_functional = Python.import("torchvision.transforms.functional")

let cfg = wan.configs.WAN_CONFIGS["t2v-14B"]
let wan_vace = wan.WanVace(
  config: cfg,
  checkpoint_dir: "/home/liu/workspace/Wan2.1/Wan2.1-VACE-14B",
  device_id: 0,
  rank: 0,
  t5_fsdp: false,
  dit_fsdp: false,
  use_usp: false,
  t5_cpu: false
)

wan_vace.text_encoder.model.cpu()

let text_encoder_state_dict = wan_vace.text_encoder.model.state_dict()

let prompt =
  "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/Wan2.1/Wan2.1-VACE-1.3B/google/umt5-xxl/spiece.model")
var tokens2 = sentencePiece.encode(prompt).map { return $0.id }
tokens2.append(1)
print(tokens2)

func UMT5TextEmbedding(vocabularySize: Int, embeddingSize: Int, name: String) -> Model {
  let tokenEmbed = Embedding(
    Float16.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: name)
  return tokenEmbed
}

func UMT5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int, upcast: Bool) -> (
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
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, positionBias], [out]))
}

func UMT5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int, upcast: Bool) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true, name: "w0")
  let wi_1 = Dense(count: intermediateSize, noBias: true, name: "w1")
  var out = wi_1(x) .* wi_0(x).GELU(approximate: .tanh)
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  out = wo(out)
  return (wi_0, wi_1, wo, Model([x], [out]))
}

func UMT5Block(
  prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int,
  upcast: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let relativePositionEmbedding = Embedding(
    Float16.self, vocabularySize: 32, embeddingSize: 64, name: "relative_position_embedding")
  let positionBias =
    relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, 64])
    .permuted(0, 3, 1, 2) + attentionMask
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (tokeys, toqueries, tovalues, unifyheads, attention) = UMT5LayerSelfAttention(
    k: k, h: h, b: b, t: t, outFeatures: outFeatures, upcast: upcast)
  let scaleFactor: Float = 8
  var out: Model.IO
  if upcast {
    out = x + scaleFactor * attention(norm1(x).to(.Float16), positionBias).to(of: x)
  } else {
    out = x + attention(norm1(x).to(.Float16), positionBias).to(of: x)
  }
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (wi_0, wi_1, wo, ff) = UMT5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize, upcast: upcast)
  if upcast {
    out = out + scaleFactor * ff(norm2(out).to(.Float16)).to(of: out)
  } else {
    out = out + ff(norm2(out).to(.Float16)).to(of: out)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let relative_attention_bias_weight = state_dict[
      "\(prefix).pos_embedding.embedding.weight"
    ].cpu().float().numpy()
    relativePositionEmbedding.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: relative_attention_bias_weight)))
    let layer_0_layer_norm_weight = state_dict["\(prefix).norm1.weight"].cpu().float()
      .numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: layer_0_layer_norm_weight))
    let k_weight = state_dict["\(prefix).attn.k.weight"].cpu().float().numpy()
    tokeys.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: k_weight)))
    let q_weight = state_dict["\(prefix).attn.q.weight"].cpu().float().numpy()
    toqueries.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: q_weight)))
    let v_weight = state_dict["\(prefix).attn.v.weight"].cpu().float().numpy()
    tovalues.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: v_weight)))
    let o_weight = state_dict["\(prefix).attn.o.weight"].cpu().float().numpy()
    unifyheads.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: o_weight)))
    let layer_1_layer_norm_weight = state_dict["\(prefix).norm2.weight"].cpu().float()
      .numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: layer_1_layer_norm_weight))
    let wi_0_weight = state_dict["\(prefix).ffn.gate.0.weight"].cpu().float()
      .numpy()
    wi_0.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: wi_0_weight)))
    let wi_1_weight = state_dict["\(prefix).ffn.fc1.weight"].cpu().float()
      .numpy()
    wi_1.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: wi_1_weight)))
    let wo_weight = state_dict["\(prefix).ffn.fc2.weight"].cpu().float().numpy()
    wo.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: wo_weight)))
  }
  return (reader, Model([x, attentionMask, relativePositionBuckets], [out]))
}

func UMT5ForConditionalGeneration(b: Int, t: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let textEmbed = UMT5TextEmbedding(vocabularySize: 256_384, embeddingSize: 4_096, name: "shared")
  var out = textEmbed(x).to(.Float32)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<24 {
    let (reader, block) = UMT5Block(
      prefix: "blocks.\(i)", k: 64, h: 64, b: b, t: t, outFeatures: 4_096,
      intermediateSize: 10_240, upcast: i >= 12)
    out = block(out, attentionMask, relativePositionBuckets)
    readers.append(reader)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out).to(.Float16)
  let reader: (PythonObject) -> Void = { state_dict in
    let vocab = state_dict["token_embedding.weight"].cpu().float().numpy()
    textEmbed.weight.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vocab)))
    for reader in readers {
      reader(state_dict)
    }
    let final_layer_norm_weight = state_dict["norm.weight"].cpu().float()
      .numpy()
    finalNorm.weight.copy(from: try! Tensor<Float>(numpy: final_layer_norm_weight))
  }
  return (reader, Model([x, attentionMask, relativePositionBuckets], [out]))
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

torch.set_autocast_enabled("cuda", true)
torch.set_autocast_dtype("cuda", torch.bfloat16)

let x = torch.randn([16, 21, 60, 104]).to(torch.float).cuda()
let vace_c = torch.randn([96, 21, 60, 104]).to(torch.float).cuda()
let t = torch.tensor([900], dtype: torch.float, device: torch.device("cuda:0"))
let context = torch.randn([28, 4096]).to(torch.float).cuda()
print(wan_vace.model)
wan_vace.model.half().to(torch.device("cuda:0"))
let out = wan_vace.model([x], t, [vace_c], [context], 21 * 30 * 52)
// print(out)

let wan_state_dict = wan_vace.model.state_dict()

func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

func WanAttentionBlock(
  prefix: String, weightsPrefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let context = Input()
  let c = (0..<6).map { _ in Input() }
  let rot = Input()
  let modulations = (0..<6).map {
    Parameter<Float>(.GPU(2), .HWC(1, 1, k * h), name: "\(weightsPrefix)attn_ada_ln_\($0)")
  }
  let chunks = zip(c, modulations).map { $0 + $1 }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = ((1 + chunks[1]) .* xNorm1(x) + chunks[0]).to(.Float16)
  let xToKeys = Dense(count: k * h, name: "\(weightsPrefix)x_k")
  let xToQueries = Dense(count: k * h, name: "\(weightsPrefix)x_q")
  let xToValues = Dense(count: k * h, name: "\(weightsPrefix)x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  let queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xQ, right: rot)
  let keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xK, right: rot)
  let values = xV
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
  let xUnifyheads = Dense(count: k * h, name: "\(weightsPrefix)x_o")
  out = xUnifyheads(out)
  out = x + chunks[2] .* out.to(of: x)
  let xNorm3 = LayerNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_3")
  xOut = xNorm3(out).to(.Float16)
  let contextToKeys = Dense(count: k * h, name: "\(weightsPrefix)c_k")
  let xToContextQueries = Dense(count: k * h, name: "\(weightsPrefix)x_c_q")
  let contextToValues = Dense(count: k * h, name: "\(weightsPrefix)c_v")
  var cK = contextToKeys(context)
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)c_norm_k")
  cK = contextNormK(cK).reshaped([b, t, h, k])
  var cQ = xToContextQueries(xOut)
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_c_norm_q")
  cQ = contextNormQ(cQ).reshaped([b, hw, h, k])
  let cV = contextToValues(context).reshaped([b, t, h, k])
  let crossAttention = ScaledDotProductAttention(
    scale: 1 / Float(k).squareRoot(), flags: [.Float16])
  let crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  let contextUnifyheads = Dense(count: k * h, name: "\(weightsPrefix)c_o")
  out = out + contextUnifyheads(crossOut).to(of: out)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, upcast: false, name: "\(weightsPrefix)x")
  out =
    out + xFF(((1 + chunks[4]) .* xNorm2(out) + chunks[3]).to(.Float16)).to(of: out) .* chunks[5]
  let reader: (PythonObject) -> Void = { state_dict in
    let modulation_bias = state_dict["\(prefix).modulation"]
      .to(
        torch.float
      ).cpu().numpy()
    for i in 0..<6 {
      modulations[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: modulation_bias[0..<1, i..<(i + 1), 0..<(k * h)])))
    }
    let self_attn_q_weight = state_dict["\(prefix).self_attn.q.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let self_attn_q_bias = state_dict["\(prefix).self_attn.q.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    xToQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_q_weight)))
    xToQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_q_bias)))
    let self_attn_k_weight = state_dict["\(prefix).self_attn.k.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let self_attn_k_bias = state_dict["\(prefix).self_attn.k.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    xToKeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_k_weight)))
    xToKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_k_bias)))
    let self_attn_v_weight = state_dict["\(prefix).self_attn.v.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let self_attn_v_bias = state_dict["\(prefix).self_attn.v.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    xToValues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_v_weight)))
    xToValues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_v_bias)))
    let self_attn_norm_k_weight = state_dict["\(prefix).self_attn.norm_k.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    normK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_norm_k_weight)))
    let self_attn_norm_q_weight = state_dict["\(prefix).self_attn.norm_q.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    normQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_norm_q_weight)))
    let self_attn_o_weight = state_dict["\(prefix).self_attn.o.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let self_attn_o_bias = state_dict["\(prefix).self_attn.o.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    xUnifyheads.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_o_weight)))
    xUnifyheads.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: self_attn_o_bias)))
    let cross_attn_q_weight = state_dict["\(prefix).cross_attn.q.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let cross_attn_q_bias = state_dict["\(prefix).cross_attn.q.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    xToContextQueries.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_q_weight)))
    xToContextQueries.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_q_bias)))
    let cross_attn_k_weight = state_dict["\(prefix).cross_attn.k.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let cross_attn_k_bias = state_dict["\(prefix).cross_attn.k.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    contextToKeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_k_weight)))
    contextToKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_k_bias)))
    let cross_attn_v_weight = state_dict["\(prefix).cross_attn.v.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let cross_attn_v_bias = state_dict["\(prefix).cross_attn.v.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    contextToValues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_v_weight)))
    contextToValues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_v_bias)))
    let cross_attn_norm_k_weight = state_dict["\(prefix).cross_attn.norm_k.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    contextNormK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_norm_k_weight)))
    let cross_attn_norm_q_weight = state_dict["\(prefix).cross_attn.norm_q.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    contextNormQ.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_norm_q_weight)))
    let cross_attn_o_weight = state_dict["\(prefix).cross_attn.o.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let cross_attn_o_bias = state_dict["\(prefix).cross_attn.o.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    contextUnifyheads.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_o_weight)))
    contextUnifyheads.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_o_bias)))
    let norm3_weight = state_dict["\(prefix).norm3.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let norm3_bias = state_dict["\(prefix).norm3.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    xNorm3.weight.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm3_weight)))
    xNorm3.bias.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: norm3_bias)))
    let ffn_0_weight = state_dict["\(prefix).ffn.0.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let ffn_0_bias = state_dict["\(prefix).ffn.0.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    xLinear1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ffn_0_weight)))
    xLinear1.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ffn_0_bias)))
    let ffn_2_weight = state_dict["\(prefix).ffn.2.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let ffn_2_bias = state_dict["\(prefix).ffn.2.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    xOutProjection.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ffn_2_weight)))
    xOutProjection.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: ffn_2_bias)))
  }
  return (reader, Model([x, context, rot] + c, [out]))
}

func TimeEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func MLPProj(inChannels: Int, outChannels: Int, name: String) -> (Model, Model, Model, Model, Model)
{
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_0")
  let fc0 = Dense(count: inChannels, name: "\(name)_embedder_0")
  var out = fc0(ln1(x)).GELU()
  let fc2 = Dense(count: outChannels, name: "\(name)_embedder_1")
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_1")
  out = ln2(fc2(out))
  return (ln1, ln2, fc0, fc2, Model([x], [out]))
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

func WanVace(
  channels: Int, layers: Int, vaceLayers: [Int], intermediateSize: Int, time: Int, height: Int,
  width: Int,
  textLength: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let v = Input()
  let imgIn = Dense(count: channels, name: "x_embedder")
  let xIn = imgIn(x)
  let vIn = Dense(count: channels, name: "vace_x_embedder")
  var out = vIn(v)
  let beforeProj = Dense(count: channels, name: "vace_before_proj")
  out = beforeProj(out) + xIn
  let txt = Input()
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: channels, name: "c")
  let context = contextEmbedder(txt)
  let t = Input()
  let rot = Input()
  let (timeInMlp0, timeInMlp2, timeIn) = TimeEmbedder(channels: channels, name: "t")
  let vector = timeIn(t).reshaped([1, 1, channels])
  let vectorIn = vector.swish()
  let timeProjections = (0..<6).map { Dense(count: channels, name: "vace_ada_ln_\($0)") }
  let tOut = timeProjections.map { $0(vectorIn) }
  let h = height / 2
  let w = width / 2
  var readers = [(PythonObject) -> Void]()
  out = out.to(.Float32)
  var hints = [Int: Model.IO]()
  var lastHint: Model.IO? = nil
  for (i, n) in vaceLayers.enumerated() {
    let (reader, block) = WanAttentionBlock(
      prefix: "vace_blocks.\(i)", weightsPrefix: "vace_", k: 128, h: channels / 128, b: 1,
      t: textLength, hw: time * h * w, intermediateSize: intermediateSize)
    out = block([out, context, rot] + tOut)
    if let lastHint = lastHint {
      out.add(dependencies: [lastHint])
    }
    readers.append(reader)
    let afterProj = Dense(count: channels, name: "vace_after_proj")
    lastHint = afterProj(out)
    hints[n] = lastHint
    readers.append({
      let after_proj_weight = $0["vace_blocks.\(i).after_proj.weight"].to(
        torch.float
      ).cpu().numpy()
      let after_proj_bias = $0["vace_blocks.\(i).after_proj.bias"].to(torch.float)
        .cpu().numpy()
      afterProj.weight.copy(
        from: Tensor<Float>(from: try! Tensor<Float>(numpy: after_proj_weight)))
      afterProj.bias.copy(
        from: Tensor<Float>(from: try! Tensor<Float>(numpy: after_proj_bias)))
    })
  }
  out = xIn.to(.Float32)
  if let lastHint = lastHint {
    out.add(dependencies: [lastHint])
  }
  for i in 0..<layers {
    let (reader, block) = WanAttentionBlock(
      prefix: "blocks.\(i)", weightsPrefix: "", k: 128, h: channels / 128, b: 1, t: textLength,
      hw: time * h * w, intermediateSize: intermediateSize)
    out = block([out, context, rot] + tOut)
    readers.append(reader)
    if let hint = hints[i] {
      out = out + hint
    }
  }
  let scale = Parameter<Float>(.GPU(2), .HWC(1, 1, channels), name: "ada_ln_0")
  let shift = Parameter<Float>(.GPU(2), .HWC(1, 1, channels), name: "ada_ln_1")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = ((1 + scale + vector) .* normFinal(out) + (vector + shift)).to(.Float16)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let patch_embedding_weight = state_dict["patch_embedding.weight"].to(
      torch.float
    ).cpu().numpy()
    let patch_embedding_bias = state_dict["patch_embedding.bias"].to(torch.float)
      .cpu().numpy()
    imgIn.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: patch_embedding_weight)))
    imgIn.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: patch_embedding_bias)))
    let vace_patch_embedding_weight = state_dict["vace_patch_embedding.weight"].to(
      torch.float
    ).cpu().numpy()
    let vace_patch_embedding_bias = state_dict["vace_patch_embedding.bias"].to(torch.float)
      .cpu().numpy()
    vIn.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vace_patch_embedding_weight)))
    vIn.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vace_patch_embedding_bias)))
    let vace_blocks_0_before_proj_weight = state_dict["vace_blocks.0.before_proj.weight"].to(
      torch.float
    ).cpu().numpy()
    let vace_blocks_0_before_proj_bias = state_dict["vace_blocks.0.before_proj.bias"].to(
      torch.float
    )
    .cpu().numpy()
    beforeProj.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vace_blocks_0_before_proj_weight)))
    beforeProj.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: vace_blocks_0_before_proj_bias)))
    let text_embedding_0_weight = state_dict["text_embedding.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let text_embedding_0_bias = state_dict["text_embedding.0.bias"].to(torch.float)
      .cpu().numpy()
    cLinear1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: text_embedding_0_weight)))
    cLinear1.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: text_embedding_0_bias)))
    let text_embedding_2_weight = state_dict["text_embedding.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let text_embedding_2_bias = state_dict["text_embedding.2.bias"].to(torch.float)
      .cpu().numpy()
    cLinear2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: text_embedding_2_weight)))
    cLinear2.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: text_embedding_2_bias)))
    let time_embedding_0_weight = state_dict["time_embedding.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let time_embedding_0_bias = state_dict["time_embedding.0.bias"].to(torch.float)
      .cpu().numpy()
    timeInMlp0.weight.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: time_embedding_0_weight)))
    timeInMlp0.bias.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: time_embedding_0_bias)))
    let time_embedding_2_weight = state_dict["time_embedding.2.weight"].to(
      torch.float
    ).cpu().numpy()
    let time_embedding_2_bias = state_dict["time_embedding.2.bias"].to(torch.float)
      .cpu().numpy()
    timeInMlp2.weight.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: time_embedding_2_weight)))
    timeInMlp2.bias.copy(
      from: Tensor<Float>(from: try! Tensor<Float>(numpy: time_embedding_2_bias)))
    let time_projection_1_weight = state_dict["time_projection.1.weight"].to(
      torch.float
    ).cpu().numpy()
    let time_projection_1_bias = state_dict["time_projection.1.bias"].to(torch.float)
      .cpu().numpy()
    for i in 0..<6 {
      timeProjections[i].weight.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: time_projection_1_weight[(i * channels)..<((i + 1) * channels), ...])))
      timeProjections[i].bias.copy(
        from: Tensor<Float>(
          from: try! Tensor<Float>(
            numpy: time_projection_1_bias[(i * channels)..<((i + 1) * channels)])))
    }
    for reader in readers {
      reader(state_dict)
    }
    let modulation_bias = state_dict["head.modulation"].to(torch.float)
      .cpu().numpy()
    shift.weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(
          numpy: modulation_bias[0..<1, 0..<1, 0..<channels])))
    scale.weight.copy(
      from: Tensor<Float>(
        from: try! Tensor<Float>(
          numpy: modulation_bias[0..<1, 1..<2, 0..<channels])))
    let head_head_weight = state_dict["head.head.weight"].to(torch.float)
      .cpu().numpy()
    let head_head_bias = state_dict["head.head.bias"].to(torch.float)
      .cpu().numpy()
    projOut.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: head_head_weight)))
    projOut.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: head_head_bias)))
  }
  return (Model([x, v, txt, t, rot], [out]), reader)
}

graph.withNoGrad {
  var rotNdTensor = graph.variable(.CPU, .NHWC(1, 21 * 30 * 52, 1, 128), of: Float.self)
  for t in 0..<21 {
    for y in 0..<30 {
      for x in 0..<52 {
        let i = t * 30 * 52 + y * 52 + x
        for k in 0..<22 {
          let theta = Double(t) * 1.0 / pow(10_000, Double(k) / 22)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, k * 2] = Float(costheta)
          rotNdTensor[0, i, 0, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<21 {
          let theta = Double(y) * 1.0 / pow(10_000, Double(k) / 21)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, (k + 22) * 2] = Float(costheta)
          rotNdTensor[0, i, 0, (k + 22) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<21 {
          let theta = Double(x) * 1.0 / pow(10_000, Double(k) / 21)
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotNdTensor[0, i, 0, (k + 22 + 21) * 2] = Float(costheta)
          rotNdTensor[0, i, 0, (k + 22 + 21) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  let (vace, vaceReader) = WanVace(
    // channels: 1536, layers: 30, vaceLayers: (0..<15).map { $0 * 2 }, intermediateSize: 8960, time: 21, height: 60, width: 104,
    channels: 5120, layers: 40, vaceLayers: (0..<8).map { $0 * 5 }, intermediateSize: 13824,
    time: 21, height: 60, width: 104,
    textLength: 512)
  let vaceTensor = graph.variable(
    Tensor<Float16>(
      from: try! Tensor<Float>(numpy: vace_c.to(torch.float).cpu().numpy())
    ).toGPU(2)
  ).reshaped(format: .NHWC, shape: [1, 96, 21 * 30, 2, 52, 2]).permuted(0, 2, 4, 1, 3, 5).copied()
    .reshaped(format: .NHWC, shape: [1, 21 * 30 * 52, 96 * 2 * 2])
  let xTensor = graph.variable(
    Tensor<Float16>(
      from: try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy())
    ).toGPU(2)
  ).reshaped(format: .NHWC, shape: [1, 16, 21 * 30, 2, 52, 2]).permuted(0, 2, 4, 1, 3, 5).copied()
    .reshaped(format: .NHWC, shape: [1, 21 * 30 * 52, 16 * 2 * 2])
  let txt = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: context.to(torch.float).cpu().numpy()))
      .reshaped(.WC(28, 4096)).toGPU(2))
  var txtIn = graph.variable(.GPU(2), .WC(512, 4096), of: Float16.self)
  txtIn.full(0)
  txtIn[0..<28, 0..<4096] = txt
  txtIn = txtIn.reshaped(.HWC(1, 512, 4096))
  let timestep = timeEmbedding(timesteps: 900, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let tGPU = graph.variable(Tensor<Float>(from: timestep)).toGPU(2)
  let rotNdTensorGPU = DynamicGraph.Tensor<Float16>(from: rotNdTensor).toGPU(2)
  vace.compile(inputs: xTensor, vaceTensor, txtIn, tGPU, rotNdTensorGPU)
  vaceReader(wan_state_dict)
  debugPrint(vace(inputs: xTensor, vaceTensor, txtIn, tGPU, rotNdTensorGPU))
  graph.openStore("/home/liu/workspace/swift-diffusion/wan_v2.1_14b_vace_f16.ckpt") {
    $0.write("dit", model: vace)
  }
}
