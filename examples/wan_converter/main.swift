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

let cfg = wan.configs.WAN_CONFIGS["i2v-14B"]
let wan_i2v = wan.WanI2V(
  config: cfg,
  checkpoint_dir: "/home/liu/workspace/Wan2.1/Wan2.1-I2V-14B-720P",
  device_id: 0,
  rank: 0,
  t5_fsdp: false,
  dit_fsdp: false,
  use_usp: false,
  t5_cpu: false
)

var img = Image.open("/home/liu/workspace/Wan2.1/examples/i2v_input.JPG").convert("RGB")
img = torchvision_transforms_functional.to_tensor(img).sub_(0.5).div_(0.5).to(wan_i2v.device)

let visual_state_dict = wan_i2v.clip.model.visual.state_dict()
/*
img = img[Python.slice(Python.None, Python.None, Python.None), Python.None, Python.slice(Python.None, Python.None, Python.None), Python.slice(Python.None, Python.None, Python.None)]
let clip_context = wan_i2v.clip.visual([img])
let resized = torch_nn_functional.interpolate(img.transpose(0, 1), size: [224, 224], mode: "bicubic", align_corners: false)
let transformed = wan_i2v.clip.transforms.transforms[-1](resized.mul_(0.5).add_(0.5))
print(transformed.shape)
print(clip_context)
print(visual_state_dict.keys())
*/

func CLIPSelfAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, tokeys, tovalues, unifyheads, Model([x], [out]))
}

func CLIPResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let (toqueries, tokeys, tovalues, unifyheads, attention) = CLIPSelfAttention(
    k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu().numpy()
    ln1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    ln1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let in_proj_weight = state_dict["\(prefix).attn.to_qkv.weight"].type(torch.float).cpu().numpy()
    let in_proj_bias = state_dict["\(prefix).attn.to_qkv.bias"].type(torch.float).cpu().numpy()
    toqueries.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[..<(k * h), ...]))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_proj_bias[..<(k * h)]))
    tokeys.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(k * h)..<(2 * k * h), ...]))
    tokeys.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(k * h)..<(2 * k * h)]))
    tovalues.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(2 * k * h)..., ...]))
    tovalues.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(2 * k * h)...]))
    let out_proj_weight = state_dict["\(prefix).attn.proj.weight"].type(torch.float).cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).attn.proj.bias"].type(torch.float).cpu().numpy()
    unifyheads.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_proj_bias))
    let ln_2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu().numpy()
    ln2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_2_weight))
    ln2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_2_bias))
    let c_fc_weight = state_dict["\(prefix).mlp.0.weight"].type(torch.float).cpu().numpy()
    let c_fc_bias = state_dict["\(prefix).mlp.0.bias"].type(torch.float).cpu().numpy()
    fc.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: c_fc_weight))
    fc.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: c_fc_bias))
    let c_proj_weight = state_dict["\(prefix).mlp.2.weight"].type(torch.float).cpu().numpy()
    let c_proj_bias = state_dict["\(prefix).mlp.2.bias"].type(torch.float).cpu().numpy()
    proj.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: c_proj_weight))
    proj.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: c_proj_bias))
  }
  return (reader, Model([x], [out]))
}

func VisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type,
  grid: Int, width: Int, layers: Int, heads: Int, batchSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let classEmbedding = Parameter<T>(.GPU(0), .HWC(1, 1, width))
  let positionalEmbedding = Parameter<T>(.GPU(0), .HWC(1, grid * grid + 1, width))
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], noBias: true,
    hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPre(out)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (reader, block) = CLIPResidualAttentionBlock(
      prefix: "transformer.\(i)", k: width / heads, h: heads, b: batchSize,
      t: grid * grid + 1)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
    readers.append(reader)
  }
  let lnPost = LayerNorm(epsilon: 1e-5, axis: [1], name: "post_layernorm")
  out = lnPost(out.reshaped([batchSize, width], strides: [width, 1]))
  let reader: (PythonObject) -> Void = { state_dict in
    let class_embedding = state_dict["cls_embedding"].type(torch.float).cpu().numpy()
    classEmbedding.weight.copy(from: try! Tensor<Float>(numpy: class_embedding))
    let positional_embedding = state_dict["pos_embedding"].type(torch.float).cpu().numpy()
    positionalEmbedding.weight.copy(from: try! Tensor<Float>(numpy: positional_embedding))
    let conv1_weight = state_dict["patch_embedding.weight"].type(torch.float).cpu().numpy()
    conv1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let ln_pre_weight = state_dict["pre_norm.weight"].type(torch.float).cpu().numpy()
    let ln_pre_bias = state_dict["pre_norm.bias"].type(torch.float).cpu().numpy()
    lnPre.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_pre_weight))
    lnPre.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_pre_bias))
    for reader in readers {
      reader(state_dict)
    }
    let ln_post_weight = state_dict["post_norm.weight"].type(torch.float).cpu().numpy()
    let ln_post_bias = state_dict["post_norm.bias"].type(torch.float).cpu().numpy()
    lnPost.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_post_weight))
    lnPost.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_post_bias))
  }
  return (reader, Model([x], [out]))
}

graph.withNoGrad {
  /*
  let imgTensor = graph.variable(try! Tensor<Float>(numpy: transformed.cpu().numpy())).toGPU(0)
  let (reader, vit) = VisionTransformer(
    Float.self,
    grid: 16, width: 1280, layers: 32, heads: 16, batchSize: 1)
  let _ = vit(inputs: imgTensor)
  reader(visual_state_dict)
  let out = vit(inputs: imgTensor)[0].as(of: Float.self).toCPU()
  debugPrint(out)
  // graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_xlm_roberta_large_vit_h14_f32.ckpt") {
  //   $0.write("vision_model", model: vit)
  // }
  */
}

wan_i2v.text_encoder.model.cpu()

let text_encoder_state_dict = wan_i2v.text_encoder.model.state_dict()

let prompt =
  "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

let sentencePiece = SentencePiece(
  file: "/home/liu/workspace/Wan2.1/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model")
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

graph.withNoGrad {
  /*
  let tokensTensor2 = graph.variable(.CPU, .C(256), of: Int32.self)
  for i in 0..<256 {
    tokensTensor2[i] = i < tokens2.count ? tokens2[i] : 0
  }
  let (reader, textModel) = UMT5ForConditionalGeneration(b: 1, t: 256)
  let relativePositionBuckets = relativePositionBuckets(
    sequenceLength: 256, numBuckets: 32, maxDistance: 128)
  let tokensTensorGPU = tokensTensor2.toGPU(0)
  var attentionMask = Tensor<Float16>(.CPU, .NCHW(1, 1, 1, 256))
  for i in 0..<256 {
    attentionMask[0, 0, 0, i] = i < tokens2.count ? 0 : -Float16.greatestFiniteMagnitude
  }
  let attentionMaskGPU = graph.variable(attentionMask.toGPU(0))
  let relativePositionBucketsGPU = graph.variable(relativePositionBuckets.toGPU(0))
  textModel.compile(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)
  reader(text_encoder_state_dict)
  let output = textModel(inputs: tokensTensorGPU, attentionMaskGPU, relativePositionBucketsGPU)[0]
    .as(of: Float16.self)
  graph.openStore("/home/liu/workspace/swift-diffusion/umt5_xxl_encoder_f16.ckpt") {
    $0.write("text_model", model: textModel)
  }
  debugPrint(output[0..<tokens2.count, 0..<4096])
  */
}

torch.set_autocast_enabled("cuda", true)
torch.set_autocast_dtype("cuda", torch.bfloat16)

let x = torch.randn([16, 21, 60, 104]).to(torch.float).cuda()
let y = torch.randn([20, 21, 60, 104]).to(torch.float).cuda()
let t = torch.tensor([900], dtype: torch.float, device: torch.device("cuda:0"))
let context = torch.randn([28, 4096]).to(torch.float).cuda()
let clip_fea = torch.randn([1, 257, 1280]).to(torch.float).cuda()
print(wan_i2v.model)
/*
wan_i2v.model.half().to(torch.device("cuda:0"))
let out = wan_i2v.model([x], t, [context], 21 * 30 * 52, clip_fea, [y])
*/
// print(out)

let wan_state_dict = wan_i2v.model.state_dict()
print(wan_state_dict.keys())

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
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let context = Input()
  let contextImg = Input()
  let c = (0..<6).map { _ in Input() }
  let rot = Input()
  let modulations = (0..<6).map {
    Parameter<Float>(.GPU(2), .HWC(1, 1, k * h), name: "attn_ada_ln_\($0)")
  }
  let chunks = zip(c, modulations).map { $0 + $1 }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = ((1 + chunks[1]) .* xNorm1(x) + chunks[0]).to(.Float16)
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  let queries = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xQ, right: rot)
  let keys = (1 / Float(k).squareRoot().squareRoot()) * Functional.cmul(left: xK, right: rot)
  let values = xV
  // Now run attention.
  let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
  var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  out = xUnifyheads(out)
  out = x + chunks[2] .* out.to(of: x)
  let xNorm3 = LayerNorm(epsilon: 1e-6, axis: [2], name: "x_norm_3")
  xOut = xNorm3(out).to(.Float16)
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let xToContextQueries = Dense(count: k * h, name: "x_c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var cK = contextToKeys(context)
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "c_norm_k")
  cK = contextNormK(cK).reshaped([b, t, h, k])
  var cQ = xToContextQueries(xOut)
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "x_c_norm_q")
  cQ = contextNormQ(cQ).reshaped([b, hw, h, k])
  let cV = contextToValues(context).reshaped([b, t, h, k])
  let crossAttention = ScaledDotProductAttention(
    scale: 1 / Float(k).squareRoot(), flags: [.Float16])
  let crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  let contextImgToKeys = Dense(count: k * h, name: "c_img_k")
  let contextImgToValues = Dense(count: k * h, name: "c_img_v")
  var cImgK = contextImgToKeys(contextImg)
  let contextImgNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "c_img_norm_k")
  cImgK = contextImgNormK(cImgK).reshaped([b, 257, h, k])
  let cImgV = contextImgToValues(contextImg).reshaped([b, 257, h, k])
  let crossAttentionImg = ScaledDotProductAttention(
    scale: 1 / Float(k).squareRoot(), flags: [.Float16])
  let crossOutImg = crossAttentionImg(cQ, cImgK, cImgV).reshaped([b, hw, k * h])
  crossOutImg.add(dependencies: [crossOut])
  let contextUnifyheads = Dense(count: k * h, name: "c_o")
  out = out + contextUnifyheads(crossOut + crossOutImg).to(of: out)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, upcast: false, name: "x")
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
    let cross_attn_k_img_weight = state_dict["\(prefix).cross_attn.k_img.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let cross_attn_k_img_bias = state_dict["\(prefix).cross_attn.k_img.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    contextImgToKeys.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_k_img_weight)))
    contextImgToKeys.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_k_img_bias)))
    let cross_attn_v_img_weight = state_dict["\(prefix).cross_attn.v_img.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    let cross_attn_v_img_bias = state_dict["\(prefix).cross_attn.v_img.bias"]
      .to(
        torch.float
      ).cpu().numpy()
    contextImgToValues.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_v_img_weight)))
    contextImgToValues.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_v_img_bias)))
    let cross_attn_norm_k_img_weight = state_dict["\(prefix).cross_attn.norm_k_img.weight"]
      .to(
        torch.float
      ).cpu().numpy()
    contextImgNormK.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: cross_attn_norm_k_img_weight)))
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
  return (reader, Model([x, context, contextImg, rot] + c, [out]))
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

func Wan(
  channels: Int, layers: Int, intermediateSize: Int, time: Int, height: Int, width: Int,
  textLength: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let imgIn = Dense(count: channels, name: "x_embedder")
  let clip = Input()
  var out = imgIn(x)
  let txt = Input()
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: channels, name: "c")
  let context = contextEmbedder(txt)
  let t = Input()
  let rot = Input()
  let (timeInMlp0, timeInMlp2, timeIn) = TimeEmbedder(channels: channels, name: "t")
  let (clipLn1, clipLn2, clipMlp0, clipMlp2, clipIn) = MLPProj(
    inChannels: 1280, outChannels: channels, name: "clip")
  let clipContext = clipIn(clip)
  let vector = timeIn(t).reshaped([1, 1, channels])
  let vectorIn = vector.swish()
  let timeProjections = (0..<6).map { Dense(count: channels, name: "ada_ln_\($0)") }
  let tOut = timeProjections.map { $0(vectorIn) }
  let h = height / 2
  let w = width / 2
  var readers = [(PythonObject) -> Void]()
  out = out.to(.Float32)
  for i in 0..<layers {
    let (reader, block) = WanAttentionBlock(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: 1, t: textLength, hw: time * h * w,
      intermediateSize: intermediateSize)
    out = block([out, context, clipContext, rot] + tOut)
    readers.append(reader)
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
    let img_emb_proj_0_weight = state_dict["img_emb.proj.0.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_emb_proj_0_bias = state_dict["img_emb.proj.0.bias"].to(torch.float)
      .cpu().numpy()
    clipLn1.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_emb_proj_0_weight)))
    clipLn1.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_emb_proj_0_bias)))
    let img_emb_proj_1_weight = state_dict["img_emb.proj.1.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_emb_proj_1_bias = state_dict["img_emb.proj.1.bias"].to(torch.float)
      .cpu().numpy()
    clipMlp0.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_emb_proj_1_weight)))
    clipMlp0.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_emb_proj_1_bias)))
    let img_emb_proj_3_weight = state_dict["img_emb.proj.3.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_emb_proj_3_bias = state_dict["img_emb.proj.3.bias"].to(torch.float)
      .cpu().numpy()
    clipMlp2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_emb_proj_3_weight)))
    clipMlp2.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_emb_proj_3_bias)))
    let img_emb_proj_4_weight = state_dict["img_emb.proj.4.weight"].to(
      torch.float
    ).cpu().numpy()
    let img_emb_proj_4_bias = state_dict["img_emb.proj.4.bias"].to(torch.float)
      .cpu().numpy()
    clipLn2.weight.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_emb_proj_4_weight)))
    clipLn2.bias.copy(
      from: Tensor<Float16>(from: try! Tensor<Float>(numpy: img_emb_proj_4_bias)))
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
  return (Model([x, txt, clip, t, rot], [out]), reader)
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
  let (wan, reader) = Wan(
    // channels: 1536, layers: 30, intermediateSize: 8960, time: 21, height: 60, width: 104,
    channels: 5120, layers: 40, intermediateSize: 13824, time: 21, height: 60, width: 104,
    textLength: 512)
  let xTensor = graph.variable(
    Tensor<Float16>(
      from: try! Tensor<Float>(numpy: torch.concat([x, y], dim: 0).to(torch.float).cpu().numpy())
    ).toGPU(2)
  ).reshaped(format: .NHWC, shape: [1, 36, 21 * 30, 2, 52, 2]).permuted(0, 2, 4, 1, 3, 5).copied()
    .reshaped(format: .NHWC, shape: [1, 21 * 30 * 52, 36 * 2 * 2])
  let txt = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: context.to(torch.float).cpu().numpy()))
      .reshaped(.WC(28, 4096)).toGPU(2))
  var txtIn = graph.variable(.GPU(2), .WC(512, 4096), of: Float16.self)
  txtIn.full(0)
  txtIn[0..<28, 0..<4096] = txt
  txtIn = txtIn.reshaped(.HWC(1, 512, 4096))
  let clip = graph.variable(
    Tensor<Float16>(from: try! Tensor<Float>(numpy: clip_fea.to(torch.float).cpu().numpy()))
      .reshaped(.HWC(1, 257, 1280)).toGPU(2))
  let timestep = timeEmbedding(timesteps: 900, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
  let tGPU = graph.variable(Tensor<Float>(from: timestep)).toGPU(2)
  let rotNdTensorGPU = DynamicGraph.Tensor<Float16>(from: rotNdTensor).toGPU(2)
  wan.compile(inputs: xTensor, txtIn, clip, tGPU, rotNdTensorGPU)
  graph.openStore("/slow/Data/wan_v2.1_14b_i2v_720p_f16.ckpt") {
    $0.read("dit", model: wan)
  }
  // reader(wan_state_dict)
  debugPrint(wan(inputs: xTensor, txtIn, clip, tGPU, rotNdTensorGPU))
  /*
  graph.openStore("/slow/Data/wan_v2.1_14b_i2v_720p_f16.ckpt") {
    $0.write("dit", model: wan)
  }
  */
}

exit(0)

let z = torch.randn([16, 3, 60, 104]).to(torch.float).cuda()
let zOut = wan_i2v.vae.decode([z])[0]
wan_i2v.vae.encode([zOut])
print(wan_i2v.vae.model.encoder)
let vae_state_dict = wan_i2v.vae.model.state_dict()

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
    let norm1_weight = state_dict["\(prefix).residual.0.gamma"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    let conv1_weight = state_dict["\(prefix).residual.2.weight"].to(torch.float).cpu().numpy()
    let conv1_bias = state_dict["\(prefix).residual.2.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).residual.3.gamma"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    let conv2_weight = state_dict["\(prefix).residual.6.weight"].to(torch.float).cpu().numpy()
    let conv2_bias = state_dict["\(prefix).residual.6.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).shortcut.weight"].to(torch.float)
        .cpu()
        .numpy()
      let nin_shortcut_bias = state_dict["\(prefix).shortcut.bias"].to(torch.float).cpu()
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
    prefix: "decoder.middle.0", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlockCausal3D(
    prefix: "decoder.middle.1", inChannels: previousChannel, depth: startDepth,
    height: startHeight, width: startWidth)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlockCausal3D(
    prefix: "decoder.middle.2", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock2(out)
  var width = startWidth
  var height = startHeight
  var depth = startDepth
  var readers = [(PythonObject) -> Void]()
  var k = 0
  for (i, channel) in channels.enumerated().reversed() {
    for _ in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlockCausal3D(
        prefix: "decoder.upsamples.\(k)",
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
        let upLayer = k
        let reader: (PythonObject) -> Void = { state_dict in
          let time_conv_weight = state_dict["decoder.upsamples.\(upLayer).time_conv.weight"]
            .to(torch.float)
            .cpu().numpy()
          let time_conv_bias = state_dict["decoder.upsamples.\(upLayer).time_conv.bias"].to(
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
      let upLayer = k
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict["decoder.upsamples.\(upLayer).resample.1.weight"]
          .to(torch.float)
          .cpu().numpy()
        let conv_bias = state_dict["decoder.upsamples.\(upLayer).resample.1.bias"].to(
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
    let post_quant_conv_weight = state_dict["conv2.weight"].to(torch.float).cpu().numpy()
    let post_quant_conv_bias = state_dict["conv2.bias"].to(torch.float).cpu().numpy()
    postQuantConv.weight.copy(from: try! Tensor<Float>(numpy: post_quant_conv_weight))
    postQuantConv.bias.copy(from: try! Tensor<Float>(numpy: post_quant_conv_bias))
    let conv_in_weight = state_dict["decoder.conv1.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["decoder.conv1.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["decoder.head.0.gamma"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    let conv_out_weight = state_dict["decoder.head.2.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["decoder.head.2.bias"].to(torch.float).cpu().numpy()
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
        prefix: "encoder.downsamples.\(k)", inChannels: previousChannel,
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
          "encoder.downsamples.\(downLayer).resample.1.weight"
        ].to(
          torch.float
        ).cpu()
          .numpy()
        let conv_bias = state_dict["encoder.downsamples.\(downLayer).resample.1.bias"]
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
          let time_conv_weight = state_dict["encoder.downsamples.\(upLayer).time_conv.weight"]
            .to(torch.float)
            .cpu().numpy()
          let time_conv_bias = state_dict["encoder.downsamples.\(upLayer).time_conv.bias"].to(
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
    prefix: "encoder.middle.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlockCausal3D(
    prefix: "encoder.middle.1", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlockCausal3D(
    prefix: "encoder.middle.2", inChannels: previousChannel,
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
    let conv_in_weight = state_dict["encoder.conv1.weight"].to(torch.float).cpu().numpy()
    let conv_in_bias = state_dict["encoder.conv1.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    let norm_out_weight = state_dict["encoder.head.0.gamma"].to(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    let conv_out_weight = state_dict["encoder.head.2.weight"].to(torch.float).cpu().numpy()
    let conv_out_bias = state_dict["encoder.head.2.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: conv_out_bias))
    let quant_conv_weight = state_dict["conv1.weight"].to(torch.float).cpu().numpy()
    let quant_conv_bias = state_dict["conv1.bias"].to(torch.float).cpu().numpy()
    quantConv.weight.copy(from: try! Tensor<Float>(numpy: quant_conv_weight))
    quantConv.bias.copy(from: try! Tensor<Float>(numpy: quant_conv_bias))
  }
  return (reader, Model([x], [out]))
}

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
  var zTensor = graph.variable(try! Tensor<Float>(numpy: z.to(torch.float).cpu().numpy())).reshaped(
    format: .NCHW, shape: [1, 16, 3, 60, 104]
  ).toGPU(1)
  // zTensor = zTensor / std + mean
  let (decoderReader, decoder) = DecoderCausal3D(
    channels: [96, 192, 384, 384], numRepeat: 2, startWidth: 104, startHeight: 60, startDepth: 3)
  decoder.compile(inputs: zTensor)
  decoderReader(vae_state_dict)
  let image = decoder(inputs: zTensor)[0].as(of: Float.self)
  let (encoderReader, encoder) = EncoderCausal3D(
    channels: [96, 192, 384, 384], numRepeat: 2, startWidth: 104, startHeight: 60, startDepth: 3)
  encoder.compile(inputs: image)
  encoderReader(vae_state_dict)
  debugPrint(encoder(inputs: image)[0].as(of: Float.self))
  graph.openStore("/home/liu/workspace/swift-diffusion/wan_v2.1_video_vae_f32.ckpt") {
    $0.write("decoder", model: decoder)
    $0.write("encoder", model: encoder)
  }
  */
}

/*
let video = wan_t2v.generate(
    prompt,
    size: PythonObject(tupleOf: 832, 480),
    frame_num: 81,
    shift: 5,
    sample_solver: "dpm++",
    sampling_steps: 40,
    guide_scale: 6,
    seed: 42,
    offload_model: true)

wan_utils_utils.cache_video(
    tensor: video[Python.None],
    save_file: "/home/liu/workspace/swift-diffusion/wan_1.3b.mp4",
    fps: cfg.sample_fps,
    nrow: 1,
    normalize: true,
    value_range: PythonObject(tupleOf: -1, 1))
*/
