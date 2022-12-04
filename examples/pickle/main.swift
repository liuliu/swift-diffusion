import Collections
import Fickling
import Foundation
import NNC
import ZIPFoundation

public typealias UseFloatingPoint = Float16

let filename = "/fast/Data/SD/PaperCut_v1.ckpt"

let archive = Archive(url: URL(fileURLWithPath: filename), accessMode: .read)!

let entry = archive["archive/data.pkl"]!

var data = Data()
let _ = try archive.extract(entry) { data.append($0) }
let interpreter = Interpreter.from(data: data)

extension Model.Parameters {
  func copy<T: TensorNumeric>(
    from tensorDescriptor: TensorDescriptor, zip: Archive, of type: T.Type
  ) throws {
    var v = 1
    for i in stride(from: tensorDescriptor.shape.count - 1, through: 0, by: -1) {
      precondition(tensorDescriptor.strides[i] == v)
      v *= tensorDescriptor.shape[i]
    }
    let entry = archive["archive/data/\(tensorDescriptor.storage.name)"]!
    var data = Data()
    let _ = try archive.extract(entry) { data.append($0) }
    data.withUnsafeMutableBytes {
      guard let address = $0.baseAddress else { return }
      let tensor: AnyTensor
      if tensorDescriptor.storage.dataType == .Float16 {
        tensor = Tensor<Float16>(
          .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
          unsafeMutablePointer: address.assumingMemoryBound(to: Float16.self), bindLifetimeOf: entry
        )
      } else {
        tensor = Tensor<Float>(
          .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
          unsafeMutablePointer: address.assumingMemoryBound(to: Float.self), bindLifetimeOf: entry)
      }
      copy(from: Tensor<T>(from: tensor))
    }
  }
}

struct Storage {
  var name: String
  var size: Int
  var dataType: DataType
}

struct TensorDescriptor {
  var storage: Storage
  var storageOffset: Int
  var shape: [Int]
  var strides: [Int]
}

interpreter.intercept(module: "UNPICKLER", function: "persistent_load") { module, function, args in
  guard args.count >= 5, let global = args[1] as? Interpreter.GlobalObject,
    let name = args[2] as? String, let size = args[4] as? Int
  else { return [nil] }
  guard global.function == "HalfStorage" || global.function == "FloatStorage" else { return [nil] }
  let storage = Storage(
    name: name, size: size, dataType: global.function == "HalfStorage" ? .Float16 : .Float32)
  return [storage]
}
interpreter.intercept(module: "torch._utils", function: "_rebuild_tensor_v2") {
  module, function, args in
  guard args.count >= 5, let storage = args[0] as? Storage, let storageOffset = args[1] as? Int,
    let shape = args[2] as? [Int],
    let strides = args[3] as? [Int]
  else { return [nil] }
  let tensorDescriptor = TensorDescriptor(
    storage: storage, storageOffset: storageOffset, shape: shape, strides: strides)
  return [tensorDescriptor]
}
interpreter.intercept(module: nil, function: nil) { module, function, args in
  return [nil]
}
while try interpreter.step() {}
let model =
  (interpreter.rootObject as? OrderedDictionary<String, Any>)
  ?? OrderedDictionary<String, Any>(
    uniqueKeysWithValues: (interpreter.rootObject as? [String: Any])!)
let state_dict =
  (model["state_dict"] as? OrderedDictionary<String, Any>)
  ?? OrderedDictionary<String, Any>(uniqueKeysWithValues: (model["state_dict"] as? [String: Any])!)

/// CLIP Text Model

func CLIPTextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (
  Model, Model, Model
) {
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(T.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + positionEmbed(positions)
  return (tokenEmbed, positionEmbed, Model([tokens, positions], [embedding], name: "embeddings"))
}

func CLIPAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let casualAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + casualAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, casualAttentionMask], [out]))
}

func QuickGELU() -> Model {
  let x = Input()
  let y = x .* Sigmoid()(1.702 * x)
  return Model([x], [y])
}

func CLIPMLP(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize)
  var out = fc1(x)
  out = QuickGELU()(out)
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

func CLIPEncoderLayer(k: Int, h: Int, b: Int, t: Int, intermediateSize: Int) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model
) {
  let x = Input()
  let casualAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  var out = layerNorm1(x)
  let (tokeys, toqueries, tovalues, unifyheads, attention) = CLIPAttention(k: k, h: h, b: b, t: t)
  out = attention(out, casualAttentionMask) + x
  let residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm2(out)
  let (fc1, fc2, mlp) = CLIPMLP(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = mlp(out) + residual
  return (
    layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2,
    Model([x, casualAttentionMask], [out])
  )
}

func CLIPTextModel<T: TensorNumeric>(
  _ dataType: T.Type,
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> (
  Model, Model, [Model], [Model], [Model], [Model], [Model], [Model], [Model], [Model], Model, Model
) {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let (tokenEmbed, positionEmbed, embedding) = CLIPTextEmbedding(
    T.self,
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  var layerNorm1s = [Model]()
  var tokeyss = [Model]()
  var toqueriess = [Model]()
  var tovaluess = [Model]()
  var unifyheadss = [Model]()
  var layerNorm2s = [Model]()
  var fc1s = [Model]()
  var fc2s = [Model]()
  let k = embeddingSize / numHeads
  for _ in 0..<numLayers {
    let (layerNorm1, tokeys, toqueries, tovalues, unifyheads, layerNorm2, fc1, fc2, encoderLayer) =
      CLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    layerNorm1s.append(layerNorm1)
    tokeyss.append(tokeys)
    toqueriess.append(toqueries)
    tovaluess.append(tovalues)
    unifyheadss.append(unifyheads)
    layerNorm2s.append(layerNorm2)
    fc1s.append(fc1)
    fc2s.append(fc2)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return (
    tokenEmbed, positionEmbed, layerNorm1s, tokeyss, toqueriess, tovaluess, unifyheadss,
    layerNorm2s, fc1s, fc2s, finalLayerNorm, Model([tokens, positions, casualAttentionMask], [out])
  )
}

let (
  tokenEmbed, positionEmbed, layerNorm1s, tokeys, toqueries, tovalues, unifyheads, layerNorm2s,
  fc1s, fc2s, finalLayerNorm, textModel
) = CLIPTextModel(
  UseFloatingPoint.self,
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
  batchSize: 1, intermediateSize: 3072)

let graph = DynamicGraph()

let tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
for i in 0..<77 {
  tokensTensor[i] = 0
  tokensTensor[i + 77] = 0
  positionTensor[i] = Int32(i)
  positionTensor[i + 77] = Int32(i)
}

let casualAttentionMask = graph.variable(Tensor<UseFloatingPoint>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -UseFloatingPoint.greatestFiniteMagnitude
  }
}

try graph.withNoGrad {
  textModel.compile(
    inputs: tokensTensor.toGPU(0), positionTensor.toGPU(0), casualAttentionMask.toGPU(0))
  let vocab =
    state_dict["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    as! TensorDescriptor
  let pos =
    state_dict["cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"]
    as! TensorDescriptor
  try tokenEmbed.parameters.copy(from: vocab, zip: archive, of: UseFloatingPoint.self)
  try positionEmbed.parameters.copy(from: pos, zip: archive, of: UseFloatingPoint.self)

  for i in 0..<12 {
    let layer_norm_1_weight =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm1.weight"
      ] as! TensorDescriptor
    let layer_norm_1_bias =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm1.bias"
      ] as! TensorDescriptor
    try layerNorm1s[i].parameters(for: .weight).copy(
      from: layer_norm_1_weight, zip: archive, of: UseFloatingPoint.self)
    try layerNorm1s[i].parameters(for: .bias).copy(
      from: layer_norm_1_bias, zip: archive, of: UseFloatingPoint.self)

    let k_proj_weight =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.k_proj.weight"
      ] as! TensorDescriptor
    let k_proj_bias =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.k_proj.bias"
      ] as! TensorDescriptor
    try tokeys[i].parameters(for: .weight).copy(
      from: k_proj_weight, zip: archive, of: UseFloatingPoint.self)
    try tokeys[i].parameters(for: .bias).copy(
      from: k_proj_bias, zip: archive, of: UseFloatingPoint.self)

    let v_proj_weight =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.v_proj.weight"
      ] as! TensorDescriptor
    let v_proj_bias =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.v_proj.bias"
      ] as! TensorDescriptor
    try tovalues[i].parameters(for: .weight).copy(
      from: v_proj_weight, zip: archive, of: UseFloatingPoint.self)
    try tovalues[i].parameters(for: .bias).copy(
      from: v_proj_bias, zip: archive, of: UseFloatingPoint.self)

    let q_proj_weight =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.q_proj.weight"
      ] as! TensorDescriptor
    let q_proj_bias =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.q_proj.bias"
      ] as! TensorDescriptor
    try toqueries[i].parameters(for: .weight).copy(
      from: q_proj_weight, zip: archive, of: UseFloatingPoint.self)
    try toqueries[i].parameters(for: .bias).copy(
      from: q_proj_bias, zip: archive, of: UseFloatingPoint.self)

    let out_proj_weight =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.out_proj.weight"
      ]
      as! TensorDescriptor
    let out_proj_bias =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).self_attn.out_proj.bias"
      ] as! TensorDescriptor
    try unifyheads[i].parameters(for: .weight).copy(
      from: out_proj_weight, zip: archive, of: UseFloatingPoint.self)
    try unifyheads[i].parameters(for: .bias).copy(
      from: out_proj_bias, zip: archive, of: UseFloatingPoint.self)

    let layer_norm_2_weight =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm2.weight"
      ] as! TensorDescriptor
    let layer_norm_2_bias =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).layer_norm2.bias"
      ] as! TensorDescriptor
    try layerNorm2s[i].parameters(for: .weight).copy(
      from: layer_norm_2_weight, zip: archive, of: UseFloatingPoint.self)
    try layerNorm2s[i].parameters(for: .bias).copy(
      from: layer_norm_2_bias, zip: archive, of: UseFloatingPoint.self)

    let fc1_weight =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc1.weight"
      ]
      as! TensorDescriptor
    let fc1_bias =
      state_dict["cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc1.bias"]
      as! TensorDescriptor
    try fc1s[i].parameters(for: .weight).copy(
      from: fc1_weight, zip: archive, of: UseFloatingPoint.self)
    try fc1s[i].parameters(for: .bias).copy(from: fc1_bias, zip: archive, of: UseFloatingPoint.self)

    let fc2_weight =
      state_dict[
        "cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc2.weight"
      ]
      as! TensorDescriptor
    let fc2_bias =
      state_dict["cond_stage_model.transformer.text_model.encoder.layers.\(i).mlp.fc2.bias"]
      as! TensorDescriptor
    try fc2s[i].parameters(for: .weight).copy(
      from: fc2_weight, zip: archive, of: UseFloatingPoint.self)
    try fc2s[i].parameters(for: .bias).copy(from: fc2_bias, zip: archive, of: UseFloatingPoint.self)
  }

  let final_layer_norm_weight =
    state_dict[
      "cond_stage_model.transformer.text_model.final_layer_norm.weight"
    ]
    as! TensorDescriptor
  let final_layer_norm_bias =
    state_dict["cond_stage_model.transformer.text_model.final_layer_norm.bias"]
    as! TensorDescriptor
  try finalLayerNorm.parameters(for: .weight).copy(
    from: final_layer_norm_weight, zip: archive, of: UseFloatingPoint.self)
  try finalLayerNorm.parameters(for: .bias).copy(
    from: final_layer_norm_bias, zip: archive, of: UseFloatingPoint.self)

  graph.openStore("/home/liu/workspace/swift-diffusion/text_model.ckpt") {
    $0.write("text_model", model: textModel)
  }
}
