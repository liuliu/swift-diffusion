import Collections
import Fickling
import Foundation
import NNC
import ZIPFoundation

public typealias UseFloatingPoint = Float16
/*
let file1 = "/home/liu/workspace/swift-diffusion/clip_vit_l14_f16.ckpt"
let file2 = "/home/liu/workspace/swift-diffusion/text_model.ckpt"

let graph = DynamicGraph()

graph.openStore(file1) { store1 in
  let keys1 = store1.keys
  graph.openStore(file2) { store2 in
    let keys2 = store2.keys
    precondition(keys1 == keys2)
    for key in keys2 {
      let tensor1 = store1.read(key)!
      let tensor2 = store2.read(key)!
      let tensor1Size = tensor1.shape.reduce(1, *)
      let tensor2Size = tensor2.shape.reduce(1, *)
      precondition(tensor1Size == tensor2Size)
      let tensor1r = Tensor<UseFloatingPoint>(tensor1).reshaped(.C(tensor1Size)).toCPU()
      let tensor2r = Tensor<UseFloatingPoint>(tensor2).reshaped(.C(tensor2Size)).toCPU()
      for i in 0..<tensor1Size {
        if tensor1r[i] != tensor2r[i] {
          print(
            "\(key) loc \(i), v1 \(tensor1.shape) \(tensor1r[i]), v2 \(tensor2.shape) \(tensor2r[i])"
          )
          // break
        }
      }
    }
  }
}
*/
let filename = "/home/liu/workspace/swift-diffusion/EmWat69.pt"

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
      if !(tensorDescriptor.strides[i] == v) {
        print(tensorDescriptor.shape)
        print(tensorDescriptor.strides)
        break
      }
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

extension TensorDescriptor {
  func inflate<T: TensorNumeric>(from zip: Archive, of type: T.Type) throws -> Tensor<T> {
    var v = 1
    for i in stride(from: shape.count - 1, through: 0, by: -1) {
      precondition(strides[i] == v)
      v *= shape[i]
    }
    let entry = archive["archive/data/\(storage.name)"]!
    var data = Data()
    let _ = try archive.extract(entry) { data.append($0) }
    return data.withUnsafeMutableBytes {
      guard let address = $0.baseAddress else { fatalError() }
      let tensor: AnyTensor
      if storage.dataType == .Float16 {
        tensor = Tensor<Float16>(
          .CPU, format: .NCHW, shape: TensorShape(shape),
          unsafeMutablePointer: address.assumingMemoryBound(to: Float16.self), bindLifetimeOf: entry
        )
      } else {
        tensor = Tensor<Float>(
          .CPU, format: .NCHW, shape: TensorShape(shape),
          unsafeMutablePointer: address.assumingMemoryBound(to: Float.self), bindLifetimeOf: entry
        )
      }
      return Tensor<T>(from: tensor).copied()
    }
  }
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
interpreter.intercept(module: "torch.nn.modules.container", function: "ParameterDict") { module, function, _ in
  return [Interpreter.Dictionary(.unordered)]
}
interpreter.intercept(module: "torch._utils", function: "_rebuild_tensor_v2") {
  module, function, args in
  guard args.count >= 5, let storage = args[0] as? Storage, let storageOffset = args[1] as? Int,
    let shape = args[2] as? [Int],
    let strides = args[3] as? [Int]
  else { return [nil] }
  precondition(storageOffset == 0)
  let tensorDescriptor = TensorDescriptor(
    storage: storage, storageOffset: storageOffset, shape: shape, strides: strides)
  return [tensorDescriptor]
}
interpreter.intercept(module: "torch._utils", function: "_rebuild_parameter") { _, _, args in
  guard let tensorDescriptor = args.first as? TensorDescriptor else { return [nil] }
  return [tensorDescriptor]
}
interpreter.intercept(module: nil, function: nil) { module, function, args in
  return [nil]
}
while try interpreter.step() {}
let graph = DynamicGraph()
let model = (interpreter.rootObject as? Interpreter.Dictionary)!

print((model["string_to_token"] as! Interpreter.Dictionary).dictionary)
let stringToToken = (model["string_to_param"] as? Interpreter.Dictionary)!
// let parameters = (stringToToken["_parameters"] as? Interpreter.Dictionary)!
// print(parameters.dictionary)
let token = stringToToken["*"] as! TensorDescriptor
// let token = model["<birb-style>"] as! TensorDescriptor
print(token)
let tensor = try token.inflate(from: archive, of: Float16.self)
debugPrint(tensor)

graph.openStore("/home/liu/workspace/swift-diffusion/textual_inversion.ckpt") {
  $0.write("string_to_param", tensor: tensor)
}
fatalError()


let state_dict = (model["state_dict"] as? Interpreter.Dictionary) ?? model

var renameVAE = [String: Any]()
state_dict.forEach { key, value in
  if key.hasPrefix("encoder.") || key.hasPrefix("decoder.") || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.") {
    renameVAE[key] = value
  }
}
for (key, value) in renameVAE {
  state_dict["first_stage_model.\(key)"] = value
}

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

/// UNet Model.

func timeEmbedding(timesteps: Int, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * Float(timesteps)
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

func TimeEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func ResBlock(b: Int, outChannels: Int, skipConnection: Bool) -> (
  Model, Model, Model, Model, Model, Model?, Model
) {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: outChannels)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([b, outChannels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var skipModel: Model? = nil
  if skipConnection {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]))
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = skip
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
  }
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel,
    Model([x, emb], [out])
  )
}

func SelfAttention(k: Int, h: Int, b: Int, hw: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, hw])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func CrossAttention(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model, Model, Model)
{
  let x = Input()
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(c).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, c], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let fc10 = Dense(count: intermediateSize)
  let fc11 = Dense(count: intermediateSize)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc10, fc11, fc2, Model([x], [out]))
}

func BasicTransformerBlock(k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model,
  Model
) {
  let x = Input()
  let c = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (tokeys2, toqueries2, tovalues2, unifyheads2, attn2) = CrossAttention(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, c) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  return (
    layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2, toqueries2,
    tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, Model([x, c], [out])
  )
}

func SpatialTransformer(
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, t: Int, intermediateSize: Int
) -> (
  Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model, Model,
  Model, Model, Model, Model
) {
  let x = Input()
  let c = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).permuted(0, 2, 1)
  let (
    layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2, toqueries2,
    tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, block
  ) = BasicTransformerBlock(k: k, h: h, b: b, hw: hw, t: t, intermediateSize: intermediateSize)
  out = block(out, c).reshaped([b, height, width, k * h]).permuted(0, 3, 1, 2)
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  return (
    norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
    toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, fc2, projOut, Model([x, c], [out])
  )
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Bool, channels: Int, numHeads: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((Interpreter.Dictionary) throws -> Void, Model) {
  let x = Input()
  let emb = Input()
  let c = Input()
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  var norm: Model? = nil
  var projIn: Model? = nil
  var layerNorm1: Model? = nil
  var tokeys1: Model? = nil
  var toqueries1: Model? = nil
  var tovalues1: Model? = nil
  var unifyheads1: Model? = nil
  var layerNorm2: Model? = nil
  var tokeys2: Model? = nil
  var toqueries2: Model? = nil
  var tovalues2: Model? = nil
  var unifyheads2: Model? = nil
  var layerNorm3: Model? = nil
  var fc10: Model? = nil
  var fc11: Model? = nil
  var tfc2: Model? = nil
  var projOut: Model? = nil
  if attentionBlock {
    let transformer: Model
    (
      norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
      toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, tfc2, projOut, transformer
    ) = SpatialTransformer(
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer(out, c)
  }
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    let in_layers_0_weight =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.in_layers.0.weight"
      ] as! TensorDescriptor
    let in_layers_0_bias =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.in_layers.0.bias"
      ] as! TensorDescriptor
    try inLayerNorm.parameters(for: .weight).copy(
      from: in_layers_0_weight, zip: archive, of: UseFloatingPoint.self)
    try inLayerNorm.parameters(for: .bias).copy(
      from: in_layers_0_bias, zip: archive, of: UseFloatingPoint.self)
    let in_layers_2_weight =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.in_layers.2.weight"
      ] as! TensorDescriptor
    let in_layers_2_bias =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.in_layers.2.bias"
      ] as! TensorDescriptor
    try inLayerConv2d.parameters(for: .weight).copy(
      from: in_layers_2_weight, zip: archive, of: UseFloatingPoint.self)
    try inLayerConv2d.parameters(for: .bias).copy(
      from: in_layers_2_bias, zip: archive, of: UseFloatingPoint.self)
    let emb_layers_1_weight =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.weight"
      ] as! TensorDescriptor
    let emb_layers_1_bias =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.bias"
      ] as! TensorDescriptor
    try embLayer.parameters(for: .weight).copy(
      from: emb_layers_1_weight, zip: archive, of: UseFloatingPoint.self)
    try embLayer.parameters(for: .bias).copy(
      from: emb_layers_1_bias, zip: archive, of: UseFloatingPoint.self)
    let out_layers_0_weight =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.out_layers.0.weight"
      ] as! TensorDescriptor
    let out_layers_0_bias =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.out_layers.0.bias"
      ] as! TensorDescriptor
    try outLayerNorm.parameters(for: .weight).copy(
      from: out_layers_0_weight, zip: archive, of: UseFloatingPoint.self)
    try outLayerNorm.parameters(for: .bias).copy(
      from: out_layers_0_bias, zip: archive, of: UseFloatingPoint.self)
    let out_layers_3_weight =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.out_layers.3.weight"
      ] as! TensorDescriptor
    let out_layers_3_bias =
      state_dict[
        "model.diffusion_model.\(prefix).\(layerStart).0.out_layers.3.bias"
      ] as! TensorDescriptor
    try outLayerConv2d.parameters(for: .weight).copy(
      from: out_layers_3_weight, zip: archive, of: UseFloatingPoint.self)
    try outLayerConv2d.parameters(for: .bias).copy(
      from: out_layers_3_bias, zip: archive, of: UseFloatingPoint.self)
    if let skipModel = skipModel {
      let skip_connection_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).0.skip_connection.weight"
        ] as! TensorDescriptor
      let skip_connection_bias =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).0.skip_connection.bias"
        ] as! TensorDescriptor
      try skipModel.parameters(for: .weight).copy(
        from: skip_connection_weight, zip: archive, of: UseFloatingPoint.self)
      try skipModel.parameters(for: .bias).copy(
        from: skip_connection_bias, zip: archive, of: UseFloatingPoint.self)
    }
    if let norm = norm, let projIn = projIn, let layerNorm1 = layerNorm1, let tokeys1 = tokeys1,
      let toqueries1 = toqueries1, let tovalues1 = tovalues1, let unifyheads1 = unifyheads1,
      let layerNorm2 = layerNorm2, let tokeys2 = tokeys2, let toqueries2 = toqueries2,
      let tovalues2 = tovalues2, let unifyheads2 = unifyheads2, let layerNorm3 = layerNorm3,
      let fc10 = fc10, let fc11 = fc11, let tfc2 = tfc2, let projOut = projOut
    {
      let norm_weight =
        state_dict["model.diffusion_model.\(prefix).\(layerStart).1.norm.weight"]
        as! TensorDescriptor
      let norm_bias =
        state_dict["model.diffusion_model.\(prefix).\(layerStart).1.norm.bias"]
        as! TensorDescriptor
      try norm.parameters(for: .weight).copy(
        from: norm_weight, zip: archive, of: UseFloatingPoint.self)
      try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: UseFloatingPoint.self)
      let proj_in_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.proj_in.weight"
        ]
        as! TensorDescriptor
      let proj_in_bias =
        state_dict["model.diffusion_model.\(prefix).\(layerStart).1.proj_in.bias"]
        as! TensorDescriptor
      try projIn.parameters(for: .weight).copy(
        from: proj_in_weight, zip: archive, of: UseFloatingPoint.self)
      try projIn.parameters(for: .bias).copy(
        from: proj_in_bias, zip: archive, of: UseFloatingPoint.self)
      let attn1_to_k_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_k.weight"
        ] as! TensorDescriptor
      try tokeys1.parameters(for: .weight).copy(
        from: attn1_to_k_weight, zip: archive, of: UseFloatingPoint.self)
      let attn1_to_q_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_q.weight"
        ] as! TensorDescriptor
      try toqueries1.parameters(for: .weight).copy(
        from: attn1_to_q_weight, zip: archive, of: UseFloatingPoint.self)
      let attn1_to_v_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_v.weight"
        ] as! TensorDescriptor
      try tovalues1.parameters(for: .weight).copy(
        from: attn1_to_v_weight, zip: archive, of: UseFloatingPoint.self)
      let attn1_to_out_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_out.0.weight"
        ] as! TensorDescriptor
      let attn1_to_out_bias =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn1.to_out.0.bias"
        ] as! TensorDescriptor
      try unifyheads1.parameters(for: .weight).copy(
        from: attn1_to_out_weight, zip: archive, of: UseFloatingPoint.self)
      try unifyheads1.parameters(for: .bias).copy(
        from: attn1_to_out_bias, zip: archive, of: UseFloatingPoint.self)
      let ff_net_0_proj_weight = try
        (state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.0.proj.weight"
        ] as! TensorDescriptor).inflate(from: archive, of: UseFloatingPoint.self)
      let ff_net_0_proj_bias = try
        (state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.0.proj.bias"
        ] as! TensorDescriptor).inflate(from: archive, of: UseFloatingPoint.self)
      fc10.parameters(for: .weight).copy(
        from: ff_net_0_proj_weight[0..<intermediateSize, 0..<ff_net_0_proj_weight.shape[1]])
      fc10.parameters(for: .bias).copy(
        from: ff_net_0_proj_bias[0..<intermediateSize])
      fc11.parameters(for: .weight).copy(
        from: ff_net_0_proj_weight[
          intermediateSize..<ff_net_0_proj_weight.shape[0], 0..<ff_net_0_proj_weight.shape[1]])
      fc11.parameters(for: .bias).copy(
        from: ff_net_0_proj_bias[intermediateSize..<ff_net_0_proj_bias.shape[0]])
      let ff_net_2_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.2.weight"
        ] as! TensorDescriptor
      let ff_net_2_bias =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.ff.net.2.bias"
        ] as! TensorDescriptor
      try tfc2.parameters(for: .weight).copy(
        from: ff_net_2_weight, zip: archive, of: UseFloatingPoint.self)
      try tfc2.parameters(for: .bias).copy(
        from: ff_net_2_bias, zip: archive, of: UseFloatingPoint.self)
      let attn2_to_k_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_k.weight"
        ] as! TensorDescriptor
      try tokeys2.parameters(for: .weight).copy(
        from: attn2_to_k_weight, zip: archive, of: UseFloatingPoint.self)
      let attn2_to_q_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_q.weight"
        ] as! TensorDescriptor
      try toqueries2.parameters(for: .weight).copy(
        from: attn2_to_q_weight, zip: archive, of: UseFloatingPoint.self)
      let attn2_to_v_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_v.weight"
        ] as! TensorDescriptor
      try tovalues2.parameters(for: .weight).copy(
        from: attn2_to_v_weight, zip: archive, of: UseFloatingPoint.self)
      let attn2_to_out_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_out.0.weight"
        ] as! TensorDescriptor
      let attn2_to_out_bias =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.attn2.to_out.0.bias"
        ] as! TensorDescriptor
      try unifyheads2.parameters(for: .weight).copy(
        from: attn2_to_out_weight, zip: archive, of: UseFloatingPoint.self)
      try unifyheads2.parameters(for: .bias).copy(
        from: attn2_to_out_bias, zip: archive, of: UseFloatingPoint.self)
      let norm1_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm1.weight"
        ]
        as! TensorDescriptor
      let norm1_bias =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm1.bias"
        ]
        as! TensorDescriptor
      try layerNorm1.parameters(for: .weight).copy(
        from: norm1_weight, zip: archive, of: UseFloatingPoint.self)
      try layerNorm1.parameters(for: .bias).copy(
        from: norm1_bias, zip: archive, of: UseFloatingPoint.self)
      let norm2_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm2.weight"
        ]
        as! TensorDescriptor
      let norm2_bias =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm2.bias"
        ]
        as! TensorDescriptor
      try layerNorm2.parameters(for: .weight).copy(
        from: norm2_weight, zip: archive, of: UseFloatingPoint.self)
      try layerNorm2.parameters(for: .bias).copy(
        from: norm2_bias, zip: archive, of: UseFloatingPoint.self)
      let norm3_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm3.weight"
        ]
        as! TensorDescriptor
      let norm3_bias =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.transformer_blocks.0.norm3.bias"
        ]
        as! TensorDescriptor
      try layerNorm3.parameters(for: .weight).copy(
        from: norm3_weight, zip: archive, of: UseFloatingPoint.self)
      try layerNorm3.parameters(for: .bias).copy(
        from: norm3_bias, zip: archive, of: UseFloatingPoint.self)
      let proj_out_weight =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.proj_out.weight"
        ] as! TensorDescriptor
      let proj_out_bias =
        state_dict[
          "model.diffusion_model.\(prefix).\(layerStart).1.proj_out.bias"
        ]
        as! TensorDescriptor
      try projOut.parameters(for: .weight).copy(
        from: proj_out_weight, zip: archive, of: UseFloatingPoint.self)
      try projOut.parameters(for: .bias).copy(
        from: proj_out_bias, zip: archive, of: UseFloatingPoint.self)
    }
  }
  if attentionBlock {
    return (reader, Model([x, emb, c], [out]))
  } else {
    return (reader, Model([x, emb], [out]))
  }
}

func MiddleBlock(
  channels: Int, numHeads: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  x: Model.IO, emb: Model.IO, c: Model.IO
) -> ((Interpreter.Dictionary) throws -> Void, Model.IO) {
  precondition(channels % numHeads == 0)
  let k = channels / numHeads
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let (
    norm, projIn, layerNorm1, tokeys1, toqueries1, tovalues1, unifyheads1, layerNorm2, tokeys2,
    toqueries2, tovalues2, unifyheads2, layerNorm3, fc10, fc11, tfc2, projOut, transformer
  ) = SpatialTransformer(
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width, t: embeddingSize,
    intermediateSize: channels * 4)
  out = transformer(out, c)
  let (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    let intermediateSize = channels * 4
    let in_layers_0_0_weight =
      state_dict["model.diffusion_model.middle_block.0.in_layers.0.weight"]
      as! TensorDescriptor
    let in_layers_0_0_bias =
      state_dict["model.diffusion_model.middle_block.0.in_layers.0.bias"]
      as! TensorDescriptor
    try inLayerNorm1.parameters(for: .weight).copy(
      from: in_layers_0_0_weight, zip: archive, of: UseFloatingPoint.self)
    try inLayerNorm1.parameters(for: .bias).copy(
      from: in_layers_0_0_bias, zip: archive, of: UseFloatingPoint.self)
    let in_layers_0_2_weight =
      state_dict["model.diffusion_model.middle_block.0.in_layers.2.weight"]
      as! TensorDescriptor
    let in_layers_0_2_bias =
      state_dict["model.diffusion_model.middle_block.0.in_layers.2.bias"]
      as! TensorDescriptor
    try inLayerConv2d1.parameters(for: .weight).copy(
      from: in_layers_0_2_weight, zip: archive, of: UseFloatingPoint.self)
    try inLayerConv2d1.parameters(for: .bias).copy(
      from: in_layers_0_2_bias, zip: archive, of: UseFloatingPoint.self)
    let emb_layers_0_1_weight =
      state_dict[
        "model.diffusion_model.middle_block.0.emb_layers.1.weight"
      ]
      as! TensorDescriptor
    let emb_layers_0_1_bias =
      state_dict["model.diffusion_model.middle_block.0.emb_layers.1.bias"]
      as! TensorDescriptor
    try embLayer1.parameters(for: .weight).copy(
      from: emb_layers_0_1_weight, zip: archive, of: UseFloatingPoint.self)
    try embLayer1.parameters(for: .bias).copy(
      from: emb_layers_0_1_bias, zip: archive, of: UseFloatingPoint.self)
    let out_layers_0_0_weight =
      state_dict[
        "model.diffusion_model.middle_block.0.out_layers.0.weight"
      ]
      as! TensorDescriptor
    let out_layers_0_0_bias =
      state_dict[
        "model.diffusion_model.middle_block.0.out_layers.0.bias"
      ] as! TensorDescriptor
    try outLayerNorm1.parameters(for: .weight).copy(
      from: out_layers_0_0_weight, zip: archive, of: UseFloatingPoint.self)
    try outLayerNorm1.parameters(for: .bias).copy(
      from: out_layers_0_0_bias, zip: archive, of: UseFloatingPoint.self)
    let out_layers_0_3_weight =
      state_dict[
        "model.diffusion_model.middle_block.0.out_layers.3.weight"
      ]
      as! TensorDescriptor
    let out_layers_0_3_bias =
      state_dict["model.diffusion_model.middle_block.0.out_layers.3.bias"]
      as! TensorDescriptor
    try outLayerConv2d1.parameters(for: .weight).copy(
      from: out_layers_0_3_weight, zip: archive, of: UseFloatingPoint.self)
    try outLayerConv2d1.parameters(for: .bias).copy(
      from: out_layers_0_3_bias, zip: archive, of: UseFloatingPoint.self)
    let norm_weight =
      state_dict["model.diffusion_model.middle_block.1.norm.weight"]
      as! TensorDescriptor
    let norm_bias =
      state_dict["model.diffusion_model.middle_block.1.norm.bias"] as! TensorDescriptor
    try norm.parameters(for: .weight).copy(
      from: norm_weight, zip: archive, of: UseFloatingPoint.self)
    try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: UseFloatingPoint.self)
    let proj_in_weight =
      state_dict["model.diffusion_model.middle_block.1.proj_in.weight"]
      as! TensorDescriptor
    let proj_in_bias =
      state_dict["model.diffusion_model.middle_block.1.proj_in.bias"]
      as! TensorDescriptor
    try projIn.parameters(for: .weight).copy(
      from: proj_in_weight, zip: archive, of: UseFloatingPoint.self)
    try projIn.parameters(for: .bias).copy(
      from: proj_in_bias, zip: archive, of: UseFloatingPoint.self)
    let attn1_to_k_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight"
      ] as! TensorDescriptor
    try tokeys1.parameters(for: .weight).copy(
      from: attn1_to_k_weight, zip: archive, of: UseFloatingPoint.self)
    let attn1_to_q_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight"
      ] as! TensorDescriptor
    try toqueries1.parameters(for: .weight).copy(
      from: attn1_to_q_weight, zip: archive, of: UseFloatingPoint.self)
    let attn1_to_v_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight"
      ] as! TensorDescriptor
    try tovalues1.parameters(for: .weight).copy(
      from: attn1_to_v_weight, zip: archive, of: UseFloatingPoint.self)
    let attn1_to_out_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight"
      ] as! TensorDescriptor
    let attn1_to_out_bias =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias"
      ] as! TensorDescriptor
    try unifyheads1.parameters(for: .weight).copy(
      from: attn1_to_out_weight, zip: archive, of: UseFloatingPoint.self)
    try unifyheads1.parameters(for: .bias).copy(
      from: attn1_to_out_bias, zip: archive, of: UseFloatingPoint.self)
    let ff_net_0_proj_weight = try
      (state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight"
      ] as! TensorDescriptor).inflate(from: archive, of: UseFloatingPoint.self)
    let ff_net_0_proj_bias = try
      (state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias"
      ] as! TensorDescriptor).inflate(from: archive, of: UseFloatingPoint.self)
    fc10.parameters(for: .weight).copy(
      from: ff_net_0_proj_weight[0..<intermediateSize, 0..<ff_net_0_proj_weight.shape[1]])
    fc10.parameters(for: .bias).copy(
      from: ff_net_0_proj_bias[0..<intermediateSize])
    fc11.parameters(for: .weight).copy(
      from: ff_net_0_proj_weight[
        intermediateSize..<ff_net_0_proj_weight.shape[0], 0..<ff_net_0_proj_weight.shape[1]])
    fc11.parameters(for: .bias).copy(
      from: ff_net_0_proj_bias[intermediateSize..<ff_net_0_proj_bias.shape[0]])
    let ff_net_2_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight"
      ] as! TensorDescriptor
    let ff_net_2_bias =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias"
      ] as! TensorDescriptor
    try tfc2.parameters(for: .weight).copy(
      from: ff_net_2_weight, zip: archive, of: UseFloatingPoint.self)
    try tfc2.parameters(for: .bias).copy(
      from: ff_net_2_bias, zip: archive, of: UseFloatingPoint.self)
    let attn2_to_k_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight"
      ] as! TensorDescriptor
    try tokeys2.parameters(for: .weight).copy(
      from: attn2_to_k_weight, zip: archive, of: UseFloatingPoint.self)
    let attn2_to_q_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight"
      ] as! TensorDescriptor
    try toqueries2.parameters(for: .weight).copy(
      from: attn2_to_q_weight, zip: archive, of: UseFloatingPoint.self)
    let attn2_to_v_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight"
      ] as! TensorDescriptor
    try tovalues2.parameters(for: .weight).copy(
      from: attn2_to_v_weight, zip: archive, of: UseFloatingPoint.self)
    let attn2_to_out_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight"
      ] as! TensorDescriptor
    let attn2_to_out_bias =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias"
      ] as! TensorDescriptor
    try unifyheads2.parameters(for: .weight).copy(
      from: attn2_to_out_weight, zip: archive, of: UseFloatingPoint.self)
    try unifyheads2.parameters(for: .bias).copy(
      from: attn2_to_out_bias, zip: archive, of: UseFloatingPoint.self)
    let norm1_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight"
      ]
      as! TensorDescriptor
    let norm1_bias =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias"
      ]
      as! TensorDescriptor
    try layerNorm1.parameters(for: .weight).copy(
      from: norm1_weight, zip: archive, of: UseFloatingPoint.self)
    try layerNorm1.parameters(for: .bias).copy(
      from: norm1_bias, zip: archive, of: UseFloatingPoint.self)
    let norm2_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight"
      ]
      as! TensorDescriptor
    let norm2_bias =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias"
      ]
      as! TensorDescriptor
    try layerNorm2.parameters(for: .weight).copy(
      from: norm2_weight, zip: archive, of: UseFloatingPoint.self)
    try layerNorm2.parameters(for: .bias).copy(
      from: norm2_bias, zip: archive, of: UseFloatingPoint.self)
    let norm3_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight"
      ]
      as! TensorDescriptor
    let norm3_bias =
      state_dict[
        "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias"
      ]
      as! TensorDescriptor
    try layerNorm3.parameters(for: .weight).copy(
      from: norm3_weight, zip: archive, of: UseFloatingPoint.self)
    try layerNorm3.parameters(for: .bias).copy(
      from: norm3_bias, zip: archive, of: UseFloatingPoint.self)
    let proj_out_weight =
      state_dict[
        "model.diffusion_model.middle_block.1.proj_out.weight"
      ] as! TensorDescriptor
    let proj_out_bias =
      state_dict["model.diffusion_model.middle_block.1.proj_out.bias"]
      as! TensorDescriptor
    try projOut.parameters(for: .weight).copy(
      from: proj_out_weight, zip: archive, of: UseFloatingPoint.self)
    try projOut.parameters(for: .bias).copy(
      from: proj_out_bias, zip: archive, of: UseFloatingPoint.self)
    let in_layers_2_0_weight =
      state_dict["model.diffusion_model.middle_block.2.in_layers.0.weight"]
      as! TensorDescriptor
    let in_layers_2_0_bias =
      state_dict["model.diffusion_model.middle_block.2.in_layers.0.bias"]
      as! TensorDescriptor
    try inLayerNorm2.parameters(for: .weight).copy(
      from: in_layers_2_0_weight, zip: archive, of: UseFloatingPoint.self)
    try inLayerNorm2.parameters(for: .bias).copy(
      from: in_layers_2_0_bias, zip: archive, of: UseFloatingPoint.self)
    let in_layers_2_2_weight =
      state_dict["model.diffusion_model.middle_block.2.in_layers.2.weight"]
      as! TensorDescriptor
    let in_layers_2_2_bias =
      state_dict["model.diffusion_model.middle_block.2.in_layers.2.bias"]
      as! TensorDescriptor
    try inLayerConv2d2.parameters(for: .weight).copy(
      from: in_layers_2_2_weight, zip: archive, of: UseFloatingPoint.self)
    try inLayerConv2d2.parameters(for: .bias).copy(
      from: in_layers_2_2_bias, zip: archive, of: UseFloatingPoint.self)
    let emb_layers_2_1_weight =
      state_dict[
        "model.diffusion_model.middle_block.2.emb_layers.1.weight"
      ]
      as! TensorDescriptor
    let emb_layers_2_1_bias =
      state_dict["model.diffusion_model.middle_block.2.emb_layers.1.bias"]
      as! TensorDescriptor
    try embLayer2.parameters(for: .weight).copy(
      from: emb_layers_2_1_weight, zip: archive, of: UseFloatingPoint.self)
    try embLayer2.parameters(for: .bias).copy(
      from: emb_layers_2_1_bias, zip: archive, of: UseFloatingPoint.self)
    let out_layers_2_0_weight =
      state_dict[
        "model.diffusion_model.middle_block.2.out_layers.0.weight"
      ]
      as! TensorDescriptor
    let out_layers_2_0_bias =
      state_dict[
        "model.diffusion_model.middle_block.2.out_layers.0.bias"
      ] as! TensorDescriptor
    try outLayerNorm2.parameters(for: .weight).copy(
      from: out_layers_2_0_weight, zip: archive, of: UseFloatingPoint.self)
    try outLayerNorm2.parameters(for: .bias).copy(
      from: out_layers_2_0_bias, zip: archive, of: UseFloatingPoint.self)
    let out_layers_2_3_weight =
      state_dict[
        "model.diffusion_model.middle_block.2.out_layers.3.weight"
      ]
      as! TensorDescriptor
    let out_layers_2_3_bias =
      state_dict["model.diffusion_model.middle_block.2.out_layers.3.bias"]
      as! TensorDescriptor
    try outLayerConv2d2.parameters(for: .weight).copy(
      from: out_layers_2_3_weight, zip: archive, of: UseFloatingPoint.self)
    try outLayerConv2d2.parameters(for: .bias).copy(
      from: out_layers_2_3_bias, zip: archive, of: UseFloatingPoint.self)
  }
  return (reader, out)
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO
) -> ((Interpreter.Dictionary) throws -> Void, [Model.IO], Model.IO) {
  let conv2d = Convolution(
    groups: 1, filters: 320, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(Interpreter.Dictionary) throws -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for _ in 0..<numRepeat {
      let (reader, inputLayer) = BlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      if attentionBlock {
        out = inputLayer(out, emb, c)
      } else {
        out = inputLayer(out, emb)
      }
      passLayers.append(out)
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
        let op_weight =
          state_dict["model.diffusion_model.input_blocks.\(downLayer).0.op.weight"]
          as! TensorDescriptor
        let op_bias =
          state_dict["model.diffusion_model.input_blocks.\(downLayer).0.op.bias"]
          as! TensorDescriptor
        try downsample.parameters(for: .weight).copy(
          from: op_weight, zip: archive, of: UseFloatingPoint.self)
        try downsample.parameters(for: .bias).copy(
          from: op_bias, zip: archive, of: UseFloatingPoint.self)
      }
      readers.append(reader)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    let input_blocks_0_0_weight =
      state_dict["model.diffusion_model.input_blocks.0.0.weight"]
      as! TensorDescriptor
    let input_blocks_0_0_bias =
      state_dict["model.diffusion_model.input_blocks.0.0.bias"] as! TensorDescriptor
    try conv2d.parameters(for: .weight).copy(
      from: input_blocks_0_0_weight, zip: archive, of: UseFloatingPoint.self)
    try conv2d.parameters(for: .bias).copy(
      from: input_blocks_0_0_bias, zip: archive, of: UseFloatingPoint.self)
    for reader in readers {
      try reader(state_dict)
    }
  }
  return (reader, passLayers, out)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeads: Int, batchSize: Int, startHeight: Int, startWidth: Int,
  embeddingSize: Int, attentionRes: Set<Int>, x: Model.IO, emb: Model.IO, c: Model.IO,
  inputs: [Model.IO]
) -> ((Interpreter.Dictionary) throws -> Void, Model.IO) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [(Interpreter.Dictionary) throws -> Void]()
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes.contains(ds)
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let (reader, outputLayer) = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeads: numHeads, batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      if attentionBlock {
        out = outputLayer(out, emb, c)
      } else {
        out = outputLayer(out, emb)
      }
      readers.append(reader)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock ? 2 : 1
        let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
          let op_weight =
            state_dict[
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight"
            ] as! TensorDescriptor
          let op_bias =
            state_dict[
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias"
            ]
            as! TensorDescriptor
          try conv2d.parameters(for: .weight).copy(
            from: op_weight, zip: archive, of: UseFloatingPoint.self)
          try conv2d.parameters(for: .bias).copy(
            from: op_bias, zip: archive, of: UseFloatingPoint.self)
        }
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    for reader in readers {
      try reader(state_dict)
    }
  }
  return (reader, out)
}

func UNet(batchSize: Int) -> ((Interpreter.Dictionary) throws -> Void, Model) {
  let x = Input()
  let t_emb = Input()
  let c = Input()
  let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
  let emb = timeEmbed(t_emb)
  let attentionRes = Set([4, 2, 1])
  let (inputReader, inputs, inputBlocks) = InputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: 64,
    startWidth: 64, embeddingSize: 77, attentionRes: attentionRes, x: x, emb: emb, c: c)
  var out = inputBlocks
  let (middleReader, middleBlock) = MiddleBlock(
    channels: 1280, numHeads: 8, batchSize: batchSize, height: 8, width: 8, embeddingSize: 77,
    x: out,
    emb: emb, c: c)
  out = middleBlock
  let (outputReader, outputBlocks) = OutputBlocks(
    channels: [320, 640, 1280, 1280], numRepeat: 2, numHeads: 8, batchSize: batchSize,
    startHeight: 64,
    startWidth: 64, embeddingSize: 77, attentionRes: attentionRes, x: out, emb: emb, c: c,
    inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    let time_embed_0_weight =
      state_dict["model.diffusion_model.time_embed.0.weight"] as! TensorDescriptor
    let time_embed_0_bias =
      state_dict["model.diffusion_model.time_embed.0.bias"] as! TensorDescriptor
    let time_embed_2_weight =
      state_dict["model.diffusion_model.time_embed.2.weight"] as! TensorDescriptor
    let time_embed_2_bias =
      state_dict["model.diffusion_model.time_embed.2.bias"] as! TensorDescriptor
    try fc0.parameters(for: .weight).copy(
      from: time_embed_0_weight, zip: archive, of: UseFloatingPoint.self)
    try fc0.parameters(for: .bias).copy(
      from: time_embed_0_bias, zip: archive, of: UseFloatingPoint.self)
    try fc2.parameters(for: .weight).copy(
      from: time_embed_2_weight, zip: archive, of: UseFloatingPoint.self)
    try fc2.parameters(for: .bias).copy(
      from: time_embed_2_bias, zip: archive, of: UseFloatingPoint.self)
    try inputReader(state_dict)
    try middleReader(state_dict)
    try outputReader(state_dict)
    let out_0_weight = state_dict["model.diffusion_model.out.0.weight"] as! TensorDescriptor
    let out_0_bias = state_dict["model.diffusion_model.out.0.bias"] as! TensorDescriptor
    try outNorm.parameters(for: .weight).copy(
      from: out_0_weight, zip: archive, of: UseFloatingPoint.self)
    try outNorm.parameters(for: .bias).copy(
      from: out_0_bias, zip: archive, of: UseFloatingPoint.self)
    let out_2_weight = state_dict["model.diffusion_model.out.2.weight"] as! TensorDescriptor
    let out_2_bias = state_dict["model.diffusion_model.out.2.bias"] as! TensorDescriptor
    try outConv2d.parameters(for: .weight).copy(
      from: out_2_weight, zip: archive, of: UseFloatingPoint.self)
    try outConv2d.parameters(for: .bias).copy(
      from: out_2_bias, zip: archive, of: UseFloatingPoint.self)
  }
  return (reader, Model([x, t_emb, c], [out]))
}

func ResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> (
  (Interpreter.Dictionary) throws -> Void, Model
) {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  out = Swish()(out)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = norm2(out)
  out = Swish()(out)
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = nin(x) + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    let norm1_weight = state_dict["first_stage_model.\(prefix).norm1.weight"] as! TensorDescriptor
    let norm1_bias = state_dict["first_stage_model.\(prefix).norm1.bias"] as! TensorDescriptor
    try norm1.parameters(for: .weight).copy(from: norm1_weight, zip: archive, of: UseFloatingPoint.self)
    try norm1.parameters(for: .bias).copy(from: norm1_bias, zip: archive, of: UseFloatingPoint.self)
    let conv1_weight = state_dict["first_stage_model.\(prefix).conv1.weight"] as! TensorDescriptor
    let conv1_bias = state_dict["first_stage_model.\(prefix).conv1.bias"] as! TensorDescriptor
    try conv1.parameters(for: .weight).copy(from: conv1_weight, zip: archive, of: UseFloatingPoint.self)
    try conv1.parameters(for: .bias).copy(from: conv1_bias, zip: archive, of: UseFloatingPoint.self)
    let norm2_weight = state_dict["first_stage_model.\(prefix).norm2.weight"] as! TensorDescriptor
    let norm2_bias = state_dict["first_stage_model.\(prefix).norm2.bias"] as! TensorDescriptor
    try norm2.parameters(for: .weight).copy(from: norm2_weight, zip: archive, of: UseFloatingPoint.self)
    try norm2.parameters(for: .bias).copy(from: norm2_bias, zip: archive, of: UseFloatingPoint.self)
    let conv2_weight = state_dict["first_stage_model.\(prefix).conv2.weight"] as! TensorDescriptor
    let conv2_bias = state_dict["first_stage_model.\(prefix).conv2.bias"] as! TensorDescriptor
    try conv2.parameters(for: .weight).copy(from: conv2_weight, zip: archive, of: UseFloatingPoint.self)
    try conv2.parameters(for: .bias).copy(from: conv2_bias, zip: archive, of: UseFloatingPoint.self)
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["first_stage_model.\(prefix).nin_shortcut.weight"] as! TensorDescriptor
      let nin_shortcut_bias = state_dict["first_stage_model.\(prefix).nin_shortcut.bias"] as! TensorDescriptor
      try ninShortcut.parameters(for: .weight).copy(
        from: nin_shortcut_weight, zip: archive, of: UseFloatingPoint.self)
      try ninShortcut.parameters(for: .bias).copy(from: nin_shortcut_bias, zip: archive, of: UseFloatingPoint.self)
    }
  }
  return (reader, Model([x], [out]))
}

func AttnBlock(prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int) -> (
  (Interpreter.Dictionary) throws -> Void, Model
) {
  let x = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let hw = width * height
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let k = tokeys(out).reshaped([batchSize, inChannels, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    batchSize, inChannels, hw,
  ])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let v = tovalues(out).reshaped([batchSize, inChannels, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = x + projOut(out.reshaped([batchSize, inChannels, height, width]))
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    let norm_weight = state_dict["first_stage_model.\(prefix).norm.weight"] as! TensorDescriptor
    let norm_bias = state_dict["first_stage_model.\(prefix).norm.bias"] as! TensorDescriptor
    try norm.parameters(for: .weight).copy(from: norm_weight, zip: archive, of: UseFloatingPoint.self)
    try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: UseFloatingPoint.self)
    let k_weight = state_dict["first_stage_model.\(prefix).k.weight"] as! TensorDescriptor
    let k_bias = state_dict["first_stage_model.\(prefix).k.bias"] as! TensorDescriptor
    try tokeys.parameters(for: .weight).copy(from: k_weight, zip: archive, of: UseFloatingPoint.self)
    try tokeys.parameters(for: .bias).copy(from: k_bias, zip: archive, of: UseFloatingPoint.self)
    let q_weight = state_dict["first_stage_model.\(prefix).q.weight"] as! TensorDescriptor
    let q_bias = state_dict["first_stage_model.\(prefix).q.bias"] as! TensorDescriptor
    try toqueries.parameters(for: .weight).copy(from: q_weight, zip: archive, of: UseFloatingPoint.self)
    try toqueries.parameters(for: .bias).copy(from: q_bias, zip: archive, of: UseFloatingPoint.self)
    let v_weight = state_dict["first_stage_model.\(prefix).v.weight"] as! TensorDescriptor
    let v_bias = state_dict["first_stage_model.\(prefix).v.bias"] as! TensorDescriptor
    try tovalues.parameters(for: .weight).copy(from: v_weight, zip: archive, of: UseFloatingPoint.self)
    try tovalues.parameters(for: .bias).copy(from: v_bias, zip: archive, of: UseFloatingPoint.self)
    let proj_out_weight = state_dict["first_stage_model.\(prefix).proj_out.weight"] as! TensorDescriptor
    let proj_out_bias = state_dict["first_stage_model.\(prefix).proj_out.bias"] as! TensorDescriptor
    try projOut.parameters(for: .weight).copy(from: proj_out_weight, zip: archive, of: UseFloatingPoint.self)
    try projOut.parameters(for: .bias).copy(from: proj_out_bias, zip: archive, of: UseFloatingPoint.self)
  }
  return (reader, Model([x], [out]))
}

func Encoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((Interpreter.Dictionary) throws -> Void, Model)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var readers = [(Interpreter.Dictionary) throws -> Void]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (reader, block) = ResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", outChannels: channel, shortcut: previousChannel != channel)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      // Conv always pad left first, then right, and pad top first then bottom.
      // Thus, we cannot have (0, 1, 0, 1) (left 0, right 1, top 0, bottom 1) padding as in
      // Stable Diffusion. Instead, we pad to (2, 1, 2, 1) and simply discard the first row and first column.
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])))
      out = conv2d(out).reshaped(
        [batchSize, channel, height, width], offset: [0, 0, 1, 1],
        strides: [channel * (height + 1) * (width + 1), (height + 1) * (width + 1), width + 1, 1])
      let downLayer = i
      let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
        let conv_weight = state_dict["first_stage_model.encoder.down.\(downLayer).downsample.conv.weight"] as! TensorDescriptor
        let conv_bias = state_dict["first_stage_model.encoder.down.\(downLayer).downsample.conv.bias"] as! TensorDescriptor
        try conv2d.parameters(for: .weight).copy(from: conv_weight, zip: archive, of: UseFloatingPoint.self)
        try conv2d.parameters(for: .bias).copy(from: conv_bias, zip: archive, of: UseFloatingPoint.self)
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize, width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "encoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 8, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let quantConv2d = Convolution(
    groups: 1, filters: 8, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = quantConv2d(out)
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    let conv_in_weight = state_dict["first_stage_model.encoder.conv_in.weight"] as! TensorDescriptor
    let conv_in_bias = state_dict["first_stage_model.encoder.conv_in.bias"] as! TensorDescriptor
    try convIn.parameters(for: .weight).copy(from: conv_in_weight, zip: archive, of: UseFloatingPoint.self)
    try convIn.parameters(for: .bias).copy(from: conv_in_bias, zip: archive, of: UseFloatingPoint.self)
    for reader in readers {
      try reader(state_dict)
    }
    try midBlockReader1(state_dict)
    try midAttnReader1(state_dict)
    try midBlockReader2(state_dict)
    let norm_out_weight = state_dict["first_stage_model.encoder.norm_out.weight"] as! TensorDescriptor
    let norm_out_bias = state_dict["first_stage_model.encoder.norm_out.bias"] as! TensorDescriptor
    try normOut.parameters(for: .weight).copy(from: norm_out_weight, zip: archive, of: UseFloatingPoint.self)
    try normOut.parameters(for: .bias).copy(from: norm_out_bias, zip: archive, of: UseFloatingPoint.self)
    let conv_out_weight = state_dict["first_stage_model.encoder.conv_out.weight"] as! TensorDescriptor
    let conv_out_bias = state_dict["first_stage_model.encoder.conv_out.bias"] as! TensorDescriptor
    try convOut.parameters(for: .weight).copy(from: conv_out_weight, zip: archive, of: UseFloatingPoint.self)
    try convOut.parameters(for: .bias).copy(from: conv_out_bias, zip: archive, of: UseFloatingPoint.self)
    let quant_conv_weight = state_dict["first_stage_model.quant_conv.weight"] as! TensorDescriptor
    let quant_conv_bias = state_dict["first_stage_model.quant_conv.bias"] as! TensorDescriptor
    try quantConv2d.parameters(for: .weight).copy(
      from: quant_conv_weight, zip: archive, of: UseFloatingPoint.self)
    try quantConv2d.parameters(for: .bias).copy(
      from: quant_conv_bias, zip: archive, of: UseFloatingPoint.self)
  }
  return (reader, Model([x], [out]))
}

func Decoder(channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int)
  -> ((Interpreter.Dictionary) throws -> Void, Model)
{
  let x = Input()
  let postQuantConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  var out = postQuantConv2d(x)
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convIn(out)
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "decoder.mid.block_1", outChannels: previousChannel, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: previousChannel, batchSize: batchSize, width: startWidth,
    height: startHeight)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "decoder.mid.block_2", outChannels: previousChannel, shortcut: false)
  out = midBlock2(out)
  var readers = [(Interpreter.Dictionary) throws -> Void]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (reader, block) = ResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", outChannels: channel, shortcut: previousChannel != channel)
      readers.append(reader)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv2d(out)
      let upLayer = i
      let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
        let conv_weight = state_dict["first_stage_model.decoder.up.\(upLayer).upsample.conv.weight"] as! TensorDescriptor
        let conv_bias = state_dict["first_stage_model.decoder.up.\(upLayer).upsample.conv.bias"] as! TensorDescriptor
        try conv2d.parameters(for: .weight).copy(from: conv_weight, zip: archive, of: UseFloatingPoint.self)
        try conv2d.parameters(for: .bias).copy(from: conv_bias, zip: archive, of: UseFloatingPoint.self)
      }
      readers.append(reader)
    }
  }
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = Swish()(out)
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (Interpreter.Dictionary) throws -> Void = { state_dict in
    let post_quant_conv_weight = state_dict["first_stage_model.post_quant_conv.weight"] as! TensorDescriptor
    let post_quant_conv_bias = state_dict["first_stage_model.post_quant_conv.bias"] as! TensorDescriptor
    try postQuantConv2d.parameters(for: .weight).copy(
      from: post_quant_conv_weight, zip: archive, of: UseFloatingPoint.self)
    try postQuantConv2d.parameters(for: .bias).copy(
      from: post_quant_conv_bias, zip: archive, of: UseFloatingPoint.self)
    let conv_in_weight = state_dict["first_stage_model.decoder.conv_in.weight"] as! TensorDescriptor
    let conv_in_bias = state_dict["first_stage_model.decoder.conv_in.bias"] as! TensorDescriptor
    try convIn.parameters(for: .weight).copy(from: conv_in_weight, zip: archive, of: UseFloatingPoint.self)
    try convIn.parameters(for: .bias).copy(from: conv_in_bias, zip: archive, of: UseFloatingPoint.self)
    try midBlockReader1(state_dict)
    try midAttnReader1(state_dict)
    try midBlockReader2(state_dict)
    for reader in readers {
      try reader(state_dict)
    }
    let norm_out_weight = state_dict["first_stage_model.decoder.norm_out.weight"] as! TensorDescriptor
    let norm_out_bias = state_dict["first_stage_model.decoder.norm_out.bias"] as! TensorDescriptor
    try normOut.parameters(for: .weight).copy(from: norm_out_weight, zip: archive, of: UseFloatingPoint.self)
    try normOut.parameters(for: .bias).copy(from: norm_out_bias, zip: archive, of: UseFloatingPoint.self)
    let conv_out_weight = state_dict["first_stage_model.decoder.conv_out.weight"] as! TensorDescriptor
    let conv_out_bias = state_dict["first_stage_model.decoder.conv_out.bias"] as! TensorDescriptor
    try convOut.parameters(for: .weight).copy(from: conv_out_weight, zip: archive, of: UseFloatingPoint.self)
    try convOut.parameters(for: .bias).copy(from: conv_out_bias, zip: archive, of: UseFloatingPoint.self)
  }
  return (reader, Model([x], [out]))
}

let (
  tokenEmbed, positionEmbed, layerNorm1s, tokeys, toqueries, tovalues, unifyheads, layerNorm2s,
  fc1s, fc2s, finalLayerNorm, textModel
) = CLIPTextModel(
  UseFloatingPoint.self,
  vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 12, numHeads: 12,
  batchSize: 1, intermediateSize: 3072)

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

let t_emb = graph.variable(
  Tensor<UseFloatingPoint>(
    from: timeEmbedding(timesteps: 981, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000))
).toGPU(0)
let xTensor = graph.variable(.CPU, .NCHW(2, 4, 64, 64), of: UseFloatingPoint.self).toGPU(0)
let cTensor = graph.variable(.CPU, .CHW(2, 77, 768), of: UseFloatingPoint.self).toGPU(0)
let (unetReader, unet) = UNet(batchSize: 2)

let encoderTensor = graph.variable(.CPU, .NCHW(1, 3, 512, 512), of: UseFloatingPoint.self).toGPU(0)
let (encoderReader, encoder) = Encoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64, startHeight: 64)

let decoderTensor = graph.variable(.CPU, .NCHW(1, 4, 64, 64), of: UseFloatingPoint.self).toGPU(0)
let (decoderReader, decoder) = Decoder(
  channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64, startHeight: 64)

try graph.withNoGrad {
  /// Load UNet.
  unet.compile(inputs: xTensor, t_emb, cTensor)
  try unetReader(state_dict)
  graph.openStore("/home/liu/workspace/swift-diffusion/unet.ckpt") {
    $0.write("unet", model: unet)
  }

  /// Load text model.
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

  /// Load Autoencoder.
  decoder.compile(inputs: decoderTensor)
  encoder.compile(inputs: encoderTensor)
  try decoderReader(state_dict)
  try encoderReader(state_dict)
  graph.openStore("/home/liu/workspace/swift-diffusion/autoencoder.ckpt") {
    $0.write("decoder", model: decoder)
    $0.write("encoder", model: encoder)
  }
}

