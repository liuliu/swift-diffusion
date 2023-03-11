import Diffusion
import Foundation
import NNC
import ZIPFoundation
import Fickling
import Collections

public struct Storage {
  var name: String
  var size: Int
  var dataType: DataType
}

public struct TensorDescriptor {
  public var storage: Storage
  public var storageOffset: Int
  public var shape: [Int]
  public var strides: [Int]
}

public final class SafeTensors {
  private var data: Data
  private let bufferStart: Int
  public let states: [String: TensorDescriptor]
  public init?(url: URL) {
    guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
    guard data.count >= 8 else { return nil }
    let headerSize = data.withUnsafeBytes { $0.load(as: UInt64.self) }
    // It doesn't make sense for my use-case has more than 10MiB header.
    guard headerSize > 0 && headerSize < data.count + 8 && headerSize < 1_024 * 1_024 * 10 else {
      return nil
    }
    guard
      let jsonDict = try? JSONSerialization.jsonObject(
        with: data[8..<(8 + headerSize)]) as? [String: Any]
    else { return nil }
    var states = [String: TensorDescriptor]()
    for (key, value) in jsonDict {
      guard let value = value as? [String: Any], let offsets = value["data_offsets"] as? [Int],
        let dtype = (value["dtype"] as? String)?.lowercased(), let shape = value["shape"] as? [Int],
        offsets.count == 2 && shape.count > 0
      else { continue }
      let offsetStart = offsets[0]
      let offsetEnd = offsets[1]
      guard offsetEnd > offsetStart && offsetEnd <= data.count else { continue }
      guard !(shape.contains { $0 <= 0 }) else { continue }
      guard
        dtype == "f32" || dtype == "f16" || dtype == "float16" || dtype == "float32"
          || dtype == "float" || dtype == "half"
      else { continue }
      let dataType: DataType =
        dtype == "f32" || dtype == "float32" || dtype == "float" ? .Float32 : .Float16
      var strides = [Int]()
      var v = 1
      for i in stride(from: shape.count - 1, through: 0, by: -1) {
        strides.append(v)
        v *= shape[i]
      }
      strides.reverse()
      let tensorDescriptor = TensorDescriptor(
        storage: Storage(name: key, size: offsetEnd - offsetStart, dataType: dataType),
        storageOffset: offsetStart, shape: shape, strides: strides)
      states[key] = tensorDescriptor
    }
    self.data = data
    self.states = states
    bufferStart = 8 + Int(headerSize)
  }
  public func with<T>(_ tensorDescriptor: TensorDescriptor, block: (AnyTensor) throws -> T) throws
    -> T
  {
    // Don't subrange data, otherwise it will materialize the data into memory. Accessing the underlying
    // bytes directly, this way, it is just the mmap bytes, and we won't cause spike in memory usage.
    return try data.withUnsafeMutableBytes {
      guard let address = $0.baseAddress else { fatalError() }
      let tensor: AnyTensor
      if tensorDescriptor.storage.dataType == .Float16 {
        tensor = Tensor<Float16>(
          .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
          unsafeMutablePointer: (address + bufferStart + tensorDescriptor.storageOffset)
            .assumingMemoryBound(
              to: Float16.self), bindLifetimeOf: self
        )
      } else {
        tensor = Tensor<Float>(
          .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
          unsafeMutablePointer: (address + bufferStart + tensorDescriptor.storageOffset)
            .assumingMemoryBound(
              to: Float.self), bindLifetimeOf: self
        )
      }
      return try block(tensor)
    }
  }
}

let filename = "/home/liu/workspace/swift-diffusion/lucyCyberpunk_35Epochs.safetensors"
/*
let archive = Archive(url: URL(fileURLWithPath: filename), accessMode: .read)!
let entry = archive["archive/data.pkl"]!
var data = Data()
let _ = try archive.extract(entry) { data.append($0) }
let interpreter = Interpreter.from(data: data)
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
let model = (interpreter.rootObject as? Interpreter.Dictionary)!
model.forEach { key, value in
  print(key)
}
*/
let safeTensors = SafeTensors(url: URL(fileURLWithPath: filename))!

let unetDiffusersMap: [(String, String)] = [("input_blocks.1.0.", "down_blocks.0.resnets.0."), ("input_blocks.1.1.", "down_blocks.0.attentions.0."), ("input_blocks.2.0.", "down_blocks.0.resnets.1."), ("input_blocks.2.1.", "down_blocks.0.attentions.1."), ("output_blocks.0.0.", "up_blocks.0.resnets.0."), ("output_blocks.1.0.", "up_blocks.0.resnets.1."), ("output_blocks.2.0.", "up_blocks.0.resnets.2."), ("input_blocks.3.0.op.", "down_blocks.0.downsamplers.0.conv."), ("output_blocks.2.1.", "up_blocks.0.upsamplers.0."), ("input_blocks.4.0.", "down_blocks.1.resnets.0."), ("input_blocks.4.1.", "down_blocks.1.attentions.0."), ("input_blocks.5.0.", "down_blocks.1.resnets.1."), ("input_blocks.5.1.", "down_blocks.1.attentions.1."), ("output_blocks.3.0.", "up_blocks.1.resnets.0."), ("output_blocks.3.1.", "up_blocks.1.attentions.0."), ("output_blocks.4.0.", "up_blocks.1.resnets.1."), ("output_blocks.4.1.", "up_blocks.1.attentions.1."), ("output_blocks.5.0.", "up_blocks.1.resnets.2."), ("output_blocks.5.1.", "up_blocks.1.attentions.2."), ("input_blocks.6.0.op.", "down_blocks.1.downsamplers.0.conv."), ("output_blocks.5.2.", "up_blocks.1.upsamplers.0."), ("input_blocks.7.0.", "down_blocks.2.resnets.0."), ("input_blocks.7.1.", "down_blocks.2.attentions.0."), ("input_blocks.8.0.", "down_blocks.2.resnets.1."), ("input_blocks.8.1.", "down_blocks.2.attentions.1."), ("output_blocks.6.0.", "up_blocks.2.resnets.0."), ("output_blocks.6.1.", "up_blocks.2.attentions.0."), ("output_blocks.7.0.", "up_blocks.2.resnets.1."), ("output_blocks.7.1.", "up_blocks.2.attentions.1."), ("output_blocks.8.0.", "up_blocks.2.resnets.2."), ("output_blocks.8.1.", "up_blocks.2.attentions.2."), ("input_blocks.9.0.op.", "down_blocks.2.downsamplers.0.conv."), ("output_blocks.8.2.", "up_blocks.2.upsamplers.0."), ("input_blocks.10.0.", "down_blocks.3.resnets.0."), ("input_blocks.11.0.", "down_blocks.3.resnets.1."), ("output_blocks.9.0.", "up_blocks.3.resnets.0."), ("output_blocks.9.1.", "up_blocks.3.attentions.0."), ("output_blocks.10.0.", "up_blocks.3.resnets.1."), ("output_blocks.10.1.", "up_blocks.3.attentions.1."), ("output_blocks.11.0.", "up_blocks.3.resnets.2."), ("output_blocks.11.1.", "up_blocks.3.attentions.2."), ("middle_block.1.", "mid_block.attentions.0."), ("middle_block.0.", "mid_block.resnets.0."), ("middle_block.2.", "mid_block.resnets.1.")]

let vaeDiffusersMap: [(String, String)] = [("nin_shortcut", "conv_shortcut"), ("norm_out", "conv_norm_out"), ("mid.attn_1.", "mid_block.attentions.0."), ("encoder.down.0.block.0.", "encoder.down_blocks.0.resnets.0."), ("encoder.down.0.block.1.", "encoder.down_blocks.0.resnets.1."), ("down.0.downsample.", "down_blocks.0.downsamplers.0."), ("up.3.upsample.", "up_blocks.0.upsamplers.0."), ("decoder.up.3.block.0.", "decoder.up_blocks.0.resnets.0."), ("decoder.up.3.block.1.", "decoder.up_blocks.0.resnets.1."), ("decoder.up.3.block.2.", "decoder.up_blocks.0.resnets.2."), ("encoder.down.1.block.0.", "encoder.down_blocks.1.resnets.0."), ("encoder.down.1.block.1.", "encoder.down_blocks.1.resnets.1."), ("down.1.downsample.", "down_blocks.1.downsamplers.0."), ("up.2.upsample.", "up_blocks.1.upsamplers.0."), ("decoder.up.2.block.0.", "decoder.up_blocks.1.resnets.0."), ("decoder.up.2.block.1.", "decoder.up_blocks.1.resnets.1."), ("decoder.up.2.block.2.", "decoder.up_blocks.1.resnets.2."), ("encoder.down.2.block.0.", "encoder.down_blocks.2.resnets.0."), ("encoder.down.2.block.1.", "encoder.down_blocks.2.resnets.1."), ("down.2.downsample.", "down_blocks.2.downsamplers.0."), ("up.1.upsample.", "up_blocks.2.upsamplers.0."), ("decoder.up.1.block.0.", "decoder.up_blocks.2.resnets.0."), ("decoder.up.1.block.1.", "decoder.up_blocks.2.resnets.1."), ("decoder.up.1.block.2.", "decoder.up_blocks.2.resnets.2."), ("encoder.down.3.block.0.", "encoder.down_blocks.3.resnets.0."), ("encoder.down.3.block.1.", "encoder.down_blocks.3.resnets.1."), ("decoder.up.0.block.0.", "decoder.up_blocks.3.resnets.0."), ("decoder.up.0.block.1.", "decoder.up_blocks.3.resnets.1."), ("decoder.up.0.block.2.", "decoder.up_blocks.3.resnets.2."), ("mid.block_1.", "mid_block.resnets.0."), ("mid.block_2.", "mid_block.resnets.1.")]

let jsonDecoder = JSONDecoder()
var unetMap = try jsonDecoder.decode([String].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/unet.json")))
for i in stride(from: 0, to: unetMap.count, by: 2) {
  // Replacing to diffuser terminology, remove first two, then join with _ except the last part.
  for namePairs in unetDiffusersMap {
    if unetMap[i].contains(namePairs.0) {
      unetMap[i] = unetMap[i].replacingOccurrences(of: namePairs.0, with: namePairs.1)
      break
    }
  }
  let parts = unetMap[i].components(separatedBy: ".")
  unetMap[i] = parts[2..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]
}
var textModelMap = try jsonDecoder.decode([String].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/text_model.json")))
for i in stride(from: 0, to: textModelMap.count, by: 2) {
  let parts = textModelMap[i].components(separatedBy: ".")
  textModelMap[i] = parts[2..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]
}
let keys = safeTensors.states.keys
var keysSet = Set(keys)
for key in keys {
  let parts = key.components(separatedBy: "_")
  let newParts = String(parts[2..<parts.count].joined(separator: "_")).components(separatedBy: ".")
  let newKey = newParts[0..<newParts.count - 2].joined(separator: ".") + ".weight"
  if unetMap.contains(newKey) {
    keysSet.remove(key)
  }
  if textModelMap.contains(newKey) {
    keysSet.remove(key)
  }
}
var unetMapCount = [String: Int]()
for i in stride(from: 0, to: unetMap.count, by: 2) {
  unetMapCount[unetMap[i]] = unetMapCount[unetMap[i], default: 0] + 1
}
for key in unetMapCount.keys {
  if unetMapCount[key] ?? 0 <= 1 {
    unetMapCount[key] = nil
  }
}
var textModelMapCount = [String: Int]()
for i in stride(from: 0, to: textModelMap.count, by: 2) {
  textModelMapCount[textModelMap[i]] = textModelMapCount[textModelMap[i], default: 0] + 1
}
for key in textModelMapCount.keys {
  if textModelMapCount[key] ?? 0 <= 1 {
    textModelMapCount[key] = nil
  }
}
let graph = DynamicGraph()
try graph.openStore("/home/liu/workspace/swift-diffusion/lora.ckpt") { store in
  for (key, descriptor) in safeTensors.states {
    let parts = key.components(separatedBy: "_")
    let newParts = String(parts[2..<parts.count].joined(separator: "_")).components(separatedBy: ".")
    let newKey = newParts[0..<newParts.count - 2].joined(separator: ".") + ".weight"
    if let index = unetMap.firstIndex(of: newKey) {
      if key.hasSuffix("up.weight") {
        try safeTensors.with(descriptor) {
          let f16 = Tensor<Float16>(from: $0)
          if unetMapCount[newKey] ?? 0 >= 2 {
            store.write("__unet__[\(unetMap[index + 1])]__up__", tensor: f16[0..<(f16.shape[0] / 2), 0..<f16.shape[1]].copied())
            store.write("__unet__[\(unetMap[index + 5])]__up__", tensor: f16[(f16.shape[0] / 2)..<f16.shape[0], 0..<f16.shape[1]].copied())
          } else {
            store.write("__unet__[\(unetMap[index + 1])]__up__", tensor: f16)
          }
        }
      } else if key.hasSuffix("down.weight") {
        try safeTensors.with(descriptor) {
          let f16 = Tensor<Float16>(from: $0)
          store.write("__unet__[\(unetMap[index + 1])]__down__", tensor: f16)
          if unetMapCount[newKey] ?? 0 >= 2 {
            store.write("__unet__[\(unetMap[index + 5])]__down__", tensor: f16)
          }
        }
      }
    } else if let index = textModelMap.firstIndex(of: newKey) {
      if key.hasSuffix("up.weight") {
        try safeTensors.with(descriptor) {
          let f16 = Tensor<Float16>(from: $0)
          store.write("__text_model__[\(textModelMap[index + 1])]__up__", tensor: f16)
        }
      } else if key.hasSuffix("down.weight") {
        try safeTensors.with(descriptor) {
          let f16 = Tensor<Float16>(from: $0)
          store.write("__text_model__[\(textModelMap[index + 1])]__down__", tensor: f16)
        }
      }
    }
  }
}
