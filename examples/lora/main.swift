import Diffusion
import Foundation
import NNC

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
}

let safeTensors = SafeTensors(url: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/lucyCyberpunk_35Epochs.safetensors"))!

print(safeTensors.states.keys)
print(safeTensors.states["lora_te_text_model_encoder_layers_7_self_attn_out_proj.lora_up.weight"])
print(safeTensors.states["lora_te_text_model_encoder_layers_7_self_attn_out_proj.lora_down.weight"])
