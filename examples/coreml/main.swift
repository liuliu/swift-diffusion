import Diffusion
import Foundation
import NNC

public typealias UseFloatingPoint = Float16

struct KeyAndRange: Codable {
  var key: String
  var idx: Int
  var count: Int
  var bias: Bool? = nil
}

let graph = DynamicGraph()
var data1 = try Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/coreml-stable-diffusion-v1-5_split_einsum_compiled/UnetChunk1.mlmodelc/weights/weight.bin"))
var data2 = try Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/coreml-stable-diffusion-v1-5_split_einsum_compiled/UnetChunk2.mlmodelc/weights/weight.bin"))
let model2 = try Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/coreml-stable-diffusion-v1-5_split_einsum_compiled/UnetChunk2.mlmodelc/model.mil"))
let jsonDecoder = JSONDecoder()
let keyAndRange1 = try jsonDecoder.decode([KeyAndRange].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/examples/coreml/data_layout1.json")))
let keyAndRange2 = try jsonDecoder.decode([KeyAndRange].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/examples/coreml/data_layout2.json")))
let keys1 = Dictionary<String, KeyAndRange>(uniqueKeysWithValues: keyAndRange1.map { ($0.key, $0) })
let keys2 = Dictionary<String, KeyAndRange>(uniqueKeysWithValues: keyAndRange2.map { ($0.key, $0) })
for kr in keyAndRange1 {
  print("\"\(kr.key)\": TensorNameAndBlobOffset(name: \"\(kr.key)\", offset: \(kr.idx * 2), isFirstChunk: true, isLayerNormBias: \(kr.bias ?? false)),")
}
for kr in keyAndRange2 {
  print("\"\(kr.key)\": TensorNameAndBlobOffset(name: \"\(kr.key)\", offset: \(kr.idx * 2), isFirstChunk: false, isLayerNormBias: \(kr.bias ?? false)),")
}
/*
graph.openStore("/home/liu/workspace/swift-diffusion/wd_v1.3_f16.ckpt") {
  for key in $0.keys {
    guard let tensor = $0.read(key) else { continue }
    var f16tensor = Tensor<UseFloatingPoint>(tensor)
    if let range1 = keys1[key] {
      if range1.bias == true {
        let biastensor = graph.constant(f16tensor)
        let weighttensor = graph.constant(Tensor<UseFloatingPoint>($0.read(key.dropLast(2) + "0]")!))
        f16tensor = (biastensor .* Functional.reciprocal(weighttensor)).rawValue.toCPU()
      } else {
        f16tensor = f16tensor.toCPU()
      }
      f16tensor.withUnsafeBytes {
        let u8p = $0.baseAddress!.assumingMemoryBound(to: UInt8.self)
        for i in 0..<$0.count {
          data1[i + range1.idx * 2] = 0 // u8p[i]
        }
      }
    } else if let range2 = keys2[key] {
      if range2.bias == true {
        let biastensor = graph.constant(f16tensor)
        let weighttensor = graph.constant(Tensor<UseFloatingPoint>($0.read(key.dropLast(2) + "0]")!))
        f16tensor = (biastensor .* Functional.reciprocal(weighttensor)).rawValue.toCPU()
      } else {
        f16tensor = f16tensor.toCPU()
      }
      f16tensor.withUnsafeBytes {
        let u8p = $0.baseAddress!.assumingMemoryBound(to: UInt8.self)
        for i in 0..<$0.count {
          data2[i + range2.idx * 2] = 0 // u8p[i]
        }
      }
    } else if key == "__unet__[t-406-1]" { // These are immediate values in model2 file.
      f16tensor = f16tensor.toCPU()
      print(String(format: "%a, %a, %a, %a", Double(f16tensor[0]), Double(f16tensor[1]), Double(f16tensor[2]), Double(f16tensor[3])))
    }
  }
}
try data1.write(to: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/weight1.bin"))
try data2.write(to: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/weight2.bin"))
*/
/*
var keyAndRange = [KeyAndRange]()
graph.openStore("/home/liu/workspace/swift-diffusion/sd_v1.5_f16.ckpt") {
  for key in $0.keys {
    guard let tensor = $0.read(key) else { continue }
    print("checking key \(key)")
    let f16tensor = Tensor<UseFloatingPoint>(tensor).toCPU()
    f16tensor.withUnsafeBytes {
      let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
      let tcount = $0.count / 2
      data1.withUnsafeBytes {
        let udatap = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
        let datacount = $0.count / 2
        var found = false
        for i in 0..<(datacount - tcount) {
          var flag = true
          for j in 0..<tcount {
            if abs(udatap[i + j] - f16p[j]) > 1e-3 {
              flag = false
              break
            }
          }
          if flag {
            print("key \(key) on data \(i)..<\(i + tcount)")
            keyAndRange.append(KeyAndRange(key: key, idx: i, count: tcount))
            found = true
            break
          }
        }
        if !found {
          print("cannot find for \(f16tensor)")
        }
      }
    }
  }
}

let jsonEncoder = JSONEncoder()
jsonEncoder.outputFormatting = .prettyPrinted
let jsonData = try jsonEncoder.encode(keyAndRange)
try jsonData.write(to: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/data_layout1.json"))
let jsonDecoder = JSONDecoder()
let keyAndRange1 = try jsonDecoder.decode([KeyAndRange].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/data_layout1.json")))
let keyAndRange2 = try jsonDecoder.decode([KeyAndRange].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/data_layout2.json")))
var keys = [String]()
graph.openStore("/home/liu/workspace/swift-diffusion/sd_v1.5_f16.ckpt") {
  keys = $0.keys
  /*
  let introKeys1 = ["__unet__[t-406-1]", "__unet__[t-181-1]", "__unet__[t-182-1]"]
  let introKeys2 = ["__unet__[t-406-1]", "__unet__[t-187-1]", "__unet__[t-188-1]", "__unet__[t-193-1]", "__unet__[t-194-1]", "__unet__[t-200-1]", "__unet__[t-201-1]", "__unet__[t-223-1]", "__unet__[t-224-1]", "__unet__[t-246-1]", "__unet__[t-247-1]", "__unet__[t-270-1]", "__unet__[t-271-1]", "__unet__[t-293-1]", "__unet__[t-294-1]"]
  for key in introKeys1 {
    print("checking key \(key)")
    let f16tensor = Tensor<UseFloatingPoint>($0.read(key)!).toCPU()
    f16tensor.withUnsafeBytes {
      let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
      let tcount = $0.count / 2
      data1.withUnsafeBytes {
        let udatap = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
        let datacount = $0.count / 2
        var found = false
        for i in 0..<(datacount - tcount) {
          var flag = true
          for j in 0..<tcount {
            if abs(udatap[i + j] - f16p[j]) > 1e-5 {
              flag = false
              break
            }
          }
          if flag {
            print("key \(key) on data \(i)..<\(i + tcount)")
            found = true
            break
          }
        }
        if !found {
          print("cannot find for \(f16tensor)")
        }
      }
    }
  }
  */
}
for kv1 in keyAndRange1 {
  for kv2 in keyAndRange2 {
    if kv1.key == kv2.key {
      print("key in 1: \(kv1.key)")
      print("key in 2: \(kv2.key)")
      print("everything else  in 1: \(kv1), \(kv2)")
    }
  }
}
for kv1 in keyAndRange1 {
  for kv2 in keyAndRange1 {
    guard kv1.key != kv2.key else { continue }
    if kv1.idx < kv2.idx + kv2.count && kv1.idx + kv1.count > kv2.idx {
      // overlap.
      print("overlap between \(kv1), \(kv2) in layout1")
    }
  }
}
for kv1 in keyAndRange2 {
  for kv2 in keyAndRange2 {
    guard kv1.key != kv2.key else { continue }
    if kv1.idx < kv2.idx + kv2.count && kv1.idx + kv1.count > kv2.idx {
      // overlap.
      print("overlap between \(kv1), \(kv2) in layout2")
    }
  }
}
var existingKeys = Set<String>()
for kv1 in keyAndRange1 {
  existingKeys.insert(kv1.key)
}
for kv2 in keyAndRange2 {
  existingKeys.insert(kv2.key)
}

graph.openStore("/home/liu/workspace/swift-diffusion/sd_v1.5_f16.ckpt") {
  for key in keys {
    if !existingKeys.contains(key) {
      // guard key.hasSuffix("-1]") else { continue }
      let biastensor = graph.constant(Tensor<UseFloatingPoint>($0.read(key)!))
      print("check for key \(key), \(biastensor)")
      /*
      let weighttensor = graph.constant(Tensor<UseFloatingPoint>($0.read(key.dropLast(2) + "0]")!))
      guard weighttensor.shape.reduce(1, *) == biastensor.shape.reduce(1, *) else { continue }
      let f16tensor = (biastensor .* Functional.reciprocal(weighttensor)).rawValue.toCPU()
      */
      let f16tensor = biastensor.rawValue.toCPU()
      debugPrint(f16tensor)
      f16tensor.withUnsafeBytes {
        let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
        let tcount = $0.count / 2
        data2.withUnsafeBytes {
          let udatap = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
          let datacount = $0.count / 2
          var found = false
          for i in 0..<(datacount - tcount + 1) {
            var flag = true
            for j in 0..<tcount {
              if abs(udatap[i + j] - f16p[j]) > 1e-4 {
                flag = false
                break
              }
            }
            if flag {
              print("key \(key) on data \(i)..<\(i + tcount)")
              found = true
              break
            }
          }
          if !found {
            print("cannot find for \(f16tensor)")
          }
        }
      }
    }
  }
}
*/
