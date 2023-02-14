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
var data1 = try Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/instruct_pix2pix/UnetChunk1.mlmodelc/weights/weight.bin"))
var data2 = try Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/instruct_pix2pix/UnetChunk2.mlmodelc/weights/weight.bin"))
let model2 = try String(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/instruct_pix2pix/UnetChunk2.mlmodelc/model.mil"), encoding: .utf8)
let jsonDecoder = JSONDecoder()
let keyAndRange1 = try jsonDecoder.decode([KeyAndRange].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/instruct_pix2pix_data_layout1.json")))
let keyAndRange2 = try jsonDecoder.decode([KeyAndRange].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/instruct_pix2pix_data_layout2.json")))
let keys1 = Dictionary<String, KeyAndRange>(uniqueKeysWithValues: keyAndRange1.map { ($0.key, $0) })
let keys2 = Dictionary<String, KeyAndRange>(uniqueKeysWithValues: keyAndRange2.map { ($0.key, $0) })
for kr in keyAndRange1 {
  print("\"\(kr.key)\": TensorNameAndBlobOffset(offset: \(kr.idx * 2), isFirstChunk: true, isLayerNormBias: \(kr.bias ?? false)),")
}
for kr in keyAndRange2 {
  print("\"\(kr.key)\": TensorNameAndBlobOffset(offset: \(kr.idx * 2), isFirstChunk: false, isLayerNormBias: \(kr.bias ?? false)),")
}
graph.openStore("/home/liu/workspace/swift-diffusion/instruct_pix2pix_22000_f16.ckpt") {
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
/*
var keyAndRange1 = [KeyAndRange]()
var keyAndRange2 = [KeyAndRange]()
graph.openStore("/home/liu/workspace/swift-diffusion/instruct_pix2pix_22000_f16.ckpt") {
  for key in $0.keys {
    guard let tensor = $0.read(key) else { continue }
    print("checking key \(key)")
    let f16tensor = Tensor<UseFloatingPoint>(tensor).toCPU()
    var f16tensorB: Tensor<UseFloatingPoint>?
    if key.hasSuffix("-1]") {
      let biastensor = graph.constant(Tensor<UseFloatingPoint>($0.read(key)!))
      let weighttensor = graph.constant(Tensor<UseFloatingPoint>($0.read(key.dropLast(2) + "0]")!))
      if weighttensor.shape.reduce(1, *) == biastensor.shape.reduce(1, *) {
        f16tensorB = (biastensor .* Functional.reciprocal(weighttensor)).rawValue.toCPU()
      } else {
        f16tensorB = nil
      }
    } else {
      f16tensorB = nil
    }
    var found = false
    data1.withUnsafeBytes {
      let udatap = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
      let datacount = $0.count / 2
      f16tensor.withUnsafeBytes {
        let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
        let tcount = $0.count / 2
        for i in 0..<(datacount - tcount + 1) {
          var flag = true
          for j in 0..<tcount {
            if abs(udatap[i + j] - f16p[j]) > 1e-2 {
              flag = false
              break
            }
          }
          if flag {
            print("key \(key) on data \(i)..<\(i + tcount)")
            keyAndRange1.append(KeyAndRange(key: key, idx: i, count: tcount))
            found = true
            break
          }
        }
      }
      if !found, let f16tensorB = f16tensorB {
        f16tensorB.withUnsafeBytes {
          let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
          let tcount = $0.count / 2
          for i in 0..<(datacount - tcount + 1) {
            var flag = true
            for j in 0..<tcount {
              if abs(udatap[i + j] - f16p[j]) > 1e-2 {
                flag = false
                break
              }
            }
            if flag {
              print("key \(key) on data \(i)..<\(i + tcount)")
              keyAndRange1.append(KeyAndRange(key: key, idx: i, count: tcount, bias: true))
              found = true
              break
            }
          }
        }
      }
    }
    if !found {
      data2.withUnsafeBytes {
        let udatap = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
        let datacount = $0.count / 2
        f16tensor.withUnsafeBytes {
          let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
          let tcount = $0.count / 2
          for i in 0..<(datacount - tcount + 1) {
            var flag = true
            for j in 0..<tcount {
              if abs(udatap[i + j] - f16p[j]) > 1e-2 {
                flag = false
                break
              }
            }
            if flag {
              print("key \(key) on data \(i)..<\(i + tcount)")
              keyAndRange2.append(KeyAndRange(key: key, idx: i, count: tcount))
              found = true
              break
            }
          }
        }
        if !found, let f16tensorB = f16tensorB {
          f16tensorB.withUnsafeBytes {
            let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
            let tcount = $0.count / 2
            for i in 0..<(datacount - tcount + 1) {
              var flag = true
              for j in 0..<tcount {
                if abs(udatap[i + j] - f16p[j]) > 1e-2 {
                  flag = false
                  break
                }
              }
              if flag {
                print("key \(key) on data \(i)..<\(i + tcount)")
                keyAndRange2.append(KeyAndRange(key: key, idx: i, count: tcount, bias: true))
                found = true
                break
              }
            }
          }
        }
      }
    }
    if !found {
      print("cannot find for \(f16tensor)")
    }
  }
}

let jsonEncoder = JSONEncoder()
jsonEncoder.outputFormatting = .prettyPrinted
let jsonData1 = try jsonEncoder.encode(keyAndRange1)
try jsonData1.write(to: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/data_layout1.json"))
let jsonData2 = try jsonEncoder.encode(keyAndRange2)
try jsonData2.write(to: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/data_layout2.json"))
*/
/*
let jsonDecoder = JSONDecoder()
var keyAndRange1 = try jsonDecoder.decode([KeyAndRange].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/sd_v1_inpainting_data_layout1.json")))
var keyAndRange2 = try jsonDecoder.decode([KeyAndRange].self, from: Data(contentsOf: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/sd_v1_inpainting_data_layout2.json")))
var keys = [String]()
graph.openStore("/home/liu/workspace/swift-diffusion/sd_v2.0_inpainting_f16.ckpt") {
  keys = $0.keys
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

graph.openStore("/home/liu/workspace/swift-diffusion/sd_v2.0_inpainting_f16.ckpt") {
  for key in keys {
    guard !existingKeys.contains(key) else { continue }
    guard let tensor = $0.read(key) else { continue }
    print("checking key \(key)")
    let f16tensor = Tensor<UseFloatingPoint>(tensor).toCPU()
    var f16tensorB: Tensor<UseFloatingPoint>?
    if key.hasSuffix("-1]") {
      let biastensor = graph.constant(Tensor<UseFloatingPoint>($0.read(key)!))
      let weighttensor = graph.constant(Tensor<UseFloatingPoint>($0.read(key.dropLast(2) + "0]")!))
      if weighttensor.shape.reduce(1, *) == biastensor.shape.reduce(1, *) {
        f16tensorB = (biastensor .* Functional.reciprocal(weighttensor)).rawValue.toCPU()
      } else {
        f16tensorB = nil
      }
    } else {
      f16tensorB = nil
    }
    var found = false
    data1.withUnsafeBytes {
      let udatap = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
      let datacount = $0.count / 2
      f16tensor.withUnsafeBytes {
        let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
        let tcount = $0.count / 2
        for i in 0..<(datacount - tcount + 1) {
          var flag = true
          for j in 0..<tcount {
            if abs(udatap[i + j] - f16p[j]) > 1e-2 {
              flag = false
              break
            }
          }
          if flag {
            print("key \(key) on data \(i)..<\(i + tcount)")
            keyAndRange1.append(KeyAndRange(key: key, idx: i, count: tcount))
            found = true
            break
          }
        }
      }
      if !found, let f16tensorB = f16tensorB {
        f16tensorB.withUnsafeBytes {
          let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
          let tcount = $0.count / 2
          for i in 0..<(datacount - tcount + 1) {
            var flag = true
            for j in 0..<tcount {
              if abs(udatap[i + j] - f16p[j]) > 1e-2 {
                flag = false
                break
              }
            }
            if flag {
              print("key \(key) on data \(i)..<\(i + tcount)")
              keyAndRange1.append(KeyAndRange(key: key, idx: i, count: tcount, bias: true))
              found = true
              break
            }
          }
        }
      }
    }
    if !found {
      data2.withUnsafeBytes {
        let udatap = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
        let datacount = $0.count / 2
        f16tensor.withUnsafeBytes {
          let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
          let tcount = $0.count / 2
          for i in 0..<(datacount - tcount + 1) {
            var flag = true
            for j in 0..<tcount {
              if abs(udatap[i + j] - f16p[j]) > 1e-2 {
                flag = false
                break
              }
            }
            if flag {
              print("key \(key) on data \(i)..<\(i + tcount)")
              keyAndRange2.append(KeyAndRange(key: key, idx: i, count: tcount))
              found = true
              break
            }
          }
        }
        if !found, let f16tensorB = f16tensorB {
          f16tensorB.withUnsafeBytes {
            let f16p = $0.baseAddress!.assumingMemoryBound(to: Float16.self)
            let tcount = $0.count / 2
            for i in 0..<(datacount - tcount + 1) {
              var flag = true
              for j in 0..<tcount {
                if abs(udatap[i + j] - f16p[j]) > 1e-2 {
                  flag = false
                  break
                }
              }
              if flag {
                print("key \(key) on data \(i)..<\(i + tcount)")
                keyAndRange2.append(KeyAndRange(key: key, idx: i, count: tcount, bias: true))
                found = true
                break
              }
            }
          }
        }
      }
    }
    if !found {
      print("cannot find for \(f16tensor)")
    }
  }
}
let jsonEncoder = JSONEncoder()
jsonEncoder.outputFormatting = .prettyPrinted
let jsonData1 = try jsonEncoder.encode(keyAndRange1)
try jsonData1.write(to: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/updated_data_layout1.json"))
let jsonData2 = try jsonEncoder.encode(keyAndRange2)
try jsonData2.write(to: URL(fileURLWithPath: "/home/liu/workspace/swift-diffusion/updated_data_layout2.json"))
*/
