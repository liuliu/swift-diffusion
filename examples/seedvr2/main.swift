import Diffusion
import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

typealias FloatType = Float

setbuf(stdout, nil)

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

let sys = Python.import("sys")
let osPath = Python.import("os.path")
let site = Python.import("site")

let repoRoot = "/home/liu/workspace/swift-diffusion"
let seedVRRoot = "/home/liu/workspace/SeedVR"
let seedVRExampleRoot = "\(repoRoot)/examples/seedvr2"

let userSitePackages = site.getusersitepackages()
if (Bool(osPath.isdir(userSitePackages)) ?? false)
  && (Bool(sys.path.__contains__(userSitePackages)) ?? false) == false
{
  sys.path.insert(0, userSitePackages)
}
let systemDistPackages = "/usr/lib/python3/dist-packages"
if (Bool(osPath.isdir(systemDistPackages)) ?? false)
  && (Bool(sys.path.__contains__(systemDistPackages)) ?? false) == false
{
  sys.path.insert(0, systemDistPackages)
}
if (Bool(sys.path.__contains__(seedVRExampleRoot)) ?? false) == false {
  sys.path.insert(0, seedVRExampleRoot)
}
if (Bool(sys.path.__contains__(seedVRRoot)) ?? false) == false {
  sys.path.insert(0, seedVRRoot)
}

let torch = Python.import("torch")
let random = Python.import("random")
let numpy = Python.import("numpy")
let seedvr2Reference = Python.import("reference")
let processInfo = ProcessInfo.processInfo
let forceCPU = processInfo.environment["SEEDVR2_FORCE_CPU"] == "1"

torch.set_grad_enabled(false)

let hasCUDA = !forceCPU && (Bool(torch.cuda.is_available()) ?? false)
let torchDevice = hasCUDA ? torch.device("cuda") : torch.device("cpu")
let disableTF32 = processInfo.environment["SEEDVR2_DISABLE_TF32"] == "1"

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
if hasCUDA {
  torch.cuda.manual_seed_all(42)
  if disableTF32 {
    torch.backends.cuda.matmul.allow_tf32 = false
    torch.backends.cudnn.allow_tf32 = false
  }
}

let swiftDevice = 0

func logStep(_ message: String) {
  FileHandle.standardError.write(Data((message + "\n").utf8))
}

func placeOnDevice(_ tensor: DynamicGraph.Tensor<Float>) -> DynamicGraph.Tensor<Float> {
  if hasCUDA {
    return tensor.toGPU(swiftDevice)
  }
  return tensor
}

func placeOnDevice(_ tensor: DynamicGraph.Tensor<BFloat16>) -> DynamicGraph.Tensor<BFloat16> {
  if hasCUDA {
    return tensor.toGPU(swiftDevice)
  }
  return tensor
}

func copiedToCPU(_ tensor: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
  tensor.as(of: Float.self).rawValue.toCPU()
}

func rematerializeOnDevice(_ graph: DynamicGraph, _ tensor: DynamicGraph.Tensor<Float>)
  -> DynamicGraph.Tensor<Float>
{
  placeOnDevice(graph.variable(tensor.rawValue.toCPU()))
}

func rematerializeOnDevice(_ graph: DynamicGraph, _ tensor: Tensor<Float>)
  -> DynamicGraph.Tensor<Float>
{
  placeOnDevice(graph.variable(tensor))
}

func reducedMaxAbsDiff5D(
  _ graph: DynamicGraph, _ swiftTensor: DynamicGraph.Tensor<Float>, _ reference: Tensor<Float>
) -> Float {
  let count = reference.shape.reduce(1, *)
  let referenceTensor = placeOnDevice(graph.variable(reference))
  let diff = Functional.abs(swiftTensor - referenceTensor).reshaped(.NC(1, count)).reduced(
    .max, axis: [1])
  return diff.toCPU()[0, 0]
}

func reducedApproxMaxRelativeDiff5D(
  _ graph: DynamicGraph, _ swiftTensor: DynamicGraph.Tensor<Float>, _ reference: Tensor<Float>
) -> Float {
  let count = reference.shape.reduce(1, *)
  let referenceTensor = placeOnDevice(graph.variable(reference))
  let denom = Functional.abs(referenceTensor) + 1e-6
  let diff = (Functional.abs(swiftTensor - referenceTensor) .* Functional.reciprocal(denom))
    .reshaped(.NC(1, count)).reduced(.max, axis: [1])
  return diff.toCPU()[0, 0]
}

func maxAbsDiff5D(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float
{
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  precondition(swiftTensor.shape[4] == torchTensor.shape[4])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - Float(torchTensor[i, c, d, h, w]))
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
  }
  return maxDiff
}

func maxAbsDiff5D(_ swiftTensor: Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  precondition(swiftTensor.shape[4] == torchTensor.shape[4])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - Float(torchTensor[i, c, d, h, w]))
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
  }
  return maxDiff
}

func manualConv3DMaxAbsDiff(
  input: Tensor<Float>, weight: Tensor<Float>, bias: Tensor<Float>, output: Tensor<Float>
) -> Float {
  precondition(input.shape.count == 5)
  precondition(weight.shape.count == 5)
  precondition(output.shape.count == 5)
  let outChannels = output.shape[1]
  let outputDepth = output.shape[2]
  let outputHeight = output.shape[3]
  let outputWidth = output.shape[4]
  let inChannels = input.shape[1]
  var maxDiff: Float = 0
  for od in 0..<outputDepth {
    for oh in 0..<outputHeight {
      for ow in 0..<outputWidth {
        for oc in 0..<outChannels {
          var sum = bias[oc]
          for ic in 0..<inChannels {
            for kd in 0..<3 {
              let id = od + kd
              for kh in 0..<3 {
                let ih = oh + kh - 1
                if ih < 0 || ih >= outputHeight {
                  continue
                }
                for kw in 0..<3 {
                  let iw = ow + kw - 1
                  if iw < 0 || iw >= outputWidth {
                    continue
                  }
                  sum += weight[oc, ic, kd, kh, kw] * input[0, ic, id, ih, iw]
                }
              }
            }
          }
          let diff = abs(sum - output[0, oc, od, oh, ow])
          if diff > maxDiff {
            maxDiff = diff
          }
        }
      }
    }
  }
  return maxDiff
}

func maxRelativeDiff5D(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>)
  -> Float
{
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  precondition(swiftTensor.shape[4] == torchTensor.shape[4])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let ref = Float(torchTensor[i, c, d, h, w])
            let denom = max(abs(ref), 1e-6)
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - ref) / denom
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
  }
  return maxDiff
}

func maxRelativeDiff5D(_ swiftTensor: Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  precondition(swiftTensor.shape[4] == torchTensor.shape[4])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let ref = Float(torchTensor[i, c, d, h, w])
            let denom = max(abs(ref), 1e-6)
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - ref) / denom
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
  }
  return maxDiff
}

func maxRelativeDiffDetails5D(
  _ swiftTensor: Tensor<Float>, _ torchTensor: Tensor<Float>
) -> (Float, Float, Float, Float, [Int]) {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  var maxDiff: Float = 0
  var maxAbs: Float = 0
  var maxRef: Float = 0
  var maxSwift: Float = 0
  var maxIndex = [0, 0, 0, 0, 0]
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let ref = Float(torchTensor[i, c, d, h, w])
            let swift = Float(swiftTensor[i, c, d, h, w])
            let absDiff = abs(swift - ref)
            let denom = max(abs(ref), 1e-6)
            let diff = absDiff / denom
            if diff > maxDiff {
              maxDiff = diff
              maxAbs = absDiff
              maxRef = ref
              maxSwift = swift
              maxIndex = [i, c, d, h, w]
            }
          }
        }
      }
    }
  }
  return (maxDiff, maxAbs, maxRef, maxSwift, maxIndex)
}

func temporalSliceMaxAbsDiff5D(
  _ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>
) -> [Float] {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  var diffs = [Float](repeating: 0, count: torchTensor.shape[2])
  for d in 0..<torchTensor.shape[2] {
    var maxDiff: Float = 0
    for i in 0..<torchTensor.shape[0] {
      for c in 0..<torchTensor.shape[1] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - Float(torchTensor[i, c, d, h, w]))
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
    diffs[d] = maxDiff
  }
  return diffs
}

func temporalSliceMaxAbsDiff5D(
  _ swiftTensor: Tensor<Float>, _ torchTensor: Tensor<Float>
) -> [Float] {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  var diffs = [Float](repeating: 0, count: torchTensor.shape[2])
  for d in 0..<torchTensor.shape[2] {
    var maxDiff: Float = 0
    for i in 0..<torchTensor.shape[0] {
      for c in 0..<torchTensor.shape[1] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - Float(torchTensor[i, c, d, h, w]))
            if diff > maxDiff {
              maxDiff = diff
            }
          }
        }
      }
    }
    diffs[d] = maxDiff
  }
  return diffs
}

func borderVsInteriorMaxAbsDiff5D(
  _ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>
) -> (Float, Float) {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  var borderMax: Float = 0
  var interiorMax: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - Float(torchTensor[i, c, d, h, w]))
            if h == 0 || h == torchTensor.shape[3] - 1 || w == 0 || w == torchTensor.shape[4] - 1 {
              if diff > borderMax {
                borderMax = diff
              }
            } else if diff > interiorMax {
              interiorMax = diff
            }
          }
        }
      }
    }
  }
  return (borderMax, interiorMax)
}

func borderVsInteriorMaxAbsDiff5D(
  _ swiftTensor: Tensor<Float>, _ torchTensor: Tensor<Float>
) -> (Float, Float) {
  precondition(swiftTensor.shape.count == 5)
  precondition(torchTensor.shape.count == 5)
  var borderMax: Float = 0
  var interiorMax: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for d in 0..<torchTensor.shape[2] {
        for h in 0..<torchTensor.shape[3] {
          for w in 0..<torchTensor.shape[4] {
            let diff = abs(Float(swiftTensor[i, c, d, h, w]) - Float(torchTensor[i, c, d, h, w]))
            if h == 0 || h == torchTensor.shape[3] - 1 || w == 0 || w == torchTensor.shape[4] - 1 {
              if diff > borderMax {
                borderMax = diff
              }
            } else if diff > interiorMax {
              interiorMax = diff
            }
          }
        }
      }
    }
  }
  return (borderMax, interiorMax)
}

func maxAbsDiff4D(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float
{
  precondition(swiftTensor.shape.count == 4)
  precondition(torchTensor.shape.count == 4)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for h in 0..<torchTensor.shape[2] {
        for w in 0..<torchTensor.shape[3] {
          let diff = abs(Float(swiftTensor[i, c, h, w]) - Float(torchTensor[i, c, h, w]))
          if diff > maxDiff {
            maxDiff = diff
          }
        }
      }
    }
  }
  return maxDiff
}

func maxRelativeDiff4D(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>)
  -> Float
{
  precondition(swiftTensor.shape.count == 4)
  precondition(torchTensor.shape.count == 4)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  precondition(swiftTensor.shape[2] == torchTensor.shape[2])
  precondition(swiftTensor.shape[3] == torchTensor.shape[3])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for c in 0..<torchTensor.shape[1] {
      for h in 0..<torchTensor.shape[2] {
        for w in 0..<torchTensor.shape[3] {
          let ref = Float(torchTensor[i, c, h, w])
          let denom = max(abs(ref), 1e-6)
          let diff = abs(Float(swiftTensor[i, c, h, w]) - ref) / denom
          if diff > maxDiff {
            maxDiff = diff
          }
        }
      }
    }
  }
  return maxDiff
}

func reducedMaxAbsDiff4D(
  _ graph: DynamicGraph, _ swiftTensor: DynamicGraph.Tensor<Float>, _ reference: Tensor<Float>
) -> Float {
  let count = reference.shape.reduce(1, *)
  let referenceTensor = placeOnDevice(graph.variable(reference))
  let diff = Functional.abs(swiftTensor - referenceTensor).reshaped(.NC(1, count)).reduced(
    .max, axis: [1])
  return diff.toCPU()[0, 0]
}

func reducedApproxMaxRelativeDiff4D(
  _ graph: DynamicGraph, _ swiftTensor: DynamicGraph.Tensor<Float>, _ reference: Tensor<Float>
) -> Float {
  let count = reference.shape.reduce(1, *)
  let referenceTensor = placeOnDevice(graph.variable(reference))
  let denom = Functional.abs(referenceTensor) + 1e-6
  let diff = (Functional.abs(swiftTensor - referenceTensor) .* Functional.reciprocal(denom))
    .reshaped(.NC(1, count)).reduced(.max, axis: [1])
  return diff.toCPU()[0, 0]
}

func printReducedDiff5D(
  _ label: String, _ graph: DynamicGraph, _ swiftTensor: DynamicGraph.Tensor<Float>,
  _ reference: Tensor<Float>
) {
  print("\(label) max abs diff:", reducedMaxAbsDiff5D(graph, swiftTensor, reference))
  print("\(label) max rel diff:", reducedApproxMaxRelativeDiff5D(graph, swiftTensor, reference))
}

func printReducedDiff4D(
  _ label: String, _ graph: DynamicGraph, _ swiftTensor: DynamicGraph.Tensor<Float>,
  _ reference: Tensor<Float>
) {
  print("\(label) max abs diff:", reducedMaxAbsDiff4D(graph, swiftTensor, reference))
  print("\(label) max rel diff:", reducedApproxMaxRelativeDiff4D(graph, swiftTensor, reference))
}

func maxAbsDiff2D(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>) -> Float
{
  precondition(swiftTensor.shape.count == 2)
  precondition(torchTensor.shape.count == 2)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for j in 0..<torchTensor.shape[1] {
      let diff = abs(Float(swiftTensor[i, j]) - Float(torchTensor[i, j]))
      if diff > maxDiff {
        maxDiff = diff
      }
    }
  }
  return maxDiff
}

func maxAbsDiff2DTensor(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var maxDiff: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let diff = abs(Float(lhs[i, j]) - Float(rhs[i, j]))
      if diff > maxDiff {
        maxDiff = diff
      }
    }
  }
  return maxDiff
}

func maxRelativeDiff2DTensor(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var maxDiff: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let ref = Float(rhs[i, j])
      let denom = max(abs(ref), 1e-6)
      let diff = abs(Float(lhs[i, j]) - ref) / denom
      if diff > maxDiff {
        maxDiff = diff
      }
    }
  }
  return maxDiff
}

func maxGlobalRelativeDiff2DTensor(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var maxDiff: Float = 0
  var maxRef: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let ref = Float(rhs[i, j])
      let diff = abs(Float(lhs[i, j]) - ref)
      if diff > maxDiff {
        maxDiff = diff
      }
      let refAbs = abs(ref)
      if refAbs > maxRef {
        maxRef = refAbs
      }
    }
  }
  return maxDiff / max(maxRef, 1e-6)
}

func minMax2DTensor(_ x: Tensor<Float>) -> (Float, Float) {
  precondition(x.shape.count == 2)
  var minValue = Float.greatestFiniteMagnitude
  var maxValue = -Float.greatestFiniteMagnitude
  for i in 0..<x.shape[0] {
    for j in 0..<x.shape[1] {
      let value = Float(x[i, j])
      if value < minValue {
        minValue = value
      }
      if value > maxValue {
        maxValue = value
      }
    }
  }
  return (minValue, maxValue)
}

func maxAbsDiff2DTensorDetail(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> (
  diff: Float, row: Int, col: Int, lhs: Float, rhs: Float
) {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var bestDiff: Float = 0
  var bestRow = 0
  var bestCol = 0
  var bestLhs: Float = 0
  var bestRhs: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let lhsValue = Float(lhs[i, j])
      let rhsValue = Float(rhs[i, j])
      let diff = abs(lhsValue - rhsValue)
      if diff > bestDiff {
        bestDiff = diff
        bestRow = i
        bestCol = j
        bestLhs = lhsValue
        bestRhs = rhsValue
      }
    }
  }
  return (bestDiff, bestRow, bestCol, bestLhs, bestRhs)
}

func formatFloatValues(_ values: [Float]) -> String {
  values.map { String(format: "% .6g", $0) }.joined(separator: ", ")
}

func tensor2DRowString(_ x: Tensor<Float>, row: Int, cols: Int) -> String {
  let count = min(cols, x.shape[1])
  var values = [Float]()
  values.reserveCapacity(count)
  for col in 0..<count {
    values.append(Float(x[row, col]))
  }
  return formatFloatValues(values)
}

func tensor2DDiffRowString(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>, row: Int, cols: Int)
  -> String
{
  let count = min(cols, lhs.shape[1])
  var values = [Float]()
  values.reserveCapacity(count)
  for col in 0..<count {
    values.append(Float(lhs[row, col]) - Float(rhs[row, col]))
  }
  return formatFloatValues(values)
}

func printTensor2DEyeball(
  _ label: String, _ lhs: Tensor<Float>, _ rhs: Tensor<Float>, cols: Int = 8
) {
  let lhsMinMax = minMax2DTensor(lhs)
  let rhsMinMax = minMax2DTensor(rhs)
  let detail = maxAbsDiff2DTensorDetail(lhs, rhs)
  print(
    "\(label) eyeball stats:",
    "shape:", lhs.shape,
    "swift_min:", lhsMinMax.0,
    "swift_max:", lhsMinMax.1,
    "ref_min:", rhsMinMax.0,
    "ref_max:", rhsMinMax.1,
    "max_diff:", detail.diff,
    "max_diff_index:", [detail.row, detail.col],
    "swift_at_max:", detail.lhs,
    "ref_at_max:", detail.rhs)
  let rows = Array(Set([0, lhs.shape[0] / 2, detail.row])).sorted()
  for row in rows {
    print("\(label) eyeball row \(row) swift:", tensor2DRowString(lhs, row: row, cols: cols))
    print("\(label) eyeball row \(row) ref:", tensor2DRowString(rhs, row: row, cols: cols))
    print(
      "\(label) eyeball row \(row) diff:", tensor2DDiffRowString(lhs, rhs, row: row, cols: cols))
  }
}

func maxRelativeDiff2D(_ swiftTensor: DynamicGraph.Tensor<Float>, _ torchTensor: Tensor<Float>)
  -> Float
{
  precondition(swiftTensor.shape.count == 2)
  precondition(torchTensor.shape.count == 2)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  var maxDiff: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for j in 0..<torchTensor.shape[1] {
      let ref = Float(torchTensor[i, j])
      let denom = max(abs(ref), 1e-6)
      let diff = abs(Float(swiftTensor[i, j]) - ref) / denom
      if diff > maxDiff {
        maxDiff = diff
      }
    }
  }
  return maxDiff
}

func copyParameter(_ parameter: Model.Parameters, from tensor: Tensor<Float>, asBFloat16: Bool) {
  if asBFloat16 {
    parameter.copy(from: Tensor<BFloat16>(from: tensor))
  } else {
    parameter.copy(from: tensor)
  }
}

func seedVR2TimeEmbedding(timestep: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int)
  -> Tensor<
    Float
  >
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timestep
    let sinFreq = sin(freq)
    let cosFreq = cos(freq)
    for j in 0..<batchSize {
      embedding[j, i] = sinFreq
      embedding[j, i + half] = cosFreq
    }
  }
  return embedding
}

func seedVR2TemporalReplicatePad(
  _ x: Model.IO, batchSize: Int, channels: Int, depth: Int, height: Int, width: Int, left: Int
) -> Model.IO {
  if left == 0 {
    return x
  }
  return x.padded(.replicate, begin: [0, 0, left, 0, 0], end: [0, 0, 0, 0, 0])
}

func seedVR2FrameWiseGroupNorm(
  _ x: Model.IO, channels: Int, depth: Int, height: Int, width: Int, name: String
) -> (GroupNorm, Model.IO) {
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: name)
  let frameWise = x.permuted(0, 2, 1, 3, 4).copied().reshaped([depth, channels, height, width])
  let normalized = norm(frameWise)
  let restored = normalized.reshaped([1, depth, channels, height, width]).permuted(0, 2, 1, 3, 4)
    .copied()
  return (norm, restored)
}

func SeedVR2ResnetBlock3D(
  prefix: String, inChannels: Int, outChannels: Int, depth: Int, height: Int, width: Int,
  flattenOutput: Bool = false
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (norm1, norm1Out) = seedVR2FrameWiseGroupNorm(
    x, channels: inChannels, depth: depth, height: height, width: width, name: "norm1")
  var out = norm1Out
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv1")
  out = conv1(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: inChannels, depth: depth, height: height, width: width, left: 2))
  let (norm2, norm2Out) = seedVR2FrameWiseGroupNorm(
    out, channels: outChannels, depth: depth, height: height, width: width, name: "norm2")
  out = norm2Out
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv2")
  out = conv2(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: outChannels, depth: depth, height: height, width: width, left: 2)
  )
  let convShortcut: Convolution?
  if inChannels != outChannels {
    let shortcut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "conv_shortcut")
    out = shortcut(x) + out
    convShortcut = shortcut
  } else {
    out = x + out
    convShortcut = nil
  }
  let reader: (PythonObject) -> Void = { stateDict in
    let norm1Weight = stateDict["\(prefix).norm1.weight"].to(torch.float).cpu().numpy()
    let norm1Bias = stateDict["\(prefix).norm1.bias"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1Weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1Bias))
    let conv1Weight = stateDict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1Bias = stateDict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1Weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1Bias))
    let norm2Weight = stateDict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2Bias = stateDict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2Weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2Bias))
    let conv2Weight = stateDict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2Bias = stateDict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2Weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2Bias))
    if let convShortcut = convShortcut {
      let shortcutWeight = stateDict["\(prefix).conv_shortcut.weight"].to(torch.float).cpu().numpy()
      let shortcutBias = stateDict["\(prefix).conv_shortcut.bias"].to(torch.float).cpu().numpy()
      convShortcut.weight.copy(from: try! Tensor<Float>(numpy: shortcutWeight))
      convShortcut.bias.copy(from: try! Tensor<Float>(numpy: shortcutBias))
    }
  }
  let output: Model.IO
  if flattenOutput {
    output = out.reshaped(.NC(1, outChannels * depth * height * width))
  } else {
    output = out
  }
  return (reader, Model([x], [output.copied()]))
}

func SeedVR2ResnetBlock3DProbe(
  prefix: String, inChannels: Int, outChannels: Int, depth: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (norm1, norm1Out) = seedVR2FrameWiseGroupNorm(
    x, channels: inChannels, depth: depth, height: height, width: width, name: "norm1")
  let act1Out = norm1Out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv1")
  let conv1TemporalPadded = seedVR2TemporalReplicatePad(
    act1Out, batchSize: 1, channels: inChannels, depth: depth, height: height, width: width, left: 2
  )
  let conv1Out = conv1(conv1TemporalPadded)
  let (norm2, norm2Out) = seedVR2FrameWiseGroupNorm(
    conv1Out, channels: outChannels, depth: depth, height: height, width: width, name: "norm2")
  let act2Out = norm2Out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv2")
  let conv2TemporalPadded = seedVR2TemporalReplicatePad(
    act2Out, batchSize: 1, channels: outChannels, depth: depth, height: height, width: width,
    left: 2)
  let conv2Out = conv2(conv2TemporalPadded)
  let convShortcut: Convolution?
  let out: Model.IO
  if inChannels != outChannels {
    let shortcut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "conv_shortcut")
    out = shortcut(x) + conv2Out
    convShortcut = shortcut
  } else {
    out = x + conv2Out
    convShortcut = nil
  }
  let reader: (PythonObject) -> Void = { stateDict in
    let norm1Weight = stateDict["\(prefix).norm1.weight"].to(torch.float).cpu().numpy()
    let norm1Bias = stateDict["\(prefix).norm1.bias"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1Weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1Bias))
    let conv1Weight = stateDict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1Bias = stateDict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1Weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1Bias))
    let norm2Weight = stateDict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2Bias = stateDict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2Weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2Bias))
    let conv2Weight = stateDict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2Bias = stateDict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2Weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2Bias))
    if let convShortcut = convShortcut {
      let shortcutWeight = stateDict["\(prefix).conv_shortcut.weight"].to(torch.float).cpu().numpy()
      let shortcutBias = stateDict["\(prefix).conv_shortcut.bias"].to(torch.float).cpu().numpy()
      convShortcut.weight.copy(from: try! Tensor<Float>(numpy: shortcutWeight))
      convShortcut.bias.copy(from: try! Tensor<Float>(numpy: shortcutBias))
    }
  }
  return (
    reader,
    Model(
      [
        x
      ],
      [
        norm1Out.copied(),
        act1Out.copied(),
        conv1Out.copied(),
        norm2Out.copied(),
        act2Out.copied(),
        conv2Out.copied(),
        out.copied(),
      ])
  )
}

func SeedVR2ResnetBlock3DSingleProbe(
  prefix: String, inChannels: Int, outChannels: Int, depth: Int, height: Int, width: Int,
  outputKey: String, flattenOutput: Bool = false
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (norm1, norm1Out) = seedVR2FrameWiseGroupNorm(
    x, channels: inChannels, depth: depth, height: height, width: width, name: "norm1")
  let act1Out = norm1Out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv1")
  let conv1TemporalPadded = seedVR2TemporalReplicatePad(
    act1Out, batchSize: 1, channels: inChannels, depth: depth, height: height, width: width, left: 2
  )
  let conv1Out = conv1(conv1TemporalPadded)
  let (norm2, norm2Out) = seedVR2FrameWiseGroupNorm(
    conv1Out, channels: outChannels, depth: depth, height: height, width: width, name: "norm2")
  let act2Out = norm2Out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv2")
  let conv2TemporalPadded = seedVR2TemporalReplicatePad(
    act2Out, batchSize: 1, channels: outChannels, depth: depth, height: height, width: width,
    left: 2)
  let conv2Out = conv2(conv2TemporalPadded)
  let convShortcut: Convolution?
  let out: Model.IO
  if inChannels != outChannels {
    let shortcut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      name: "conv_shortcut")
    out = shortcut(x) + conv2Out
    convShortcut = shortcut
  } else {
    out = x + conv2Out
    convShortcut = nil
  }
  let output: Model.IO
  switch outputKey {
  case "norm1":
    output = norm1Out
  case "act1":
    output = act1Out
  case "conv1":
    output = conv1Out
  case "norm2":
    output = norm2Out
  case "act2":
    output = act2Out
  case "conv2":
    output = conv2Out
  case "output":
    output = out
  default:
    fatalError("Unsupported SeedVR2 resnet probe output: \(outputKey)")
  }
  let finalOutput: Model.IO
  if flattenOutput {
    finalOutput = output.reshaped(.NC(1, outChannels * depth * height * width))
  } else {
    finalOutput = output
  }
  let reader: (PythonObject) -> Void = { stateDict in
    let norm1Weight = stateDict["\(prefix).norm1.weight"].to(torch.float).cpu().numpy()
    let norm1Bias = stateDict["\(prefix).norm1.bias"].to(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1Weight))
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1Bias))
    let conv1Weight = stateDict["\(prefix).conv1.weight"].to(torch.float).cpu().numpy()
    let conv1Bias = stateDict["\(prefix).conv1.bias"].to(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1Weight))
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1Bias))
    let norm2Weight = stateDict["\(prefix).norm2.weight"].to(torch.float).cpu().numpy()
    let norm2Bias = stateDict["\(prefix).norm2.bias"].to(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2Weight))
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2Bias))
    let conv2Weight = stateDict["\(prefix).conv2.weight"].to(torch.float).cpu().numpy()
    let conv2Bias = stateDict["\(prefix).conv2.bias"].to(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2Weight))
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2Bias))
    if let convShortcut = convShortcut {
      let shortcutWeight = stateDict["\(prefix).conv_shortcut.weight"].to(torch.float).cpu().numpy()
      let shortcutBias = stateDict["\(prefix).conv_shortcut.bias"].to(torch.float).cpu().numpy()
      convShortcut.weight.copy(from: try! Tensor<Float>(numpy: shortcutWeight))
      convShortcut.bias.copy(from: try! Tensor<Float>(numpy: shortcutBias))
    }
  }
  return (reader, Model([x], [finalOutput.copied()]))
}

func SeedVR2AttentionBlock2D(
  prefix: String, channels: Int, batchSize: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: "group_norm")
  var out = norm(x)
  let hw = height * width
  let toQueries = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), name: "to_q")
  let toKeys = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), name: "to_k")
  let toValues = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), name: "to_v")
  let q = ((1.0 / Float(channels).squareRoot()) * toQueries(out)).reshaped([
    batchSize, channels, hw,
  ])
  let k = toKeys(out).reshaped([batchSize, channels, hw])
  let v = toValues(out).reshaped([batchSize, channels, hw])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let toOut = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    name: "to_out")
  out = x + toOut(out.reshaped([batchSize, channels, height, width]))
  let reader: (PythonObject) -> Void = { stateDict in
    let normWeight = stateDict["\(prefix).group_norm.weight"].to(torch.float).cpu().numpy()
    let normBias = stateDict["\(prefix).group_norm.bias"].to(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: normWeight))
    norm.bias.copy(from: try! Tensor<Float>(numpy: normBias))
    let qWeight = stateDict["\(prefix).to_q.weight"].to(torch.float).cpu().numpy()
    let qBias = stateDict["\(prefix).to_q.bias"].to(torch.float).cpu().numpy()
    toQueries.weight.copy(from: try! Tensor<Float>(numpy: qWeight))
    toQueries.bias.copy(from: try! Tensor<Float>(numpy: qBias))
    let kWeight = stateDict["\(prefix).to_k.weight"].to(torch.float).cpu().numpy()
    let kBias = stateDict["\(prefix).to_k.bias"].to(torch.float).cpu().numpy()
    toKeys.weight.copy(from: try! Tensor<Float>(numpy: kWeight))
    toKeys.bias.copy(from: try! Tensor<Float>(numpy: kBias))
    let vWeight = stateDict["\(prefix).to_v.weight"].to(torch.float).cpu().numpy()
    let vBias = stateDict["\(prefix).to_v.bias"].to(torch.float).cpu().numpy()
    toValues.weight.copy(from: try! Tensor<Float>(numpy: vWeight))
    toValues.bias.copy(from: try! Tensor<Float>(numpy: vBias))
    let outWeight = stateDict["\(prefix).to_out.0.weight"].to(torch.float).cpu().numpy()
    let outBias = stateDict["\(prefix).to_out.0.bias"].to(torch.float).cpu().numpy()
    toOut.weight.copy(from: try! Tensor<Float>(numpy: outWeight))
    toOut.bias.copy(from: try! Tensor<Float>(numpy: outBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DecoderMidBlock3D(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (resnet0Reader, resnet0) = SeedVR2ResnetBlock3D(
    prefix: "decoder.mid_block.resnets.0", inChannels: 512, outChannels: 512, depth: depth,
    height: height, width: width)
  var out = resnet0(x)
  let (attnReader, attn) = SeedVR2AttentionBlock2D(
    prefix: "decoder.mid_block.attentions.0", channels: 512, batchSize: depth, height: height,
    width: width)
  let frameWise = out.permuted(0, 2, 1, 3, 4).contiguous().reshaped([depth, 512, height, width])
  out = attn(frameWise).reshaped([1, depth, 512, height, width]).permuted(0, 2, 1, 3, 4)
    .contiguous()
  let (resnet1Reader, resnet1) = SeedVR2ResnetBlock3D(
    prefix: "decoder.mid_block.resnets.1", inChannels: 512, outChannels: 512, depth: depth,
    height: height, width: width)
  out = resnet1(out)
  let reader: (PythonObject) -> Void = { stateDict in
    resnet0Reader(stateDict)
    attnReader(stateDict)
    resnet1Reader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Upsample3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int, temporalUp: Bool,
  spatialUp: Bool
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let temporalRatio = temporalUp ? 2 : 1
  let spatialRatio = spatialUp ? 2 : 1
  let upscaleRatio = temporalRatio * spatialRatio * spatialRatio
  let upscaleConv = Convolution(
    groups: 1, filters: channels * upscaleRatio, filterSize: [1, 1, 1],
    hint: Hint(stride: [1, 1, 1]),
    name: "upscale_conv")
  var out = upscaleConv(x)
  out = out.reshaped([1, spatialRatio, spatialRatio, temporalRatio, channels, depth, height, width])
    .permuted(0, 4, 5, 3, 6, 1, 7, 2).contiguous()
  let upDepthRaw = depth * temporalRatio
  let upHeight = height * spatialRatio
  let upWidth = width * spatialRatio
  out = out.reshaped([1, channels, upDepthRaw, upHeight, upWidth])
  if temporalUp {
    let first = out.reshaped(
      [1, channels, 1, upHeight, upWidth],
      strides: [
        channels * upDepthRaw * upHeight * upWidth, upDepthRaw * upHeight * upWidth,
        upHeight * upWidth, upWidth, 1,
      ]
    ).contiguous()
    let rest = out.reshaped(
      [1, channels, upDepthRaw - 2, upHeight, upWidth],
      offset: [0, 0, 2, 0, 0],
      strides: [
        channels * upDepthRaw * upHeight * upWidth, upDepthRaw * upHeight * upWidth,
        upHeight * upWidth, upWidth, 1,
      ]
    ).contiguous()
    out = Functional.concat(axis: 2, first, rest)
  }
  let upDepth = temporalUp ? 1 + (depth - 1) * temporalRatio : upDepthRaw
  let conv = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv")
  out = conv(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: channels, depth: upDepth, height: upHeight, width: upWidth,
      left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let upscaleWeight = stateDict["\(prefix).upscale_conv.weight"].to(torch.float).cpu().numpy()
    let upscaleBias = stateDict["\(prefix).upscale_conv.bias"].to(torch.float).cpu().numpy()
    upscaleConv.weight.copy(from: try! Tensor<Float>(numpy: upscaleWeight))
    upscaleConv.bias.copy(from: try! Tensor<Float>(numpy: upscaleBias))
    let convWeight = stateDict["\(prefix).conv.weight"].to(torch.float).cpu().numpy()
    let convBias = stateDict["\(prefix).conv.bias"].to(torch.float).cpu().numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: convWeight))
    conv.bias.copy(from: try! Tensor<Float>(numpy: convBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Upsample3DProbe(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int, temporalUp: Bool,
  spatialUp: Bool
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let temporalRatio = temporalUp ? 2 : 1
  let spatialRatio = spatialUp ? 2 : 1
  let upscaleRatio = temporalRatio * spatialRatio * spatialRatio
  let upscaleConv = Convolution(
    groups: 1, filters: channels * upscaleRatio, filterSize: [1, 1, 1],
    hint: Hint(stride: [1, 1, 1]),
    name: "upscale_conv")
  let afterUpscale = upscaleConv(x)
  let afterRearrange = afterUpscale.reshaped(
    [1, spatialRatio, spatialRatio, temporalRatio, channels, depth, height, width]
  ).permuted(0, 4, 5, 3, 6, 1, 7, 2).contiguous().reshaped(
    [1, channels, depth * temporalRatio, height * spatialRatio, width * spatialRatio])
  let afterRemoveHead: Model.IO
  let upDepth: Int
  if temporalUp {
    let upDepthRaw = depth * temporalRatio
    let first = afterRearrange.reshaped(
      [1, channels, 1, height * spatialRatio, width * spatialRatio],
      strides: [
        channels * upDepthRaw * height * spatialRatio * width * spatialRatio,
        upDepthRaw * height * spatialRatio * width * spatialRatio,
        height * spatialRatio * width * spatialRatio, width * spatialRatio, 1,
      ]
    ).contiguous()
    let rest = afterRearrange.reshaped(
      [1, channels, upDepthRaw - 2, height * spatialRatio, width * spatialRatio],
      offset: [0, 0, 2, 0, 0],
      strides: [
        channels * upDepthRaw * height * spatialRatio * width * spatialRatio,
        upDepthRaw * height * spatialRatio * width * spatialRatio,
        height * spatialRatio * width * spatialRatio, width * spatialRatio, 1,
      ]
    ).contiguous()
    afterRemoveHead = Functional.concat(axis: 2, first, rest)
    upDepth = 1 + (depth - 1) * temporalRatio
  } else {
    afterRemoveHead = afterRearrange
    upDepth = depth
  }
  let conv = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv")
  let out = conv(
    seedVR2TemporalReplicatePad(
      afterRemoveHead, batchSize: 1, channels: channels, depth: upDepth,
      height: height * spatialRatio, width: width * spatialRatio, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let upscaleWeight = stateDict["\(prefix).upscale_conv.weight"].to(torch.float).cpu().numpy()
    let upscaleBias = stateDict["\(prefix).upscale_conv.bias"].to(torch.float).cpu().numpy()
    upscaleConv.weight.copy(from: try! Tensor<Float>(numpy: upscaleWeight))
    upscaleConv.bias.copy(from: try! Tensor<Float>(numpy: upscaleBias))
    let convWeight = stateDict["\(prefix).conv.weight"].to(torch.float).cpu().numpy()
    let convBias = stateDict["\(prefix).conv.bias"].to(torch.float).cpu().numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: convWeight))
    conv.bias.copy(from: try! Tensor<Float>(numpy: convBias))
  }
  return (reader, Model([x], [afterUpscale, afterRearrange, afterRemoveHead, out]))
}

func SeedVR2DiTTextIn() -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let txtIn = Dense(count: 2560, name: "txt_in")
  let out = txtIn(x)
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["txt_in.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["txt_in.bias"].to(torch.float).cpu().numpy()
    txtIn.weight.copy(from: try! Tensor<Float>(numpy: weight))
    txtIn.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DiTTimeEmbedding() -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let projIn = Dense(count: 2560, name: "proj_in")
  let projHid = Dense(count: 2560, name: "proj_hid")
  let projOut = Dense(count: 15360, name: "proj_out")
  var out = projIn(x).swish()
  out = projHid(out).swish()
  out = projOut(out)
  let reader: (PythonObject) -> Void = { stateDict in
    let projInWeight = stateDict["emb_in.proj_in.weight"].to(torch.float).cpu().numpy()
    let projInBias = stateDict["emb_in.proj_in.bias"].to(torch.float).cpu().numpy()
    projIn.weight.copy(from: try! Tensor<Float>(numpy: projInWeight))
    projIn.bias.copy(from: try! Tensor<Float>(numpy: projInBias))
    let projHidWeight = stateDict["emb_in.proj_hid.weight"].to(torch.float).cpu().numpy()
    let projHidBias = stateDict["emb_in.proj_hid.bias"].to(torch.float).cpu().numpy()
    projHid.weight.copy(from: try! Tensor<Float>(numpy: projHidWeight))
    projHid.bias.copy(from: try! Tensor<Float>(numpy: projHidBias))
    let projOutWeight = stateDict["emb_in.proj_out.weight"].to(torch.float).cpu().numpy()
    let projOutBias = stateDict["emb_in.proj_out.bias"].to(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: projOutWeight))
    projOut.bias.copy(from: try! Tensor<Float>(numpy: projOutBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DiTPatchIn(frames: Int, latentHeight: Int, latentWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let proj = Dense(count: 2560, name: "proj")
  var out = x.reshaped([frames, latentHeight, latentWidth, 33], format: .NHWC)
  out = out.reshaped(
    [frames, latentHeight / 2, 2, latentWidth / 2, 2, 33], format: .NHWC
  ).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    frames * (latentHeight / 2) * (latentWidth / 2), 132,
  ])
  out = proj(out)
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["vid_in.proj.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["vid_in.proj.bias"].to(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DiTOutputHead(frames: Int, latentHeight: Int, latentWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let mod = Input()
  let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
  let proj = Dense(count: 64, name: "proj")
  let patchHeight = latentHeight / 2
  let patchWidth = latentWidth / 2
  let afterNorm = norm(x)
  let afterAda =
    afterNorm .* seedVR2ModRow(mod, index: 1, width: 2560)
    + seedVR2ModRow(mod, index: 0, width: 2560)
  var out = proj(afterAda).reshaped([frames, patchHeight, patchWidth, 2, 2, 16], format: .NHWC)
  out = out.permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    frames * latentHeight * latentWidth, 16,
  ])
  let reader: (PythonObject) -> Void = { stateDict in
    let normWeight = stateDict["vid_out_norm.weight"].to(torch.float).cpu().numpy()
    norm.weight.copy(from: try! Tensor<Float>(numpy: normWeight))
    let weight = stateDict["vid_out.proj.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["vid_out.proj.bias"].to(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x, mod], [afterNorm.copied(), afterAda.copied(), out.copied()]))
}

func seedVR2ModRow(_ x: Model.IO, index: Int, width: Int) -> Model.IO {
  x.reshaped([1, width], offset: [index, 0], strides: [width, 1]).contiguous()
}

func seedVR2FlattenHeads(_ x: Model.IO, seqLen: Int, heads: Int, headDim: Int) -> Model.IO {
  x.contiguous().reshaped([seqLen, heads * headDim])
}

func seedVR2ApplyRotary3D(
  _ x: Model.IO, freqs: Model.IO, seqLen: Int, heads: Int, headDim: Int = 128, rotDim: Int = 126
) -> Model.IO {
  let xHeadFirst = x.permuted(1, 0, 2).contiguous().copied()
  let xRot = xHeadFirst.reshaped(
    [heads, seqLen, rotDim / 2, 2], offset: [0, 0, 0],
    strides: [seqLen * headDim, headDim, 2, 1]
  ).contiguous()
  let xPass = xHeadFirst.reshaped(
    [heads, seqLen, headDim - rotDim], offset: [0, 0, rotDim],
    strides: [seqLen * headDim, headDim, 1]
  ).contiguous()
  let angles = freqs.reshaped(
    [seqLen, rotDim / 2], offset: [0, 0, 0],
    strides: [rotDim, 2, 1]
  ).contiguous()
  let rot = Functional.concat(
    axis: 3,
    angles.cos().reshaped([1, seqLen, rotDim / 2, 1]),
    angles.sin().reshaped([1, seqLen, rotDim / 2, 1])
  ).contiguous()
  let rotated = Functional.cmul(left: xRot, right: rot).reshaped([heads, seqLen, rotDim])
    .contiguous()
  return Functional.concat(axis: 2, rotated, xPass).permuted(1, 0, 2).contiguous()
}

func seedVR2BlockModTensor(
  emb: Tensor<Float>, stateDict: PythonObject, blockIndex: Int, branch: String, layer: String
) -> Tensor<Float> {
  let layerOffset = layer == "attn" ? 0 : 3
  let branchPrefix: String
  if Bool(stateDict.__contains__("blocks.\(blockIndex).ada.\(branch).\(layer)_shift")) ?? false {
    branchPrefix = "blocks.\(blockIndex).ada.\(branch)"
  } else {
    branchPrefix = "blocks.\(blockIndex).ada.all"
  }
  let shift = try! Tensor<Float>(
    numpy: stateDict["\(branchPrefix).\(layer)_shift"].to(torch.float).cpu().numpy())
  let scale = try! Tensor<Float>(
    numpy: stateDict["\(branchPrefix).\(layer)_scale"].to(torch.float).cpu().numpy())
  let gate = try! Tensor<Float>(
    numpy: stateDict["\(branchPrefix).\(layer)_gate"].to(torch.float).cpu().numpy())
  var mod = Tensor<Float>(.CPU, .NC(3, 2560))
  for i in 0..<2560 {
    mod[0, i] = Float(emb[0, i * 6 + layerOffset]) + shift[i]
    mod[1, i] = Float(emb[0, i * 6 + layerOffset + 1]) + scale[i]
    mod[2, i] = Float(emb[0, i * 6 + layerOffset + 2]) + gate[i]
  }
  return mod
}

func seedVR2BlockModTensor(
  emb: DynamicGraph.Tensor<Float>, stateDict: PythonObject, blockIndex: Int, branch: String,
  layer: String
) -> Tensor<Float> {
  seedVR2BlockModTensor(
    emb: copiedToCPU(emb), stateDict: stateDict, blockIndex: blockIndex, branch: branch,
    layer: layer)
}

func seedVR2OutputModTensor(emb: Tensor<Float>, stateDict: PythonObject, offset: Int) -> Tensor<
  Float
> {
  let shift = try! Tensor<Float>(
    numpy: stateDict["vid_out_ada.out_shift"].to(torch.float).cpu().numpy())
  let scale = try! Tensor<Float>(
    numpy: stateDict["vid_out_ada.out_scale"].to(torch.float).cpu().numpy())
  var mod = Tensor<Float>(.CPU, .NC(2, 2560))
  for i in 0..<2560 {
    mod[0, i] = Float(emb[0, i * 6 + offset]) + shift[i]
    mod[1, i] = Float(emb[0, i * 6 + offset + 1]) + scale[i]
  }
  return mod
}

func SeedVR2RotaryProbe(seqLen: Int) -> Model {
  let q = Input()
  let k = Input()
  let freqs = Input()
  let qOut = seedVR2ApplyRotary3D(q, freqs: freqs, seqLen: seqLen, heads: 20)
  let kOut = seedVR2ApplyRotary3D(k, freqs: freqs, seqLen: seqLen, heads: 20)
  return Model([q, k, freqs], [qOut, kOut])
}

func seedVR2FullAttentionSDPA(
  vidQ: Model.IO, vidK: Model.IO, vidV: Model.IO,
  txtQ: Model.IO, txtK: Model.IO, txtV: Model.IO,
  vidLen: Int, txtLen: Int
) -> (Model.IO, Model.IO) {
  let totalLen = vidLen + txtLen
  let q = Functional.concat(axis: 0, vidQ.copied(), txtQ.copied()).reshaped(
    [1, totalLen, 20, 128], format: .NHWC
  ).to(.BFloat16)
    .contiguous()
  let k = Functional.concat(axis: 0, vidK.copied(), txtK.copied()).reshaped(
    [1, totalLen, 20, 128], format: .NHWC
  ).to(.BFloat16)
    .contiguous()
  let v = Functional.concat(axis: 0, vidV.copied(), txtV.copied()).reshaped(
    [1, totalLen, 20, 128], format: .NHWC
  ).to(.BFloat16)
    .contiguous()
  let out = ScaledDotProductAttention(scale: 1.0 / Float(128).squareRoot(), flags: [.Float16])(
    q, k, v
  ).to(
    .Float32
  ).reshaped([totalLen, 20, 128], format: .NHWC).contiguous()
  let vidOut = out.reshaped(
    [vidLen, 20, 128], offset: [0, 0, 0], strides: [20 * 128, 128, 1]
  ).contiguous().copied()
  let txtOut = out.reshaped(
    [txtLen, 20, 128], offset: [vidLen, 0, 0], strides: [20 * 128, 128, 1]
  ).contiguous().copied()
  return (vidOut, txtOut)
}

func SeedVR2DiTAttnProjectOnly(
  vidLen: Int, txtLen: Int, bfloat16Weights: Bool = false
) -> ((PythonObject, Int) -> Void, Model) {
  let vid = Input()
  let txt = Input()
  let vidAttnMod = Input()
  let txtAttnMod = Input()
  let ropeVidQ = Input()
  let ropeVidK = Input()
  let vidV = Input()
  let ropeTxtQ = Input()
  let ropeTxtK = Input()
  let txtV = Input()

  let (vidAttn3D, txtAttn3D) = seedVR2FullAttentionSDPA(
    vidQ: ropeVidQ, vidK: ropeVidK, vidV: vidV,
    txtQ: ropeTxtQ, txtK: ropeTxtK, txtV: txtV,
    vidLen: vidLen, txtLen: txtLen)
  let vidAttnFlat = seedVR2FlattenHeads(vidAttn3D, seqLen: vidLen, heads: 20, headDim: 128).copied()
  let txtAttnFlat = seedVR2FlattenHeads(txtAttn3D, seqLen: txtLen, heads: 20, headDim: 128).copied()
  let vidAttnOut = Dense(count: 2560, name: "vid_attn_out")
  let txtAttnOut = Dense(count: 2560, name: "txt_attn_out")
  let vidAttnProjected = vidAttnOut(vidAttnFlat).copied()
  let txtAttnProjected = txtAttnOut(txtAttnFlat).copied()
  let vidAfterAttn = vid + vidAttnProjected .* seedVR2ModRow(vidAttnMod, index: 2, width: 2560)
  let txtAfterAttn = txt + txtAttnProjected .* seedVR2ModRow(txtAttnMod, index: 2, width: 2560)

  let reader: (PythonObject, Int) -> Void = { stateDict, layerIndex in
    let prefix = "blocks.\(layerIndex)"
    let sharedWeights =
      (Bool(stateDict.__contains__("\(prefix).attn.proj_out.vid.weight")) ?? false) == false
    if sharedWeights {
      copyParameter(
        vidAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.all.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.all.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.all.bias"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.all.bias"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    } else {
      copyParameter(
        vidAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.vid.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.txt.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.vid.bias"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.txt.bias"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    }
  }
  return (
    reader,
    Model(
      [vid, txt, vidAttnMod, txtAttnMod, ropeVidQ, ropeVidK, vidV, ropeTxtQ, ropeTxtK, txtV],
      [vidAfterAttn, txtAfterAttn, vidAttnProjected, txtAttnProjected, vidAttn3D, txtAttn3D])
  )
}

func SeedVR2DiTAttnProjectOnlyLastLayer(
  vidLen: Int, txtLen: Int, bfloat16Weights: Bool = false
) -> ((PythonObject, Int) -> Void, Model) {
  let vid = Input()
  let txt = Input()
  let vidAttnMod = Input()
  let ropeVidQ = Input()
  let ropeVidK = Input()
  let vidV = Input()
  let ropeTxtQ = Input()
  let ropeTxtK = Input()
  let txtV = Input()

  let (vidAttn3D, txtAttn3D) = seedVR2FullAttentionSDPA(
    vidQ: ropeVidQ, vidK: ropeVidK, vidV: vidV,
    txtQ: ropeTxtQ, txtK: ropeTxtK, txtV: txtV,
    vidLen: vidLen, txtLen: txtLen)
  let vidAttnFlat = seedVR2FlattenHeads(vidAttn3D, seqLen: vidLen, heads: 20, headDim: 128).copied()
  let txtAttnFlat = seedVR2FlattenHeads(txtAttn3D, seqLen: txtLen, heads: 20, headDim: 128).copied()
  let vidAttnOut = Dense(count: 2560, name: "vid_attn_out")
  let txtAttnOut = Dense(count: 2560, name: "txt_attn_out")
  let vidAttnProjected = vidAttnOut(vidAttnFlat).copied()
  let txtAttnProjected = txtAttnOut(txtAttnFlat).copied()
  let vidAfterAttn = vid + vidAttnProjected .* seedVR2ModRow(vidAttnMod, index: 2, width: 2560)
  let txtAfterAttn = txt + txtAttnProjected

  let reader: (PythonObject, Int) -> Void = { stateDict, layerIndex in
    let prefix = "blocks.\(layerIndex)"
    let sharedWeights =
      (Bool(stateDict.__contains__("\(prefix).attn.proj_out.vid.weight")) ?? false) == false
    if sharedWeights {
      copyParameter(
        vidAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.all.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.all.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.all.bias"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.all.bias"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    } else {
      copyParameter(
        vidAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.vid.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtAttnOut.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.txt.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.vid.bias"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtAttnOut.bias,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.proj_out.txt.bias"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    }
  }
  return (
    reader,
    Model(
      [vid, txt, vidAttnMod, ropeVidQ, ropeVidK, vidV, ropeTxtQ, ropeTxtK, txtV],
      [vidAfterAttn, txtAfterAttn, vidAttnProjected, txtAttnProjected, vidAttn3D, txtAttn3D])
  )
}

func SeedVR2DiTMlpOnly(
  vidLen: Int, txtLen: Int, bfloat16Weights: Bool = false
) -> ((PythonObject, Int) -> Void, Model) {
  let vidAfterAttn = Input()
  let txtAfterAttn = Input()
  let vidMlpMod = Input()
  let txtMlpMod = Input()

  let vidMlpNorm = RMSNorm(epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "vid_mlp_norm")
  let txtMlpNorm = RMSNorm(epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "txt_mlp_norm")
  let vidMlpIn =
    vidMlpNorm(vidAfterAttn) .* seedVR2ModRow(vidMlpMod, index: 1, width: 2560)
    + seedVR2ModRow(vidMlpMod, index: 0, width: 2560)
  let txtMlpIn =
    txtMlpNorm(txtAfterAttn) .* seedVR2ModRow(txtMlpMod, index: 1, width: 2560)
    + seedVR2ModRow(txtMlpMod, index: 0, width: 2560)
  let vidMlpGate = Dense(count: 6912, noBias: true, name: "vid_mlp_gate")
  let vidMlpInProj = Dense(count: 6912, noBias: true, name: "vid_mlp_in")
  let vidMlpOutProj = Dense(count: 2560, noBias: true, name: "vid_mlp_out")
  let txtMlpGate = Dense(count: 6912, noBias: true, name: "txt_mlp_gate")
  let txtMlpInProj = Dense(count: 6912, noBias: true, name: "txt_mlp_in")
  let txtMlpOutProj = Dense(count: 2560, noBias: true, name: "txt_mlp_out")

  let vidGate = vidMlpGate(vidMlpIn.copied()).copied()
  let vidInner = vidMlpInProj(vidMlpIn.copied()).copied()
  let vidMlpOut = vidMlpOutProj(
    ((vidGate.to(.Float32) .* vidGate.to(.Float32).sigmoid()) .* vidInner.to(.Float32)).copied()
  ).copied()
  let txtGate = txtMlpGate(txtMlpIn.copied()).copied()
  let txtInner = txtMlpInProj(txtMlpIn.copied()).copied()
  let txtMlpOut = txtMlpOutProj(
    ((txtGate.to(.Float32) .* txtGate.to(.Float32).sigmoid()) .* txtInner.to(.Float32)).copied()
  ).copied()
  let vidFinal = (vidAfterAttn + vidMlpOut .* seedVR2ModRow(vidMlpMod, index: 2, width: 2560)).to(
    of: vidAfterAttn)
  let txtFinal = (txtAfterAttn + txtMlpOut .* seedVR2ModRow(txtMlpMod, index: 2, width: 2560)).to(
    of: txtAfterAttn)

  let reader: (PythonObject, Int) -> Void = { stateDict, layerIndex in
    let prefix = "blocks.\(layerIndex)"
    let sharedWeights =
      (Bool(stateDict.__contains__("\(prefix).mlp.vid.proj_in_gate.weight")) ?? false) == false
    if sharedWeights {
      copyParameter(
        vidMlpGate.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_in_gate.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtMlpGate.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_in_gate.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidMlpInProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_in.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtMlpInProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_in.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidMlpOutProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_out.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtMlpOutProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_out.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    } else {
      copyParameter(
        vidMlpGate.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.vid.proj_in_gate.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtMlpGate.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.txt.proj_in_gate.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidMlpInProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.vid.proj_in.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtMlpInProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.txt.proj_in.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidMlpOutProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.vid.proj_out.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtMlpOutProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.txt.proj_out.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    }
  }
  return (
    reader,
    Model(
      [vidAfterAttn, txtAfterAttn, vidMlpMod, txtMlpMod],
      [vidFinal, txtFinal, vidMlpIn, txtMlpIn, vidMlpOut, txtMlpOut])
  )
}

func SeedVR2DiTMlpOnlyLastLayer(
  vidLen: Int, txtLen: Int, bfloat16Weights: Bool = false
) -> ((PythonObject, Int) -> Void, Model) {
  let vidAfterAttn = Input()
  let txtAfterAttn = Input()
  let vidMlpMod = Input()

  let vidMlpNorm = RMSNorm(epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "vid_mlp_norm")
  let vidMlpIn =
    vidMlpNorm(vidAfterAttn) .* seedVR2ModRow(vidMlpMod, index: 1, width: 2560)
    + seedVR2ModRow(vidMlpMod, index: 0, width: 2560)
  let vidMlpGate = Dense(count: 6912, noBias: true, name: "vid_mlp_gate")
  let vidMlpInProj = Dense(count: 6912, noBias: true, name: "vid_mlp_in")
  let vidMlpOutProj = Dense(count: 2560, noBias: true, flags: [.Float32], name: "vid_mlp_out")
  let vidGate = vidMlpGate(vidMlpIn.copied()).copied()
  let vidInner = vidMlpInProj(vidMlpIn.copied()).copied()
  let vidMlpOut = vidMlpOutProj(
    ((vidGate.to(.Float32) .* vidGate.to(.Float32).sigmoid()) .* vidInner.to(.Float32)).copied()
  ).copied()
  let vidFinal = (vidAfterAttn + vidMlpOut .* seedVR2ModRow(vidMlpMod, index: 2, width: 2560)).to(
    of: vidAfterAttn)
  let txtFinal = (txtAfterAttn + txtAfterAttn).to(of: txtAfterAttn)

  let reader: (PythonObject, Int) -> Void = { stateDict, layerIndex in
    let prefix = "blocks.\(layerIndex)"
    let sharedWeights =
      (Bool(stateDict.__contains__("\(prefix).mlp.vid.proj_in_gate.weight")) ?? false) == false
    if sharedWeights {
      copyParameter(
        vidMlpGate.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_in_gate.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidMlpInProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_in.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidMlpOutProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.all.proj_out.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    } else {
      copyParameter(
        vidMlpGate.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.vid.proj_in_gate.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidMlpInProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.vid.proj_in.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidMlpOutProj.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).mlp.vid.proj_out.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    }
  }
  return (
    reader,
    Model(
      [vidAfterAttn, txtAfterAttn, vidMlpMod],
      [vidFinal, txtFinal, vidMlpIn, vidMlpOut])
  )
}

func SeedVR2DiTAttnInProbe(vidLen: Int, txtLen: Int) -> Model {
  let vid = Input()
  let txt = Input()
  let vidAttnMod = Input()
  let txtAttnMod = Input()
  let vidAttnNorm = RMSNorm(
    epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "vid_attn_norm")
  let txtAttnNorm = RMSNorm(
    epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "txt_attn_norm")
  let vidAttnIn =
    vidAttnNorm(vid) .* seedVR2ModRow(vidAttnMod, index: 1, width: 2560)
    + seedVR2ModRow(vidAttnMod, index: 0, width: 2560)
  let txtAttnIn =
    txtAttnNorm(txt) .* seedVR2ModRow(txtAttnMod, index: 1, width: 2560)
    + seedVR2ModRow(txtAttnMod, index: 0, width: 2560)
  return Model([vid, txt, vidAttnMod, txtAttnMod], [vidAttnIn, txtAttnIn])
}

func SeedVR2DiTAttnInProbeLastLayer(vidLen: Int, txtLen: Int) -> Model {
  let vid = Input()
  let txt = Input()
  let vidAttnMod = Input()
  let vidAttnNorm = RMSNorm(
    epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "vid_attn_norm")
  let txtAttnNorm = RMSNorm(
    epsilon: 1e-5, axis: [1], elementwiseAffine: false, name: "txt_attn_norm")
  let vidAttnIn =
    vidAttnNorm(vid) .* seedVR2ModRow(vidAttnMod, index: 1, width: 2560)
    + seedVR2ModRow(vidAttnMod, index: 0, width: 2560)
  let txtAttnIn = txtAttnNorm(txt)
  return Model([vid, txt, vidAttnMod], [vidAttnIn, txtAttnIn])
}

func SeedVR2DiTQKVFromAttnInputsProbe(
  vidLen: Int, txtLen: Int, bfloat16Weights: Bool = false
) -> ((PythonObject, Int) -> Void, Model) {
  let vidAttnIn = Input()
  let txtAttnIn = Input()
  let vidQProj = Dense(count: 2560, noBias: true, name: "qkv_from_attn_vid_q")
  let vidKProj = Dense(count: 2560, noBias: true, name: "qkv_from_attn_vid_k")
  let vidVProj = Dense(count: 2560, noBias: true, name: "qkv_from_attn_vid_v")
  let txtQProj = Dense(count: 2560, noBias: true, name: "qkv_from_attn_txt_q")
  let txtKProj = Dense(count: 2560, noBias: true, name: "qkv_from_attn_txt_k")
  let txtVProj = Dense(count: 2560, noBias: true, name: "qkv_from_attn_txt_v")
  let vidNormQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "qkv_from_attn_vid_norm_q")
  let vidNormK = RMSNorm(epsilon: 1e-5, axis: [2], name: "qkv_from_attn_vid_norm_k")
  let txtNormQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "qkv_from_attn_txt_norm_q")
  let txtNormK = RMSNorm(epsilon: 1e-5, axis: [2], name: "qkv_from_attn_txt_norm_k")
  let vidAttnDense = vidAttnIn.copied()
  let txtAttnDense = txtAttnIn.copied()
  let vidQ = vidNormQ(vidQProj(vidAttnDense).reshaped([vidLen, 20, 128]))
  let vidK = vidNormK(vidKProj(vidAttnDense).reshaped([vidLen, 20, 128]))
  let vidV = vidVProj(vidAttnDense).reshaped([vidLen, 20, 128])
  let txtQ = txtNormQ(txtQProj(txtAttnDense).reshaped([txtLen, 20, 128]))
  let txtK = txtNormK(txtKProj(txtAttnDense).reshaped([txtLen, 20, 128]))
  let txtV = txtVProj(txtAttnDense).reshaped([txtLen, 20, 128])
  let vidQFlat = vidQ.reshaped([vidLen, 2560])
  let vidKFlat = vidK.reshaped([vidLen, 2560])
  let vidVFlat = vidV.reshaped([vidLen, 2560])
  let txtQFlat = txtQ.reshaped([txtLen, 2560])
  let txtKFlat = txtK.reshaped([txtLen, 2560])
  let txtVFlat = txtV.reshaped([txtLen, 2560])
  let reader: (PythonObject, Int) -> Void = { stateDict, layerIndex in
    let prefix = "blocks.\(layerIndex)"
    let sharedWeights =
      (Bool(stateDict.__contains__("\(prefix).attn.proj_qkv.vid.weight")) ?? false) == false
    if sharedWeights {
      let allQKV = stateDict["\(prefix).attn.proj_qkv.all.weight"].to(torch.float).cpu()
      let allNormQ = try! Tensor<Float>(
        numpy: stateDict["\(prefix).attn.norm_q.all.weight"].to(torch.float).cpu().numpy())
      let allNormK = try! Tensor<Float>(
        numpy: stateDict["\(prefix).attn.norm_k.all.weight"].to(torch.float).cpu().numpy())
      let qWeight = try! Tensor<Float>(numpy: allQKV[..<2560, ...].numpy())
      let kWeight = try! Tensor<Float>(numpy: allQKV[2560..<5120, ...].numpy())
      let vWeight = try! Tensor<Float>(numpy: allQKV[5120..., ...].numpy())
      copyParameter(vidQProj.weight, from: qWeight, asBFloat16: bfloat16Weights)
      copyParameter(vidKProj.weight, from: kWeight, asBFloat16: bfloat16Weights)
      copyParameter(vidVProj.weight, from: vWeight, asBFloat16: bfloat16Weights)
      copyParameter(txtQProj.weight, from: qWeight, asBFloat16: bfloat16Weights)
      copyParameter(txtKProj.weight, from: kWeight, asBFloat16: bfloat16Weights)
      copyParameter(txtVProj.weight, from: vWeight, asBFloat16: bfloat16Weights)
      copyParameter(vidNormQ.weight, from: allNormQ, asBFloat16: bfloat16Weights)
      copyParameter(vidNormK.weight, from: allNormK, asBFloat16: bfloat16Weights)
      copyParameter(txtNormQ.weight, from: allNormQ, asBFloat16: bfloat16Weights)
      copyParameter(txtNormK.weight, from: allNormK, asBFloat16: bfloat16Weights)
    } else {
      let vidQKV = stateDict["\(prefix).attn.proj_qkv.vid.weight"].to(torch.float).cpu()
      let txtQKV = stateDict["\(prefix).attn.proj_qkv.txt.weight"].to(torch.float).cpu()
      copyParameter(
        vidQProj.weight, from: try! Tensor<Float>(numpy: vidQKV[..<2560, ...].numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidKProj.weight, from: try! Tensor<Float>(numpy: vidQKV[2560..<5120, ...].numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidVProj.weight, from: try! Tensor<Float>(numpy: vidQKV[5120..., ...].numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtQProj.weight, from: try! Tensor<Float>(numpy: txtQKV[..<2560, ...].numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtKProj.weight, from: try! Tensor<Float>(numpy: txtQKV[2560..<5120, ...].numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtVProj.weight, from: try! Tensor<Float>(numpy: txtQKV[5120..., ...].numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidNormQ.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.norm_q.vid.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        vidNormK.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.norm_k.vid.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtNormQ.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.norm_q.txt.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
      copyParameter(
        txtNormK.weight,
        from: try! Tensor<Float>(
          numpy: stateDict["\(prefix).attn.norm_k.txt.weight"].to(torch.float).cpu().numpy()),
        asBFloat16: bfloat16Weights)
    }
  }
  return (
    reader,
    Model(
      [vidAttnIn, txtAttnIn],
      [
        vidQ, vidK, vidV, txtQ, txtK, txtV, vidQFlat, vidKFlat, vidVFlat, txtQFlat, txtKFlat,
        txtVFlat,
      ])
  )
}

func SeedVR2DecoderConvIn(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: 512, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_in")
  let out = convIn(
    seedVR2TemporalReplicatePad(
      x, batchSize: 1, channels: 16, depth: depth, height: height, width: width, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let convInWeight = stateDict["decoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let convInBias = stateDict["decoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: convInWeight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: convInBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2UpDecoderBlock3D(
  prefix: String, inChannels: Int, outChannels: Int, depth: Int, height: Int, width: Int,
  addUpsample: Bool, temporalUp: Bool, spatialUp: Bool
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (resnet0Reader, resnet0) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.0", inChannels: inChannels, outChannels: outChannels, depth: depth,
    height: height, width: width)
  var out = resnet0(x)
  let (resnet1Reader, resnet1) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.1", inChannels: outChannels, outChannels: outChannels,
    depth: depth, height: height, width: width)
  out = resnet1(out)
  let (resnet2Reader, resnet2) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.2", inChannels: outChannels, outChannels: outChannels,
    depth: depth, height: height, width: width)
  out = resnet2(out)
  let upsampleReader: ((PythonObject) -> Void)?
  if addUpsample {
    let (reader, upsample) = SeedVR2Upsample3D(
      prefix: "\(prefix).upsamplers.0", channels: outChannels, depth: depth, height: height,
      width: width, temporalUp: temporalUp, spatialUp: spatialUp)
    out = upsample(out)
    upsampleReader = reader
  } else {
    upsampleReader = nil
  }
  let reader: (PythonObject) -> Void = { stateDict in
    resnet0Reader(stateDict)
    resnet1Reader(stateDict)
    resnet2Reader(stateDict)
    upsampleReader?(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2DecoderPostProcess(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let convNormOut = GroupNorm(
    axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: "conv_norm_out")
  var out = x.permuted(0, 2, 1, 3, 4).copied().reshaped([depth, 128, height, width])
  out = convNormOut(out).reshaped([1, depth, 128, height, width]).permuted(0, 2, 1, 3, 4)
    .copied()
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_out")
  out = convOut(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: 128, depth: depth, height: height, width: width, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let normWeight = stateDict["decoder.conv_norm_out.weight"].to(torch.float).cpu().numpy()
    let normBias = stateDict["decoder.conv_norm_out.bias"].to(torch.float).cpu().numpy()
    convNormOut.weight.copy(from: try! Tensor<Float>(numpy: normWeight))
    convNormOut.bias.copy(from: try! Tensor<Float>(numpy: normBias))
    let convWeight = stateDict["decoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let convBias = stateDict["decoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: convWeight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: convBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Decoder3D(startDepth: Int, startHeight: Int, startWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (convInReader, convIn) = SeedVR2DecoderConvIn(
    depth: startDepth, height: startHeight, width: startWidth)
  var out = convIn(x)

  let (midBlockReader, midBlock) = SeedVR2DecoderMidBlock3D(
    depth: startDepth, height: startHeight, width: startWidth)
  out = midBlock(out)

  let (upBlock0Reader, upBlock0) = SeedVR2UpDecoderBlock3D(
    prefix: "decoder.up_blocks.0", inChannels: 512, outChannels: 512, depth: startDepth,
    height: startHeight, width: startWidth, addUpsample: true, temporalUp: true, spatialUp: true)
  out = upBlock0(out)
  let depth1 = 1 + (startDepth - 1) * 2
  let height1 = startHeight * 2
  let width1 = startWidth * 2

  let (upBlock1Reader, upBlock1) = SeedVR2UpDecoderBlock3D(
    prefix: "decoder.up_blocks.1", inChannels: 512, outChannels: 512, depth: depth1,
    height: height1, width: width1, addUpsample: true, temporalUp: true, spatialUp: true)
  out = upBlock1(out)
  let depth2 = 1 + (depth1 - 1) * 2
  let height2 = height1 * 2
  let width2 = width1 * 2

  let (upBlock2Reader, upBlock2) = SeedVR2UpDecoderBlock3D(
    prefix: "decoder.up_blocks.2", inChannels: 512, outChannels: 256, depth: depth2,
    height: height2, width: width2, addUpsample: true, temporalUp: false, spatialUp: true)
  out = upBlock2(out)
  let depth3 = depth2
  let height3 = height2 * 2
  let width3 = width2 * 2

  let (upBlock3Reader, upBlock3) = SeedVR2UpDecoderBlock3D(
    prefix: "decoder.up_blocks.3", inChannels: 256, outChannels: 128, depth: depth3,
    height: height3, width: width3, addUpsample: false, temporalUp: false, spatialUp: false)
  out = upBlock3(out)

  let (postReader, postProcess) = SeedVR2DecoderPostProcess(
    depth: depth3, height: height3, width: width3)
  out = postProcess(out)

  let reader: (PythonObject) -> Void = { stateDict in
    convInReader(stateDict)
    midBlockReader(stateDict)
    upBlock0Reader(stateDict)
    upBlock1Reader(stateDict)
    upBlock2Reader(stateDict)
    upBlock3Reader(stateDict)
    postReader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2EncoderConvIn(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_in")
  let out = convIn(
    seedVR2TemporalReplicatePad(
      x, batchSize: 1, channels: 3, depth: depth, height: height, width: width, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["encoder.conv_in.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["encoder.conv_in.bias"].to(torch.float).cpu().numpy()
    convIn.weight.copy(from: try! Tensor<Float>(numpy: weight))
    convIn.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Downsample3D(
  prefix: String, channels: Int, temporalDown: Bool, spatialDown: Bool, depth: Int, height: Int,
  width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let temporalRatio = temporalDown ? 2 : 1
  let spatialRatio = spatialDown ? 2 : 1
  let temporalKernel = temporalDown ? 3 : 1
  let spatialKernel = spatialDown ? 3 : 1
  var out: Model.IO = x
  if spatialDown {
    out = out.padded(.zero, begin: [0, 0, 0, 0, 0], end: [0, 0, 0, 1, 1])
  }
  let conv = Convolution(
    groups: 1, filters: channels, filterSize: [temporalKernel, spatialKernel, spatialKernel],
    hint: Hint(stride: [temporalRatio, spatialRatio, spatialRatio]),
    name: "conv")
  if temporalDown {
    out = conv(
      seedVR2TemporalReplicatePad(
        out, batchSize: 1, channels: channels, depth: depth, height: height + (spatialDown ? 1 : 0),
        width: width + (spatialDown ? 1 : 0), left: 2))
  } else {
    out = conv(out)
  }
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["\(prefix).conv.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["\(prefix).conv.bias"].to(torch.float).cpu().numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: weight))
    conv.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Downsample3DProbe(
  prefix: String, channels: Int, temporalDown: Bool, spatialDown: Bool, depth: Int, height: Int,
  width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let temporalRatio = temporalDown ? 2 : 1
  let spatialRatio = spatialDown ? 2 : 1
  let temporalKernel = temporalDown ? 3 : 1
  let spatialKernel = spatialDown ? 3 : 1
  var spatialPadded: Model.IO = x
  if spatialDown {
    spatialPadded = spatialPadded.padded(.zero, begin: [0, 0, 0, 0, 0], end: [0, 0, 0, 1, 1])
  }
  let conv = Convolution(
    groups: 1, filters: channels, filterSize: [temporalKernel, spatialKernel, spatialKernel],
    hint: Hint(stride: [temporalRatio, spatialRatio, spatialRatio]),
    name: "conv")
  let out: Model.IO
  if temporalDown {
    out = conv(
      seedVR2TemporalReplicatePad(
        spatialPadded, batchSize: 1, channels: channels, depth: depth,
        height: height + (spatialDown ? 1 : 0),
        width: width + (spatialDown ? 1 : 0), left: 2))
  } else {
    out = conv(spatialPadded)
  }
  let reader: (PythonObject) -> Void = { stateDict in
    let weight = stateDict["\(prefix).conv.weight"].to(torch.float).cpu().numpy()
    let bias = stateDict["\(prefix).conv.bias"].to(torch.float).cpu().numpy()
    conv.weight.copy(from: try! Tensor<Float>(numpy: weight))
    conv.bias.copy(from: try! Tensor<Float>(numpy: bias))
  }
  return (reader, Model([x], [spatialPadded, out]))
}

func SeedVR2DownEncoderBlock3D(
  prefix: String, inChannels: Int, outChannels: Int, addDownsample: Bool, temporalDown: Bool,
  spatialDown: Bool, depth: Int, height: Int, width: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (resnet0Reader, resnet0) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.0", inChannels: inChannels, outChannels: outChannels, depth: depth,
    height: height, width: width)
  var out = resnet0(x)
  let (resnet1Reader, resnet1) = SeedVR2ResnetBlock3D(
    prefix: "\(prefix).resnets.1", inChannels: outChannels, outChannels: outChannels,
    depth: depth, height: height, width: width)
  out = resnet1(out)
  let downsampleReader: ((PythonObject) -> Void)?
  if addDownsample {
    let (reader, downsample) = SeedVR2Downsample3D(
      prefix: "\(prefix).downsamplers.0", channels: outChannels, temporalDown: temporalDown,
      spatialDown: spatialDown, depth: depth, height: height, width: width)
    out = downsample(out)
    downsampleReader = reader
  } else {
    downsampleReader = nil
  }
  let reader: (PythonObject) -> Void = { stateDict in
    resnet0Reader(stateDict)
    resnet1Reader(stateDict)
    downsampleReader?(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2EncoderMidBlock3D(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (resnet0Reader, resnet0) = SeedVR2ResnetBlock3D(
    prefix: "encoder.mid_block.resnets.0", inChannels: 512, outChannels: 512, depth: depth,
    height: height, width: width)
  var out = resnet0(x)
  let (attnReader, attn) = SeedVR2AttentionBlock2D(
    prefix: "encoder.mid_block.attentions.0", channels: 512, batchSize: depth, height: height,
    width: width)
  let frameWise = out.permuted(0, 2, 1, 3, 4).contiguous().reshaped([depth, 512, height, width])
  out = attn(frameWise).reshaped([1, depth, 512, height, width]).permuted(0, 2, 1, 3, 4)
    .contiguous()
  let (resnet1Reader, resnet1) = SeedVR2ResnetBlock3D(
    prefix: "encoder.mid_block.resnets.1", inChannels: 512, outChannels: 512, depth: depth,
    height: height, width: width)
  out = resnet1(out)
  let reader: (PythonObject) -> Void = { stateDict in
    resnet0Reader(stateDict)
    attnReader(stateDict)
    resnet1Reader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2EncoderPostProcess(depth: Int, height: Int, width: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let convNormOut = GroupNorm(
    axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: "conv_norm_out")
  var out = x.permuted(0, 2, 1, 3, 4).copied().reshaped([depth, 512, height, width])
  out = convNormOut(out).reshaped([1, depth, 512, height, width]).permuted(0, 2, 1, 3, 4)
    .copied()
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
    name: "conv_out")
  out = convOut(
    seedVR2TemporalReplicatePad(
      out, batchSize: 1, channels: 512, depth: depth, height: height, width: width, left: 2))
  let reader: (PythonObject) -> Void = { stateDict in
    let normWeight = stateDict["encoder.conv_norm_out.weight"].to(torch.float).cpu().numpy()
    let normBias = stateDict["encoder.conv_norm_out.bias"].to(torch.float).cpu().numpy()
    convNormOut.weight.copy(from: try! Tensor<Float>(numpy: normWeight))
    convNormOut.bias.copy(from: try! Tensor<Float>(numpy: normBias))
    let convWeight = stateDict["encoder.conv_out.weight"].to(torch.float).cpu().numpy()
    let convBias = stateDict["encoder.conv_out.bias"].to(torch.float).cpu().numpy()
    convOut.weight.copy(from: try! Tensor<Float>(numpy: convWeight))
    convOut.bias.copy(from: try! Tensor<Float>(numpy: convBias))
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2Encoder3D(startDepth: Int, startHeight: Int, startWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (convInReader, convIn) = SeedVR2EncoderConvIn(
    depth: startDepth, height: startHeight, width: startWidth)
  var out = convIn(x)

  let (downBlock0Reader, downBlock0) = SeedVR2DownEncoderBlock3D(
    prefix: "encoder.down_blocks.0", inChannels: 128, outChannels: 128, addDownsample: true,
    temporalDown: false, spatialDown: true, depth: startDepth, height: startHeight,
    width: startWidth)
  out = downBlock0(out)
  let depth1 = startDepth
  let height1 = startHeight / 2
  let width1 = startWidth / 2

  let (downBlock1Reader, downBlock1) = SeedVR2DownEncoderBlock3D(
    prefix: "encoder.down_blocks.1", inChannels: 128, outChannels: 256, addDownsample: true,
    temporalDown: true, spatialDown: true, depth: depth1, height: height1, width: width1)
  out = downBlock1(out)
  let depth2 = (depth1 + 1) / 2
  let height2 = height1 / 2
  let width2 = width1 / 2

  let (downBlock2Reader, downBlock2) = SeedVR2DownEncoderBlock3D(
    prefix: "encoder.down_blocks.2", inChannels: 256, outChannels: 512, addDownsample: true,
    temporalDown: true, spatialDown: true, depth: depth2, height: height2, width: width2)
  out = downBlock2(out)
  let depth3 = (depth2 + 1) / 2
  let height3 = height2 / 2
  let width3 = width2 / 2

  let (downBlock3Reader, downBlock3) = SeedVR2DownEncoderBlock3D(
    prefix: "encoder.down_blocks.3", inChannels: 512, outChannels: 512, addDownsample: false,
    temporalDown: false, spatialDown: false, depth: depth3, height: height3, width: width3)
  out = downBlock3(out)

  let (midBlockReader, midBlock) = SeedVR2EncoderMidBlock3D(
    depth: depth3, height: height3, width: width3)
  out = midBlock(out)

  let (postReader, postProcess) = SeedVR2EncoderPostProcess(
    depth: depth3, height: height3, width: width3)
  out = postProcess(out)

  let reader: (PythonObject) -> Void = { stateDict in
    convInReader(stateDict)
    downBlock0Reader(stateDict)
    downBlock1Reader(stateDict)
    downBlock2Reader(stateDict)
    downBlock3Reader(stateDict)
    midBlockReader(stateDict)
    postReader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2VAE3D(startDepth: Int, startHeight: Int, startWidth: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let (encoderReader, encoder) = SeedVR2Encoder3D(
    startDepth: startDepth, startHeight: startHeight, startWidth: startWidth)
  let moments = encoder(x)
  let latentDepth1 = (startDepth + 1) / 2
  let latentDepth = (latentDepth1 + 1) / 2
  let latentHeight = startHeight / 8
  let latentWidth = startWidth / 8
  let latent = moments.reshaped(
    [1, 16, latentDepth, latentHeight, latentWidth],
    strides: [
      32 * latentDepth * latentHeight * latentWidth, latentDepth * latentHeight * latentWidth,
      latentHeight * latentWidth, latentWidth, 1,
    ]
  ).contiguous()
  let (decoderReader, decoder) = SeedVR2Decoder3D(
    startDepth: latentDepth, startHeight: latentHeight, startWidth: latentWidth)
  let out = decoder(latent)
  let reader: (PythonObject) -> Void = { stateDict in
    encoderReader(stateDict)
    decoderReader(stateDict)
  }
  return (reader, Model([x], [out.copied()]))
}

func SeedVR2MomentsToLatent(depth: Int, height: Int, width: Int) -> Model {
  let x = Input()
  let latent = x.reshaped(
    [1, 16, depth, height, width],
    strides: [32 * depth * height * width, depth * height * width, height * width, width, 1]
  ).contiguous()
  return Model([x], [latent])
}

let vaeReference = seedvr2Reference.SeedVR2Reference(
  repo_root: seedVRRoot,
  checkpoint_root: "\(seedVRRoot)/SeedVR2-3B",
  device: hasCUDA ? "cuda" : "cpu",
  load_dit: false,
  load_vae: true)

logStep("SeedVR2 vae state_dict start")
let vaeStateDict = vaeReference.runner.vae.state_dict()
logStep("SeedVR2 vae encoder_probe start")
let vaeEncoderProbe = vaeReference.make_encoder_probe(depth: 5, height: 96, width: 160, seed: 42)
logStep("SeedVR2 vae decoder_probe start")
let vaeDecoderProbe = vaeReference.make_decoder_probe(depth: 2, height: 12, width: 20, seed: 42)
logStep("SeedVR2 vae mode_probe start")
let vaeModeProbe = vaeReference.make_vae_mode_probe(frames: 5, height: 96, width: 160, seed: 42)

if processInfo.environment["SEEDVR2_RUN_VAE"] == "1" {
  let graph = DynamicGraph()
  graph.maxConcurrency = .limit(1)
  graph.withNoGrad {
    func materialize(_ value: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
      copiedToCPU(value)
    }

    func loadInput(_ probe: PythonObject, _ key: String) -> DynamicGraph.Tensor<Float> {
      placeOnDevice(
        graph.variable(try! Tensor<Float>(numpy: probe[key].to(torch.float).cpu().numpy())))
    }

    func loadTensor(_ probe: PythonObject, _ key: String) -> Tensor<Float> {
      try! Tensor<Float>(numpy: probe[key].to(torch.float).cpu().numpy())
    }

    func printParity(_ name: String, _ swift: Tensor<Float>, _ reference: Tensor<Float>) {
      print("\(name) max abs diff:", maxAbsDiff5D(swift, reference))
      print("\(name) max rel diff:", maxRelativeDiff5D(swift, reference))
    }

    func runEncoderSequential(_ input: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
      logStep("SeedVR2 vae encoder start")
      let (convInReader, convIn) = SeedVR2EncoderConvIn(depth: 5, height: 96, width: 160)
      convIn.compile(inputs: input)
      convInReader(vaeStateDict)
      var stage = materialize(convIn(inputs: input)[0].as(of: Float.self))

      let down0Input = rematerializeOnDevice(graph, stage)
      let (down0Reader, down0) = SeedVR2DownEncoderBlock3D(
        prefix: "encoder.down_blocks.0", inChannels: 128, outChannels: 128, addDownsample: true,
        temporalDown: false, spatialDown: true, depth: 5, height: 96, width: 160)
      down0.compile(inputs: down0Input)
      down0Reader(vaeStateDict)
      stage = materialize(down0(inputs: down0Input)[0].as(of: Float.self))

      let down1Input = rematerializeOnDevice(graph, stage)
      let (down1Reader, down1) = SeedVR2DownEncoderBlock3D(
        prefix: "encoder.down_blocks.1", inChannels: 128, outChannels: 256, addDownsample: true,
        temporalDown: true, spatialDown: true, depth: 5, height: 48, width: 80)
      down1.compile(inputs: down1Input)
      down1Reader(vaeStateDict)
      stage = materialize(down1(inputs: down1Input)[0].as(of: Float.self))

      let down2Input = rematerializeOnDevice(graph, stage)
      let (down2Reader, down2) = SeedVR2DownEncoderBlock3D(
        prefix: "encoder.down_blocks.2", inChannels: 256, outChannels: 512, addDownsample: true,
        temporalDown: true, spatialDown: true, depth: 3, height: 24, width: 40)
      down2.compile(inputs: down2Input)
      down2Reader(vaeStateDict)
      stage = materialize(down2(inputs: down2Input)[0].as(of: Float.self))

      let down3Input = rematerializeOnDevice(graph, stage)
      let (down3Reader, down3) = SeedVR2DownEncoderBlock3D(
        prefix: "encoder.down_blocks.3", inChannels: 512, outChannels: 512, addDownsample: false,
        temporalDown: false, spatialDown: false, depth: 2, height: 12, width: 20)
      down3.compile(inputs: down3Input)
      down3Reader(vaeStateDict)
      stage = materialize(down3(inputs: down3Input)[0].as(of: Float.self))

      let midInput = rematerializeOnDevice(graph, stage)
      let (midReader, mid) = SeedVR2EncoderMidBlock3D(depth: 2, height: 12, width: 20)
      mid.compile(inputs: midInput)
      midReader(vaeStateDict)
      stage = materialize(mid(inputs: midInput)[0].as(of: Float.self))

      let postInput = rematerializeOnDevice(graph, stage)
      let (postReader, post) = SeedVR2EncoderPostProcess(depth: 2, height: 12, width: 20)
      post.compile(inputs: postInput)
      postReader(vaeStateDict)
      return materialize(post(inputs: postInput)[0].as(of: Float.self))
    }

    func runDecoderSequential(_ input: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
      logStep("SeedVR2 vae decoder start")
      let (convInReader, convIn) = SeedVR2DecoderConvIn(depth: 2, height: 12, width: 20)
      convIn.compile(inputs: input)
      convInReader(vaeStateDict)
      var stage = materialize(convIn(inputs: input)[0].as(of: Float.self))

      let midInput = rematerializeOnDevice(graph, stage)
      let (midReader, mid) = SeedVR2DecoderMidBlock3D(depth: 2, height: 12, width: 20)
      mid.compile(inputs: midInput)
      midReader(vaeStateDict)
      stage = materialize(mid(inputs: midInput)[0].as(of: Float.self))

      let up0Input = rematerializeOnDevice(graph, stage)
      let (up0Reader, up0) = SeedVR2UpDecoderBlock3D(
        prefix: "decoder.up_blocks.0", inChannels: 512, outChannels: 512, depth: 2,
        height: 12, width: 20, addUpsample: true, temporalUp: true, spatialUp: true)
      up0.compile(inputs: up0Input)
      up0Reader(vaeStateDict)
      stage = materialize(up0(inputs: up0Input)[0].as(of: Float.self))

      let up1Input = rematerializeOnDevice(graph, stage)
      let (up1Reader, up1) = SeedVR2UpDecoderBlock3D(
        prefix: "decoder.up_blocks.1", inChannels: 512, outChannels: 512, depth: 3,
        height: 24, width: 40, addUpsample: true, temporalUp: true, spatialUp: true)
      up1.compile(inputs: up1Input)
      up1Reader(vaeStateDict)
      stage = materialize(up1(inputs: up1Input)[0].as(of: Float.self))

      let up2Input = rematerializeOnDevice(graph, stage)
      let (up2Reader, up2) = SeedVR2UpDecoderBlock3D(
        prefix: "decoder.up_blocks.2", inChannels: 512, outChannels: 256, depth: 5,
        height: 48, width: 80, addUpsample: true, temporalUp: false, spatialUp: true)
      up2.compile(inputs: up2Input)
      up2Reader(vaeStateDict)
      stage = materialize(up2(inputs: up2Input)[0].as(of: Float.self))

      let up3Input = rematerializeOnDevice(graph, stage)
      let (up3Reader, up3) = SeedVR2UpDecoderBlock3D(
        prefix: "decoder.up_blocks.3", inChannels: 256, outChannels: 128, depth: 5,
        height: 96, width: 160, addUpsample: false, temporalUp: false, spatialUp: false)
      up3.compile(inputs: up3Input)
      up3Reader(vaeStateDict)
      stage = materialize(up3(inputs: up3Input)[0].as(of: Float.self))

      let postInput = rematerializeOnDevice(graph, stage)
      let (postReader, post) = SeedVR2DecoderPostProcess(depth: 5, height: 96, width: 160)
      post.compile(inputs: postInput)
      postReader(vaeStateDict)
      return materialize(post(inputs: postInput)[0].as(of: Float.self))
    }

    logStep("SeedVR2 vae validation start")
    let encoderInput = loadInput(vaeEncoderProbe, "input")
    let swiftEncoderOutput = runEncoderSequential(encoderInput)
    let torchEncoderOutput = loadTensor(vaeEncoderProbe, "output")
    printParity("SeedVR2 vae.encoder", swiftEncoderOutput, torchEncoderOutput)

    let decoderInput = loadInput(vaeDecoderProbe, "input")
    let swiftDecoderOutput = runDecoderSequential(decoderInput)
    let torchDecoderOutput = loadTensor(vaeDecoderProbe, "output")
    printParity("SeedVR2 vae.decoder", swiftDecoderOutput, torchDecoderOutput)

    let vaeModeInput = loadInput(vaeModeProbe, "input")
    let swiftModeMoments = runEncoderSequential(vaeModeInput)
    let torchModeMoments = loadTensor(vaeModeProbe, "moments")
    printParity("SeedVR2 vae.mode moments", swiftModeMoments, torchModeMoments)

    let torchModeLatent = loadTensor(vaeModeProbe, "latent")
    let momentsToLatentInput = rematerializeOnDevice(graph, swiftModeMoments)
    let momentsToLatent = SeedVR2MomentsToLatent(depth: 2, height: 12, width: 20)
    momentsToLatent.compile(inputs: momentsToLatentInput)
    let swiftModeLatent = materialize(
      momentsToLatent(inputs: momentsToLatentInput)[0].as(of: Float.self))
    printParity("SeedVR2 vae.mode latent", swiftModeLatent, torchModeLatent)

    let torchModeDecoded = loadTensor(vaeModeProbe, "output")
    let swiftModeDecoded = runDecoderSequential(rematerializeOnDevice(graph, swiftModeLatent))
    printParity("SeedVR2 vae.mode decoded", swiftModeDecoded, torchModeDecoded)

    let decodedRelDetails = maxRelativeDiffDetails5D(swiftModeDecoded, torchModeDecoded)
    print(
      "SeedVR2 vae.mode decoded max rel detail:",
      "rel", decodedRelDetails.0,
      "abs", decodedRelDetails.1,
      "ref", decodedRelDetails.2,
      "swift", decodedRelDetails.3,
      "index", decodedRelDetails.4)

    printParity("SeedVR2 vae full", swiftModeDecoded, torchModeDecoded)
  }
  exit(0)
}

let ditReference = seedvr2Reference.SeedVR2Reference(
  repo_root: seedVRRoot,
  checkpoint_root: "\(seedVRRoot)/SeedVR2-3B",
  device: hasCUDA ? "cuda" : "cpu",
  load_dit: true,
  load_vae: false)

if processInfo.environment["SEEDVR2_RUN_DIT_SINGLE_WINDOW"] != "0" {
  let ditFrames = 1
  let ditLatentHeight = 6
  let ditLatentWidth = 6
  let ditVidLen = ditFrames * (ditLatentHeight / 2) * (ditLatentWidth / 2)
  let ditTxtLen = 58
  let printEyeball = processInfo.environment["SEEDVR2_PRINT_EYEBALL"] == "1"

  logStep("SeedVR2 dit.single state_dict start")
  let ditStateDict = ditReference.runner.dit.state_dict()
  logStep("SeedVR2 dit.single rope_probe start")
  let ditSingleProbe = ditReference.make_dit_block_probe(
    layer_idx: 0, frames: ditFrames, latent_height: ditLatentHeight, latent_width: ditLatentWidth,
    timestep: 500.0,
    disable_rope: false)
  logStep("SeedVR2 dit.single official_probe start")
  let ditSingleOfficialProbe = ditReference.make_dit_body_probe_official(
    frames: ditFrames, latent_height: ditLatentHeight, latent_width: ditLatentWidth, timestep: 500.0
  )

  graph.withNoGrad {
    func loadTensor(_ probe: PythonObject, _ key: String) -> Tensor<Float> {
      try! Tensor<Float>(numpy: probe[key].to(torch.float).cpu().numpy())
    }

    let embCPU = loadTensor(ditSingleOfficialProbe, "emb")
    let compileVid = rematerializeOnDevice(
      graph, loadTensor(ditSingleOfficialProbe, "layer0_vid_input"))
    let compileTxt = rematerializeOnDevice(
      graph, loadTensor(ditSingleOfficialProbe, "layer0_txt_input"))
    let vidFreqs = placeOnDevice(graph.variable(loadTensor(ditSingleProbe, "window0_vid_freqs")))
    let txtFreqs = placeOnDevice(graph.variable(loadTensor(ditSingleProbe, "window0_txt_freqs")))

    let layer0VidAttnMod = rematerializeOnDevice(
      graph,
      seedVR2BlockModTensor(
        emb: embCPU, stateDict: ditStateDict, blockIndex: 0, branch: "vid", layer: "attn"))
    let layer0TxtAttnMod = rematerializeOnDevice(
      graph,
      seedVR2BlockModTensor(
        emb: embCPU, stateDict: ditStateDict, blockIndex: 0, branch: "txt", layer: "attn"))
    let layer0VidMlpMod = rematerializeOnDevice(
      graph,
      seedVR2BlockModTensor(
        emb: embCPU, stateDict: ditStateDict, blockIndex: 0, branch: "vid", layer: "mlp"))
    let layer0TxtMlpMod = rematerializeOnDevice(
      graph,
      seedVR2BlockModTensor(
        emb: embCPU, stateDict: ditStateDict, blockIndex: 0, branch: "txt", layer: "mlp"))

    logStep("SeedVR2 dit.single compile body")
    let attnInProbe = SeedVR2DiTAttnInProbe(vidLen: ditVidLen, txtLen: ditTxtLen)
    attnInProbe.compile(inputs: compileVid, compileTxt, layer0VidAttnMod, layer0TxtAttnMod)
    let attnInProbeLast = SeedVR2DiTAttnInProbeLastLayer(vidLen: ditVidLen, txtLen: ditTxtLen)
    attnInProbeLast.compile(inputs: compileVid, compileTxt, layer0VidAttnMod)

    let (singleQKVFromAttnReader, singleQKVFromAttn) = SeedVR2DiTQKVFromAttnInputsProbe(
      vidLen: ditVidLen, txtLen: ditTxtLen)
    singleQKVFromAttn.compile(
      inputs:
        rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "vid_attn_in")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "txt_attn_in")))

    let vidRotaryProbe = SeedVR2RotaryProbe(seqLen: ditVidLen)
    vidRotaryProbe.compile(
      inputs:
        rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "vid_q")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "vid_k")),
      vidFreqs)
    let txtRotaryProbe = SeedVR2RotaryProbe(seqLen: ditTxtLen)
    txtRotaryProbe.compile(
      inputs:
        rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "txt_q")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "txt_k")),
      txtFreqs)

    let (attnProjectReader, attnProject) = SeedVR2DiTAttnProjectOnly(
      vidLen: ditVidLen, txtLen: ditTxtLen)
    attnProject.compile(
      inputs:
        compileVid, compileTxt, layer0VidAttnMod, layer0TxtAttnMod,
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_vid_q_rope")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_vid_k_rope")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_vid_v")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_txt_q_rope")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_txt_k_rope")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_txt_v")))
    let (attnProjectLastReader, attnProjectLast) = SeedVR2DiTAttnProjectOnlyLastLayer(
      vidLen: ditVidLen, txtLen: ditTxtLen)
    attnProjectLast.compile(
      inputs:
        compileVid, compileTxt, layer0VidAttnMod,
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_vid_q_rope")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_vid_k_rope")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_vid_v")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_txt_q_rope")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_txt_k_rope")),
      rematerializeOnDevice(graph, loadTensor(ditSingleProbe, "window0_txt_v")))

    let (mlpReader, mlp) = SeedVR2DiTMlpOnly(vidLen: ditVidLen, txtLen: ditTxtLen)
    mlp.compile(inputs: compileVid, compileTxt, layer0VidMlpMod, layer0TxtMlpMod)
    let (mlpLastReader, mlpLast) = SeedVR2DiTMlpOnlyLastLayer(
      vidLen: ditVidLen, txtLen: ditTxtLen)
    mlpLast.compile(inputs: compileVid, compileTxt, layer0VidMlpMod)

    logStep("SeedVR2 dit.single compile input/output")
    let rawVidInput = rematerializeOnDevice(graph, loadTensor(ditSingleOfficialProbe, "vid_input"))
    let rawTxtInput = rematerializeOnDevice(graph, loadTensor(ditSingleOfficialProbe, "txt_input"))
    let timeInput = placeOnDevice(
      graph.variable(
        seedVR2TimeEmbedding(timestep: 500.0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)))

    let (txtInReader, txtInModel) = SeedVR2DiTTextIn()
    txtInModel.compile(inputs: rawTxtInput)
    let (patchInReader, patchInModel) = SeedVR2DiTPatchIn(
      frames: ditFrames, latentHeight: ditLatentHeight, latentWidth: ditLatentWidth)
    patchInModel.compile(inputs: rawVidInput)
    let (embReader, embModel) = SeedVR2DiTTimeEmbedding()
    embModel.compile(inputs: timeInput)
    let (outputHeadReader, outputHead) = SeedVR2DiTOutputHead(
      frames: ditFrames, latentHeight: ditLatentHeight, latentWidth: ditLatentWidth)
    outputHead.compile(
      inputs:
        compileVid,
      rematerializeOnDevice(
        graph, seedVR2OutputModTensor(emb: embCPU, stateDict: ditStateDict, offset: 0)))

    txtInReader(ditStateDict)
    patchInReader(ditStateDict)
    embReader(ditStateDict)
    outputHeadReader(ditStateDict)

    let fullTxtInput = txtInModel(inputs: rawTxtInput)[0].as(of: Float.self).copied()
    let fullVidInput = patchInModel(inputs: rawVidInput)[0].as(of: Float.self).copied()
    let fullEmb = embModel(inputs: timeInput)[0].as(of: Float.self).copied()
    let fullEmbCPU = copiedToCPU(fullEmb)
    print(
      "SeedVR2 dit.single init vid max abs diff:",
      maxAbsDiff2DTensor(
        copiedToCPU(fullVidInput), loadTensor(ditSingleOfficialProbe, "layer0_vid_input")))
    print(
      "SeedVR2 dit.single init txt max abs diff:",
      maxAbsDiff2DTensor(
        copiedToCPU(fullTxtInput), loadTensor(ditSingleOfficialProbe, "layer0_txt_input")))
    print(
      "SeedVR2 dit.single init emb max abs diff:",
      maxAbsDiff2DTensor(fullEmbCPU, embCPU))

    let summaryLayers: Set<Int> = [0, 1, 9, 30, 31]
    var chainVid = fullVidInput
    var chainTxt = fullTxtInput
    var worstChainedVidLayer = 0
    var worstChainedTxtLayer = 0
    var worstChainedVid: Float = 0
    var worstChainedTxt: Float = 0

    for layerIndex in 0..<32 {
      let layerVidAttnMod = rematerializeOnDevice(
        graph,
        seedVR2BlockModTensor(
          emb: fullEmbCPU, stateDict: ditStateDict, blockIndex: layerIndex, branch: "vid",
          layer: "attn"))
      let layerTxtAttnMod = rematerializeOnDevice(
        graph,
        seedVR2BlockModTensor(
          emb: fullEmbCPU, stateDict: ditStateDict, blockIndex: layerIndex, branch: "txt",
          layer: "attn"))
      let layerVidMlpMod = rematerializeOnDevice(
        graph,
        seedVR2BlockModTensor(
          emb: fullEmbCPU, stateDict: ditStateDict, blockIndex: layerIndex, branch: "vid",
          layer: "mlp"))
      let layerTxtMlpMod = rematerializeOnDevice(
        graph,
        seedVR2BlockModTensor(
          emb: fullEmbCPU, stateDict: ditStateDict, blockIndex: layerIndex, branch: "txt",
          layer: "mlp"))

      let layerAttnInOutputs: [DynamicGraph.AnyTensor]
      if layerIndex == 31 {
        layerAttnInOutputs = attnInProbeLast(inputs: chainVid, chainTxt, layerVidAttnMod)
      } else {
        layerAttnInOutputs = attnInProbe(
          inputs: chainVid, chainTxt, layerVidAttnMod, layerTxtAttnMod)
      }
      singleQKVFromAttnReader(ditStateDict, layerIndex)
      let layerQKVOutputs = singleQKVFromAttn(inputs: layerAttnInOutputs[0], layerAttnInOutputs[1])
      let layerVidRotaryOutputs = vidRotaryProbe(
        inputs: layerQKVOutputs[0], layerQKVOutputs[1], vidFreqs)
      let layerTxtRotaryOutputs = txtRotaryProbe(
        inputs: layerQKVOutputs[3], layerQKVOutputs[4], txtFreqs)

      let layerAttnOutputs: [DynamicGraph.AnyTensor]
      if layerIndex == 31 {
        attnProjectLastReader(ditStateDict, layerIndex)
        layerAttnOutputs = attnProjectLast(
          inputs:
            chainVid, chainTxt, layerVidAttnMod,
          layerVidRotaryOutputs[0], layerVidRotaryOutputs[1], layerQKVOutputs[2],
          layerTxtRotaryOutputs[0], layerTxtRotaryOutputs[1], layerQKVOutputs[5])
      } else {
        attnProjectReader(ditStateDict, layerIndex)
        layerAttnOutputs = attnProject(
          inputs:
            chainVid, chainTxt, layerVidAttnMod, layerTxtAttnMod,
          layerVidRotaryOutputs[0], layerVidRotaryOutputs[1], layerQKVOutputs[2],
          layerTxtRotaryOutputs[0], layerTxtRotaryOutputs[1], layerQKVOutputs[5])
      }

      let layerOutputs: [DynamicGraph.AnyTensor]
      if layerIndex == 31 {
        mlpLastReader(ditStateDict, layerIndex)
        layerOutputs = mlpLast(inputs: layerAttnOutputs[0], layerAttnOutputs[1], layerVidMlpMod)
      } else {
        mlpReader(ditStateDict, layerIndex)
        layerOutputs = mlp(
          inputs: layerAttnOutputs[0], layerAttnOutputs[1], layerVidMlpMod, layerTxtMlpMod)
      }
      chainVid = layerOutputs[0].as(of: Float.self).copied()
      chainTxt = layerOutputs[1].as(of: Float.self).copied()

      let layerVidDiff = maxAbsDiff2DTensor(
        copiedToCPU(chainVid),
        loadTensor(ditSingleOfficialProbe, "layer\(layerIndex)_vid_before_out"))
      let layerTxtDiff = maxAbsDiff2DTensor(
        copiedToCPU(chainTxt),
        loadTensor(ditSingleOfficialProbe, "layer\(layerIndex)_txt_before_out"))
      if layerVidDiff > worstChainedVid {
        worstChainedVid = layerVidDiff
        worstChainedVidLayer = layerIndex
      }
      if layerTxtDiff > worstChainedTxt {
        worstChainedTxt = layerTxtDiff
        worstChainedTxtLayer = layerIndex
      }
      if summaryLayers.contains(layerIndex) {
        print("SeedVR2 dit.single layer\(layerIndex) vid max abs diff:", layerVidDiff)
        print("SeedVR2 dit.single layer\(layerIndex) txt max abs diff:", layerTxtDiff)
      }
    }

    print(
      "SeedVR2 dit.single body worst vid max abs diff:",
      worstChainedVid, "layer:", worstChainedVidLayer)
    print(
      "SeedVR2 dit.single body worst txt max abs diff:",
      worstChainedTxt, "layer:", worstChainedTxtLayer)

    let swiftBodyVid = copiedToCPU(chainVid)
    let swiftBodyTxt = copiedToCPU(chainTxt)
    let torchBodyVid = loadTensor(ditSingleOfficialProbe, "vid_before_out")
    let torchBodyTxt = loadTensor(ditSingleOfficialProbe, "txt_before_out")
    print(
      "SeedVR2 dit.single body vid max abs diff:", maxAbsDiff2DTensor(swiftBodyVid, torchBodyVid))
    print(
      "SeedVR2 dit.single body vid global max rel diff:",
      maxGlobalRelativeDiff2DTensor(swiftBodyVid, torchBodyVid))
    print(
      "SeedVR2 dit.single body txt max abs diff:", maxAbsDiff2DTensor(swiftBodyTxt, torchBodyTxt))
    print(
      "SeedVR2 dit.single body txt global max rel diff:",
      maxGlobalRelativeDiff2DTensor(swiftBodyTxt, torchBodyTxt))
    if printEyeball {
      printTensor2DEyeball("SeedVR2 dit.single body vid", swiftBodyVid, torchBodyVid)
      printTensor2DEyeball("SeedVR2 dit.single body txt", swiftBodyTxt, torchBodyTxt)
    }

    let torchAfterNorm = loadTensor(ditSingleOfficialProbe, "vid_after_norm")
    let torchAfterAda = loadTensor(ditSingleOfficialProbe, "vid_after_ada")
    let torchFullOutput = loadTensor(ditSingleOfficialProbe, "output")
    let tailOutputs = outputHead(
      inputs:
        chainVid,
      rematerializeOnDevice(
        graph, seedVR2OutputModTensor(emb: fullEmbCPU, stateDict: ditStateDict, offset: 0)))
    let swiftAfterNorm = copiedToCPU(tailOutputs[0].as(of: Float.self))
    let swiftAfterAda = copiedToCPU(tailOutputs[1].as(of: Float.self))
    let swiftFullOutput = copiedToCPU(tailOutputs[2].as(of: Float.self))
    print(
      "SeedVR2 dit.single tail after_norm max abs diff:",
      maxAbsDiff2DTensor(swiftAfterNorm, torchAfterNorm))
    print(
      "SeedVR2 dit.single tail after_ada max abs diff:",
      maxAbsDiff2DTensor(swiftAfterAda, torchAfterAda))
    print(
      "SeedVR2 dit.single full output max abs diff:",
      maxAbsDiff2DTensor(swiftFullOutput, torchFullOutput))
    print(
      "SeedVR2 dit.single full output global max rel diff:",
      maxGlobalRelativeDiff2DTensor(swiftFullOutput, torchFullOutput))
    if printEyeball {
      printTensor2DEyeball("SeedVR2 dit.single full output", swiftFullOutput, torchFullOutput)
    }
  }
  exit(0)
}

exit(0)
