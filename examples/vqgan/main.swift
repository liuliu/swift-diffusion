import Foundation
import NNC

var df = DataFrame(
  fromCSV: "/home/liu/workspace/swift-diffusion/sdxl_vae_regression.csv", automaticUseHeader: false)!

let graph = DynamicGraph()

let linear = Dense(count: 3)

var adamOptimizer = AdamOptimizer(graph, rate: 0.01)
adamOptimizer.parameters = [linear.parameters]

let scaleFactor: Float = 0.13025

df["x"] = df["0", "1", "2", "3"].map {
  (c0: String, c1: String, c2: String, c3: String) -> Tensor<Float> in
  return Tensor<Float>(
    [
      Float(c0)! * scaleFactor, Float(c1)! * scaleFactor, Float(c2)! * scaleFactor,
      Float(c3)! * scaleFactor,
    ], .CPU, .C(4))
}

func f(x: Float) -> Float {
  if x >= 0.0031308 {
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055
  } else {
    return 12.92 * x
  }
}

func linear_srgb_to_oklab(r: Float, g: Float, b: Float) -> (Float, Float, Float) {
  let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
  let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
  let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

  let l_ = cbrtf(l)
  let m_ = cbrtf(m)
  let s_ = cbrtf(s)

  return (
    0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
  )
}

df["y"] = df["4", "5", "6"].map { (c0: String, c1: String, c2: String) -> Tensor<Float> in
  return Tensor<Float>([Float(c0)!, Float(c1)!, Float(c2)!], .CPU, .C(3))
  // let r = f(x: Float(c0)! / 255)
  // let g = f(x: Float(c1)! / 255)
  // let b = f(x: Float(c2)! / 255)
  // let (okl, oka, okb) = linear_srgb_to_oklab(r: r, g: g, b: b)
  // return Tensor<Float>([okl, oka, okb], .CPU, .C(3))
}

var batchedDf = df["x", "y"].combine(size: 128, repeating: 1)

let weight = graph.variable(.CPU, .NC(3, 4), of: Float.self)
let bias = graph.variable(.CPU, .C(3), of: Float.self)

for epoch in 0..<10 {
  batchedDf.shuffle()
  var totalLoss: Double = 0
  for (i, batch) in batchedDf["x", "y"].enumerated() {
    let x = graph.variable((batch[0] as! Tensor<Float>).toGPU(0))
    let target = graph.variable((batch[1] as! Tensor<Float>).toGPU(0))
    let y = linear(inputs: x)[0].as(of: Float.self)
    let mseLoss = MSELoss()
    let loss = mseLoss(y, target: target)[0].as(of: Float.self)
    loss.backward(to: x)
    adamOptimizer.step()
    let value = loss.reduced(.mean, axis: [0]).toCPU()[0, 0]
    totalLoss += Double(value)
    if i % 1000 == 999 {
      print("epoch: \(epoch), \(i), loss: \(totalLoss / Double(i + 1))")
    }
  }
  if epoch == 2 {
    adamOptimizer.rate = 0.001
  }
  if epoch == 5 {
    adamOptimizer.rate = 0.0001
  }
  print("epoch: \(epoch), loss: \(totalLoss / Double(batchedDf.count))")
  linear.weight.copy(to: weight)
  linear.bias.copy(to: bias)
  debugPrint(weight)
  debugPrint(bias)
}

linear.weight.copy(to: weight)
linear.bias.copy(to: bias)
debugPrint(weight)
debugPrint(bias)
