import Foundation
import NNC

var df = DataFrame(
  fromCSV: "/home/liu/workspace/swift-diffusion/regression.csv", automaticUseHeader: false)!

let graph = DynamicGraph()

let linear = Dense(count: 3)

var adamOptimizer = AdamOptimizer(graph, rate: 0.01)
adamOptimizer.parameters = [linear.parameters]

df["x"] = df["0", "1", "2", "3"].map {
  (c0: String, c1: String, c2: String, c3: String) -> Tensor<Float> in
  return Tensor<Float>([Float(c0)!, Float(c1)!, Float(c2)!, Float(c3)!], .CPU, .C(4))
}

df["y"] = df["4", "5", "6"].map { (c0: String, c1: String, c2: String) -> Tensor<Float> in
  return Tensor<Float>([Float(c0)!, Float(c1)!, Float(c2)!], .CPU, .C(3))
}

var batchedDf = df["x", "y"].combine(size: 128, repeating: 1)

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
  print("epoch: \(epoch), loss: \(totalLoss / Double(batchedDf.count))")
}

let weight = graph.variable(.CPU, .NC(3, 4), of: Float.self)
let bias = graph.variable(.CPU, .C(3), of: Float.self)
linear.weight.copy(to: weight)
linear.bias.copy(to: bias)
debugPrint(weight)
debugPrint(bias)
