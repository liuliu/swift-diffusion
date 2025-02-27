import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let numpy = Python.import("numpy")
let torch = Python.import("torch")

let graph = DynamicGraph()

graph.openStore(
  "/home/liu/workspace/draw-things-community/skyreels_v1_hunyuan_i2v_f16.ckpt",
  // "/fast/Data/SD/flux_1_dev_f16.ckpt",
  flags: .readOnly
) { store in
  let keys = store.keys
  graph.openStore(
    "/home/liu/workspace/draw-things-community/skyreels_v1_hunyuan_i2v_q5p.ckpt",
    // "/fast/Data/SD/flux_1_dev_q5p.ckpt",
    flags: .readOnly
  ) { bench in
    graph.openStore(
      "/home/liu/workspace/draw-things-community/skyreels_v1_hunyuan_i2v_q5p_svd.ckpt",
      // "/home/liu/workspace/swift-diffusion/flux_1_dev_q5p_svd.ckpt",
      flags: .truncateWhenClose
    ) { writer in
      graph.openStore(
        "/home/liu/workspace/draw-things-community/skyreels_v1_hunyuan_i2v_q5p.ckpt",
        // "/fast/Data/SD/flux_1_dev_q5p.ckpt",
        flags: .readOnly
      ) {
        for key in keys {
          guard var codec = $0.codec(for: key) else { continue }
          guard codec == .q5p else {
            codec.subtract([.externalData, .jit, .externalOnDemand])
            guard let tensor = $0.read(key, codec: codec.union([.jit, .externalData])) else {
              continue
            }
            print("transwrite key \(key)")
            writer.write(key, tensor: tensor, codec: codec)
            continue
          }
          guard let tensor = (store.read(key).map { Tensor<Float32>(from: $0).toCPU() }) else {
            continue
          }
          print("key \(key)")
          let f32 = torch.from_numpy(tensor)
          /*
          let benchTensor =
            (bench.read(key, codec: [.q5p, .q8p]).map { Tensor<Float32>(from: $0).toCPU() })
          let qTensor =
            (writer.read(key, codec: [.q4p, .q5p, .q8p]).map { Tensor<Float32>(from: $0).toCPU() })!
          let upTensor = (writer.read("\(key)__up__").map { Tensor<Float32>(from: $0).toCPU() })!
          let downTensor =
            (writer.read("\(key)__down__").map { Tensor<Float32>(from: $0).toCPU() })!
          let b32 = torch.from_numpy(benchTensor)
          let q32 = torch.from_numpy(qTensor)
          let tup = torch.from_numpy(upTensor)
          let tdown = torch.from_numpy(downTensor)
          let qr = q32 + torch.matmul(tup, tdown)
          let qrdiff = (f32 - qr)
          let qdiff = (f32 - b32)
          print("\(key) \((qrdiff * qrdiff).sum()) \((qdiff * qdiff).sum())")
          continue
          */
          let (du, ds, dv) = torch.linalg.svd(f32.double()).tuple3
          let up = torch.matmul(du[..., 0..<32], torch.diag(ds[0..<32])).half().float()  // truncate to half then float.
          let down = dv[0..<32, ...].half().float()
          let restr = torch.matmul(up, down)
          let rdiff = (f32 - restr)
          writer.write(
            "\(key)__up__", tensor: Tensor<Float16>(from: try! Tensor<Float>(numpy: up.numpy())))
          writer.write(
            "\(key)__down__", tensor: Tensor<Float16>(from: try! Tensor<Float>(numpy: down.numpy()))
          )
          writer.write(
            key, tensor: Tensor<Float16>(from: try! Tensor<Float>(numpy: rdiff.numpy())),
            codec: [.q5p, .ezm7])
        }
      }
    }
  }
}
