import NNC

let graph = DynamicGraph()

graph.openStore(
  "/home/liu/workspace/swift-diffusion/fooocus_inpaint_sd_xl_v2.6_f16.ckpt",
  flags: .truncateWhenClose
) { store in
  let keys = store.keys
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/fooocus_inpaint_sd_xl_v2.6_q6p_q8p.ckpt",
    flags: .truncateWhenClose
  ) {
    for key in keys {
      guard let tensor = store.read(key) else { continue }
      if key.contains("visual_proj") {
        $0.write(key, tensor: tensor)
        continue
      }
      let shape = tensor.shape
      print("write \(key) \(tensor)")
      var n = 0
      for i in 0..<shape.count {
        if shape[i] > 1 {
          n += 1
        }
      }
      if shape.count == 2 && n > 1 {
        $0.write(key, tensor: tensor, codec: [.q6p, .ezm7])
      } else if shape.count == 4 && n > 1 {
        $0.write(key, tensor: tensor, codec: [.q8p, .ezm7])
      } else {
        $0.write(key, tensor: tensor, codec: [.ezm7])
      }
    }
  }
}
