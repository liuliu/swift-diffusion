import NNC

let graph = DynamicGraph()

graph.openStore(
  "/home/liu/workspace/swift-diffusion/llava_llama_3_8b_v1.1_f16.ckpt",
  flags: .truncateWhenClose
) { store in
  let keys = store.keys
  graph.openStore(
    "/home/liu/workspace/swift-diffusion/llava_llama_3_8b_v1.1_q6p.ckpt",
    flags: .truncateWhenClose
  ) {
    for key in keys {
      guard let tensor = (store.read(key).map { Tensor<Float16>(from: $0).toCPU() }) else {
        continue
      }
      if key.contains("__stage_c_fixed__") && (key.contains("key") || key.contains("value")) {
        continue
      }
      if key.contains("text_emb") || key.contains("effnet") || key.contains("previewer") {
        $0.write(key, tensor: tensor)
        continue
      }
      let shape = tensor.shape
      print("write \(key) \(tensor)")
      if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-")  // || key.contains("ada_ln")
        || key.contains("_embeddings") || key.contains("register_tokens")
        || key.contains("refiner_")
      {
        $0.write(key, tensor: tensor)
        continue
      }
      if key.contains("shift_table") || key.contains("t_block") || key.contains("zero_conv") {
        $0.write(key, tensor: tensor)
        continue
      }
      if key.contains("norm") {
        $0.write(key, tensor: tensor, codec: [.ezm7])
        continue
      }
      var n = 0
      for i in 0..<shape.count {
        if shape[i] > 1 {
          n += 1
        }
      }
      /*
      if n > 1 && key.contains("_proj-") || key.contains("-o-") {
        $0.write(key, tensor: tensor, codec: [.q8p, .ezm7])
        continue
      }
      */
      /*
      if n > 1 {
        if key.contains("fc") {
          $0.write(key, tensor: tensor, codec: [.q8p, .ezm7])
        } else {
          $0.write(key, tensor: tensor, codec: [.q8p, .ezm7])
        }
      } else {
        $0.write(key, tensor: tensor, codec: [.ezm7])
      }
      */
      /*
      if key.contains("relative_position_embedding") || key.contains("shared") {
        $0.write(key, tensor: tensor)
        continue
      }
      if key.contains("norm") {
        $0.write(key, tensor: tensor, codec: [.ezm7])
        continue
      } else {
        if key.contains("w") {
          $0.write(key, tensor: tensor, codec: [.q4p, .ezm7])
        } else {
          $0.write(key, tensor: tensor, codec: [.q6p, .ezm7])
        }
        continue
      }
      */
      if n > 1 && key.contains("ada_ln") {
        $0.write(key, tensor: tensor, codec: [.q8p, .ezm7])
        continue
      }
      if shape.count == 2 && n > 1 {
        $0.write(key, tensor: tensor, codec: [.q6p, .ezm7])
      } else if shape.count == 4 && n > 1 {
        $0.write(key, tensor: tensor, codec: [.q8p, .ezm7])
      } else {
        $0.write(key, tensor: tensor, codec: [.ezm7])
      }
      /*
      if keys.contains("vision_proj") {
        if shape.count == 2 && n > 1 {
          $0.write(key, tensor: tensor, codec: [.q8p, .ezm7])
        } else if shape.count == 4 && n > 1 {
          $0.write(key, tensor: tensor, codec: [.q8p, .ezm7])
        } else {
          $0.write(key, tensor: tensor, codec: [.ezm7])
        }
      } else {
        if shape.count == 2 && n > 1 {
          $0.write(key, tensor: tensor, codec: [.q6p, .ezm7])
        } else if shape.count == 4 && n > 1 {
          $0.write(key, tensor: tensor, codec: [.q6p, .ezm7])
        } else {
          $0.write(key, tensor: tensor, codec: [.ezm7])
        }
      }
      */
    }
  }
}
