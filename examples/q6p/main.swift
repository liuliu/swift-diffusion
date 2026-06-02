import Foundation
import NNC

let graph = DynamicGraph()

private let imatrixPath = "/home/liu/workspace/s4nnc/qwen"  // 36_27b_ud_q4_k_xl_tensor_imatrix.csv"
private let inputPath = "/fast/Data/hidream_o1_dev_2604_f16.ckpt"
private let outputPath = "/fast/Data/hidream_o1_dev_2604_q6p.ckpt"

private struct QuantizationEntry {
  let format: String
  let imatrix: Tensor<Float>?
}

private func loadImatrixCSV(_ path: String) -> [String: QuantizationEntry] {
  let csv: String
  do {
    csv = try String(contentsOfFile: path, encoding: .utf8)
  } catch {
    print("failed to load imatrix csv \(path): \(error)")
    return [:]
  }
  var entries = [String: QuantizationEntry]()
  for line in csv.components(separatedBy: .newlines).dropFirst() where !line.isEmpty {
    let fields = line.split(separator: ",", maxSplits: 2, omittingEmptySubsequences: false)
    guard fields.count >= 2 else { continue }
    let key = String(fields[0])
    let format = String(fields[1]).trimmingCharacters(in: .whitespacesAndNewlines)
    let imatrix: Tensor<Float>?
    if fields.count == 3,
      let data = Data(
        base64Encoded: String(fields[2]).trimmingCharacters(in: .whitespacesAndNewlines))
    {
      let floats = data.withUnsafeBytes { bytes -> [Float] in
        let buffer = bytes.bindMemory(to: Float.self)
        return Array(buffer)
      }
      imatrix =
        floats.isEmpty
        ? nil
        : Tensor<Float>(floats, kind: .CPU, format: .NCHW, shape: [floats.count])
    } else {
      imatrix = nil
    }
    entries[key] = QuantizationEntry(format: format, imatrix: imatrix)
  }
  return entries
}

private let quantizationTable = loadImatrixCSV(imatrixPath)

private func strippedStoreName(_ key: String) -> String? {
  let prefix = "__text_model__[t-"
  guard key.hasPrefix(prefix), key.hasSuffix("]") else { return nil }
  var name = String(key.dropFirst(prefix.count).dropLast())
  for suffix in ["-0-0", "-0-1"] {
    if name.hasSuffix(suffix) {
      name.removeLast(suffix.count)
      break
    }
  }
  return name
}

private func csvQuantizationKey(for key: String) -> String? {
  guard let name = strippedStoreName(key) else { return nil }
  switch name {
  case "lm_head":
    return "output.weight"
  case "model.language_model.embed_tokens":
    return "token_embd.weight"
  case "model.language_model.norm":
    return "output_norm.weight"
  default:
    break
  }

  let layerPrefix = "model.language_model.layers."
  guard name.hasPrefix(layerPrefix) else { return nil }
  let remainder = name.dropFirst(layerPrefix.count)
  guard let dot = remainder.firstIndex(of: ".") else { return nil }
  let layer = remainder[..<dot]
  let layerKey = remainder[remainder.index(after: dot)...]
  let blockPrefix = "blk.\(layer)."

  switch layerKey {
  case "input_layernorm":
    return "\(blockPrefix)attn_norm.weight"
  case "post_attention_layernorm":
    return "\(blockPrefix)post_attention_norm.weight"
  case "mlp.down_proj":
    return "\(blockPrefix)ffn_down.weight"
  case "mlp.gate_proj":
    return "\(blockPrefix)ffn_gate.weight"
  case "mlp.up_proj":
    return "\(blockPrefix)ffn_up.weight"
  case "linear_attn.A_log":
    return "\(blockPrefix)ssm_a"
  case "linear_attn.conv1d.weight":
    return "\(blockPrefix)ssm_conv1d.weight"
  case "linear_attn.dt_bias":
    return "\(blockPrefix)ssm_dt.bias"
  case "linear_attn.in_proj_a":
    return "\(blockPrefix)ssm_alpha.weight"
  case "linear_attn.in_proj_b":
    return "\(blockPrefix)ssm_beta.weight"
  case "linear_attn.in_proj_qkv":
    return "\(blockPrefix)attn_qkv.weight"
  case "linear_attn.in_proj_z":
    return "\(blockPrefix)attn_gate.weight"
  case "linear_attn.norm":
    return "\(blockPrefix)ssm_norm.weight"
  case "linear_attn.out_proj":
    return "\(blockPrefix)ssm_out.weight"
  case "self_attn.k_norm":
    return "\(blockPrefix)attn_k_norm.weight"
  case "self_attn.k_proj":
    return "\(blockPrefix)attn_k.weight"
  case "self_attn.o_proj":
    return "\(blockPrefix)attn_output.weight"
  case "self_attn.q_gate_proj":
    return "\(blockPrefix)attn_q.weight"
  case "self_attn.q_norm":
    return "\(blockPrefix)attn_q_norm.weight"
  case "self_attn.q_proj":
    return "\(blockPrefix)attn_q.weight"
  case "self_attn.v_proj":
    return "\(blockPrefix)attn_v.weight"
  default:
    return nil
  }
}

private func codec(for format: String) -> DynamicGraph.Store.Codec? {
  switch format {
  case "Q5_K":
    return [.i8x(.q5k), .ezm7]
  case "Q4_K":
    return [.i8x(.q4k), .ezm7]
  case "IQ4_XS":
    return [.i8x(.q4k), .ezm7]
  case "Q3_K":
    return [.i8x(.q3k), .ezm7]
  case "Q2_K":
    return [.i8x(.q2k), .ezm7]
  case "IQ2_S":
    return [.i8x(.iq2s), .ezm7]
  case "IQ2_XS":
    return [.i8x(.iq2xs), .ezm7]
  case "IQ2_XXS":
    return [.i8x(.iq2xxs), .ezm7]
  case "IQ3_S":
    return [.i8x(.iq3s), .ezm7]
  case "IQ3_XXS":
    return [.i8x(.iq3xxs), .ezm7]
  default:
    return nil
  }
}

graph.openStore(
  inputPath, flags: [.readOnly]
) { store in
  let keys = store.keys
  graph.openStore(
    outputPath,
    flags: .truncateWhenClose
  ) {
    for key in keys {
      guard let anyTensor = store.read(key) else { continue }
      guard anyTensor.dataType != .Float32 else {
        // If it is already in FP32, skip transcode to FP16. Only useful for UMT5 XXL / Wan v2.1 models.
        let tensor = Tensor<Float16>(from: anyTensor).toCPU()
        // let tensor = Tensor<Float32>(anyTensor).toCPU()
        $0.write(key, tensor: tensor)
        continue
      }
      let tensor: AnyTensor
      if anyTensor.dataType == .BFloat16 {
        tensor = Tensor<BFloat16>(from: anyTensor).toCPU()
      } else {
        tensor = Tensor<Float16>(from: anyTensor).toCPU()
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
      if key.contains("embedder") || key.contains("pos_embed") || key.contains("-linear-")
        || key.contains("-linear_final-")
        || key.contains("-proj_out-") || key.contains("-audio_proj_out-")
        || key.contains("_embeddings") || key.contains("register_tokens")
        || (key.contains("refiner_") && !key.contains("noise_refiner_")
          && !key.contains("context_refiner_"))
        || key.contains("position_embedding") || key.contains("-shared-")
        || key.contains("_pad_token")  // Z-Image related.
        || key.contains("_registers") || key.contains("_connector") || key.contains("_extractor")  // LTX-2 related.
        || key.contains("token_embedding")  // Anima related.
        || key.contains("positive_embedding")  // SeedVR2 related.
        || key.contains("negative_embedding")  // SeedVR2 related.
        || key.contains("embed_tokens")  // Qwen 3.5 related.
        || key.contains("patch_embed")  // Qwen 3.5 related.
        || key.contains("linear_attn.conv1d.")  // Qwen 3.5 related.
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
      var imatrix: Tensor<Float>? = nil
      if let csvKey = csvQuantizationKey(for: key), let entry = quantizationTable[csvKey] {
        if let csvCodec = codec(for: entry.format) {
          $0.write(key, tensor: tensor, codec: csvCodec, imatrix: entry.imatrix)
          continue
        }
        imatrix = entry.imatrix
      }
      if n > 1 && (key.contains("ada_ln") || key.contains("adaln_")) {
        $0.write(key, tensor: tensor, codec: [.q8p, .ezm7], imatrix: imatrix)
        continue
      }
      if (shape.count == 2 || shape.count == 3) && n > 1 {
        if shape.count == 2 {
          $0.write(key, tensor: tensor, codec: [.q6p, .ezm7], imatrix: imatrix)
        } else {
          $0.write(key, tensor: tensor, codec: [.q6p, .ezm7], imatrix: imatrix)
        }
      } else if shape.count == 4 && n > 1 {
        $0.write(key, tensor: tensor, codec: [.q8p, .ezm7], imatrix: imatrix)
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
