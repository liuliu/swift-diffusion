import Foundation
import NNC

struct Flux3Checkpoints {
  let dit: String
  let vae: String
  let text: String
  var tokenizer: String { text + ".tokenizer.json" }

  init(environment: [String: String] = ProcessInfo.processInfo.environment) {
    dit = environment["FLUX3_DIT_CHECKPOINT"] ?? "/slow/Data/flux_3_image_klein_dit_bf16.ckpt"
    vae = environment["FLUX3_VAE_CHECKPOINT"] ?? "/slow/Data/flux_3_image_klein_vae_f32.ckpt"
    text =
      environment["FLUX3_TEXT_CHECKPOINT"] ?? "/slow/Data/flux_3_image_klein_text_model_f16.ckpt"
    precondition(Set([dit, vae, text]).count == 3, "Each component needs its own checkpoint file")
  }

  func path(for key: String) -> String {
    switch key {
    case "dit": return dit
    case "text_model": return text
    case "encoder", "decoder", "latent_mean", "latent_std": return vae
    default: preconditionFailure("Unknown checkpoint component: \(key)")
    }
  }
}

func flux3TokenInputs(_ tokens: [Int32], valid: Int) -> (Tensor<Int32>, Tensor<Int32>) {
  var ids = Tensor<Int32>(.CPU, format: .NHWC, shape: [tokens.count])
  var positions = Tensor<Int32>(.CPU, format: .NHWC, shape: [tokens.count])
  for i in tokens.indices {
    ids[i] = tokens[i]
    positions[i] = i < valid ? Int32(i) : 1
  }
  return (ids, positions)
}

func flux3IDs(text: Int, height: Int, width: Int) -> Tensor<Float> {
  var ids = Tensor<Float>(.CPU, .NC(text + height * width, 4))
  for i in 0..<(text + height * width) {
    for axis in 0..<4 { ids[i, axis] = 0 }
    if i < text {
      ids[i, 3] = Float(i)
    } else {
      ids[i, 1] = Float((i - text) / width) * 0.5
      ids[i, 2] = Float((i - text) % width) * 0.5
    }
  }
  return ids
}

func flux3Schedule(steps: Int) -> [Float] {
  precondition(steps > 0)
  return (0...steps).map {
    let t = 1 - Float($0) / Float(steps)
    return 6.93 * t / (1 + (6.93 - 1) * t)
  }
}

// SplitMix64 + Box-Muller provides deterministic Gaussian noise entirely in Swift.
// Its seed stream is independent of PyTorch's RNG; no Python-generated fixture is used.
func flux3Noise(seed: UInt64, count: Int) -> Tensor<Float> {
  var state = seed
  func uniform() -> Double {
    state &+= 0x9e37_79b9_7f4a_7c15
    var z = state
    z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
    z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
    z ^= z >> 31
    return (Double(z >> 11) + 0.5) / 9_007_199_254_740_992
  }
  var out = Tensor<Float>(.CPU, .HWC(1, count, 128))
  for i in stride(from: 0, to: count * 128, by: 2) {
    let radius = sqrt(-2 * log(uniform()))
    let angle = 2 * Double.pi * uniform()
    out[0, i / 128, i % 128] = Float(radius * cos(angle))
    out[0, (i + 1) / 128, (i + 1) % 128] = Float(radius * sin(angle))
  }
  return out
}

// Undo 2x2 patchification and latent normalization outside the VAE graph.
func flux3DecodeLatents(
  _ tokens: Tensor<Float>, mean: Tensor<Float>, std: Tensor<Float>, height: Int, width: Int
) -> Tensor<Float> {
  let h = height / 16
  let w = width / 16
  let packed = tokens.reshaped(.NC(h * w, 128))
  var out = Tensor<Float>(.CPU, .NCHW(1, 32, h * 2, w * 2))
  for y in 0..<h {
    for x in 0..<w {
      for c in 0..<32 {
        for dy in 0..<2 {
          for dx in 0..<2 {
            let k = c * 4 + dy * 2 + dx
            out[0, c, y * 2 + dy, x * 2 + dx] = packed[y * w + x, k] * std[k] + mean[k]
          }
        }
      }
    }
  }
  return out
}
