import Foundation
import NNC

enum Q21VAEConfig {
  static let encoderChannels = [96, 192, 384, 768, 768]
  static let decoderChannels = [1_152, 1_152, 576, 288, 144]
  static let latentChannels = 64
  static let pixelChannels = 4
}

struct Q21VAEBinding {
  let model: Model
  let key: String
  var norm = false
}

// Single-image specialization: temporal resampling is inactive. FP16 uses
// temporal-kernel-one convolutions for cuDNN FP32 accumulation.
func Q21VAE(
  encoder: Bool, height: Int, width: Int, device: Int = 1, precision: DataType = .Float32
)
  -> (Model, [Q21VAEBinding])
{
  precondition(precision == .Float32 || precision == .Float16)
  let input = Input()
  var bindings = [Q21VAEBinding]()
  var h = height
  var w = width
  func conv(
    _ key: String, _ channels: Int, from: Int, kernel: Int = 3,
    down: Bool = false, up: Bool = false
  ) -> (Model.IO) -> Model.IO {
    let border = down ? 0 : kernel / 2
    let stride = down ? 2 : 1
    let spatialScale = up ? 2 : 1
    let inputHeight = h * spatialScale + (down ? 1 : 0)
    let inputWidth = w * spatialScale + (down ? 1 : 0)
    let outputHeight = h * spatialScale / stride
    let outputWidth = w * spatialScale / stride
    let half = precision == .Float16
    let model = Convolution(
      groups: 1, filters: channels, filterSize: half ? [1, kernel, kernel] : [kernel, kernel],
      hint: Hint(
        stride: half ? [1, stride, stride] : [stride, stride],
        border: Hint.Border(
          begin: half ? [0, border, border] : [border, border],
          end: half ? [0, border, border] : [border, border])),
      name: key.replacingOccurrences(of: ".", with: "_"))
    bindings.append(.init(model: model, key: key))
    return { x in
      if half {
        return model(x.reshaped([1, from, 1, inputHeight, inputWidth]))
          .reshaped([1, channels, outputHeight, outputWidth])
      }
      return model(x)
    }
  }
  func channelNorm(_ x: Model.IO, _ key: String, channels: Int) -> Model.IO {
    let name = key.replacingOccurrences(of: ".", with: "_")
    let gamma: Model
    if precision == .Float16 {
      gamma = Parameter<Float16>(.GPU(device), .NCHW(1, channels, 1, 1), name: name)
    } else {
      gamma = Parameter<Float>(.GPU(device), .NCHW(1, channels, 1, 1), name: name)
    }
    bindings.append(.init(model: gamma, key: key, norm: true))
    // Normalize in FP32, then round before applying scale/gamma, as in PyTorch.
    let normalizedInput = x.to(.Float32)
    let sum = (normalizedInput .* normalizedInput).reduced(.sum, axis: [1])
    let inverse = sum.clamped(1e-24...).pow(-0.5)
    return (normalizedInput .* inverse).to(precision) * Float(channels).squareRoot() .* gamma()
  }
  func silu(_ x: Model.IO) -> Model.IO { x.to(.Float32).swish().to(precision) }
  func residual(_ x: Model.IO, _ key: String, from: Int, to: Int) -> Model.IO {
    let shortcut = from == to ? x : conv(key + ".conv_shortcut", to, from: from, kernel: 1)(x)
    let a = conv(key + ".conv1", to, from: from)(
      silu(channelNorm(x, key + ".norm1", channels: from)))
    let b = conv(key + ".conv2", to, from: to)(silu(channelNorm(a, key + ".norm2", channels: to)))
    return b + shortcut
  }
  func attention(_ x: Model.IO, _ key: String, channels: Int) -> Model.IO {
    let normalized = channelNorm(x, key + ".norm", channels: channels)
    let qkv = conv(key + ".to_qkv", 3 * channels, from: channels, kernel: 1)(normalized)
      .to(.Float32).reshaped([3 * channels, h * w])
    let chunks = (0..<3).map { i in
      qkv.reshaped([channels, h * w], offset: [i * channels, 0], strides: [h * w, 1]).contiguous()
    }
    let scores =
      Matmul(transposeA: (0, 1))(chunks[0], chunks[1]) * (1 / Float(channels).squareRoot())
    let probabilities = scores.softmax()
    let attended = Matmul(transposeB: (0, 1))(chunks[2], probabilities).reshaped([
      1, channels, h, w,
    ])
    return x + conv(key + ".proj", channels, from: channels, kernel: 1)(attended.to(precision))
  }
  func middle(_ x: Model.IO, _ key: String, channels: Int) -> Model.IO {
    let a = residual(x, key + ".resnets.0", from: channels, to: channels)
    let b = attention(a, key + ".attentions.0", channels: channels)
    return residual(b, key + ".resnets.1", from: channels, to: channels)
  }
  func downShortcut(_ x: Model.IO, from: Int, to: Int, temporal: Int, spatial: Int) -> Model.IO {
    let nh = h / spatial
    let nw = w / spatial
    var folded = x.reshaped([from, nh, spatial, nw, spatial]).permuted(0, 2, 4, 1, 3).contiguous()
      .reshaped([from, 1, spatial * spatial, nh * nw])
    if temporal == 2 { folded = Functional.concat(axis: 1, folded * 0, folded) }
    let group = from * temporal * spatial * spatial / to
    return folded.reshaped([1, to, group, nh, nw]).reduced(.mean, axis: [2]).reshaped([
      1, to, nh, nw,
    ])
  }
  func upShortcut(_ x: Model.IO, from: Int, to: Int, temporal: Int) -> Model.IO {
    let repeats = to * temporal * 4 / from
    let row = x.reshaped([from, 1, h, w])
    var duplicated = row
    if repeats > 1 {
      for _ in 1..<repeats { duplicated = Functional.concat(axis: 1, duplicated, row) }
    }
    // First-image semantics: keep only the last of the duplicated temporal frames.
    let selected = duplicated.reshaped([to, temporal, 4, h * w])
      .reshaped(
        [to, 1, 4, h * w], offset: [0, temporal - 1, 0, 0],
        strides: [temporal * 4 * h * w, 4 * h * w, h * w, 1]
      ).contiguous()
    return selected.reshaped([to, 2, 2, h, w]).permuted(0, 3, 1, 4, 2).contiguous().reshaped([
      1, to, h * 2, w * 2,
    ])
  }
  var x: Model.IO
  if encoder {
    x = conv("encoder.conv_in", 96, from: 4)(input)
    var channels = 96
    for (stage, next) in Q21VAEConfig.encoderChannels.enumerated() {
      let key = "encoder.down_blocks.\(stage)"
      let down = stage < 4
      let shortcut = downShortcut(
        x, from: channels, to: next, temporal: stage > 0 && stage < 4 ? 2 : 1, spatial: down ? 2 : 1
      )
      for block in 0..<2 {
        x = residual(x, key + ".resnets.\(block)", from: block == 0 ? channels : next, to: next)
      }
      if down {
        x = conv(key + ".downsampler.resample.1", next, from: next, down: true)(
          x.padded(.zero, begin: [0, 0, 0, 0], end: [0, 0, 1, 1]))
      }
      x = x + shortcut
      channels = next
      if down {
        h /= 2
        w /= 2
      }
    }
    x = middle(x, "encoder.mid_block", channels: channels)
    x = conv("encoder.conv_out", 128, from: channels)(
      silu(channelNorm(x, "encoder.norm_out", channels: channels)))
    x = conv("quant_conv", 128, from: 128, kernel: 1)(x)
  } else {
    x = conv("post_quant_conv", 64, from: 64, kernel: 1)(input)
    x = conv("decoder.conv_in", 1_152, from: 64)(x)
    x = middle(x, "decoder.mid_block", channels: 1_152)
    var channels = 1_152
    for (stage, next) in Q21VAEConfig.decoderChannels.enumerated() {
      let key = "decoder.up_blocks.\(stage)"
      let up = stage < 4
      let shortcut = up ? upShortcut(x, from: channels, to: next, temporal: stage < 3 ? 2 : 1) : nil
      for block in 0..<3 {
        x = residual(x, key + ".resnets.\(block)", from: block == 0 ? channels : next, to: next)
      }
      if up {
        x = conv(key + ".upsampler.resample.1", next, from: next, up: true)(
          Upsample(.nearest, widthScale: 2, heightScale: 2)(x))
        x = x + shortcut!
        h *= 2
        w *= 2
      }
      channels = next
    }
    x = conv("decoder.conv_out", 4, from: channels)(
      silu(channelNorm(x, "decoder.norm_out", channels: channels)))
  }
  return (Model([input], [x]), bindings)
}
