import Diffusion
import Foundation
import NNC
import PNG

let deviceID = Int(ProcessInfo.processInfo.environment["MINIMAX_H3_DEVICE"] ?? "0") ?? 0
let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

struct H3RuntimeOptions {
  var prompt =
    "A cinematic wide shot of ocean waves at sunset, with synchronized surf and wind sounds."
  var height = H3Config.defaultHeight
  var width = H3Config.defaultWidth
  var frames = H3Config.defaultFrames
  var steps = 50
  var seed: UInt64 = 42
  var output = "minimax_h3_output"
  var dryRun = false
  var textOnly = false
}

func parseOptions() -> H3RuntimeOptions {
  var options = H3RuntimeOptions()
  var index = 1
  let arguments = CommandLine.arguments
  while index < arguments.count {
    let argument = arguments[index]
    func value() -> String {
      precondition(index + 1 < arguments.count, "Missing value after \(argument)")
      index += 1
      return arguments[index]
    }
    switch argument {
    case "--prompt": options.prompt = value()
    case "--height": options.height = Int(value())!
    case "--width": options.width = Int(value())!
    case "--frames": options.frames = Int(value())!
    case "--steps": options.steps = Int(value())!
    case "--seed": options.seed = UInt64(value())!
    case "--output": options.output = value()
    case "--dry-run": options.dryRun = true
    case "--text-only": options.textOnly = true
    default: preconditionFailure("Unknown argument: \(argument)")
    }
    index += 1
  }
  precondition(options.height % 32 == 0 && options.width % 32 == 0)
  precondition(options.steps >= 2)
  options.frames = alignFrameCount(options.frames)
  precondition(options.frames >= 124 && options.frames <= 364, "H3 supports 5-15 second clips")
  return options
}

let options = parseOptions()
let environment = ProcessInfo.processInfo.environment
let textStore =
  environment["MINIMAX_H3_QWEN_STORE"]
  ?? "/slow/Data/minimax_h3_qwen_3_vl_f16.ckpt"
let ditStore =
  environment["MINIMAX_H3_DIT_STORE"]
  ?? "/slow/Data/minimax_h3_dit_f32.ckpt"
let vaeStore =
  environment["MINIMAX_H3_VAE_STORE"]
  ?? "/slow/Data/minimax_h3_vae_f32.ckpt"

let specialTokens: [String: Int32] = [
  "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
  "<|object_ref_start|>": 151646, "<|object_ref_end|>": 151647,
  "<|box_start|>": 151648, "<|box_end|>": 151649, "<|quad_start|>": 151650,
  "<|quad_end|>": 151651, "<|vision_start|>": 151652, "<|vision_end|>": 151653,
  "<|vision_pad|>": 151654, "<|image_pad|>": 151655, "<|video_pad|>": 151656,
  "<tool_call>": 151657, "</tool_call>": 151658, "<|fim_prefix|>": 151659,
  "<|fim_middle|>": 151660, "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662,
  "<|repo_name|>": 151663, "<|file_sep|>": 151664, "<tool_response>": 151665,
  "</tool_response>": 151666, "<think>": 151667, "</think>": 151668,
]

func resourcePath(_ name: String) -> String {
  let sourceRelative = "examples/minimax_h3_assets/\(name)"
  if FileManager.default.fileExists(atPath: sourceRelative) { return sourceRelative }
  let executable = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
  let candidates = [
    executable.appendingPathComponent("minimax_h3_assets/\(name)").path,
    executable.appendingPathComponent(
      "minimax_h3.runfiles/_main/examples/minimax_h3_assets/\(name)"
    ).path,
  ]
  return candidates.first(where: FileManager.default.fileExists(atPath:)) ?? sourceRelative
}

func qwenRotary(sequenceLength: Int) -> Tensor<Float16> {
  var rotary = Tensor<Float16>(
    .CPU, .NHWC(1, sequenceLength, 1, H3QwenConfig.headDim))
  for row in 0..<sequenceLength {
    for index in 0..<(H3QwenConfig.headDim / 2) {
      let angle =
        Double(row)
        * pow(
          H3QwenConfig.ropeTheta,
          -2 * Double(index) / Double(H3QwenConfig.headDim))
      rotary[0, row, 0, 2 * index] = Float16(cos(angle))
      rotary[0, row, 0, 2 * index + 1] = Float16(sin(angle))
    }
  }
  return rotary
}

func timestepFrequencies(_ timesteps: [Float]) -> Tensor<Float> {
  let half = H3Config.timestepFrequencySize / 2
  var frequencies = Tensor<Float>(.CPU, .WC(timesteps.count, H3Config.timestepFrequencySize))
  for row in 0..<timesteps.count {
    for index in 0..<half {
      let inverse = exp(-log(10_000.0) * Float(index) / Float(half))
      let angle = timesteps[row] * inverse
      frequencies[row, index] = cos(angle)
      frequencies[row, index + half] = sin(angle)
    }
  }
  return frequencies
}

func shiftedSigmas(points: Int, shift: Float) -> [Float] {
  (0..<points).map { index in
    let base = 1 - Float(index) / Float(points - 1)
    return shift * base / (1 + (shift - 1) * base)
  }
}

// Load and move one parameter at a time. This matches the Flux.2 converter pattern and avoids
// ever requiring the complete Qwen or DiT checkpoint to fit in device memory first.
func loadUnified(_ key: String, model: Model, storePath: String) {
  graph.openStore(storePath, flags: [.readOnly]) { store in
    let count = model.parameters.count
    for index in 0..<model.parameters.count {
      let parameter = model.parameters[index]
      let tensorKey = "__\(key)__[\(parameter.name)]"
      guard let tensor = store.read(tensorKey, kind: .CPU) else {
        preconditionFailure("Missing checkpoint parameter \(tensorKey)")
      }
      parameter.copy(from: tensor)
      parameter.to(.unifiedMemory)
      if (index + 1) % 100 == 0 || index + 1 == count {
        print("MiniMax-H3 loaded \(key) parameter \(index + 1)/\(count)")
      }
    }
  }
}

func encodePrompt(_ prompt: String) -> Tensor<BlockFloat> {
  let tokenizer = TiktokenTokenizer(
    vocabulary: resourcePath("vocab.json"), merges: resourcePath("merges.txt"),
    specialTokens: specialTokens)
  let tokens = tokenizer.tokenize(text: prompt, addSpecialTokens: false).0
  precondition(!tokens.isEmpty, "Prompt tokenized to an empty sequence")
  print("MiniMax-H3 prompt tokens:", tokens.count)
  return graph.withNoGrad {
    var tokenCPU = Tensor<Int32>(.CPU, .C(tokens.count))
    for index in tokens.indices { tokenCPU[index] = tokens[index] }
    let tokenTensor = graph.variable(tokenCPU.toGPU(deviceID))
    let rotary = graph.variable(qwenRotary(sequenceLength: tokens.count).toGPU(deviceID))
    let (textModel, _) = H3QwenExportModel(sequenceLength: tokens.count)
    textModel.maxConcurrency = .limit(1)
    textModel.compile(inputs: tokenTensor, rotary)
    loadUnified("text_model", model: textModel, storePath: textStore)
    let output = textModel(inputs: tokenTensor, rotary)[0].as(of: Float16.self).rawValue.toCPU()
      .reshaped(.WC(tokens.count, H3QwenConfig.hiddenSize))
    return Tensor<BlockFloat>(from: output)
  }
}

func unpatchVideoRows(
  _ rows: Tensor<Float>, frames: Int, height: Int, width: Int
) -> Tensor<Float> {
  let patchHeight = height / 2
  let patchWidth = width / 2
  precondition(rows.shape == [frames * patchHeight * patchWidth, H3Config.videoPatchSize])
  var latents = Tensor<Float>(.CPU, format: .NCHW, shape: [24, frames, height, width])
  rows.withUnsafeBytes { sourceBytes in
    latents.withUnsafeMutableBytes { destinationBytes in
      let source = sourceBytes.baseAddress!.assumingMemoryBound(to: Float.self)
      let destination = destinationBytes.baseAddress!.assumingMemoryBound(to: Float.self)
      for frame in 0..<frames {
        for y in 0..<patchHeight {
          for x in 0..<patchWidth {
            let row = (frame * patchHeight + y) * patchWidth + x
            for channel in 0..<24 {
              for patchY in 0..<2 {
                for patchX in 0..<2 {
                  let feature = channel * 4 + patchY * 2 + patchX
                  let target =
                    ((channel * frames + frame) * height + y * 2 + patchY) * width
                    + x * 2 + patchX
                  destination[target] = source[row * H3Config.videoPatchSize + feature]
                }
              }
            }
          }
        }
      }
    }
  }
  return latents
}

func runDenoiser(text: Tensor<BlockFloat>) -> (Tensor<Float>, Tensor<Float>) {
  let latentFrames = videoLatentFrameCount(options.frames)
  let latentHeight = options.height / H3Config.videoSpatialCompression
  let latentWidth = options.width / H3Config.videoSpatialCompression
  let audioLatents = audioLatentFrameCount(options.frames)
  let layout = makeT2VALayout(
    textTokenCount: text.shape[0], latentFrames: latentFrames, latentHeight: latentHeight,
    latentWidth: latentWidth, audioLatents: audioLatents)
  let videoLength = layout.videoRange.count
  let audioLength = layout.audioRange.count
  print("MiniMax-H3 packed text/audio/video rows:", text.shape[0], audioLength, videoLength)
  return graph.withNoGrad {
    DynamicGraph.setSeed(UInt32(truncatingIfNeeded: options.seed))
    var video = graph.variable(
      .GPU(deviceID), .HWC(1, videoLength, H3Config.videoPatchSize), of: Float.self)
    var audio = graph.variable(
      .GPU(deviceID), .HWC(1, audioLength, H3Config.audioChannels), of: Float.self)
    video.randn()
    audio.randn()
    let textTensor = graph.variable(
      text.reshaped(.HWC(1, text.shape[0], text.shape[1])).toGPU(deviceID))
    let rotary = graph.variable(h3RotaryTensor(positionIDs: layout.positionIDs).toGPU(deviceID))
    var adalnCPU = Tensor<BlockFloat>(.CPU, .WC(layout.sequenceLength, 6))
    var timestepCPU = Tensor<BlockFloat>(.CPU, .WC(layout.sequenceLength, 2))
    for row in layout.textRange {
      adalnCPU[row, 1] = 1
      timestepCPU[row, 0] = 1
    }
    for row in layout.videoRange {
      adalnCPU[row, 0] = 1
      timestepCPU[row, 0] = 1
    }
    for row in layout.audioRange {
      adalnCPU[row, 5] = 1
      timestepCPU[row, 1] = 1
    }
    let adaln = graph.variable(adalnCPU.toGPU(deviceID))
    let timestepSelection = graph.variable(timestepCPU.toGPU(deviceID))
    let initialTemb = graph.variable(
      .GPU(deviceID), .WC(2, H3Config.timestepSize), of: Float.self)
    let (dit, _) = H3JointTransformer(
      textLength: text.shape[0], audioLength: audioLength, videoLength: videoLength,
      timestepCount: 2)
    dit.maxConcurrency = .limit(1)
    dit.compile(inputs: video, audio, textTensor, rotary, adaln, timestepSelection, initialTemb)
    loadUnified("dit", model: dit, storePath: ditStore)

    let initialFrequencies = graph.variable(timestepFrequencies([0, 0]).toGPU(deviceID))
    let (timeModel, _) = H3TimestepEmbedding(timestepCount: 2)
    timeModel.maxConcurrency = .limit(1)
    timeModel.compile(inputs: initialFrequencies)
    loadUnified("time_embedder", model: timeModel, storePath: ditStore)

    let videoSigmas = shiftedSigmas(points: options.steps, shift: H3Config.videoFlowShift)
    let audioSigmas = shiftedSigmas(points: options.steps, shift: H3Config.audioFlowShift)
    for step in 0..<(options.steps - 1) {
      let videoTimestep = 1 - videoSigmas[step]
      let audioTimestep = 1 - audioSigmas[step]
      let frequencies = graph.variable(
        timestepFrequencies([videoTimestep, audioTimestep]).toGPU(deviceID))
      let temb = timeModel(inputs: frequencies)[0].as(of: Float.self)
      let velocity = dit(
        inputs: video, audio, textTensor, rotary, adaln, timestepSelection, temb)
      let videoVelocity = velocity[0].as(of: Float.self)
      let audioVelocity = velocity[1].as(of: Float.self)
      let videoDelta = (1 - videoSigmas[step + 1] / videoSigmas[step]) * videoSigmas[step]
      let audioDelta = (1 - audioSigmas[step + 1] / audioSigmas[step]) * audioSigmas[step]
      video = video + videoDelta * videoVelocity
      audio = audio + audioDelta * audioVelocity
      print("MiniMax-H3 denoise step \(step + 1)/\(options.steps - 1)")
    }
    let videoCPU = video.rawValue.toCPU().reshaped(.WC(videoLength, H3Config.videoPatchSize))
    let audioCPU = audio.rawValue.toCPU().reshaped(.WC(audioLength, H3Config.audioChannels))
    return (
      unpatchVideoRows(videoCPU, frames: latentFrames, height: latentHeight, width: latentWidth),
      audioCPU
    )
  }
}

let videoLatentMean: [Float] = [
  0.85809034, -0.96065915, 1.066164, -0.50903255, -0.2727582, -1.3675414,
  -0.2553255, -0.26907554, -0.5376841, -0.04640973, 0.66573703, 0.19690128,
  -0.5460608, -0.4035342, -0.23683025, 0.25928453, -0.30133945, 0.21134199,
  -1.1206849, 0.35819334, -0.042251438, 0.260483, 0.22864093, 0.7056032,
]
let videoLatentStd: [Float] = [
  1.2223774, 1.2767264, 1.6831775, 1.7549455, 1.5636216, 2.1941435,
  0.9653138, 1.0569886, 0.8419489, 0.7729953, 1.8955938, 0.94684184,
  0.79968095, 0.449889, 0.71974, 0.6936293, 2.961095, 2.76942,
  3.0496185, 2.1088054, 3.2762263, 3.1627357, 2.2816813, 2.6127844,
]
let audioLatentMean: [Float] = [
  -0.020211687, 0.38764665, -0.0439828, -0.28591514, 0.08179686, -0.3578264,
  0.04062381, -0.01552535, -0.22336248, 0.18210068, 0.2941779, -0.07901168,
  -0.056815073, -0.36990282, -0.31616315, 0.5905951, -0.05213957, 0.01367316,
  -0.03691648, 0.09732661, -0.33946624, -0.30685678, -0.24504599, -0.03469852,
  0.02868032, -0.2121778, -0.16782632, 0.3221288, -0.12230559, 0.43566048,
  -0.05025992, 0.39792585,
]
let audioLatentStd: [Float] = [
  1.6895524, 2.7626374, 1.7945344, 1.6801682, 1.6390227, 2.7788298, 1.765909,
  1.6199758, 2.6336527, 1.8539357, 2.5056498, 1.8110192, 1.9579657, 1.6685498,
  1.4922469, 3.2986703, 1.9491805, 1.8720003, 1.833408, 1.648807, 1.6176958,
  1.9131449, 1.5695245, 1.6943659, 1.8318421, 1.5540638, 1.9344931, 1.5991982,
  1.718046, 1.630722, 1.8661226, 1.5613768,
]

func appendLE<T: FixedWidthInteger>(_ value: T, to bytes: inout [UInt8]) {
  var littleEndian = value.littleEndian
  withUnsafeBytes(of: &littleEndian) { bytes.append(contentsOf: $0) }
}

func writeStereoWAV(_ channels: [Tensor<Float>], path: String) {
  precondition(channels.count == 2 && channels[0].shape == channels[1].shape)
  let sampleCount = channels[0].shape.reduce(1, *)
  let dataBytes = sampleCount * 2 * MemoryLayout<Int16>.size
  var bytes = Array("RIFF".utf8)
  appendLE(UInt32(36 + dataBytes), to: &bytes)
  bytes.append(contentsOf: Array("WAVEfmt ".utf8))
  appendLE(UInt32(16), to: &bytes)
  appendLE(UInt16(1), to: &bytes)
  appendLE(UInt16(2), to: &bytes)
  appendLE(UInt32(H3Config.audioOutputRate), to: &bytes)
  appendLE(UInt32(H3Config.audioOutputRate * 4), to: &bytes)
  appendLE(UInt16(4), to: &bytes)
  appendLE(UInt16(16), to: &bytes)
  bytes.append(contentsOf: Array("data".utf8))
  appendLE(UInt32(dataBytes), to: &bytes)
  for index in 0..<sampleCount {
    for channel in 0..<2 {
      let value = max(-1, min(1, channels[channel][index]))
      appendLE(Int16((value * 32_767).rounded()), to: &bytes)
    }
  }
  try! Data(bytes).write(to: URL(fileURLWithPath: path))
}

func decodeAudio(_ rows: Tensor<Float>, outputDirectory: String) {
  precondition(rows.shape[0] % 2 == 0 && rows.shape[1] == H3Config.audioChannels)
  let latentWidth = rows.shape[0] / 2
  let dummy = graph.variable(
    .GPU(deviceID), .NCHW(1, H3Config.audioChannels, 1, latentWidth), of: Float.self)
  let (_, decoder) = H3AudioDecoder(latentWidth: latentWidth)
  decoder.maxConcurrency = .limit(1)
  decoder.compile(inputs: dummy)
  loadUnified("audio_decoder", model: decoder, storePath: vaeStore)
  var decoded = [Tensor<Float>]()
  for channel in 0..<2 {
    var input = Tensor<Float>(
      .CPU, format: .NCHW, shape: [1, H3Config.audioChannels, 1, latentWidth])
    for latent in 0..<latentWidth {
      for feature in 0..<H3Config.audioChannels {
        input[0, feature, 0, latent] =
          rows[channel * latentWidth + latent, feature] * audioLatentStd[feature]
          + audioLatentMean[feature]
      }
    }
    let variable = graph.variable(input.toGPU(deviceID))
    decoded.append(
      decoder(inputs: variable)[0].as(of: Float.self).rawValue.toCPU()
        .reshaped(.C(latentWidth * 800)))
  }
  let path = outputDirectory + "/audio.wav"
  writeStereoWAV(decoded, path: path)
  print("MiniMax-H3 wrote", path)
}

func denormalizeVideoLatents(_ normalized: Tensor<Float>) -> Tensor<Float> {
  var latents = normalized
  let frames = latents.shape[1]
  let height = latents.shape[2]
  let width = latents.shape[3]
  latents.withUnsafeMutableBytes { bytes in
    let values = bytes.baseAddress!.assumingMemoryBound(to: Float.self)
    let plane = frames * height * width
    for channel in 0..<24 {
      for index in 0..<plane {
        let offset = channel * plane + index
        values[offset] = values[offset] * videoLatentStd[channel] + videoLatentMean[channel]
      }
    }
  }
  return latents
}

func makeVideoDecoderInput(
  _ latents: Tensor<Float>, temporalStart: Int, temporalCount: Int,
  yStart: Int, xStart: Int, tileHeight: Int, tileWidth: Int
) -> Tensor<VideoFloat> {
  var input = Tensor<VideoFloat>(
    .CPU, .HWC(1, temporalCount * tileHeight * tileWidth, H3Config.videoChannels))
  for frame in 0..<temporalCount {
    for y in 0..<tileHeight {
      for x in 0..<tileWidth {
        let row = (frame * tileHeight + y) * tileWidth + x
        for channel in 0..<H3Config.videoChannels {
          input[0, row, channel] = VideoFloat(
            latents[channel, temporalStart + frame, yStart + y, xStart + x])
        }
      }
    }
  }
  return input
}

func stitchVideoTiles(
  _ tiles: [[Tensor<Float>]], yPlan: H3VideoTilePlan, xPlan: H3VideoTilePlan,
  frames: Int, height: Int, width: Int
) -> Tensor<Float> {
  var output = Tensor<Float>(.CPU, format: .NCHW, shape: [1, 3, frames, height, width])
  for tileY in tiles.indices {
    for tileX in tiles[tileY].indices {
      let tile = tiles[tileY][tileX]
      let tileHeight = yPlan.lengths[tileY]
      let tileWidth = xPlan.lengths[tileX]
      let topOverlap = tileY > 0 ? yPlan.overlaps[tileY - 1] : 0
      let leftOverlap = tileX > 0 ? xPlan.overlaps[tileX - 1] : 0
      let copyHeight = tileY + 1 < tiles.count ? tileHeight - yPlan.overlaps[tileY] : tileHeight
      let copyWidth = tileX + 1 < tiles[tileY].count ? tileWidth - xPlan.overlaps[tileX] : tileWidth
      for channel in 0..<3 {
        for frame in 0..<frames {
          for y in 0..<copyHeight {
            for x in 0..<copyWidth {
              var value = tile[0, channel, frame, y, x]
              if topOverlap > 0 && y < topOverlap {
                let previous = tiles[tileY - 1][tileX]
                let weight = Float(y) / Float(topOverlap)
                value =
                  previous[0, channel, frame, tileHeight - topOverlap + y, x]
                  * (1 - weight) + value * weight
              }
              if leftOverlap > 0 && x < leftOverlap {
                let previous = tiles[tileY][tileX - 1]
                let weight = Float(x) / Float(leftOverlap)
                value =
                  previous[0, channel, frame, y, tileWidth - leftOverlap + x]
                  * (1 - weight) + value * weight
              }
              output[0, channel, frame, yPlan.starts[tileY] + y, xPlan.starts[tileX] + x] = value
            }
          }
        }
      }
    }
  }
  return output
}

func writePNGFrame(
  _ frame: Int, from clip: Tensor<Float>, outputIndex: Int, outputDirectory: String,
  previous: Tensor<Float>? = nil, previousFrame: Int = 0, blendWeight: Float = 1
) {
  let height = clip.shape[3]
  let width = clip.shape[4]
  let mean: [Float] = [0.485, 0.456, 0.406]
  let std: [Float] = [0.229, 0.224, 0.225]
  var pixels = [PNG.RGBA<UInt8>](repeating: .init(0), count: height * width)
  for y in 0..<height {
    for x in 0..<width {
      var channels = [UInt8](repeating: 0, count: 3)
      for channel in 0..<3 {
        var value = clip[0, channel, frame, y, x]
        if let previous {
          value =
            previous[0, channel, previousFrame, y, x] * (1 - blendWeight)
            + value * blendWeight
        }
        value = max(0, min(1, value * std[channel] + mean[channel]))
        channels[channel] = UInt8((value * 255).rounded())
      }
      pixels[y * width + x] = PNG.RGBA(channels[0], channels[1], channels[2], 255)
    }
  }
  let image = PNG.Data.Rectangular(
    packing: pixels, size: (width, height),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  let filename = String(format: "%@/frame_%04d.png", outputDirectory, outputIndex)
  try! image.compress(path: filename, level: 4)
}

func decodeVideo(_ normalizedLatents: Tensor<Float>, outputDirectory: String) {
  let latents = denormalizeVideoLatents(normalizedLatents)
  let latentFrames = latents.shape[1]
  let latentHeight = latents.shape[2]
  let latentWidth = latents.shape[3]
  let temporalPlan = videoTemporalDecodePlan(latentFrames: latentFrames)
  precondition(!temporalPlan.clipStarts.isEmpty, "At least seven video latent frames are required")
  var padded = latents
  if temporalPlan.repeatedTailTokens > 0 {
    var expanded = Tensor<Float>(
      .CPU, format: .NCHW,
      shape: [
        24, temporalPlan.paddedTokenCount - H3Config.videoTokenDrop, latentHeight, latentWidth,
      ])
    for channel in 0..<24 {
      for frame in 0..<expanded.shape[1] {
        let sourceFrame = min(frame, latentFrames - 1)
        for y in 0..<latentHeight {
          for x in 0..<latentWidth {
            expanded[channel, frame, y, x] = latents[channel, sourceFrame, y, x]
          }
        }
      }
    }
    padded = expanded
  }

  let yPlan = splitVideoTiles(length: options.height)
  let xPlan = splitVideoTiles(length: options.width)
  let temporalCount = temporalPlan.tokensPerChunk + temporalPlan.tokenOverlap
  let firstTileHeight = yPlan.lengths[0] / H3Config.videoSpatialCompression
  let firstTileWidth = xPlan.lengths[0] / H3Config.videoSpatialCompression
  let numPatches = temporalCount * firstTileHeight * firstTileWidth
  let dummy = graph.variable(
    .GPU(deviceID), .HWC(1, numPatches, H3Config.videoChannels), of: VideoFloat.self)
  let decoderRotary = graph.variable(
    videoDecoderRotary(
      latentFrames: temporalCount, latentHeight: firstTileHeight, latentWidth: firstTileWidth
    ).toGPU(deviceID))
  let zero = graph.variable(
    .GPU(deviceID), .HWC(1, 1, H3VideoDecoderConfig.width), of: Float.self)
  zero.full(0)
  let (decoder, _) = H3VideoDecoder(numPatches: numPatches, deviceID: deviceID)
  decoder.maxConcurrency = .limit(1)
  decoder.compile(inputs: dummy, decoderRotary, zero)
  loadUnified("video_decoder", model: decoder, storePath: vaeStore)

  var previousClip: Tensor<Float>? = nil
  var outputFrame = 0
  for (clipIndex, temporalStart) in temporalPlan.clipStarts.enumerated() {
    print("MiniMax-H3 video decode clip \(clipIndex + 1)/\(temporalPlan.clipStarts.count)")
    var tileRows = [[Tensor<Float>]]()
    for (tileY, yStartPixels) in yPlan.starts.enumerated() {
      var row = [Tensor<Float>]()
      for (tileX, xStartPixels) in xPlan.starts.enumerated() {
        let tileHeight = yPlan.lengths[tileY] / H3Config.videoSpatialCompression
        let tileWidth = xPlan.lengths[tileX] / H3Config.videoSpatialCompression
        precondition(tileHeight == firstTileHeight && tileWidth == firstTileWidth)
        let input = makeVideoDecoderInput(
          padded, temporalStart: temporalStart, temporalCount: temporalCount,
          yStart: yStartPixels / H3Config.videoSpatialCompression,
          xStart: xStartPixels / H3Config.videoSpatialCompression,
          tileHeight: tileHeight, tileWidth: tileWidth)
        let variable = graph.variable(input.toGPU(deviceID))
        let rows = Tensor<Float>(
          from: decoder(inputs: variable, decoderRotary, zero)[0].as(of: VideoFloat.self)
            .rawValue.toCPU()
        ).reshaped(
          .WC(
            numPatches,
            3 * H3VideoDecoderConfig.temporalPatch * H3VideoDecoderConfig.spatialPatch
              * H3VideoDecoderConfig.spatialPatch))
        row.append(
          unpatchifyVideoDecoderRows(
            rows, latentFrames: temporalCount, latentHeight: tileHeight,
            latentWidth: tileWidth))
      }
      tileRows.append(row)
    }
    let clip = stitchVideoTiles(
      tileRows, yPlan: yPlan, xPlan: xPlan,
      frames: temporalCount * H3VideoDecoderConfig.temporalPatch,
      height: options.height, width: options.width)
    let firstStart = temporalPlan.framePrePadding
    for localFrame in 0..<H3Config.framesPerChunk {
      if localFrame < temporalPlan.frameOverlap, let previousClip {
        writePNGFrame(
          firstStart + localFrame, from: clip, outputIndex: outputFrame,
          outputDirectory: outputDirectory, previous: previousClip,
          previousFrame: temporalCount * H3VideoDecoderConfig.temporalPatch
            - temporalPlan.frameOverlap + localFrame,
          blendWeight: Float(localFrame) / Float(temporalPlan.frameOverlap))
      } else {
        writePNGFrame(
          firstStart + localFrame, from: clip, outputIndex: outputFrame,
          outputDirectory: outputDirectory)
      }
      outputFrame += 1
    }
    previousClip = clip
  }
  if let previousClip {
    let overlapStart =
      temporalCount * H3VideoDecoderConfig.temporalPatch
      - temporalPlan.frameOverlap
    for localFrame in 0..<temporalPlan.frameOverlap {
      writePNGFrame(
        overlapStart + localFrame, from: previousClip, outputIndex: outputFrame,
        outputDirectory: outputDirectory)
      outputFrame += 1
    }
  }
  let paddedFrames =
    temporalPlan.repeatedTailTokens == 0
    ? 0 : temporalPlan.repeatedTailTokens * H3VideoDecoderConfig.temporalPatch
  precondition(outputFrame - paddedFrames == options.frames)
  if paddedFrames > 0 {
    print("MiniMax-H3 warning: trailing padded frame files require removal for this frame count")
  }
}

func muxOutput(outputDirectory: String) {
  guard FileManager.default.isExecutableFile(atPath: "/usr/bin/ffmpeg") else {
    print("MiniMax-H3 ffmpeg not found; leaving PNG frames and audio.wav in", outputDirectory)
    return
  }
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/usr/bin/ffmpeg")
  process.arguments = [
    "-y", "-framerate", String(H3Config.fps), "-i", outputDirectory + "/frame_%04d.png",
    "-i", outputDirectory + "/audio.wav", "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-shortest", outputDirectory + "/output.mp4",
  ]
  try! process.run()
  process.waitUntilExit()
  precondition(process.terminationStatus == 0, "ffmpeg mux failed")
  print("MiniMax-H3 wrote", outputDirectory + "/output.mp4")
}

let latentFrames = videoLatentFrameCount(options.frames)
let latentHeight = options.height / H3Config.videoSpatialCompression
let latentWidth = options.width / H3Config.videoSpatialCompression
let audioLatents = audioLatentFrameCount(options.frames)
print(
  "MiniMax-H3 T2VA plan: \(options.width)x\(options.height), \(options.frames) frames,",
  "\(latentFrames)x\(latentHeight)x\(latentWidth) video latents, \(audioLatents) audio latents,",
  "\(options.steps - 1) denoiser evaluations")
if options.height < 768 && !options.dryRun {
  print("MiniMax-H3 warning: the released checkpoint is intended for a 768-pixel short edge")
}
if options.dryRun {
  exit(0)
}

try! FileManager.default.createDirectory(
  atPath: options.output, withIntermediateDirectories: true)
let text = encodePrompt(options.prompt)
if options.textOnly {
  print("MiniMax-H3 text embedding shape:", text.shape)
  exit(0)
}
let (videoLatents, audioRows) = runDenoiser(text: text)
decodeAudio(audioRows, outputDirectory: options.output)
decodeVideo(videoLatents, outputDirectory: options.output)
muxOutput(outputDirectory: options.output)
