import Diffusion
import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

// MiniMax-H3 converter and parity harness. H3 and its Qwen3-VL text encoder are
// distributed and executed in BF16. The video
// decoder is a separate FP16-autocast model, so keep its scalar type distinct.
typealias BlockFloat = BFloat16
typealias VideoFloat = Float16

setbuf(stdout, nil)

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

let environment = ProcessInfo.processInfo.environment
let mode = CommandLine.arguments.dropFirst().first ?? "self-test"
let modelRoot = environment["MINIMAX_H3_MODEL"] ?? "/slow/Data/MiniMax-H3"
let qwenExportPath =
  environment["MINIMAX_H3_QWEN_EXPORT_PATH"]
  ?? "/slow/Data/minimax_h3_qwen_3_vl_f16.ckpt"
let ditExportPath =
  environment["MINIMAX_H3_DIT_EXPORT_PATH"]
  ?? "/slow/Data/minimax_h3_dit_f32.ckpt"
let ref2vaDitExportPath =
  environment["MINIMAX_H3_REF2VA_DIT_EXPORT_PATH"]
  ?? "/slow/Data/minimax_h3_ref2va_dit_f32.ckpt"
let vaeExportPath =
  environment["MINIMAX_H3_VAE_EXPORT_PATH"]
  ?? "/slow/Data/minimax_h3_vae_f32.ckpt"
let deviceID = Int(environment["MINIMAX_H3_DEVICE"] ?? "0") ?? 0

enum H3Config {
  static let hiddenSize = 5_376
  static let layers = 50
  static let refinerLayers = 2
  static let heads = 56
  static let headDim = 128
  static let innerAttentionSize = heads * headDim
  static let intermediateSize = 14_336
  static let videoChannels = 24
  static let audioChannels = 32
  static let patch = (t: 1, h: 2, w: 2)
  static let videoPatchSize = videoChannels * patch.t * patch.h * patch.w
  static let textSize = 5_120
  static let timestepFrequencySize = 256
  static let timestepHiddenSize = 5_376
  static let timestepSize = 2_688
  static let ropeFrequencySize = 16
  static let rotarySize = 2 * 3 * ropeFrequencySize
  static let ropeTheta: Double = 10_000
  static let normEpsilon: Float = 1e-5
  static let modalityCount = 3

  static let fps = 24
  static let audioLatentsPerSecond = 40
  static let audioOutputRate = 32_000
  static let videoSpatialCompression = 16
  static let framesPerChunk = 17
  static let latentsPerChunk = 5
  static let videoTokenDrop = 3
  static let defaultFrames = 124
  static let defaultHeight = 768
  static let defaultWidth = 1_344
  static let videoFlowShift: Float = 12
  static let audioFlowShift: Float = 3
}

enum H3QwenConfig {
  static let hiddenSize = 5_120
  static let layers = 64
  static let featureLayer = 50
  static let heads = 64
  static let keyValueHeads = 8
  static let headDim = 128
  static let intermediateSize = 25_600
  static let vocabularySize = 151_936
  static let normEpsilon: Float = 1e-6
  static let ropeTheta: Double = 5_000_000
}

enum H3VideoDecoderConfig {
  static let width = 2_048
  static let layers = 36
  static let heads = 32
  static let headDim = 64
  static let intermediateSize = 8_192
  static let rotarySize = 48
  static let registerTokens = 4
  static let spatialPatch = 16
  static let temporalPatch = 4
  static let normEpsilon: Float = 1e-5
}

enum H3ConditioningVideoEncoderConfig {
  static let channels = [128, 256, 256, 512, 512, 1_024]
  static let spatialStrides = [2, 2, 2, 2, 1, 1]
  static let temporalStrides = [1, 2, 2, 1, 1, 1]
  static let blocksPerLevel = 2
  static let latentChannels = 24
  static let normEpsilon: Float = 1e-6
}

enum H3Tag: Int32 {
  case video = 0
  case text = 1
  case audio = 2
}

struct H3PackedLayout {
  let sequenceLength: Int
  let textRange: Range<Int>
  let audioRange: Range<Int>
  let videoRange: Range<Int>
  let positionIDs: Tensor<Float>
  let tokenTags: [Int32]
}

func alignFrameCount(_ requested: Int) -> Int {
  precondition(requested > 0)
  var count = requested
  while count % H3Config.framesPerChunk != H3Config.latentsPerChunk {
    count += 1
  }
  return count
}

func videoLatentFrameCount(_ frames: Int) -> Int {
  precondition(frames % H3Config.framesPerChunk == H3Config.latentsPerChunk)
  return (frames - H3Config.latentsPerChunk) / H3Config.framesPerChunk
    * H3Config.latentsPerChunk + 2
}

func audioLatentFrameCount(_ frames: Int) -> Int {
  Int((Double(frames) / Double(H3Config.fps) * Double(H3Config.audioLatentsPerSecond)).rounded())
}

struct H3VideoTilePlan {
  let starts: [Int]
  let lengths: [Int]
  let overlaps: [Int]
}

func splitVideoTiles(length: Int, tileSize: Int = 256, minimumOverlap: Int = 64)
  -> H3VideoTilePlan
{
  precondition(length > 0 && tileSize > 0 && minimumOverlap >= 0)
  if tileSize >= length {
    return H3VideoTilePlan(starts: [0], lengths: [length], overlaps: [])
  }
  var tileCount = (length + tileSize - 1) / tileSize
  while tileSize * tileCount - minimumOverlap * (tileCount - 1) < length {
    tileCount += 1
  }
  var overlaps = [Int](repeating: minimumOverlap, count: tileCount - 1)
  let remaining = tileSize * tileCount - overlaps.reduce(0, +) - length
  for index in 0..<(remaining / H3Config.videoSpatialCompression) {
    overlaps[index % overlaps.count] += H3Config.videoSpatialCompression
  }
  var starts = [0]
  for index in 0..<(tileCount - 1) {
    starts.append(starts.last! + tileSize - overlaps[index])
  }
  return H3VideoTilePlan(
    starts: starts, lengths: [Int](repeating: tileSize, count: tileCount),
    overlaps: overlaps)
}

struct H3VideoTemporalDecodePlan {
  let framePrePadding: Int
  let tokensPerChunk: Int
  let tokenOverlap: Int
  let frameOverlap: Int
  let paddedTokenCount: Int
  let repeatedTailTokens: Int
  let clipStarts: [Int]
}

func videoTemporalDecodePlan(latentFrames: Int) -> H3VideoTemporalDecodePlan {
  precondition(latentFrames > 0)
  let temporalRatio = H3VideoDecoderConfig.temporalPatch
  let framePrePadding = (temporalRatio - H3Config.framesPerChunk % temporalRatio) % temporalRatio
  let tokensPerChunk =
    (H3Config.framesPerChunk + temporalRatio - 1) / temporalRatio
  let tokenOverlap =
    (tokensPerChunk - H3Config.videoTokenDrop % tokensPerChunk) % tokensPerChunk
  let frameOverlap = max(tokenOverlap * temporalRatio - framePrePadding, 0)
  let tokenCountBeforePadding = latentFrames + H3Config.videoTokenDrop
  let repeatedTailTokens =
    (tokensPerChunk - tokenCountBeforePadding % tokensPerChunk) % tokensPerChunk
  let paddedTokenCount = tokenCountBeforePadding + repeatedTailTokens
  let chunkCount = paddedTokenCount / tokensPerChunk - 1
  return H3VideoTemporalDecodePlan(
    framePrePadding: framePrePadding, tokensPerChunk: tokensPerChunk,
    tokenOverlap: tokenOverlap, frameOverlap: frameOverlap,
    paddedTokenCount: paddedTokenCount, repeatedTailTokens: repeatedTailTokens,
    clipStarts: (0..<chunkCount).map { $0 * tokensPerChunk })
}

func videoDecoderRotary(latentFrames: Int, latentHeight: Int, latentWidth: Int)
  -> Tensor<VideoFloat>
{
  let patchCount = latentFrames * latentHeight * latentWidth
  let suffixCount = H3VideoDecoderConfig.registerTokens + 1
  var rotary = Tensor<VideoFloat>(
    .CPU, .NHWC(1, patchCount + suffixCount, 1, H3VideoDecoderConfig.headDim))
  let frequencyCount = H3VideoDecoderConfig.rotarySize / 6
  let inverseFrequencies = (0..<frequencyCount).map { index in
    pow(100.0, -Double(index) * 6.0 / Double(H3VideoDecoderConfig.rotarySize))
  }
  var row = 0
  for frame in 0..<latentFrames {
    let t = 2 * (Double(frame) + 0.5) / Double(latentFrames) - 1
    for yIndex in 0..<latentHeight {
      let y = 2 * (Double(yIndex) + 0.5) / Double(latentHeight) - 1
      for xIndex in 0..<latentWidth {
        let x = 2 * (Double(xIndex) + 0.5) / Double(latentWidth) - 1
        var angleIndex = 0
        for coordinate in [t, y, x] {
          for inverseFrequency in inverseFrequencies {
            let angle = 2 * Double.pi * coordinate * inverseFrequency
            rotary[0, row, 0, 2 * angleIndex] = VideoFloat(cos(angle))
            rotary[0, row, 0, 2 * angleIndex + 1] = VideoFloat(sin(angle))
            angleIndex += 1
          }
        }
        for index in H3VideoDecoderConfig.rotarySize..<H3VideoDecoderConfig.headDim {
          rotary[0, row, 0, index] = VideoFloat(index % 2 == 0 ? 1 : 0)
        }
        row += 1
      }
    }
  }
  for suffix in 0..<suffixCount {
    for index in 0..<H3VideoDecoderConfig.headDim {
      rotary[0, patchCount + suffix, 0, index] = VideoFloat(index % 2 == 0 ? 1 : 0)
    }
  }
  return rotary
}

func unpatchifyVideoDecoderRows(
  _ rows: Tensor<Float>, latentFrames: Int, latentHeight: Int, latentWidth: Int
) -> Tensor<Float> {
  precondition(rows.shape == [latentFrames * latentHeight * latentWidth, 3 * 4 * 16 * 16])
  let temporalPatch = H3VideoDecoderConfig.temporalPatch
  let spatialPatch = H3VideoDecoderConfig.spatialPatch
  let outputFrames = latentFrames * temporalPatch
  let outputHeight = latentHeight * spatialPatch
  let outputWidth = latentWidth * spatialPatch
  let featureCount = 3 * temporalPatch * spatialPatch * spatialPatch
  var output = Tensor<Float>(
    .CPU, format: .NCHW, shape: [1, 3, outputFrames, outputHeight, outputWidth])
  // The C backend cannot materialize the reference's eight-dimensional tensor view directly.
  // Reorder contiguous CPU storage explicitly; this is also suitable for one decoded spatial tile.
  rows.withUnsafeBytes { sourceBytes in
    output.withUnsafeMutableBytes { outputBytes in
      let source = sourceBytes.baseAddress!.assumingMemoryBound(to: Float.self)
      let destination = outputBytes.baseAddress!.assumingMemoryBound(to: Float.self)
      for frame in 0..<latentFrames {
        for y in 0..<latentHeight {
          for x in 0..<latentWidth {
            let token = (frame * latentHeight + y) * latentWidth + x
            for channel in 0..<3 {
              for patchFrame in 0..<temporalPatch {
                for patchY in 0..<spatialPatch {
                  for patchX in 0..<spatialPatch {
                    let feature =
                      (((channel * temporalPatch + patchFrame) * spatialPatch + patchY)
                        * spatialPatch + patchX)
                    let outputIndex =
                      (((channel * outputFrames + frame * temporalPatch + patchFrame)
                        * outputHeight + y * spatialPatch + patchY) * outputWidth
                        + x * spatialPatch + patchX)
                    destination[outputIndex] = source[token * featureCount + feature]
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return output
}

private let ropeFrameRescale = 5.0 / 3.0
private let ropeFramesPerLatent = [1.0, 4.0, 4.0, 4.0, 4.0]

func temporalPositionGrid(count: Int, origin: Double) -> [Double] {
  var result = [Double](repeating: 0, count: count)
  var position = origin
  for i in 0..<count {
    result[i] = position
    position += ropeFrameRescale * ropeFramesPerLatent[i % ropeFramesPerLatent.count]
  }
  return result
}

func spatialPositionGrid(dimension: Int, patch: Int, squareRootArea: Double) -> [Double] {
  let count = dimension / patch
  let ratio = Double(dimension) / squareRootArea
  let left = (1 - ratio) / 2
  return (0..<count).map { index in
    // This is numpy.linspace(start, stop, count, endpoint=False), not torch.linspace.
    (left + ratio * Double(index) / Double(count)) * 32
  }
}

func makeT2VALayout(
  textTokenCount: Int, latentFrames: Int, latentHeight: Int, latentWidth: Int,
  audioLatents: Int
) -> H3PackedLayout {
  precondition(latentHeight % H3Config.patch.h == 0)
  precondition(latentWidth % H3Config.patch.w == 0)
  let rowsPerFrame =
    latentHeight / H3Config.patch.h * latentWidth / H3Config.patch.w
  let audioRows = audioLatents * 2
  let videoRows = latentFrames * rowsPerFrame
  let textRange = 0..<textTokenCount
  let audioRange = textRange.upperBound..<(textRange.upperBound + audioRows)
  let videoRange = audioRange.upperBound..<(audioRange.upperBound + videoRows)
  let sequenceLength = videoRange.upperBound

  var positions = Tensor<Float>(.CPU, .WC(sequenceLength, 3))
  for i in textRange {
    positions[i, 0] = Float(i)
    positions[i, 1] = 0
    positions[i, 2] = 0
  }

  let squareRootArea = sqrt(Double(latentHeight * latentWidth))
  let heightGrid = spatialPositionGrid(
    dimension: latentHeight, patch: H3Config.patch.h, squareRootArea: squareRootArea)
  let widthGrid = spatialPositionGrid(
    dimension: latentWidth, patch: H3Config.patch.w, squareRootArea: squareRootArea)
  for channel in 0..<2 {
    for i in 0..<audioLatents {
      let row = audioRange.lowerBound + channel * audioLatents + i
      positions[row, 0] = Float(Double(textTokenCount + i))
      positions[row, 1] = 0
      positions[row, 2] = Float(channel == 0 ? widthGrid.first! : widthGrid.last!)
    }
  }

  let temporalGrid = temporalPositionGrid(count: latentFrames, origin: Double(textTokenCount))
  for frame in 0..<latentFrames {
    for y in 0..<heightGrid.count {
      for x in 0..<widthGrid.count {
        let spatialIndex = y * widthGrid.count + x
        let row = videoRange.lowerBound + frame * rowsPerFrame + spatialIndex
        positions[row, 0] = Float(temporalGrid[frame])
        positions[row, 1] = Float(heightGrid[y])
        positions[row, 2] = Float(widthGrid[x])
      }
    }
  }

  var tags = [Int32](repeating: H3Tag.text.rawValue, count: sequenceLength)
  for i in audioRange { tags[i] = H3Tag.audio.rawValue }
  for i in videoRange { tags[i] = H3Tag.video.rawValue }
  return H3PackedLayout(
    sequenceLength: sequenceLength, textRange: textRange, audioRange: audioRange,
    videoRange: videoRange, positionIDs: positions, tokenTags: tags)
}

func h3RotaryTensor(positionIDs: Tensor<Float>) -> Tensor<BlockFloat> {
  precondition(positionIDs.shape.count == 2 && positionIDs.shape[1] == 3)
  let rows = positionIDs.shape[0]
  var rotary = Tensor<BlockFloat>(.CPU, .NHWC(1, rows, 1, H3Config.headDim))
  for row in 0..<rows {
    for angleIndex in 0..<48 {
      let axis = angleIndex / 16
      let frequencyIndex = angleIndex % 16
      let inverseFrequency = pow(10_000.0, -Double(frequencyIndex) * 2.0 / 32.0)
      let angle = Double(positionIDs[row, axis]) * inverseFrequency
      rotary[0, row, 0, angleIndex * 2] = BlockFloat(cos(angle))
      rotary[0, row, 0, angleIndex * 2 + 1] = BlockFloat(sin(angle))
    }
    for index in 96..<H3Config.headDim {
      rotary[0, row, 0, index] = BlockFloat(index % 2 == 0 ? 1 : 0)
    }
  }
  return rotary
}

func shiftedSigmaGrid(points: Int, shift: Float) -> [Float] {
  precondition(points >= 2 && shift > 0)
  return (0..<points).map { index in
    let base = 1 - Float(index) / Float(points - 1)
    return shift * base / (1 + (shift - 1) * base)
  }
}

func schedulerStep(sample: Float, velocity: Float, timestep: Float, sigma: Float, nextSigma: Float)
  -> Float
{
  let denoised = sample + (1 - timestep) * velocity
  let ratio = nextSigma / sigma
  return ratio * sample + (1 - ratio) * denoised
}

struct TokenParityMetrics {
  let allFinite: Bool
  let maxAbsoluteDifference: Float
  let maxRelativeDifference: Float
  let meanAbsoluteDifference: Float
  let rootMeanSquareError: Float
  let normalizedRootMeanSquareError: Float
  let referenceMaximumMagnitude: Float
  let referenceRootMeanSquare: Float
  let maximumDifferenceToken: Int
  let maximumDifferenceFeature: Int
  let actualAtMaximumDifference: Float
  let expectedAtMaximumDifference: Float
  let minimumCosineSimilarity: Float
  let meanCosineSimilarity: Float
  let worstCosineToken: Int
}

func tokenParityMetrics(_ actual: Tensor<Float>, _ expected: Tensor<Float>) -> TokenParityMetrics {
  precondition(actual.shape == expected.shape && actual.shape.count == 2)
  let tokenCount = actual.shape[0]
  let featureCount = actual.shape[1]
  var maximumAbsoluteDifference: Float = 0
  var maximumReferenceMagnitude: Float = 0
  var minimumCosine: Float = 1
  var absoluteDifferenceSum: Double = 0
  var squaredDifferenceSum: Double = 0
  var referenceSquaredSum: Double = 0
  var finiteElementCount = 0
  var maximumDifferenceToken = 0
  var maximumDifferenceFeature = 0
  var actualAtMaximumDifference: Float = 0
  var expectedAtMaximumDifference: Float = 0
  var cosineSum: Float = 0
  var worstToken = 0
  var allFinite = true
  for token in 0..<tokenCount {
    var dot: Double = 0
    var actualSquared: Double = 0
    var expectedSquared: Double = 0
    for feature in 0..<featureCount {
      let lhs = actual[token, feature]
      let rhs = expected[token, feature]
      if !lhs.isFinite || !rhs.isFinite {
        allFinite = false
        continue
      }
      let difference = abs(lhs - rhs)
      if difference > maximumAbsoluteDifference {
        maximumAbsoluteDifference = difference
        maximumDifferenceToken = token
        maximumDifferenceFeature = feature
        actualAtMaximumDifference = lhs
        expectedAtMaximumDifference = rhs
      }
      maximumReferenceMagnitude = max(maximumReferenceMagnitude, abs(rhs))
      absoluteDifferenceSum += Double(difference)
      squaredDifferenceSum += Double(difference) * Double(difference)
      referenceSquaredSum += Double(rhs) * Double(rhs)
      finiteElementCount += 1
      dot += Double(lhs) * Double(rhs)
      actualSquared += Double(lhs) * Double(lhs)
      expectedSquared += Double(rhs) * Double(rhs)
    }
    let denominator = sqrt(actualSquared * expectedSquared)
    let cosine =
      denominator > 1e-20 ? Float(dot / denominator) : (actualSquared == expectedSquared ? 1 : 0)
    cosineSum += cosine
    if cosine < minimumCosine {
      minimumCosine = cosine
      worstToken = token
    }
  }
  if !allFinite {
    return TokenParityMetrics(
      allFinite: false, maxAbsoluteDifference: .infinity,
      maxRelativeDifference: .infinity, meanAbsoluteDifference: .infinity,
      rootMeanSquareError: .infinity, normalizedRootMeanSquareError: .infinity,
      referenceMaximumMagnitude: maximumReferenceMagnitude, referenceRootMeanSquare: .infinity,
      maximumDifferenceToken: maximumDifferenceToken,
      maximumDifferenceFeature: maximumDifferenceFeature,
      actualAtMaximumDifference: actualAtMaximumDifference,
      expectedAtMaximumDifference: expectedAtMaximumDifference, minimumCosineSimilarity: -1,
      meanCosineSimilarity: -1, worstCosineToken: worstToken)
  }
  let elementCount = Double(max(finiteElementCount, 1))
  let rootMeanSquareError = Float(sqrt(squaredDifferenceSum / elementCount))
  let referenceRootMeanSquare = Float(sqrt(referenceSquaredSum / elementCount))
  return TokenParityMetrics(
    allFinite: allFinite,
    maxAbsoluteDifference: maximumAbsoluteDifference,
    maxRelativeDifference: maximumAbsoluteDifference / max(maximumReferenceMagnitude, 1e-6),
    meanAbsoluteDifference: Float(absoluteDifferenceSum / elementCount),
    rootMeanSquareError: rootMeanSquareError,
    normalizedRootMeanSquareError: rootMeanSquareError / max(referenceRootMeanSquare, 1e-6),
    referenceMaximumMagnitude: maximumReferenceMagnitude,
    referenceRootMeanSquare: referenceRootMeanSquare,
    maximumDifferenceToken: maximumDifferenceToken,
    maximumDifferenceFeature: maximumDifferenceFeature,
    actualAtMaximumDifference: actualAtMaximumDifference,
    expectedAtMaximumDifference: expectedAtMaximumDifference,
    minimumCosineSimilarity: minimumCosine,
    meanCosineSimilarity: cosineSum / Float(max(tokenCount, 1)),
    worstCosineToken: worstToken)
}

func printMetrics(_ label: String, _ metrics: TokenParityMetrics) {
  print("\(label) all finite:", metrics.allFinite)
  print("\(label) max abs / max ref:", metrics.maxRelativeDifference)
  print(
    "\(label) max abs diff / max ref magnitude:", metrics.maxAbsoluteDifference,
    metrics.referenceMaximumMagnitude)
  print(
    "\(label) mean abs / RMSE / ref RMS / NRMSE:", metrics.meanAbsoluteDifference,
    metrics.rootMeanSquareError, metrics.referenceRootMeanSquare,
    metrics.normalizedRootMeanSquareError)
  print(
    "\(label) max-diff token/feature actual/reference:", metrics.maximumDifferenceToken,
    metrics.maximumDifferenceFeature, metrics.actualAtMaximumDifference,
    metrics.expectedAtMaximumDifference)
  print(
    "\(label) token cosine min/mean:", metrics.minimumCosineSimilarity,
    metrics.meanCosineSimilarity)
  print("\(label) worst cosine token:", metrics.worstCosineToken)
}

let site = Python.import("site")
let sys = Python.import("sys")
let osPath = Python.import("os.path")

func movePythonPathToFront(_ path: String) {
  while Bool(sys.path.__contains__(path)) ?? false { sys.path.remove(path) }
  sys.path.insert(0, path)
}

let userSitePackages = String(site.getusersitepackages()) ?? ""
if Bool(osPath.isdir(userSitePackages)) ?? false,
  !(Bool(sys.path.__contains__(userSitePackages)) ?? false)
{
  movePythonPathToFront(userSitePackages)
}
let systemDistPackages = "/usr/lib/python3/dist-packages"
if !(Bool(sys.path.__contains__(systemDistPackages)) ?? false) {
  movePythonPathToFront(systemDistPackages)
}

let builtins = Python.import("builtins")
let types = Python.import("types")
let torch = Python.import("torch")
let h3Reference = types.ModuleType("minimax_h3_swift_reference")
builtins.exec(
  #"""
  import gc
  import json
  import math
  import os
  import sys
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from safetensors.torch import load_file

  HIDDEN = 5376
  HEADS = 56
  HEAD_DIM = 128
  INNER = HEADS * HEAD_DIM
  FFN = 14336
  TIME = 2688
  ROTARY = 96

  class ShardedStateDict:
      def __init__(self, root, index_name="diffusion_pytorch_model.safetensors.index.json"):
          index_path = os.path.join(root, index_name)
          with open(index_path) as f:
              self.weight_map = json.load(f)["weight_map"]
          self.root = root
          self.cached_name = None
          self.cached = None

      def __contains__(self, key):
          return key in self.weight_map

      def __getitem__(self, key):
          name = self.weight_map[key]
          if name != self.cached_name:
              self.cached = load_file(os.path.join(self.root, name), device="cpu")
              self.cached_name = name
          return self.cached[key]

      def release(self):
          self.cached = None
          self.cached_name = None
          gc.collect()

  class SingleStateDict:
      def __init__(self, path):
          self.state = load_file(path, device="cpu")
      def __contains__(self, key):
          return key in self.state
      def __getitem__(self, key):
          return self.state[key]
      def release(self):
          self.state = None
          gc.collect()

  def audio_state_keys(root):
      state = load_file(os.path.join(root, "audio_vae", "diffusion_pytorch_model.safetensors"), device="cpu")
      return [key for key in state.keys() if key.startswith(("dec_in_proj", "decoder.conv_pre", "decoder.ups.0", "decoder.resblocks.0", "decoder.activation_post", "decoder.conv_post"))]

  def weight_norm_value(state, prefix):
      return torch._weight_norm(state[prefix + ".weight_v"], state[prefix + ".weight_g"], 0)

  def audio_activation(x, state, prefix):
      ratio, kernel = 2, 12
      pad = kernel // ratio - 1
      pad_left = pad * ratio + (kernel - ratio) // 2
      pad_right = pad * ratio + (kernel - ratio + 1) // 2
      channels = x.shape[1]
      up_filter = state[prefix + ".upsample.filter"].to(x.device)
      x = F.pad(x, (pad, pad), mode="replicate")
      x = ratio * F.conv_transpose1d(x, up_filter.expand(channels, -1, -1), stride=ratio, groups=channels)
      x = x[..., pad_left:-pad_right]
      alpha = state[prefix + ".act.alpha"].to(x.device).exp()[None, :, None]
      beta = (state[prefix + ".act.beta"].to(x.device).exp() + 1e-9).reciprocal()[None, :, None]
      x = x + beta * torch.sin(alpha * x).pow(2)
      down_filter = state[prefix + ".downsample.lowpass.filter"].to(x.device)
      x = F.pad(x, (5, 6), mode="replicate")
      return F.conv1d(x, down_filter.expand(channels, -1, -1), stride=2, groups=channels)

  def load_audio_amp(root):
      state = SingleStateDict(os.path.join(root, "audio_vae", "diffusion_pytorch_model.safetensors"))
      return {"state": state}

  @torch.no_grad()
  def run_audio_amp_case(pack, width, seed, device):
      torch.manual_seed(seed)
      state = pack["state"]
      x = torch.randn(1, 512, width, device=f"cuda:{device}", dtype=torch.float32)
      out = x
      prefix = "decoder.resblocks.0"
      for index, dilation in enumerate((1, 3, 5)):
          residual = audio_activation(out, state, f"{prefix}.activations.{index * 2}")
          weight = weight_norm_value(state, f"{prefix}.convs1.{index}").to(out.device)
          bias = state[f"{prefix}.convs1.{index}.bias"].to(out.device)
          residual = F.conv1d(residual, weight, bias, padding=(3 - 1) * dilation // 2, dilation=dilation)
          residual = audio_activation(residual, state, f"{prefix}.activations.{index * 2 + 1}")
          weight = weight_norm_value(state, f"{prefix}.convs2.{index}").to(out.device)
          bias = state[f"{prefix}.convs2.{index}.bias"].to(out.device)
          residual = F.conv1d(residual, weight, bias, padding=(3 - 1) // 2)
          out = out + residual
      return {"x": x[0].cpu(), "output": out[0].cpu()}

  def audio_amp_forward(x, state, prefix, kernel):
      out = x
      for index, dilation in enumerate((1, 3, 5)):
          residual = audio_activation(out, state, f"{prefix}.activations.{index * 2}")
          weight = weight_norm_value(state, f"{prefix}.convs1.{index}").to(out.device)
          bias = state[f"{prefix}.convs1.{index}.bias"].to(out.device)
          residual = F.conv1d(residual, weight, bias, padding=(kernel - 1) * dilation // 2, dilation=dilation)
          residual = audio_activation(residual, state, f"{prefix}.activations.{index * 2 + 1}")
          weight = weight_norm_value(state, f"{prefix}.convs2.{index}").to(out.device)
          bias = state[f"{prefix}.convs2.{index}.bias"].to(out.device)
          residual = F.conv1d(residual, weight, bias, padding=(kernel - 1) // 2)
          out = out + residual
      return out

  @torch.no_grad()
  def run_audio_decoder_case(pack, width, seed, device):
      torch.manual_seed(seed)
      state = pack["state"]
      dev = torch.device(f"cuda:{device}")
      x = torch.randn(1, 32, width, device=dev, dtype=torch.float32)
      weight = state["dec_in_proj.weight"].to(dev)
      bias = state["dec_in_proj.bias"].to(dev)
      out = F.conv1d(x, weight, bias)
      weight = weight_norm_value(state, "decoder.conv_pre").to(dev)
      out = F.conv1d(out, weight, state["decoder.conv_pre.bias"].to(dev), padding=3)
      rates, kernels, res_kernels = (5, 5, 2, 2, 2, 2, 2), (9, 9, 4, 4, 4, 4, 4), (3, 7, 11)
      for layer, (rate, kernel) in enumerate(zip(rates, kernels)):
          weight = weight_norm_value(state, f"decoder.ups.{layer}.0").to(dev)
          bias = state[f"decoder.ups.{layer}.0.bias"].to(dev)
          out = F.conv_transpose1d(out, weight, bias, stride=rate, padding=(kernel - rate) // 2)
          branches = [audio_amp_forward(out, state, f"decoder.resblocks.{layer * 3 + branch}", res_kernel) for branch, res_kernel in enumerate(res_kernels)]
          out = sum(branches) / 3
      out = audio_activation(out, state, "decoder.activation_post")
      weight = weight_norm_value(state, "decoder.conv_post").to(dev)
      out = F.conv1d(out, weight, padding=3).clamp(-1, 1)
      return {"x": x.cpu(), "output": out.cpu()}

  def release_audio_amp(pack):
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()

  def permute_qk_weight(weight):
      weight = weight.view(HEADS, HEAD_DIM, weight.shape[-1])
      leading = weight[:, :ROTARY].view(HEADS, 2, ROTARY // 2, weight.shape[-1]).transpose(1, 2)
      return torch.cat((leading.reshape(HEADS, ROTARY, weight.shape[-1]), weight[:, ROTARY:]), dim=1)

  def permute_qk_norm(weight):
      weight = weight.view(HEAD_DIM)
      leading = weight[:ROTARY].view(2, ROTARY // 2).transpose(0, 1)
      return torch.cat((leading.reshape(ROTARY), weight[ROTARY:]), dim=0)

  def rotary_tensor(position_ids):
      inv = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32.0))
      freqs = position_ids.double().unsqueeze(-1) * inv.double().view(1, 1, -1)
      freqs = torch.cat(tuple(freqs.unbind(1)), dim=-1)
      freqs = torch.cat((freqs, freqs), dim=-1)
      cos, sin = freqs.cos().float(), freqs.sin().float()
      pairwise = torch.empty(position_ids.shape[0], HEAD_DIM, dtype=torch.float32)
      pairwise[:, :ROTARY:2] = cos[:, :ROTARY // 2]
      pairwise[:, 1:ROTARY:2] = sin[:, :ROTARY // 2]
      pairwise[:, ROTARY::2] = 1
      pairwise[:, ROTARY + 1::2] = 0
      return pairwise[None, :, None]

  def apply_rope(x, position_ids):
      inv = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32, device=position_ids.device) / 32.0))
      freqs = position_ids.double().unsqueeze(-1) * inv.double().view(1, 1, -1)
      freqs = torch.cat(tuple(freqs.unbind(1)), dim=-1)
      freqs = torch.cat((freqs, freqs), dim=-1)
      cos = freqs.cos().to(x.dtype)[None, :, None]
      sin = freqs.sin().to(x.dtype)[None, :, None]
      rotated, passed = x[..., :ROTARY], x[..., ROTARY:]
      x1, x2 = rotated.chunk(2, dim=-1)
      rotated = rotated * cos + torch.cat((-x2, x1), dim=-1) * sin
      return torch.cat((rotated, passed), dim=-1)

  class Attention(nn.Module):
      def __init__(self):
          super().__init__()
          self.to_q = nn.Linear(HIDDEN, INNER, bias=False)
          self.to_k = nn.Linear(HIDDEN, INNER, bias=False)
          self.to_v = nn.Linear(HIDDEN, INNER, bias=False)
          self.norm_q = nn.RMSNorm(HEAD_DIM, eps=1e-5)
          self.norm_k = nn.RMSNorm(HEAD_DIM, eps=1e-5)
          self.to_out = nn.Linear(INNER, HIDDEN, bias=False)

      def forward(self, x, position_ids):
          q = self.norm_q(self.to_q(x).view(1, -1, HEADS, HEAD_DIM))
          k = self.norm_k(self.to_k(x).view(1, -1, HEADS, HEAD_DIM))
          v = self.to_v(x).view(1, -1, HEADS, HEAD_DIM)
          if position_ids is not None:
              q, k = apply_rope(q, position_ids), apply_rope(k, position_ids)
          out = F.scaled_dot_product_attention(
              q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
              dropout_p=0.0, is_causal=False).transpose(1, 2)
          return self.to_out(out.reshape(1, -1, INNER))

  class Block(nn.Module):
      def __init__(self):
          super().__init__()
          self.norm1 = nn.RMSNorm(HIDDEN, eps=1e-5)
          self.attn = Attention()
          self.norm2 = nn.RMSNorm(HIDDEN, eps=1e-5)
          self.ff_in = nn.Linear(HIDDEN, 2 * FFN, bias=False)
          self.ff_out = nn.Linear(FFN, HIDDEN, bias=False)
          self.adaln = nn.Linear(TIME, 6 * HIDDEN * 3, bias=True)

      def forward(self, x, temb, indices, position_ids):
          mods = self.adaln(F.silu(temb).to(self.adaln.weight.dtype)).view(-1, 6 * HIDDEN)
          shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mods.chunk(6, dim=-1)
          def rows(t): return t.index_select(0, indices)[None]
          normed = self.norm1(x)
          normed = normed * (1 + rows(scale_a)) + rows(shift_a)
          x = x + rows(gate_a) * self.attn(normed, position_ids)
          normed = self.norm2(x)
          normed = normed * (1 + rows(scale_f)) + rows(shift_f)
          up, gate = self.ff_in(normed).chunk(2, dim=-1)
          return x + rows(gate_f) * self.ff_out(up * F.silu(gate))

  def transformer_block_state(state, index):
      prefix = f"transformer_blocks.{index}."
      return {
          "norm1.weight": state[prefix + "norm1.weight"],
          "attn.to_q.weight": state[prefix + "attn.to_q.weight"],
          "attn.to_k.weight": state[prefix + "attn.to_k.weight"],
          "attn.to_v.weight": state[prefix + "attn.to_v.weight"],
          "attn.norm_q.weight": state[prefix + "attn.norm_q.weight"],
          "attn.norm_k.weight": state[prefix + "attn.norm_k.weight"],
          "attn.to_out.weight": state[prefix + "attn.to_out.0.weight"],
          "norm2.weight": state[prefix + "norm2.weight"],
          "ff_in.weight": state[prefix + "ff.net.0.proj.weight"],
          "ff_out.weight": state[prefix + "ff.net.2.weight"],
          "adaln.weight": state[prefix + "adaln_proj.linear.weight"],
          "adaln.bias": state[prefix + "adaln_proj.linear.bias"],
      }

  def load_block(root, index, device, transformer_subdir="transformer"):
      state = ShardedStateDict(os.path.join(root, transformer_subdir))
      block = Block().to(dtype=torch.bfloat16)
      block.load_state_dict(transformer_block_state(state, index), strict=True)
      block = block.to(device).eval()
      return {"block": block, "state": state}

  @torch.no_grad()
  def run_block_case(pack, sequence_length, seed, device):
      torch.manual_seed(seed)
      device = torch.device(f"cuda:{device}")
      x = torch.randn(1, sequence_length, HIDDEN, device=device, dtype=torch.bfloat16)
      # A transformer block receives the bounded output of H3's timestep MLP, not a
      # unit-variance unconstrained vector. Test that operating range here; the MLP
      # itself is validated separately in FP32.
      temb = 0.01 * torch.randn(2, TIME, device=device, dtype=torch.float32)
      tags = torch.tensor([(i % 3) for i in range(sequence_length)], device=device)
      timestep_indices = torch.tensor([(i % 2) for i in range(sequence_length)], device=device)
      indices = timestep_indices * 3 + tags
      position_ids = torch.randn(sequence_length, 3, device=device, dtype=torch.float64)
      output = pack["block"](x, temb, indices, position_ids)
      return {
          "x": x.float().cpu(),
          "temb_activated": F.silu(temb).to(torch.bfloat16).cpu(),
          "selection": F.one_hot(indices, num_classes=6).to(torch.bfloat16).cpu(),
          "rotary": rotary_tensor(position_ids.cpu()).to(torch.bfloat16),
          "output": output[0].float().cpu(),
      }

  def release_block(pack):
      pack["block"] = None
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()

  class RefinerBlock(nn.Module):
      def __init__(self):
          super().__init__()
          self.norm1 = nn.RMSNorm(HIDDEN, eps=1e-5)
          self.attn = Attention()
          self.norm2 = nn.RMSNorm(HIDDEN, eps=1e-5)
          self.ff_in = nn.Linear(HIDDEN, 2 * FFN, bias=False)
          self.ff_out = nn.Linear(FFN, HIDDEN, bias=False)

      def forward(self, x):
          x = x + self.attn(self.norm1(x), None)
          residual = x
          x = self.norm2(x)
          up, gate = self.ff_in(x).chunk(2, dim=-1)
          return residual + self.ff_out(up * F.silu(gate))

  class TokenRefiner(nn.Module):
      def __init__(self):
          super().__init__()
          self.blocks = nn.ModuleList([RefinerBlock(), RefinerBlock()])
          self.final_norm = nn.RMSNorm(HIDDEN, eps=1e-5)

      def forward(self, x):
          for block in self.blocks:
              x = block(x)
          return self.final_norm(x)

  def load_refiner(root, device):
      state = ShardedStateDict(os.path.join(root, "transformer"))
      model = TokenRefiner().to(dtype=torch.bfloat16)
      mapped = {"final_norm.weight": state["token_refiner.final_norm.weight"]}
      for index in range(2):
          src = f"token_refiner.refiner_blocks.{index}."
          dst = f"blocks.{index}."
          mapped.update({
              dst + "norm1.weight": state[src + "norm1.weight"],
              dst + "attn.to_q.weight": state[src + "attn.to_q.weight"],
              dst + "attn.to_k.weight": state[src + "attn.to_k.weight"],
              dst + "attn.to_v.weight": state[src + "attn.to_v.weight"],
              dst + "attn.norm_q.weight": state[src + "attn.norm_q.weight"],
              dst + "attn.norm_k.weight": state[src + "attn.norm_k.weight"],
              dst + "attn.to_out.weight": state[src + "attn.to_out.0.weight"],
              dst + "norm2.weight": state[src + "norm2.weight"],
              dst + "ff_in.weight": state[src + "ff.net.0.proj.weight"],
              dst + "ff_out.weight": state[src + "ff.net.2.weight"],
          })
      model.load_state_dict(mapped, strict=True)
      return {"model": model.to(device).eval(), "state": state}

  @torch.no_grad()
  def run_refiner_case(pack, sequence_length, seed, device):
      torch.manual_seed(seed)
      x = torch.randn(1, sequence_length, HIDDEN, device=f"cuda:{device}", dtype=torch.bfloat16)
      output = pack["model"](x)
      return {"x": x.float().cpu(), "output": output[0].float().cpu()}

  def release_refiner(pack):
      pack["model"] = None
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()

  class TimeEmbedding(nn.Module):
      def __init__(self):
          super().__init__()
          self.linear_1 = nn.Linear(256, 5376, bias=True)
          self.linear_2 = nn.Linear(5376, 2688, bias=True)

      def forward(self, x):
          return self.linear_2(F.silu(self.linear_1(x)))

  def representative_temb(state, device):
      model = TimeEmbedding().float()
      model.load_state_dict({
          "linear_1.weight": state["time_embedder.linear_1.weight"],
          "linear_1.bias": state["time_embedder.linear_1.bias"],
          "linear_2.weight": state["time_embedder.linear_2.weight"],
          "linear_2.bias": state["time_embedder.linear_2.bias"],
      }, strict=True)
      model = model.to(device).eval()
      timesteps = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)
      half = 128
      frequencies = torch.exp(
          -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half)
      args = timesteps[:, None] * frequencies[None]
      embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
      result = model(embedding)
      del model
      return result

  def load_time_embedding(root, device, transformer_subdir="transformer"):
      state = ShardedStateDict(os.path.join(root, transformer_subdir))
      model = TimeEmbedding().float()
      model.load_state_dict({
          "linear_1.weight": state["time_embedder.linear_1.weight"],
          "linear_1.bias": state["time_embedder.linear_1.bias"],
          "linear_2.weight": state["time_embedder.linear_2.weight"],
          "linear_2.bias": state["time_embedder.linear_2.bias"],
      }, strict=True)
      return {"model": model.to(device).eval(), "state": state}

  @torch.no_grad()
  def run_time_embedding_case(pack, device):
      timesteps = torch.tensor([0.2, 0.8], dtype=torch.float32, device=f"cuda:{device}")
      half = 128
      frequencies = torch.exp(
          -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
      args = timesteps[:, None] * frequencies[None]
      embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
      output = pack["model"](embedding)
      return {"frequencies": embedding.cpu(), "output": output.cpu()}

  def release_time_embedding(pack):
      pack["model"] = None
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()

  class NormOut(nn.Module):
      def __init__(self):
          super().__init__()
          self.norm = nn.RMSNorm(HIDDEN, eps=1e-5)
          self.linear = nn.Linear(TIME, 2 * HIDDEN, bias=True)

      def forward(self, x, temb, timestep_indices):
          shift, scale = self.linear(F.silu(temb).to(self.linear.weight.dtype)).chunk(2, dim=-1)
          return self.norm(x) * (1 + scale.index_select(0, timestep_indices)[None]) + shift.index_select(0, timestep_indices)[None]

  class OneBlockShell(nn.Module):
      def __init__(self):
          super().__init__()
          self.proj_in = nn.Linear(96, HIDDEN, bias=True).float()
          self.audio_proj_in = nn.Linear(32, HIDDEN, bias=True).float()
          self.context_embedder = nn.Linear(5120, HIDDEN, bias=True).to(torch.bfloat16)
          self.token_refiner = TokenRefiner().to(torch.bfloat16)
          self.block = Block().to(torch.bfloat16)
          self.norm_out = NormOut().to(torch.bfloat16)
          self.proj_out = nn.Linear(HIDDEN, 96, bias=True).float()
          self.audio_proj_out = nn.Linear(HIDDEN, 32, bias=True).float()

      def forward(self, video, audio, text, temb, indices, timestep_indices, position_ids):
          text = self.token_refiner(self.context_embedder(text.to(torch.bfloat16)))
          x = torch.cat((text, self.audio_proj_in(audio).to(torch.bfloat16), self.proj_in(video).to(torch.bfloat16)), dim=1)
          x = self.block(x, temb, indices, position_ids)
          x = self.norm_out(x, temb, timestep_indices).float()
          text_len, audio_len = text.shape[1], audio.shape[1]
          return self.proj_out(x[:, text_len + audio_len:]), self.audio_proj_out(x[:, text_len:text_len + audio_len])

  def load_one_block_shell(root, device, transformer_subdir="transformer"):
      state = ShardedStateDict(os.path.join(root, transformer_subdir))
      shell = OneBlockShell()
      mapped = {
          "proj_in.weight": state["proj_in.weight"], "proj_in.bias": state["proj_in.bias"],
          "audio_proj_in.weight": state["audio_proj_in.weight"], "audio_proj_in.bias": state["audio_proj_in.bias"],
          "context_embedder.weight": state["context_embedder.weight"], "context_embedder.bias": state["context_embedder.bias"],
          "token_refiner.final_norm.weight": state["token_refiner.final_norm.weight"],
          "norm_out.norm.weight": state["norm_out.norm.weight"],
          "norm_out.linear.weight": state["norm_out.linear.weight"], "norm_out.linear.bias": state["norm_out.linear.bias"],
          "proj_out.weight": state["proj_out.weight"], "proj_out.bias": state["proj_out.bias"],
          "audio_proj_out.weight": state["audio_proj_out.weight"], "audio_proj_out.bias": state["audio_proj_out.bias"],
      }
      for index in range(2):
          src, dst = f"token_refiner.refiner_blocks.{index}.", f"token_refiner.blocks.{index}."
          mapped.update({
              dst + "norm1.weight": state[src + "norm1.weight"], dst + "attn.to_q.weight": state[src + "attn.to_q.weight"],
              dst + "attn.to_k.weight": state[src + "attn.to_k.weight"], dst + "attn.to_v.weight": state[src + "attn.to_v.weight"],
              dst + "attn.norm_q.weight": state[src + "attn.norm_q.weight"], dst + "attn.norm_k.weight": state[src + "attn.norm_k.weight"],
              dst + "attn.to_out.weight": state[src + "attn.to_out.0.weight"], dst + "norm2.weight": state[src + "norm2.weight"],
              dst + "ff_in.weight": state[src + "ff.net.0.proj.weight"], dst + "ff_out.weight": state[src + "ff.net.2.weight"],
          })
      src, dst = "transformer_blocks.0.", "block."
      mapped.update({
          dst + "norm1.weight": state[src + "norm1.weight"], dst + "attn.to_q.weight": state[src + "attn.to_q.weight"],
          dst + "attn.to_k.weight": state[src + "attn.to_k.weight"], dst + "attn.to_v.weight": state[src + "attn.to_v.weight"],
          dst + "attn.norm_q.weight": state[src + "attn.norm_q.weight"], dst + "attn.norm_k.weight": state[src + "attn.norm_k.weight"],
          dst + "attn.to_out.weight": state[src + "attn.to_out.0.weight"], dst + "norm2.weight": state[src + "norm2.weight"],
          dst + "ff_in.weight": state[src + "ff.net.0.proj.weight"], dst + "ff_out.weight": state[src + "ff.net.2.weight"],
          dst + "adaln.weight": state[src + "adaln_proj.linear.weight"], dst + "adaln.bias": state[src + "adaln_proj.linear.bias"],
      })
      shell.load_state_dict(mapped, strict=True)
      return {"model": shell.to(device).eval(), "state": state}

  @torch.no_grad()
  def run_one_block_shell_case(pack, seed, device):
      torch.manual_seed(seed)
      dev = torch.device(f"cuda:{device}")
      text_len, audio_len, video_len = 2, 2, 2
      video = torch.randn(1, video_len, 96, device=dev, dtype=torch.float32)
      audio = torch.randn(1, audio_len, 32, device=dev, dtype=torch.float32)
      text = torch.randn(1, text_len, 5120, device=dev, dtype=torch.bfloat16)
      temb = representative_temb(pack["state"], dev)
      tags = torch.tensor([1, 1, 2, 2, 0, 0], device=dev)
      timestep_indices = torch.tensor([1, 1, 0, 0, 0, 0], device=dev)
      indices = timestep_indices * 3 + tags
      positions = torch.randn(text_len + audio_len + video_len, 3, device=dev, dtype=torch.float64)
      video_out, audio_out = pack["model"](video, audio, text, temb, indices, timestep_indices, positions)
      return {
          "video": video.cpu(), "audio": audio.cpu(), "text": text.float().cpu(), "temb": temb.cpu(),
          "adaln_selection": F.one_hot(indices, num_classes=6).to(torch.bfloat16).cpu(),
          "timestep_selection": F.one_hot(timestep_indices, num_classes=2).to(torch.bfloat16).cpu(),
          "rotary": rotary_tensor(positions.cpu()).to(torch.bfloat16),
          "video_output": video_out[0].cpu(), "audio_output": audio_out[0].cpu(),
      }

  @torch.no_grad()
  def run_full_shell_case(pack, layers, seed, device):
      torch.manual_seed(seed)
      dev = torch.device(f"cuda:{device}")
      model, state = pack["model"], pack["state"]
      text_len, audio_len, video_len = 2, 2, 2
      video = torch.randn(1, video_len, 96, device=dev, dtype=torch.float32)
      audio = torch.randn(1, audio_len, 32, device=dev, dtype=torch.float32)
      text = torch.randn(1, text_len, 5120, device=dev, dtype=torch.bfloat16)
      temb = representative_temb(state, dev)
      tags = torch.tensor([1, 1, 2, 2, 0, 0], device=dev)
      timestep_indices = torch.tensor([1, 1, 0, 0, 0, 0], device=dev)
      indices = timestep_indices * 3 + tags
      positions = torch.randn(text_len + audio_len + video_len, 3, device=dev, dtype=torch.float64)
      text_rows = model.token_refiner(model.context_embedder(text))
      out = torch.cat((
          text_rows, model.audio_proj_in(audio).to(torch.bfloat16),
          model.proj_in(video).to(torch.bfloat16)), dim=1)
      for index in range(layers):
          layer_model = Block().to(dtype=torch.bfloat16)
          layer_model.load_state_dict(transformer_block_state(state, index), strict=True)
          layer_model = layer_model.to(dev).eval()
          out = layer_model(out, temb, indices, positions)
          torch.cuda.synchronize(dev)
          del layer_model
      pre_norm = out.float()
      out = model.norm_out(out, temb, timestep_indices).float()
      video_out = model.proj_out(out[:, text_len + audio_len:])
      audio_out = model.audio_proj_out(out[:, text_len:text_len + audio_len])
      return {
          "video": video.cpu(), "audio": audio.cpu(), "text": text.float().cpu(), "temb": temb.cpu(),
          "adaln_selection": F.one_hot(indices, num_classes=6).to(torch.bfloat16).cpu(),
          "timestep_selection": F.one_hot(timestep_indices, num_classes=2).to(torch.bfloat16).cpu(),
          "rotary": rotary_tensor(positions.cpu()).to(torch.bfloat16),
          "pre_norm": pre_norm[0].cpu(), "hidden": out[0].cpu(),
          "video_output": video_out[0].cpu(), "audio_output": audio_out[0].cpu(),
      }

  def release_one_block_shell(pack):
      pack["model"] = None
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()

  VWIDTH, VHEADS, VHEAD_DIM, VFFN, VROTARY = 2048, 32, 64, 8192, 48

  def video_permute_weight(weight):
      weight = weight.view(VHEADS, VHEAD_DIM, weight.shape[-1])
      leading = weight[:, :VROTARY].view(VHEADS, 2, VROTARY // 2, weight.shape[-1]).transpose(1, 2)
      return torch.cat((leading.reshape(VHEADS, VROTARY, weight.shape[-1]), weight[:, VROTARY:]), dim=1)

  def video_permute_bias(bias):
      bias = bias.view(VHEADS, VHEAD_DIM)
      leading = bias[:, :VROTARY].view(VHEADS, 2, VROTARY // 2).transpose(1, 2)
      return torch.cat((leading.reshape(VHEADS, VROTARY), bias[:, VROTARY:]), dim=1)

  def video_rotary_tensor(length):
      inv = 1.0 / (100.0 ** torch.arange(0, 1, 6 / VROTARY, dtype=torch.float32))
      positions = torch.randn(1, length, 3, dtype=torch.float32)
      angles = 2 * math.pi * positions[:, :, :, None] * inv[None, None, None]
      angles = angles.flatten(2, 3).tile(2)[0]
      pairwise = torch.empty(length, VHEAD_DIM, dtype=torch.float32)
      pairwise[:, :VROTARY:2] = angles.cos()[:, :VROTARY // 2]
      pairwise[:, 1:VROTARY:2] = angles.sin()[:, :VROTARY // 2]
      pairwise[:, VROTARY::2] = 1
      pairwise[:, VROTARY + 1::2] = 0
      return pairwise[None, :, None]

  def video_apply_rope(x, rot):
      leading, passed = x[..., :VROTARY], x[..., VROTARY:]
      x1, x2 = leading.chunk(2, dim=-1)
      cos_half = rot[..., :VROTARY:2]
      sin_half = rot[..., 1:VROTARY:2]
      cos = torch.cat((cos_half, cos_half), dim=-1)
      sin = torch.cat((sin_half, sin_half), dim=-1)
      return torch.cat((leading * cos + torch.cat((-x2, x1), dim=-1) * sin, passed), dim=-1)

  class VideoDecoderBlock(nn.Module):
      def __init__(self):
          super().__init__()
          self.norm1 = nn.RMSNorm(VWIDTH, eps=1e-5)
          self.to_q = nn.Linear(VWIDTH, VWIDTH, bias=True)
          self.to_k = nn.Linear(VWIDTH, VWIDTH, bias=True)
          self.to_v = nn.Linear(VWIDTH, VWIDTH, bias=True)
          self.to_out = nn.Linear(VWIDTH, VWIDTH, bias=True)
          self.scale1 = nn.Parameter(torch.zeros(VWIDTH))
          self.norm2 = nn.RMSNorm(VWIDTH, eps=1e-5)
          self.ff_in = nn.Linear(VWIDTH, 2 * VFFN, bias=True)
          self.ff_out = nn.Linear(VFFN, VWIDTH, bias=True)
          self.scale2 = nn.Parameter(torch.zeros(VWIDTH))

      def forward(self, x, rot):
          normed = F.rms_norm(x.float(), (VWIDTH,), self.norm1.weight.float(), eps=1e-5).to(x.dtype)
          q = self.to_q(normed).view(1, -1, VHEADS, VHEAD_DIM)
          k = self.to_k(normed).view(1, -1, VHEADS, VHEAD_DIM)
          v = self.to_v(normed).view(1, -1, VHEADS, VHEAD_DIM)
          q = F.rms_norm(q.float(), (VHEAD_DIM,), None, eps=1e-5).to(q.dtype)
          k = F.rms_norm(k.float(), (VHEAD_DIM,), None, eps=1e-5).to(k.dtype)
          q, k = video_apply_rope(q, rot), video_apply_rope(k, rot)
          attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
          x = x + self.to_out(attn.reshape(1, -1, VWIDTH)) * self.scale1
          normed = F.rms_norm(x.float(), (VWIDTH,), self.norm2.weight.float(), eps=1e-5).to(x.dtype)
          up, gate = self.ff_in(normed).chunk(2, dim=-1)
          return x + self.ff_out(up * F.silu(gate)) * self.scale2

  def video_decoder_block_state(state, index):
      prefix = f"decoder.transformer_blocks.{index}."
      return {
          "norm1.weight": state[prefix + "norm1.weight"], "to_q.weight": state[prefix + "attn.to_q.weight"],
          "to_q.bias": state[prefix + "attn.to_q.bias"], "to_k.weight": state[prefix + "attn.to_k.weight"],
          "to_k.bias": state[prefix + "attn.to_k.bias"], "to_v.weight": state[prefix + "attn.to_v.weight"],
          "to_v.bias": state[prefix + "attn.to_v.bias"], "to_out.weight": state[prefix + "attn.to_out.0.weight"],
          "to_out.bias": state[prefix + "attn.to_out.0.bias"], "scale1": state[prefix + "scale1"],
          "norm2.weight": state[prefix + "norm2.weight"], "ff_in.weight": state[prefix + "ff.net.0.proj.weight"],
          "ff_in.bias": state[prefix + "ff.net.0.proj.bias"], "ff_out.weight": state[prefix + "ff.net.2.weight"],
          "ff_out.bias": state[prefix + "ff.net.2.bias"], "scale2": state[prefix + "scale2"],
      }

  def load_video_decoder_block(root, index, device):
      state = ShardedStateDict(os.path.join(root, "vae"))
      model = VideoDecoderBlock().float()
      model.load_state_dict(video_decoder_block_state(state, index), strict=True)
      return {"model": model.to(device).eval(), "state": state}

  @torch.no_grad()
  def run_video_decoder_block_case(pack, length, seed, device):
      torch.manual_seed(seed)
      dev = torch.device(f"cuda:{device}")
      x = torch.randn(1, length, VWIDTH, device=dev, dtype=torch.float16)
      rot = video_rotary_tensor(length).to(device=dev, dtype=torch.float16)
      with torch.autocast("cuda", dtype=torch.float16):
          output = pack["model"](x, rot)
      return {"x": x.float().cpu(), "rotary": rot.cpu(), "output": output[0].float().cpu()}

  @torch.no_grad()
  def run_video_decoder_shell_case(pack, patches, seed, device):
      torch.manual_seed(seed)
      dev = torch.device(f"cuda:{device}")
      state, block = pack["state"], pack["model"]
      x = torch.randn(1, patches, 24, device=dev, dtype=torch.float32)
      sequence_length = patches + 5
      rot = video_rotary_tensor(sequence_length).to(device=dev, dtype=torch.float16)
      zero = torch.zeros(1, 1, VWIDTH, device=dev, dtype=torch.float16)
      with torch.autocast("cuda", dtype=torch.float16):
          out = F.linear(
              x, state["post_quant_conv.weight"].view(24, 24).to(dev),
              state["post_quant_conv.bias"].to(dev))
          out = F.linear(
              out, state["decoder.proj_in.weight"].to(dev),
              state["decoder.proj_in.bias"].to(dev))
          registers = state["decoder.register_tokens"].to(dev).view(1, 4, VWIDTH)
          out = torch.cat((out, registers, zero), dim=1)
          out = block(out, rot)
          out = F.layer_norm(
              out.float(), (VWIDTH,), state["decoder.norm_out.weight"].to(dev),
              state["decoder.norm_out.bias"].to(dev), eps=1e-5).to(torch.float16)
          out = F.linear(
              out, state["decoder.proj_out.weight"].to(dev),
              state["decoder.proj_out.bias"].to(dev))[:, :patches]
      return {
          "x": x.cpu(), "zero": zero.cpu(), "rotary": rot.cpu(),
          "output": out[0].float().cpu(),
      }

  @torch.no_grad()
  def run_video_decoder_full_case(pack, patches, layers, seed, device):
      torch.manual_seed(seed)
      dev = torch.device(f"cuda:{device}")
      state, block = pack["state"], pack["model"]
      x = torch.randn(1, patches, 24, device=dev, dtype=torch.float32)
      sequence_length = patches + 5
      rot = video_rotary_tensor(sequence_length).to(device=dev, dtype=torch.float16)
      zero = torch.zeros(1, 1, VWIDTH, device=dev, dtype=torch.float16)
      with torch.autocast("cuda", dtype=torch.float16):
          out = F.linear(
              x, state["post_quant_conv.weight"].view(24, 24).to(dev),
              state["post_quant_conv.bias"].to(dev))
          out = F.linear(
              out, state["decoder.proj_in.weight"].to(dev),
              state["decoder.proj_in.bias"].to(dev))
          out = torch.cat((out, state["decoder.register_tokens"].to(dev).view(1, 4, VWIDTH), zero), dim=1)
          for index in range(layers):
              layer_model = VideoDecoderBlock().float()
              layer_model.load_state_dict(video_decoder_block_state(state, index), strict=True)
              layer_model = layer_model.to(dev).eval()
              out = layer_model(out, rot)
              torch.cuda.synchronize(dev)
              del layer_model
          hidden = out
          out = F.layer_norm(
              out.float(), (VWIDTH,), state["decoder.norm_out.weight"].to(dev),
              state["decoder.norm_out.bias"].to(dev), eps=1e-5).to(torch.float16)
          out = F.linear(
              out, state["decoder.proj_out.weight"].to(dev),
              state["decoder.proj_out.bias"].to(dev))[:, :patches]
      return {
          "x": x.cpu(), "zero": zero.cpu(), "rotary": rot.cpu(),
          "hidden": hidden[0].float().cpu(), "output": out[0].float().cpu(),
      }

  @torch.no_grad()
  def run_video_decoder_chain_case(pack, length, layers, seed, device):
      torch.manual_seed(seed)
      dev = torch.device(f"cuda:{device}")
      state, block = pack["state"], pack["model"]
      x = torch.randn(1, length, VWIDTH, device=dev, dtype=torch.float16)
      rot = video_rotary_tensor(length).to(device=dev, dtype=torch.float16)
      out = x
      with torch.autocast("cuda", dtype=torch.float16):
          for index in range(layers):
              block.load_state_dict(video_decoder_block_state(state, index), strict=True)
              out = block(out, rot)
              torch.cuda.synchronize(dev)
      return {"x": x.float().cpu(), "rotary": rot.cpu(), "output": out[0].float().cpu()}

  def release_video_decoder_block(pack):
      pack["model"] = None
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()

  QWIDTH = 5120
  QHEADS = 64
  QKV_HEADS = 8
  QHEAD_DIM = 128
  QFFN = 25600

  def qwen_permute_weight(weight, heads):
      return weight.view(heads, 2, QHEAD_DIM // 2, weight.shape[-1]).transpose(1, 2)

  def qwen_permute_norm(weight):
      return weight.view(2, QHEAD_DIM // 2).transpose(0, 1)

  def qwen_rotary_tensor(length):
      inv = 1.0 / (5000000.0 ** (torch.arange(0, QHEAD_DIM, 2, dtype=torch.float32) / QHEAD_DIM))
      freqs = torch.arange(length, dtype=torch.float64).unsqueeze(-1) * inv.double().unsqueeze(0)
      pairwise = torch.empty(length, QHEAD_DIM, dtype=torch.float32)
      pairwise[:, 0::2] = freqs.cos().float()
      pairwise[:, 1::2] = freqs.sin().float()
      return pairwise[None, :, None]

  def qwen_apply_rope(x, positions):
      inv = 1.0 / (5000000.0 ** (torch.arange(0, QHEAD_DIM, 2, dtype=torch.float32, device=x.device) / QHEAD_DIM))
      freqs = positions.double().unsqueeze(-1) * inv.double().unsqueeze(0)
      cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
      x1, x2 = x.chunk(2, dim=-1)
      return x * torch.cat((cos, cos), dim=-1)[None, :, None] + torch.cat((-x2, x1), dim=-1) * torch.cat((sin, sin), dim=-1)[None, :, None]

  class QwenAttention(nn.Module):
      def __init__(self):
          super().__init__()
          self.q_proj = nn.Linear(QWIDTH, QHEADS * QHEAD_DIM, bias=False)
          self.k_proj = nn.Linear(QWIDTH, QKV_HEADS * QHEAD_DIM, bias=False)
          self.v_proj = nn.Linear(QWIDTH, QKV_HEADS * QHEAD_DIM, bias=False)
          self.o_proj = nn.Linear(QHEADS * QHEAD_DIM, QWIDTH, bias=False)
          self.q_norm = nn.RMSNorm(QHEAD_DIM, eps=1e-6)
          self.k_norm = nn.RMSNorm(QHEAD_DIM, eps=1e-6)

      def forward(self, x):
          length = x.shape[1]
          q = self.q_norm(self.q_proj(x).view(1, length, QHEADS, QHEAD_DIM))
          k = self.k_norm(self.k_proj(x).view(1, length, QKV_HEADS, QHEAD_DIM))
          v = self.v_proj(x).view(1, length, QKV_HEADS, QHEAD_DIM)
          positions = torch.arange(length, dtype=torch.float64, device=x.device)
          q, k = qwen_apply_rope(q, positions), qwen_apply_rope(k, positions)
          k = k.repeat_interleave(QHEADS // QKV_HEADS, dim=2)
          v = v.repeat_interleave(QHEADS // QKV_HEADS, dim=2)
          out = F.scaled_dot_product_attention(
              q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
              dropout_p=0.0, is_causal=True).transpose(1, 2)
          return self.o_proj(out.reshape(1, length, QHEADS * QHEAD_DIM))

  class QwenBlock(nn.Module):
      def __init__(self):
          super().__init__()
          self.input_layernorm = nn.RMSNorm(QWIDTH, eps=1e-6)
          self.self_attn = QwenAttention()
          self.post_attention_layernorm = nn.RMSNorm(QWIDTH, eps=1e-6)
          self.gate_proj = nn.Linear(QWIDTH, QFFN, bias=False)
          self.up_proj = nn.Linear(QWIDTH, QFFN, bias=False)
          self.down_proj = nn.Linear(QFFN, QWIDTH, bias=False)

      def forward(self, x):
          x = x + self.self_attn(self.input_layernorm(x))
          residual = x
          x = self.post_attention_layernorm(x)
          return residual + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

  def load_qwen_block(root, index, device):
      state = ShardedStateDict(os.path.join(root, "text_encoder"), "model.safetensors.index.json")
      block = QwenBlock().to(dtype=torch.bfloat16)
      prefix = f"model.language_model.layers.{index}."
      mapped = {
          "input_layernorm.weight": state[prefix + "input_layernorm.weight"],
          "self_attn.q_proj.weight": state[prefix + "self_attn.q_proj.weight"],
          "self_attn.k_proj.weight": state[prefix + "self_attn.k_proj.weight"],
          "self_attn.v_proj.weight": state[prefix + "self_attn.v_proj.weight"],
          "self_attn.o_proj.weight": state[prefix + "self_attn.o_proj.weight"],
          "self_attn.q_norm.weight": state[prefix + "self_attn.q_norm.weight"],
          "self_attn.k_norm.weight": state[prefix + "self_attn.k_norm.weight"],
          "post_attention_layernorm.weight": state[prefix + "post_attention_layernorm.weight"],
          "gate_proj.weight": state[prefix + "mlp.gate_proj.weight"],
          "up_proj.weight": state[prefix + "mlp.up_proj.weight"],
          "down_proj.weight": state[prefix + "mlp.down_proj.weight"],
      }
      block.load_state_dict(mapped, strict=True)
      return {"block": block.to(device).eval(), "state": state}

  @torch.no_grad()
  def run_qwen_block_case(pack, sequence_length, seed, device):
      torch.manual_seed(seed)
      x = torch.randn(1, sequence_length, QWIDTH, device=f"cuda:{device}", dtype=torch.bfloat16)
      output = pack["block"](x)
      return {
          "x": x[0].float().cpu(),
          "rotary": qwen_rotary_tensor(sequence_length).to(torch.bfloat16),
          "output": output[0].float().cpu(),
      }

  def release_qwen_block(pack):
      pack["block"] = None
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()

  def conditioning_video_frame_norm(norm, x):
      batch, channels, frames, height, width = x.shape
      x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
      x = norm(x)
      return x.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

  def conditioning_video_conv(x, conv, spatial=1):
      if spatial:
          x = F.pad(x, (spatial, spatial, spatial, spatial, 0, 0), mode="reflect")
      x = F.pad(x, (0, 0, 0, 0, 2, 0), mode="constant")
      return conv(x)

  class ConditioningVideoResBlock(nn.Module):
      def __init__(self, in_channels, out_channels):
          super().__init__()
          self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
          self.conv1 = nn.Conv3d(in_channels, out_channels, 3)
          self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
          self.conv2 = nn.Conv3d(out_channels, out_channels, 3)
          if in_channels != out_channels:
              self.nin_shortcut = nn.Conv3d(in_channels, out_channels, 1)

      def forward(self, x):
          out = conditioning_video_frame_norm(self.norm1, x)
          out = conditioning_video_conv(F.silu(out), self.conv1)
          out = conditioning_video_frame_norm(self.norm2, out)
          out = conditioning_video_conv(F.silu(out), self.conv2)
          return (self.nin_shortcut(x) if hasattr(self, "nin_shortcut") else x) + out

  class ConditioningVideoDownsample(nn.Module):
      def __init__(self, channels, temporal_stride, spatial_stride):
          super().__init__()
          self.conv = nn.Conv3d(
              channels, channels, 3, stride=(temporal_stride, spatial_stride, spatial_stride))
          self.spatial_stride = spatial_stride

      def forward(self, x):
          if self.spatial_stride == 2:
              x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
          x = F.pad(x, (0, 0, 0, 0, 2, 0), mode="constant")
          return self.conv(x)

  class ConditioningVideoEncoderBody(nn.Module):
      def __init__(self):
          super().__init__()
          channels = (128, 256, 256, 512, 512, 1024)
          temporal = (1, 2, 2, 1, 1, 1)
          spatial = (2, 2, 2, 2, 1, 1)
          self.conv_in = nn.Conv3d(3, 128, 3)
          self.down = nn.ModuleList()
          current = 128
          for level, target in enumerate(channels):
              stage = nn.Module()
              stage.block = nn.ModuleList()
              for _ in range(2):
                  stage.block.append(ConditioningVideoResBlock(current, target))
                  current = target
              if temporal[level] * spatial[level] > 1:
                  stage.downsample = ConditioningVideoDownsample(
                      current, temporal[level], spatial[level])
              self.down.append(stage)
          self.norm_out = nn.GroupNorm(32, current, eps=1e-6)
          self.conv_out = nn.Conv3d(current, 48, 3)

      def forward(self, x):
          x = conditioning_video_conv(x, self.conv_in)
          for stage in self.down:
              for block in stage.block:
                  x = block(x)
              if hasattr(stage, "downsample"):
                  x = stage.downsample(x)
          x = conditioning_video_frame_norm(self.norm_out, x)
          return conditioning_video_conv(F.silu(x), self.conv_out)

  class ConditioningVideoEncoder(nn.Module):
      def __init__(self):
          super().__init__()
          self.encoder = ConditioningVideoEncoderBody()
          self.quant_conv = nn.Conv3d(48, 48, 1)

      def forward(self, x):
          return self.quant_conv(self.encoder(x))

  def load_conditioning_video_encoder(root, device):
      variant_root = os.path.join(root, "FL2VA")
      state = SingleStateDict(os.path.join(variant_root, "video_vae", "source", "model.safetensors"))
      model = ConditioningVideoEncoder().float()
      selected = {
          key: value for key, value in state.state.items()
          if key.startswith("encoder.") or key.startswith("quant_conv.")
      }
      model.load_state_dict(selected, strict=True)
      return {"model": model.to(device).eval(), "state": state}

  @torch.no_grad()
  def run_conditioning_video_encoder_case(pack, frames, height, width, seed, device):
      torch.manual_seed(seed)
      x = torch.randn(1, 3, frames, height, width, device=f"cuda:{device}", dtype=torch.float32)
      output = pack["model"](x)
      output = output.permute(0, 2, 3, 4, 1).contiguous().view(-1, 48)
      return {"x": x.cpu(), "output": output.cpu()}

  def release_conditioning_video_encoder(pack):
      pack["model"] = None
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()

  def conditioning_audio_snake(x, alpha):
      alpha = alpha.to(device=x.device, dtype=x.dtype)
      return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)

  def conditioning_audio_residual(x, state, prefix, dilation):
      alpha = state[prefix + ".block.0.alpha"].to(x.device)
      out = conditioning_audio_snake(x, alpha)
      weight = weight_norm_value(state, prefix + ".block.1").to(x.device)
      bias = state[prefix + ".block.1.bias"].to(x.device)
      out = F.conv1d(out, weight, bias, padding=3 * dilation, dilation=dilation)
      alpha = state[prefix + ".block.2.alpha"].to(x.device)
      out = conditioning_audio_snake(out, alpha)
      weight = weight_norm_value(state, prefix + ".block.3").to(x.device)
      bias = state[prefix + ".block.3.bias"].to(x.device)
      return x + F.conv1d(out, weight, bias)

  def conditioning_audio_projection(x, state):
      x = x.transpose(1, 2)
      norm1 = F.layer_norm(
          x, (2048,), state["pre_block.norm1.weight"].to(x.device),
          state["pre_block.norm1.bias"].to(x.device), 1e-5)
      qkv_bias = torch.cat((
          state["pre_block.attn.q_bias"], state["pre_block.attn.zero_k_bias"],
          state["pre_block.attn.v_bias"])).to(x.device)
      qkv = F.linear(norm1, state["pre_block.attn.qkv.weight"].to(x.device), qkv_bias)
      q, k, v = qkv.view(1, x.shape[1], 3, 8, 256).permute(2, 0, 3, 1, 4).unbind(0)
      attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
      attended = attended.mean(dim=1)
      attended = F.adaptive_avg_pool1d(attended, 32)
      attended = F.linear(
          attended, state["pre_block.attn.proj.weight"].to(x.device),
          state["pre_block.attn.proj.bias"].to(x.device))
      norm3 = F.layer_norm(
          x, (2048,), state["pre_block.norm3.weight"].to(x.device),
          state["pre_block.norm3.bias"].to(x.device), 1e-5)
      projected = F.linear(
          norm3, state["pre_block.proj.weight"].to(x.device),
          state["pre_block.proj.bias"].to(x.device))
      x = projected + attended
      norm2 = F.layer_norm(
          x, (32,), state["pre_block.norm2.weight"].to(x.device),
          state["pre_block.norm2.bias"].to(x.device), 1e-5)
      mlp = F.layer_norm(
          norm2, (32,), state["pre_block.mlp.norm.weight"].to(x.device),
          state["pre_block.mlp.norm.bias"].to(x.device), 1e-5)
      gate = F.gelu(F.linear(
          mlp, state["pre_block.mlp.w0.weight"].to(x.device),
          state["pre_block.mlp.w0.bias"].to(x.device)), approximate="tanh")
      up = F.linear(
          mlp, state["pre_block.mlp.w1.weight"].to(x.device),
          state["pre_block.mlp.w1.bias"].to(x.device))
      mlp = F.linear(
          gate * up, state["pre_block.mlp.w2.weight"].to(x.device),
          state["pre_block.mlp.w2.bias"].to(x.device))
      return x + mlp

  def load_conditioning_audio_encoder(root):
      state = SingleStateDict(os.path.join(root, "FL2VA", "audio_vae", "model.safetensors"))
      return {"state": state}

  def conditioning_audio_qkv_bias(state):
      return torch.cat((
          state["pre_block.attn.q_bias"], state["pre_block.attn.zero_k_bias"],
          state["pre_block.attn.v_bias"]))

  @torch.no_grad()
  def run_conditioning_audio_encoder_case(pack, length, seed, device):
      torch.manual_seed(seed)
      dev = torch.device(f"cuda:{device}")
      state = pack["state"]
      x = torch.randn(1, 1, length, device=dev, dtype=torch.float32)
      out = F.conv1d(
          x, weight_norm_value(state, "encoder.block.0").to(dev),
          state["encoder.block.0.bias"].to(dev), padding=3)
      rates = (2, 4, 4, 5, 5)
      for level, rate in enumerate(rates):
          prefix = f"encoder.block.{level + 1}.block"
          for unit, dilation in enumerate((1, 3, 9)):
              out = conditioning_audio_residual(out, state, f"{prefix}.{unit}", dilation)
          out = conditioning_audio_snake(out, state[f"{prefix}.3.alpha"].to(dev))
          out = F.conv1d(
              out, weight_norm_value(state, f"{prefix}.4").to(dev),
              state[f"{prefix}.4.bias"].to(dev), stride=rate, padding=math.ceil(rate / 2))
      out = conditioning_audio_snake(out, state["encoder.block.6.alpha"].to(dev))
      out = F.conv1d(
          out, weight_norm_value(state, "encoder.block.7").to(dev),
          state["encoder.block.7.bias"].to(dev), padding=1)
      encoder_output = out.transpose(1, 2)
      out = conditioning_audio_projection(out, state)
      mean = F.linear(
          out, state["mean_proj.weight"].squeeze(-1).to(dev), state["mean_proj.bias"].to(dev))
      logs = F.linear(
          out, state["logs_proj.weight"].squeeze(-1).to(dev), state["logs_proj.bias"].to(dev))
      return {
          "x": x.cpu(), "encoder_output": encoder_output[0].cpu(),
          "projected": out[0].cpu(), "output": torch.cat((mean, logs), dim=-1)[0].cpu()}

  def release_conditioning_audio_encoder(pack):
      pack["state"].release()
      gc.collect()
      torch.cuda.empty_cache()
  """#,
  h3Reference.__dict__)

func copyBF16DenseTensor(
  _ dense: Model, weight sourceWeight: PythonObject, bias sourceBias: PythonObject? = nil,
  transform: ((PythonObject) -> PythonObject)? = nil
) {
  var weight = sourceWeight
  if let transform { weight = transform(weight) }
  let weightCPU = weight.to(torch.float).cpu().numpy()
  dense.weight.copy(from: Tensor<BlockFloat>(from: try! Tensor<Float>(numpy: weightCPU)))
  dense.weight.to(.unifiedMemory)
  if let sourceBias {
    let biasCPU = sourceBias.to(torch.float).cpu().numpy()
    dense.bias.copy(from: Tensor<BlockFloat>(from: try! Tensor<Float>(numpy: biasCPU)))
    dense.bias.to(.unifiedMemory)
  }
}

func copyBF16Dense(
  _ dense: Model, state: PythonObject, weight key: String, bias biasKey: String? = nil,
  transform: ((PythonObject) -> PythonObject)? = nil
) {
  copyBF16DenseTensor(
    dense, weight: state[key], bias: biasKey.map { state[$0] }, transform: transform)
}

func copyBF16Norm(_ norm: Model, state: PythonObject, _ key: String, permute: Bool = false) {
  let source = permute ? h3Reference.permute_qk_norm(state[key]) : state[key]
  norm.weight.copy(
    from: Tensor<BlockFloat>(from: try! Tensor<Float>(numpy: source.to(torch.float).cpu().numpy())))
  norm.weight.to(.unifiedMemory)
}

func copyFloatDenseTensor(
  _ dense: Model, weight sourceWeight: PythonObject, bias sourceBias: PythonObject? = nil,
  transform: ((PythonObject) -> PythonObject)? = nil
) {
  let transformedWeight = transform.map { $0(sourceWeight) } ?? sourceWeight
  let weight = transformedWeight.to(torch.float).cpu().numpy()
  dense.weight.copy(from: try! Tensor<Float>(numpy: weight))
  dense.weight.to(.unifiedMemory)
  if let sourceBias {
    let bias = sourceBias.to(torch.float).cpu().numpy()
    dense.bias.copy(from: try! Tensor<Float>(numpy: bias))
    dense.bias.to(.unifiedMemory)
  }
}

func copyFloatDense(
  _ dense: Model, state: PythonObject, weight key: String, bias biasKey: String? = nil,
  transform: ((PythonObject) -> PythonObject)? = nil
) {
  copyFloatDenseTensor(
    dense, weight: state[key], bias: biasKey.map { state[$0] }, transform: transform)
}

private var h3ActivationDebugCounts = [String: Int]()

func h3DebugActivation(_ input: Model.IO, label: String) -> Model.IO {
  let environment = ProcessInfo.processInfo.environment
  guard environment["H3_DEBUG_ACTIVATIONS"] == "1" else {
    return input
  }
  if let prefix = environment["H3_DEBUG_ACTIVATION_PREFIX"],
    !label.hasPrefix(prefix) && !label.hasPrefix("dit.") && !label.hasPrefix("time_embedder.")
  {
    return input
  }
  return input.debug { tensors, _ in
    let tensor = Tensor<Float>(from: tensors[0]!).toCPU()
    var maximumMagnitude: Float = 0
    var squaredSum: Double = 0
    var finiteCount = 0
    var nonfiniteCount = 0
    tensor.withUnsafeBytes { bytes in
      let values = bytes.baseAddress!.assumingMemoryBound(to: Float.self)
      let count = bytes.count / MemoryLayout<Float>.stride
      for index in 0..<count {
        let value = values[index]
        if value.isFinite {
          maximumMagnitude = max(maximumMagnitude, abs(value))
          squaredSum += Double(value) * Double(value)
          finiteCount += 1
        } else {
          nonfiniteCount += 1
        }
      }
    }
    let call = h3ActivationDebugCounts[label, default: 0]
    h3ActivationDebugCounts[label] = call + 1
    let rms = finiteCount > 0 ? sqrt(squaredSum / Double(finiteCount)) : .infinity
    print(
      "H3_ACT label=\(label) call=\(call) max=\(maximumMagnitude) rms=\(rms) "
        + "finite=\(finiteCount) nonfinite=\(nonfiniteCount)")
  }
}

func H3Attention(
  prefix: String, sequenceLength: Int, rotary: Bool
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = rotary ? Input() : nil
  let toQ = Dense(count: H3Config.innerAttentionSize, noBias: true, name: "q")
  let toK = Dense(count: H3Config.innerAttentionSize, noBias: true, name: "k")
  let toV = Dense(count: H3Config.innerAttentionSize, noBias: true, name: "v")
  let normQ = RMSNorm(epsilon: H3Config.normEpsilon, axis: [3], name: "norm_q")
  let normK = RMSNorm(epsilon: H3Config.normEpsilon, axis: [3], name: "norm_k")
  let qRaw = toQ(x.to(.Float32)).reshaped(
    .NHWC(1, sequenceLength, H3Config.heads, H3Config.headDim))
  let kRaw = toK(x.to(.Float32)).reshaped(
    .NHWC(1, sequenceLength, H3Config.heads, H3Config.headDim))
  var q = normQ(h3DebugActivation(qRaw, label: "\(prefix).q_raw"))
  var k = normK(h3DebugActivation(kRaw, label: "\(prefix).k_raw"))
  let v = h3DebugActivation(
    toV(x.to(.Float32)).reshaped(
      .NHWC(1, sequenceLength, H3Config.heads, H3Config.headDim)),
    label: "\(prefix).v")
  if let rot {
    q = Functional.cmul(left: q, right: rot.to(.Float32))
    k = Functional.cmul(left: k, right: rot.to(.Float32))
  }
  // Only fused attention operates in BF16; the rest of the block remains FP32.
  let attention = ScaledDotProductAttention(
    scale: 1 / Float(H3Config.headDim).squareRoot())
  let attended = h3DebugActivation(
    attention(q.to(.BFloat16), k.to(.BFloat16), v.to(.BFloat16)).to(.Float32)
      .reshaped([1, sequenceLength, H3Config.innerAttentionSize]),
    label: "\(prefix).attended")
  let out = Dense(count: H3Config.hiddenSize, noBias: true, name: "o")
  let reader: (PythonObject) -> Void = { state in
    let transform: ((PythonObject) -> PythonObject)? =
      rotary
      ? { h3Reference.permute_qk_weight($0) } : nil
    copyFloatDense(toQ, state: state, weight: "\(prefix).to_q.weight", transform: transform)
    copyFloatDense(toK, state: state, weight: "\(prefix).to_k.weight", transform: transform)
    copyFloatDense(toV, state: state, weight: "\(prefix).to_v.weight")
    copyFloatNorm(normQ, state: state, "\(prefix).norm_q.weight", permute: rotary)
    copyFloatNorm(normK, state: state, "\(prefix).norm_k.weight", permute: rotary)
    copyFloatDense(out, state: state, weight: "\(prefix).to_out.0.weight")
  }
  if let rot {
    return (
      Model([x, rot], [h3DebugActivation(out(attended), label: "\(prefix).out_proj")]),
      reader
    )
  }
  return (Model([x], [h3DebugActivation(out(attended), label: "\(prefix).out_proj")]), reader)
}

func H3SwiGLU(prefix: String) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let up = Dense(count: H3Config.intermediateSize, noBias: true, name: "up")
  let gate = Dense(count: H3Config.intermediateSize, noBias: true, name: "gate")
  let down = Dense(count: H3Config.hiddenSize, noBias: true, name: "down")
  let input = x.to(.Float32)
  let upOut = up(input)
  let gateOut = gate(input)
  let product = h3DebugActivation(upOut .* gateOut.swish(), label: "\(prefix).product")
  let out = h3DebugActivation(down(product), label: "\(prefix).down")
  let reader: (PythonObject) -> Void = { state in
    let combined = state["\(prefix).net.0.proj.weight"]
    let upWeight = combined[..<H3Config.intermediateSize, ...]
    let gateWeight = combined[H3Config.intermediateSize..<(2 * H3Config.intermediateSize), ...]
    copyFloatDenseTensor(up, weight: upWeight)
    copyFloatDenseTensor(gate, weight: gateWeight)
    copyFloatDense(down, state: state, weight: "\(prefix).net.2.weight")
  }
  return (Model([x], [out]), reader)
}

func H3TransformerBlock(
  prefix: String, sequenceLength: Int, timestepCount: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let selection = Input()
  let activatedTimestep = Input()
  var modulationDense = [[Model]]()
  for chunk in 0..<6 {
    modulationDense.append(
      (0..<H3Config.modalityCount).map { modality in
        Dense(count: H3Config.hiddenSize, name: "adaln_\(chunk)_\(modality)")
      })
  }
  var modulations = [Model.IO]()
  for chunk in 0..<6 {
    let projected = modulationDense[chunk].enumerated().map { modality, dense in
      h3DebugActivation(
        dense(activatedTimestep), label: "\(prefix).adaln_\(chunk)_\(modality)")
    }
    var rows = [Model.IO]()
    for timestep in 0..<timestepCount {
      for modality in 0..<H3Config.modalityCount {
        rows.append(
          projected[modality].reshaped(
            [1, H3Config.hiddenSize], offset: [timestep, 0],
            strides: [H3Config.hiddenSize, 1]
          ).contiguous())
      }
    }
    var table = rows[0]
    for row in rows.dropFirst() {
      table = Functional.concat(axis: 0, table, row)
    }
    modulations.append(
      h3DebugActivation(
        Matmul()(selection.to(.Float32), table.to(.Float32))
          .reshaped([1, sequenceLength, H3Config.hiddenSize]),
        label: "\(prefix).modulation_\(chunk)"))
  }
  let norm1 = RMSNorm(epsilon: H3Config.normEpsilon, axis: [2], name: "norm1")
  let (attention, attentionReader) = H3Attention(
    prefix: "\(prefix).attn", sequenceLength: sequenceLength, rotary: true)
  let normed1 = norm1(x.to(.Float32))
  let scale1 = 1 + modulations[1]
  var out = h3DebugActivation(
    normed1 .* scale1 + modulations[0], label: "\(prefix).attn_input")
  out = x.to(.Float32) + modulations[2] .* attention(out, rot)
  out = h3DebugActivation(out, label: "\(prefix).post_attn")
  let residual = out
  let norm2 = RMSNorm(epsilon: H3Config.normEpsilon, axis: [2], name: "norm2")
  let (feedForward, feedForwardReader) = H3SwiGLU(prefix: "\(prefix).ff")
  let normed2 = norm2(out)
  let scale2 = 1 + modulations[4]
  out = normed2 .* scale2 + modulations[3]
  out = residual + modulations[5] .* feedForward(out)
  out = h3DebugActivation(out, label: "\(prefix).output")
  let reader: (PythonObject) -> Void = { state in
    copyFloatNorm(norm1, state: state, "\(prefix).norm1.weight")
    copyFloatNorm(norm2, state: state, "\(prefix).norm2.weight")
    attentionReader(state)
    feedForwardReader(state)
    let combinedWeight = state["\(prefix).adaln_proj.linear.weight"]
    let combinedBias = state["\(prefix).adaln_proj.linear.bias"]
    for chunk in 0..<6 {
      for modality in 0..<H3Config.modalityCount {
        let row = modality * 6 + chunk
        let lower = row * H3Config.hiddenSize
        let upper = lower + H3Config.hiddenSize
        let weight = combinedWeight[lower..<upper, ...]
        let bias = combinedBias[lower..<upper]
        copyFloatDenseTensor(modulationDense[chunk][modality], weight: weight, bias: bias)
      }
    }
  }
  return (Model([x, rot, selection, activatedTimestep], [out]), reader)
}

func H3TokenRefinerBlock(prefix: String, sequenceLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: H3Config.normEpsilon, axis: [2], name: "norm1")
  let (attention, attentionReader) = H3Attention(
    prefix: "\(prefix).attn", sequenceLength: sequenceLength, rotary: false)
  var out = x.to(.Float32) + attention(norm1(x.to(.Float32)))
  let norm2 = RMSNorm(epsilon: H3Config.normEpsilon, axis: [2], name: "norm2")
  let (feedForward, feedForwardReader) = H3SwiGLU(prefix: "\(prefix).ff")
  out = out + feedForward(norm2(out))
  let reader: (PythonObject) -> Void = { state in
    copyFloatNorm(norm1, state: state, "\(prefix).norm1.weight")
    copyFloatNorm(norm2, state: state, "\(prefix).norm2.weight")
    attentionReader(state)
    feedForwardReader(state)
  }
  return (Model([x], [out]), reader)
}

func H3TokenRefiner(sequenceLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  var out: Model.IO = x
  var readers = [(PythonObject) -> Void]()
  for layer in 0..<H3Config.refinerLayers {
    let (block, reader) = H3TokenRefinerBlock(
      prefix: "token_refiner.refiner_blocks.\(layer)", sequenceLength: sequenceLength)
    out = block(out)
    readers.append(reader)
  }
  let norm = RMSNorm(epsilon: H3Config.normEpsilon, axis: [2], name: "final_norm")
  out = norm(out)
  let reader: (PythonObject) -> Void = { state in
    for reader in readers { reader(state) }
    copyFloatNorm(norm, state: state, "token_refiner.final_norm.weight")
  }
  return (Model([x], [out]), reader)
}

func H3TimestepEmbedding(timestepCount: Int) -> (Model, (PythonObject) -> Void) {
  let frequencies = Input()
  let linear1 = Dense(count: H3Config.timestepHiddenSize, name: "linear_1")
  let linear2 = Dense(count: H3Config.timestepSize, name: "linear_2")
  let out = h3DebugActivation(
    linear2(linear1(frequencies).swish()), label: "time_embedder.output")
  let reader: (PythonObject) -> Void = { state in
    copyFloatDense(
      linear1, state: state, weight: "time_embedder.linear_1.weight",
      bias: "time_embedder.linear_1.bias")
    copyFloatDense(
      linear2, state: state, weight: "time_embedder.linear_2.weight",
      bias: "time_embedder.linear_2.bias")
  }
  return (Model([frequencies], [out]), reader)
}

func H3JointTransformer(
  textLength: Int, audioLength: Int, videoLength: Int, timestepCount: Int,
  layers: Int = H3Config.layers, includeHidden: Bool = false
) -> (Model, (PythonObject) -> Void) {
  let video = Input()
  let audio = Input()
  let text = Input()
  let rot = Input()
  let adalnSelection = Input()
  let timestepSelection = Input()
  let temb = Input()
  let sequenceLength = textLength + audioLength + videoLength

  // The checkpoint intentionally keeps these two projections in FP32.
  let videoInput = Dense(count: H3Config.hiddenSize, name: "proj_in")
  let audioInput = Dense(count: H3Config.hiddenSize, name: "audio_proj_in")
  let textInput = Dense(count: H3Config.hiddenSize, name: "context_embedder")
  let (refiner, refinerReader) = H3TokenRefiner(sequenceLength: textLength)
  let textRows = refiner(textInput(text.to(.Float32)))
  let audioRows = audioInput(audio)
  let videoRows = videoInput(video)
  var out = h3DebugActivation(
    Functional.concat(axis: 1, textRows, audioRows, videoRows), label: "dit.input")
  let activatedTemb = h3DebugActivation(temb.swish().to(.Float32), label: "dit.temb_activated")
  var readers = [(PythonObject) -> Void]()
  for layer in 0..<layers {
    let (block, reader) = H3TransformerBlock(
      prefix: "transformer_blocks.\(layer)", sequenceLength: sequenceLength,
      timestepCount: timestepCount)
    out = block(out, rot, adalnSelection, activatedTemb)
    readers.append(reader)
  }
  // Do not construct this diagnostic edge for export-only models. ccv counts every
  // outgoing edge during functional graph traversal, including an unreturned edge.
  let preNorm = includeHidden ? out.to(.Float32) : nil

  let outputNorm = RMSNorm(epsilon: H3Config.normEpsilon, axis: [2], name: "norm_out")
  let outputShift = Dense(count: H3Config.hiddenSize, name: "norm_out_shift")
  let outputScale = Dense(count: H3Config.hiddenSize, name: "norm_out_scale")
  let shift = Matmul()(timestepSelection.to(.Float32), outputShift(activatedTemb))
    .reshaped([1, sequenceLength, H3Config.hiddenSize])
  let scale = Matmul()(timestepSelection.to(.Float32), outputScale(activatedTemb))
    .reshaped([1, sequenceLength, H3Config.hiddenSize])
  out = h3DebugActivation(outputNorm(out) .* (1 + scale) + shift, label: "dit.norm_out")
  let audioOutRows = out.reshaped(
    [1, audioLength, H3Config.hiddenSize], offset: [0, textLength, 0],
    strides: [sequenceLength * H3Config.hiddenSize, H3Config.hiddenSize, 1]
  ).contiguous().to(.Float32)
  let videoOutRows = out.reshaped(
    [1, videoLength, H3Config.hiddenSize], offset: [0, textLength + audioLength, 0],
    strides: [sequenceLength * H3Config.hiddenSize, H3Config.hiddenSize, 1]
  ).contiguous().to(.Float32)
  let videoOutput = Dense(count: H3Config.videoPatchSize, name: "proj_out")
  let audioOutput = Dense(count: H3Config.audioChannels, name: "audio_proj_out")

  let reader: (PythonObject) -> Void = { state in
    copyFloatDense(videoInput, state: state, weight: "proj_in.weight", bias: "proj_in.bias")
    copyFloatDense(
      audioInput, state: state, weight: "audio_proj_in.weight", bias: "audio_proj_in.bias")
    copyFloatDense(
      textInput, state: state, weight: "context_embedder.weight", bias: "context_embedder.bias")
    refinerReader(state)
    for reader in readers { reader(state) }
    copyFloatNorm(outputNorm, state: state, "norm_out.norm.weight")
    let normWeight = state["norm_out.linear.weight"]
    let normBias = state["norm_out.linear.bias"]
    copyFloatDenseTensor(
      outputShift, weight: normWeight[..<H3Config.hiddenSize, ...],
      bias: normBias[..<H3Config.hiddenSize])
    copyFloatDenseTensor(
      outputScale,
      weight: normWeight[H3Config.hiddenSize..<(2 * H3Config.hiddenSize), ...],
      bias: normBias[H3Config.hiddenSize..<(2 * H3Config.hiddenSize)])
    copyFloatDense(videoOutput, state: state, weight: "proj_out.weight", bias: "proj_out.bias")
    copyFloatDense(
      audioOutput, state: state, weight: "audio_proj_out.weight", bias: "audio_proj_out.bias")
  }
  let projectedVideo = h3DebugActivation(videoOutput(videoOutRows), label: "dit.video_output")
  let projectedAudio = h3DebugActivation(audioOutput(audioOutRows), label: "dit.audio_output")
  let outputs: [Model.IO] =
    includeHidden
    ? [projectedVideo, projectedAudio, preNorm!, out.to(.Float32)]
    : [projectedVideo, projectedAudio]
  return (
    Model(
      [video, audio, text, rot, adalnSelection, timestepSelection, temb], outputs),
    reader
  )
}

func copyFloatNorm(
  _ norm: Model, state: PythonObject, _ key: String, permute: Bool = false
) {
  let source = permute ? h3Reference.permute_qk_norm(state[key]) : state[key]
  let value = source.to(torch.float).cpu().numpy()
  norm.weight.copy(from: try! Tensor<Float>(numpy: value))
  norm.weight.to(.unifiedMemory)
}

func copyVideoDecoderDense(
  _ dense: Model, state: PythonObject, weight key: String, bias biasKey: String,
  permuteQK: Bool = false
) {
  var weight = state[key]
  var bias = state[biasKey]
  if permuteQK {
    weight = h3Reference.video_permute_weight(weight)
    bias = h3Reference.video_permute_bias(bias)
  }
  let weightCPU = weight.to(torch.float).cpu().numpy()
  let biasCPU = bias.to(torch.float).cpu().numpy()
  dense.weight.copy(from: Tensor<VideoFloat>(from: try! Tensor<Float>(numpy: weightCPU)))
  dense.bias.copy(from: Tensor<VideoFloat>(from: try! Tensor<Float>(numpy: biasCPU)))
  dense.weight.to(.unifiedMemory)
  dense.bias.to(.unifiedMemory)
}

func copyVideoDecoderDenseTensor(
  _ dense: Model, weight sourceWeight: PythonObject, bias sourceBias: PythonObject
) {
  let weightCPU = sourceWeight.to(torch.float).cpu().numpy()
  let biasCPU = sourceBias.to(torch.float).cpu().numpy()
  dense.weight.copy(from: Tensor<VideoFloat>(from: try! Tensor<Float>(numpy: weightCPU)))
  dense.bias.copy(from: Tensor<VideoFloat>(from: try! Tensor<Float>(numpy: biasCPU)))
  dense.weight.to(.unifiedMemory)
  dense.bias.to(.unifiedMemory)
}

func copyH3FloatConvolution(_ convolution: Model, state: PythonObject, prefix: String) {
  let weight = state["\(prefix).weight"].to(torch.float).cpu().numpy()
  let bias = state["\(prefix).bias"].to(torch.float).cpu().numpy()
  convolution.weight.copy(from: try! Tensor<Float>(numpy: weight))
  convolution.bias.copy(from: try! Tensor<Float>(numpy: bias))
  convolution.weight.to(.unifiedMemory)
  convolution.bias.to(.unifiedMemory)
}

func h3ConditioningVideoCausalPad(_ x: Model.IO, spatial: Int = 1) -> Model.IO {
  var out = x
  if spatial > 0 {
    out = out.padded(
      .reflect, begin: [0, 0, 0, spatial, spatial],
      end: [0, 0, 0, spatial, spatial])
  }
  return out.padded(.zero, begin: [0, 0, 2, 0, 0], end: [0, 0, 0, 0, 0])
}

func h3ConditioningVideoFrameNorm(
  _ x: Model.IO, channels: Int, frames: Int, height: Int, width: Int, name: String
) -> (GroupNorm, Model.IO) {
  let norm = GroupNorm(
    axis: 1, groups: 32, epsilon: H3ConditioningVideoEncoderConfig.normEpsilon,
    reduce: [2, 3], name: name)
  let framesAsBatch = x.permuted(0, 2, 1, 3, 4).copied()
    .reshaped([frames, channels, height, width])
  let normalized = norm(framesAsBatch)
  return (
    norm,
    normalized.reshaped([1, frames, channels, height, width])
      .permuted(0, 2, 1, 3, 4).copied()
  )
}

func H3ConditioningVideoResBlock(
  prefix: String, inChannels: Int, outChannels: Int,
  frames: Int, height: Int, width: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let (norm1, normalized1) = h3ConditioningVideoFrameNorm(
    x, channels: inChannels, frames: frames, height: height, width: width, name: "norm1")
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1]), name: "conv1")
  var out = conv1(h3ConditioningVideoCausalPad(normalized1.swish()))
  let (norm2, normalized2) = h3ConditioningVideoFrameNorm(
    out, channels: outChannels, frames: frames, height: height, width: width, name: "norm2")
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1]), name: "conv2")
  out = conv2(h3ConditioningVideoCausalPad(normalized2.swish()))
  let shortcut: Convolution?
  if inChannels != outChannels {
    let projection = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1],
      hint: Hint(stride: [1, 1, 1]), name: "nin_shortcut")
    out = projection(x) + out
    shortcut = projection
  } else {
    out = x + out
    shortcut = nil
  }
  let reader: (PythonObject) -> Void = { state in
    copyFloatGroupNorm(norm1, state: state, prefix: "\(prefix).norm1")
    copyFloatGroupNorm(norm2, state: state, prefix: "\(prefix).norm2")
    copyH3FloatConvolution(conv1, state: state, prefix: "\(prefix).conv1")
    copyH3FloatConvolution(conv2, state: state, prefix: "\(prefix).conv2")
    if let shortcut {
      copyH3FloatConvolution(shortcut, state: state, prefix: "\(prefix).nin_shortcut")
    }
  }
  return (Model([x], [out]), reader)
}

func copyFloatGroupNorm(_ norm: Model, state: PythonObject, prefix: String) {
  norm.weight.copy(
    from: try! Tensor<Float>(numpy: state["\(prefix).weight"].to(torch.float).cpu().numpy()))
  norm.bias.copy(
    from: try! Tensor<Float>(numpy: state["\(prefix).bias"].to(torch.float).cpu().numpy()))
  norm.weight.to(.unifiedMemory)
  norm.bias.to(.unifiedMemory)
}

func H3ConditioningVideoEncoder(frames: Int, height: Int, width: Int)
  -> (Model, (PythonObject) -> Void)
{
  precondition(frames > 0 && height % 16 == 0 && width % 16 == 0)
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1]), name: "conv_in")
  var out = convIn(h3ConditioningVideoCausalPad(x))
  var currentChannels = 128
  var currentFrames = frames
  var currentHeight = height
  var currentWidth = width
  var readers = [(PythonObject) -> Void]()
  for level in 0..<H3ConditioningVideoEncoderConfig.channels.count {
    let targetChannels = H3ConditioningVideoEncoderConfig.channels[level]
    for blockIndex in 0..<H3ConditioningVideoEncoderConfig.blocksPerLevel {
      let (block, reader) = H3ConditioningVideoResBlock(
        prefix: "encoder.down.\(level).block.\(blockIndex)",
        inChannels: currentChannels, outChannels: targetChannels,
        frames: currentFrames, height: currentHeight, width: currentWidth)
      out = block(out)
      currentChannels = targetChannels
      readers.append(reader)
    }
    let temporalStride = H3ConditioningVideoEncoderConfig.temporalStrides[level]
    let spatialStride = H3ConditioningVideoEncoderConfig.spatialStrides[level]
    if temporalStride * spatialStride > 1 {
      let downsample = Convolution(
        groups: 1, filters: currentChannels, filterSize: [3, 3, 3],
        hint: Hint(stride: [temporalStride, spatialStride, spatialStride]),
        name: "downsample_\(level)")
      if spatialStride == 2 {
        out = out.padded(
          .reflect, begin: [0, 0, 0, 0, 0], end: [0, 0, 0, 1, 1])
      }
      out = downsample(
        out.padded(.zero, begin: [0, 0, 2, 0, 0], end: [0, 0, 0, 0, 0]))
      readers.append { state in
        copyH3FloatConvolution(
          downsample, state: state, prefix: "encoder.down.\(level).downsample.conv")
      }
      currentFrames = (currentFrames + temporalStride - 1) / temporalStride
      currentHeight /= spatialStride
      currentWidth /= spatialStride
    }
  }
  let (normOut, normalizedOut) = h3ConditioningVideoFrameNorm(
    out, channels: currentChannels, frames: currentFrames,
    height: currentHeight, width: currentWidth, name: "norm_out")
  let convOut = Convolution(
    groups: 1, filters: 48, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1]), name: "conv_out")
  out = convOut(h3ConditioningVideoCausalPad(normalizedOut.swish()))
  let quant = Convolution(
    groups: 1, filters: 48, filterSize: [1, 1, 1],
    hint: Hint(stride: [1, 1, 1]), name: "quant_conv")
  out = quant(out)
  let reader: (PythonObject) -> Void = { state in
    copyH3FloatConvolution(convIn, state: state, prefix: "encoder.conv_in")
    for reader in readers { reader(state) }
    copyFloatGroupNorm(normOut, state: state, prefix: "encoder.norm_out")
    copyH3FloatConvolution(convOut, state: state, prefix: "encoder.conv_out")
    copyH3FloatConvolution(quant, state: state, prefix: "quant_conv")
  }
  return (Model([x], [out]), reader)
}

func copyH3AudioEncoderConvolution(_ convolution: Model, state: PythonObject, prefix: String) {
  let value = h3Reference.weight_norm_value(state, prefix).to(torch.float).unsqueeze(2).cpu()
    .numpy()
  convolution.weight.copy(from: try! Tensor<Float>(numpy: value))
  convolution.bias.copy(
    from: try! Tensor<Float>(numpy: state["\(prefix).bias"].to(torch.float).cpu().numpy()))
  convolution.weight.to(.unifiedMemory)
  convolution.bias.to(.unifiedMemory)
}

func copyH3FloatLayerNorm(_ norm: Model, state: PythonObject, prefix: String) {
  norm.weight.copy(
    from: try! Tensor<Float>(numpy: state["\(prefix).weight"].to(torch.float).cpu().numpy()))
  norm.bias.copy(
    from: try! Tensor<Float>(numpy: state["\(prefix).bias"].to(torch.float).cpu().numpy()))
  norm.weight.to(.unifiedMemory)
  norm.bias.to(.unifiedMemory)
}

func H3ConditioningAudioSnake(prefix: String, channels: Int, name: String)
  -> (Model, (PythonObject) -> Void)
{
  let x = Input()
  let alpha = Parameter<Float>(
    .GPU(deviceID), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_alpha")
  let reciprocal = Parameter<Float>(
    .GPU(deviceID), .NCHW(1, channels, 1, 1), trainable: false,
    name: "\(name)_reciprocal")
  let out = x + reciprocal .* (alpha .* x).sin().pow(2)
  let reader: (PythonObject) -> Void = { state in
    let alphaValue = state["\(prefix).alpha"].to(torch.float)
    alpha.weight.copy(
      from: try! Tensor<Float>(numpy: alphaValue.view(1, -1, 1, 1).cpu().numpy()))
    reciprocal.weight.copy(
      from: try! Tensor<Float>(
        numpy: (alphaValue + 1e-9).reciprocal()
          .view(1, -1, 1, 1).cpu().numpy()))
    alpha.weight.to(.unifiedMemory)
    reciprocal.weight.to(.unifiedMemory)
  }
  return (Model([x], [out]), reader)
}

func H3ConditioningAudioResidualUnit(
  prefix: String, channels: Int, width: Int, dilation: Int, name: String
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let (snake1, snake1Reader) = H3ConditioningAudioSnake(
    prefix: "\(prefix).block.0", channels: channels, name: "\(name)_snake1")
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [1, 7], dilation: [1, dilation],
    hint: Hint(
      stride: [1, 1],
      border: Hint.Border(
        begin: [0, 3 * dilation], end: [0, 3 * dilation])), name: "\(name)_conv1")
  let (snake2, snake2Reader) = H3ConditioningAudioSnake(
    prefix: "\(prefix).block.2", channels: channels, name: "\(name)_snake2")
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), name: "\(name)_conv2")
  let out = x + conv2(snake2(conv1(snake1(x))))
  let reader: (PythonObject) -> Void = { state in
    snake1Reader(state)
    copyH3AudioEncoderConvolution(conv1, state: state, prefix: "\(prefix).block.1")
    snake2Reader(state)
    copyH3AudioEncoderConvolution(conv2, state: state, prefix: "\(prefix).block.3")
  }
  return (Model([x], [out]), reader)
}

func H3ConditioningAudioEncoderBlock(
  prefix: String, channels: Int, outputChannels: Int, width: Int, stride: Int, name: String
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  var out: Model.IO = x
  var readers = [(PythonObject) -> Void]()
  for (index, dilation) in [1, 3, 9].enumerated() {
    let (unit, reader) = H3ConditioningAudioResidualUnit(
      prefix: "\(prefix).\(index)", channels: channels, width: width,
      dilation: dilation, name: "\(name)_res\(index)")
    out = unit(out)
    readers.append(reader)
  }
  let (snake, snakeReader) = H3ConditioningAudioSnake(
    prefix: "\(prefix).3", channels: channels, name: "\(name)_snake")
  let downsample = Convolution(
    groups: 1, filters: outputChannels, filterSize: [1, 2 * stride],
    hint: Hint(
      stride: [1, stride],
      border: Hint.Border(
        begin: [0, (stride + 1) / 2], end: [0, (stride + 1) / 2])),
    name: "\(name)_downsample")
  out = downsample(snake(out))
  let reader: (PythonObject) -> Void = { state in
    for reader in readers { reader(state) }
    snakeReader(state)
    copyH3AudioEncoderConvolution(downsample, state: state, prefix: "\(prefix).4")
  }
  return (Model([x], [out]), reader)
}

func H3ConditioningAudioProjection(length: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "norm1")
  let qkv = Dense(count: 6_144, name: "qkv")
  let projectedQKV = qkv(norm1(x))
  let q = projectedQKV.reshaped(
    [1, length, 2_048], offset: [0, 0, 0],
    strides: [length * 6_144, 6_144, 1]
  ).contiguous().reshaped(.NHWC(1, length, 8, 256))
  let k = projectedQKV.reshaped(
    [1, length, 2_048], offset: [0, 0, 2_048],
    strides: [length * 6_144, 6_144, 1]
  ).contiguous().reshaped(.NHWC(1, length, 8, 256))
  let v = projectedQKV.reshaped(
    [1, length, 2_048], offset: [0, 0, 4_096],
    strides: [length * 6_144, 6_144, 1]
  ).contiguous().reshaped(.NHWC(1, length, 8, 256))
  var attended = ScaledDotProductAttention(
    scale: 1 / Float(256).squareRoot(), isCausal: true)(
      q.to(.BFloat16), k.to(.BFloat16), v.to(.BFloat16)
    ).to(.Float32)
  attended = attended.reduced(.mean, axis: [2]).reshaped([1, length, 32, 8])
    .reduced(.mean, axis: [3]).reshaped([1, length, 32])
  let attentionOutput = Dense(count: 32, name: "attention_output")
  attended = attentionOutput(attended)
  let norm3 = LayerNorm(epsilon: 1e-5, axis: [2], name: "norm3")
  let inputProjection = Dense(count: 32, name: "input_projection")
  var out = inputProjection(norm3(x)) + attended
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "norm2")
  let mlpNorm = LayerNorm(epsilon: 1e-5, axis: [2], name: "mlp_norm")
  let mlpGate = Dense(count: 64, name: "mlp_gate")
  let mlpUp = Dense(count: 64, name: "mlp_up")
  let mlpDown = Dense(count: 32, name: "mlp_down")
  let mlpInput = mlpNorm(norm2(out))
  out = out + mlpDown(mlpGate(mlpInput).GELU(approximate: .tanh) .* mlpUp(mlpInput))
  let reader: (PythonObject) -> Void = { state in
    copyH3FloatLayerNorm(norm1, state: state, prefix: "pre_block.norm1")
    copyFloatDenseTensor(
      qkv, weight: state["pre_block.attn.qkv.weight"],
      bias: h3Reference.conditioning_audio_qkv_bias(state))
    copyFloatDense(
      attentionOutput, state: state, weight: "pre_block.attn.proj.weight",
      bias: "pre_block.attn.proj.bias")
    copyH3FloatLayerNorm(norm3, state: state, prefix: "pre_block.norm3")
    copyFloatDense(
      inputProjection, state: state, weight: "pre_block.proj.weight",
      bias: "pre_block.proj.bias")
    copyH3FloatLayerNorm(norm2, state: state, prefix: "pre_block.norm2")
    copyH3FloatLayerNorm(mlpNorm, state: state, prefix: "pre_block.mlp.norm")
    copyFloatDense(
      mlpGate, state: state, weight: "pre_block.mlp.w0.weight", bias: "pre_block.mlp.w0.bias")
    copyFloatDense(
      mlpUp, state: state, weight: "pre_block.mlp.w1.weight", bias: "pre_block.mlp.w1.bias")
    copyFloatDense(
      mlpDown, state: state, weight: "pre_block.mlp.w2.weight", bias: "pre_block.mlp.w2.bias")
  }
  return (Model([x], [out]), reader)
}

func H3ConditioningAudioEncoder(inputLength: Int, includeHidden: Bool = false)
  -> (Model, (PythonObject) -> Void)
{
  precondition(inputLength > 0 && inputLength % 800 == 0)
  let x = Input()
  let input = Convolution(
    groups: 1, filters: 64, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])),
    name: "encoder_input")
  var out = input(x)
  var width = inputLength
  var channels = 64
  var readers = [(PythonObject) -> Void]()
  for (level, stride) in [2, 4, 4, 5, 5].enumerated() {
    let (block, reader) = H3ConditioningAudioEncoderBlock(
      prefix: "encoder.block.\(level + 1).block", channels: channels,
      outputChannels: channels * 2, width: width, stride: stride,
      name: "encoder_block_\(level)")
    out = block(out)
    channels *= 2
    width /= stride
    readers.append(reader)
  }
  let (finalSnake, finalSnakeReader) = H3ConditioningAudioSnake(
    prefix: "encoder.block.6", channels: channels, name: "encoder_final_snake")
  let finalConv = Convolution(
    groups: 1, filters: 2_048, filterSize: [1, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 1], end: [0, 1])),
    name: "encoder_final_conv")
  out = finalConv(finalSnake(out)).permuted(0, 3, 1, 2).copied()
    .reshaped([1, width, 2_048])
  let encoderOutput = out
  let (projection, projectionReader) = H3ConditioningAudioProjection(length: width)
  out = projection(out)
  let projected = out
  let mean = Dense(count: 32, name: "mean_proj")
  let logs = Dense(count: 32, name: "logs_proj")
  let output = Functional.concat(axis: 2, mean(out), logs(out))
  let reader: (PythonObject) -> Void = { state in
    copyH3AudioEncoderConvolution(input, state: state, prefix: "encoder.block.0")
    for reader in readers { reader(state) }
    finalSnakeReader(state)
    copyH3AudioEncoderConvolution(finalConv, state: state, prefix: "encoder.block.7")
    projectionReader(state)
    copyFloatDenseTensor(
      mean, weight: state["mean_proj.weight"].squeeze(-1), bias: state["mean_proj.bias"])
    copyFloatDenseTensor(
      logs, weight: state["logs_proj.weight"].squeeze(-1), bias: state["logs_proj.bias"])
  }
  return (
    Model([x], includeHidden ? [output, encoderOutput, projected] : [output]), reader
  )
}

func H3VideoDecoderAttention(prefix: String, sequenceLength: Int) -> (Model, (PythonObject) -> Void)
{
  let x = Input()
  let rot = Input()
  let q = Dense(count: H3VideoDecoderConfig.width, name: "to_q")
  let k = Dense(count: H3VideoDecoderConfig.width, name: "to_k")
  let v = Dense(count: H3VideoDecoderConfig.width, name: "to_v")
  let qNorm = RMSNorm(
    epsilon: H3VideoDecoderConfig.normEpsilon, axis: [3], elementwiseAffine: false)
  let kNorm = RMSNorm(
    epsilon: H3VideoDecoderConfig.normEpsilon, axis: [3], elementwiseAffine: false)
  var queries = q(x).reshaped(
    .NHWC(
      1, sequenceLength, H3VideoDecoderConfig.heads, H3VideoDecoderConfig.headDim))
  var keys = k(x).reshaped(
    .NHWC(
      1, sequenceLength, H3VideoDecoderConfig.heads, H3VideoDecoderConfig.headDim))
  let values = v(x).reshaped(
    .NHWC(
      1, sequenceLength, H3VideoDecoderConfig.heads, H3VideoDecoderConfig.headDim))
  queries = Functional.cmul(left: qNorm(queries.to(.Float32)).to(.Float16), right: rot)
  keys = Functional.cmul(left: kNorm(keys.to(.Float32)).to(.Float16), right: rot)
  let attention = ScaledDotProductAttention(
    scale: 1 / Float(H3VideoDecoderConfig.headDim).squareRoot(), flags: [.Float16])
  let attended = attention(queries, keys, values).reshaped([
    sequenceLength, H3VideoDecoderConfig.width,
  ])
  let output = Dense(count: H3VideoDecoderConfig.width, name: "to_out")
  let reader: (PythonObject) -> Void = { state in
    copyVideoDecoderDense(
      q, state: state, weight: "\(prefix).to_q.weight", bias: "\(prefix).to_q.bias",
      permuteQK: true)
    copyVideoDecoderDense(
      k, state: state, weight: "\(prefix).to_k.weight", bias: "\(prefix).to_k.bias",
      permuteQK: true)
    copyVideoDecoderDense(
      v, state: state, weight: "\(prefix).to_v.weight", bias: "\(prefix).to_v.bias")
    copyVideoDecoderDense(
      output, state: state, weight: "\(prefix).to_out.0.weight",
      bias: "\(prefix).to_out.0.bias")
  }
  return (Model([x, rot], [output(attended)]), reader)
}

func H3VideoDecoderBlock(
  prefix: String, sequenceLength: Int, deviceID: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(
    epsilon: H3VideoDecoderConfig.normEpsilon, axis: [2], name: "norm1")
  let (attention, attentionReader) = H3VideoDecoderAttention(
    prefix: "\(prefix).attn", sequenceLength: sequenceLength)
  let scale1 = Parameter<Float>(
    .GPU(deviceID), .HWC(1, 1, H3VideoDecoderConfig.width), name: "scale1")
  var out =
    x.to(.Float32)
    + attention(norm1(x.to(.Float32)).to(.Float16), rot).to(.Float32) .* scale1
  let residual = out
  let norm2 = RMSNorm(
    epsilon: H3VideoDecoderConfig.normEpsilon, axis: [2], name: "norm2")
  let up = Dense(count: H3VideoDecoderConfig.intermediateSize, name: "ff_up")
  let gate = Dense(count: H3VideoDecoderConfig.intermediateSize, name: "ff_gate")
  let down = Dense(count: H3VideoDecoderConfig.width, name: "ff_down")
  let normed = norm2(out.to(.Float32)).to(.Float16)
  let scale2 = Parameter<Float>(
    .GPU(deviceID), .HWC(1, 1, H3VideoDecoderConfig.width), name: "scale2")
  out = residual + down(up(normed) .* gate(normed).swish()).to(.Float32) .* scale2
  let reader: (PythonObject) -> Void = { state in
    copyFloatNorm(norm1, state: state, "\(prefix).norm1.weight")
    copyFloatNorm(norm2, state: state, "\(prefix).norm2.weight")
    attentionReader(state)
    let combinedWeight = state["\(prefix).ff.net.0.proj.weight"]
    let combinedBias = state["\(prefix).ff.net.0.proj.bias"]
    copyVideoDecoderDenseTensor(
      up, weight: combinedWeight[..<H3VideoDecoderConfig.intermediateSize, ...],
      bias: combinedBias[..<H3VideoDecoderConfig.intermediateSize])
    copyVideoDecoderDenseTensor(
      gate,
      weight: combinedWeight[
        H3VideoDecoderConfig.intermediateSize..<(2 * H3VideoDecoderConfig.intermediateSize), ...],
      bias: combinedBias[
        H3VideoDecoderConfig.intermediateSize..<(2 * H3VideoDecoderConfig.intermediateSize)])
    copyVideoDecoderDense(
      down, state: state, weight: "\(prefix).ff.net.2.weight", bias: "\(prefix).ff.net.2.bias")
    scale1.weight.copy(
      from: tensorFromPython(state["\(prefix).scale1"])
        .reshaped(.HWC(1, 1, H3VideoDecoderConfig.width)))
    scale1.weight.to(.unifiedMemory)
    scale2.weight.copy(
      from: tensorFromPython(state["\(prefix).scale2"])
        .reshaped(.HWC(1, 1, H3VideoDecoderConfig.width)))
    scale2.weight.to(.unifiedMemory)
  }
  return (Model([x, rot], [out]), reader)
}

func H3VideoDecoder(
  numPatches: Int, deviceID: Int, layers: Int = H3VideoDecoderConfig.layers,
  includeHidden: Bool = false
) -> (Model, (PythonObject) -> Void) {
  let latentTokens = Input()
  let rot = Input()
  let zeroToken = Input()
  let postQuant = Dense(count: H3Config.videoChannels, name: "post_quant_conv")
  let input = Dense(count: H3VideoDecoderConfig.width, name: "decoder_proj_in")
  var out = input(postQuant(latentTokens)).to(.Float32)
  let registers = Parameter<Float>(
    .GPU(deviceID), .HWC(1, H3VideoDecoderConfig.registerTokens, H3VideoDecoderConfig.width),
    name: "register_tokens")
  out = Functional.concat(axis: 1, out, registers, zeroToken)
  let sequenceLength = numPatches + H3VideoDecoderConfig.registerTokens + 1
  var readers = [(PythonObject) -> Void]()
  for layer in 0..<layers {
    let (block, reader) = H3VideoDecoderBlock(
      prefix: "decoder.transformer_blocks.\(layer)", sequenceLength: sequenceLength,
      deviceID: deviceID)
    out = block(out, rot)
    readers.append(reader)
  }
  let hidden = out
  let norm = LayerNorm(
    epsilon: H3VideoDecoderConfig.normEpsilon, axis: [2], name: "decoder_norm_out")
  out = norm(out.to(.Float32)).to(.Float16)
  let output = Dense(
    count: 3 * H3VideoDecoderConfig.temporalPatch * H3VideoDecoderConfig.spatialPatch
      * H3VideoDecoderConfig.spatialPatch,
    name: "decoder_proj_out")
  out = output(out).reshaped(
    [
      1, numPatches,
      3 * H3VideoDecoderConfig.temporalPatch
        * H3VideoDecoderConfig.spatialPatch * H3VideoDecoderConfig.spatialPatch,
    ],
    offset: [0, 0, 0],
    strides: [
      sequenceLength * 3 * H3VideoDecoderConfig.temporalPatch
        * H3VideoDecoderConfig.spatialPatch * H3VideoDecoderConfig.spatialPatch,
      3 * H3VideoDecoderConfig.temporalPatch * H3VideoDecoderConfig.spatialPatch
        * H3VideoDecoderConfig.spatialPatch, 1,
    ]
  ).contiguous()
  let reader: (PythonObject) -> Void = { state in
    copyVideoDecoderDenseTensor(
      postQuant,
      weight: state["post_quant_conv.weight"].view(
        H3Config.videoChannels, H3Config.videoChannels),
      bias: state["post_quant_conv.bias"])
    copyVideoDecoderDense(
      input, state: state, weight: "decoder.proj_in.weight", bias: "decoder.proj_in.bias")
    let registerValue = tensorFromPython(state["decoder.register_tokens"])
    registers.weight.copy(
      from: registerValue.reshaped(
        .HWC(1, H3VideoDecoderConfig.registerTokens, H3VideoDecoderConfig.width)))
    registers.weight.to(.unifiedMemory)
    for reader in readers { reader(state) }
    copyFloatNorm(norm, state: state, "decoder.norm_out.weight")
    let normBias = state["decoder.norm_out.bias"].to(torch.float).cpu().numpy()
    norm.bias.copy(from: try! Tensor<Float>(numpy: normBias))
    norm.bias.to(.unifiedMemory)
    copyVideoDecoderDense(
      output, state: state, weight: "decoder.proj_out.weight", bias: "decoder.proj_out.bias")
  }
  return (Model([latentTokens, rot, zeroToken], includeHidden ? [out, hidden] : [out]), reader)
}

func H3AudioSnakeBeta(prefix: String, channels: Int, name: String) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let alpha = Parameter<Float>(
    .GPU(deviceID), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_alpha")
  let beta = Parameter<Float>(
    .GPU(deviceID), .NCHW(1, channels, 1, 1), trainable: false, name: "\(name)_beta")
  let out = x + beta .* (x .* alpha).sin().pow(2)
  let reader: (PythonObject) -> Void = { state in
    let alphaValue = state["\(prefix).alpha"].to(torch.float).exp().view(1, -1, 1, 1).cpu()
      .numpy()
    let betaValue = (state["\(prefix).beta"].to(torch.float).exp() + 1e-9).pow(-1)
      .view(1, -1, 1, 1).cpu().numpy()
    alpha.weight.copy(from: try! Tensor<Float>(numpy: alphaValue))
    beta.weight.copy(from: try! Tensor<Float>(numpy: betaValue))
    alpha.weight.to(.unifiedMemory)
    beta.weight.to(.unifiedMemory)
  }
  return (reader, Model([x], [out]))
}

func H3AudioActivation(
  prefix: String, channels: Int, width: Int, name: String
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let ratio = 2
  let kernelSize = 12
  let upWidth = width * ratio
  let pad = kernelSize / ratio - 1
  let inputWidth = width + 2 * pad
  let rawWidth = (inputWidth - 1) * ratio + kernelSize
  let padLeft = pad * ratio + (kernelSize - ratio) / 2
  let upsample = ConvolutionTranspose(
    groups: 1, filters: 1, filterSize: [1, kernelSize], noBias: true,
    hint: Hint(stride: [1, ratio]), name: "\(name)_upsample")
  let (snakeReader, snake) = H3AudioSnakeBeta(
    prefix: "\(prefix).act", channels: channels, name: "\(name)_snake")
  let downsample = Convolution(
    groups: 1, filters: 1, filterSize: [1, kernelSize], noBias: true,
    hint: Hint(stride: [1, ratio]), name: "\(name)_downsample")
  var out = x.reshaped([channels, 1, 1, width])
  out = out.padded(.replicate, begin: [0, 0, 0, pad], end: [0, 0, 0, pad])
  out = Float(ratio) * upsample(out)
  out = out.reshaped(
    [channels, 1, 1, upWidth], offset: [0, 0, 0, padLeft],
    strides: [rawWidth, rawWidth, rawWidth, 1]
  ).contiguous()
  out = snake(out.reshaped([1, channels, 1, upWidth]))
  out = out.reshaped([channels, 1, 1, upWidth])
  out = downsample(
    out.padded(.replicate, begin: [0, 0, 0, 5], end: [0, 0, 0, 6]))
  out = out.reshaped([1, channels, 1, width])
  let reader: (PythonObject) -> Void = { state in
    snakeReader(state)
    let upFilter = state["\(prefix).upsample.filter"].to(torch.float).cpu().numpy()
    upsample.weight.copy(from: try! Tensor<Float>(numpy: upFilter))
    upsample.weight.to(.unifiedMemory)
    let downFilter = state["\(prefix).downsample.lowpass.filter"].to(torch.float).cpu().numpy()
    downsample.weight.copy(from: try! Tensor<Float>(numpy: downFilter))
    downsample.weight.to(.unifiedMemory)
  }
  return (reader, Model([x], [out]))
}

func copyH3AudioWeightNormConv(
  _ conv: Model, state: PythonObject, prefix: String, bias: Bool = true
) {
  let weight = h3Reference.weight_norm_value(state, prefix).to(torch.float).cpu().numpy()
  conv.weight.copy(from: try! Tensor<Float>(numpy: weight))
  conv.weight.to(.unifiedMemory)
  if bias {
    let value = state["\(prefix).bias"].to(torch.float).cpu().numpy()
    conv.bias.copy(from: try! Tensor<Float>(numpy: value))
    conv.bias.to(.unifiedMemory)
  }
}

func H3AudioAMPBlock(
  prefix: String, channels: Int, width: Int, kernelSize: Int, name: String
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  var out: Model.IO = x
  var readers = [(PythonObject) -> Void]()
  for (index, dilation) in [1, 3, 5].enumerated() {
    let residual = out
    let (act1Reader, act1) = H3AudioActivation(
      prefix: "\(prefix).activations.\(index * 2)", channels: channels, width: width,
      name: "\(name)_a\(index)_0")
    let conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize], dilation: [1, dilation],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(
          begin: [0, (kernelSize - 1) * dilation / 2],
          end: [0, (kernelSize - 1) * dilation / 2])), name: "\(name)_c\(index)_0")
    let (act2Reader, act2) = H3AudioActivation(
      prefix: "\(prefix).activations.\(index * 2 + 1)", channels: channels, width: width,
      name: "\(name)_a\(index)_1")
    let conv2 = Convolution(
      groups: 1, filters: channels, filterSize: [1, kernelSize],
      hint: Hint(
        stride: [1, 1],
        border: Hint.Border(
          begin: [0, (kernelSize - 1) / 2], end: [0, (kernelSize - 1) / 2])),
      name: "\(name)_c\(index)_1")
    out = residual + conv2(act2(conv1(act1(out))))
    readers.append(act1Reader)
    readers.append(act2Reader)
    readers.append { state in
      copyH3AudioWeightNormConv(conv1, state: state, prefix: "\(prefix).convs1.\(index)")
      copyH3AudioWeightNormConv(conv2, state: state, prefix: "\(prefix).convs2.\(index)")
    }
  }
  return ({ state in for reader in readers { reader(state) } }, Model([x], [out]))
}

func H3AudioDecoder(latentWidth: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let input = Convolution(
    groups: 1, filters: 2_048, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), name: "dec_in_proj")
  let pre = Convolution(
    groups: 1, filters: 1_024, filterSize: [1, 7],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])),
    name: "audio_conv_pre")
  var out = pre(input(x))
  let rates = [5, 5, 2, 2, 2, 2, 2]
  let kernels = [9, 9, 4, 4, 4, 4, 4]
  let resKernels = [3, 7, 11]
  var width = latentWidth
  var readers = [(PythonObject) -> Void]()
  for layer in 0..<rates.count {
    let channels = 1_024 / (1 << (layer + 1))
    let padding = (kernels[layer] - rates[layer]) / 2
    let up = ConvolutionTranspose(
      groups: 1, filters: channels, filterSize: [1, kernels[layer]],
      hint: Hint(
        stride: [1, rates[layer]],
        border: Hint.Border(begin: [0, padding], end: [0, padding])),
      name: "audio_up_\(layer)")
    out = up(out)
    width *= rates[layer]
    readers.append { state in
      copyH3AudioWeightNormConv(up, state: state, prefix: "decoder.ups.\(layer).0")
    }
    var branches = [Model.IO]()
    for branch in 0..<resKernels.count {
      let blockIndex = layer * resKernels.count + branch
      let (reader, block) = H3AudioAMPBlock(
        prefix: "decoder.resblocks.\(blockIndex)", channels: channels, width: width,
        kernelSize: resKernels[branch], name: "audio_amp_\(blockIndex)")
      readers.append(reader)
      branches.append(block(out))
    }
    out = (1.0 / Float(branches.count)) * branches.dropFirst().reduce(branches[0]) { $0 + $1 }
  }
  let (postActivationReader, postActivation) = H3AudioActivation(
    prefix: "decoder.activation_post", channels: 8, width: width, name: "audio_post")
  readers.append(postActivationReader)
  out = postActivation(out)
  let post = Convolution(
    groups: 1, filters: 1, filterSize: [1, 7], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 3], end: [0, 3])),
    name: "audio_conv_post")
  out = post(out).clamped(-1...1)
  let reader: (PythonObject) -> Void = { state in
    let inputWeight = state["dec_in_proj.weight"].to(torch.float).view(2_048, 32, 1, 1).cpu()
      .numpy()
    input.weight.copy(from: try! Tensor<Float>(numpy: inputWeight))
    input.bias.copy(
      from: try! Tensor<Float>(numpy: state["dec_in_proj.bias"].to(torch.float).cpu().numpy()))
    input.weight.to(.unifiedMemory)
    input.bias.to(.unifiedMemory)
    copyH3AudioWeightNormConv(pre, state: state, prefix: "decoder.conv_pre")
    for reader in readers { reader(state) }
    copyH3AudioWeightNormConv(post, state: state, prefix: "decoder.conv_post", bias: false)
  }
  return (reader, Model([x], [out]))
}

// This is the same Qwen3-VL decoder block structure already used by the Flux.2,
// Ideogram 4, Krea 2, and Anima conversions. H3 reads hidden_states[50] directly
// from the language model, so the conversion stops after block 49 in the composed
// encoder; this single-layer unit is also useful for mandatory numeric validation.
func H3QwenAttention(prefix: String, sequenceLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let q = Dense(
    count: H3QwenConfig.heads * H3QwenConfig.headDim, noBias: true, name: "q_proj")
  let k = Dense(
    count: H3QwenConfig.keyValueHeads * H3QwenConfig.headDim, noBias: true, name: "k_proj")
  let v = Dense(
    count: H3QwenConfig.keyValueHeads * H3QwenConfig.headDim, noBias: true, name: "v_proj")
  let qNorm = RMSNorm(epsilon: H3QwenConfig.normEpsilon, axis: [3], name: "q_norm")
  let kNorm = RMSNorm(epsilon: H3QwenConfig.normEpsilon, axis: [3], name: "k_norm")
  var queries = q(x.to(.Float32)).reshaped(
    .NHWC(
      1, sequenceLength, H3QwenConfig.heads, H3QwenConfig.headDim))
  var keys = k(x.to(.Float32)).reshaped(
    .NHWC(
      1, sequenceLength, H3QwenConfig.keyValueHeads, H3QwenConfig.headDim))
  let values = v(x.to(.Float32)).reshaped(
    .NHWC(
      1, sequenceLength, H3QwenConfig.keyValueHeads, H3QwenConfig.headDim))
  queries = Functional.cmul(left: qNorm(queries), right: rot.to(.Float32))
  keys = Functional.cmul(left: kNorm(keys), right: rot.to(.Float32))
  let attention = ScaledDotProductAttention(
    scale: 1 / Float(H3QwenConfig.headDim).squareRoot(), isCausal: true)
  let attended = attention(
    queries.to(.BFloat16), keys.to(.BFloat16), values.to(.BFloat16)
  ).to(.Float32).reshaped([
    sequenceLength, H3QwenConfig.heads * H3QwenConfig.headDim,
  ])
  let output = Dense(count: H3QwenConfig.hiddenSize, noBias: true, name: "o_proj")
  let reader: (PythonObject) -> Void = { state in
    copyFloatDense(
      q, state: state, weight: "\(prefix).q_proj.weight",
      transform: { h3Reference.qwen_permute_weight($0, H3QwenConfig.heads) })
    copyFloatDense(
      k, state: state, weight: "\(prefix).k_proj.weight",
      transform: { h3Reference.qwen_permute_weight($0, H3QwenConfig.keyValueHeads) })
    copyFloatDense(v, state: state, weight: "\(prefix).v_proj.weight")
    copyFloatDense(output, state: state, weight: "\(prefix).o_proj.weight")
    let qNormValue = h3Reference.qwen_permute_norm(state["\(prefix).q_norm.weight"])
    qNorm.weight.copy(from: try! Tensor<Float>(numpy: qNormValue.to(torch.float).cpu().numpy()))
    qNorm.weight.to(.unifiedMemory)
    let kNormValue = h3Reference.qwen_permute_norm(state["\(prefix).k_norm.weight"])
    kNorm.weight.copy(from: try! Tensor<Float>(numpy: kNormValue.to(torch.float).cpu().numpy()))
    kNorm.weight.to(.unifiedMemory)
  }
  return (Model([x, rot], [output(attended)]), reader)
}

func H3QwenBlock(prefix: String, sequenceLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let inputNorm = RMSNorm(
    epsilon: H3QwenConfig.normEpsilon, axis: [1], name: "input_layernorm")
  let (attention, attentionReader) = H3QwenAttention(
    prefix: "\(prefix).self_attn", sequenceLength: sequenceLength)
  var out = x.to(.Float32) + attention(inputNorm(x.to(.Float32)), rot)
  let residual = out
  let postAttentionNorm = RMSNorm(
    epsilon: H3QwenConfig.normEpsilon, axis: [1], name: "post_attention_layernorm")
  out = postAttentionNorm(out)
  let gate = Dense(count: H3QwenConfig.intermediateSize, noBias: true, name: "gate_proj")
  let up = Dense(count: H3QwenConfig.intermediateSize, noBias: true, name: "up_proj")
  let down = Dense(count: H3QwenConfig.hiddenSize, noBias: true, name: "down_proj")
  let gateOut = gate(out)
  let upOut = up(out)
  out = residual + down(gateOut.swish() .* upOut)
  let reader: (PythonObject) -> Void = { state in
    attentionReader(state)
    copyFloatNorm(inputNorm, state: state, "\(prefix).input_layernorm.weight")
    copyFloatNorm(
      postAttentionNorm, state: state, "\(prefix).post_attention_layernorm.weight")
    copyFloatDense(gate, state: state, weight: "\(prefix).mlp.gate_proj.weight")
    copyFloatDense(up, state: state, weight: "\(prefix).mlp.up_proj.weight")
    copyFloatDense(down, state: state, weight: "\(prefix).mlp.down_proj.weight")
  }
  return (Model([x, rot], [out]), reader)
}

func H3QwenTextEncoder(sequenceLength: Int) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let embedding = Embedding(
    BlockFloat.self, vocabularySize: H3QwenConfig.vocabularySize,
    embeddingSize: H3QwenConfig.hiddenSize, name: "tok_embeddings")
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  // hidden_states[0] is the embedding. H3 consumes hidden_states[50], hence the
  // un-normalized residual stream after decoder block 49.
  for layer in 0..<H3QwenConfig.featureLayer {
    let prefix = "model.language_model.layers.\(layer)"
    let (block, reader) = H3QwenBlock(prefix: prefix, sequenceLength: sequenceLength)
    out = block(out, rot)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state in
    let value = state["model.language_model.embed_tokens.weight"].to(torch.float).cpu().numpy()
    embedding.parameters.copy(
      from: Tensor<BlockFloat>(from: try! Tensor<Float>(numpy: value)))
    embedding.parameters.to(.unifiedMemory)
    for reader in readers { reader(state) }
  }
  return (Model([tokens, rot], [out]), reader)
}

func copyQwenExportDense(
  _ dense: Model, state: PythonObject, weight key: String,
  transform: ((PythonObject) -> PythonObject)? = nil
) {
  let source = transform.map { $0(state[key]) } ?? state[key]
  dense.weight.copy(
    from: Tensor<Float16>(
      from: try! Tensor<Float>(numpy: source.to(torch.float).cpu().numpy())))
  dense.weight.to(.unifiedMemory)
}

func copyQwenExportNorm(
  _ norm: Model, state: PythonObject, _ key: String, permute: Bool = false
) {
  let source = permute ? h3Reference.qwen_permute_norm(state[key]) : state[key]
  norm.weight.copy(
    from: Tensor<Float16>(
      from: try! Tensor<Float>(numpy: source.to(torch.float).cpu().numpy())))
  norm.weight.to(.unifiedMemory)
}

// Standard Qwen3-VL FP16 store model. Names deliberately match the existing
// qwen_3_vl_4b / qwen_3_vl_8b stores under /slow/Data.
func H3QwenExportAttention(prefix: String, sequenceLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let k = Dense(
    count: H3QwenConfig.keyValueHeads * H3QwenConfig.headDim, noBias: true, name: "k_proj")
  let q = Dense(
    count: H3QwenConfig.heads * H3QwenConfig.headDim, noBias: true, name: "q_proj")
  let v = Dense(
    count: H3QwenConfig.keyValueHeads * H3QwenConfig.headDim, noBias: true, name: "v_proj")
  var keys = k(x).reshaped(
    .NHWC(
      1, sequenceLength, H3QwenConfig.keyValueHeads, H3QwenConfig.headDim))
  let normK = RMSNorm(epsilon: H3QwenConfig.normEpsilon, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries = q(x).reshaped(
    .NHWC(
      1, sequenceLength, H3QwenConfig.heads, H3QwenConfig.headDim))
  let normQ = RMSNorm(epsilon: H3QwenConfig.normEpsilon, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = v(x).reshaped(
    .NHWC(
      1, sequenceLength, H3QwenConfig.keyValueHeads, H3QwenConfig.headDim))
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  let attended = ScaledDotProductAttention(
    scale: 1 / Float(H3QwenConfig.headDim).squareRoot(), isCausal: true)(
      queries, keys, values
    ).reshaped([sequenceLength, H3QwenConfig.heads * H3QwenConfig.headDim])
  let output = Dense(count: H3QwenConfig.hiddenSize, noBias: true, name: "out_proj")
  let reader: (PythonObject) -> Void = { state in
    copyQwenExportDense(
      q, state: state, weight: "\(prefix).self_attn.q_proj.weight",
      transform: { h3Reference.qwen_permute_weight($0, H3QwenConfig.heads) })
    copyQwenExportDense(
      k, state: state, weight: "\(prefix).self_attn.k_proj.weight",
      transform: { h3Reference.qwen_permute_weight($0, H3QwenConfig.keyValueHeads) })
    copyQwenExportDense(v, state: state, weight: "\(prefix).self_attn.v_proj.weight")
    copyQwenExportDense(output, state: state, weight: "\(prefix).self_attn.o_proj.weight")
    copyQwenExportNorm(
      normQ, state: state, "\(prefix).self_attn.q_norm.weight", permute: true)
    copyQwenExportNorm(
      normK, state: state, "\(prefix).self_attn.k_norm.weight", permute: true)
  }
  return (Model([x, rot], [output(attended)]), reader)
}

func H3QwenExportBlock(prefix: String, sequenceLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let inputNorm = RMSNorm(
    epsilon: H3QwenConfig.normEpsilon, axis: [1], name: "input_layernorm")
  let (attention, attentionReader) = H3QwenExportAttention(
    prefix: prefix, sequenceLength: sequenceLength)
  var out = x + attention(inputNorm(x), rot)
  let residual = out
  let postNorm = RMSNorm(
    epsilon: H3QwenConfig.normEpsilon, axis: [1], name: "post_attention_layernorm")
  let mlpInput = Input()
  let gate = Dense(
    count: H3QwenConfig.intermediateSize, noBias: true, name: "mlp_gate_proj")
  let up = Dense(count: H3QwenConfig.intermediateSize, noBias: true, name: "mlp_up_proj")
  let down = Dense(count: H3QwenConfig.hiddenSize, noBias: true, name: "mlp_down_proj")
  let feedForward = Model(
    [mlpInput], [down(up(mlpInput) .* gate(mlpInput).swish())], name: "mlp")
  out = residual + feedForward(postNorm(out))
  let reader: (PythonObject) -> Void = { state in
    attentionReader(state)
    copyQwenExportNorm(inputNorm, state: state, "\(prefix).input_layernorm.weight")
    copyQwenExportNorm(postNorm, state: state, "\(prefix).post_attention_layernorm.weight")
    copyQwenExportDense(gate, state: state, weight: "\(prefix).mlp.gate_proj.weight")
    copyQwenExportDense(up, state: state, weight: "\(prefix).mlp.up_proj.weight")
    copyQwenExportDense(down, state: state, weight: "\(prefix).mlp.down_proj.weight")
  }
  return (Model([x, rot], [out]), reader)
}

func H3QwenExportModel(sequenceLength: Int) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let embedding = Embedding(
    Float16.self, vocabularySize: H3QwenConfig.vocabularySize,
    embeddingSize: H3QwenConfig.hiddenSize, name: "tok_embeddings")
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  for layer in 0..<H3QwenConfig.featureLayer {
    let prefix = "model.language_model.layers.\(layer)"
    let (block, reader) = H3QwenExportBlock(prefix: prefix, sequenceLength: sequenceLength)
    out = block(out, rot)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state in
    let value = state["model.language_model.embed_tokens.weight"].to(torch.float).cpu().numpy()
    embedding.parameters.copy(from: Tensor<Float16>(from: try! Tensor<Float>(numpy: value)))
    embedding.parameters.to(.unifiedMemory)
    for reader in readers { reader(state) }
  }
  return (Model([tokens, rot], [out]), reader)
}

func tensorFromPython(_ object: PythonObject) -> Tensor<Float> {
  try! Tensor<Float>(numpy: object.to(torch.float).cpu().numpy())
}

func runQwenBlockParity() -> Bool {
  let sequenceLength = Int(environment["MINIMAX_H3_TEST_TOKENS"] ?? "6") ?? 6
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-qwen-block outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let layer = H3QwenConfig.featureLayer - 1
  let pack = h3Reference.load_qwen_block(modelRoot, layer, deviceID)
  let testCase = h3Reference.run_qwen_block_case(pack, sequenceLength, 43, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let reference = tensorFromPython(testCase["output"])
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(Tensor<BlockFloat>(from: xCPU).toGPU(deviceID))
    let rotary = graph.variable(
      Tensor<BlockFloat>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, sequenceLength, 1, H3QwenConfig.headDim)))
    let prefix = "model.language_model.layers.\(layer)"
    let (block, reader) = H3QwenBlock(prefix: prefix, sequenceLength: sequenceLength)
    block.maxConcurrency = .limit(1)
    print("MiniMax-H3 Qwen layer \(layer): compiling Swift graph")
    block.compile(inputs: x, rotary)
    print("MiniMax-H3 Qwen layer \(layer): loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 Qwen layer \(layer): executing Swift graph")
    let output = Tensor<Float>(
      from: block(inputs: x, rotary)[0].as(of: Float.self).rawValue.toCPU()
        .reshaped(.WC(sequenceLength, H3QwenConfig.hiddenSize)))
    print("MiniMax-H3 Qwen layer \(layer): Swift graph complete")
    return output
  }
  h3Reference.release_qwen_block(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 Qwen feature block \(layer)", metrics)
  return metrics.maxRelativeDifference <= 0.02 && metrics.minimumCosineSimilarity >= 0.99
    && metrics.meanCosineSimilarity >= 0.999
}

func runTokenRefinerParity() -> Bool {
  let sequenceLength = Int(environment["MINIMAX_H3_TEST_TOKENS"] ?? "6") ?? 6
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-refiner outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_refiner(modelRoot, deviceID)
  let testCase = h3Reference.run_refiner_case(pack, sequenceLength, 44, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let reference = tensorFromPython(testCase["output"])
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(Tensor<BlockFloat>(from: xCPU).toGPU(deviceID))
    let (refiner, reader) = H3TokenRefiner(sequenceLength: sequenceLength)
    refiner.maxConcurrency = .limit(1)
    print("MiniMax-H3 token refiner: compiling Swift graph")
    refiner.compile(inputs: x)
    print("MiniMax-H3 token refiner: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 token refiner: executing Swift graph")
    let output = Tensor<Float>(
      from: refiner(inputs: x)[0].as(of: Float.self).rawValue.toCPU()
        .reshaped(.WC(sequenceLength, H3Config.hiddenSize)))
    print("MiniMax-H3 token refiner: Swift graph complete")
    return output
  }
  h3Reference.release_refiner(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 two-block token refiner", metrics)
  return metrics.maxRelativeDifference <= 0.02 && metrics.minimumCosineSimilarity >= 0.99
    && metrics.meanCosineSimilarity >= 0.999
}

func runTimestepEmbeddingParity(transformerSubdirectory: String = "transformer") -> Bool {
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-time outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_time_embedding(modelRoot, deviceID, transformerSubdirectory)
  let testCase = h3Reference.run_time_embedding_case(pack, deviceID)
  let state = pack["state"]
  let frequenciesCPU = tensorFromPython(testCase["frequencies"])
  let reference = tensorFromPython(testCase["output"])
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let frequencies = graph.variable(frequenciesCPU.toGPU(deviceID))
    let (timeEmbedding, reader) = H3TimestepEmbedding(timestepCount: 2)
    timeEmbedding.maxConcurrency = .limit(1)
    print("MiniMax-H3 timestep MLP: compiling Swift graph")
    timeEmbedding.compile(inputs: frequencies)
    print("MiniMax-H3 timestep MLP: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 timestep MLP: executing Swift graph")
    let output = timeEmbedding(inputs: frequencies)[0].as(of: Float.self).rawValue.toCPU()
      .reshaped(.WC(2, H3Config.timestepSize))
    print("MiniMax-H3 timestep MLP: Swift graph complete")
    return output
  }
  h3Reference.release_time_embedding(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 FP32 timestep MLP", metrics)
  return metrics.maxRelativeDifference <= 1e-5 && metrics.minimumCosineSimilarity >= 0.99999
    && metrics.meanCosineSimilarity >= 0.99999
}

func runOneBlockShellParity() -> Bool {
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-shell outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_one_block_shell(modelRoot, deviceID)
  let testCase = h3Reference.run_one_block_shell_case(pack, 45, deviceID)
  let state = pack["state"]
  let videoCPU = tensorFromPython(testCase["video"])
  let audioCPU = tensorFromPython(testCase["audio"])
  let textCPU = tensorFromPython(testCase["text"])
  let tembCPU = tensorFromPython(testCase["temb"])
  let adalnSelectionCPU = tensorFromPython(testCase["adaln_selection"])
  let timestepSelectionCPU = tensorFromPython(testCase["timestep_selection"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let referenceVideo = tensorFromPython(testCase["video_output"])
  let referenceAudio = tensorFromPython(testCase["audio_output"])
  let outputs = graph.withNoGrad { () -> (Tensor<Float>, Tensor<Float>) in
    let video = graph.variable(videoCPU.toGPU(deviceID))
    let audio = graph.variable(audioCPU.toGPU(deviceID))
    let text = graph.variable(Tensor<BlockFloat>(from: textCPU).toGPU(deviceID))
    let temb = graph.variable(tembCPU.toGPU(deviceID))
    let adalnSelection = graph.variable(
      Tensor<BlockFloat>(from: adalnSelectionCPU).toGPU(deviceID))
    let timestepSelection = graph.variable(
      Tensor<BlockFloat>(from: timestepSelectionCPU).toGPU(deviceID))
    let rotary = graph.variable(
      Tensor<VideoFloat>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, 6, 1, H3Config.headDim)))
    let (shell, reader) = H3JointTransformer(
      textLength: 2, audioLength: 2, videoLength: 2, timestepCount: 2, layers: 1)
    shell.maxConcurrency = .limit(1)
    print("MiniMax-H3 one-block T2VA shell: compiling Swift graph")
    shell.compile(inputs: video, audio, text, rotary, adalnSelection, timestepSelection, temb)
    print("MiniMax-H3 one-block T2VA shell: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 one-block T2VA shell: executing Swift graph")
    let result = shell(
      inputs: video, audio, text, rotary, adalnSelection, timestepSelection, temb)
    let videoOutput = result[0].as(of: Float.self).rawValue.toCPU().reshaped(.WC(2, 96))
    let audioOutput = result[1].as(of: Float.self).rawValue.toCPU().reshaped(.WC(2, 32))
    print("MiniMax-H3 one-block T2VA shell: Swift graph complete")
    return (videoOutput, audioOutput)
  }
  h3Reference.release_one_block_shell(pack)
  let videoMetrics = tokenParityMetrics(outputs.0, referenceVideo)
  let audioMetrics = tokenParityMetrics(outputs.1, referenceAudio)
  printMetrics("MiniMax-H3 one-block shell video", videoMetrics)
  printMetrics("MiniMax-H3 one-block shell audio", audioMetrics)
  return videoMetrics.maxRelativeDifference <= 0.03
    && videoMetrics.minimumCosineSimilarity >= 0.99
    && videoMetrics.meanCosineSimilarity >= 0.999
    && audioMetrics.maxRelativeDifference <= 0.03
    && audioMetrics.minimumCosineSimilarity >= 0.99
    && audioMetrics.meanCosineSimilarity >= 0.999
}

func runFullJointTransformerParity(
  transformerSubdirectory: String = "transformer", allowRef2VAAudioAmplification: Bool = false
) -> Bool {
  let layers = Int(environment["MINIMAX_H3_TRANSFORMER_LAYERS"] ?? "50") ?? 50
  precondition(layers > 0 && layers <= H3Config.layers)
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-transformer-full outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_one_block_shell(modelRoot, deviceID, transformerSubdirectory)
  print("MiniMax-H3 full joint transformer: running Python reference with", layers, "blocks")
  let testCase = h3Reference.run_full_shell_case(pack, layers, 52, deviceID)
  let state = pack["state"]
  let videoCPU = tensorFromPython(testCase["video"])
  let audioCPU = tensorFromPython(testCase["audio"])
  let textCPU = tensorFromPython(testCase["text"])
  let tembCPU = tensorFromPython(testCase["temb"])
  let adalnSelectionCPU = tensorFromPython(testCase["adaln_selection"])
  let timestepSelectionCPU = tensorFromPython(testCase["timestep_selection"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let referenceVideo = tensorFromPython(testCase["video_output"])
  let referenceAudio = tensorFromPython(testCase["audio_output"])
  let referencePreNorm = tensorFromPython(testCase["pre_norm"])
  let referenceHidden = tensorFromPython(testCase["hidden"])
  let outputs = graph.withNoGrad {
    () -> (Tensor<Float>, Tensor<Float>, Tensor<Float>, Tensor<Float>) in
    let video = graph.variable(videoCPU.toGPU(deviceID))
    let audio = graph.variable(audioCPU.toGPU(deviceID))
    let text = graph.variable(Tensor<BlockFloat>(from: textCPU).toGPU(deviceID))
    let temb = graph.variable(tembCPU.toGPU(deviceID))
    let adalnSelection = graph.variable(
      Tensor<BlockFloat>(from: adalnSelectionCPU).toGPU(deviceID))
    let timestepSelection = graph.variable(
      Tensor<BlockFloat>(from: timestepSelectionCPU).toGPU(deviceID))
    let rotary = graph.variable(
      Tensor<BlockFloat>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, 6, 1, H3Config.headDim)))
    let (transformer, reader) = H3JointTransformer(
      textLength: 2, audioLength: 2, videoLength: 2, timestepCount: 2,
      layers: layers, includeHidden: true)
    transformer.maxConcurrency = .limit(1)
    print("MiniMax-H3 full joint transformer: compiling Swift graph")
    transformer.compile(
      inputs: video, audio, text, rotary, adalnSelection, timestepSelection, temb)
    print("MiniMax-H3 full joint transformer: loading", layers, "blocks into unified memory")
    reader(state)
    print("MiniMax-H3 full joint transformer: executing Swift graph")
    let result = transformer(
      inputs: video, audio, text, rotary, adalnSelection, timestepSelection, temb)
    let videoOutput = result[0].as(of: Float.self).rawValue.toCPU().reshaped(.WC(2, 96))
    let audioOutput = result[1].as(of: Float.self).rawValue.toCPU().reshaped(.WC(2, 32))
    let preNorm = result[2].as(of: Float.self).rawValue.toCPU()
      .reshaped(.WC(6, H3Config.hiddenSize))
    let hidden = result[3].as(of: Float.self).rawValue.toCPU()
      .reshaped(.WC(6, H3Config.hiddenSize))
    print("MiniMax-H3 full joint transformer: Swift graph complete")
    return (videoOutput, audioOutput, preNorm, hidden)
  }
  h3Reference.release_one_block_shell(pack)
  let preNormMetrics = tokenParityMetrics(outputs.2, referencePreNorm)
  let hiddenMetrics = tokenParityMetrics(outputs.3, referenceHidden)
  let videoMetrics = tokenParityMetrics(outputs.0, referenceVideo)
  let audioMetrics = tokenParityMetrics(outputs.1, referenceAudio)
  printMetrics("MiniMax-H3 joint pre-norm (\(layers) blocks)", preNormMetrics)
  printMetrics("MiniMax-H3 joint post-norm (\(layers) blocks)", hiddenMetrics)
  printMetrics("MiniMax-H3 joint video head (\(layers) blocks)", videoMetrics)
  printMetrics("MiniMax-H3 joint audio head (\(layers) blocks)", audioMetrics)
  // The independently seeded block checks remain the strict conversion gate. At full
  // depth, backend rounding compounds through a high-gain synthetic hidden stream, so
  // judge the externally consumed denoiser heads using scale-aware error and direction.
  let audioNRMSELimit: Float = allowRef2VAAudioAmplification ? 0.25 : 0.20
  let audioMeanCosineLimit: Float = allowRef2VAAudioAmplification ? 0.975 : 0.98
  return videoMetrics.allFinite && audioMetrics.allFinite
    && videoMetrics.maxRelativeDifference <= 0.15
    && videoMetrics.normalizedRootMeanSquareError <= 0.10
    && videoMetrics.minimumCosineSimilarity >= 0.99
    && videoMetrics.meanCosineSimilarity >= 0.995
    && audioMetrics.maxRelativeDifference <= 0.25
    && audioMetrics.normalizedRootMeanSquareError <= audioNRMSELimit
    && audioMetrics.minimumCosineSimilarity >= 0.97
    && audioMetrics.meanCosineSimilarity >= audioMeanCosineLimit
}

func runVideoDecoderBlockParity() -> Bool {
  let sequenceLength = Int(environment["MINIMAX_H3_TEST_TOKENS"] ?? "6") ?? 6
  let blockIndex = Int(environment["MINIMAX_H3_VIDEO_BLOCK_INDEX"] ?? "0") ?? 0
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-video-block outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_video_decoder_block(modelRoot, blockIndex, deviceID)
  let testCase = h3Reference.run_video_decoder_block_case(pack, sequenceLength, 46, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let reference = tensorFromPython(testCase["output"])
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(Tensor<VideoFloat>(from: xCPU).toGPU(deviceID))
    let rotary = graph.variable(
      Tensor<VideoFloat>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, sequenceLength, 1, H3VideoDecoderConfig.headDim)))
    let prefix = "decoder.transformer_blocks.\(blockIndex)"
    let (block, reader) = H3VideoDecoderBlock(
      prefix: prefix, sequenceLength: sequenceLength, deviceID: deviceID)
    block.maxConcurrency = .limit(1)
    print("MiniMax-H3 video decoder block \(blockIndex): compiling Swift graph")
    block.compile(inputs: x, rotary)
    print("MiniMax-H3 video decoder block \(blockIndex): loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 video decoder block \(blockIndex): executing Swift graph")
    let output = Tensor<Float>(
      from: block(inputs: x, rotary)[0].as(of: Float.self).rawValue.toCPU()
        .reshaped(.WC(sequenceLength, H3VideoDecoderConfig.width)))
    print("MiniMax-H3 video decoder block \(blockIndex): Swift graph complete")
    return output
  }
  h3Reference.release_video_decoder_block(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 video decoder block \(blockIndex)", metrics)
  return metrics.maxRelativeDifference <= 0.02 && metrics.minimumCosineSimilarity >= 0.99
    && metrics.meanCosineSimilarity >= 0.999
}

func runConditioningVideoEncoderParity() -> Bool {
  let frames = Int(environment["MINIMAX_H3_VIDEO_ENCODER_FRAMES"] ?? "5") ?? 5
  let height = Int(environment["MINIMAX_H3_VIDEO_ENCODER_HEIGHT"] ?? "32") ?? 32
  let width = Int(environment["MINIMAX_H3_VIDEO_ENCODER_WIDTH"] ?? "32") ?? 32
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-video-encoder outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_conditioning_video_encoder(modelRoot, deviceID)
  let testCase = h3Reference.run_conditioning_video_encoder_case(
    pack, frames, height, width, 71, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let reference = tensorFromPython(testCase["output"])
  let outputFrames = (frames + 3) / 4
  let outputHeight = height / 16
  let outputWidth = width / 16
  let tokenCount = outputFrames * outputHeight * outputWidth
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(xCPU.toGPU(deviceID))
    let (encoder, reader) = H3ConditioningVideoEncoder(
      frames: frames, height: height, width: width)
    encoder.maxConcurrency = .limit(1)
    print("MiniMax-H3 conditioning video encoder: compiling Swift graph")
    encoder.compile(inputs: x)
    print("MiniMax-H3 conditioning video encoder: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 conditioning video encoder: executing Swift graph")
    let output = encoder(inputs: x)[0].as(of: Float.self)
      .permuted(0, 2, 3, 4, 1).copied().rawValue.toCPU()
      .reshaped(.WC(tokenCount, 48))
    print("MiniMax-H3 conditioning video encoder: Swift graph complete")
    return output
  }
  h3Reference.release_conditioning_video_encoder(pack)
  let expected = reference.reshaped(.WC(tokenCount, 48))
  let metrics = tokenParityMetrics(result, expected)
  printMetrics("MiniMax-H3 FP32 conditioning video encoder", metrics)
  return metrics.maxRelativeDifference <= 1e-3
    && metrics.normalizedRootMeanSquareError <= 1e-3
    && metrics.minimumCosineSimilarity >= 0.9999
    && metrics.meanCosineSimilarity >= 0.99999
}

func runConditioningAudioEncoderParity() -> Bool {
  let inputLength = Int(environment["MINIMAX_H3_AUDIO_ENCODER_SAMPLES"] ?? "1600") ?? 1600
  precondition(inputLength % 800 == 0)
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-audio-encoder outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_conditioning_audio_encoder(modelRoot)
  let testCase = h3Reference.run_conditioning_audio_encoder_case(
    pack, inputLength, 73, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let reference = tensorFromPython(testCase["output"])
    .reshaped(.WC(inputLength / 800, 64))
  let encoderReference = tensorFromPython(testCase["encoder_output"])
    .reshaped(.WC(inputLength / 800, 2_048))
  let projectedReference = tensorFromPython(testCase["projected"])
    .reshaped(.WC(inputLength / 800, 32))
  var encoderResult = Tensor<Float>(.CPU, .WC(inputLength / 800, 2_048))
  var projectedResult = Tensor<Float>(.CPU, .WC(inputLength / 800, 32))
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(
      xCPU.reshaped(.NCHW(1, 1, 1, inputLength)).toGPU(deviceID))
    let (encoder, reader) = H3ConditioningAudioEncoder(
      inputLength: inputLength, includeHidden: true)
    encoder.maxConcurrency = .limit(1)
    print("MiniMax-H3 conditioning audio encoder: compiling Swift graph")
    encoder.compile(inputs: x)
    print("MiniMax-H3 conditioning audio encoder: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 conditioning audio encoder: executing Swift graph")
    let outputs = encoder(inputs: x)
    let output = outputs[0].as(of: Float.self).rawValue.toCPU()
      .reshaped(.WC(inputLength / 800, 64))
    encoderResult = outputs[1].as(of: Float.self).rawValue.toCPU()
      .reshaped(.WC(inputLength / 800, 2_048))
    projectedResult = outputs[2].as(of: Float.self).rawValue.toCPU()
      .reshaped(.WC(inputLength / 800, 32))
    print("MiniMax-H3 conditioning audio encoder: Swift graph complete")
    return output
  }
  h3Reference.release_conditioning_audio_encoder(pack)
  printMetrics(
    "MiniMax-H3 FP32 conditioning audio convolutional encoder",
    tokenParityMetrics(encoderResult, encoderReference))
  printMetrics(
    "MiniMax-H3 FP32 conditioning audio attention projection",
    tokenParityMetrics(projectedResult, projectedReference))
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 FP32 conditioning audio encoder", metrics)
  return metrics.maxRelativeDifference <= 3e-3
    && metrics.normalizedRootMeanSquareError <= 2e-3
    && metrics.minimumCosineSimilarity >= 0.999
    && metrics.meanCosineSimilarity >= 0.9999
}

func runVideoDecoderChainParity() -> Bool {
  let sequenceLength = Int(environment["MINIMAX_H3_TEST_TOKENS"] ?? "6") ?? 6
  let layers = Int(environment["MINIMAX_H3_VIDEO_LAYERS"] ?? "2") ?? 2
  guard Bool(torch.cuda.is_available()) ?? false else { return false }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_video_decoder_block(modelRoot, 0, deviceID)
  let testCase = h3Reference.run_video_decoder_chain_case(
    pack, sequenceLength, layers, 51, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let reference = tensorFromPython(testCase["output"])
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = Input()
    let rotary = Input()
    var out: Model.IO = x
    var readers = [(PythonObject) -> Void]()
    for layer in 0..<layers {
      let (block, reader) = H3VideoDecoderBlock(
        prefix: "decoder.transformer_blocks.\(layer)", sequenceLength: sequenceLength,
        deviceID: deviceID)
      out = block(out, rotary)
      readers.append(reader)
    }
    let chain = Model([x, rotary], [out])
    chain.maxConcurrency = .limit(1)
    let xVariable = graph.variable(Tensor<VideoFloat>(from: xCPU).toGPU(deviceID))
    let rotaryVariable = graph.variable(
      Tensor<VideoFloat>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, sequenceLength, 1, H3VideoDecoderConfig.headDim)))
    print("MiniMax-H3 video decoder chain: compiling", layers, "blocks")
    chain.compile(inputs: xVariable, rotaryVariable)
    for reader in readers { reader(state) }
    let output = Tensor<Float>(
      from: chain(inputs: xVariable, rotaryVariable)[0].as(of: Float.self).rawValue.toCPU()
        .reshaped(.WC(sequenceLength, H3VideoDecoderConfig.width)))
    return output
  }
  h3Reference.release_video_decoder_block(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 video decoder chain (\(layers) blocks)", metrics)
  return metrics.maxRelativeDifference <= 0.03 && metrics.minimumCosineSimilarity >= 0.99
    && metrics.meanCosineSimilarity >= 0.999
}

func runVideoDecoderShellParity() -> Bool {
  let patchCount = Int(environment["MINIMAX_H3_VIDEO_TEST_PATCHES"] ?? "2") ?? 2
  let sequenceLength = patchCount + H3VideoDecoderConfig.registerTokens + 1
  let outputWidth =
    3 * H3VideoDecoderConfig.temporalPatch
    * H3VideoDecoderConfig.spatialPatch * H3VideoDecoderConfig.spatialPatch
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-video-decoder outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_video_decoder_block(modelRoot, 0, deviceID)
  let testCase = h3Reference.run_video_decoder_shell_case(pack, patchCount, 49, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let zeroCPU = tensorFromPython(testCase["zero"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let reference = tensorFromPython(testCase["output"]).reshaped(.WC(patchCount, outputWidth))
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(Tensor<VideoFloat>(from: xCPU).toGPU(deviceID))
    let zero = graph.variable(zeroCPU.toGPU(deviceID))
    let rotary = graph.variable(
      Tensor<VideoFloat>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, sequenceLength, 1, H3VideoDecoderConfig.headDim)))
    let (decoder, reader) = H3VideoDecoder(
      numPatches: patchCount, deviceID: deviceID, layers: 1)
    decoder.maxConcurrency = .limit(1)
    print("MiniMax-H3 one-block video decoder shell: compiling Swift graph")
    decoder.compile(inputs: x, rotary, zero)
    print("MiniMax-H3 one-block video decoder shell: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 one-block video decoder shell: executing Swift graph")
    let output = Tensor<Float>(
      from: decoder(inputs: x, rotary, zero)[0].as(of: VideoFloat.self).rawValue.toCPU()
        .reshaped(.WC(patchCount, outputWidth)))
    print("MiniMax-H3 one-block video decoder shell: Swift graph complete")
    return output
  }
  h3Reference.release_video_decoder_block(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 one-block video decoder shell", metrics)
  return metrics.maxRelativeDifference <= 0.02 && metrics.minimumCosineSimilarity >= 0.99
    && metrics.meanCosineSimilarity >= 0.999
}

func runFullVideoDecoderParity() -> Bool {
  let patchCount = Int(environment["MINIMAX_H3_VIDEO_TEST_PATCHES"] ?? "1") ?? 1
  let layers = Int(environment["MINIMAX_H3_VIDEO_LAYERS"] ?? "36") ?? 36
  precondition(layers > 0 && layers <= H3VideoDecoderConfig.layers)
  let sequenceLength = patchCount + H3VideoDecoderConfig.registerTokens + 1
  let outputWidth =
    3 * H3VideoDecoderConfig.temporalPatch
    * H3VideoDecoderConfig.spatialPatch * H3VideoDecoderConfig.spatialPatch
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-video-full outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_video_decoder_block(modelRoot, 0, deviceID)
  print("MiniMax-H3 full video decoder: running Python reference with", layers, "blocks")
  let testCase = h3Reference.run_video_decoder_full_case(
    pack, patchCount, layers, 50, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let zeroCPU = tensorFromPython(testCase["zero"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let reference = tensorFromPython(testCase["output"]).reshaped(.WC(patchCount, outputWidth))
  let referenceHidden = tensorFromPython(testCase["hidden"])
  let result = graph.withNoGrad { () -> (Tensor<Float>, Tensor<Float>) in
    let x = graph.variable(Tensor<VideoFloat>(from: xCPU).toGPU(deviceID))
    let zero = graph.variable(zeroCPU.toGPU(deviceID))
    let rotary = graph.variable(
      Tensor<VideoFloat>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, sequenceLength, 1, H3VideoDecoderConfig.headDim)))
    let (decoder, reader) = H3VideoDecoder(
      numPatches: patchCount, deviceID: deviceID, layers: layers, includeHidden: true)
    decoder.maxConcurrency = .limit(1)
    print("MiniMax-H3 full video decoder: compiling Swift graph")
    decoder.compile(inputs: x, rotary, zero)
    print("MiniMax-H3 full video decoder: loading", layers, "blocks into unified memory")
    reader(state)
    print("MiniMax-H3 full video decoder: executing Swift graph")
    let decoderOutputs = decoder(inputs: x, rotary, zero)
    let output = Tensor<Float>(
      from: decoderOutputs[0].as(of: VideoFloat.self).rawValue.toCPU()
        .reshaped(.WC(patchCount, outputWidth)))
    let hidden = Tensor<Float>(
      from: decoderOutputs[1].as(of: Float.self).rawValue.toCPU()
        .reshaped(.WC(sequenceLength, H3VideoDecoderConfig.width)))
    print("MiniMax-H3 full video decoder: Swift graph complete")
    return (output, hidden)
  }
  h3Reference.release_video_decoder_block(pack)
  let hiddenMetrics = tokenParityMetrics(result.1, referenceHidden)
  printMetrics("MiniMax-H3 full video decoder hidden (\(layers) blocks)", hiddenMetrics)
  let metrics = tokenParityMetrics(result.0, reference)
  printMetrics("MiniMax-H3 full video decoder (\(layers) blocks)", metrics)
  return metrics.maxRelativeDifference <= 0.03 && metrics.minimumCosineSimilarity >= 0.99
    && metrics.meanCosineSimilarity >= 0.999
}

func runAudioAMPParity() -> Bool {
  let width = Int(environment["MINIMAX_H3_AUDIO_TEST_WIDTH"] ?? "20") ?? 20
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-audio-amp outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_audio_amp(modelRoot)
  let testCase = h3Reference.run_audio_amp_case(pack, width, 47, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let reference = tensorFromPython(testCase["output"])
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(
      xCPU.reshaped(.NCHW(1, 512, 1, width)).toGPU(deviceID))
    let (reader, block) = H3AudioAMPBlock(
      prefix: "decoder.resblocks.0", channels: 512, width: width,
      kernelSize: 3, name: "audio_amp_probe")
    block.maxConcurrency = .limit(1)
    print("MiniMax-H3 audio AMP block 0: compiling Swift graph")
    block.compile(inputs: x)
    print("MiniMax-H3 audio AMP block 0: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 audio AMP block 0: executing Swift graph")
    let output = block(inputs: x)[0].as(of: Float.self).rawValue.toCPU()
      .reshaped(.WC(512, width))
    print("MiniMax-H3 audio AMP block 0: Swift graph complete")
    return output
  }
  h3Reference.release_audio_amp(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 audio AMP block 0", metrics)
  return metrics.maxRelativeDifference <= 1e-3 && metrics.minimumCosineSimilarity >= 0.9999
    && metrics.meanCosineSimilarity >= 0.99999
}

func runAudioDecoderParity() -> Bool {
  let latentWidth = Int(environment["MINIMAX_H3_AUDIO_TEST_WIDTH"] ?? "2") ?? 2
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-audio-decoder outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_audio_amp(modelRoot)
  let testCase = h3Reference.run_audio_decoder_case(pack, latentWidth, 48, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let outputWidth = latentWidth * 800
  let reference = tensorFromPython(testCase["output"]).reshaped(.WC(1, outputWidth))
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(xCPU.reshaped(.NCHW(1, 32, 1, latentWidth)).toGPU(deviceID))
    let (reader, decoder) = H3AudioDecoder(latentWidth: latentWidth)
    decoder.maxConcurrency = .limit(1)
    print("MiniMax-H3 complete audio decoder: compiling Swift graph")
    decoder.compile(inputs: x)
    print("MiniMax-H3 complete audio decoder: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 complete audio decoder: executing Swift graph")
    let output = decoder(inputs: x)[0].as(of: Float.self).rawValue.toCPU()
      .reshaped(.WC(1, outputWidth))
    print("MiniMax-H3 complete audio decoder: Swift graph complete")
    return output
  }
  h3Reference.release_audio_amp(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 complete audio decoder", metrics)
  return metrics.maxRelativeDifference <= 2e-3 && metrics.minimumCosineSimilarity >= 0.999
    && metrics.meanCosineSimilarity >= 0.999
}

func runTransformerBlockParity(transformerSubdirectory: String = "transformer") -> Bool {
  let sequenceLength = Int(environment["MINIMAX_H3_TEST_TOKENS"] ?? "9") ?? 9
  let blockIndex = Int(environment["MINIMAX_H3_BLOCK_INDEX"] ?? "0") ?? 0
  precondition(blockIndex >= 0 && blockIndex < H3Config.layers)
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run parity-block outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let pack = h3Reference.load_block(modelRoot, blockIndex, deviceID, transformerSubdirectory)
  let testCase = h3Reference.run_block_case(pack, sequenceLength, 42, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let tembCPU = tensorFromPython(testCase["temb_activated"])
  let selectionCPU = tensorFromPython(testCase["selection"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let reference = tensorFromPython(testCase["output"])
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(Tensor<BlockFloat>(from: xCPU).toGPU(deviceID))
    let temb = graph.variable(tembCPU.toGPU(deviceID))
    let selection = graph.variable(Tensor<BlockFloat>(from: selectionCPU).toGPU(deviceID))
    let rotary = graph.variable(
      Tensor<BlockFloat>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, sequenceLength, 1, H3Config.headDim)))
    let (block, reader) = H3TransformerBlock(
      prefix: "transformer_blocks.\(blockIndex)", sequenceLength: sequenceLength, timestepCount: 2)
    block.maxConcurrency = .limit(1)
    print("MiniMax-H3 block parity: compiling Swift graph")
    block.compile(inputs: x, rotary, selection, temb)
    print("MiniMax-H3 block parity: loading Swift weights")
    reader(state)
    print("MiniMax-H3 block parity: executing Swift graph")
    let output = Tensor<Float>(
      from: block(inputs: x, rotary, selection, temb)[0].as(of: Float.self).rawValue.toCPU()
        .reshaped(.WC(sequenceLength, H3Config.hiddenSize)))
    print("MiniMax-H3 block parity: Swift graph complete")
    return output
  }
  h3Reference.release_block(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 transformer block \(blockIndex)", metrics)
  return metrics.maxRelativeDifference <= 0.05 && metrics.minimumCosineSimilarity >= 0.99
    && metrics.meanCosineSimilarity >= 0.999
}

func runQwenExportBlockParity() -> Bool {
  let sequenceLength = Int(environment["MINIMAX_H3_TEST_TOKENS"] ?? "6") ?? 6
  guard Bool(torch.cuda.is_available()) ?? false else {
    print("CUDA is not visible. Run Qwen export outside the sandbox.")
    return false
  }
  torch.set_grad_enabled(false)
  let layer = H3QwenConfig.featureLayer - 1
  let pack = h3Reference.load_qwen_block(modelRoot, layer, deviceID)
  let testCase = h3Reference.run_qwen_block_case(pack, sequenceLength, 43, deviceID)
  let state = pack["state"]
  let xCPU = tensorFromPython(testCase["x"])
  let rotaryCPU = tensorFromPython(testCase["rotary"])
  let reference = tensorFromPython(testCase["output"])
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let x = graph.variable(Tensor<Float16>(from: xCPU).toGPU(deviceID))
    let rotary = graph.variable(
      Tensor<Float16>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, sequenceLength, 1, H3QwenConfig.headDim)))
    let prefix = "model.language_model.layers.\(layer)"
    let (block, reader) = H3QwenExportBlock(prefix: prefix, sequenceLength: sequenceLength)
    block.maxConcurrency = .limit(1)
    print("MiniMax-H3 Qwen FP16 export block: compiling")
    block.compile(inputs: x, rotary)
    print("MiniMax-H3 Qwen FP16 export block: loading weights")
    reader(state)
    print("MiniMax-H3 Qwen FP16 export block: executing")
    return Tensor<Float>(
      from: block(inputs: x, rotary)[0].as(of: Float16.self).rawValue.toCPU()
        .reshaped(.WC(sequenceLength, H3QwenConfig.hiddenSize)))
  }
  h3Reference.release_qwen_block(pack)
  let metrics = tokenParityMetrics(result, reference)
  printMetrics("MiniMax-H3 Qwen FP16 export block \(layer)", metrics)
  return metrics.allFinite && metrics.maxRelativeDifference <= 0.02
    && metrics.minimumCosineSimilarity >= 0.99
    && metrics.meanCosineSimilarity >= 0.999
}

func exportQwenTextModel() {
  precondition(runQwenExportBlockParity(), "MiniMax-H3 Qwen FP16 export parity failed")
  let sequenceLength = Int(environment["MINIMAX_H3_QWEN_EXPORT_TOKENS"] ?? "256") ?? 256
  print("MiniMax-H3 Qwen export path:", qwenExportPath)
  print("MiniMax-H3 Qwen export layers/tokens:", H3QwenConfig.featureLayer, sequenceLength)
  let state = h3Reference.ShardedStateDict(
    osPath.join(modelRoot, "text_encoder"), "model.safetensors.index.json")
  graph.withNoGrad {
    let tokensCPU = graph.variable(.CPU, format: .NHWC, shape: [sequenceLength], of: Int32.self)
    for index in 0..<sequenceLength { tokensCPU[index] = 0 }
    let tokens = tokensCPU.toGPU(deviceID)
    let rotaryCPU = tensorFromPython(h3Reference.qwen_rotary_tensor(sequenceLength))
    let rotary = graph.variable(
      Tensor<Float16>(from: rotaryCPU).toGPU(deviceID)
        .reshaped(.NHWC(1, sequenceLength, 1, H3QwenConfig.headDim)))
    let (model, reader) = H3QwenExportModel(sequenceLength: sequenceLength)
    model.maxConcurrency = .limit(1)
    print("MiniMax-H3 Qwen export: compiling")
    model.compile(inputs: tokens, rotary)
    print("MiniMax-H3 Qwen export: loading 50 layers into unified memory")
    reader(state)
    print("MiniMax-H3 Qwen export: writing store")
    graph.openStore(qwenExportPath) {
      $0.write("text_model", model: model)
    }
  }
  state.release()
  print("MiniMax-H3 Qwen export: done")
}

func exportMainDiT(
  transformerSubdirectory: String = "transformer", exportPath: String = ditExportPath,
  requireFullParity: Bool = false
) {
  precondition(
    runTransformerBlockParity(transformerSubdirectory: transformerSubdirectory),
    "MiniMax-H3 DiT export block parity failed")
  precondition(
    runTimestepEmbeddingParity(transformerSubdirectory: transformerSubdirectory),
    "MiniMax-H3 timestep MLP export parity failed")
  if requireFullParity {
    precondition(
      runFullJointTransformerParity(
        transformerSubdirectory: transformerSubdirectory,
        allowRef2VAAudioAmplification: transformerSubdirectory == "transformer_ref"),
      "MiniMax-H3 full DiT export parity failed")
  }
  let textLength = Int(environment["MINIMAX_H3_EXPORT_TEXT_TOKENS"] ?? "256") ?? 256
  let latentFrames = videoLatentFrameCount(H3Config.defaultFrames)
  let latentHeight = H3Config.defaultHeight / H3Config.videoSpatialCompression
  let latentWidth = H3Config.defaultWidth / H3Config.videoSpatialCompression
  let audioLatents = audioLatentFrameCount(H3Config.defaultFrames)
  let layout = makeT2VALayout(
    textTokenCount: textLength, latentFrames: latentFrames,
    latentHeight: latentHeight, latentWidth: latentWidth, audioLatents: audioLatents)
  let videoLength = layout.videoRange.count
  let audioLength = layout.audioRange.count
  print("MiniMax-H3 DiT source:", transformerSubdirectory)
  print("MiniMax-H3 DiT export path:", exportPath)
  print("MiniMax-H3 DiT text/audio/video rows:", textLength, audioLength, videoLength)
  let state = h3Reference.ShardedStateDict(osPath.join(modelRoot, transformerSubdirectory))
  graph.withNoGrad {
    let video = graph.variable(
      .CPU, format: .NHWC, shape: [1, videoLength, H3Config.videoPatchSize], of: Float.self
    ).toGPU(deviceID)
    let audio = graph.variable(
      .CPU, format: .NHWC, shape: [1, audioLength, H3Config.audioChannels], of: Float.self
    ).toGPU(deviceID)
    let text = graph.variable(
      .CPU, format: .NHWC, shape: [1, textLength, H3Config.textSize], of: BlockFloat.self
    ).toGPU(deviceID)
    let rotary = graph.variable(h3RotaryTensor(positionIDs: layout.positionIDs).toGPU(deviceID))
    var adalnSelectionCPU = Tensor<BlockFloat>(.CPU, .WC(layout.sequenceLength, 6))
    var timestepSelectionCPU = Tensor<BlockFloat>(.CPU, .WC(layout.sequenceLength, 2))
    for row in 0..<layout.sequenceLength {
      let timestep = layout.tokenTags[row] == H3Tag.audio.rawValue ? 1 : 0
      adalnSelectionCPU[row, timestep * H3Config.modalityCount + Int(layout.tokenTags[row])] = 1
      timestepSelectionCPU[row, timestep] = 1
    }
    let adalnSelection = graph.variable(adalnSelectionCPU.toGPU(deviceID))
    let timestepSelection = graph.variable(timestepSelectionCPU.toGPU(deviceID))
    let temb = graph.variable(
      .CPU, format: .NHWC, shape: [2, H3Config.timestepSize], of: Float.self
    ).toGPU(deviceID)
    let (model, reader) = H3JointTransformer(
      textLength: textLength, audioLength: audioLength, videoLength: videoLength,
      timestepCount: 2)
    model.maxConcurrency = .limit(1)
    print("MiniMax-H3 DiT export: compiling default T2V graph")
    model.compile(
      inputs: video, audio, text, rotary, adalnSelection, timestepSelection, temb)
    print("MiniMax-H3 DiT export: loading 50 layers into unified memory")
    reader(state)
    print("MiniMax-H3 DiT export: writing store")
    graph.openStore(exportPath) { $0.write("dit", model: model) }

    let frequencies = graph.variable(
      .CPU, format: .NHWC, shape: [2, H3Config.timestepFrequencySize], of: Float.self
    ).toGPU(deviceID)
    let (timeEmbedding, timeReader) = H3TimestepEmbedding(timestepCount: 2)
    timeEmbedding.maxConcurrency = .limit(1)
    timeEmbedding.compile(inputs: frequencies)
    timeReader(state)
    graph.openStore(exportPath) { $0.write("time_embedder", model: timeEmbedding) }
  }
  state.release()
  print("MiniMax-H3 DiT export: done")
}

func exportTimestepEmbedding() {
  precondition(runTimestepEmbeddingParity(), "MiniMax-H3 timestep MLP export parity failed")
  let state = h3Reference.ShardedStateDict(osPath.join(modelRoot, "transformer"))
  graph.withNoGrad {
    let frequencies = graph.variable(
      .CPU, format: .NHWC, shape: [2, H3Config.timestepFrequencySize], of: Float.self
    ).toGPU(deviceID)
    let (model, reader) = H3TimestepEmbedding(timestepCount: 2)
    model.maxConcurrency = .limit(1)
    print("MiniMax-H3 timestep MLP export: compiling")
    model.compile(inputs: frequencies)
    print("MiniMax-H3 timestep MLP export: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 timestep MLP export: appending to", ditExportPath)
    graph.openStore(ditExportPath) { $0.write("time_embedder", model: model) }
  }
  state.release()
  print("MiniMax-H3 timestep MLP export: done")
}

func exportVideoDecoder() {
  precondition(runFullVideoDecoderParity(), "MiniMax-H3 video decoder export parity failed")
  let latentFrames = 5
  let latentHeight = 16
  let latentWidth = 16
  let numPatches = latentFrames * latentHeight * latentWidth
  print("MiniMax-H3 video decoder export path:", vaeExportPath)
  print("MiniMax-H3 video decoder tile:", latentFrames, latentHeight, latentWidth)
  let state = h3Reference.ShardedStateDict(osPath.join(modelRoot, "vae"))
  graph.withNoGrad {
    let latents = graph.variable(
      .CPU, format: .NHWC, shape: [1, numPatches, H3Config.videoChannels], of: VideoFloat.self
    ).toGPU(deviceID)
    let rotary = graph.variable(
      videoDecoderRotary(
        latentFrames: latentFrames, latentHeight: latentHeight, latentWidth: latentWidth
      ).toGPU(deviceID))
    let zero = graph.variable(
      .CPU, format: .NHWC, shape: [1, 1, H3VideoDecoderConfig.width], of: Float.self
    ).toGPU(deviceID)
    let (model, reader) = H3VideoDecoder(numPatches: numPatches, deviceID: deviceID)
    model.maxConcurrency = .limit(1)
    print("MiniMax-H3 video decoder export: compiling")
    model.compile(inputs: latents, rotary, zero)
    print("MiniMax-H3 video decoder export: loading weights")
    reader(state)
    print("MiniMax-H3 video decoder export: writing store")
    graph.openStore(vaeExportPath) { $0.write("video_decoder", model: model) }
  }
  state.release()
  print("MiniMax-H3 video decoder export: done")
}

func exportConditioningVideoEncoder() {
  precondition(
    runConditioningVideoEncoderParity(), "MiniMax-H3 conditioning video encoder parity failed")
  let frames = 1
  let height = 32
  let width = 32
  print("MiniMax-H3 conditioning video encoder export path:", vaeExportPath)
  let state = h3Reference.SingleStateDict(
    osPath.join(modelRoot, "FL2VA", "video_vae", "source", "model.safetensors"))
  graph.withNoGrad {
    let x = graph.variable(
      .CPU, format: .NCHW, shape: [1, 3, frames, height, width], of: Float.self
    ).toGPU(deviceID)
    let (model, reader) = H3ConditioningVideoEncoder(
      frames: frames, height: height, width: width)
    model.maxConcurrency = .limit(1)
    print("MiniMax-H3 conditioning video encoder export: compiling")
    model.compile(inputs: x)
    print("MiniMax-H3 conditioning video encoder export: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 conditioning video encoder export: writing store")
    graph.openStore(vaeExportPath) { $0.write("video_encoder", model: model) }
  }
  state.release()
  print("MiniMax-H3 conditioning video encoder export: done")
}

func exportConditioningAudioEncoder() {
  precondition(
    runConditioningAudioEncoderParity(), "MiniMax-H3 conditioning audio encoder parity failed")
  let inputLength = 1_600
  print("MiniMax-H3 conditioning audio encoder export path:", vaeExportPath)
  let state = h3Reference.SingleStateDict(
    osPath.join(modelRoot, "FL2VA", "audio_vae", "model.safetensors"))
  graph.withNoGrad {
    let x = graph.variable(
      .CPU, format: .NCHW, shape: [1, 1, 1, inputLength], of: Float.self
    ).toGPU(deviceID)
    let (model, reader) = H3ConditioningAudioEncoder(inputLength: inputLength)
    model.maxConcurrency = .limit(1)
    print("MiniMax-H3 conditioning audio encoder export: compiling")
    model.compile(inputs: x)
    print("MiniMax-H3 conditioning audio encoder export: loading unified-memory weights")
    reader(state)
    print("MiniMax-H3 conditioning audio encoder export: writing store")
    graph.openStore(vaeExportPath) { $0.write("audio_encoder", model: model) }
  }
  state.release()
  print("MiniMax-H3 conditioning audio encoder export: done")
}

func exportAudioDecoder() {
  precondition(runAudioDecoderParity(), "MiniMax-H3 audio decoder export parity failed")
  let latentWidth = audioLatentFrameCount(H3Config.defaultFrames)
  print("MiniMax-H3 audio decoder export path:", vaeExportPath)
  print("MiniMax-H3 audio decoder latent width:", latentWidth)
  let state = h3Reference.SingleStateDict(
    osPath.join(modelRoot, "audio_vae", "diffusion_pytorch_model.safetensors"))
  graph.withNoGrad {
    let latents = graph.variable(
      .CPU, format: .NCHW, shape: [1, H3Config.audioChannels, 1, latentWidth], of: Float.self
    ).toGPU(deviceID)
    let (reader, model) = H3AudioDecoder(latentWidth: latentWidth)
    model.maxConcurrency = .limit(1)
    print("MiniMax-H3 audio decoder export: compiling")
    model.compile(inputs: latents)
    print("MiniMax-H3 audio decoder export: loading weights")
    reader(state)
    print("MiniMax-H3 audio decoder export: writing store")
    graph.openStore(vaeExportPath) { $0.write("audio_decoder", model: model) }
  }
  state.release()
  print("MiniMax-H3 audio decoder export: done")
}

func runFoundationSelfTest() {
  precondition(alignFrameCount(120) == 124)
  precondition(videoLatentFrameCount(124) == 37)
  precondition(audioLatentFrameCount(124) == 207)
  let layout = makeT2VALayout(
    textTokenCount: 3, latentFrames: 2, latentHeight: 4, latentWidth: 4,
    audioLatents: 2)
  precondition(layout.sequenceLength == 15)
  precondition(layout.textRange == 0..<3)
  precondition(layout.audioRange == 3..<7)
  precondition(layout.videoRange == 7..<15)
  precondition(layout.tokenTags == [1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0])
  let temporalPlan = videoTemporalDecodePlan(latentFrames: 37)
  precondition(temporalPlan.framePrePadding == 3)
  precondition(temporalPlan.tokensPerChunk == 5 && temporalPlan.tokenOverlap == 2)
  precondition(temporalPlan.frameOverlap == 5 && temporalPlan.repeatedTailTokens == 0)
  precondition(temporalPlan.clipStarts == [0, 5, 10, 15, 20, 25, 30])
  let heightTiles = splitVideoTiles(length: 768)
  precondition(heightTiles.starts == [0, 160, 336, 512])
  precondition(heightTiles.overlaps == [96, 80, 80])
  let widthTiles = splitVideoTiles(length: 1_344)
  precondition(widthTiles.starts == [0, 176, 352, 528, 704, 896, 1_088])
  precondition(widthTiles.overlaps == [80, 80, 80, 80, 64, 64])
  let decoderRotary = videoDecoderRotary(latentFrames: 1, latentHeight: 1, latentWidth: 2)
  precondition(decoderRotary.shape == [1, 7, 1, H3VideoDecoderConfig.headDim])
  var patchRows = Tensor<Float>(.CPU, .WC(2, 3 * 4 * 16 * 16))
  patchRows[0, ((2 * 4 + 3) * 16 + 5) * 16 + 7] = 7
  patchRows[1, ((1 * 4 + 2) * 16 + 4) * 16 + 6] = 9
  let pixels = unpatchifyVideoDecoderRows(
    patchRows, latentFrames: 1, latentHeight: 1, latentWidth: 2)
  precondition(pixels.shape == [1, 3, 4, 16, 32])
  precondition(pixels[0, 2, 3, 5, 7] == 7)
  precondition(pixels[0, 1, 2, 4, 22] == 9)
  let videoSigmas = shiftedSigmaGrid(points: 3, shift: H3Config.videoFlowShift)
  precondition(videoSigmas[0] == 1 && videoSigmas[2] == 0)
  let timestep = 1 - videoSigmas[0]
  let stepped = schedulerStep(
    sample: 0.25, velocity: -0.5, timestep: timestep, sigma: videoSigmas[0],
    nextSigma: videoSigmas[1])
  precondition(stepped.isFinite)
  print("MiniMax-H3 foundation self-test passed")
  print(
    "default T2VA geometry:", H3Config.defaultWidth, "x", H3Config.defaultHeight,
    H3Config.defaultFrames, "frames;", videoLatentFrameCount(H3Config.defaultFrames),
    "video latents;", audioLatentFrameCount(H3Config.defaultFrames), "audio latents/channel")
}

switch mode {
case "self-test":
  runFoundationSelfTest()
case "parity-block":
  precondition(runTransformerBlockParity(), "MiniMax-H3 transformer block parity failed")
case "parity-ref2va-block":
  precondition(
    runTransformerBlockParity(transformerSubdirectory: "transformer_ref"),
    "MiniMax-H3 Ref2VA transformer block parity failed")
case "parity-qwen-block":
  precondition(runQwenBlockParity(), "MiniMax-H3 Qwen block parity failed")
case "export-qwen":
  exportQwenTextModel()
case "export-dit":
  exportMainDiT()
case "export-ref2va-dit":
  exportMainDiT(
    transformerSubdirectory: "transformer_ref", exportPath: ref2vaDitExportPath,
    requireFullParity: true)
case "export-time-embedding":
  exportTimestepEmbedding()
case "export-video-decoder":
  exportVideoDecoder()
case "export-video-encoder":
  exportConditioningVideoEncoder()
case "export-audio-encoder":
  exportConditioningAudioEncoder()
case "export-audio-decoder":
  exportAudioDecoder()
case "export-vae":
  exportConditioningVideoEncoder()
  exportVideoDecoder()
  exportConditioningAudioEncoder()
  exportAudioDecoder()
case "parity-refiner":
  precondition(runTokenRefinerParity(), "MiniMax-H3 token refiner parity failed")
case "parity-time":
  precondition(runTimestepEmbeddingParity(), "MiniMax-H3 timestep MLP parity failed")
case "parity-shell":
  precondition(runOneBlockShellParity(), "MiniMax-H3 one-block shell parity failed")
case "parity-transformer-full":
  precondition(
    runFullJointTransformerParity(), "MiniMax-H3 full joint transformer parity failed")
case "parity-ref2va-transformer-full":
  precondition(
    runFullJointTransformerParity(
      transformerSubdirectory: "transformer_ref", allowRef2VAAudioAmplification: true),
    "MiniMax-H3 full Ref2VA joint transformer parity failed")
case "parity-video-block":
  precondition(runVideoDecoderBlockParity(), "MiniMax-H3 video decoder block parity failed")
case "parity-video-encoder":
  precondition(
    runConditioningVideoEncoderParity(), "MiniMax-H3 conditioning video encoder parity failed")
case "parity-audio-encoder":
  precondition(
    runConditioningAudioEncoderParity(), "MiniMax-H3 conditioning audio encoder parity failed")
case "parity-video-chain":
  precondition(runVideoDecoderChainParity(), "MiniMax-H3 video decoder chain parity failed")
case "parity-video-decoder":
  precondition(runVideoDecoderShellParity(), "MiniMax-H3 video decoder shell parity failed")
case "parity-video-full":
  precondition(runFullVideoDecoderParity(), "MiniMax-H3 full video decoder parity failed")
case "inspect-audio":
  for key in h3Reference.audio_state_keys(modelRoot) { print(key) }
case "parity-audio-amp":
  precondition(runAudioAMPParity(), "MiniMax-H3 audio AMP parity failed")
case "parity-audio-decoder":
  precondition(runAudioDecoderParity(), "MiniMax-H3 audio decoder parity failed")
default:
  print("MiniMax-H3 mode is not implemented yet:", mode)
  exit(2)
}
