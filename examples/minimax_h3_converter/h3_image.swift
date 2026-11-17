import Foundation
import NNC
import NNCPythonConversion
import PythonKit

func h3ImageDecoderRotary(height: Int, width: Int) -> Tensor<Float> {
  let patches = height * width
  var result = Tensor<Float>(.CPU, .NHWC(1, patches + 5, 1, 64))
  for row in 0..<(patches + 5) {
    for feature in 0..<64 { result[0, row, 0, feature] = feature % 2 == 0 ? 1 : 0 }
  }
  for y in 0..<height {
    for x in 0..<width {
      let coordinates = [
        2 * ((Float(y) + 0.5) / Float(height)) - 1,
        2 * ((Float(x) + 0.5) / Float(width)) - 1,
      ]
      for axis in 0..<2 {
        for frequency in 0..<8 {
          let inverse: Float = 1 / pow(100, Float(frequency) / 8)
          let angle = (2 * Float.pi * coordinates[axis]) * inverse
          let index = 2 * (axis * 8 + frequency)
          result[0, y * width + x, 0, index] = cos(angle)
          result[0, y * width + x, 0, index + 1] = sin(angle)
        }
      }
    }
  }
  return result
}

@discardableResult
func checkH3ImageParity(
  _ label: String, _ actual: Tensor<Float>, _ expected: Tensor<Float>,
  maxRelativeDifference: Float = 1e-3
)
  -> TokenParityMetrics
{
  precondition(actual.shape == expected.shape, "\(label): shape mismatch")
  let metrics = tokenParityMetrics(actual, expected)
  printMetrics(label, metrics)
  precondition(
    metrics.allFinite && metrics.maxRelativeDifference < maxRelativeDifference
      && metrics.normalizedRootMeanSquareError < 1e-3
      && metrics.minimumCosineSimilarity > 0.99999,
    "\(label): FP32 parity failed")
  return metrics
}

func runH3ImageVAEParity(export: Bool = false) {
  precondition(Bool(torch.cuda.is_available()) == true, "Run GPU parity outside the sandbox")
  let exportPath = "/slow/Data/minimax_h3_image_vae_f32.ckpt"
  let stagingPath = exportPath + ".partial"
  if export {
    for path in [exportPath, stagingPath, exportPath + ".json"] {
      precondition(!FileManager.default.fileExists(atPath: path), "Refusing to overwrite \(path)")
    }
  }
  let assets = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().appendingPathComponent("minimax_h3_assets").path
  movePythonPathToFront(assets)
  let reference = Python.import("h3_image_reference")
  let pack = reference.load_vae(deviceID)
  let state = pack["state"]
  let configuration = H3VAEDecoderArchitecture.image
  // Square, non-square, and a complete production-size 256px VAE tile.
  for (index, size) in [(64, 64), (64, 96), (256, 256)].enumerated() {
    let (height, width) = size
    let latentHeight = height / 16
    let latentWidth = width / 16
    let patches = latentHeight * latentWidth
    print("H3-Image VAE: official FP32 case \(height)x\(width)")
    let test = reference.vae_case(pack, height, width, 61 + index)
    let rotaryCPU = h3ImageDecoderRotary(height: latentHeight, width: latentWidth)
    checkH3ImageParity(
      "H3-Image 2D decoder RoPE \(height)x\(width)",
      rotaryCPU.reshaped(.WC(patches + 5, 64)), tensorFromPython(test["rotary"]))
    graph.withNoGrad {
      let pixels = graph.variable(tensorFromPython(test["pixels"]).toGPU(deviceID))
      let (encoder, encoderReader) = H3ConditioningVideoEncoder(
        frames: 1, height: height, width: width)
      encoder.maxConcurrency = .limit(1)
      encoder.compile(inputs: pixels)
      encoderReader(state)
      let momentsTensor = encoder(inputs: pixels)[0].as(of: Float.self)
      let moments = momentsTensor.permuted(0, 2, 3, 4, 1).contiguous()
        .reshaped(.WC(patches, 48)).rawValue.toCPU()
      checkH3ImageParity(
        "H3-Image encoder moments \(height)x\(width)", moments,
        tensorFromPython(test["moments"]))

      let latent = graph.variable(
        tensorFromPython(test["latent_rows"]).reshaped(.HWC(1, patches, 24)).toGPU(deviceID))
      let rotary = graph.variable(rotaryCPU.toGPU(deviceID))
      let zero = graph.variable(.GPU(deviceID), .HWC(1, 1, configuration.width), of: Float.self)
      zero.full(0)
      let (decoder, decoderReader) = H3VideoDecoder(
        numPatches: patches, deviceID: deviceID, layers: configuration.layers,
        configuration: configuration, useFloat16: false)
      decoder.maxConcurrency = .limit(1)
      decoder.compile(inputs: latent, rotary, zero)
      decoderReader(state)
      let decoded = decoder(inputs: latent, rotary, zero)[0].as(of: Float.self)
        .rawValue.toCPU().reshaped(.WC(patches, 768))
      checkH3ImageParity(
        "H3-Image decoder pixels \(height)x\(width)", decoded,
        tensorFromPython(test["decoded"]))

      var mean = Tensor<Float>(.CPU, .HWC(1, patches, 24))
      for row in 0..<patches {
        for channel in 0..<24 { mean[0, row, channel] = moments[row, channel] }
      }
      let reconstructed = decoder(
        inputs: graph.variable(mean.toGPU(deviceID)), rotary, zero)[0].as(of: Float.self)
        .rawValue.toCPU().reshaped(.WC(patches, 768))
      let meanValues = (0..<patches).flatMap { row in
        (0..<24).map { channel in mean[0, row, channel] }
      }
      let sharedLatentReference = tensorFromPython(
        reference.decode_rows(pack, meanValues, latentHeight, latentWidth))
      checkH3ImageParity(
        "H3-Image decoder on Swift encoder latent \(height)x\(width)", reconstructed,
        sharedLatentReference)
      printMetrics(
        "H3-Image Python-only propagated encoder error \(height)x\(width)",
        tokenParityMetrics(sharedLatentReference, tensorFromPython(test["reconstructed"])))
      // Composition can amplify the small FP32 convolution/reduction differences.
      // Keep tight RMS/cosine gates, and check the decoder on identical latents
      // above so a localized propagated error cannot hide a decoder mismatch.
      checkH3ImageParity(
        "H3-Image encoder-to-decoder \(height)x\(width)", reconstructed,
        tensorFromPython(test["reconstructed"]), maxRelativeDifference: 5e-3)
      if export && index == 2 {
        state.validate_coverage()
        print("H3-Image VAE parity passed; writing both models to", stagingPath)
        graph.openStore(stagingPath) { store in
          store.write("video_encoder", model: encoder)
          store.write("video_decoder", model: decoder)
        }
        graph.openStore(stagingPath) { store in
          try! store.read("video_encoder", model: encoder, strict: true)
          try! store.read("video_decoder", model: decoder, strict: true)
        }
        let reloadedMoments = encoder(inputs: pixels)[0].as(of: Float.self)
          .permuted(0, 2, 3, 4, 1).contiguous().reshaped(.WC(patches, 48)).rawValue.toCPU()
        let reloadedPixels = decoder(inputs: latent, rotary, zero)[0].as(of: Float.self)
          .rawValue.toCPU().reshaped(.WC(patches, 768))
        for (name, actual, expected) in [
          ("encoder", reloadedMoments, moments), ("decoder", reloadedPixels, decoded),
        ] {
          let metrics = tokenParityMetrics(actual, expected)
          printMetrics("H3-Image VAE strict reload \(name)", metrics)
          precondition(metrics.allFinite && metrics.maxAbsoluteDifference == 0)
        }
        let encoderNames = (0..<encoder.parameters.count).map { encoder.parameters[$0].name }
        let decoderNames = (0..<decoder.parameters.count).map { decoder.parameters[$0].name }
        let audit = reference.audit_vae_export(stagingPath, pack, encoderNames, decoderNames)
        try! FileManager.default.moveItem(atPath: stagingPath, toPath: exportPath)
        reference.write_vae_export_metadata(exportPath, pack, audit)
        print("H3-Image VAE export verified:", exportPath)
      }
    }
  }
  state.validate_coverage()
  reference.release_vae(pack)
  if !export {
    print("H3-Image VAE FP32 parity passed; no checkpoint written (export approval required)")
  }
}

func runH3ImageDiTParity(export: Bool = false, float16Storage: Bool = false) {
  precondition(Bool(torch.cuda.is_available()) == true, "Run GPU parity outside the sandbox")
  precondition(
    environment["H3_FP32_ATTENTION_CONTROL"] == "1", "Use FP32 attention for baseline parity")
  let precision = float16Storage ? "f16" : "f32"
  let exportPath = "/slow/Data/minimax_h3_image_dit_\(precision).ckpt"
  let stagingPath = exportPath + ".partial"
  if export {
    for path in [exportPath, stagingPath, exportPath + ".inference.json"] {
      precondition(!FileManager.default.fileExists(atPath: path), "Refusing to overwrite \(path)")
    }
  }
  let assets = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().appendingPathComponent("minimax_h3_assets").path
  movePythonPathToFront(assets)
  let reference = Python.import("h3_image_reference")
  let source = h3Reference.ShardedStateDict("/slow/Data/MiniMax-H3-Image/transformer_diffusers")
  let pack = reference.load_image_dit(source, deviceID, float16Storage)
  let state = pack["state"]
  for hasReferences in [false, true] {
    let label = hasReferences ? "reference layout" : "T2I layout"
    let textLength = hasReferences ? 4 : 3
    let imageLength = hasReferences ? 6 : 4
    let length = textLength + imageLength
    let timestepCount = hasReferences ? 3 : 2
    var baseline: PythonObject = Python.None
    if float16Storage {
      print("H3-Image DiT: official native-weight FP32 baseline \(label)")
      state.rounded = false
      baseline = reference.image_dit_case(pack, hasReferences)
      state.rounded = true
    }
    print("H3-Image DiT: official 50-layer \(precision) weights, FP32 arithmetic, \(label)")
    let test = reference.image_dit_case(pack, hasReferences)
    if float16Storage {
      let rounded = tensorFromPython(test["image_output"]).reshaped(.WC(imageLength, 576))
      let native = tensorFromPython(baseline["image_output"]).reshaped(.WC(imageLength, 576))
      for head in 0..<6 {
        var roundedHead = Tensor<Float>(.CPU, .WC(imageLength, 96))
        var nativeHead = Tensor<Float>(.CPU, .WC(imageLength, 96))
        for row in 0..<imageLength {
          for channel in 0..<96 {
            roundedHead[row, channel] = rounded[row, head * 96 + channel]
            nativeHead[row, channel] = native[row, head * 96 + channel]
          }
        }
        let metrics = tokenParityMetrics(roundedHead, nativeHead)
        printMetrics("H3-Image FP16 storage vs native PDD head \(head) \(label)", metrics)
        precondition(metrics.allFinite, "FP16-rounded official output is non-finite")
      }
    }
    graph.withNoGrad {
      let image = graph.variable(tensorFromPython(test["image"]).toGPU(deviceID))
      let text = graph.variable(tensorFromPython(test["text"]).toGPU(deviceID))
      let rotary = graph.variable(
        tensorFromPython(test["rotary"])
          .reshaped(.NHWC(1, length, 1, H3Config.headDim)).toGPU(deviceID))
      let adaln = graph.variable(tensorFromPython(test["adaln_selection"]).toGPU(deviceID))
      let timestep = graph.variable(tensorFromPython(test["timestep_selection"]).toGPU(deviceID))
      let frequencies = graph.variable(tensorFromPython(test["frequencies"]).toGPU(deviceID))
      // Converted submodels are checked before the full graph, on the exact
      // intermediates captured from the official forward.
      do {
        let (model, reader) = H3TimestepEmbedding(timestepCount: timestepCount)
        model.compile(inputs: frequencies)
        reader(state)
        checkH3ImageParity(
          "H3-Image timestep \(label)",
          model(inputs: frequencies)[0].as(of: Float.self).rawValue.toCPU(),
          tensorFromPython(test["temb"]))
      }
      do {
        let input = graph.variable(tensorFromPython(test["refiner_input"]).toGPU(deviceID))
        let (model, reader) = H3TokenRefiner(sequenceLength: textLength)
        model.maxConcurrency = .limit(1)
        model.compile(inputs: input)
        reader(state)
        checkH3ImageParity(
          "H3-Image token refiner \(label)",
          model(inputs: input)[0].as(of: Float.self).rawValue.toCPU()
            .reshaped(.WC(textLength, H3Config.hiddenSize)),
          tensorFromPython(test["refiner_output"]).reshaped(.WC(textLength, H3Config.hiddenSize)))
      }
      do {
        let input = graph.variable(tensorFromPython(test["block_input"]).toGPU(deviceID))
        let activatedTemb = graph.variable(
          tensorFromPython(torch.nn.functional.silu(test["temb"]))
            .toGPU(deviceID))
        let (model, reader) = H3TransformerBlock(
          prefix: "transformer_blocks.0", sequenceLength: length, timestepCount: timestepCount)
        model.maxConcurrency = .limit(1)
        model.compile(inputs: input, rotary, adaln, activatedTemb)
        reader(state)
        checkH3ImageParity(
          "H3-Image first transformer block \(label)",
          model(inputs: input, rotary, adaln, activatedTemb)[0].as(of: Float.self).rawValue.toCPU()
            .reshaped(.WC(length, H3Config.hiddenSize)),
          tensorFromPython(test["block_output"]).reshaped(.WC(length, H3Config.hiddenSize)))
      }
      let (model, reader) = H3JointTransformer(
        textLength: textLength, audioLength: 0, videoLength: imageLength,
        timestepCount: timestepCount, includeHidden: true, pddSteps: 6, includeAudio: false)
      model.maxConcurrency = .limit(1)
      print("H3-Image: compiling image-only Swift DiT")
      model.compile(inputs: image, text, rotary, adaln, timestep, frequencies)
      let names = (0..<model.parameters.count).map { model.parameters[$0].name }
      precondition(Set(names).count == names.count, "Duplicate parameter names")
      precondition(!names.contains { $0.contains("audio_proj") })
      source.accessed.clear()
      if export && hasReferences { state.enable_export_audit() }
      print("H3-Image: loading all 50 layers into unified memory")
      reader(state)
      state.validate_coverage()
      source.release()
      print("H3-Image: executing Swift \(label)")
      let result = model(inputs: image, text, rotary, adaln, timestep, frequencies)
      let actual = result[0].as(of: Float.self).rawValue.toCPU().reshaped(.WC(imageLength, 576))
      let expected = tensorFromPython(test["image_output"]).reshaped(.WC(imageLength, 576))
      checkH3ImageParity(
        "H3-Image all six heads \(label)", actual.reshaped(.WC(imageLength * 6, 96)),
        expected.reshaped(.WC(imageLength * 6, 96)))
      for head in 0..<6 {
        var actualHead = Tensor<Float>(.CPU, .WC(imageLength, 96))
        var expectedHead = Tensor<Float>(.CPU, .WC(imageLength, 96))
        for row in 0..<imageLength {
          for channel in 0..<96 {
            actualHead[row, channel] = actual[row, head * 96 + channel]
            expectedHead[row, channel] = expected[row, head * 96 + channel]
          }
        }
        checkH3ImageParity("H3-Image PDD head \(head) \(label)", actualHead, expectedHead)
      }
      printMetrics(
        "H3-Image pre-norm residual \(label)",
        tokenParityMetrics(
          result[1].as(of: Float.self).rawValue.toCPU().reshaped(.WC(length, H3Config.hiddenSize)),
          tensorFromPython(test["pre_norm"]).reshaped(.WC(length, H3Config.hiddenSize))))
      checkH3ImageParity(
        "H3-Image final norm \(label)",
        result[2].as(of: Float.self).rawValue.toCPU().reshaped(.WC(length, H3Config.hiddenSize)),
        tensorFromPython(test["hidden"]).reshaped(.WC(length, H3Config.hiddenSize)))
      if export && hasReferences {
        // Both complete layouts and every output head passed in this session.
        print("H3-Image DiT parity passed; writing dit to", stagingPath)
        if float16Storage {
          // Same streaming writer as Pika: no intermediate FP32 checkpoint.
          try! graph.openStore(stagingPath) { store in
            try! store.write("dit", model: model, strict: true) { name, tensor in
              let value = Tensor<Float>(from: tensor).toCPU()
              try! store.write(name, tensor: Tensor<Float16>(from: value), strict: true)
              return .skip
            }
          }.get()
        } else {
          graph.openStore(stagingPath) { $0.write("dit", model: model) }
        }
        print("H3-Image DiT: strict reload from staged checkpoint")
        graph.openStore(stagingPath) { try! $0.read("dit", model: model, strict: true) }
        let reloaded = model(inputs: image, text, rotary, adaln, timestep, frequencies)[0]
          .as(of: Float.self).rawValue.toCPU().reshaped(.WC(imageLength * 6, 96))
        let metrics = tokenParityMetrics(reloaded, actual.reshaped(.WC(imageLength * 6, 96)))
        printMetrics("H3-Image DiT strict reload, all six heads", metrics)
        precondition(metrics.allFinite && metrics.maxAbsoluteDifference == 0)
        let audit = reference.audit_dit_export(stagingPath, state, names)
        try! FileManager.default.moveItem(atPath: stagingPath, toPath: exportPath)
        reference.write_dit_export_metadata(exportPath, audit)
        print("H3-Image DiT export verified:", exportPath)
      }
    }
  }
  if !export {
    print(
      "H3-Image DiT \(precision) storage parity passed; no checkpoint written (export approval required)"
    )
  }
}
