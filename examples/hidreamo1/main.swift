import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

func envString(_ name: String, _ defaultValue: String) -> String {
  ProcessInfo.processInfo.environment[name] ?? defaultValue
}

func envInt(_ name: String, _ defaultValue: Int) -> Int {
  guard let value = ProcessInfo.processInfo.environment[name], let intValue = Int(value) else {
    return defaultValue
  }
  return intValue
}

func envFloat(_ name: String, _ defaultValue: Float) -> Float {
  guard let value = ProcessInfo.processInfo.environment[name], let floatValue = Float(value) else {
    return defaultValue
  }
  return floatValue
}

func envFlag(_ name: String) -> Bool {
  guard let value = ProcessInfo.processInfo.environment[name] else { return false }
  return value == "1" || value.lowercased() == "true" || value.lowercased() == "yes"
}

enum HiDreamO1Paths {
  static let modelPath = envString("HIDREAMO1_MODEL_PATH", "/slow/Data/HiDream-O1-Image-Dev")
  static let referenceRoot = envString(
    "HIDREAMO1_REFERENCE_ROOT",
    "/home/liu/workspace/swift-diffusion/examples/hidreamo1")
  static let ltxEnvSitePackages =
    "/home/liu/workspace/ltx2/LTX-2/_env/lib/python3.12/site-packages"
  static let userSitePackages = Python.import("site").getusersitepackages()
  static let systemDistPackages = "/usr/lib/python3/dist-packages"
}

enum HiDreamO1Config {
  static let patchSize = 32
  static let imageChannels = 3
  static let patchDimension = imageChannels * patchSize * patchSize
  static let hiddenSize = 4_096
  static let timestepFrequencySize = 256
  static let pixelBottleneckSize = hiddenSize / 4
  static let samplePatchTokens = envInt("HIDREAMO1_PARITY_PATCH_TOKENS", 4)
  static let parityHeight = envInt("HIDREAMO1_PARITY_HEIGHT", 64)
  static let parityWidth = envInt("HIDREAMO1_PARITY_WIDTH", 64)
  static let generationHeight = envInt("HIDREAMO1_GENERATION_HEIGHT", 512)
  static let generationWidth = envInt("HIDREAMO1_GENERATION_WIDTH", 512)
  static let generationLayers = envInt("HIDREAMO1_GENERATION_LAYERS", 36)
  static let generationDevice = envInt("HIDREAMO1_SWIFT_DEVICE", 3)
  static let generationSeed = envInt("HIDREAMO1_SEED", 32)
  static let generationNoiseScale = envFloat("HIDREAMO1_NOISE_SCALE", 7.5)
  static let generationNoiseClipStd = envFloat("HIDREAMO1_NOISE_CLIP_STD", 2.5)
  static let storePath = envString(
    "HIDREAMO1_STORE_PATH", "/slow/Data/HiDream-O1-Image-Dev/hidream_o1_dev_f32.ckpt")
  static let denoiserParityHeight = envInt("HIDREAMO1_DENOISER_PARITY_HEIGHT", 64)
  static let denoiserParityWidth = envInt("HIDREAMO1_DENOISER_PARITY_WIDTH", 64)
  static let denoiserParityLayers = envInt("HIDREAMO1_DENOISER_PARITY_LAYERS", 1)
  static let denoiserParitySeed = envInt("HIDREAMO1_DENOISER_PARITY_SEED", 123)
  static let denoiserParityTimestep = envFloat("HIDREAMO1_DENOISER_PARITY_TIMESTEP", 0.001)
  static let outputPath = envString(
    "HIDREAMO1_OUTPUT_PATH", "/slow/Data/HiDream-O1-Image-Dev/o1_swift_dev.png")
  static let devFirstTimestep: Float = 0.001  // 1 - 999 / 1000.
  static let devTimesteps: [Float] = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
  ]
}

func preparePythonPath() {
  let sys = Python.import("sys")
  let paths = [
    HiDreamO1Paths.referenceRoot,
    HiDreamO1Paths.ltxEnvSitePackages,
    String(HiDreamO1Paths.userSitePackages)!,
    HiDreamO1Paths.systemDistPackages,
  ]
  for path in paths.reversed() {
    if !Bool(sys.path.__contains__(path))! {
      sys.path.insert(0, path)
    }
  }
}

preparePythonPath()
let torch = Python.import("torch")
torch.set_grad_enabled(false)
let reference = Python.import("reference")

func loadLog(_ message: String) {
  if envFlag("HIDREAMO1_VERBOSE_LOAD") {
    print(message)
  }
}

func tensorFromPython(_ tensor: PythonObject) -> Tensor<Float> {
  try! Tensor<Float>(numpy: tensor.type(torch.float).cpu().numpy())
}

func tensorInt32FromPython(_ tensor: PythonObject) -> Tensor<Int32> {
  try! Tensor<Int32>(numpy: tensor.type(torch.int32).cpu().numpy())
}

func maxAbsAndRelativeDiff2D<T: TensorNumeric & BinaryFloatingPoint>(
  _ swiftTensor: DynamicGraph.Tensor<T>, _ torchTensor: Tensor<Float>
) -> (Float, Float) {
  precondition(swiftTensor.shape.count == 2)
  precondition(torchTensor.shape.count == 2)
  precondition(swiftTensor.shape[0] == torchTensor.shape[0])
  precondition(swiftTensor.shape[1] == torchTensor.shape[1])
  var maxDiff: Float = 0
  var maxReferenceAbs: Float = 0
  for i in 0..<torchTensor.shape[0] {
    for j in 0..<torchTensor.shape[1] {
      let referenceAbs = abs(torchTensor[i, j])
      maxReferenceAbs = max(maxReferenceAbs, referenceAbs)
      maxDiff = max(maxDiff, abs(Float(swiftTensor[i, j]) - torchTensor[i, j]))
    }
  }
  return (maxDiff, maxDiff / max(1e-6, maxReferenceAbs))
}

func printSamples<T: TensorNumeric & BinaryFloatingPoint>(
  _ label: String, _ swiftTensor: DynamicGraph.Tensor<T>, _ torchTensor: Tensor<Float>,
  count: Int = 8
) {
  let total = min(count, torchTensor.shape.reduce(1, *))
  let width = torchTensor.shape[1]
  var values = [String]()
  for index in 0..<total {
    let i = index / width
    let j = index % width
    values.append("\(Float(swiftTensor[i, j]))/\(torchTensor[i, j])")
  }
  print("\(label) samples swift/reference:", values.joined(separator: ", "))
}

func qkvInterleavedWeight(_ tensor: PythonObject, heads: Int, headDimension: Int) -> Tensor<Float> {
  let numpy = tensor.type(torch.float).view(heads, 2, headDimension / 2, -1).transpose(1, 2).cpu()
    .numpy()
  return try! Tensor<Float>(numpy: numpy)
}

func qkvInterleavedNorm(_ tensor: PythonObject, headDimension: Int) -> Tensor<Float> {
  let numpy = tensor.type(torch.float).view(2, headDimension / 2).transpose(0, 1).cpu().numpy()
  return try! Tensor<Float>(numpy: numpy)
}

func qwen3VLMRotary(positionIDs: Tensor<Int32>, headDimension: Int = 128, theta: Double = 5_000_000)
  -> Tensor<Float>
{
  precondition(positionIDs.shape.count == 3)
  precondition(positionIDs.shape[0] == 3)
  let batch = positionIDs.shape[1]
  let tokenLength = positionIDs.shape[2]
  precondition(batch == 1)
  let half = headDimension / 2
  let mropeSection = [24, 20, 20]
  var rotary = Tensor<Float>(.CPU, .NHWC(batch, tokenLength, 1, headDimension))
  for token in 0..<tokenLength {
    for i in 0..<half {
      var axis = 0
      if i < mropeSection[1] * 3 && i % 3 == 1 {
        axis = 1
      } else if i < mropeSection[2] * 3 && i % 3 == 2 {
        axis = 2
      }
      let pos = Double(positionIDs[axis, 0, token])
      let freq = pos / pow(theta, Double(i) / Double(half))
      rotary[0, token, 0, i * 2] = Float(cos(freq))
      rotary[0, token, 0, i * 2 + 1] = Float(sin(freq))
    }
  }
  return rotary
}

func mixedAttentionMask(tokenTypes: Tensor<Int32>) -> Tensor<Float> {
  precondition(tokenTypes.shape.count == 2)
  precondition(tokenTypes.shape[0] == 1)
  let tokenLength = tokenTypes.shape[1]
  var mask = Tensor<Float>(.CPU, .NHWC(1, 1, tokenLength, tokenLength))
  let masked = -Float.greatestFiniteMagnitude
  for i in 0..<tokenLength {
    let isGen = tokenTypes[0, i] != 0
    for j in 0..<tokenLength {
      mask[0, 0, i, j] = (!isGen && j > i) ? masked : 0
    }
  }
  return mask
}

func timeEmbedding(timestep: Float, embeddingSize: Int = HiDreamO1Config.timestepFrequencySize)
  -> Tensor<Float>
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(1, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq = timestep * 1000 * exp(-log(Float(10_000)) * Float(i) / Float(half))
    embedding[0, i] = cos(freq)
    embedding[0, i + half] = sin(freq)
  }
  return embedding
}

func TimestepEmbedder() -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let fc0 = Dense(count: HiDreamO1Config.hiddenSize, name: "t_embedder_mlp_0")
  let fc1 = Dense(count: HiDreamO1Config.hiddenSize, name: "t_embedder_mlp_2")
  let out = fc1(fc0(x).swish())
  let reader: (PythonObject) -> Void = { stateDict in
    fc0.weight.copy(
      from: tensorFromPython(stateDict["model.t_embedder1.mlp.0.weight"]))
    fc0.bias.copy(
      from: tensorFromPython(stateDict["model.t_embedder1.mlp.0.bias"]))
    fc1.weight.copy(
      from: tensorFromPython(stateDict["model.t_embedder1.mlp.2.weight"]))
    fc1.bias.copy(
      from: tensorFromPython(stateDict["model.t_embedder1.mlp.2.bias"]))
  }
  return (Model([x], [out]), reader)
}

func PixelEmbedder() -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let proj1 = Dense(
    count: HiDreamO1Config.pixelBottleneckSize, noBias: true, name: "x_embedder_proj1")
  let proj2 = Dense(count: HiDreamO1Config.hiddenSize, name: "x_embedder_proj2")
  let out = proj2(proj1(x))
  let reader: (PythonObject) -> Void = { stateDict in
    proj1.weight.copy(
      from: tensorFromPython(stateDict["model.x_embedder.proj1.weight"]))
    proj2.weight.copy(
      from: tensorFromPython(stateDict["model.x_embedder.proj2.weight"]))
    proj2.bias.copy(
      from: tensorFromPython(stateDict["model.x_embedder.proj2.bias"]))
  }
  return (Model([x], [out]), reader)
}

func FinalLayer() -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let linear = Dense(count: HiDreamO1Config.patchDimension, name: "final_layer2")
  let out = linear(x)
  let reader: (PythonObject) -> Void = { stateDict in
    linear.weight.copy(
      from: tensorFromPython(stateDict["model.final_layer2.linear.weight"]))
    linear.bias.copy(
      from: tensorFromPython(stateDict["model.final_layer2.linear.bias"]))
  }
  return (Model([x], [out]), reader)
}

func PixelRoundTrip() -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let (embedder, embedderReader) = PixelEmbedder()
  let (final, finalReader) = FinalLayer()
  let out = final(embedder(x))
  let reader: (PythonObject) -> Void = { stateDict in
    embedderReader(stateDict)
    finalReader(stateDict)
  }
  return (Model([x], [out]), reader)
}

func HiDreamO1FeedForward(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let gate = Dense(count: intermediateSize, noBias: true, name: "mlp_gate_proj")
  let up = Dense(count: intermediateSize, noBias: true, name: "mlp_up_proj")
  let down = Dense(count: hiddenSize, noBias: true, name: "mlp_down_proj")
  let out = down(gate(x).swish() .* up(x))
  return (gate, up, down, Model([x], [out]))
}

func HiDreamO1SelfAttention(
  prefix: String, width: Int, headDimension: Int, heads: Int, kvHeads: Int, tokenLength: Int
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = Input()
  let attentionMask = Input()
  let toKeys = Dense(count: headDimension * kvHeads, noBias: true, name: "k_proj")
  let toQueries = Dense(count: headDimension * heads, noBias: true, name: "q_proj")
  let toValues = Dense(count: headDimension * kvHeads, noBias: true, name: "v_proj")
  var keys = toKeys(x).reshaped([1, tokenLength, kvHeads, headDimension])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "k_norm")
  keys = normK(keys)
  var queries = toQueries(x).reshaped([1, tokenLength, heads, headDimension])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "q_norm")
  queries = normQ(queries)
  let values = toValues(x).reshaped([1, tokenLength, kvHeads, headDimension]).transposed(1, 2)
  queries = Functional.cmul(left: queries, right: rot).transposed(1, 2)
  keys = Functional.cmul(left: keys, right: rot).transposed(1, 2)
  var outs = [Model.IO]()
  let headsPerKVHead = heads / kvHeads
  for i in 0..<kvHeads {
    let query = queries.reshaped(
      [1, headsPerKVHead, tokenLength, headDimension],
      offset: [0, i * headsPerKVHead, 0, 0],
      strides: [heads * tokenLength * headDimension, tokenLength * headDimension, headDimension, 1])
    let key = keys.reshaped(
      [1, 1, tokenLength, headDimension],
      offset: [0, i, 0, 0],
      strides: [
        kvHeads * tokenLength * headDimension, tokenLength * headDimension, headDimension, 1,
      ])
    let value = values.reshaped(
      [1, 1, tokenLength, headDimension],
      offset: [0, i, 0, 0],
      strides: [
        kvHeads * tokenLength * headDimension, tokenLength * headDimension, headDimension, 1,
      ])
    var dot =
      (1.0 / Float(headDimension).squareRoot()) * Matmul(transposeB: (2, 3))(query, key)
      + attentionMask
    if let last = outs.last {
      dot.add(dependencies: [last])
    }
    dot = dot.reshaped([headsPerKVHead * tokenLength, tokenLength]).softmax()
    dot = dot.reshaped([1, headsPerKVHead, tokenLength, tokenLength])
    outs.append(dot * value)
  }
  var out = Concat(axis: 1)(outs).reshaped([1, heads, tokenLength, headDimension])
    .transposed(1, 2).reshaped([tokenLength, heads * headDimension])
  let unifyHeads = Dense(count: width, noBias: true, name: "o_proj")
  out = unifyHeads(out)
  let reader: (PythonObject) -> Void = { stateDict in
    toQueries.weight.copy(
      from: qkvInterleavedWeight(
        stateDict["\(prefix).self_attn.q_proj.weight"], heads: heads,
        headDimension: headDimension))
    normQ.weight.copy(
      from: qkvInterleavedNorm(
        stateDict["\(prefix).self_attn.q_norm.weight"], headDimension: headDimension))
    toKeys.weight.copy(
      from: qkvInterleavedWeight(
        stateDict["\(prefix).self_attn.k_proj.weight"], heads: kvHeads,
        headDimension: headDimension))
    normK.weight.copy(
      from: qkvInterleavedNorm(
        stateDict["\(prefix).self_attn.k_norm.weight"], headDimension: headDimension))
    toValues.weight.copy(from: tensorFromPython(stateDict["\(prefix).self_attn.v_proj.weight"]))
    unifyHeads.weight.copy(from: tensorFromPython(stateDict["\(prefix).self_attn.o_proj.weight"]))
  }
  return (Model([x, rot, attentionMask], [out]), reader)
}

func HiDreamO1DecoderLayer(layerIdx: Int, tokenLength: Int) -> (Model, (PythonObject) -> Void) {
  let prefix = "model.language_model.layers.\(layerIdx)"
  let x = Input()
  let rot = Input()
  let attentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attentionReader) = HiDreamO1SelfAttention(
    prefix: prefix, width: HiDreamO1Config.hiddenSize, headDimension: 128, heads: 32, kvHeads: 8,
    tokenLength: tokenLength)
  out = x + attention(out, rot, attentionMask)
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (gate, up, down, feedForward) = HiDreamO1FeedForward(
    hiddenSize: HiDreamO1Config.hiddenSize, intermediateSize: 12_288)
  out = residual + feedForward(out)
  let reader: (PythonObject) -> Void = { stateDict in
    attentionReader(stateDict)
    norm1.weight.copy(from: tensorFromPython(stateDict["\(prefix).input_layernorm.weight"]))
    norm2.weight.copy(
      from: tensorFromPython(stateDict["\(prefix).post_attention_layernorm.weight"]))
    gate.weight.copy(from: tensorFromPython(stateDict["\(prefix).mlp.gate_proj.weight"]))
    up.weight.copy(from: tensorFromPython(stateDict["\(prefix).mlp.up_proj.weight"]))
    down.weight.copy(from: tensorFromPython(stateDict["\(prefix).mlp.down_proj.weight"]))
  }
  return (Model([x, rot, attentionMask], [out]), reader)
}

func HiDreamO1Denoiser(
  textPrefixLength: Int, imageTokenCount: Int, layerCount: Int
) -> ((PythonObject) -> Void, Model) {
  let textPrefix = Input()
  let timestepFrequency = Input()
  let pixelPatches = Input()
  let rot = Input()
  let attentionMask = Input()
  let (timestepEmbedder, timestepReader) = TimestepEmbedder()
  let (pixelEmbedder, pixelReader) = PixelEmbedder()
  let tEmbedding = timestepEmbedder(timestepFrequency)
  let pixelEmbedding = pixelEmbedder(pixelPatches)
  var out = Concat(axis: 0)(textPrefix, tEmbedding, pixelEmbedding)
  var readers = [(PythonObject) -> Void]()
  for layerIdx in 0..<layerCount {
    let (layer, reader) = HiDreamO1DecoderLayer(
      layerIdx: layerIdx, tokenLength: textPrefixLength + 1 + imageTokenCount)
    out = layer(out, rot, attentionMask)
    readers.append { stateDict in
      loadLog("load model.language_model.layers.\(layerIdx)")
      reader(stateDict)
    }
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out)
  let (finalLayer, finalReader) = FinalLayer()
  var predicted = finalLayer(out)
  predicted = predicted.reshaped(
    [imageTokenCount, HiDreamO1Config.patchDimension],
    offset: [textPrefixLength + 1, 0],
    strides: [HiDreamO1Config.patchDimension, 1])
  let reader: (PythonObject) -> Void = { stateDict in
    timestepReader(stateDict)
    pixelReader(stateDict)
    for reader in readers {
      reader(stateDict)
    }
    norm.weight.copy(from: tensorFromPython(stateDict["model.language_model.norm.weight"]))
    finalReader(stateDict)
  }
  return (
    reader, Model([textPrefix, timestepFrequency, pixelPatches, rot, attentionMask], [predicted])
  )
}

func loadHiDreamO1DenoiserWeights(_ denoiser: Model, reader: (PythonObject) -> Void) {
  if envFlag("HIDREAMO1_LOAD_WEIGHTS") {
    print("hidream-o1 reading denoiser store:", HiDreamO1Config.storePath)
    graph.openStore(HiDreamO1Config.storePath, flags: [.readOnly]) {
      try! $0.read("denoiser", model: denoiser, strict: true)
    }
  } else {
    reader(reference.lazy_state(HiDreamO1Paths.modelPath))
  }
  if envFlag("HIDREAMO1_WRITE_WEIGHTS") {
    print("hidream-o1 writing denoiser store:", HiDreamO1Config.storePath)
    graph.openStore(HiDreamO1Config.storePath) {
      $0.write("denoiser", model: denoiser)
    }
  }
}

func runSampleConstructionProbe() {
  let prompt =
    "A friendly golden retriever sitting in a sunlit park, holding a white sign that says 'HiDream O1 works' in clear black letters."
  let sample = reference.build_t2i_text_sample(
    HiDreamO1Paths.modelPath, prompt, HiDreamO1Config.parityHeight, HiDreamO1Config.parityWidth)
  let inputIDs = sample["input_ids"]
  let positionIDs = sample["position_ids"]
  let tokenTypes = sample["token_types"]
  let vinputMask = sample["vinput_mask"]
  print("hidream-o1 sample input_ids shape:", inputIDs.shape)
  print("hidream-o1 sample position_ids shape:", positionIDs.shape)
  print("hidream-o1 sample token_types shape:", tokenTypes.shape)
  print("hidream-o1 sample vinput_mask true count:", vinputMask.sum().item())
  print(
    "hidream-o1 sample last text token ids:",
    inputIDs[0, Python.slice(-8, Python.None, Python.None)].tolist())
}

func runTimestepParity(stateDict: PythonObject) -> Bool {
  graph.withNoGrad {
    let timestep = HiDreamO1Config.devFirstTimestep
    let tFreqTorch = reference.timestep_frequency(timestep)
    let expected = tensorFromPython(reference.timestep_embedder(stateDict, timestep))
    let input = graph.variable(tensorFromPython(tFreqTorch))
    let (model, reader) = TimestepEmbedder()
    model.compile(inputs: input)
    reader(stateDict)
    let swiftOutput = model(inputs: input)[0].as(of: Float.self).toCPU()
    let (maxAbs, relative) = maxAbsAndRelativeDiff2D(swiftOutput, expected)
    print("hidream-o1 timestep parity shape:", swiftOutput.shape, expected.shape)
    printSamples("hidream-o1 timestep", swiftOutput, expected)
    print("hidream-o1 timestep max-abs diff:", maxAbs, "relative:", relative)
    return maxAbs <= 1e-4 && relative <= 1e-5
  }
}

func runPixelEmbedderParity(stateDict: PythonObject) -> Bool {
  graph.withNoGrad {
    let xTorch = reference.random_patch_tokens(HiDreamO1Config.samplePatchTokens)
    let expected = tensorFromPython(reference.pixel_embedder(stateDict, xTorch))
    let input = graph.variable(tensorFromPython(xTorch))
    let (model, reader) = PixelEmbedder()
    model.compile(inputs: input)
    reader(stateDict)
    let swiftOutput = model(inputs: input)[0].as(of: Float.self).toCPU()
    let (maxAbs, relative) = maxAbsAndRelativeDiff2D(swiftOutput, expected)
    print("hidream-o1 pixel embedder parity shape:", swiftOutput.shape, expected.shape)
    printSamples("hidream-o1 pixel embedder", swiftOutput, expected)
    print("hidream-o1 pixel embedder max-abs diff:", maxAbs, "relative:", relative)
    return maxAbs <= 1e-4 && relative <= 1e-5
  }
}

func runFinalLayerParity(stateDict: PythonObject) -> Bool {
  graph.withNoGrad {
    let xTorch = reference.random_hidden(HiDreamO1Config.samplePatchTokens)
    let expected = tensorFromPython(reference.final_layer(stateDict, xTorch))
    let input = graph.variable(tensorFromPython(xTorch))
    let (model, reader) = FinalLayer()
    model.compile(inputs: input)
    reader(stateDict)
    let swiftOutput = model(inputs: input)[0].as(of: Float.self).toCPU()
    let (maxAbs, relative) = maxAbsAndRelativeDiff2D(swiftOutput, expected)
    print("hidream-o1 final layer parity shape:", swiftOutput.shape, expected.shape)
    printSamples("hidream-o1 final layer", swiftOutput, expected)
    print("hidream-o1 final layer max-abs diff:", maxAbs, "relative:", relative)
    return maxAbs <= 1e-4 && relative <= 1e-5
  }
}

func runPixelRoundTripParity(stateDict: PythonObject) -> Bool {
  graph.withNoGrad {
    let xTorch = reference.random_patch_tokens(HiDreamO1Config.samplePatchTokens, seed: 45)
    let expected = tensorFromPython(reference.pixel_roundtrip(stateDict, xTorch))
    let input = graph.variable(tensorFromPython(xTorch))
    let (model, reader) = PixelRoundTrip()
    model.compile(inputs: input)
    reader(stateDict)
    let swiftOutput = model(inputs: input)[0].as(of: Float.self).toCPU()
    let (maxAbs, relative) = maxAbsAndRelativeDiff2D(swiftOutput, expected)
    print("hidream-o1 pixel roundtrip parity shape:", swiftOutput.shape, expected.shape)
    printSamples("hidream-o1 pixel roundtrip", swiftOutput, expected)
    print("hidream-o1 pixel roundtrip max-abs diff:", maxAbs, "relative:", relative)
    return maxAbs <= 2e-4 && relative <= 2e-5
  }
}

func runDecoderLayerParity(stateDict: PythonObject) -> Bool {
  graph.withNoGrad {
    let prompt = "A friendly golden retriever sitting in a sunlit park."
    let sample = reference.build_t2i_text_sample(
      HiDreamO1Paths.modelPath, prompt, HiDreamO1Config.parityHeight, HiDreamO1Config.parityWidth)
    let positionIDs = tensorInt32FromPython(sample["position_ids"])
    let tokenTypes = tensorInt32FromPython(sample["token_types"])
    let tokenLength = tokenTypes.shape[1]
    let hiddenStatesTorch = reference.random_hidden_sequence(tokenLength)
    let expected = tensorFromPython(
      reference.qwen3vl_decoder_layer(
        stateDict, hiddenStatesTorch, sample["position_ids"], sample["token_types"], 0)[0])
    let hiddenStates = graph.variable(tensorFromPython(hiddenStatesTorch[0]))
    let rot = graph.variable(qwen3VLMRotary(positionIDs: positionIDs))
    let attentionMask = graph.variable(mixedAttentionMask(tokenTypes: tokenTypes))
    let (model, reader) = HiDreamO1DecoderLayer(layerIdx: 0, tokenLength: tokenLength)
    model.compile(inputs: hiddenStates, rot, attentionMask)
    reader(stateDict)
    let swiftOutput = model(inputs: hiddenStates, rot, attentionMask)[0].as(of: Float.self).toCPU()
    let (maxAbs, relative) = maxAbsAndRelativeDiff2D(swiftOutput, expected)
    print("hidream-o1 decoder layer0 parity shape:", swiftOutput.shape, expected.shape)
    printSamples("hidream-o1 decoder layer0", swiftOutput, expected)
    print("hidream-o1 decoder layer0 max-abs diff:", maxAbs, "relative:", relative)
    return maxAbs <= 2e-3 && relative <= 2e-4
  }
}

func runDenoiserParity() -> Bool {
  graph.withNoGrad {
    let prompt = envString(
      "HIDREAMO1_DENOISER_PARITY_PROMPT",
      "A friendly golden retriever sitting in a sunlit park.")
    let height = HiDreamO1Config.denoiserParityHeight
    let width = HiDreamO1Config.denoiserParityWidth
    let layerCount = HiDreamO1Config.denoiserParityLayers
    let timestep = HiDreamO1Config.denoiserParityTimestep
    let device = HiDreamO1Config.generationDevice
    let referenceDevice = envString("HIDREAMO1_DENOISER_REFERENCE_DEVICE", "cuda:\(device)")
    print(
      "hidream-o1 denoiser parity size:", width, "x", height, "layers:", layerCount,
      "timestep:", timestep, "swift device:", device, "reference device:", referenceDevice)
    let sample = reference.generation_inputs(HiDreamO1Paths.modelPath, prompt, height, width)
    let textPrefixCPU = tensorFromPython(sample["text_prefix_embeddings"])
    let positionIDs = tensorInt32FromPython(sample["position_ids"])
    let tokenTypes = tensorInt32FromPython(sample["token_types"])
    let imageTokenCount = Int(sample["image_token_count"])!
    let textPrefixLength = textPrefixCPU.shape[0]
    let tokenLength = tokenTypes.shape[1]
    precondition(tokenLength == textPrefixLength + 1 + imageTokenCount)
    let patchesTorch = reference.random_patch_tokens(
      imageTokenCount, seed: HiDreamO1Config.denoiserParitySeed)
    print("hidream-o1 computing torch denoiser reference")
    let expected = tensorFromPython(
      reference.denoiser_step(
        HiDreamO1Paths.modelPath, prompt, patchesTorch, timestep, height, width, layerCount,
        referenceDevice, true))
    print("hidream-o1 compiling swift denoiser parity graph")
    let textPrefix = graph.variable(textPrefixCPU).toGPU(device)
    let tFreq = graph.variable(timeEmbedding(timestep: timestep)).toGPU(device)
    let patches = graph.variable(tensorFromPython(patchesTorch)).toGPU(device)
    let rot = graph.variable(qwen3VLMRotary(positionIDs: positionIDs)).toGPU(device)
    let attentionMask = graph.variable(mixedAttentionMask(tokenTypes: tokenTypes)).toGPU(device)
    let (reader, denoiser) = HiDreamO1Denoiser(
      textPrefixLength: textPrefixLength, imageTokenCount: imageTokenCount, layerCount: layerCount)
    denoiser.compile(inputs: textPrefix, tFreq, patches, rot, attentionMask)
    print("hidream-o1 loading swift denoiser parity weights")
    loadHiDreamO1DenoiserWeights(denoiser, reader: reader)
    let swiftOutput = denoiser(inputs: textPrefix, tFreq, patches, rot, attentionMask)[0].as(
      of: Float.self
    ).toCPU()
    let (maxAbs, relative) = maxAbsAndRelativeDiff2D(swiftOutput, expected)
    print("hidream-o1 denoiser parity shape:", swiftOutput.shape, expected.shape)
    printSamples("hidream-o1 denoiser", swiftOutput, expected)
    print("hidream-o1 denoiser max-abs diff:", maxAbs, "relative:", relative)
    return maxAbs <= 5e-3 && relative <= 5e-4
  }
}

func savePatchImage(_ patches: DynamicGraph.Tensor<Float>, height: Int, width: Int, path: String) {
  precondition(height % HiDreamO1Config.patchSize == 0)
  precondition(width % HiDreamO1Config.patchSize == 0)
  precondition(patches.shape.count == 2)
  let patchSize = HiDreamO1Config.patchSize
  let patchArea = patchSize * patchSize
  let heightPatches = height / patchSize
  let widthPatches = width / patchSize
  precondition(patches.shape[0] == heightPatches * widthPatches)
  precondition(patches.shape[1] == HiDreamO1Config.patchDimension)
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: height * width)
  func encode(_ value: Float) -> UInt8 {
    guard value.isFinite else { return 0 }
    let scaled = Int(((value + 1) * 0.5 * 255).rounded())
    return UInt8(min(max(scaled, 0), 255))
  }
  for patchY in 0..<heightPatches {
    for patchX in 0..<widthPatches {
      let token = patchY * widthPatches + patchX
      for yInPatch in 0..<patchSize {
        for xInPatch in 0..<patchSize {
          let y = patchY * patchSize + yInPatch
          let x = patchX * patchSize + xInPatch
          let offset = yInPatch * patchSize + xInPatch
          let index = y * width + x
          rgba[index].r = encode(patches[token, offset])
          rgba[index].g = encode(patches[token, patchArea + offset])
          rgba[index].b = encode(patches[token, patchArea * 2 + offset])
        }
      }
    }
  }
  let directory = URL(fileURLWithPath: path).deletingLastPathComponent().path
  try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (width, height),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! image.compress(path: path, level: 4)
}

func runGeneration() {
  graph.withNoGrad {
    precondition(HiDreamO1Config.generationHeight % HiDreamO1Config.patchSize == 0)
    precondition(HiDreamO1Config.generationWidth % HiDreamO1Config.patchSize == 0)
    DynamicGraph.setSeed(UInt32(HiDreamO1Config.generationSeed + 1))
    let prompt = envString(
      "HIDREAMO1_PROMPT",
      "A friendly golden retriever sitting in a sunlit park, holding a white sign that says 'HiDream O1 works' in clear black letters."
    )
    print("hidream-o1 generation prompt:", prompt)
    print(
      "hidream-o1 generation size:", HiDreamO1Config.generationWidth, "x",
      HiDreamO1Config.generationHeight, "layers:", HiDreamO1Config.generationLayers,
      "device:", HiDreamO1Config.generationDevice)
    let sample = reference.generation_inputs(
      HiDreamO1Paths.modelPath, prompt, HiDreamO1Config.generationHeight,
      HiDreamO1Config.generationWidth)
    let textPrefixCPU = tensorFromPython(sample["text_prefix_embeddings"])
    let positionIDs = tensorInt32FromPython(sample["position_ids"])
    let tokenTypes = tensorInt32FromPython(sample["token_types"])
    let imageTokenCount = Int(sample["image_token_count"])!
    let textPrefixLength = textPrefixCPU.shape[0]
    let tokenLength = tokenTypes.shape[1]
    precondition(tokenLength == textPrefixLength + 1 + imageTokenCount)
    let device = HiDreamO1Config.generationDevice
    let textPrefix = graph.variable(textPrefixCPU).toGPU(device)
    let tFreq0 = graph.variable(timeEmbedding(timestep: HiDreamO1Config.devFirstTimestep)).toGPU(
      device)
    let rot = graph.variable(qwen3VLMRotary(positionIDs: positionIDs)).toGPU(device)
    let attentionMask = graph.variable(mixedAttentionMask(tokenTypes: tokenTypes))
      .toGPU(device)
    var z = graph.variable(
      .GPU(device), .NC(imageTokenCount, HiDreamO1Config.patchDimension), of: Float.self)
    z.randn(std: HiDreamO1Config.generationNoiseScale, mean: 0)
    let (reader, denoiser) = HiDreamO1Denoiser(
      textPrefixLength: textPrefixLength, imageTokenCount: imageTokenCount,
      layerCount: HiDreamO1Config.generationLayers)
    print("hidream-o1 compiling denoiser")
    denoiser.compile(inputs: textPrefix, tFreq0, z, rot, attentionMask)
    print("hidream-o1 loading denoiser weights")
    let loadStart = Date()
    loadHiDreamO1DenoiserWeights(denoiser, reader: reader)
    print("hidream-o1 weights loaded in", Date().timeIntervalSince(loadStart), "seconds")
    let steps = HiDreamO1Config.devTimesteps
    let runStart = Date()
    for (stepIdx, timestep) in steps.enumerated() {
      let tPixel = 1 - timestep / 1000
      let tFreq = graph.variable(timeEmbedding(timestep: tPixel)).toGPU(device)
      let xPred = denoiser(inputs: textPrefix, tFreq, z, rot, attentionMask)[0]
        .as(of: Float.self)
      let sigmaNext = stepIdx + 1 < steps.count ? steps[stepIdx + 1] / 1000 : 0
      if sigmaNext > 0 {
        let noise = graph.variable(
          .GPU(device), .NC(imageTokenCount, HiDreamO1Config.patchDimension), of: Float.self)
        noise.randn(std: 1, mean: 0)
        let clip = HiDreamO1Config.generationNoiseClipStd
        let clippedNoise = clip > 0 ? noise.clamped((-clip)...clip) : noise
        z =
          ((1 - sigmaNext) * xPred)
          + (sigmaNext * HiDreamO1Config.generationNoiseScale) * clippedNoise
      } else {
        z = xPred
      }
      print(
        "hidream-o1 generation step \(stepIdx + 1)/\(steps.count) timestep \(Int(timestep)) elapsed",
        Date().timeIntervalSince(runStart))
    }
    let output = z.toCPU()
    savePatchImage(
      output, height: HiDreamO1Config.generationHeight, width: HiDreamO1Config.generationWidth,
      path: HiDreamO1Config.outputPath)
    print("hidream-o1 wrote image:", HiDreamO1Config.outputPath)
  }
}

print("hidream-o1 model path:", HiDreamO1Paths.modelPath)
if !envFlag("HIDREAMO1_SKIP_PARITY") {
  runSampleConstructionProbe()
  let auxStateDict = reference.load_aux_state(HiDreamO1Paths.modelPath)
  let layer0StateDict = reference.load_layer_state(HiDreamO1Paths.modelPath, 0)
  let passed = [
    runTimestepParity(stateDict: auxStateDict),
    runPixelEmbedderParity(stateDict: auxStateDict),
    runFinalLayerParity(stateDict: auxStateDict),
    runPixelRoundTripParity(stateDict: auxStateDict),
    runDecoderLayerParity(stateDict: layer0StateDict),
  ].allSatisfy { $0 }

  if !passed {
    fatalError("hidream-o1 initial parity failed")
  }
  print("hidream-o1 initial parity passed")
} else {
  print("hidream-o1 parity skipped by HIDREAMO1_SKIP_PARITY")
}
if envFlag("HIDREAMO1_RUN_DENOISER_PARITY") {
  if !runDenoiserParity() {
    fatalError("hidream-o1 denoiser parity failed")
  }
  print("hidream-o1 denoiser parity passed")
}
if envFlag("HIDREAMO1_RUN_GENERATION") {
  runGeneration()
}
