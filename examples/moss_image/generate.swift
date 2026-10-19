import Foundation
import NNC
import NNCPythonConversion
import PythonKit

private func generationText(_ prompt: String) -> Tensor<Float> {
  let pack = reference.text_case(prompt)
  let values = try! Tensor<Int32>(numpy: pack["tokens"])
  let length = values.shape[0]
  let graph = DynamicGraph()
  graph.maxConcurrency = .limit(1)
  return graph.withNoGrad {
    let tokens = graph.variable(values.reshaped(format: .NHWC, shape: values.shape).toGPU(device))
    let rotary = graph.variable(mossTextRotary(Float.self, length: length).toGPU(device))
    var causal = Tensor<Float>(.CPU, .NHWC(1, 1, length, length))
    for i in 0..<length {
      for j in 0..<length { causal[0, 0, i, j] = j > i ? -.infinity : 0 }
    }
    let mask = graph.variable(causal.toGPU(device))
    let zeros = Tensor<Float>(
      Array(repeating: 0, count: 32 * 128 * 128), .CPU, .HWC(32, 128, 128))
    let state = graph.variable(zeros.toGPU(device))
    let padding = graph.variable(
      Tensor<Float>(
        Array(repeating: 0, count: 3 * 8_192), .CPU, .NHWC(1, 3, 1, 8_192)
      ).toGPU(device))
    var triangle = Tensor<Float>(.CPU, .NHWC(7, 32, 64, 64))
    for level in 0..<7 {
      let span = 1 << level
      for head in 0..<32 {
        for i in 0..<64 {
          for j in 0..<64 {
            let include =
              level == 0
              ? j <= i
              : i / span == j / span && i % span >= span / 2 && j % span < span / 2
            triangle[level, head, i, j] = include ? 1 : 0
          }
        }
      }
    }
    let lower = graph.variable(triangle.toGPU(device))
    let (model, bindings, _) = MOSSTextEncoder(Float.self, length: length, device: device)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: tokens, rotary, mask, state, padding, lower)
    load(Float.self, bindings: bindings, state: reference.State("text_encoder"), component: "text")
    let result = model(inputs: tokens, rotary, mask, state, padding, lower)[0]
      .as(of: Float.self).rawValue.toCPU()
    print("Encoded prompt:", length, "tokens")
    return result
  }
}

func runGeneration() {
  precondition(precision == "float32", "Generation currently uses the validated FP32 path")
  let prompt =
    env["MOSS_PROMPT"]
    ?? "A red teapot on a wooden table beside a window, soft morning light."
  let negativePrompt = env["MOSS_NEGATIVE_PROMPT"] ?? ""
  let height = Int(env["MOSS_HEIGHT"] ?? "1024")!
  let width = Int(env["MOSS_WIDTH"] ?? "1024")!
  let steps = Int(env["MOSS_STEPS"] ?? "50")!
  let seed = Int(env["MOSS_SEED"] ?? "42")!
  let guidance: Float = 3
  let negativeDevice = Int(env["MOSS_NEGATIVE_DEVICE"] ?? "3")!
  let output = env["MOSS_OUTPUT"] ?? directory + "/artifacts/moss_swift_teapot.png"
  precondition(height > 0 && width > 0 && height % 16 == 0 && width % 16 == 0 && steps > 0)
  precondition(negativeDevice != device, "FP32 CFG branches use separate GPUs")
  let start = Date()
  print("Generate:", width, "x", height, "steps", steps, "seed", seed, "CFG", guidance)
  print("Prompt:", prompt)
  let positiveFeatures = generationText(prompt)
  let negativeFeatures = generationText(negativePrompt)
  let setup = reference.generation_setup(height, width, steps, seed)
  let timesteps = [Float](setup["timesteps"])!
  let sigmas = [Float](setup["sigmas"])!
  let h = height / 16
  let w = width / 16
  let graph = DynamicGraph()
  graph.maxConcurrency = .limit(1)
  let latent: Tensor<Float> = graph.withNoGrad {
    var current = graph.variable(cpu(setup["noise"]).toGPU(device)).reshaped(.WC(h * w, 128))
    func branch(_ features: Tensor<Float>, _ target: Int, _ skip: Set<Int>)
      -> (Model, DynamicGraph.Tensor<Float>, DynamicGraph.Tensor<Float>)
    {
      let length = features.shape[0]
      let context = graph.variable(features.toGPU(target)).reshaped(.WC(length, 4_096))
      let rotary = graph.variable(
        mossDiTRotary(Float.self, height: h, width: w, textLength: length).toGPU(target))
      let time = graph.variable(mossTimeEmbedding(timesteps[0]).toGPU(target))
      let input = graph.variable(cpu(setup["noise"]).toGPU(target)).reshaped(.WC(h * w, 128))
      let (model, bindings, _) = MOSSDiT(
        Float.self, textLength: length, height: h, width: w, device: target, skipLayers: skip)
      model.maxConcurrency = .limit(1)
      model.compile(inputs: input, context, rotary, time)
      let state = reference.State("transformer")
      reference.check_dit_keys(state, bindings.map { $0.key }, Array(skip))
      load(Float.self, bindings: bindings, state: state, component: "dit")
      return (model, context, rotary)
    }
    let (positive, positiveText, positiveRotary) = branch(positiveFeatures, device, [])
    let (negative, negativeText, negativeRotary) = branch(negativeFeatures, negativeDevice, [22])
    for i in timesteps.indices {
      let begin = Date()
      let negativeInput = graph.variable(current.rawValue.toCPU().toGPU(negativeDevice))
      let negativeTime = graph.variable(mossTimeEmbedding(timesteps[i]).toGPU(negativeDevice))
      let uncond = negative(inputs: negativeInput, negativeText, negativeRotary, negativeTime)[0]
        .as(of: Float.self).rawValue.toCPU()
      let negativePrediction = graph.variable(uncond.toGPU(device))
      let time = graph.variable(mossTimeEmbedding(timesteps[i]).toGPU(device))
      let conditional = positive(inputs: current, positiveText, positiveRotary, time)[0]
        .as(of: Float.self)
      let velocity = negativePrediction + (conditional - negativePrediction) * guidance
      current = current + velocity * (sigmas[i + 1] - sigmas[i])
      let sample = current.rawValue.toCPU()
      var peak: Float = 0
      for token in 0..<(h * w) {
        for channel in 0..<128 {
          let value = sample[token, channel]
          precondition(value.isFinite, "Non-finite latent at step \(i + 1)")
          peak = max(peak, abs(value))
        }
      }
      print(
        "Step", i + 1, "/", timesteps.count, "t", timesteps[i], "peak", peak,
        "seconds", String(format: "%.2f", Date().timeIntervalSince(begin)))
    }
    return current.rawValue.toCPU()
  }
  reference.decode_generation(
    latent.makeNumpyArray(), height, width, output, referenceDevice,
    prompt, negativePrompt, seed, guidance, timesteps, sigmas, Date().timeIntervalSince(start))
}
