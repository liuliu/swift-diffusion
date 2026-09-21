import Foundation
import NNC
import NNCPythonConversion
import PythonKit

func q21EncodePrompt(_ prompt: String) -> Tensor<Float> {
  let inputs = reference.text_inputs(prompt)
  let result = graph.withNoGrad { () -> Tensor<Float> in
    let tokenCPU = try! Tensor<Int32>(numpy: inputs["tokens"])
    let length = tokenCPU.shape[0]
    let tokens = graph.variable(tokenCPU.toGPU(device))
    let rotary = graph.variable(q21TextRotary(length: length).toGPU(device))
    let (model, _) = Q21Text(Float16.self, length: length)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: tokens, rotary)
    graph.openStore(textCheckpoint) { try! $0.read("text_model", model: model, strict: true) }
    let output = Tensor<Float>(
      from: model(inputs: tokens, rotary)[0].as(of: Float16.self).rawValue.toCPU()
    ).makeNumpyArray()
    let trimmed = output[Python.slice(inputs["drop"], Python.None)]
    return cpu(trimmed)
  }
  return result
}

func q21Generate() {
  let prompt =
    "A red ceramic teapot on a wooden table beside a window, soft morning light, a small white cup and a folded linen napkin, realistic photograph."
  let height = 512
  let width = 512
  let steps = 40
  let seed = 42
  let outputDirectory = directory + "/artifacts/examples/generation"
  try! FileManager.default.createDirectory(
    atPath: outputDirectory, withIntermediateDirectories: true)
  let condition = q21EncodePrompt(prompt)
  let inputs = reference.generation_setup(
    condition.makeNumpyArray(), height, width, steps, seed, pythonDevice)
  let latent = graph.withNoGrad { () -> Tensor<Float> in
    let pixels = height / 16 * (width / 16)
    let text = condition.shape[0]
    var x = graph.variable(Tensor<Float>(from: cpu(inputs["initial"])).toGPU(device))
    let context = graph.variable(Tensor<Float>(from: cpu(inputs["context"])).toGPU(device))
    let rotary = graph.variable(
      q21Rotary(text: text, height: height / 16, width: width / 16).toGPU(device))
    let time = graph.variable(q21TimeEmbedding(Float(inputs["timesteps"][0])!).toGPU(device))
    // FP16 main computation with FP32 residuals; no FFN scaling is needed.
    let (model, _) = Q21DiT(
      Float16.self, text: text, height: height / 16, width: width / 16, device: device,
      mixedPrecision: true)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: x, context, time, rotary)
    graph.openStore(ditCheckpoint) { try! $0.read("dit", model: model, strict: true) }
    for step in 0..<steps {
      let timestep = Float(inputs["timesteps"][step])!
      let time = graph.variable(q21TimeEmbedding(timestep).toGPU(device))
      let velocity = model(inputs: x, context, time, rotary)[0].as(of: Float.self).reshaped(
        .NC(pixels, 64))
      let delta = Float(inputs["sigmas"][step + 1])! - Float(inputs["sigmas"][step])!
      x = DynamicGraph.Tensor<Float>(
        from: DynamicGraph.Tensor<Float>(from: x) + delta
          * DynamicGraph.Tensor<Float>(from: velocity))
      print("Swift denoising", step + 1, "/", steps)
      fflush(stdout)
    }
    return Tensor<Float>(from: x.rawValue.toCPU())
  }
  q21DecodeGeneration(
    latent: latent, height: height, width: width, outputDirectory: outputDirectory)
}

func q21DecodeGeneration(
  latent: Tensor<Float>, height: Int, width: Int, outputDirectory: String
) {
  let latentInput = reference.generation_latent_input(latent.makeNumpyArray(), height, width)
  graph.withNoGrad {
    let input = graph.variable(
      cpu(latentInput).reshaped(.NCHW(1, 64, height / 16, width / 16)).toGPU(device))
    let (decoder, _) = Q21VAE(
      encoder: false, height: height / 16, width: width / 16, device: device)
    decoder.maxConcurrency = .limit(1)
    decoder.compile(inputs: input)
    graph.openStore(vaeCheckpoint) { try! $0.read("decoder", model: decoder, strict: true) }
    let actual = decoder(inputs: input)[0].as(of: Float.self).rawValue.toCPU().makeNumpyArray()
    reference.save_rgba(actual, outputDirectory + "/swift.png")
  }
  print("Generation images:", outputDirectory)
}
