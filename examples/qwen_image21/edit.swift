import Foundation
import NNC
import NNCPythonConversion
import PythonKit

func q21Segments(_ object: PythonObject) -> [Q21Segment] {
  object.map { row in
    Q21Segment(
      image: Bool(row[0])!, length: Int(row[1])!, sourceOffset: Int(row[2])!, height: Int(row[3])!,
      width: Int(row[4])!)
  }
}

func q21VisionForward<T: TensorNumeric>(
  _ type: T.Type, patches: PythonObject, grids: [Q21VisionGrid]
) -> [Tensor<Float>] {
  graph.withNoGrad {
    let positions = q21VisionPositions(grids)
    let x = graph.variable(Tensor<T>(from: cpu(patches)).toGPU(device))
    let ids = graph.variable(positions.ids.toGPU(device))
    let weights = graph.variable(positions.weights.toGPU(device))
    let rotary = graph.variable(positions.rotary.toGPU(device))
    let (model, _) = Q21VisionModel(type, grids: grids)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: x, ids, weights, rotary)
    graph.openStore(textCheckpoint) { try! $0.read("vision_model", model: model, strict: true) }
    let results = model(inputs: x, ids, weights, rotary).map {
      Tensor<Float>(from: $0.as(of: T.self).rawValue.toCPU())
    }
    return results
  }
}

func q21EncodeEdit(paths: [String], prompt: String, resolution: Int)
  -> (
    Tensor<Float>, PythonObject
  )
{
  let inputs = reference.edit_inputs(paths, prompt, resolution)
  let grids: [Q21VisionGrid] = inputs["grids"].map { (Int($0[0])!, Int($0[1])!, Int($0[2])!) }
  let vision = q21VisionForward(
    Float.self, patches: inputs["patches"], grids: grids)
  let context = graph.withNoGrad { () -> Tensor<Float> in
    let tokenCPU = try! Tensor<Int32>(numpy: inputs["tokens"])
    let length = tokenCPU.shape[0]
    let spans = inputs["spans"].map {
      Q21VisualSpan(start: Int($0[0])!, height: Int($0[1])!, width: Int($0[2])!)
    }
    let positions = q21MultimodalPositions(length: length, visual: spans)
    let tokens = graph.variable(tokenCPU.toGPU(device))
    let rotary = graph.variable(q21MultimodalRotary(Float.self, positions).toGPU(device))
    let features = vision.map { graph.variable(Tensor<Float>(from: $0).toGPU(device)) }
    let (model, _) = Q21Text(Float.self, length: length, visual: spans)
    model.maxConcurrency = .limit(1)
    let mask = graph.variable(q21AttentionMask(text: length, pixels: 0).toGPU(device))
    model.compile(inputs: [tokens, rotary] + features + [mask])
    graph.openStore(textCheckpoint) { try! $0.read("text_model", model: model, strict: true) }
    let output = Tensor<Float>(
      from: model(inputs: tokens, rotary, features[0], features[1], features[2], features[3], mask)[
        0
      ].as(
        of: Float.self
      ).rawValue.toCPU()
    ).makeNumpyArray()
    let trimmed = output[Python.slice(inputs["drop"], Python.None)]
    return cpu(trimmed)
  }
  return (context, inputs)
}

func q21Edit() {
  let paths = [directory + "/artifacts/examples/generation/swift.png"]
  let prompt = "Change the teapot to blue and keep the rest of the scene unchanged."
  let height = 512
  let width = 512
  let resolution = 512
  let steps = 40
  let seed = 44
  let outputDirectory = directory + "/artifacts/examples/edit"
  precondition(
    paths.allSatisfy { FileManager.default.fileExists(atPath: $0) },
    "Run generate first, or set the input images above")
  try! FileManager.default.createDirectory(
    atPath: outputDirectory, withIntermediateDirectories: true)
  let (condition, textInputs) = q21EncodeEdit(paths: paths, prompt: prompt, resolution: resolution)
  var normalizedLatents = [PythonObject]()
  for image in textInputs["vae_inputs"] {
    graph.withNoGrad {
      let tensor = cpu(image)
      let h = tensor.shape[2]
      let w = tensor.shape[3]
      let input = graph.variable(tensor.reshaped(.NCHW(1, 4, h, w)).toGPU(device))
      let (encoder, _) = Q21VAE(encoder: true, height: h, width: w, device: device)
      encoder.maxConcurrency = .limit(1)
      encoder.compile(inputs: input)
      graph.openStore(vaeCheckpoint) { try! $0.read("encoder", model: encoder, strict: true) }
      let moments = encoder(inputs: input)[0].as(of: Float.self).rawValue.toCPU().makeNumpyArray()
      normalizedLatents.append(reference.normalize_image_latent(moments))
    }
  }
  let inputs = reference.edit_generation_setup(
    condition.makeNumpyArray(), textInputs, normalizedLatents, height, width, steps, seed,
    pythonDevice)
  let latent = graph.withNoGrad { () -> Tensor<Float> in
    let pixels = height / 16 * (width / 16)
    let segments = q21Segments(inputs["segments"])
    let contextCPU = cpu(inputs["context"])
    let context = graph.variable(contextCPU.toGPU(device))
    let source = graph.variable(cpu(inputs["condition_latent"]).toGPU(device))
    var x = graph.variable(cpu(inputs["initial"]).toGPU(device))
    let rotary = graph.variable(q21LayoutRotary(segments).toGPU(device))
    let mask = graph.variable(q21LayoutMask(segments).toGPU(device))
    let time = graph.variable(q21TimeEmbedding(Float(inputs["timesteps"][0])!).toGPU(device))
    let (model, _) = Q21DiT(
      Float.self, text: contextCPU.shape[0], height: height / 16, width: width / 16, device: device,
      layout: segments)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: Functional.concat(axis: 0, source, x), context, time, rotary, mask)
    graph.openStore(ditCheckpoint) { try! $0.read("dit", model: model, strict: true) }
    for step in 0..<steps {
      let timestep = Float(inputs["timesteps"][step])!
      let time = graph.variable(q21TimeEmbedding(timestep).toGPU(device))
      let image = Functional.concat(axis: 0, source, x)
      let velocity = model(inputs: image, context, time, rotary, mask)[0].as(of: Float.self)
        .reshaped(.NC(pixels, 64))
      let delta = Float(inputs["sigmas"][step + 1])! - Float(inputs["sigmas"][step])!
      x = x + delta * velocity
      print("Swift edit denoising", step + 1, "/", steps)
      fflush(stdout)
    }
    return x.rawValue.toCPU()
  }
  q21DecodeGeneration(
    latent: latent, height: height, width: width, outputDirectory: outputDirectory)
}
