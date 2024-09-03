import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let onnx = Python.import("onnx")

let onnxruntime = Python.import("onnxruntime")

let model = onnx.load(
  "/home/liu/workspace/Kolors/ipadapter_FaceID/models/antelopev2/glintr100.onnx")

let initializers = model.graph.initializer

var stateDict = [String: PythonObject]()

for tensor in initializers {
  stateDict[String(tensor.name)!] = tensor
}

let namespace: PythonObject = [:]

Python.exec(
  """
  import onnx

  def extract_submodel(model, output_names):
      # Create a graph proto for the new model
      graph = onnx.GraphProto()
      graph.name = f"Extracted_from_{model.graph.name}"

      # Set to keep track of added nodes and inputs
      added_nodes = set()
      added_inputs = set()

      def add_node_and_ancestors(node):
          if node.name in added_nodes:
              return
          added_nodes.add(node.name)

          # Add input nodes recursively
          for input_name in node.input:
              input_node = next((n for n in model.graph.node if input_name in n.output), None)
              if input_node:
                  add_node_and_ancestors(input_node)
              elif input_name not in added_inputs:
                  # This is a model input or an initializer
                  input_value_info = next((vi for vi in model.graph.input if vi.name == input_name), None)
                  if input_value_info:
                      graph.input.append(input_value_info)
                      added_inputs.add(input_name)
                  else:
                      # Check if it's an initializer
                      initializer = next((init for init in model.graph.initializer if init.name == input_name), None)
                      if initializer:
                          graph.initializer.append(initializer)
                      else:
                          print(f"Warning: Input {input_name} not found in model inputs or initializers")

          # Add the node itself
          graph.node.append(node)

      # Function to find the node that produces a given output
      def find_output_node(output_name):
          for node in model.graph.node:
              if output_name in node.output:
                  return node
          return None

      # Start from the output nodes and work backwards
      for output_name in output_names:
          output_node = find_output_node(output_name)
          if output_node:
              add_node_and_ancestors(output_node)
          else:
              print(f"Warning: Node producing output {output_name} not found")

          # Add the output to the new graph
          output_value_info = next((vi for vi in model.graph.output if vi.name == output_name), None)
          if not output_value_info:
              # If it's an intermediate output, create a new ValueInfo
              output_value_info = onnx.helper.make_tensor_value_info(
                  output_name,
                  onnx.TensorProto.FLOAT,  # Assume float, adjust if needed
                  None  # Shape will be inferred later
              )
          graph.output.append(output_value_info)

      # Create a new model with the extracted graph
      new_model = onnx.helper.make_model(graph, producer_name="ONNX_Extractor")

      # Copy over any metadata from the original model
      new_model.ir_version = model.ir_version
      new_model.producer_name = model.producer_name
      new_model.producer_version = model.producer_version
      new_model.domain = model.domain
      new_model.model_version = model.model_version
      new_model.doc_string = model.doc_string

      # Infer shapes for the new model
      new_model = onnx.shape_inference.infer_shapes(new_model)

      # Verify the new model
      onnx.checker.check_model(new_model)

      return new_model
  """, namespace)

let outputY = "1333"

let submodel = namespace["extract_submodel"](model, [outputY])

let session = onnxruntime.InferenceSession(submodel.SerializeToString())

let numpy = Python.import("numpy")

numpy.random.seed(42)

let torch = Python.import("torch")

let image = numpy.random.normal(size: PythonObject(tupleOf: 1, 3, 112, 112)).astype(numpy.float32)

let output = session.run([outputY], ["input.1": image])

print(output)

func PReLU(count: Int, name: String) -> (Model, ([String: PythonObject]) -> Void) {
  let x = Input()
  let weight = Parameter<Float>(.GPU(0), .CHW(count, 1, 1))
  let out = x.ReLU() - (-x).ReLU() .* weight
  let reader: ([String: PythonObject]) -> Void = { stateDict in
    let t = onnx.numpy_helper.to_array(stateDict[name])
    weight.weight.copy(
      from: try! Tensor<Float>(numpy: t))
  }
  return (Model([x], [out]), reader)
}

func BatchNorm(count: Int, name: String) -> (Model, ([String: PythonObject]) -> Void) {
  let x = Input()
  let weight = Parameter<Float>(.GPU(0), .CHW(count, 1, 1))
  let bias = Parameter<Float>(.GPU(0), .CHW(count, 1, 1))
  let out = x .* weight + bias
  let reader: ([String: PythonObject]) -> Void = { stateDict in
    let bn1_weight = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["\(name).weight"]))
    let bn1_bias = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["\(name).bias"]))
    let bn1_running_mean = torch.from_numpy(
      onnx.numpy_helper.to_array(stateDict["\(name).running_mean"]))
    let bn1_running_var = torch.from_numpy(
      onnx.numpy_helper.to_array(stateDict["\(name).running_var"]))
    let norm_weight = bn1_weight / torch.sqrt(bn1_running_var + 1e-5)
    let norm_bias =
      bn1_bias - bn1_weight * bn1_running_mean
      / torch.sqrt(bn1_running_var + 1e-5)
    weight.weight.copy(from: try! Tensor<Float>(numpy: norm_weight.numpy()))
    bias.weight.copy(from: try! Tensor<Float>(numpy: norm_bias.numpy()))
  }
  return (Model([x], [out]), reader)
}

func ResnetBlock(prefix: (Int, Int, String), inChannels: Int, outChannels: Int, downsample: Bool)
  -> (Model, ([String: PythonObject]) -> Void)
{
  var readers = [([String: PythonObject]) -> Void]()
  let x = Input()
  let bn1 = BatchNorm(count: inChannels, name: prefix.2)
  readers.append(bn1.1)
  var out = bn1.0(x)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let prelu = PReLU(count: outChannels, name: "\(prefix.1)")
  readers.append(prelu.1)
  out = prelu.0(out)
  let skip: Model?
  let conv2: Model
  if downsample {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    let convSkip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])))
    out = conv2(out) + convSkip(x)
    skip = convSkip
  } else {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
    out = conv2(out) + x
    skip = nil
  }
  let reader: ([String: PythonObject]) -> Void = { stateDict in
    for reader in readers {
      reader(stateDict)
    }
    let conv1_weight = onnx.numpy_helper.to_array(stateDict["\(prefix.0)"])
    let conv1_bias = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 1)"])
    conv1.weight.copy(
      from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.bias.copy(
      from: try! Tensor<Float>(numpy: conv1_bias))
    let conv2_weight = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 3)"])
    let conv2_bias = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 4)"])
    conv2.weight.copy(
      from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.bias.copy(
      from: try! Tensor<Float>(numpy: conv2_bias))
    if let skip = skip {
      let skip_weight = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 6)"])
      let skip_bias = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 7)"])
      skip.weight.copy(
        from: try! Tensor<Float>(numpy: skip_weight))
      skip.bias.copy(
        from: try! Tensor<Float>(numpy: skip_bias))
    }
  }
  return (Model([x], [out]), reader)
}

func ResnetLayer(prefix: (Int, Int, String), inChannels: Int, outChannels: Int, layers: Int) -> (
  Model, ([String: PythonObject]) -> Void
) {
  var readers = [([String: PythonObject]) -> Void]()
  let x = Input()
  // First block is to downsample.
  let firstBlock = ResnetBlock(
    prefix: (prefix.0, prefix.1, "\(prefix.2).0.bn1"), inChannels: inChannels,
    outChannels: outChannels, downsample: true)
  readers.append(firstBlock.1)
  var out = firstBlock.0(x)
  if layers > 1 {
    for i in 0..<(layers - 1) {
      let block = ResnetBlock(
        prefix: (prefix.0 + i * 6 + 9, prefix.1 + i + 1, "\(prefix.2).\(i + 1).bn1"),
        inChannels: outChannels, outChannels: outChannels, downsample: false)
      readers.append(block.1)
      out = block.0(out)
    }
  }
  let reader: ([String: PythonObject]) -> Void = { stateDict in
    for reader in readers {
      reader(stateDict)
    }
  }
  return (Model([x], [out]), reader)
}

func ArcFace(batchSize: Int) -> (Model, ([String: PythonObject]) -> Void) {
  let x = Input()
  let conv = Convolution(
    groups: 1, filters: 64, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv(x)
  var readers = [([String: PythonObject]) -> Void]()
  let prelu = PReLU(count: 64, name: "1643")
  readers.append(prelu.1)
  out = prelu.0(out)
  let layer1 = ResnetLayer(
    prefix: (1338, 1644, "layer1"), inChannels: 64, outChannels: 64, layers: 3)
  readers.append(layer1.1)
  out = layer1.0(out)
  let layer2 = ResnetLayer(
    prefix: (1359, 1647, "layer2"), inChannels: 64, outChannels: 128, layers: 13)
  readers.append(layer2.1)
  out = layer2.0(out)
  let layer3 = ResnetLayer(
    prefix: (1440, 1660, "layer3"), inChannels: 128, outChannels: 256, layers: 30)
  readers.append(layer3.1)
  out = layer3.0(out)
  let layer4 = ResnetLayer(
    prefix: (1623, 1690, "layer4"), inChannels: 256, outChannels: 512, layers: 3)
  readers.append(layer4.1)
  out = layer4.0(out)
  let fc = Dense(count: 512)
  out = fc(out.reshaped([batchSize, 512 * 7 * 7]))
  let reader: ([String: PythonObject]) -> Void = { stateDict in
    let t1335 = onnx.numpy_helper.to_array(stateDict["1335"])
    let t1336 = onnx.numpy_helper.to_array(stateDict["1336"])
    conv.weight.copy(
      from: try! Tensor<Float>(numpy: t1335))
    conv.bias.copy(
      from: try! Tensor<Float>(numpy: t1336))
    for reader in readers {
      reader(stateDict)
    }
    let bn2_weight = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["bn2.weight"]))
    let bn2_bias = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["bn2.bias"]))
    let bn2_running_mean = torch.from_numpy(
      onnx.numpy_helper.to_array(stateDict["bn2.running_mean"]))
    let bn2_running_var = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["bn2.running_var"]))
    let norm1_weight = bn2_weight / torch.sqrt(bn2_running_var + 1e-5)
    let norm1_bias =
      bn2_bias - bn2_weight * bn2_running_mean
      / torch.sqrt(bn2_running_var + 1e-5)
    let fc_weight = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["fc.weight"]))
    let fc_bias = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["fc.bias"]))
    let features_weight = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["features.weight"]))
    let features_bias = torch.from_numpy(onnx.numpy_helper.to_array(stateDict["features.bias"]))
    let features_running_mean = torch.from_numpy(
      onnx.numpy_helper.to_array(stateDict["features.running_mean"]))
    let features_running_var = torch.from_numpy(
      onnx.numpy_helper.to_array(stateDict["features.running_var"]))
    let norm2_weight = features_weight / torch.sqrt(features_running_var + 1e-5)
    let norm2_bias =
      features_bias - features_weight * features_running_mean
      / torch.sqrt(features_running_var + 1e-5)
    let fused_weight =
      fc_weight * norm1_weight.view(-1, 1).repeat(1, 7 * 7).view(1, -1) * norm2_weight.view(-1, 1)
    let fused_bias =
      norm2_weight
      * (torch.matmul(fc_weight, norm1_bias.view(-1, 1).repeat(1, 7 * 7).view(-1)) + fc_bias)
      + norm2_bias
    fc.weight.copy(
      from: try! Tensor<Float>(numpy: fused_weight.numpy()))
    fc.bias.copy(
      from: try! Tensor<Float>(numpy: fused_bias.numpy()))
  }
  return (Model([x], [out]), reader)
}

let graph = DynamicGraph()
graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: image)).toGPU(0)
  // Already processed out.
  let (arc, arcReader) = ArcFace(batchSize: 1)
  arc.compile(inputs: xTensor)
  arcReader(stateDict)
  let embedding = arc(inputs: xTensor)[0].as(of: Float.self)
  debugPrint(embedding)
}
