import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let onnx = Python.import("onnx")

let onnx_tools_update_model_dims = Python.import("onnx.tools.update_model_dims")

let onnxruntime = Python.import("onnxruntime")

let cv2 = Python.import("cv2")

var model = onnx.load(
  "/home/liu/workspace/Kolors/ipadapter_FaceID/models/antelopev2/scrfd_10g_bnkps.onnx")

let initializers = model.graph.initializer

var stateDict = [String: PythonObject]()

for tensor in initializers {
  stateDict[String(tensor.name)!] = tensor
}

let namespace: PythonObject = [:]

Python.exec(
  """
  import onnx
  import numpy as np

  def update_unsqueeze_nodes(model):
    # Update Unsqueeze nodes to be compatible with newer ONNX versions.
    for node in model.graph.node:
        if node.op_type == 'Unsqueeze':
            # In older versions, axes was an attribute. In newer versions, it's an input.
            if len(node.attribute) > 0 and any(attr.name == 'axes' for attr in node.attribute):
                axes_attr = next(attr for attr in node.attribute if attr.name == 'axes')
                axes = axes_attr.ints

                # Create a new initializer for the axes
                axes_name = node.name + "_axes"
                axes_tensor = onnx.numpy_helper.from_array(np.array(axes, dtype=np.int64), name=axes_name)
                model.graph.initializer.append(axes_tensor)

                # Update the node
                node.input.append(axes_name)
                node.attribute.remove(axes_attr)

    return model

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
      # Update Unsqueeze nodes
      new_model = update_unsqueeze_nodes(new_model)

      # Infer shapes for the new model
      new_model = onnx.shape_inference.infer_shapes(new_model)

      # Verify the new model
      onnx.checker.check_model(new_model)

      return new_model
  """, namespace)

let outputY = ["494", "497", "500", "471", "474", "477", "448", "451", "454"]

let submodel = model  // namespace["extract_submodel"](model, outputY)

let session = onnxruntime.InferenceSession(submodel.SerializeToString())

let numpy = Python.import("numpy")

let diffusers = Python.import("diffusers")

let image = diffusers.utils.load_image(
  "/home/liu/workspace/Kolors/ipadapter_FaceID/assets/image1.png")
let img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

numpy.random.seed(42)

let blob = cv2.dnn.blobFromImage(
  img, 1.0 / 128, PythonObject(tupleOf: 640, 640), PythonObject(tupleOf: 127.5, 127.5, 127.5),
  swapRB: true)
print(blob.shape)

let output = session.run(outputY, ["input.1": blob])

print(output)

func ResnetBlock(prefix: Int, channels: Int, downsample: Bool) -> (
  Model, ([String: PythonObject]) -> Void
) {
  let x = Input()
  let conv1: Model
  if downsample {
    conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [3, 3],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  } else {
    conv1 = Convolution(
      groups: 1, filters: channels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  }
  var out = conv1(x).ReLU()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let skip: Model?
  if downsample {
    let convSkip = Convolution(
      groups: 1, filters: channels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
    out =
      conv2(out)
      + convSkip(
        AveragePool(
          filterSize: [2, 2],
          hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])))(x))
    skip = convSkip
  } else {
    out = conv2(out) + x
    skip = nil
  }
  let reader: ([String: PythonObject]) -> Void = { stateDict in
    let t559 = onnx.numpy_helper.to_array(stateDict["\(prefix)"])
    let t561 = onnx.numpy_helper.to_array(stateDict["\(prefix + 2)"])
    conv1.weight.copy(
      from: try! Tensor<Float>(numpy: t559))
    conv1.bias.copy(
      from: try! Tensor<Float>(numpy: t561))
    let t563 = onnx.numpy_helper.to_array(stateDict["\(prefix + 4)"])
    let t565 = onnx.numpy_helper.to_array(stateDict["\(prefix + 6)"])
    conv2.weight.copy(
      from: try! Tensor<Float>(numpy: t563))
    conv2.bias.copy(
      from: try! Tensor<Float>(numpy: t565))
    if let skip = skip {
      let t591 = onnx.numpy_helper.to_array(stateDict["\(prefix + 8)"])
      let t593 = onnx.numpy_helper.to_array(stateDict["\(prefix + 10)"])
      skip.weight.copy(
        from: try! Tensor<Float>(numpy: t591))
      skip.bias.copy(
        from: try! Tensor<Float>(numpy: t593))
    }
  }
  return (Model([x], [out]), reader)
}

func BoxHead(prefix: (Int, String), multiplier: Float) -> (Model, ([String: PythonObject]) -> Void)
{
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 80, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv1(x).ReLU()
  let conv2 = Convolution(
    groups: 1, filters: 80, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out).ReLU()
  let conv3 = Convolution(
    groups: 1, filters: 80, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv3(out).ReLU()
  let cls = Convolution(
    groups: 1, filters: 2, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var outs = [Model.IO]()
  outs.append(cls(out).sigmoid())
  let reg = Convolution(
    groups: 1, filters: 8, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  outs.append(reg(out) * multiplier)
  let kps = Convolution(
    groups: 1, filters: 20, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  outs.append(kps(out))
  let reader: ([String: PythonObject]) -> Void = { stateDict in
    let t667 = onnx.numpy_helper.to_array(stateDict["\(prefix.0)"])
    let t669 = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 2)"])
    conv1.weight.copy(
      from: try! Tensor<Float>(numpy: t667))
    conv1.bias.copy(
      from: try! Tensor<Float>(numpy: t669))
    let t671 = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 4)"])
    let t673 = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 6)"])
    conv2.weight.copy(
      from: try! Tensor<Float>(numpy: t671))
    conv2.bias.copy(
      from: try! Tensor<Float>(numpy: t673))
    let t675 = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 8)"])
    let t677 = onnx.numpy_helper.to_array(stateDict["\(prefix.0 + 10)"])
    conv3.weight.copy(
      from: try! Tensor<Float>(numpy: t675))
    conv3.bias.copy(
      from: try! Tensor<Float>(numpy: t677))
    let bbox_head_stride_cls_weight = onnx.numpy_helper.to_array(
      stateDict["bbox_head.stride_cls.\(prefix.1).weight"])
    let bbox_head_stride_cls_bias = onnx.numpy_helper.to_array(
      stateDict["bbox_head.stride_cls.\(prefix.1).bias"])
    cls.weight.copy(
      from: try! Tensor<Float>(numpy: bbox_head_stride_cls_weight))
    cls.bias.copy(
      from: try! Tensor<Float>(numpy: bbox_head_stride_cls_bias))
    let bbox_head_stride_reg_weight = onnx.numpy_helper.to_array(
      stateDict["bbox_head.stride_reg.\(prefix.1).weight"])
    let bbox_head_stride_reg_bias = onnx.numpy_helper.to_array(
      stateDict["bbox_head.stride_reg.\(prefix.1).bias"])
    reg.weight.copy(
      from: try! Tensor<Float>(numpy: bbox_head_stride_reg_weight))
    reg.bias.copy(
      from: try! Tensor<Float>(numpy: bbox_head_stride_reg_bias))
    let bbox_head_stride_kps_weight = onnx.numpy_helper.to_array(
      stateDict["bbox_head.stride_kps.\(prefix.1).weight"])
    let bbox_head_stride_kps_bias = onnx.numpy_helper.to_array(
      stateDict["bbox_head.stride_kps.\(prefix.1).bias"])
    kps.weight.copy(
      from: try! Tensor<Float>(numpy: bbox_head_stride_kps_weight))
    kps.bias.copy(
      from: try! Tensor<Float>(numpy: bbox_head_stride_kps_bias))
  }
  return (Model([x], outs), reader)
}

func RetinaFace(batchSize: Int) -> (Model, ([String: PythonObject]) -> Void) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 28, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv1(x).ReLU()
  let conv2 = Convolution(
    groups: 1, filters: 28, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out).ReLU()
  let conv3 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv3(out).ReLU()
  out = MaxPool(
    filterSize: [2, 2],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])))(out)
  var readers = [([String: PythonObject]) -> Void]()
  let resnetBlock_1_1 = ResnetBlock(prefix: 559, channels: 56, downsample: false)
  readers.append(resnetBlock_1_1.1)
  out = resnetBlock_1_1.0(out).ReLU()
  let resnetBlock_1_2 = ResnetBlock(prefix: 567, channels: 56, downsample: false)
  readers.append(resnetBlock_1_2.1)
  out = resnetBlock_1_2.0(out).ReLU()
  let resnetBlock_1_3 = ResnetBlock(prefix: 575, channels: 56, downsample: false)
  readers.append(resnetBlock_1_3.1)
  out = resnetBlock_1_3.0(out).ReLU()
  let resnetBlock_2_1 = ResnetBlock(prefix: 583, channels: 88, downsample: true)
  readers.append(resnetBlock_2_1.1)
  out = resnetBlock_2_1.0(out).ReLU()
  let resnetBlock_2_2 = ResnetBlock(prefix: 595, channels: 88, downsample: false)
  readers.append(resnetBlock_2_2.1)
  out = resnetBlock_2_2.0(out).ReLU()
  let resnetBlock_2_3 = ResnetBlock(prefix: 603, channels: 88, downsample: false)
  readers.append(resnetBlock_2_3.1)
  out = resnetBlock_2_3.0(out).ReLU()
  let resnetBlock_2_4 = ResnetBlock(prefix: 611, channels: 88, downsample: false)
  readers.append(resnetBlock_2_4.1)
  out = resnetBlock_2_4.0(out).ReLU()
  var layer2Out = out
  let resnetBlock_3_1 = ResnetBlock(prefix: 619, channels: 88, downsample: true)
  readers.append(resnetBlock_3_1.1)
  out = resnetBlock_3_1.0(out).ReLU()
  let resnetBlock_3_2 = ResnetBlock(prefix: 631, channels: 88, downsample: false)
  readers.append(resnetBlock_3_2.1)
  out = resnetBlock_3_2.0(out).ReLU()
  var layer3Out = out
  let resnetBlock_4_1 = ResnetBlock(prefix: 639, channels: 224, downsample: true)
  readers.append(resnetBlock_4_1.1)
  out = resnetBlock_4_1.0(out).ReLU()
  let resnetBlock_4_2 = ResnetBlock(prefix: 651, channels: 224, downsample: false)
  readers.append(resnetBlock_4_2.1)
  out = resnetBlock_4_2.0(out).ReLU()
  let resnetBlock_4_3 = ResnetBlock(prefix: 659, channels: 224, downsample: false)
  readers.append(resnetBlock_4_3.1)
  out = resnetBlock_4_3.0(out).ReLU()
  let conv4 = Convolution(
    groups: 1, filters: 56, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
  layer2Out = conv4(layer2Out)
  let conv5 = Convolution(
    groups: 1, filters: 56, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
  layer3Out = conv5(layer3Out)
  let conv6 = Convolution(
    groups: 1, filters: 56, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW)
  out = conv6(out)
  layer3Out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out) + layer3Out
  layer2Out = Upsample(.nearest, widthScale: 2, heightScale: 2)(layer3Out) + layer2Out

  let conv7 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  layer2Out = conv7(layer2Out)

  let conv8 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let conv9 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  layer3Out = conv8(layer3Out) + conv9(layer2Out)

  let conv10 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  let conv11 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv10(out) + conv11(layer3Out)

  let conv12 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  layer3Out = conv12(layer3Out)

  let conv13 = Convolution(
    groups: 1, filters: 56, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv13(out)

  let boxHead8 = BoxHead(prefix: (667, "(8, 8)"), multiplier: 0.8463594317436218)
  readers.append(boxHead8.1)
  layer2Out = boxHead8.0(layer2Out)

  let boxHead16 = BoxHead(prefix: (679, "(16, 16)"), multiplier: 0.8996264338493347)
  readers.append(boxHead16.1)
  layer3Out = boxHead16.0(layer3Out)

  let boxHead32 = BoxHead(prefix: (691, "(32, 32)"), multiplier: 1.0812087059020996)
  readers.append(boxHead32.1)
  out = boxHead32.0(out)

  let reader: ([String: PythonObject]) -> Void = { stateDict in
    let t547 = onnx.numpy_helper.to_array(stateDict["547"])
    let t549 = onnx.numpy_helper.to_array(stateDict["549"])
    conv1.weight.copy(
      from: try! Tensor<Float>(numpy: t547))
    conv1.bias.copy(
      from: try! Tensor<Float>(numpy: t549))
    let t551 = onnx.numpy_helper.to_array(stateDict["551"])
    let t553 = onnx.numpy_helper.to_array(stateDict["553"])
    conv2.weight.copy(
      from: try! Tensor<Float>(numpy: t551))
    conv2.bias.copy(
      from: try! Tensor<Float>(numpy: t553))
    let t555 = onnx.numpy_helper.to_array(stateDict["555"])
    let t557 = onnx.numpy_helper.to_array(stateDict["557"])
    conv3.weight.copy(
      from: try! Tensor<Float>(numpy: t555))
    conv3.bias.copy(
      from: try! Tensor<Float>(numpy: t557))
    for reader in readers {
      reader(stateDict)
    }
    let neck_lateral_convs_0_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.lateral_convs.0.conv.weight"])
    let neck_lateral_convs_0_conv_bias = onnx.numpy_helper.to_array(
      stateDict["neck.lateral_convs.0.conv.bias"])
    conv4.weight.copy(
      from: try! Tensor<Float>(numpy: neck_lateral_convs_0_conv_weight))
    conv4.bias.copy(
      from: try! Tensor<Float>(numpy: neck_lateral_convs_0_conv_bias))
    let neck_lateral_convs_1_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.lateral_convs.1.conv.weight"])
    let neck_lateral_convs_1_conv_bias = onnx.numpy_helper.to_array(
      stateDict["neck.lateral_convs.1.conv.bias"])
    conv5.weight.copy(
      from: try! Tensor<Float>(numpy: neck_lateral_convs_1_conv_weight))
    conv5.bias.copy(
      from: try! Tensor<Float>(numpy: neck_lateral_convs_1_conv_bias))
    let neck_lateral_convs_2_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.lateral_convs.2.conv.weight"])
    let neck_lateral_convs_2_conv_bias = onnx.numpy_helper.to_array(
      stateDict["neck.lateral_convs.2.conv.bias"])
    conv6.weight.copy(
      from: try! Tensor<Float>(numpy: neck_lateral_convs_2_conv_weight))
    conv6.bias.copy(
      from: try! Tensor<Float>(numpy: neck_lateral_convs_2_conv_bias))

    let neck_fpn_convs_0_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.fpn_convs.0.conv.weight"])
    let neck_fpn_convs_0_conv_bias = onnx.numpy_helper.to_array(
      stateDict["neck.fpn_convs.0.conv.bias"])
    conv7.weight.copy(
      from: try! Tensor<Float>(numpy: neck_fpn_convs_0_conv_weight))
    conv7.bias.copy(
      from: try! Tensor<Float>(numpy: neck_fpn_convs_0_conv_bias))
    let neck_downsample_convs_0_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.downsample_convs.0.conv.weight"])
    let neck_downsample_convs_0_conv_bias = onnx.numpy_helper.to_array(
      stateDict["neck.downsample_convs.0.conv.bias"])
    conv9.weight.copy(
      from: try! Tensor<Float>(numpy: neck_downsample_convs_0_conv_weight))
    conv9.bias.copy(
      from: try! Tensor<Float>(numpy: neck_downsample_convs_0_conv_bias))
    let neck_fpn_convs_1_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.fpn_convs.1.conv.weight"])
    conv8.weight.copy(
      from: try! Tensor<Float>(numpy: neck_fpn_convs_1_conv_weight))
    conv8.bias.copy(
      from: try! Tensor<Float>(numpy: neck_downsample_convs_0_conv_bias))

    let neck_downsample_convs_1_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.downsample_convs.1.conv.weight"])
    let neck_downsample_convs_1_conv_bias = onnx.numpy_helper.to_array(
      stateDict["neck.downsample_convs.1.conv.bias"])
    conv11.weight.copy(
      from: try! Tensor<Float>(numpy: neck_downsample_convs_1_conv_weight))
    conv11.bias.copy(
      from: try! Tensor<Float>(numpy: neck_downsample_convs_1_conv_bias))
    let neck_fpn_convs_2_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.fpn_convs.2.conv.weight"])
    conv10.weight.copy(
      from: try! Tensor<Float>(numpy: neck_fpn_convs_2_conv_weight))
    conv10.bias.copy(
      from: try! Tensor<Float>(numpy: neck_downsample_convs_1_conv_bias))

    let neck_pafpn_convs_0_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.pafpn_convs.0.conv.weight"])
    let neck_pafpn_convs_0_conv_bias = onnx.numpy_helper.to_array(
      stateDict["neck.pafpn_convs.0.conv.bias"])
    conv12.weight.copy(
      from: try! Tensor<Float>(numpy: neck_pafpn_convs_0_conv_weight))
    conv12.bias.copy(
      from: try! Tensor<Float>(numpy: neck_pafpn_convs_0_conv_bias))

    let neck_pafpn_convs_1_conv_weight = onnx.numpy_helper.to_array(
      stateDict["neck.pafpn_convs.1.conv.weight"])
    let neck_pafpn_convs_1_conv_bias = onnx.numpy_helper.to_array(
      stateDict["neck.pafpn_convs.1.conv.bias"])
    conv13.weight.copy(
      from: try! Tensor<Float>(numpy: neck_pafpn_convs_1_conv_weight))
    conv13.bias.copy(
      from: try! Tensor<Float>(numpy: neck_pafpn_convs_1_conv_bias))
  }
  return (Model([x], [out, layer3Out, layer2Out]), reader)
}

let graph = DynamicGraph()
graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: blob)).toGPU(0)
  // Already processed out.
  let (retinaFace, retinaFaceReader) = RetinaFace(batchSize: 1)
  retinaFace.compile(inputs: xTensor)
  retinaFaceReader(stateDict)
  var embeddings = retinaFace(inputs: xTensor).map { $0.as(of: Float.self) }
  embeddings = embeddings.map { $0.permuted(0, 2, 3, 1).copied() }
  embeddings[0] = embeddings[0].reshaped(.WC(20 * 20 * 2, 1))
  embeddings[1] = embeddings[1].reshaped(.WC(20 * 20 * 2, 4))
  embeddings[2] = embeddings[2].reshaped(.WC(20 * 20 * 2, 10))
  embeddings[3] = embeddings[3].reshaped(.WC(40 * 40 * 2, 1))
  embeddings[4] = embeddings[4].reshaped(.WC(40 * 40 * 2, 4))
  embeddings[5] = embeddings[5].reshaped(.WC(40 * 40 * 2, 10))
  embeddings[6] = embeddings[6].reshaped(.WC(80 * 80 * 2, 1))
  embeddings[7] = embeddings[7].reshaped(.WC(80 * 80 * 2, 4))
  embeddings[8] = embeddings[8].reshaped(.WC(80 * 80 * 2, 10))
  graph.openStore("/home/liu/workspace/swift-diffusion/arcface_f32.ckpt") {
    $0.write("retinaface", model: retinaFace)
  }
  debugPrint(embeddings)
}
