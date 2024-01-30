import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let torch = Python.import("torch")
let numpy = Python.import("numpy")
let dpt = Python.import("depth_anything.dpt")

let depth_anything = dpt.DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").to("cuda")
  .eval()

let transforms = Python.import("torchvision.transforms")
let depth_anything_util_transform = Python.import("depth_anything.util.transform")
let cv2 = Python.import("cv2")

torch.set_grad_enabled(false)
let transform = transforms.Compose([
  depth_anything_util_transform.Resize(
    width: 518,
    height: 518,
    resize_target: false,
    keep_aspect_ratio: true,
    ensure_multiple_of: 14,
    resize_method: "lower_bound",
    image_interpolation_method: cv2.INTER_CUBIC
  ),
  depth_anything_util_transform.NormalizeImage(
    mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]),
  depth_anything_util_transform.PrepareForNet(),
])

let raw_image = cv2.imread("/home/liu/workspace/swift-diffusion/kandinsky.png")
var image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

image = transform(["image": image])["image"]
image = torch.from_numpy(image).unsqueeze(0).to("cuda")

/*
var depth = depth_anything(image)

print(depth_anything.pretrained)
*/

func DinoSelfAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, tokeys, tovalues, unifyheads, Model([x], [out]))
}

func DinoResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2])
  let (toqueries, tokeys, tovalues, unifyheads, attention) = DinoSelfAttention(
    k: k, h: h, b: b, t: t)
  let ls1 = Parameter<Float>(.GPU(0), .NC(1, h * k), initBound: 1)
  var out = x.reshaped([b * t, h * k]) + (attention(ln1(x)) .* ls1)
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  let ls2 = Parameter<Float>(.GPU(0), .NC(1, h * k), initBound: 1)
  out = out + (proj(gelu(fc(ln2(out)))) .* ls2)
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu().numpy()
    ln1.weight.copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    ln1.bias.copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let in_proj_weight = state_dict["\(prefix).attn.qkv.weight"].type(torch.float).cpu().numpy()
    let in_proj_bias = state_dict["\(prefix).attn.qkv.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[..<(k * h), ...]))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: in_proj_bias[..<(k * h)]))
    tokeys.weight.copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(k * h)..<(2 * k * h), ...]))
    tokeys.bias.copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(k * h)..<(2 * k * h)]))
    tovalues.weight.copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(2 * k * h)..., ...]))
    tovalues.bias.copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(2 * k * h)...]))
    let out_proj_weight = state_dict["\(prefix).attn.proj.weight"].type(torch.float).cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).attn.proj.bias"].type(torch.float).cpu().numpy()
    let ls_1_gamma = state_dict["\(prefix).ls1.gamma"].type(torch.float).cpu().numpy()
    ls1.weight.copy(from: try! Tensor<Float>(numpy: ls_1_gamma))
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: out_proj_bias))
    let ln_2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu().numpy()
    ln2.weight.copy(from: try! Tensor<Float>(numpy: ln_2_weight))
    ln2.bias.copy(from: try! Tensor<Float>(numpy: ln_2_bias))
    let c_fc_weight = state_dict["\(prefix).mlp.fc1.weight"].type(torch.float).cpu().numpy()
    let c_fc_bias = state_dict["\(prefix).mlp.fc1.bias"].type(torch.float).cpu().numpy()
    fc.weight.copy(from: try! Tensor<Float>(numpy: c_fc_weight))
    fc.bias.copy(from: try! Tensor<Float>(numpy: c_fc_bias))
    let c_proj_weight = state_dict["\(prefix).mlp.fc2.weight"].type(torch.float).cpu().numpy()
    let c_proj_bias = state_dict["\(prefix).mlp.fc2.bias"].type(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: c_proj_weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: c_proj_bias))
    let ls_2_gamma = state_dict["\(prefix).ls2.gamma"].type(torch.float).cpu().numpy()
    ls2.weight.copy(from: try! Tensor<Float>(numpy: ls_2_gamma))
  }
  return (reader, Model([x], [out]))
}

func DinoVisionTransformer(
  grid: Int, width: Int, layers: Int, heads: Int, batchSize: Int, intermediateLayers: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let classEmbedding = Input()
  let positionalEmbedding = Input()
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  for i in 0..<layers {
    let (reader, block) = DinoResidualAttentionBlock(
      prefix: "blocks.\(i)", k: width / heads, h: heads, b: batchSize,
      t: grid * grid + 1)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
    if i >= layers - intermediateLayers {
      outs.append(out)
    }
    readers.append(reader)
  }
  let lnPost = LayerNorm(epsilon: 1e-6, axis: [1])
  outs = outs.map { lnPost($0) }
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1_weight = state_dict["patch_embed.proj.weight"].type(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let conv1_bias = state_dict["patch_embed.proj.bias"].type(torch.float).cpu().numpy()
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    for reader in readers {
      reader(state_dict)
    }
    let ln_post_weight = state_dict["norm.weight"].type(torch.float).cpu().numpy()
    let ln_post_bias = state_dict["norm.bias"].type(torch.float).cpu().numpy()
    lnPost.weight.copy(from: try! Tensor<Float>(numpy: ln_post_weight))
    lnPost.bias.copy(from: try! Tensor<Float>(numpy: ln_post_bias))
  }
  return (reader, Model([x, classEmbedding, positionalEmbedding], outs))
}

func ResidualConvUnit(prefix: String) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv1(x.ReLU())
  let conv2 = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = x + conv2(out.ReLU())
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1_weight = state_dict["\(prefix).conv1.weight"].type(torch.float).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let conv1_bias = state_dict["\(prefix).conv1.bias"].type(torch.float).cpu().numpy()
    conv1.bias.copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let conv2_weight = state_dict["\(prefix).conv2.weight"].type(torch.float).cpu().numpy()
    conv2.weight.copy(from: try! Tensor<Float>(numpy: conv2_weight))
    let conv2_bias = state_dict["\(prefix).conv2.bias"].type(torch.float).cpu().numpy()
    conv2.bias.copy(from: try! Tensor<Float>(numpy: conv2_bias))
  }
  return (reader, Model([x], [out]))
}

func DepthHead() -> ((PythonObject) -> Void, Model) {
  let x0 = Input()
  let proj0 = Convolution(groups: 1, filters: 256, filterSize: [1, 1])
  let conv0 = ConvolutionTranspose(
    groups: 1, filters: 256, filterSize: [4, 4], hint: Hint(stride: [4, 4]))
  var out0 = conv0(proj0(x0))
  let x1 = Input()
  let proj1 = Convolution(groups: 1, filters: 512, filterSize: [1, 1])
  let conv1 = ConvolutionTranspose(
    groups: 1, filters: 512, filterSize: [2, 2], hint: Hint(stride: [2, 2]))
  var out1 = conv1(proj1(x1))
  let x2 = Input()
  let proj2 = Convolution(groups: 1, filters: 1024, filterSize: [1, 1])
  var out2 = proj2(x2)
  let x3 = Input()
  let proj3 = Convolution(groups: 1, filters: 1024, filterSize: [1, 1])
  let conv3 = Convolution(
    groups: 1, filters: 1024, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out3 = conv3(proj3(x3))

  let layer1_rn = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out0 = layer1_rn(out0)
  let layer2_rn = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out1 = layer2_rn(out1)
  let layer3_rn = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out2 = layer3_rn(out2)
  let layer4_rn = Convolution(
    groups: 1, filters: 256, filterSize: [3, 3], noBias: true,
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out3 = layer4_rn(out3)

  let (refinenet4Reader, refinenet4) = ResidualConvUnit(prefix: "scratch.refinenet4.resConfUnit2")
  out3 = Upsample(.bilinear, widthScale: 37.0 / 19.0, heightScale: 37.0 / 19.0)(refinenet4(out3))
  let refinenet4OutConv = Convolution(groups: 1, filters: 256, filterSize: [1, 1])
  out3 = refinenet4OutConv(out3)
  let (refinenet3Unit1Reader, refinenet3Unit1) = ResidualConvUnit(
    prefix: "scratch.refinenet3.resConfUnit1")
  out2 = out3 + refinenet3Unit1(out2)
  let (refinenet3Unit2Reader, refinenet3Unit2) = ResidualConvUnit(
    prefix: "scratch.refinenet3.resConfUnit2")
  out2 = Upsample(.bilinear, widthScale: 2, heightScale: 2)(refinenet3Unit2(out2))
  let refinenet3OutConv = Convolution(groups: 1, filters: 256, filterSize: [1, 1])
  out2 = refinenet3OutConv(out2)
  let (refinenet2Unit1Reader, refinenet2Unit1) = ResidualConvUnit(
    prefix: "scratch.refinenet2.resConfUnit1")
  out1 = out2 + refinenet2Unit1(out1)
  let (refinenet2Unit2Reader, refinenet2Unit2) = ResidualConvUnit(
    prefix: "scratch.refinenet2.resConfUnit2")
  out1 = Upsample(.bilinear, widthScale: 2, heightScale: 2)(refinenet2Unit2(out1))
  let refinenet2OutConv = Convolution(groups: 1, filters: 256, filterSize: [1, 1])
  out1 = refinenet2OutConv(out1)
  let (refinenet1Unit1Reader, refinenet1Unit1) = ResidualConvUnit(
    prefix: "scratch.refinenet1.resConfUnit1")
  out0 = out1 + refinenet1Unit1(out0)
  let (refinenet1Unit2Reader, refinenet1Unit2) = ResidualConvUnit(
    prefix: "scratch.refinenet1.resConfUnit2")
  out0 = Upsample(.bilinear, widthScale: 2, heightScale: 2)(refinenet1Unit2(out0))
  let refinenet1OutConv = Convolution(groups: 1, filters: 256, filterSize: [1, 1])
  out0 = refinenet1OutConv(out0)

  let outputConv1 = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out0 = Upsample(.bilinear, widthScale: 518.0 / 296.0, heightScale: 518.0 / 296.0)(
    outputConv1(out0))

  let outputConv20 = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out0 = outputConv20(out0).ReLU()
  let outputConv22 = Convolution(
    groups: 1, filters: 1, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out0 = outputConv22(out0).ReLU()

  let reader: (PythonObject) -> Void = { state_dict in
    let projects_0_weight = state_dict["projects.0.weight"].type(torch.float).cpu().numpy()
    proj0.weight.copy(from: try! Tensor<Float>(numpy: projects_0_weight))
    let projects_0_bias = state_dict["projects.0.bias"].type(torch.float).cpu().numpy()
    proj0.bias.copy(from: try! Tensor<Float>(numpy: projects_0_bias))
    let projects_1_weight = state_dict["projects.1.weight"].type(torch.float).cpu().numpy()
    proj1.weight.copy(from: try! Tensor<Float>(numpy: projects_1_weight))
    let projects_1_bias = state_dict["projects.1.bias"].type(torch.float).cpu().numpy()
    proj1.bias.copy(from: try! Tensor<Float>(numpy: projects_1_bias))
    let projects_2_weight = state_dict["projects.2.weight"].type(torch.float).cpu().numpy()
    proj2.weight.copy(from: try! Tensor<Float>(numpy: projects_2_weight))
    let projects_2_bias = state_dict["projects.2.bias"].type(torch.float).cpu().numpy()
    proj2.bias.copy(from: try! Tensor<Float>(numpy: projects_2_bias))
    let projects_3_weight = state_dict["projects.3.weight"].type(torch.float).cpu().numpy()
    proj3.weight.copy(from: try! Tensor<Float>(numpy: projects_3_weight))
    let projects_3_bias = state_dict["projects.3.bias"].type(torch.float).cpu().numpy()
    proj3.bias.copy(from: try! Tensor<Float>(numpy: projects_3_bias))
    let resize_layers_0_weight = state_dict["resize_layers.0.weight"].type(torch.float).cpu()
      .numpy()
    conv0.weight.copy(from: try! Tensor<Float>(numpy: resize_layers_0_weight))
    let resize_layers_0_bias = state_dict["resize_layers.0.bias"].type(torch.float).cpu().numpy()
    conv0.bias.copy(from: try! Tensor<Float>(numpy: resize_layers_0_bias))
    let resize_layers_1_weight = state_dict["resize_layers.1.weight"].type(torch.float).cpu()
      .numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: resize_layers_1_weight))
    let resize_layers_1_bias = state_dict["resize_layers.1.bias"].type(torch.float).cpu().numpy()
    conv1.bias.copy(from: try! Tensor<Float>(numpy: resize_layers_1_bias))
    let resize_layers_3_weight = state_dict["resize_layers.3.weight"].type(torch.float).cpu()
      .numpy()
    conv3.weight.copy(from: try! Tensor<Float>(numpy: resize_layers_3_weight))
    let resize_layers_3_bias = state_dict["resize_layers.3.bias"].type(torch.float).cpu().numpy()
    conv3.bias.copy(from: try! Tensor<Float>(numpy: resize_layers_3_bias))
    let scratch_layer1_rn_weight = state_dict["scratch.layer1_rn.weight"].type(torch.float).cpu()
      .numpy()
    layer1_rn.weight.copy(from: try! Tensor<Float>(numpy: scratch_layer1_rn_weight))
    let scratch_layer2_rn_weight = state_dict["scratch.layer2_rn.weight"].type(torch.float).cpu()
      .numpy()
    layer2_rn.weight.copy(from: try! Tensor<Float>(numpy: scratch_layer2_rn_weight))
    let scratch_layer3_rn_weight = state_dict["scratch.layer3_rn.weight"].type(torch.float).cpu()
      .numpy()
    layer3_rn.weight.copy(from: try! Tensor<Float>(numpy: scratch_layer3_rn_weight))
    let scratch_layer4_rn_weight = state_dict["scratch.layer4_rn.weight"].type(torch.float).cpu()
      .numpy()
    layer4_rn.weight.copy(from: try! Tensor<Float>(numpy: scratch_layer4_rn_weight))

    refinenet4Reader(state_dict)
    let refinenet4_out_conv_weight = state_dict["scratch.refinenet4.out_conv.weight"].type(
      torch.float
    ).cpu().numpy()
    refinenet4OutConv.weight.copy(from: try! Tensor<Float>(numpy: refinenet4_out_conv_weight))
    let refinenet4_out_conv_bias = state_dict["scratch.refinenet4.out_conv.bias"].type(torch.float)
      .cpu().numpy()
    refinenet4OutConv.bias.copy(from: try! Tensor<Float>(numpy: refinenet4_out_conv_bias))

    refinenet3Unit1Reader(state_dict)
    refinenet3Unit2Reader(state_dict)
    let refinenet3_out_conv_weight = state_dict["scratch.refinenet3.out_conv.weight"].type(
      torch.float
    ).cpu().numpy()
    refinenet3OutConv.weight.copy(from: try! Tensor<Float>(numpy: refinenet3_out_conv_weight))
    let refinenet3_out_conv_bias = state_dict["scratch.refinenet3.out_conv.bias"].type(torch.float)
      .cpu().numpy()
    refinenet3OutConv.bias.copy(from: try! Tensor<Float>(numpy: refinenet3_out_conv_bias))

    refinenet2Unit1Reader(state_dict)
    refinenet2Unit2Reader(state_dict)
    let refinenet2_out_conv_weight = state_dict["scratch.refinenet2.out_conv.weight"].type(
      torch.float
    ).cpu().numpy()
    refinenet2OutConv.weight.copy(from: try! Tensor<Float>(numpy: refinenet2_out_conv_weight))
    let refinenet2_out_conv_bias = state_dict["scratch.refinenet2.out_conv.bias"].type(torch.float)
      .cpu().numpy()
    refinenet2OutConv.bias.copy(from: try! Tensor<Float>(numpy: refinenet2_out_conv_bias))

    refinenet1Unit1Reader(state_dict)
    refinenet1Unit2Reader(state_dict)
    let refinenet1_out_conv_weight = state_dict["scratch.refinenet1.out_conv.weight"].type(
      torch.float
    ).cpu().numpy()
    refinenet1OutConv.weight.copy(from: try! Tensor<Float>(numpy: refinenet1_out_conv_weight))
    let refinenet1_out_conv_bias = state_dict["scratch.refinenet1.out_conv.bias"].type(torch.float)
      .cpu().numpy()
    refinenet1OutConv.bias.copy(from: try! Tensor<Float>(numpy: refinenet1_out_conv_bias))
    let output_conv1_weight = state_dict["scratch.output_conv1.weight"].type(torch.float).cpu()
      .numpy()
    outputConv1.weight.copy(from: try! Tensor<Float>(numpy: output_conv1_weight))
    let output_conv1_bias = state_dict["scratch.output_conv1.bias"].type(torch.float).cpu().numpy()
    outputConv1.bias.copy(from: try! Tensor<Float>(numpy: output_conv1_bias))

    let output_conv2_0_weight = state_dict["scratch.output_conv2.0.weight"].type(torch.float).cpu()
      .numpy()
    outputConv20.weight.copy(from: try! Tensor<Float>(numpy: output_conv2_0_weight))
    let output_conv2_0_bias = state_dict["scratch.output_conv2.0.bias"].type(torch.float).cpu()
      .numpy()
    outputConv20.bias.copy(from: try! Tensor<Float>(numpy: output_conv2_0_bias))
    let output_conv2_2_weight = state_dict["scratch.output_conv2.2.weight"].type(torch.float).cpu()
      .numpy()
    outputConv22.weight.copy(from: try! Tensor<Float>(numpy: output_conv2_2_weight))
    let output_conv2_2_bias = state_dict["scratch.output_conv2.2.bias"].type(torch.float).cpu()
      .numpy()
    outputConv22.bias.copy(from: try! Tensor<Float>(numpy: output_conv2_2_bias))
  }
  return (reader, Model([x0, x1, x2, x3], [out0]))
}

let (reader, vit) = DinoVisionTransformer(
  grid: 37, width: 1024, layers: 24, heads: 16, batchSize: 1, intermediateLayers: 4)
let (dptReader, depthHead) = DepthHead()

let random = Python.import("random")
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = image.cpu()
let y = depth_anything(x.cuda())
print(y)
let graph = DynamicGraph()
let xTensor = graph.variable(try! Tensor<Float>(numpy: x.numpy())).toGPU(0)
let state_dict = depth_anything.pretrained.state_dict()
let dpt_state_dict = depth_anything.depth_head.state_dict()

graph.withNoGrad {
  let class_embedding = state_dict["cls_token"].type(torch.float).cpu().numpy()
  let classEmbedding = graph.variable(try! Tensor<Float>(numpy: class_embedding)).reshaped(
    .CHW(1, 1, 1024)
  ).toGPU(0)
  let positional_embedding = state_dict["pos_embed"].type(torch.float).cpu().numpy()
  let positionalEmbedding = graph.variable(try! Tensor<Float>(numpy: positional_embedding))
    .reshaped(.CHW(1, 37 * 37 + 1, 1024)).toGPU(0)
  let _ = vit(inputs: xTensor, classEmbedding, positionalEmbedding)
  reader(state_dict)
  let outs = vit(inputs: xTensor, classEmbedding, positionalEmbedding).map { $0.as(of: Float.self) }
  let x0 = outs[0][1..<1370, 0..<1024].transposed(0, 1).reshaped(.NCHW(1, 1024, 37, 37)).copied()
  let x1 = outs[1][1..<1370, 0..<1024].transposed(0, 1).reshaped(.NCHW(1, 1024, 37, 37)).copied()
  let x2 = outs[2][1..<1370, 0..<1024].transposed(0, 1).reshaped(.NCHW(1, 1024, 37, 37)).copied()
  let x3 = outs[3][1..<1370, 0..<1024].transposed(0, 1).reshaped(.NCHW(1, 1024, 37, 37)).copied()
  let _ = depthHead(inputs: x0, x1, x2, x3)
  dptReader(dpt_state_dict)
  let out = depthHead(inputs: x0, x1, x2, x3)[0].as(of: Float.self).toCPU()
  debugPrint(out)
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: 518 * 518)
  for y in 0..<518 {
    for x in 0..<518 {
      let rgb = out[0, 0, y, x]
      rgba[y * 518 + x].r = UInt8(
        min(max(Int(rgb), 0), 255))
      rgba[y * 518 + x].g = UInt8(
        min(max(Int(rgb), 0), 255))
      rgba[y * 518 + x].b = UInt8(
        min(max(Int(rgb), 0), 255))
    }
  }
  let png = PNG.Data.Rectangular(
    packing: rgba, size: (518, 518),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! png.compress(path: "/home/liu/workspace/swift-diffusion/kandinsky_depth.png", level: 4)
}
