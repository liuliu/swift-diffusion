import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let ldm_modules_encoders_adapter = Python.import("ldm.modules.encoders.adapter")
let torch = Python.import("torch")
let random = Python.import("random")
let numpy = Python.import("numpy")

func ResnetBlock(outChannels: Int, inConv: Bool) -> (
  Model?, Model, Model, Model
) {
  let x = Input()
  let outX: Model.IO
  var skipModel: Model? = nil
  if inConv {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]))
    outX = skip(x)
    skipModel = skip
  } else {
    outX = x
  }
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = inLayerConv2d(outX)
  out = ReLU()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]))
  out = outLayerConv2d(out) + outX
  return (
    skipModel, inLayerConv2d, outLayerConv2d, Model([x], [out])
  )
}

func Adapter(channels: [Int], numRepeat: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (skipModel, inLayerConv2d, outLayerConv2d, resnetBlock) = ResnetBlock(
        outChannels: channel, inConv: previousChannel != channel)
      previousChannel = channel
      out = resnetBlock(out)
      let reader: (PythonObject) -> Void = { state_dict in
        let block1_weight = state_dict["body.\(i * numRepeat + j).block1.weight"].numpy()
        let block1_bias = state_dict["body.\(i * numRepeat + j).block1.bias"].numpy()
        inLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: block1_weight))
        inLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: block1_bias))
        let block2_weight = state_dict["body.\(i * numRepeat + j).block2.weight"].numpy()
        let block2_bias = state_dict["body.\(i * numRepeat + j).block2.bias"].numpy()
        outLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: block2_weight))
        outLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: block2_bias))
        if let skipModel = skipModel {
          let in_conv_weight = state_dict["body.\(i * numRepeat + j).in_conv.weight"].numpy()
          let in_conv_bias = state_dict["body.\(i * numRepeat + j).in_conv.bias"].numpy()
          skipModel.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_conv_weight))
          skipModel.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_conv_bias))
        }
      }
      readers.append(reader)
    }
    outs.append(out)
    if i != channels.count - 1 {
      let downsample = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
      out = downsample(out)
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["conv_in.weight"].numpy()
    let conv_in_bias = state_dict["conv_in.bias"].numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x], outs))
}

func ResnetBlockLight(outChannels: Int) -> (
  Model, Model, Model
) {
  let x = Input()
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = inLayerConv2d(x)
  out = ReLU()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outLayerConv2d(out) + x
  return (
    inLayerConv2d, outLayerConv2d, Model([x], [out])
  )
}

func Extractor(prefix: String, channel: Int, innerChannel: Int, numRepeat: Int, downsample: Bool)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  var out: Model.IO = x
  if downsample {
    let downsample = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
    out = downsample(out)
  }
  let inConv = Convolution(
    groups: 1, filters: innerChannel, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = inConv(out)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<numRepeat {
    let (inLayerConv2d, outLayerConv2d, resnetBlock) = ResnetBlockLight(outChannels: innerChannel)
    out = resnetBlock(out)
    let reader: (PythonObject) -> Void = { state_dict in
      let block1_weight = state_dict["body.\(prefix).body.\(i).block1.weight"].numpy()
      let block1_bias = state_dict["body.\(prefix).body.\(i).block1.bias"].numpy()
      inLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: block1_weight))
      inLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: block1_bias))
      let block2_weight = state_dict["body.\(prefix).body.\(i).block2.weight"].numpy()
      let block2_bias = state_dict["body.\(prefix).body.\(i).block2.bias"].numpy()
      outLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: block2_weight))
      outLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: block2_bias))
    }
    readers.append(reader)
  }
  let outConv = Convolution(
    groups: 1, filters: channel, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = outConv(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let in_conv_weight = state_dict["body.\(prefix).in_conv.weight"].numpy()
    let in_conv_bias = state_dict["body.\(prefix).in_conv.bias"].numpy()
    inConv.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_conv_weight))
    inConv.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_conv_bias))
    let out_conv_weight = state_dict["body.\(prefix).out_conv.weight"].numpy()
    let out_conv_bias = state_dict["body.\(prefix).out_conv.bias"].numpy()
    outConv.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_conv_weight))
    outConv.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_conv_bias))
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x], [out]))
}

func AdapterLight(channels: [Int], numRepeat: Int) -> ((PythonObject) -> Void, Model) {
  var readers = [(PythonObject) -> Void]()
  let x = Input()
  var out: Model.IO = x
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let (reader, extractor) = Extractor(
      prefix: "\(i)", channel: channel, innerChannel: channel / 4, numRepeat: numRepeat,
      downsample: i != 0)
    out = extractor(out)
    outs.append(out)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x], outs))
}

func QuickGELU() -> Model {
  let x = Input()
  let y = x .* Sigmoid()(1.702 * x)
  return Model([x], [y])
}

func CLIPSelfAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
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

func CLIPResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let (toqueries, tokeys, tovalues, unifyheads, attention) = CLIPSelfAttention(
    k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = QuickGELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_1_weight = state_dict["\(prefix).ln_1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).ln_1.bias"].type(torch.float).cpu().numpy()
    ln1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    ln1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let in_proj_weight = state_dict["\(prefix).attn.in_proj_weight"].type(torch.float).cpu().numpy()
    let in_proj_bias = state_dict["\(prefix).attn.in_proj_bias"].type(torch.float).cpu().numpy()
    toqueries.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[..<(k * h), ...]))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_proj_bias[..<(k * h)]))
    tokeys.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(k * h)..<(2 * k * h), ...]))
    tokeys.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(k * h)..<(2 * k * h)]))
    tovalues.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_proj_weight[(2 * k * h)..., ...]))
    tovalues.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: in_proj_bias[(2 * k * h)...]))
    let out_proj_weight = state_dict["\(prefix).attn.out_proj.weight"].type(torch.float).cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).attn.out_proj.bias"].type(torch.float).cpu().numpy()
    unifyheads.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_proj_bias))
    let ln_2_weight = state_dict["\(prefix).ln_2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).ln_2.bias"].type(torch.float).cpu().numpy()
    ln2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_2_weight))
    ln2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_2_bias))
    let c_fc_weight = state_dict["\(prefix).mlp.c_fc.weight"].type(torch.float).cpu().numpy()
    let c_fc_bias = state_dict["\(prefix).mlp.c_fc.bias"].type(torch.float).cpu().numpy()
    fc.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: c_fc_weight))
    fc.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: c_fc_bias))
    let c_proj_weight = state_dict["\(prefix).mlp.c_proj.weight"].type(torch.float).cpu().numpy()
    let c_proj_bias = state_dict["\(prefix).mlp.c_proj.bias"].type(torch.float).cpu().numpy()
    proj.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: c_proj_weight))
    proj.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: c_proj_bias))
  }
  return (reader, Model([x], [out]))
}

func StyleAdapter(width: Int, outputDim: Int, layers: Int, heads: Int, tokens: Int, batchSize: Int)
  -> ((PythonObject) -> Void, Model)
{
  let x = Input()
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = lnPre(x)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (reader, block) = CLIPResidualAttentionBlock(
      prefix: "transformer_layes.\(i)", k: width / heads, h: heads, b: batchSize, t: 257 + tokens)
    out = block(out.reshaped([batchSize, 257 + tokens, width]))
    readers.append(reader)
  }
  let lnPost = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPost(out.reshaped([batchSize, 257 + tokens, width]))
  let proj = Dense(count: outputDim, noBias: true)
  out = proj(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_pre_weight = state_dict["ln_pre.weight"].type(torch.float).cpu().numpy()
    let ln_pre_bias = state_dict["ln_pre.bias"].type(torch.float).cpu().numpy()
    lnPre.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_pre_weight))
    lnPre.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_pre_bias))
    let ln_post_weight = state_dict["ln_post.weight"].type(torch.float).cpu().numpy()
    let ln_post_bias = state_dict["ln_post.bias"].type(torch.float).cpu().numpy()
    lnPost.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_post_weight))
    lnPost.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_post_bias))
    let proj_weight = state_dict["proj"].type(torch.float).cpu().numpy()
    var projTensor = Tensor<Float>(.CPU, .NC(outputDim, width))
    for i in 0..<outputDim {
      for j in 0..<width {
        projTensor[i, j] = Float(proj_weight[j, i])!
      }
    }
    proj.parameters(for: .weight).copy(from: projTensor)
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([x], [out]))
}

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let hint = torch.randn([2, 3, 512, 512])

// let adapter = ldm_modules_encoders_adapter.Adapter(
//   cin: 64, channels: [320, 640, 1280, 1280], nums_rb: 2, ksize: 1, sk: true, use_conv: false
// ).to(torch.device("cpu"))
let adapterLight = ldm_modules_encoders_adapter.Adapter_light(
  cin: 64 * 3, channels: [320, 640, 1280, 1280], nums_rb: 4
).to(torch.device("cpu"))
// let style = torch.randn([1, 257, 1024])
// let styleAdapter = ldm_modules_encoders_adapter.StyleAdapter(
//   width: 1024, context_dim: 768, num_head: 8, n_layes: 3, num_token: 8
// ).to(torch.device("cpu"))
adapterLight.load_state_dict(
  torch.load("/home/liu/workspace/T2I-Adapter/models/t2iadapter_color_sd14v1.pth"))
let state_dict = adapterLight.state_dict()
print(state_dict.keys())
let ret = adapterLight(hint)
// print(ret[1])

// let styleEmbed = try Tensor<Float>(
//   numpy: state_dict["style_embedding"].type(torch.float).cpu().numpy())

let graph = DynamicGraph()
let hintTensor = graph.variable(try! Tensor<Float>(numpy: hint.numpy())).toGPU(0)
// let styleTensor = graph.variable(try! Tensor<Float>(numpy: style.numpy())).toGPU(0)
// let (reader, adapternet) = Adapter(channels: [320, 640, 1280, 1280], numRepeat: 2)
let (reader, adapternet) = AdapterLight(channels: [320, 640, 1280, 1280], numRepeat: 4)
// let (reader, styleadapternet) = StyleAdapter(
//   width: 1024, outputDim: 768, layers: 3, heads: 8, tokens: 8, batchSize: 1)
graph.workspaceSize = 1_024 * 1_024 * 1_024
graph.withNoGrad {
  let hintIn = hintTensor.reshaped(format: .NCHW, shape: [2, 3, 64, 8, 64, 8]).permuted(
    0, 1, 3, 5, 2, 4
  ).copied().reshaped(.NCHW(2, 64 * 3, 64, 64))
  var controls = adapternet(inputs: hintIn).map { $0.as(of: Float.self) }
  reader(state_dict)
  controls = adapternet(inputs: hintIn).map { $0.as(of: Float.self) }
  debugPrint(controls[1])
  // let styleEmbedTensor = graph.variable(styleEmbed).toGPU(0)
  // var styleAll = graph.variable(.GPU(0), .CHW(1, 257 + 8, 1024), of: Float.self)
  // styleAll[0..<1, 0..<257, 0..<1024] = styleTensor
  // styleAll[0..<1, 257..<265, 0..<1024] = styleEmbedTensor
  // var contexts = styleadapternet(inputs: styleAll)[0].as(of: Float.self)
  // reader(state_dict)
  // contexts = styleadapternet(inputs: styleAll)[0].as(of: Float.self)
  // debugPrint(contexts[0..<1, 257..<265, 0..<768])
  graph.openStore("/home/liu/workspace/swift-diffusion/adapter.ckpt") {
    // $0.write("style_embedding", variable: styleEmbedTensor)
    $0.write("adapter", model: adapternet)
  }
}
