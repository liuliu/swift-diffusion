import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let torch = Python.import("torch")
let diffusers = Python.import("diffusers")

let lcm_unet = diffusers.UNet2DConditionModel.from_pretrained(
  "latent-consistency/lcm-sdxl", torch_dtype: torch.float16, variant: "fp16")
let pipe = diffusers.DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", unet: lcm_unet, torch_dtype: torch.float16,
  variant: "fp16")

pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

let prompt = "a close-up picture of an old man standing in the rain"

let image = pipe(prompt, num_inference_steps: 4, guidance_scale: 8.0).images[0]

image.save("/home/liu/workspace/swift-diffusion/image.png")

/*
let state_dict = pipe.unet.state_dict()

print(pipe.unet)
print(state_dict.keys())

func TimeEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func LabelEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func ResBlock(b: Int, outChannels: Int, skipConnection: Bool) -> (
  Model, Model, Model, Model, Model, Model?, Model
) {
  let x = Input()
  let emb = Input()
  let inLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  var out = inLayerNorm(x)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = inLayerConv2d(out)
  let embLayer = Dense(count: outChannels)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([b, outChannels, 1, 1])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var skipModel: Model? = nil
  if skipConnection {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]))
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = skip
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
  }
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel,
    Model([x, emb], [out])
  )
}

func SelfAttention(k: Int, h: Int, b: Int, hw: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, hw, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, hw])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
}

func CrossAttentionKeysAndValues(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = Dense(count: k * h, noBias: true)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, unifyheads, Model([x, keys, values], [out]))
}

func FeedForward(hiddenSize: Int, intermediateSize: Int) -> (Model, Model, Model, Model) {
  let x = Input()
  let fc10 = Dense(count: intermediateSize)
  let fc11 = Dense(count: intermediateSize)
  var out = fc10(x)
  out = out .* GELU()(fc11(x))
  let fc2 = Dense(count: hiddenSize)
  out = fc2(out)
  return (fc10, fc11, fc2, Model([x], [out]))
}

func BasicTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let (toqueries2, unifyheads2, attn2) = CrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, keys, values) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  let reader: (PythonObject) -> Void = { state_dict in
    let attn1_to_k_weight = state_dict[
      "\(prefix).attn1.to_k.weight"
    ].float().cpu().numpy()
    tokeys1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_k_weight))
    print("\"diffusion_model.\(prefix).attn1.to_k.weight\": [\"\(tokeys1.weight.name)\"],")
    let attn1_to_q_weight = state_dict[
      "\(prefix).attn1.to_q.weight"
    ].float().cpu().numpy()
    toqueries1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_q_weight))
    print("\"diffusion_model.\(prefix).attn1.to_q.weight\": [\"\(toqueries1.weight.name)\"],")
    let attn1_to_v_weight = state_dict[
      "\(prefix).attn1.to_v.weight"
    ].float().cpu().numpy()
    tovalues1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn1_to_v_weight))
    print("\"diffusion_model.\(prefix).attn1.to_v.weight\": [\"\(tovalues1.weight.name)\"],")
    let attn1_to_out_weight = state_dict[
      "\(prefix).attn1.to_out.0.weight"
    ].float().cpu().numpy()
    let attn1_to_out_bias = state_dict[
      "\(prefix).attn1.to_out.0.bias"
    ].float().cpu().numpy()
    unifyheads1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attn1_to_out_weight))
    print("\"diffusion_model.\(prefix).attn1.to_out.0.weight\": [\"\(unifyheads1.weight.name)\"],")
    unifyheads1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn1_to_out_bias))
    print("\"diffusion_model.\(prefix).attn1.to_out.0.bias\": [\"\(unifyheads1.bias.name)\"],")
    let ff_net_0_proj_weight = state_dict[
      "\(prefix).ff.net.0.proj.weight"
    ].float().cpu().numpy()
    let ff_net_0_proj_bias = state_dict[
      "\(prefix).ff.net.0.proj.bias"
    ].float().cpu().numpy()
    fc10.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[..<intermediateSize, ...]))
    fc10.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[..<intermediateSize]))
    fc11.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_weight[intermediateSize..., ...]))
    fc11.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: ff_net_0_proj_bias[intermediateSize...]))
    print(
      "\"diffusion_model.\(prefix).ff.net.0.proj.weight\": [\"\(fc10.weight.name)\", \"\(fc11.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).ff.net.0.proj.bias\": [\"\(fc10.bias.name)\", \"\(fc11.bias.name)\"],"
    )
    let ff_net_2_weight = state_dict[
      "\(prefix).ff.net.2.weight"
    ].float().cpu().numpy()
    let ff_net_2_bias = state_dict[
      "\(prefix).ff.net.2.bias"
    ].float().cpu().numpy()
    fc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ff_net_2_weight))
    fc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ff_net_2_bias))
    print("\"diffusion_model.\(prefix).ff.net.2.weight\": [\"\(fc2.weight.name)\"],")
    print("\"diffusion_model.\(prefix).ff.net.2.bias\": [\"\(fc2.bias.name)\"],")
    let attn2_to_q_weight = state_dict[
      "\(prefix).attn2.to_q.weight"
    ].float().cpu().numpy()
    toqueries2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: attn2_to_q_weight))
    print("\"diffusion_model.\(prefix).attn2.to_q.weight\": [\"\(toqueries2.weight.name)\"],")
    let attn2_to_out_weight = state_dict[
      "\(prefix).attn2.to_out.0.weight"
    ].float().cpu().numpy()
    let attn2_to_out_bias = state_dict[
      "\(prefix).attn2.to_out.0.bias"
    ].float().cpu().numpy()
    unifyheads2.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: attn2_to_out_weight))
    unifyheads2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: attn2_to_out_bias))
    print("\"diffusion_model.\(prefix).attn2.to_out.0.weight\": [\"\(unifyheads2.weight.name)\"],")
    print("\"diffusion_model.\(prefix).attn2.to_out.0.bias\": [\"\(unifyheads2.bias.name)\"],")
    let norm1_weight = state_dict[
      "\(prefix).norm1.weight"
    ]
    .float().cpu().numpy()
    let norm1_bias = state_dict[
      "\(prefix).norm1.bias"
    ]
    .float().cpu().numpy()
    layerNorm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    layerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    print("\"diffusion_model.\(prefix).norm1.weight\": [\"\(layerNorm1.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm1.bias\": [\"\(layerNorm1.bias.name)\"],")
    let norm2_weight = state_dict[
      "\(prefix).norm2.weight"
    ]
    .float().cpu().numpy()
    let norm2_bias = state_dict[
      "\(prefix).norm2.bias"
    ]
    .float().cpu().numpy()
    layerNorm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
    layerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
    print("\"diffusion_model.\(prefix).norm2.weight\": [\"\(layerNorm2.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm2.bias\": [\"\(layerNorm2.bias.name)\"],")
    let norm3_weight = state_dict[
      "\(prefix).norm3.weight"
    ]
    .float().cpu().numpy()
    let norm3_bias = state_dict[
      "\(prefix).norm3.bias"
    ]
    .float().cpu().numpy()
    layerNorm3.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm3_weight))
    layerNorm3.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm3_bias))
    print("\"diffusion_model.\(prefix).norm3.weight\": [\"\(layerNorm3.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm3.bias\": [\"\(layerNorm3.bias.name)\"],")
  }
  return (reader, Model([x, keys, values], [out]))
}

func SpatialTransformer(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).permuted(0, 2, 1)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<depth {
    let keys = Input()
    kvs.append(keys)
    let values = Input()
    kvs.append(values)
    let (reader, block) = BasicTransformerBlock(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    out = block(out, keys, values)
    readers.append(reader)
  }
  out = out.reshaped([b, height, width, k * h]).permuted(0, 3, 1, 2)
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  let reader: (PythonObject) -> Void = { state_dict in
    let norm_weight = state_dict["\(prefix).norm.weight"]
      .float().cpu().numpy()
    let norm_bias = state_dict["\(prefix).norm.bias"].float().cpu().numpy()
    norm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_weight))
    norm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_bias))
    print("\"diffusion_model.\(prefix).norm.weight\": [\"\(norm.weight.name)\"],")
    print("\"diffusion_model.\(prefix).norm.bias\": [\"\(norm.bias.name)\"],")
    let proj_in_weight = state_dict["\(prefix).proj_in.weight"]
      .float().cpu().numpy()
    let proj_in_bias = state_dict["\(prefix).proj_in.bias"]
      .float().cpu().numpy()
    projIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_in_weight))
    projIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_in_bias))
    print("\"diffusion_model.\(prefix).proj_in.weight\": [\"\(projIn.weight.name)\"],")
    print("\"diffusion_model.\(prefix).proj_in.bias\": [\"\(projIn.bias.name)\"],")
    for reader in readers {
      reader(state_dict)
    }
    let proj_out_weight = state_dict[
      "\(prefix).proj_out.weight"
    ].float().cpu().numpy()
    let proj_out_bias = state_dict["\(prefix).proj_out.bias"]
      .float().cpu().numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
    print("\"diffusion_model.\(prefix).proj_out.weight\": [\"\(projOut.weight.name)\"],")
    print("\"diffusion_model.\(prefix).proj_out.bias\": [\"\(projOut.bias.name)\"],")
  }
  return (reader, Model([x] + kvs, [out]))
}

func BlockLayer(
  prefix: String, layerStart: Int, inLayerStart: Int, skipConnection: Bool, attentionBlock: Int,
  channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let emb = Input()
  var kvs = [Model.IO]()
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  var transformerReader: ((PythonObject) -> Void)? = nil
  if attentionBlock > 0 {
    let c = (0..<(attentionBlock * 2)).map { _ in Input() }
    let transformer: Model
    (
      transformerReader, transformer
    ) = SpatialTransformer(
      prefix: "\(prefix).\(layerStart).attentions.\(inLayerStart)",
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let in_layers_0_weight = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).norm1.weight"
    ].float().cpu().numpy()
    let in_layers_0_bias = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).norm1.bias"
    ].float().cpu().numpy()
    inLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_0_weight))
    inLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.in_layers.0.weight\": [\"\(inLayerNorm.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.in_layers.0.bias\": [\"\(inLayerNorm.bias.name)\"],"
    )
    let in_layers_2_weight = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).conv1.weight"
    ].float().cpu().numpy()
    let in_layers_2_bias = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).conv1.bias"
    ].float().cpu().numpy()
    inLayerConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: in_layers_2_weight))
    inLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.in_layers.2.weight\": [\"\(inLayerConv2d.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.in_layers.2.bias\": [\"\(inLayerConv2d.bias.name)\"],"
    )
    let emb_layers_1_weight = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).time_emb_proj.weight"
    ].float().cpu().numpy()
    let emb_layers_1_bias = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).time_emb_proj.bias"
    ].float().cpu().numpy()
    embLayer.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_1_weight))
    embLayer.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_1_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.weight\": [\"\(embLayer.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.emb_layers.1.bias\": [\"\(embLayer.bias.name)\"],"
    )
    let out_layers_0_weight = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).norm2.weight"
    ].float().cpu().numpy()
    let out_layers_0_bias = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).norm2.bias"
    ].float().cpu().numpy()
    outLayerNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_layers_0_weight))
    outLayerNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).resnets.\(inLayerStart).conv2.weight\": [\"\(outLayerNorm.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).resnets.\(inLayerStart).conv2.bias\": [\"\(outLayerNorm.bias.name)\"],"
    )
    let out_layers_3_weight = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).conv2.weight"
    ].float().cpu().numpy()
    let out_layers_3_bias = state_dict[
      "\(prefix).\(layerStart).resnets.\(inLayerStart).conv2.bias"
    ].float().cpu().numpy()
    outLayerConv2d.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_3_weight))
    outLayerConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_3_bias))
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.out_layers.3.weight\": [\"\(outLayerConv2d.weight.name)\"],"
    )
    print(
      "\"diffusion_model.\(prefix).\(layerStart).0.out_layers.3.bias\": [\"\(outLayerConv2d.bias.name)\"],"
    )
    if let skipModel = skipModel {
      let skip_connection_weight = state_dict[
        "\(prefix).\(layerStart).resnets.\(inLayerStart).conv_shortcut.weight"
      ].float().cpu().numpy()
      let skip_connection_bias = state_dict[
        "\(prefix).\(layerStart).resnets.\(inLayerStart).conv_shortcut.bias"
      ].float().cpu().numpy()
      skipModel.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: skip_connection_weight))
      skipModel.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: skip_connection_bias))
      print(
        "\"diffusion_model.\(prefix).\(layerStart).0.skip_connection.weight\": [\"\(skipModel.weight.name)\"],"
      )
      print(
        "\"diffusion_model.\(prefix).\(layerStart).0.skip_connection.bias\": [\"\(skipModel.bias.name)\"],"
      )
    }
    if let transformerReader = transformerReader {
      transformerReader(state_dict)
    }
  }
  return (reader, Model([x, emb] + kvs, [out]))
}

func MiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, x: Model.IO, emb: Model.IO
) -> ((PythonObject) -> Void, Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let kvs = (0..<(attentionBlock * 2)).map { _ in Input() }
  let transformerReader: ((PythonObject) -> Void)?
  let inLayerNorm2: Model?
  let inLayerConv2d2: Model?
  let embLayer2: Model?
  let outLayerNorm2: Model?
  let outLayerConv2d2: Model?
  if attentionBlock > 0 {
    let (
      reader, transformer
    ) = SpatialTransformer(
      prefix: "mid_block.attentions.0", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
      width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
    out = transformer([out] + kvs)
    let resBlock2: Model
    (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
      ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
    out = resBlock2(out, emb)
    transformerReader = reader
  } else {
    transformerReader = nil
    inLayerNorm2 = nil
    inLayerConv2d2 = nil
    embLayer2 = nil
    outLayerNorm2 = nil
    outLayerConv2d2 = nil
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let in_layers_0_0_weight = state_dict["mid_block.resnets.0.norm1.weight"]
      .float().cpu().numpy()
    let in_layers_0_0_bias = state_dict["mid_block.resnets.0.norm1.bias"].float().cpu()
      .numpy()
    inLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_0_0_weight))
    inLayerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_0_bias))
    print(
      "\"diffusion_model.middle_block.0.in_layers.0.weight\": [\"\(inLayerNorm1.weight.name)\"],")
    print("\"diffusion_model.middle_block.0.in_layers.0.bias\": [\"\(inLayerNorm1.bias.name)\"],")
    let in_layers_0_2_weight = state_dict["mid_block.resnets.0.conv1.weight"]
      .float().cpu().numpy()
    let in_layers_0_2_bias = state_dict["mid_block.resnets.0.conv1.bias"].float().cpu()
      .numpy()
    inLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: in_layers_0_2_weight))
    inLayerConv2d1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_0_2_bias))
    print(
      "\"diffusion_model.middle_block.0.in_layers.2.weight\": [\"\(inLayerConv2d1.weight.name)\"],")
    print("\"diffusion_model.middle_block.0.in_layers.2.bias\": [\"\(inLayerConv2d1.bias.name)\"],")
    let emb_layers_0_1_weight = state_dict["mid_block.resnets.0.time_emb_proj.weight"]
      .float().cpu().numpy()
    let emb_layers_0_1_bias = state_dict["mid_block.resnets.0.time_emb_proj.bias"].float().cpu()
      .numpy()
    embLayer1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_0_1_weight))
    embLayer1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_0_1_bias))
    print("\"diffusion_model.middle_block.0.emb_layers.1.weight\": [\"\(embLayer1.weight.name)\"],")
    print("\"diffusion_model.middle_block.0.emb_layers.1.bias\": [\"\(embLayer1.bias.name)\"],")
    let out_layers_0_0_weight = state_dict["mid_block.resnets.0.norm2.weight"]
      .float().cpu().numpy()
    let out_layers_0_0_bias = state_dict["mid_block.resnets.0.norm2.bias"].float().cpu().numpy()
    outLayerNorm1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_0_weight))
    outLayerNorm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_0_0_bias))
    print(
      "\"diffusion_model.middle_block.0.out_layers.0.weight\": [\"\(outLayerNorm1.weight.name)\"],")
    print("\"diffusion_model.middle_block.0.out_layers.0.bias\": [\"\(outLayerNorm1.bias.name)\"],")
    let out_layers_0_3_weight = state_dict["mid_block.resnets.0.conv2.weight"]
      .float().cpu().numpy()
    let out_layers_0_3_bias = state_dict["mid_block.resnets.0.conv2.bias"].float().cpu()
      .numpy()
    outLayerConv2d1.parameters(for: .weight).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_3_weight))
    outLayerConv2d1.parameters(for: .bias).copy(
      from: try! Tensor<Float>(numpy: out_layers_0_3_bias))
    print(
      "\"diffusion_model.middle_block.0.out_layers.3.weight\": [\"\(outLayerConv2d1.weight.name)\"],"
    )
    print(
      "\"diffusion_model.middle_block.0.out_layers.3.bias\": [\"\(outLayerConv2d1.bias.name)\"],")
    transformerReader?(state_dict)
    if let inLayerNorm2 = inLayerNorm2, let inLayerConv2d2 = inLayerConv2d2, let embLayer2 = embLayer2, let outLayerNorm2 = outLayerNorm2, let outLayerConv2d2 = outLayerConv2d2 {
      let in_layers_2_0_weight = state_dict["mid_block.resnets.1.norm1.weight"]
        .float().cpu().numpy()
      let in_layers_2_0_bias = state_dict["mid_block.resnets.1.norm1.bias"].float().cpu()
        .numpy()
      inLayerNorm2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: in_layers_2_0_weight))
      inLayerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_0_bias))
      print("\"diffusion_model.middle_block.2.in_layers.0.weight\": [\"\(inLayerNorm2.weight.name)\"],")
      print("\"diffusion_model.middle_block.2.in_layers.0.bias\": [\"\(inLayerNorm2.bias.name)\"],")
      let in_layers_2_2_weight = state_dict["mid_block.resnets.1.conv1.weight"]
        .float().cpu().numpy()
      let in_layers_2_2_bias = state_dict["mid_block.resnets.1.conv1.bias"].float().cpu()
        .numpy()
      inLayerConv2d2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: in_layers_2_2_weight))
      inLayerConv2d2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: in_layers_2_2_bias))
      print("\"diffusion_model.middle_block.2.in_layers.2.weight\": [\"\(inLayerConv2d2.weight.name)\"],")
      print("\"diffusion_model.middle_block.2.in_layers.2.bias\": [\"\(inLayerConv2d2.bias.name)\"],")
      let emb_layers_2_1_weight = state_dict["mid_block.resnets.1.time_emb_proj.weight"]
        .float().cpu().numpy()
      let emb_layers_2_1_bias = state_dict["mid_block.resnets.1.time_emb_proj.bias"].float().cpu()
        .numpy()
      embLayer2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: emb_layers_2_1_weight))
      embLayer2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: emb_layers_2_1_bias))
      print("\"diffusion_model.middle_block.2.emb_layers.1.weight\": [\"\(embLayer2.weight.name)\"],")
      print("\"diffusion_model.middle_block.2.emb_layers.1.bias\": [\"\(embLayer2.bias.name)\"],")
      let out_layers_2_0_weight = state_dict["mid_block.resnets.1.norm2.weight"]
        .float().cpu().numpy()
      let out_layers_2_0_bias = state_dict[
        "mid_block.resnets.1.norm2.bias"
      ].float().cpu().numpy()
      outLayerNorm2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: out_layers_2_0_weight))
      outLayerNorm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_layers_2_0_bias))
      print("\"diffusion_model.middle_block.2.out_layers.0.weight\": [\"\(outLayerNorm2.weight.name)\"],")
      print("\"diffusion_model.middle_block.2.out_layers.0.bias\": [\"\(outLayerNorm2.bias.name)\"],")
      let out_layers_2_3_weight = state_dict["mid_block.resnets.1.conv2.weight"]
        .float().cpu().numpy()
      let out_layers_2_3_bias = state_dict["mid_block.resnets.1.conv2.bias"].float().cpu()
        .numpy()
      outLayerConv2d2.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: out_layers_2_3_weight))
      outLayerConv2d2.parameters(for: .bias).copy(
        from: try! Tensor<Float>(numpy: out_layers_2_3_bias))
      print("\"diffusion_model.middle_block.2.out_layers.3.weight\": [\"\(outLayerConv2d2.weight.name)\"],")
      print("\"diffusion_model.middle_block.2.out_layers.3.bias\": [\"\(outLayerConv2d2.bias.name)\"],")
    }
  }
  return (reader, out, kvs)
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: [Int]], x: Model.IO, emb: Model.IO
) -> ((PythonObject) -> Void, [Model.IO], Model.IO, [Input]) {
  let conv2d = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    for j in 0..<numRepeat {
      let (reader, inputLayer) = BlockLayer(
        prefix: "down_blocks",
        layerStart: i, inLayerStart: j, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      let c = (0..<(attentionBlock[j] * 2)).map { _ in Input() }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      passLayers.append(out)
      readers.append(reader)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let reader: (PythonObject) -> Void = { state_dict in
        let op_weight = state_dict["down_blocks.\(i).downsamplers.0.conv.weight"].float().cpu()
          .numpy()
        let op_bias = state_dict["down_blocks.\(i).downsamplers.0.conv.bias"].float().cpu()
          .numpy()
        downsample.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: op_weight))
        downsample.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: op_bias))
        print(
          "\"diffusion_model.input_blocks.\(downLayer).0.op.weight\": [\"\(downsample.weight.name)\"],"
        )
        print(
          "\"diffusion_model.input_blocks.\(downLayer).0.op.bias\": [\"\(downsample.bias.name)\"],")
      }
      readers.append(reader)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let input_blocks_0_0_weight = state_dict["conv_in.weight"].float().cpu()
      .numpy()
    let input_blocks_0_0_bias = state_dict["conv_in.bias"].float().cpu().numpy()
    conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_weight))
    conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: input_blocks_0_0_bias))
    print("\"diffusion_model.input_blocks.0.0.weight\": [\"\(conv2d.weight.name)\"],")
    print("\"diffusion_model.input_blocks.0.0.bias\": [\"\(conv2d.bias.name)\"],")
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, passLayers, out, kvs)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: [Int]], x: Model.IO, emb: Model.IO,
  inputs: [Model.IO]
) -> ((PythonObject) -> Void, Model.IO, [Input]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat + 1)]
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let (reader, outputLayer) = BlockLayer(
        prefix: "up_blocks",
        layerStart: channels.count - 1 - i, inLayerStart: j, skipConnection: true,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      let c = (0..<(attentionBlock[j] * 2)).map { _ in Input() }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      readers.append(reader)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
        let upLayer = channels.count - 1 - i  // layerStart
        let reader: (PythonObject) -> Void = { state_dict in
          let op_weight = state_dict[
            "up_blocks.\(upLayer).upsamplers.0.conv.weight"
          ].float().cpu().numpy()
          let op_bias = state_dict["up_blocks.\(upLayer).upsamplers.0.conv.bias"]
            .float().cpu().numpy()
          conv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: op_weight))
          conv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: op_bias))
          print(
            "\"diffusion_model.output_blocks.\(upLayer).conv.weight\": [\"\(conv2d.weight.name)\"],"
          )
          print(
            "\"diffusion_model.output_blocks.\(upLayer).conv.bias\": [\"\(conv2d.bias.name)\"],")
        }
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, out, kvs)
}

func UNetXL(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int, outputAttentionRes: KeyValuePairs<Int, [Int]>
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let (timeFc0, timeFc2, timeEmbed) = TimeEmbed(modelChannels: channels[0])
  let (labelFc0, labelFc2, labelEmbed) = LabelEmbed(modelChannels: channels[0])
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (inputReader, inputs, inputBlocks, inputKVs) = InputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77,
    attentionRes: inputAttentionRes,
    x: x, emb: emb)
  var out = inputBlocks
  let (middleReader, middleBlock, middleKVs) = MiddleBlock(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 77, attentionBlock: middleAttentionBlocks, x: out, emb: emb)
  out = middleBlock
  let (outputReader, outputBlocks, outputKVs) = OutputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77,
    attentionRes: outputAttentionRes,
    x: out, emb: emb, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let time_embed_0_weight = state_dict["time_embedding.linear_1.weight"].float().cpu().numpy()
    let time_embed_0_bias = state_dict["time_embedding.linear_1.bias"].float().cpu().numpy()
    let time_embed_2_weight = state_dict["time_embedding.linear_2.weight"].float().cpu().numpy()
    let time_embed_2_bias = state_dict["time_embedding.linear_2.bias"].float().cpu().numpy()
    timeFc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_0_weight))
    timeFc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_0_bias))
    print("\"diffusion_model.time_embed.0.weight\": [\"\(timeFc0.weight.name)\"],")
    print("\"diffusion_model.time_embed.0.bias\": [\"\(timeFc0.bias.name)\"],")
    timeFc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_2_weight))
    timeFc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_2_bias))
    print("\"diffusion_model.time_embed.2.weight\": [\"\(timeFc2.weight.name)\"],")
    print("\"diffusion_model.time_embed.2.bias\": [\"\(timeFc2.bias.name)\"],")
    let label_emb_0_0_weight = state_dict["add_embedding.linear_1.weight"].float().cpu().numpy()
    let label_emb_0_0_bias = state_dict["add_embedding.linear_1.bias"].float().cpu().numpy()
    let label_emb_0_2_weight = state_dict["add_embedding.linear_2.weight"].float().cpu().numpy()
    let label_emb_0_2_bias = state_dict["add_embedding.linear_2.bias"].float().cpu().numpy()
    labelFc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: label_emb_0_0_weight))
    labelFc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: label_emb_0_0_bias))
    print("\"diffusion_model.label_emb.0.0.weight\": [\"\(labelFc0.weight.name)\"],")
    print("\"diffusion_model.label_emb.0.0.bias\": [\"\(labelFc0.bias.name)\"],")
    labelFc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: label_emb_0_2_weight))
    labelFc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: label_emb_0_2_bias))
    print("\"diffusion_model.label_emb.0.2.weight\": [\"\(labelFc2.weight.name)\"],")
    print("\"diffusion_model.label_emb.0.2.bias\": [\"\(labelFc2.bias.name)\"],")
    inputReader(state_dict)
    middleReader(state_dict)
    outputReader(state_dict)
    let out_0_weight = state_dict["conv_norm_out.weight"].float().cpu().numpy()
    let out_0_bias = state_dict["conv_norm_out.bias"].float().cpu().numpy()
    outNorm.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_0_weight))
    outNorm.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_0_bias))
    print("\"diffusion_model.out.0.weight\": [\"\(outNorm.weight.name)\"],")
    print("\"diffusion_model.out.0.bias\": [\"\(outNorm.bias.name)\"],")
    let out_2_weight = state_dict["conv_out.weight"].float().cpu().numpy()
    let out_2_bias = state_dict["conv_out.bias"].float().cpu().numpy()
    outConv2d.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: out_2_weight))
    outConv2d.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: out_2_bias))
    print("\"diffusion_model.out.2.weight\": [\"\(outConv2d.weight.name)\"],")
    print("\"diffusion_model.out.2.bias\": [\"\(outConv2d.bias.name)\"],")
  }
  return (reader, Model([x, t_emb, y] + inputKVs + middleKVs + outputKVs, [out]))
}

func CrossAttentionFixed(k: Int, h: Int, b: Int, hw: Int, t: Int) -> (Model, Model, Model) {
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
  return (tokeys, tovalues, Model([c], [keys, values]))
}

func BasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> (
  (PythonObject) -> Void, Model
) {
  let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(k: k, h: h, b: b, hw: hw, t: t)
  let reader: (PythonObject) -> Void = { state_dict in
    let attn2_to_k_weight = state_dict[
      "\(prefix).attn2.to_k.weight"
    ].float().cpu().numpy()
    tokeys2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_k_weight))
    print("\"diffusion_model.\(prefix).attn2.to_k.weight\": [\"\(tokeys2.weight.name)\"],")
    let attn2_to_v_weight = state_dict[
      "\(prefix).attn2.to_v.weight"
    ].float().cpu().numpy()
    tovalues2.weight.copy(from: try! Tensor<Float>(numpy: attn2_to_v_weight))
    print("\"diffusion_model.\(prefix).attn2.to_v.weight\": [\"\(tovalues2.weight.name)\"],")
  }
  return (reader, attn2)
}

func SpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  var readers = [(PythonObject) -> Void]()
  var outs = [Model.IO]()
  let hw = height * width
  for i in 0..<depth {
    let (reader, block) = BasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(block(c))
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, Model([c], outs))
}

func BlockLayerFixed(
  prefix: String,
  layerStart: Int, inLayerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int,
  numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> ((PythonObject) -> Void, Model) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (transformerReader, transformer) = SpatialTransformerFixed(
    prefix: "\(prefix).\(layerStart).attentions.\(inLayerStart)",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingSize,
    intermediateSize: channels * 4)
  return (transformerReader, transformer)
}

func MiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, c: Model.IO
) -> ((PythonObject) -> Void, Model.IO) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (
    transformerReader, transformer
  ) = SpatialTransformerFixed(
    prefix: "mid_block.attentions.0", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  let out = transformer(c)
  return (transformerReader, out)
}

func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: [Int]], c: Model.IO
) -> ((PythonObject) -> Void, [Model.IO]) {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    for j in 0..<numRepeat {
      if attentionBlock[j] > 0 {
        let (reader, inputLayer) = BlockLayerFixed(
          prefix: "down_blocks",
          layerStart: i, inLayerStart: j, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        previousChannel = channel
        outs.append(inputLayer(c))
        readers.append(reader)
      }
      layerStart += 1
    }
    if i != channels.count - 1 {
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, outs)
}

func OutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: [Int]], c: Model.IO
) -> ((PythonObject) -> Void, [Model.IO]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var readers = [(PythonObject) -> Void]()
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat + 1)]
    for j in 0..<(numRepeat + 1) {
      if attentionBlock[j] > 0 {
        let (reader, outputLayer) = BlockLayerFixed(
          prefix: "up_blocks",
          layerStart: channels.count - 1 - i, inLayerStart: j, skipConnection: true,
          attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        outs.append(outputLayer(c))
        readers.append(reader)
      }
      layerStart += 1
    }
  }
  let reader: (PythonObject) -> Void = { state_dict in
    for reader in readers {
      reader(state_dict)
    }
  }
  return (reader, outs)
}

func UNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int, outputAttentionRes: KeyValuePairs<Int, [Int]>
) -> ((PythonObject) -> Void, Model) {
  let c = Input()
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let (inputReader, inputBlocks) = InputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77,
    attentionRes: inputAttentionRes,
    c: c)
  var out = inputBlocks
  let middleReader: ((PythonObject) -> Void)?
  if middleAttentionBlocks > 0 {
    let middleBlockSizeMult = 1 << (channels.count - 1)
    let (reader, middleBlock) = MiddleBlockFixed(
      channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
      height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
      embeddingSize: 77, attentionBlock: middleAttentionBlocks, c: c)
    out.append(middleBlock)
    middleReader = reader
  } else {
    middleReader = nil
  }
  let (outputReader, outputBlocks) = OutputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77,
    attentionRes: outputAttentionRes,
    c: c)
  out.append(contentsOf: outputBlocks)
  let reader: (PythonObject) -> Void = { state_dict in
    inputReader(state_dict)
    middleReader?(state_dict)
    outputReader(state_dict)
  }
  return (reader, Model([c], out))
}

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let x = torch.randn([2, 4, 128, 128])
let c = torch.randn([2, 77, 2048])
let t = torch.full([1], 981)
let y = torch.randn([2, 2816])

let graph = DynamicGraph()

func timeEmbedding(timesteps: Int, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * Float(timesteps)
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

func guidanceScaleEmbedding(guidanceScale: Float, embeddingSize: Int) -> Tensor<Float> {
  // This is only slightly different from timeEmbedding by:
  // 1. sin before cos.
  // 2. w is scaled by 1000.0
  // 3. half v.s. half - 1 when scale down.
  precondition(embeddingSize % 2 == 0)
  let guidanceScale = guidanceScale * 1000
  var embedding = Tensor<Float>(.CPU, .NC(1, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(10_000)) * Float(i) / Float(half - 1)) * guidanceScale
    let sinFreq = sin(freq)
    let cosFreq = cos(freq)
    embedding[0, i] = sinFreq
    embedding[0, i + half] = cosFreq
  }
  return embedding
}

let t_emb = graph.variable(
  timeEmbedding(timesteps: 981, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000)
).toGPU(1)
let xTensor = graph.variable(try! Tensor<Float>(numpy: x.numpy())).toGPU(1)
let cTensor = graph.variable(try! Tensor<Float>(numpy: c.numpy())).toGPU(1)
let yTensor = graph.variable(try! Tensor<Float>(numpy: y.numpy())).toGPU(1)
let (readerFixed, unetFixed) = UNetXLFixed(
  batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
  inputAttentionRes: [2: [2, 2], 4: [10, 10]], middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]])
let (reader, unet) = UNetXL(
  batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
  inputAttentionRes: [2: [2, 2], 4: [10, 10]], middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]])
graph.workspaceSize = 1_024 * 1_024 * 1_024
graph.withNoGrad {
  let _ = unetFixed(inputs: cTensor)
  readerFixed(state_dict)
  let condProj = Dense(count: 320, noBias: true)
  let wTensor = graph.variable(.GPU(1), .NC(2, 256), of: Float.self)
  condProj.compile(inputs: wTensor)
  let cond_proj_weight = state_dict["time_embedding.cond_proj.weight"].float().cpu().numpy()
  condProj.weight.copy(from: try! Tensor<Float>(numpy: cond_proj_weight))
  let kvs = unetFixed(inputs: cTensor).map { $0.as(of: Float.self) }
  let _ = unet(inputs: xTensor, [t_emb, yTensor] + kvs)
  reader(state_dict)
  let attnOut = unet(inputs: xTensor, [t_emb, yTensor] + kvs)[0].as(of: Float.self)
  debugPrint(attnOut)
  graph.openStore("/home/liu/workspace/swift-diffusion/lcm_sd_xl_base_1.0_f32.ckpt") {
    $0.write("w_cond_proj", model: condProj)
    $0.write("unet_fixed", model: unetFixed)
    $0.write("unet", model: unet)
  }
}
*/
