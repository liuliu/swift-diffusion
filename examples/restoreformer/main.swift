import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let gfpgan = Python.import("gfpgan")
let torch = Python.import("torch")
let numpy = Python.import("numpy")
let random = Python.import("random")

let model = gfpgan.archs.restoreformer_arch.RestoreFormer()
let data = torch.load(
  "/home/liu/workspace/GFPGAN/gfpgan/weights/RestoreFormer.pth", map_location: "cpu")
let state_dict = data["params"]
model.load_state_dict(state_dict, strict: false)
print(state_dict.keys())
model.eval()

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let croppedFace = torch.randn([1, 3, 512, 512])
let ret = model(croppedFace)
print(ret)

/*
let compiledModel = torch.compile(model, backend: PythonFunction { (parameters: [PythonObject]) -> PythonConvertible in
  parameters[0].graph.print_tabular()
  return parameters[0].forward
})

compiledModel(croppedFace)
*/

func ResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = norm2(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    ninShortcut = nin
    out = nin(x) + out
  } else {
    ninShortcut = nil
    out = x + out
  }
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).norm1.weight"].numpy()
    let norm1_bias = state_dict["\(prefix).norm1.bias"].numpy()
    norm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let conv1_weight = state_dict["\(prefix).conv1.weight"].numpy()
    let conv1_bias = state_dict["\(prefix).conv1.bias"].numpy()
    conv1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv1_weight))
    conv1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].numpy()
    let norm2_bias = state_dict["\(prefix).norm2.bias"].numpy()
    norm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
    norm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let conv2_weight = state_dict["\(prefix).conv2.weight"].numpy()
    let conv2_bias = state_dict["\(prefix).conv2.bias"].numpy()
    conv2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv2_weight))
    conv2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv2_bias))
    if let ninShortcut = ninShortcut {
      let nin_shortcut_weight = state_dict["\(prefix).nin_shortcut.weight"].numpy()
      let nin_shortcut_bias = state_dict["\(prefix).nin_shortcut.bias"].numpy()
      ninShortcut.parameters(for: .weight).copy(
        from: try! Tensor<Float>(numpy: nin_shortcut_weight))
      ninShortcut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: nin_shortcut_bias))
    }
  }
  return (reader, Model([x], [out]))
}

func AttnBlock(
  prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int, numHeads: Int,
  crossAttention: Bool
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  let y: Model.IO?
  let y_: Model.IO
  let norm2: Model?
  if crossAttention {
    let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
    norm2 = norm
    let y0 = Input()
    y_ = norm(y0)
    y = y0
  } else {
    y_ = out
    y = nil
    norm2 = nil
  }
  let hw = width * height
  let attSize = inChannels / numHeads
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let k = tokeys(out).reshaped([batchSize * numHeads, attSize, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let q = ((1.0 / Float(attSize).squareRoot()) * toqueries(y_)).reshaped([
    batchSize * numHeads, attSize, hw,
  ])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * numHeads * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize * numHeads, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let v = tovalues(out).reshaped([batchSize * numHeads, attSize, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = x + projOut(out.reshaped([batchSize, inChannels, height, width]))
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).norm1.weight"].numpy()
    let norm1_bias = state_dict["\(prefix).norm1.bias"].numpy()
    norm1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm1_weight))
    norm1.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm1_bias))
    if let norm2 = norm2 {
      let norm2_weight = state_dict["\(prefix).norm2.weight"].numpy()
      let norm2_bias = state_dict["\(prefix).norm2.bias"].numpy()
      norm2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm2_weight))
      norm2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm2_bias))
    }
    let k_weight = state_dict["\(prefix).k.weight"].numpy()
    let k_bias = state_dict["\(prefix).k.bias"].numpy()
    tokeys.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: k_weight))
    tokeys.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: k_bias))
    let q_weight = state_dict["\(prefix).q.weight"].numpy()
    let q_bias = state_dict["\(prefix).q.bias"].numpy()
    toqueries.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: q_weight))
    toqueries.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: q_bias))
    let v_weight = state_dict["\(prefix).v.weight"].numpy()
    let v_bias = state_dict["\(prefix).v.bias"].numpy()
    tovalues.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: v_weight))
    tovalues.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: v_bias))
    let proj_out_weight = state_dict["\(prefix).proj_out.weight"].numpy()
    let proj_out_bias = state_dict["\(prefix).proj_out.bias"].numpy()
    projOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    projOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: proj_out_bias))
  }
  if let y = y {
    return (reader, Model([x, y], [out]))
  } else {
    return (reader, Model([x], [out]))
  }
}

func MultiHeadEncoder(
  ch: Int, chMult: [Int], zChannels: Int, numHeads: Int, numResBlocks: Int, x: Model.IO
) -> ((PythonObject) -> Void, [String: Model.IO]) {
  let convIn = Convolution(
    groups: 1, filters: ch, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var lastCh = ch
  var readers = [(PythonObject) -> Void]()
  var resolution = 512
  var outs = [String: Model.IO]()
  for (i, chM) in chMult.enumerated() {
    for j in 0..<numResBlocks {
      let (reader, block) = ResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", outChannels: ch * chM, shortcut: lastCh != ch * chM)
      lastCh = ch * chM
      readers.append(reader)
      out = block(out)
      if i == chMult.count - 1 {
        let (attnReader, attnBlock) = AttnBlock(
          prefix: "encoder.down.\(i).attn.\(j)", inChannels: lastCh, batchSize: 1,
          width: resolution, height: resolution, numHeads: numHeads, crossAttention: false)
        out = attnBlock(out)
        readers.append(attnReader)
      }
    }
    if i != chMult.count - 1 {
      outs["block_\(i)"] = out
      let downsample = Convolution(
        groups: 1, filters: lastCh, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])))
      resolution = resolution / 2
      out = downsample(out).reshaped(
        [1, lastCh, resolution, resolution], offset: [0, 0, 1, 1],
        strides: [
          lastCh * (resolution + 1) * (resolution + 1), (resolution + 1) * (resolution + 1),
          resolution + 1, 1,
        ])
      let reader: (PythonObject) -> Void = { state_dict in
        let downsample_weight = state_dict["encoder.down.\(i).downsample.conv.weight"].numpy()
        let downsample_bias = state_dict["encoder.down.\(i).downsample.conv.bias"].numpy()
        downsample.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: downsample_weight))
        downsample.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: downsample_bias))
      }
      readers.append(reader)
    }
  }
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: lastCh, shortcut: false)
  out = midBlock1(out)
  outs["block_\(chMult.count - 1)_attn"] = out
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: lastCh, batchSize: 1, width: resolution,
    height: resolution, numHeads: numHeads, crossAttention: false)
  out = midAttn1(out)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "encoder.mid.block_2", outChannels: lastCh, shortcut: false)
  out = midBlock2(out)
  outs["mid_attn"] = out
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: zChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  outs["out"] = out
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["encoder.conv_in.weight"].numpy()
    let conv_in_bias = state_dict["encoder.conv_in.bias"].numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    for reader in readers {
      reader(state_dict)
    }
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    let norm_out_weight = state_dict["encoder.norm_out.weight"].numpy()
    let norm_out_bias = state_dict["encoder.norm_out.bias"].numpy()
    normOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["encoder.conv_out.weight"].numpy()
    let conv_out_bias = state_dict["encoder.conv_out.bias"].numpy()
    convOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, outs)
}

func VectorQuantizer(nE: Int, eDim: Int, height: Int, width: Int) -> Model {
  let x = Input()
  let embedding = Input()
  var out = x.permuted(0, 2, 3, 1).reshaped([1 * height * width, eDim])
  let sum1 = (out .* out).reduced(.sum, axis: [1])
  let sum2 = (embedding .* embedding).reduced(.sum, axis: [1]).reshaped([1, nE])
  out = sum1 + (sum2 - 2 * Matmul(transposeB: (0, 1))(out, embedding))
  out = out.argmin(axis: 1)
  out = IndexSelect()(embedding, out.reshaped([height * width]))
  return Model([x, embedding], [out])
}

func MultiHeadDecoderTransformer(
  ch: Int, chMult: [Int], zChannels: Int, numHeads: Int, numResBlocks: Int, x: Model.IO,
  hs: [String: Model.IO]
) -> ((PythonObject) -> Void, Model.IO) {
  var lastCh = ch * chMult[chMult.count - 1]
  var resolution = 512
  for _ in 0..<chMult.count - 1 {
    resolution = resolution / 2
  }
  let convIn = Convolution(
    groups: 1, filters: lastCh, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  let (midBlockReader1, midBlock1) = ResnetBlock(
    prefix: "decoder.mid.block_1", outChannels: lastCh, shortcut: false)
  out = midBlock1(out)
  let (midAttnReader1, midAttn1) = AttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: lastCh, batchSize: 1, width: resolution,
    height: resolution, numHeads: numHeads, crossAttention: true)
  out = midAttn1(out, hs["mid_attn"]!)
  let (midBlockReader2, midBlock2) = ResnetBlock(
    prefix: "decoder.mid.block_2", outChannels: lastCh, shortcut: false)
  out = midBlock2(out)
  var readers = [(PythonObject) -> Void]()
  for (i, chM) in chMult.enumerated().reversed() {
    for j in 0..<numResBlocks + 1 {
      let (reader, block) = ResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", outChannels: ch * chM, shortcut: lastCh != ch * chM)
      lastCh = ch * chM
      readers.append(reader)
      out = block(out)
      if i == chMult.count - 1 {
        let (attnReader, attnBlock) = AttnBlock(
          prefix: "decoder.up.\(i).attn.\(j)", inChannels: lastCh, batchSize: 1, width: resolution,
          height: resolution, numHeads: numHeads, crossAttention: true)
        out = attnBlock(out, hs["block_\(i)_attn"]!)
        readers.append(attnReader)
      }
    }
    if i != 0 {
      let upsample = Upsample(.nearest, widthScale: 2, heightScale: 2)
      out = upsample(out)
      let conv = Convolution(
        groups: 1, filters: lastCh, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv(out)
      resolution = resolution * 2
      let reader: (PythonObject) -> Void = { state_dict in
        let conv_weight = state_dict["decoder.up.\(i).upsample.conv.weight"].numpy()
        let conv_bias = state_dict["decoder.up.\(i).upsample.conv.bias"].numpy()
        conv.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_weight))
        conv.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_bias))
      }
      readers.append(reader)
    }
  }
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv_in_weight = state_dict["decoder.conv_in.weight"].numpy()
    let conv_in_bias = state_dict["decoder.conv_in.bias"].numpy()
    convIn.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_in_weight))
    convIn.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_in_bias))
    midBlockReader1(state_dict)
    midAttnReader1(state_dict)
    midBlockReader2(state_dict)
    for reader in readers {
      reader(state_dict)
    }
    let norm_out_weight = state_dict["decoder.norm_out.weight"].numpy()
    let norm_out_bias = state_dict["decoder.norm_out.bias"].numpy()
    normOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    normOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: norm_out_bias))
    let conv_out_weight = state_dict["decoder.conv_out.weight"].numpy()
    let conv_out_bias = state_dict["decoder.conv_out.bias"].numpy()
    convOut.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv_out_weight))
    convOut.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: conv_out_bias))
  }
  return (reader, out)
}

func RestoreFormer(
  nEmbed: Int, embedDim: Int, ch: Int, chMult: [Int], zChannels: Int, numHeads: Int,
  numResBlocks: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let embedding = Input()
  let (encoderReader, encoderOuts) = MultiHeadEncoder(
    ch: ch, chMult: chMult, zChannels: zChannels, numHeads: numHeads, numResBlocks: numResBlocks,
    x: x)
  var out = encoderOuts["out"]!
  let quantConv = Convolution(
    groups: 1, filters: embedDim, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = quantConv(out)
  let vq = VectorQuantizer(nE: nEmbed, eDim: embedDim, height: 16, width: 16)
  out = vq(out, embedding)
  out = out.transposed(0, 1).reshaped([1, embedDim, 16, 16])
  let postQuantConv = Convolution(
    groups: 1, filters: zChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = postQuantConv(out)
  let (decoderReader, decoderOut) = MultiHeadDecoderTransformer(
    ch: ch, chMult: chMult, zChannels: zChannels, numHeads: numHeads, numResBlocks: numResBlocks,
    x: out, hs: encoderOuts)
  out = decoderOut
  let reader: (PythonObject) -> Void = { state_dict in
    encoderReader(state_dict)
    let quant_conv_weight = state_dict["quant_conv.weight"].numpy()
    let quant_conv_bias = state_dict["quant_conv.bias"].numpy()
    quantConv.weight.copy(from: try! Tensor<Float>(numpy: quant_conv_weight))
    quantConv.bias.copy(from: try! Tensor<Float>(numpy: quant_conv_bias))
    let post_quant_conv_weight = state_dict["post_quant_conv.weight"].numpy()
    let post_quant_conv_bias = state_dict["post_quant_conv.bias"].numpy()
    postQuantConv.weight.copy(from: try! Tensor<Float>(numpy: post_quant_conv_weight))
    postQuantConv.bias.copy(from: try! Tensor<Float>(numpy: post_quant_conv_bias))
    decoderReader(state_dict)
  }
  return (reader, Model([x, embedding], [out]))
}

let graph = DynamicGraph()
let croppedFaceTensor = graph.variable(try! Tensor<Float>(numpy: croppedFace.numpy())).toGPU(0)
graph.workspaceSize = 1_024 * 1_024 * 1_024
let (reader, restoreFormer) = RestoreFormer(
  nEmbed: 1024, embedDim: 256, ch: 64, chMult: [1, 2, 2, 4, 4, 8], zChannels: 256, numHeads: 8,
  numResBlocks: 2)
graph.withNoGrad {
  let embedding_weight = state_dict["quantize.embedding.weight"].numpy()
  let embeddingTensor = graph.variable(try! Tensor<Float>(numpy: embedding_weight)).toGPU(0)
  restoreFormer.compile(inputs: croppedFaceTensor, embeddingTensor)
  reader(state_dict)
  let result = restoreFormer(inputs: croppedFaceTensor, embeddingTensor)[0].as(of: Float.self)
  debugPrint(result)
}
