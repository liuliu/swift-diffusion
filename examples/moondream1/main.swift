import C_ccv
import Diffusion
import Foundation
import NNC
import PNG

public enum PythonObject {}
public typealias FloatType = Float16

func SigLIPSelfAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
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

func SigLIPResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int, MLP: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2])
  let (_, _, _, _, attention) = SigLIPSelfAttention(
    k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [1])
  let fc = Dense(count: MLP)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x], [out]))
}

func SigLIPVisionTransformer(
  gridX: Int, gridY: Int, width: Int, layers: Int, heads: Int, MLP: Int, batchSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let posEmbed = Parameter<FloatType>(
    .GPU(0), .CHW(1, 27 * 27, width), initBound: 1, name: "pos_embed")
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, gridX * gridY]).transposed(1, 2)
  out = out + posEmbed
  for i in 0..<layers {
    let (_, block) = SigLIPResidualAttentionBlock(
      prefix: "model.encoder.model.visual.blocks.\(i)", k: width / heads, h: heads, b: batchSize,
      t: gridX * gridY, MLP: MLP)
    out = block(out.reshaped([batchSize, gridX * gridY, width]))
  }
  let lnPost = LayerNorm(epsilon: 1e-6, axis: [1])
  out = lnPost(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x], [out]))
}

func MoondreamVisionProjection() -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let mlp1fc = Dense(count: 2048 * 4)
  let mlp1gelu = GELU()
  let mlp1proj = Dense(count: 2048)
  var out = mlp1proj(mlp1gelu(mlp1fc(x)))
  let ln = LayerNorm(epsilon: 1e-5, axis: [1])
  out = ln(out)
  let mlp2fc = Dense(count: 2048 * 4)
  let mlp2gelu = GELU()
  let mlp2proj = Dense(count: 2048)
  out = out + mlp2proj(mlp2gelu(mlp2fc(out)))
  let reader: (PythonObject) -> Void = { _ in
  }
  return (reader, Model([x], [out]))
}

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int), rotaryDim: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let costheta = Input()
  let sintheta = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
  let kIn = Input()
  let vIn = Input()
  var keys = tokeys(x).reshaped([b, t.1, hk, k])
  var queries = toqueries(x).reshaped([b, t.1, h, k])
  let values = tovalues(x).reshaped([b, t.1, hk, k])
  let keysRot0 = keys.reshaped(
    [b, t.1, hk, rotaryDim / 2], offset: [0, 0, 0, 0], strides: [t.1 * hk * k, hk * k, k, 1])
  let keysRot1 = keys.reshaped(
    [b, t.1, hk, rotaryDim / 2], offset: [0, 0, 0, rotaryDim / 2],
    strides: [t.1 * hk * k, hk * k, k, 1])
  let keysPass = keys.reshaped(
    [b, t.1, hk, k - rotaryDim], offset: [0, 0, 0, rotaryDim],
    strides: [t.1 * hk * k, hk * k, k, 1])
  let queriesRot0 = queries.reshaped(
    [b, t.1, h, rotaryDim / 2], offset: [0, 0, 0, 0], strides: [t.1 * h * k, h * k, k, 1])
  let queriesRot1 = queries.reshaped(
    [b, t.1, h, rotaryDim / 2], offset: [0, 0, 0, rotaryDim / 2],
    strides: [t.1 * h * k, h * k, k, 1])
  let queriesPass = queries.reshaped(
    [b, t.1, h, k - rotaryDim], offset: [0, 0, 0, rotaryDim], strides: [t.1 * h * k, h * k, k, 1])
  queries = Functional.concat(
    axis: 3, queriesRot0 .* costheta - queriesRot1 .* sintheta,
    queriesRot0 .* sintheta + queriesRot1 .* costheta, queriesPass)
  keys = Functional.concat(
    axis: 3, keysRot0 .* costheta - keysRot1 .* sintheta,
    keysRot0 .* sintheta + keysRot1 .* costheta, keysPass)
  let kOut = Functional.concat(axis: 1, kIn, keys)
  let vOut = Functional.concat(axis: 1, vIn, values)
  var out = ScaledDotProductAttention(
    scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true)(
      queries, kOut, vOut, causalAttentionMask
    ).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: k * h, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (
    Model([x, costheta, sintheta, causalAttentionMask, kIn, vIn], [out, kOut, vOut]), reader
  )
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (Model, Model, Model)
{
  let x = Input()
  let w1 = Dense(count: intermediateSize)
  var out = GELU()(w1(x))
  let w2 = Dense(count: hiddenSize)
  out = w2(out)
  return (w1, w2, Model([x], [out], name: name))
}

func TransformerBlock(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int), MLP: Int, rotaryDim: Int
) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let costheta = Input()
  let sintheta = Input()
  let causalAttentionMask = Input()
  let kIn = Input()
  let vIn = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let (attention, _) = SelfAttention(
    prefix: prefix, k: k, h: h, hk: hk, b: b, t: t, rotaryDim: rotaryDim)
  let residual = out
  let tuple = attention(out, costheta, sintheta, causalAttentionMask, kIn, vIn)
  out = tuple[0] + x
  let kOut = tuple[1]
  let vOut = tuple[2]
  let (_, _, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "ffn")
  out = out + ffn(residual)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (
    Model([x, costheta, sintheta, causalAttentionMask, kIn, vIn], [out, kOut, vOut]), reader
  )
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: Int, cachedTokenLength: Int,
  layers: Int, MLP: Int, rotaryDim: Int, heads: Int, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let textEmb = Input()
  let costheta = Input()
  let sintheta = Input()
  let causalAttentionMask = Input()
  var kvs = [Input]()
  var kvOuts = [Model.IO]()
  var out: Model.IO = textEmb
  for i in 0..<layers {
    let (layer, _) = TransformerBlock(
      prefix: "model.transformer.h.\(i)", k: width / heads, h: heads, hk: heads, b: batchSize,
      t: (cachedTokenLength + tokenLength, tokenLength), MLP: MLP, rotaryDim: rotaryDim)
    let kIn = Input()
    let vIn = Input()
    let tuple = layer(out, costheta, sintheta, causalAttentionMask, kIn, vIn)
    out = tuple[0]
    kvs.append(kIn)
    kvs.append(vIn)
    kvOuts.append(tuple[1])
    kvOuts.append(tuple[2])
  }
  let norm = LayerNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  let output = Dense(count: vocabularySize, name: "output")
  out = output(out)
  let reader: (PythonObject) -> Void = { _ in
  }
  return (Model([textEmb, costheta, sintheta, causalAttentionMask] + kvs, [out] + kvOuts), reader)
}

let u8Img = ccv_dense_matrix_new(512, 512, Int32(CCV_8U | CCV_C3), nil, 0)!
if let image = try PNG.Data.Rectangular.decompress(
  path: "/home/liu/workspace/swift-diffusion/kandinsky-512.png")
{
  let rgba = image.unpack(as: PNG.RGBA<UInt8>.self)
  for y in 0..<512 {
    for x in 0..<512 {
      let pixel = rgba[y * 512 + x]
      u8Img.pointee.data.u8[y * 512 * 3 + x * 3] = pixel.r
      u8Img.pointee.data.u8[y * 512 * 3 + x * 3 + 1] = pixel.g
      u8Img.pointee.data.u8[y * 512 * 3 + x * 3 + 2] = pixel.b
    }
  }
}

var clipImg = Tensor<FloatType>(.CPU, .NCHW(1, 3, 378, 378))
var smallerImg: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
ccv_resample(u8Img, &smallerImg, 0, 378 / 512, 378 / 512, Int32(CCV_INTER_AREA))
ccv_matrix_free(u8Img)
for y in 0..<378 {
  for x in 0..<378 {
    let (r, g, b) = (
      smallerImg!.pointee.data.u8[y * 1136 + x * 3],
      smallerImg!.pointee.data.u8[y * 1136 + x * 3 + 1],
      smallerImg!.pointee.data.u8[y * 1136 + x * 3 + 2]
    )
    clipImg[0, 0, y, x] = FloatType((Float(r) / 255 - 0.5) / 0.5)
    clipImg[0, 1, y, x] = FloatType((Float(g) / 255 - 0.5) / 0.5)
    clipImg[0, 2, y, x] = FloatType((Float(b) / 255 - 0.5) / 0.5)
  }
}
ccv_matrix_free(smallerImg)

let graph = DynamicGraph()

graph.withNoGrad {
  let (_, vit) = SigLIPVisionTransformer(
    gridX: 27, gridY: 27, width: 1152, layers: 27, heads: 16, MLP: 4304, batchSize: 1)

  let input = graph.variable(clipImg.toGPU(0))
  vit.compile(inputs: input)
  graph.openStore("/home/liu/workspace/swift-diffusion/siglip_384_f32.ckpt") {
    $0.read("vit", model: vit)
  }
  var out = vit(inputs: input)[0].as(of: FloatType.self)

  let (_, proj) = MoondreamVisionProjection()

  proj.compile(inputs: out)
  var textEmbTensor: AnyTensor? = nil
  graph.openStore("/home/liu/workspace/swift-diffusion/moondream1_f32.ckpt") {
    $0.read("vision_proj", model: proj)
    textEmbTensor = $0.read("text_emb")
  }
  out = proj(inputs: out)[0].as(of: FloatType.self)
  let textEmb = graph.variable(
    Tensor<FloatType>(from: textEmbTensor!).toGPU(0).reshaped(
      .WC(textEmbTensor!.shape[0], textEmbTensor!.shape[1])))
  let tokenizer = GPT2Tokenizer(
    vocabulary: "/home/liu/workspace/swift-diffusion/examples/moondream1/vocab.json",
    merges: "/home/liu/workspace/swift-diffusion/examples/moondream1/merges.txt")
  let eos = "<END>"
  let before = tokenizer.tokenize(text: "<image>", addSpecialTokens: true)
  let beforeTensor = Tensor<Int32>(before, .CPU, .C(before.count))
  let beforeEmb = Functional.indexSelect(
    input: textEmb, index: graph.variable(beforeTensor.toGPU(0)))
  let after = tokenizer.tokenize(
    text:
      "</image>\n\nQuestion: Describe this image and its style in a very detailed manner, follow the format of describing: what, who, where, when, how. You don't need to fill in all if they are irrelevant. Please remove What, Who, Where, When, How prefixes and make it one paragraph.\n\nAnswer:",
    addSpecialTokens: false)
  let afterTensor = Tensor<Int32>(after, .CPU, .C(after.count))
  let afterEmb = Functional.indexSelect(input: textEmb, index: graph.variable(afterTensor.toGPU(0)))
  var inputEmb = graph.variable(
    .GPU(0), .WC(beforeEmb.shape[0] + out.shape[0] + afterEmb.shape[0], 2048), of: FloatType.self)
  inputEmb[0..<beforeEmb.shape[0], 0..<2048] = beforeEmb
  inputEmb[beforeEmb.shape[0]..<(beforeEmb.shape[0] + out.shape[0]), 0..<2048] = out
  inputEmb[(beforeEmb.shape[0] + out.shape[0])..<inputEmb.shape[0], 0..<2048] = afterEmb

  let seqLen = inputEmb.shape[0]

  let phi = ModelBuilder { (tokenLengths: (cachedTokenLength: Int, tokenLength: Int), _) in
    return Transformer(
      FloatType.self, vocabularySize: 51_200, width: 2048, tokenLength: tokenLengths.tokenLength,
      cachedTokenLength: tokenLengths.cachedTokenLength, layers: 24, MLP: 2048 * 4,
      rotaryDim: 32, heads: 32, batchSize: 1
    ).0
  }
  graph.maxConcurrency = .limit(1)
  phi.maxConcurrency = .limit(1)
  var kvs = (0..<48).map { _ in
    graph.variable(.GPU(0), format: .NHWC, shape: [], of: FloatType.self)
  }
  var costhetaTensor = graph.variable(.CPU, .NHWC(1, seqLen, 1, 16), of: Float.self)
  var sinthetaTensor = graph.variable(.CPU, .NHWC(1, seqLen, 1, 16), of: Float.self)
  for i in 0..<inputEmb.shape[0] {
    for k in 0..<16 {
      let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 32)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      costhetaTensor[0, i, 0, k] = Float(costheta)
      sinthetaTensor[0, i, 0, k] = Float(sintheta)
    }
  }
  var causalAttentionMask = graph.variable(
    .CPU, .NHWC(1, 1, seqLen, seqLen), of: FloatType.self)
  causalAttentionMask.full(0)
  for i in 0..<(seqLen - 1) {
    for j in (i + 1)..<seqLen {
      causalAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
    }
  }
  var causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
  var costhetaTensorGPU = DynamicGraph.Tensor<FloatType>(from: costhetaTensor).toGPU(0)
  var sinthetaTensorGPU = DynamicGraph.Tensor<FloatType>(from: sinthetaTensor).toGPU(0)
  phi.compile(
    (cachedTokenLength: 0, tokenLength: inputEmb.shape[0]),
    inputs: [inputEmb, costhetaTensorGPU, sinthetaTensorGPU, causalAttentionMask] + kvs)
  graph.openStore("/home/liu/workspace/swift-diffusion/moondream1_f32.ckpt") {
    $0.read("phi", model: phi)
  }
  var tuple = phi(
    (cachedTokenLength: 0, tokenLength: inputEmb.shape[0]), inputs: inputEmb,
    [costhetaTensorGPU, sinthetaTensorGPU, causalAttentionMaskGPU] + kvs
  ).map { $0.as(of: FloatType.self) }
  var nextToken = tuple[0].rawValue.toCPU()
  var topV = nextToken[seqLen - 1, 0]
  var topK = 0
  for i in 1..<51_200 {
    if nextToken[seqLen - 1, i] > topV {
      topV = nextToken[seqLen - 1, i]
      topK = i
    }
  }
  var ids = [Int32(topK)]
  print(ids)
  kvs = Array(tuple[1...])
  let kvs100 = kvs.map {
    let v = graph.variable(
      .GPU(0), .NHWC($0.shape[0], 1024, $0.shape[2], $0.shape[3]), of: FloatType.self)
    v.full(0)
    return v
  }
  causalAttentionMask = graph.variable(
    .CPU, .NHWC(1, 1, 1, 1025), of: FloatType.self)
  causalAttentionMask.full(0)
  causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
  phi.compile(
    (cachedTokenLength: 1024, tokenLength: 1),
    inputs: [
      inputEmb.reshaped(.WC(1, 2048)), costhetaTensorGPU.reshaped(.NHWC(1, 1, 1, 16)),
      sinthetaTensorGPU.reshaped(.NHWC(1, 1, 1, 16)), causalAttentionMaskGPU,
    ] + kvs100, isEager: true)
  let maxTokens = 128
  var output = tokenizer.decode(ids)
  for _ in 0..<maxTokens {
    let tokensTensor = graph.variable(.CPU, format: .NHWC, shape: [1], of: Int32.self)
    tokensTensor[0] = Int32(topK)
    let inputEmb = Functional.indexSelect(input: textEmb, index: tokensTensor.toGPU(0))
    costhetaTensor = graph.variable(.CPU, .NHWC(1, 1, 1, 16), of: Float.self)
    sinthetaTensor = graph.variable(.CPU, .NHWC(1, 1, 1, 16), of: Float.self)
    let cachedTokenLength = kvs[0].shape[1]
    for k in 0..<16 {
      let theta = Double(cachedTokenLength) * 1.0 / pow(10_000, Double(k) * 2 / 32)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      costhetaTensor[0, 0, 0, k] = Float(costheta)
      sinthetaTensor[0, 0, 0, k] = Float(sintheta)
    }
    causalAttentionMask = graph.variable(
      .CPU, .NHWC(1, 1, 1, cachedTokenLength + 1), of: FloatType.self)
    causalAttentionMask.full(0)
    causalAttentionMaskGPU = causalAttentionMask.toGPU(0)
    costhetaTensorGPU = DynamicGraph.Tensor<FloatType>(from: costhetaTensor).toGPU(0)
    sinthetaTensorGPU = DynamicGraph.Tensor<FloatType>(from: sinthetaTensor).toGPU(0)
    tuple = phi(
      (cachedTokenLength: cachedTokenLength, tokenLength: 1), inputs: inputEmb,
      [costhetaTensorGPU, sinthetaTensorGPU, causalAttentionMaskGPU] + kvs
    ).map { $0.as(of: FloatType.self) }
    kvs = Array(tuple[1...])
    nextToken = tuple[0].rawValue.toCPU()
    topV = nextToken[0, 0]
    topK = 0
    for i in 1..<51_200 {
      if nextToken[0, i] > topV {
        topK = i
        topV = nextToken[0, i]
      }
    }
    ids.append(Int32(topK))
    output += tokenizer.decode([Int32(topK)])
    print(output)
    if output.hasSuffix(eos) {
      break
    }
  }
}
