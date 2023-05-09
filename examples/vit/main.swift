import NNC
import NNCPythonConversion
import PythonKit

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

func VisionTransformer(
  grid: Int, width: Int, outputDim: Int, layers: Int, heads: Int, batchSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let classEmbedding = Input()
  let positionalEmbedding = Input()
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], noBias: true,
    hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPre(out)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (reader, block) = CLIPResidualAttentionBlock(
      prefix: "transformer.resblocks.\(i)", k: width / heads, h: heads, b: batchSize,
      t: grid * grid + 1)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
    readers.append(reader)
  }
  let lnPost = LayerNorm(epsilon: 1e-5, axis: [1])
  out = lnPost(out.reshaped([batchSize, width], strides: [width, 1]))
  let proj = Dense(count: outputDim, noBias: true)
  out = proj(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1_weight = state_dict["conv1.weight"].type(torch.float).cpu().numpy()
    conv1.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let ln_pre_weight = state_dict["ln_pre.weight"].type(torch.float).cpu().numpy()
    let ln_pre_bias = state_dict["ln_pre.bias"].type(torch.float).cpu().numpy()
    lnPre.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_pre_weight))
    lnPre.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_pre_bias))
    for reader in readers {
      reader(state_dict)
    }
    let ln_post_weight = state_dict["ln_post.weight"].type(torch.float).cpu().numpy()
    let ln_post_bias = state_dict["ln_post.bias"].type(torch.float).cpu().numpy()
    lnPost.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: ln_post_weight))
    lnPost.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: ln_post_bias))
    let proj_weight = state_dict["proj"].type(torch.float).cpu().numpy()
    // Somehow I have problems with transposed numpy array.
    var projTensor = Tensor<Float>(.CPU, .NC(outputDim, width))
    for i in 0..<outputDim {
      for j in 0..<width {
        projTensor[i, j] = Float(proj_weight[j, i])!
      }
    }
    proj.parameters(for: .weight).copy(from: projTensor)
  }
  return (reader, Model([x, classEmbedding, positionalEmbedding], [out]))
}

let clip = Python.import("clip")
let torch = Python.import("torch")
let random = Python.import("random")
let numpy = Python.import("numpy")
let transformers = Python.import("transformers")

let transformer = transformers.CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let model = clip.load("ViT-L/14").tuple2.0

let state_dict = model.visual.state_dict()
print(state_dict.keys())

let x = torch.randn([1, 3, 224, 224])
let y = model.visual(x.cuda().type(model.dtype))
print(y)
let text_proj = model.text_projection.type(torch.float)
var textProj = Tensor<Float>(.CPU, .NC(768, 768))
for i in 0..<768 {
  for j in 0..<768 {
    textProj[i, j] = Float(text_proj[i, j])!
  }
}

let graph = DynamicGraph()
let xTensor = graph.variable(try! Tensor<Float>(numpy: x.numpy())).toGPU(0)
let (reader, vit) = VisionTransformer(
  grid: 16, width: 1024, outputDim: 768, layers: 24, heads: 16, batchSize: 1)
graph.workspaceSize = 1_024 * 1_024 * 1_024
graph.withNoGrad {
  let class_embedding = state_dict["class_embedding"].type(torch.float).cpu().numpy()
  let classEmbedding = graph.variable(try! Tensor<Float>(numpy: class_embedding)).reshaped(
    .CHW(1, 1, 1024)
  ).toGPU(0)
  let positional_embedding = state_dict["positional_embedding"].type(torch.float).cpu().numpy()
  let positionalEmbedding = graph.variable(try! Tensor<Float>(numpy: positional_embedding))
    .reshaped(.CHW(1, 16 * 16 + 1, 1024)).toGPU(0)
  let _ = vit(inputs: xTensor, classEmbedding, positionalEmbedding)
  reader(state_dict)
  // DynamicGraph.logLevel = .verbose
  let out = vit(inputs: xTensor, classEmbedding, positionalEmbedding)[0].as(of: Float.self).toCPU()
  for j in 0..<768 {
    print("\(j) \(out[0, j])")
  }
  let textProj = graph.variable(textProj)

  graph.openStore("/home/liu/workspace/swift-diffusion/image_model.ckpt") {
    $0.write("vit", model: vit)
    $0.write("class_embedding", variable: classEmbedding)
    $0.write("positional_embedding", variable: positionalEmbedding)
    $0.write("text_proj", variable: textProj)
  }
}
