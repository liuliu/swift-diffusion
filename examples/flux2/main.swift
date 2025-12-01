import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

typealias FloatType = Float

let torch = Python.import("torch")
let PIL = Python.import("PIL")

torch.set_grad_enabled(false)

let torch_device = torch.device("cuda")

let random = Python.import("random")
let numpy = Python.import("numpy")

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let flux2_util = Python.import("flux2.util")

var mistral = flux2_util.load_mistral_small_embedder()
let model = flux2_util.load_flow_model(model_name: "flux.2-dev", debug_mode: false, device: "cpu")

// print(mistral)
print(model)
mistral = mistral.cpu()

let x = torch.randn([1, 4096, 128]).to(torch.bfloat16).cuda()
let txt = torch.randn([1, 512, 15360]).to(torch.bfloat16).cuda() * 0.01
var img_ids = torch.zeros([64, 64, 4])
img_ids[..., ..., 1] = img_ids[..., ..., 1] + torch.arange(64)[..., Python.None]
img_ids[..., ..., 2] = img_ids[..., ..., 2] + torch.arange(64)[Python.None, ...]
img_ids = img_ids.reshape([1, 4096, 4]).cuda()
var txt_ids = torch.zeros([512, 4])
txt_ids[..., 3] = txt_ids[..., 3] + torch.arange(512)
txt_ids = txt_ids.reshape([1, 512, 4]).cuda()
let t = torch.full([1], 1).to(torch.bfloat16).cuda()
let guidance = torch.full([1], 4.0).to(torch.bfloat16).cuda()

let output = model(x, img_ids, t, txt, txt_ids, guidance)

let state_dict = model.state_dict()
print(state_dict.keys())

func timeEmbedding(timesteps: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<
  Float
> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timesteps
    let cosFreq = cos(freq)
    let sinFreq = sin(freq)
    for j in 0..<batchSize {
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

func Flux2(b: Int, h: Int, w: Int, guidanceEmbed: Bool) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let contextIn = Input()
  let xEmbedder = Dense(count: 6144, noBias: true, name: "x_embedder")
  var out = xEmbedder(x)
  let contextEmbedder = Dense(count: 6144, noBias: true, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  let reader: (PythonObject) -> Void = { state_dict in
    let img_in_weight = state_dict["img_in.weight"].to(
      torch.float
    ).cpu().numpy()
    xEmbedder.weight.copy(from: try! Tensor<Float>(numpy: img_in_weight))
    let txt_in_weight = state_dict["txt_in.weight"].to(
      torch.float
    ).cpu().numpy()
    contextEmbedder.weight.copy(from: try! Tensor<Float>(numpy: txt_in_weight))
  }
  return (reader, Model([x, contextIn], [out, context]))
}

let (reader, dit) = Flux2(b: 1, h: 64, w: 64, guidanceEmbed: true)

let graph = DynamicGraph()

graph.maxConcurrency = .limit(1)

graph.withNoGrad {
  let xTensor = graph.variable(try! Tensor<Float>(numpy: x.to(torch.float).cpu().numpy()).toGPU(1))
  let contextTensor = graph.variable(
    try! Tensor<Float>(numpy: txt.to(torch.float).cpu().numpy()).toGPU(1))
  dit.maxConcurrency = .limit(1)
  dit.compile(inputs: xTensor, contextTensor)
  reader(state_dict)
  debugPrint(dit(inputs: xTensor, contextTensor))
}

exit(0)
