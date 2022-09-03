import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let ldm_util = Python.import("ldm.util")
let torch = Python.import("torch")
let omegaconf = Python.import("omegaconf")
let random = Python.import("random")
let numpy = Python.import("numpy")

func timeEmbedding(timesteps: Int, embeddingSize: Int, maxPeriod: Int) -> Tensor<Float> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .C(embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * Float(timesteps)
    embedding[i] = cos(freq)
    embedding[i + half] = sin(freq)
  }
  return embedding
}

func TimeEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

let config = omegaconf.OmegaConf.load(
  "/home/liu/workspace/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
let pl_sd = torch.load(
  "/home/liu/workspace/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
  map_location: "cpu")
let sd = pl_sd["state_dict"]
let model = ldm_util.instantiate_from_config(config.model)
model.load_state_dict(sd, strict: false)
model.eval()
let state_dict = model.model.state_dict()
let x = torch.randn([1, 4, 64, 64])
let t = torch.full([1], 981)
let c = torch.randn([1, 77, 768])
let ret = model.model.diffusion_model(x, t, c)

let graph = DynamicGraph()

let time_embed_0_weight = state_dict["diffusion_model.time_embed.0.weight"].numpy()
let time_embed_0_bias = state_dict["diffusion_model.time_embed.0.bias"].numpy()
let time_embed_2_weight = state_dict["diffusion_model.time_embed.2.weight"].numpy()
let time_embed_2_bias = state_dict["diffusion_model.time_embed.2.bias"].numpy()

let t_emb = graph.variable(timeEmbedding(timesteps: 981, embeddingSize: 320, maxPeriod: 10_000))
let (fc0, fc2, timeEmbed) = TimeEmbed(modelChannels: 320)
let _ = timeEmbed(inputs: t_emb)
fc0.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_0_weight))
fc0.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_0_bias))
fc2.parameters(for: .weight).copy(from: try! Tensor<Float>(numpy: time_embed_2_weight))
fc2.parameters(for: .bias).copy(from: try! Tensor<Float>(numpy: time_embed_2_bias))
let emb = timeEmbed(inputs: t_emb)[0].as(of: Float.self)
