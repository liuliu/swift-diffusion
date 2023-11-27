import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let streamlit_helpers = Python.import("scripts.demo.streamlit_helpers")
let PIL = Python.import("PIL")
let numpy = Python.import("numpy")
let torch = Python.import("torch")
let pytorch_lightning = Python.import("pytorch_lightning")
let torchvision = Python.import("torchvision")

let version_dict: [String: PythonObject] = [
  "T": 14,
  "H": 512,
  "W": 512,
  "C": 4, "f": 8,
  "config": "/home/liu/workspace/generative-models/configs/inference/svd.yaml",
  "ckpt": "/home/liu/workspace/generative-models/checkpoints/svd.safetensors",
  "options": [
    "discretization": 1,
    "cfg": 2.5,
    "sigma_min": 0.002,
    "sigma_max": 700.0,
    "rho": 7.0,
    "guider": 2,
    "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
    "num_steps": 25,
  ],
]

let state = streamlit_helpers.init_st(version_dict, load_filter: false)

func load_img_for_prediction(path: String, W: Int, H: Int) -> PythonObject {
  var image = PIL.Image.open(path)
  if image.mode != "RGB" {
    image = image.convert("RGB")
  }
  let (width, height) = image.size.tuple2
  let rfs = Double(
    streamlit_helpers.get_resizing_factor(
      PythonObject(tupleOf: W, H), PythonObject(tupleOf: width, height)))!
  let resize_size = (
    Int((Double(height)! * rfs).rounded(.up)), Int((Double(width)! * rfs).rounded(.up))
  )
  let top = (resize_size.0 - H) / 2
  let left = (resize_size.1 - W) / 2
  image = numpy.array(image).transpose(2, 0, 1)
  image = torch.from_numpy(image).to(dtype: torch.float32) / 255.0
  image = image.unsqueeze(0)
  image = torch.nn.functional.interpolate(
    image, PythonObject(tupleOf: resize_size.0, resize_size.1), mode: "area", antialias: false)
  image = torchvision.transforms.functional.crop(image, top: top, left: left, height: H, width: W)
  return image * 2 - 1
}
let img = load_img_for_prediction(
  path: "/home/liu/workspace/swift-diffusion/kandinsky.png", W: 512, H: 512
).cuda()
let ukeys = Python.set(
  streamlit_helpers.get_unique_embedder_keys_from_conditioner(state["model"].conditioner))
var value_dict = streamlit_helpers.init_embedder_options(ukeys, Python.dict())
pytorch_lightning.seed_everything(23)
value_dict["image_only_indicator"] = 0
value_dict["cond_frames_without_noise"] = img
value_dict["cond_frames"] = img + 0.02 * torch.randn_like(img)
value_dict["cond_aug"] = 0.02

var options = version_dict["options"]!
options["num_frames"] = 14
let (sampler, num_rows, num_cols) = streamlit_helpers.init_sampling(options: options).tuple3
let num_samples = num_rows * num_cols
let sample = streamlit_helpers.do_sample(
  state["model"], sampler, value_dict, num_samples, 512, 512, 4, 8,
  T: 14, batch2model_input: ["num_video_frames", "image_only_indicator"],
  force_uc_zero_embeddings: options["force_uc_zero_embeddings"], return_latents: false,
  decoding_t: 1)
// streamlit_helpers.save_video_as_grid_and_mp4(sample, "/home/liu/workspace/swift-diffusion/outputs/", 14, fps: value_dict["fps"])
