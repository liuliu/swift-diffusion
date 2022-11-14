import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let basicsr = Python.import("basicsr")
let torch = Python.import("torch")
let numpy = Python.import("numpy")

let model = basicsr.archs.rrdbnet_arch.RRDBNet(
  num_in_ch: 3, num_out_ch: 3, num_feat: 64, num_block: 23, num_grow_ch: 32, scale: 4)
let esrgan = torch.load(
  "/home/liu/workspace/Real-ESRGAN/weights/RealESRGAN_x4plus.pth",
  map_location: "cpu")
let state_dict = esrgan["params_ema"]
model.load_state_dict(state_dict, strict: false)
print(state_dict.keys())
