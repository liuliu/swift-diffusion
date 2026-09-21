import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let sys = Python.import("sys")
let site = Python.import("site")
for path in [String(site.getusersitepackages())!, "/usr/lib/python3/dist-packages"] {
  if !(Bool(sys.path.__contains__(path)) ?? false) { sys.path.append(path) }
}
let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
sys.path.insert(0, directory)
let reference = Python.import("reference")
let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)
let device = 1
let pythonDevice = 0
let textCheckpoint = "/slow/Data/qwen_image_2.1_qwen_3_vl_8b.ckpt"
let ditCheckpoint = "/slow/Data/qwen_image_2.1_dit_f16.ckpt"
let vaeCheckpoint = "/slow/Data/qwen_image_2.1_vae_f32.ckpt"

func cpu(_ object: PythonObject) -> Tensor<Float> { try! Tensor<Float>(numpy: object) }

// Run generation first; editing uses its image unless you change the path in edit.swift.
switch CommandLine.arguments.dropFirst().first ?? "generate" {
case "generate": q21Generate()
case "edit": q21Edit()
case "--help", "help": print("Usage: bash examples/qwen_image21/run.sh [generate|edit]")
default: fatalError("Expected generate or edit")
}
