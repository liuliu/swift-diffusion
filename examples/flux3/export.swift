import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

// Preserve the original values in these underflow-sensitive projections.
let flux3FP32StorageWeights = [
  "time_in.out_layer.weight": "__dit__[t-t_2-0-0]",
  "ref_time_in.out_layer.weight": "__dit__[t-next_t_2-0-0]",
  "final_layer.image.adaLN_modulation.1.weight": "__dit__[t-final_modulation-0-0]",
]

// Export is an explicit mode; normal parity runs never write checkpoint weights.
let exportingFP16 = CommandLine.arguments.dropFirst().first == "export-dit-f16"
let exporting =
  exportingFP16 || ["export", "export-dit"].contains(CommandLine.arguments.dropFirst().first ?? "")
let checkpoints: Flux3Checkpoints = {
  var environment = env
  if exportingFP16 && environment["FLUX3_DIT_CHECKPOINT"] == nil {
    environment["FLUX3_DIT_CHECKPOINT"] = "/slow/Data/flux_3_image_klein_dit_f16.ckpt"
  }
  return Flux3Checkpoints(environment: environment)
}()

func exportAndVerify<T: TensorNumeric>(
  _ type: T.Type, key: String, model: Model, fresh: Model,
  inputs: [DynamicGraph.AnyTensor], expected: [DynamicGraph.AnyTensor], fp16Storage: Bool = false
) {
  let exportStorePath = checkpoints.path(for: key) + ".partial"
  print("Exporting", key, "to", exportStorePath)
  try! graph.openStore(exportStorePath) { store in
    if fp16Storage {
      // Validate FP16-rounded weights with FP32 arithmetic before exporting.
      // FP16 activation scaling is a separate change; the unscaled residuals overflow.
      try! store.write(key, model: model, strict: true) { name, tensor in
        let value = Tensor<Float>(from: tensor).toCPU()
        if flux3FP32StorageWeights.values.contains(name) {
          try! store.write(name, tensor: value, strict: true)
        } else {
          try! store.write(name, tensor: Tensor<Float16>(from: value), strict: true)
        }
        return .skip
      }
    } else {
      try! store.write(key, model: model, strict: true)
    }
  }.get()
  fresh.maxConcurrency = .limit(1)
  fresh.compile(inputs: inputs)
  try! graph.openStore(exportStorePath, flags: .readOnly) {
    try! $0.read(key, model: fresh, strict: true)
  }.get()
  let outputs = fresh(inputs: inputs[0], Array(inputs.dropFirst()))
  precondition(outputs.count == expected.count)
  for i in outputs.indices {
    let before = Tensor<Float>(from: expected[i].as(of: T.self).rawValue.toCPU())
    compare(
      "checkpoint/\(key)/\(i)", outputs[i].as(of: T.self), before.makeNumpyArray(), tolerance: 1e-7)
  }
}

func verifyDiTStoreNames() {
  var expected = Set(
    [
      "t_0", "t_2", "next_t_0", "next_t_2",
      "context_refiner_modulation", "x_refiner_modulation", "ref_refiner_modulation",
      "context_modulation", "x_modulation", "ref_modulation",
      "context_embedder", "x_embedder", "ref_embedder", "final_modulation", "linear",
    ].map { "__dit__[t-\($0)-0-0]" })
  for (name, count) in [
    ("context_refiner", Flux3Config.refinerLayers), ("x_refiner", Flux3Config.refinerLayers),
    ("ref_refiner", Flux3Config.refinerLayers), ("x", Flux3Config.layers),
  ] {
    for suffix in ["q", "k", "v", "norm_q", "norm_k", "o", "w1", "w3", "w2"] {
      for index in 0..<count {
        expected.insert("__dit__[t-\(name)_\(suffix)-\(index)-0]")
      }
    }
  }
  let actual = try! graph.openStore(checkpoints.dit + ".partial", flags: .readOnly) {
    Set($0.keys)
  }.get()
  precondition(
    actual == expected,
    "DiT store names: missing \(expected.subtracting(actual).sorted()), unexpected \(actual.subtracting(expected).sorted())"
  )
  print("Verified SQLite names for all", actual.count, "DiT tensors")
}

// Refresh the standalone DiT checkpoint after parity using a clean staging database.
func exportDiTCheckpoint() {
  let exportStorePath = checkpoints.dit + ".partial"
  precondition(
    !FileManager.default.fileExists(atPath: exportStorePath),
    "Remove the incomplete .partial export before retrying")
  if exportingFP16 {
    runDiT(Float.self, name: "float32", fp16Weights: true)
  } else {
    runDiT(BFloat16.self, name: "bfloat16")
  }
  verifyDiTStoreNames()
  if exportingFP16 {
    reference.verify_dit_storage(exportStorePath, Array(flux3FP32StorageWeights.values))
  }
  reference.finish(exportingFP16 ? "export-dit-f16" : "export-dit")
  let status = rename(exportStorePath, checkpoints.dit)
  precondition(status == 0, "Cannot replace checkpoint")
  print("DiT export complete:", checkpoints.dit)
}

func exportCheckpoint() {
  // Each unit has its own clean staging database, including both VAE graphs together.
  let paths = [checkpoints.text, checkpoints.dit, checkpoints.vae]
  for path in paths {
    precondition(
      !FileManager.default.fileExists(atPath: path), "Checkpoint already exists: \(path)")
    precondition(
      !FileManager.default.fileExists(atPath: path + ".partial"),
      "Remove the incomplete .partial export before retrying: \(path)")
  }
  precondition(
    !FileManager.default.fileExists(atPath: checkpoints.tokenizer), "Tokenizer already exists")
  let tokenizerRoot = String(Python.str(reference.ROOT))!
  let tokenizer = try! Flux3Tokenizer(path: tokenizerRoot + "/text_encoder/tokenizer.json")
  for prompt in [
    "A red ceramic teapot on a wooden table.",
    "Café e\u{301} — 中文, 日本語, 🫖!  Three   spaces.\nI'm testing 123456.\t\n",
    "A literal <|im_start|> token, <|image_pad|> and <|im_end|>.",
  ] {
    let pack = reference.text_case(prompt)
    let expected = try! Tensor<Int32>(numpy: pack["tokens"])
    let actual = tokenizer.prompt(prompt)
    precondition(
      actual.valid == Int(pack["attention"].sum())! && actual.tokens.count == expected.shape[0])
    for i in actual.tokens.indices {
      precondition(actual.tokens[i] == expected[i], "Tokenizer mismatch at \(i)")
    }
  }
  print(
    "Swift tokenizer matches Hugging Face: chat template, padding, whitespace, Unicode, special tokens"
  )
  // Every unit is checked immediately before writing, then read into a fresh graph.
  runText(Float16.self, name: "float16")
  runDiT(BFloat16.self, name: "bfloat16")
  verifyDiTStoreNames()
  runVAE()
  let state = reference.state("vae")
  let mean = cpu(reference.array(state["bn.running_mean"]))
  let std = cpu(reference.array((state["bn.running_var"] + 1e-4).sqrt()))
  try! graph.openStore(checkpoints.vae + ".partial") {
    try! $0.write("latent_mean", tensor: mean, strict: true)
    try! $0.write("latent_std", tensor: std, strict: true)
  }.get()
  // Check the native unpack/denormalize path against the reference helper.
  let fixture = reference.vae_reference(state, 32, 48, 42, pythonDevice)
  let native = flux3DecodeLatents(
    cpu(fixture["tokens"]), mean: mean, std: std, height: 32, width: 48)
  precondition(
    Bool(
      reference.metrics(
        "runtime/decode_latents", native.makeNumpyArray(),
        reference.denormalize_tokens(fixture["tokens"], state, 32, 48), 1e-6))!)
  let nativeSchedule = flux3Schedule(steps: 16)
  let schedule = Python.import("flux3.sampling").timesteps(16, 6.93).numpy()
  for i in nativeSchedule.indices {
    precondition(abs(nativeSchedule[i] - Float(schedule[i])!) < 1e-6)
  }
  reference.finish("export-checkpoint")
  try! FileManager.default.copyItem(
    atPath: tokenizerRoot + "/text_encoder/tokenizer.json",
    toPath: checkpoints.tokenizer)
  for path in paths {
    try! FileManager.default.moveItem(atPath: path + ".partial", toPath: path)
    print("Export complete:", path)
  }
}
