import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

// Export is an explicit mode; normal parity runs never write checkpoint weights.
let exporting = ["export", "export-dit"].contains(CommandLine.arguments.dropFirst().first ?? "")
let checkpoints = Flux3Checkpoints()

func exportAndVerify<T: TensorNumeric>(
  _ type: T.Type, key: String, model: Model, fresh: Model,
  inputs: [DynamicGraph.AnyTensor], expected: [DynamicGraph.AnyTensor]
) {
  let exportStorePath = checkpoints.path(for: key) + ".partial"
  print("Exporting", key, "to", exportStorePath)
  try! graph.openStore(exportStorePath) { try! $0.write(key, model: model, strict: true) }.get()
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
  runDiT(BFloat16.self, name: "bfloat16")
  verifyDiTStoreNames()
  reference.finish("export-dit")
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
