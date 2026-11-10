"""Profile native activation ranges and FP16 overflows; does not modify weights."""
import argparse
import json
import torch
import reference as r

parser = argparse.ArgumentParser()
parser.add_argument("component", choices=["dit", "text"])
parser.add_argument("--dtype", default="float16")
parser.add_argument("--device", type=int, default=2)
parser.add_argument("--case", type=int, default=0)
args = parser.parse_args()
target = f"cuda:{args.device}"
dtype = getattr(torch, args.dtype)
if args.component == "dit":
    model = r.BriaFibo2Transformer2DModel.from_pretrained(
        r.ROOT / "transformer", torch_dtype=dtype, local_files_only=True).to(target).eval()
else:
    model = r.Qwen3VLForConditionalGeneration.from_pretrained(
        r.ROOT / "text_encoder", dtype=dtype, local_files_only=True,
        attn_implementation="sdpa").to(target).eval()
rows = []
def hook(name):
    def capture(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        x = output.detach().float()
        finite = x.isfinite()
        row = dict(name=name, shape=list(x.shape), dtype=str(output.dtype),
                   max_abs=float(x[finite].abs().max()) if finite.any() else None,
                   nonfinite=int((~finite).sum()))
        rows.append(row)
        if row["nonfinite"] or (row["max_abs"] or 0) > 10000:
            print(json.dumps(row), flush=True)
        if row["nonfinite"]:
            raise FloatingPointError(name)
    return capture
handles = []
for name, module in model.named_modules():
    if name and (isinstance(module, torch.nn.Linear) or
                 type(module).__name__ in ("BriaFibo2TransformerBlock", "BriaFibo2PerceiverLayer",
                                          "Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock")):
        handles.append(module.register_forward_hook(hook(name)))
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(
                lambda m, a, name=name: hook(name + ".input")(m, (), a[0])))
try:
    if args.component == "dit":
        p = r.dit_case(args.case, "float32")
        def tensor(value): return torch.from_numpy(value).to(device=target, dtype=dtype)
        model(hidden_states=tensor(p["x"]), timestep=torch.tensor([p["timestep"]], device=target),
              encoder_hidden_states=tensor(p["context"]),
              context_latents=[tensor(x) for x in p["references"]])
    else:
        p = r.text_case(args.case)
        model(**{k: v.to(target) for k,v in p["inputs"].items()},
              output_hidden_states=True, use_cache=False)
    print("All captured outputs finite", flush=True)
except FloatingPointError as error:
    print("FIRST NONFINITE:", error, flush=True)
finally:
    for handle in handles:
        handle.remove()
    path = r.ARTIFACTS / f"ranges_{args.component}_{args.dtype}_case{args.case}.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(path, flush=True)
