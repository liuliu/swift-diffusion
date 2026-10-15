"""Locate the first unscaled FP16 activation overflow without changing model operators.

Run in the reference Python environment: python examples/flux3/audit_fp16.py --device 2
Weight conversion is audited separately by the Swift export-dit-f16 mode.
"""
import argparse
import json

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import reference as ref
import flux3.dit as module


class ActivationOverflow(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    args = parser.parse_args()
    device = f"cuda:{args.device}"
    torch.set_num_threads(8)
    cases = [
        ("random_no_reference", ref.dit_case(16, 2, 3, [], 42, "float16", .75, .5)),
        ("random_two_references", ref.dit_case(16, 3, 2, [[2, 2], [1, 3]], 43, "float16", .625, .12345)),
    ]
    prompt = ("A red ceramic teapot on a wooden table beside a window, soft morning light, "
              "a small white cup and a folded linen napkin, realistic photograph.")
    text = ref.text_case(prompt)
    features = ref.text_reference(ref.state("text"), text, "float16", args.device)
    context = ref.concatenate_features([features[f"hidden_{i}"] for i in [4, 8, 12, 16, 20, 24, 28, 32]])
    # Same FP16 text -> BF16 feature boundary used by the native Swift generator.
    context = torch.from_numpy(context).bfloat16().float().numpy()
    for label, time, target in [("prompt_512_first_step", 1., 6.93 * (15/16) / (1 + 5.93 * (15/16))),
                                ("prompt_512_late_time", .25, 0.)]:
        case = ref.dit_case(len(context), 32, 32, [], 42, "float16", time, target)
        case["context"] = context
        cases.append((label, case))
    del features
    ref.release()

    weights = ref.state("dit")
    with torch.device("meta"):
        model = module.DiT(True)
    model.load_state_dict({k: v.half() for k, v in weights.items()}, strict=True, assign=True)
    model = model.to(device).eval()
    stack, module_inputs, records = [], {}, []

    def enter(name):
        def hook(_, inputs):
            stack.append(name)
            module_inputs[name] = inputs
        return hook

    def leave(_, inputs, output):
        module_inputs.pop(stack.pop(), None)

    for name, child in model.named_modules():
        child.register_forward_pre_hook(enter(name))
        child.register_forward_hook(leave, always_call=True)

    class Inspect(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if (isinstance(out, torch.Tensor) and out.is_floating_point() and out.numel()
                    and not bool(out.isfinite().all())):
                name = stack[-1] if stack else ""
                event = dict(module=name, op=str(func), dtype=str(out.dtype), shape=list(out.shape),
                             nan=int(out.isnan().sum()), inf=int(out.isinf().sum()))
                inputs = [x for x in args if isinstance(x, torch.Tensor) and x.is_floating_point() and x.numel()]
                event["inputs"] = [dict(shape=list(x.shape), dtype=str(x.dtype),
                                        max_abs=float(x.abs().max()), finite=bool(x.isfinite().all())) for x in inputs]
                if str(func) == "aten.add.Tensor" and args[0] is module_inputs[name][0]:
                    event["operation"] = "residual addition"
                # Isolate the operation's range from its already-rounded, finite FP16 inputs.
                if all(x["finite"] for x in event["inputs"]):
                    full_args = tuple(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in args)
                    with torch.autocast("cuda", enabled=False):
                        full = func(*full_args, **(kwargs or {}))
                    event["operation_fp32_max_abs"] = float(full.abs().max())
                    event["operation_fp32_outside_fp16_range"] = int((full.abs() > 65504).sum())
                records.append(event)
                raise ActivationOverflow()
            return out

    for label, case in cases:
        nc, nx = len(case["context"]), len(case["x"])
        data = {m: torch.from_numpy(case[k]).to(device=device, dtype=torch.float16)
                for m, k in [("text", "context"), ("image", "x"), ("image_cond", "reference")]}
        ids = torch.from_numpy(case["ids"]).to(device)
        ids = dict(text=ids[:nc], image=ids[nc:nc+nx], image_cond=ids[nc+nx:])
        times = {m: torch.full((len(x),), case["time"] if m == "image" else 0,
                              device=device, dtype=x.dtype) for m, x in data.items()}
        pack = module.Packed(data, ids, times, {m: [(0, len(x))] for m, x in data.items()})
        try:
            with Inspect(), torch.autocast("cuda", dtype=torch.float16):
                out = model(pack, torch.tensor([case["target"]], device=device))
            records.append(dict(case=label, finite=True, max_abs=float(out.abs().max())))
        except ActivationOverflow:
            records[-1]["case"] = label
        print(json.dumps(records[-1]), flush=True)
    report = dict(prompt=prompt, cases=records, note="First failures only; later NaN propagation is excluded. No activation rescaling applied.")
    (ref.ARTIFACTS / "fp16-activation-audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
