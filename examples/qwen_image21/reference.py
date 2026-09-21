"""Preprocessing, noise initialization, and scheduler setup.

Neural-network inference runs in Swift. There are no reference forward passes
or parity fixtures in the generation/editing path.
"""
import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "artifacts/model"
REFERENCE = ROOT / "artifacts/reference"
# Latent normalization metadata only; architecture stays in Swift.
vae_config = json.loads((MODEL / "vae/config.json").read_text())
LATENTS_MEAN = np.asarray(vae_config["latents_mean"], np.float32)[None, :, None, None]
LATENTS_STD = np.asarray(vae_config["latents_std"], np.float32)[None, :, None, None]

def upstream(name, relative_package):
    if name == "pipeline_qwenimage21":
        import diffusers.models as models
        models.QwenImage21Transformer2DModel = upstream("transformer_qwenimage21", "models.transformers").QwenImage21Transformer2DModel
        models.AutoencoderKLQwenImage21 = upstream("autoencoder_kl_qwenimage21", "models.autoencoders").AutoencoderKLQwenImage21
    full_name = f"diffusers.{relative_package}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, REFERENCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def generation_latent_input(latent, height, width):
    latent = np.asarray(latent).reshape(height//16,width//16,64).transpose(2,0,1)[None].copy()
    # Keep statistics outside the decoder graph, exactly as the pipeline does.
    return (latent * LATENTS_STD
            + LATENTS_MEAN)


def save_rgba(array, path):
    from PIL import Image
    array = np.asarray(array)[0].transpose(1,2,0)
    array = np.clip(array/2+0.5,0,1)
    image = Image.fromarray(np.rint(array*255).astype(np.uint8), "RGBA")
    image.save(path)
    white = Image.new("RGB",image.size,"white")
    white.paste(image,mask=image.getchannel("A"))
    white.save(str(Path(path).with_suffix(".preview.png")))
    return str(path)


def normalize_image_latent(moments):
    latent = np.asarray(moments)[:,:64]
    return ((latent - LATENTS_MEAN) /
            LATENTS_STD).copy()


def text_inputs(prompt):
    from transformers import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(MODEL / "processor", local_files_only=True)
    system = "Comprehend and analyze the provided prompt."
    template = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    tokens = processor(text=[template], padding=True, padding_side="left", return_tensors="pt").input_ids
    drop = len(processor.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": system}]}], tokenize=True, return_dict=False)[0])
    return {"tokens": tokens.numpy()[0].astype(np.int32), "drop": drop}


def generation_setup(context, height, width, steps, seed, device=0):
    from diffusers import FlowMatchEulerDiscreteScheduler
    module = upstream("pipeline_qwenimage21", "pipelines.qwenimage21")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL / "scheduler", local_files_only=True)
    device = torch.device(f"cuda:{device}")
    # Match the released pipeline's CUDA RNG and C,H,W noise layout before packing.
    noise = torch.randn((1, 1, 64, height // 16, width // 16),
                        generator=torch.Generator(device=device).manual_seed(seed),
                        device=device, dtype=torch.float32)
    initial = noise.reshape(1, 64, -1).transpose(1, 2)[0]
    mu = module.calculate_shift(initial.shape[0],
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15))
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), device=device, mu=mu)
    return {"initial": initial.cpu().numpy(), "context": np.asarray(context),
            "timesteps": (scheduler.timesteps / 1000).float().cpu().numpy(),
            "sigmas": scheduler.sigmas.float().cpu().numpy()}


def edit_inputs(paths, prompt, resolution=512):
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor
    module = upstream("pipeline_qwenimage21", "pipelines.qwenimage21")
    from transformers import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(MODEL / "processor", local_files_only=True)
    image_processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=False, do_convert_grayscale=False)
    images, vae_inputs = [], []
    for path in paths:
        image = Image.open(path).convert("RGBA")
        width, height, _ = module.calculate_dimensions(resolution**2, image.width / image.height)
        image = image_processor.resize(image, width=width, height=height)
        vae_inputs.append(image_processor.preprocess(image, width=width, height=height).numpy())
        white = Image.new("RGB", image.size, "white")
        white.paste(image, mask=image.getchannel("A"))
        images.append(white)
    system = "Comprehend and analyze the provided prompt."
    placeholders = " ".join(f"<image{i+1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(images)))
    template = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{placeholders}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=[template], images=images, padding=True, padding_side="left", return_tensors="pt")
    ids = inputs.input_ids[0].numpy()
    drop = len(processor.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": system}]}], tokenize=True, return_dict=False)[0])
    image_token = processor.tokenizer.encode("<|image_pad|>")[0]
    grids = inputs.image_grid_thw.tolist()
    spans, cursor = [], 0
    for t, h, w in grids:
        start = int(np.flatnonzero(ids[cursor:] == image_token)[0]) + cursor
        length = t * h * w // 4
        assert t == 1 and (ids[start:start+length] == image_token).all()
        spans.append([start, h // 2, w // 2])
        cursor = start + length
    return {"tokens": ids.astype(np.int32), "grids": grids, "spans": spans,
            "patches": inputs.pixel_values.float().numpy(), "drop": drop,
            "image_mask": (ids[drop:] == image_token), "vae_inputs": vae_inputs}


def edit_generation_setup(condition, text_inputs, normalized_latents, height, width, steps, seed, device=0):
    result = generation_setup(condition, height, width, steps, seed, device)
    slots = text_inputs["image_mask"]
    segments, cursor, text_offset, image_offset = [], 0, 0, 0
    shapes = [(v.shape[-2], v.shape[-1]) for v in normalized_latents]
    for h, w in shapes:
        start = int(np.flatnonzero(slots[cursor:])[0]) + cursor
        if start > cursor:
            segments.append([False, start-cursor, text_offset, 0, 0])
            text_offset += start-cursor
        segments.append([True, h*w, image_offset, h, w])
        image_offset += h*w
        cursor = start + h*w//4
    if cursor < len(slots):
        segments.append([False, len(slots)-cursor, text_offset, 0, 0])
    pixels = height//16 * (width//16)
    segments.append([True, pixels, image_offset, height//16, width//16])
    result["context"] = np.asarray(condition)[~slots].copy()
    result["condition_latent"] = np.concatenate([v.reshape(64, -1).T for v in normalized_latents]).copy()
    result["segments"] = segments
    return result
