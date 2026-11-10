"""Pinned, unmodified MOSS / Transformers reference forwards; no checkpoint export."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

import numpy as np
import torch
from safetensors import safe_open

VARIANT = os.environ.get('MOSS_VARIANT', 'sft')
assert VARIANT in ('sft', 'base'), VARIANT
MODEL_NAME, REVISION = {
    'sft': ('MOSS-Image-SFT', '2c7954437269461767f533b006823e8f85973a60'),
    'base': ('MOSS-Image-Base', '32edbc8a53ff3117ed52af34ce27a031129f2d0e'),
}[VARIANT]
ROOT = Path('/slow/Data') / MODEL_NAME
assert json.loads((ROOT / 'DOWNLOAD_MANIFEST.json').read_text())['revision'] == REVISION
SHARED_ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
ARTIFACTS = SHARED_ARTIFACTS if VARIANT == 'sft' else SHARED_ARTIFACTS / VARIANT
ARTIFACTS.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
REPORT = []
_TEXT = None
_TEXT_SPEC = None
_DIT = None
_DIT_SPEC = None


class State:
    def __init__(self, component):
        self.root = ROOT / component
        filename = 'model' if component == 'text_encoder' else 'diffusion_pytorch_model'
        self.index = json.loads((self.root / f'{filename}.safetensors.index.json').read_text())['weight_map']

    def __getitem__(self, name):
        with safe_open(self.root / self.index[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name)

    def keys(self):
        return self.index.keys()


def array(x):
    return x.detach().float().cpu().contiguous().numpy()


def normalize_name(name):
    name = str(name)
    if '[' in name and name.endswith(']'):
        name = name.split('[', 1)[1][:-1]
    if name.startswith('t-'):
        name = name[2:]
    return re.sub(r'-\d+-\d+$', '', name)


def text_source(name):
    name = normalize_name(name).removeprefix('model.')
    if name.endswith('.q_gate_proj'):
        return name.removesuffix('.q_gate_proj') + '.q_proj.weight'
    return name if name.endswith(('.A_log', '.dt_bias', '.conv1d.weight')) else name + '.weight'


def interleave(x, heads, head_dim, rotary_dim):
    shape = x.shape
    x = x.reshape(heads, head_dim, -1)
    rotary = x[:, :rotary_dim].reshape(heads, 2, rotary_dim // 2, -1).transpose(1, 2)
    return torch.cat([rotary.reshape(heads, rotary_dim, -1), x[:, rotary_dim:]], 1).reshape(shape)


def weight(state, key, transform='identity', heads=0, row_start=0, row_count=0, component='dit',
           fp16_storage=False):
    name = text_source(key) if component == 'text' else str(key)
    x = state[name].float()
    if row_count:
        x = x[int(row_start):int(row_start) + int(row_count)]
    if component == 'text':
        if transform == 'qwen_q' or transform == 'qwen_gate':
            x = x.reshape(16, 2, 256, -1)[:, int(transform == 'qwen_gate')].reshape(4096, -1)
            if transform == 'qwen_q':
                x = interleave(x, 16, 256, 64)
        elif transform == 'qwen_k':
            x = interleave(x, 4, 256, 64)
        elif transform == 'qwen_norm':
            x = x + 1
            if '.self_attn.' in name:
                x = interleave(x, 1, 256, 64)
        elif transform == 'exp':
            x = x.exp()
        elif transform == 'conv1d':
            x = x.unsqueeze(-1)
    else:
        if transform == 'qk':
            x = interleave(x, int(heads), 128, 128)
        elif transform == 'qk_norm':
            x = interleave(x, 1, 128, 128)
    if fp16_storage:
        x = x.half().float()
        assert torch.isfinite(x).all(), (name, 'FP16 storage overflow')
    return array(x)


def check_text_names(names):
    names = [normalize_name(name) for name in names]
    assert len(names) == len(set(names)) == 434, len(names)
    state = State('text_encoder')
    used = {text_source(name) for name in names}
    expected = {name for name in state.keys() if name.startswith('language_model.')}
    assert used == expected, (expected-used, used-expected)
    path = Path('/slow/Data/qwen_3.5_9b_f16.ckpt')
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        keys = {row[0] for row in db.execute('SELECT name FROM tensors INDEXED BY sqlite_autoindex_tensors_1')}
    actual = {f'__text_model__[t-{name}-0-0]' for name in names}
    assert actual <= keys, actual-keys
    (ARTIFACTS / 'text_parameter_names.json').write_text(json.dumps(sorted(actual), indent=2) + '\n')
    print('Text graph: 434 parameter names match the existing SQLite checkpoint', flush=True)


def text_case(kind, length=17, seed=42):
    if kind == 'random':
        ids = torch.randint(0, 248320, (1, int(length)), generator=torch.Generator().manual_seed(int(seed)))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'tokenizer', local_files_only=True)
        formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': kind}],
                                                  tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=True)
        ids = tokenizer(formatted, return_tensors='pt', truncation=True, max_length=4096).input_ids
    return {'tokens': ids.numpy().astype(np.int32).reshape(-1), 'kind': kind}


def text_reference(pack, precision, device):
    global _TEXT, _TEXT_SPEC
    from transformers import Qwen3_5Model
    spec = (str(precision), int(device))
    if _TEXT_SPEC != spec:
        _TEXT = None
        release()
        print(f'Loading upstream Qwen3.5 reference: {spec}', flush=True)
        _TEXT = Qwen3_5Model.from_pretrained(
            ROOT / 'text_encoder', local_files_only=True, dtype=getattr(torch, precision),
            device_map={'': f'cuda:{device}'},
            attn_implementation='eager' if precision == 'float32' else 'sdpa').eval()
        _TEXT_SPEC = spec
    result = {}
    hooks = []
    for index in [0, 1, 2, 3, 7, 15, 23, 31]:
        def capture(module, args, output, index=index):
            result[f'layer_{index}'] = array(output[0] if isinstance(output, tuple) else output)[0]
        hooks.append(_TEXT.language_model.layers[index].register_forward_hook(capture))
    def named_capture(name):
        def capture(module, args, output):
            result[name] = array(output)[0]
        return capture
    hooks.append(_TEXT.language_model.embed_tokens.register_forward_hook(named_capture('embedding')))
    first = _TEXT.language_model.layers[0]
    hooks.append(first.input_layernorm.register_forward_hook(named_capture('input_norm_0')))
    hooks.append(first.linear_attn.in_proj_qkv.register_forward_hook(named_capture('qkv_0')))
    def conv_capture(module, args, output):
        result['conv_0'] = array(torch.nn.functional.silu(output[:, :, :len(pack['tokens'])]).transpose(1, 2))[0]
    hooks.append(first.linear_attn.conv1d.register_forward_hook(conv_capture))
    def core_capture(module, args):
        result['core_0'] = array(args[0])
    hooks.append(first.linear_attn.norm.register_forward_pre_hook(core_capture))
    def gated_capture(module, args):
        result['gated_0'] = array(args[0])[0]
    hooks.append(first.linear_attn.out_proj.register_forward_pre_hook(gated_capture))
    hooks.append(first.linear_attn.out_proj.register_forward_hook(named_capture('attention_0')))
    try:
        ids = torch.as_tensor(np.asarray(pack['tokens']).astype(np.int64), device=f'cuda:{device}')[None]
        result['output'] = array(_TEXT.language_model(input_ids=ids, use_cache=False).last_hidden_state)[0]
    finally:
        for hook in hooks:
            hook.remove()
    if pack['kind'] != 'random':
        np.save(ARTIFACTS / f'prompt_features_{precision}.npy', result['output'])
    return result


def save_text_features(value, precision):
    np.save(ARTIFACTS / f'swift_prompt_features_{precision}.npy', np.asarray(value))


def load_text_features(precision, swift=False):
    name = f'{"swift_" if swift else ""}prompt_features_{precision}.npy'
    path = ARTIFACTS / name
    if not path.exists() and VARIANT == 'base':
        # The pinned Base and SFT snapshots contain identical frozen encoders.
        base = json.loads((ROOT / 'DOWNLOAD_MANIFEST.json').read_text())
        sft = json.loads((Path('/slow/Data/MOSS-Image-SFT') / 'DOWNLOAD_MANIFEST.json').read_text())
        def encoder_files(manifest):
            return {f['path']: f['checksum'] for f in manifest['files']
                    if f['path'].startswith(('text_encoder/', 'tokenizer/'))}
        assert encoder_files(base) == encoder_files(sft), 'Cannot reuse different encoder features'
        path = SHARED_ARTIFACTS / name
    return np.load(path)


def upstream():
    name = 'moss_image_upstream'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'pipeline_moss_image.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def dit_case(index, precision):
    index = int(index)
    height, width, text_length, timestep, skip = [
        (2, 3, 17, 750.125, []), (3, 2, 11, 321.5, [22]), (64, 64, 0, 999.0, []),
    ][index]
    generator = torch.Generator().manual_seed(420 + index)
    noise = torch.randn((1, 128, height, width), generator=generator)
    if text_length:
        context = torch.randn((text_length, 4096), generator=generator).numpy()
    else:
        context = load_text_features(precision, swift=True)
    return dict(x=noise.flatten(2).transpose(1, 2)[0].numpy(), context=context,
                height=height, width=width, timestep=timestep, skip_layers=skip)


def check_dit_keys(state, keys, skip_layers):
    used = set(map(str, keys))
    skip = tuple(f'layers.{int(index)}.' for index in skip_layers)
    expected = {key for key in state.keys() if key != 'rope.bands' and not (skip and key.startswith(skip))}
    assert used == expected, (sorted(expected-used), sorted(used-expected))
    print(f'DiT: all {len(used)} active source tensors accounted for', flush=True)


def check_dit_store(path, written_names, fp16_storage=False):
    written_names = list(map(str, written_names))
    assert len(written_names) == len(set(written_names)) == 524, len(written_names)
    with sqlite3.connect(Path(str(path)).as_uri() + '?mode=ro', uri=True) as db:
        dtypes = dict(db.execute('SELECT name,datatype FROM tensors'))
        names = set(dtypes)
    assert names == set(written_names), (names-set(written_names), set(written_names)-names)
    expected = set()
    def add(stem, count=1, bias=False):
        expected.update(f'__dit__[t-{stem}-{i}-{parameter}]'
                        for i in range(count) for parameter in range(2 if bias else 1))
    for stem in ['context_embedder', 'x_embedder', 'time_embedder_0', 'time_embedder_2',
                 *(f'x_ada_ln_{i}' for i in range(6)), 'ada_ln_0', 'ada_ln_1']:
        add(stem, bias=True)
    for stem in ['final_norm', 'final_linear']:
        add(stem)
    for stem in ['context_q', 'context_k', 'context_v', 'context_o',
                 'context_ffn_up_proj', 'context_ffn_gate_proj', 'context_ffn_down_proj']:
        add(stem, count=2, bias=True)
    for stem in ['context_norm1', 'context_norm2', 'context_norm_q', 'context_norm_k']:
        add(stem, count=2)
    for stem in ['x_q', 'x_k', 'x_v', 'x_o', 'x_norm_q', 'x_norm_k', 'x_norm1', 'x_norm2',
                 'ffn_up_proj', 'ffn_gate_proj', 'ffn_down_proj']:
        add(stem, count=42)
    assert names == expected, (sorted(expected-names), sorted(names-expected))
    keep_float32 = {'__dit__[t-time_embedder_2-0-0]', '__dit__[t-time_embedder_2-0-1]'}
    assert all(dtype == (16384 if not fp16_storage or name in keep_float32 else 131072)
               for name, dtype in dtypes.items()), 'Unexpected checkpoint storage dtype'
    if fp16_storage:
        print('Verified storage: 522 FP16 tensors; time_embedder_2 weight and bias are FP32', flush=True)
    (ARTIFACTS / 'dit_parameter_names.json').write_text(json.dumps(sorted(names), indent=2) + '\n')
    print('Verified all 524 SQLite keys match the flat parameter naming convention', flush=True)


def dit_reference(pack, precision, device, original_context=False):
    global _DIT, _DIT_SPEC
    spec = (str(precision), int(device))
    if _DIT_SPEC != spec:
        _DIT = None
        release()
        print(f'Loading pinned upstream MOSS DiT: {spec}', flush=True)
        _DIT = upstream().MOSSImageTransformer.from_pretrained(
            ROOT / 'transformer', local_files_only=True,
            torch_dtype=getattr(torch, precision)).to(f'cuda:{device}').eval()
        _DIT_SPEC = spec
    device = f'cuda:{device}'
    result = {}
    hooks = []
    def capture(name, squeeze=True, reshape=None):
        def hook(module, args, output):
            value = array(output)
            if reshape:
                value = value.reshape(reshape)
            elif squeeze:
                value = value[0]
            result[name] = value
        return hook
    for name, module in [('image_embedding', _DIT.video_embedder),
                         ('text_projection', _DIT.text_embedder.proj)]:
        hooks.append(module.register_forward_hook(capture(name)))
    for index, block in enumerate(_DIT.text_embedder.blocks):
        hooks.append(block.register_forward_hook(capture(f'text_block_{index}')))
    hooks.append(_DIT.time_embedding.register_forward_hook(capture('time_embedding', False)))
    hooks.append(_DIT.shared_block_modulation.register_forward_hook(capture('modulation', reshape=(6, 3072))))
    hooks.append(_DIT.video_head_modulation.register_forward_hook(capture('head_modulation', reshape=(2, 3072))))
    for index in [0, 21, 22, 41]:
        hooks.append(_DIT.layers[index].register_forward_hook(capture(f'layer_{index}')))
    context = pack['context']
    if original_context:
        context = load_text_features(precision)
    context = torch.as_tensor(context, device=device, dtype=getattr(torch, precision))[None]
    height, width = int(pack['height']), int(pack['width'])
    x = torch.as_tensor(pack['x'], device=device).T.reshape(1, 128, height, width)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    try:
        # FP32 control uses math SDPA on both sides. The released BF16 path uses flash SDPA.
        backend = SDPBackend.MATH if precision == 'float32' else SDPBackend.FLASH_ATTENTION
        with sdpa_kernel(backend):
            output = _DIT(x, torch.tensor([pack['timestep']], device=device), context,
                          skip_layers=list(pack['skip_layers'])).sample
        result['output'] = array(output.permute(0, 2, 3, 1).reshape(height * width, 128))
    finally:
        for hook in hooks:
            hook.remove()
    return result


def metrics(label, actual, expected, max_relative_limit=1e-3, relative_l2_limit=1e-3):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    max_abs = float(np.abs(delta).max())
    max_relative = max_abs / max(float(np.abs(expected).max()), 1e-12)
    relative_l2 = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-12))
    passed = max_relative <= max_relative_limit and relative_l2 <= relative_l2_limit
    row = dict(label=str(label), shape=list(actual.shape), max_abs=max_abs,
               max_relative=max_relative, relative_l2=relative_l2,
               actual_sample=actual.ravel()[:8].tolist(), reference_sample=expected.ravel()[:8].tolist(),
               max_relative_limit=float(max_relative_limit), relative_l2_limit=float(relative_l2_limit),
               passed=passed)
    REPORT.append(row)
    print(json.dumps(row), flush=True)
    return passed


def finish(name):
    report = dict(model=f'OpenMOSS-Team/{MODEL_NAME}', model_revision=REVISION,
                  torch=torch.__version__, results=REPORT)
    (ARTIFACTS / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n')
    assert REPORT and all(row['passed'] for row in REPORT), 'Numeric parity failed'
    print('PARITY PASSED:', ARTIFACTS / f'{name}.json', flush=True)


def release():
    gc.collect()
    torch.cuda.empty_cache()


def generation_setup(height, width, steps, seed):
    scheduler = upstream().MOSSImageScheduler(
        num_train_timesteps=1000, shift=5.0, training_shift=1.0)
    scheduler.set_timesteps(int(steps))
    noise = torch.randn((1, 128, int(height) // 16, int(width) // 16),
                        generator=torch.Generator().manual_seed(int(seed)))
    return dict(noise=array(noise.flatten(2).transpose(1, 2)[0]),
                timesteps=scheduler.timesteps.tolist(), sigmas=scheduler.sigmas.tolist())


def decode_generation(latent, height, width, output, device, prompt, negative_prompt,
                      seed, guidance, timesteps, sigmas, denoising_seconds):
    from diffusers import AutoencoderKLFlux2
    from diffusers.image_processor import VaeImageProcessor
    from PIL.PngImagePlugin import PngInfo
    height, width = int(height), int(width)
    packed = np.asarray(latent, dtype=np.float32)
    assert packed.shape == (height // 16 * (width // 16), 128) and np.isfinite(packed).all()
    output = Path(str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.with_suffix('.latent.npy'), packed)
    print('Decode final latent with upstream FP32 Flux2 VAE', flush=True)
    vae = AutoencoderKLFlux2.from_pretrained(
        ROOT / 'vae', local_files_only=True, torch_dtype=torch.float32).to(f'cuda:{device}').eval()
    z = torch.as_tensor(packed, device=f'cuda:{device}').T.reshape(1, 128, height // 16, width // 16)
    z = z * (vae.bn.running_var + vae.config.batch_norm_eps).sqrt().view(1, -1, 1, 1)
    z = z + vae.bn.running_mean.view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2*h, 2*w)
    decoded = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    assert tuple(decoded.shape) == (1, 3, height, width) and torch.isfinite(decoded).all()
    image = VaeImageProcessor(vae_scale_factor=16).postprocess(decoded, output_type='pil')[0]
    metadata = dict(model_revision=REVISION, prompt=str(prompt), negative_prompt=str(negative_prompt),
                    width=width, height=height, seed=int(seed), steps=len(timesteps),
                    guidance=float(guidance), shift=5.0, uncond_skip_layers=[22],
                    dtype='float32', text_encoder='Swift', dit='Swift', vae='PyTorch',
                    noise_rng='PyTorch CPU', timesteps=list(timesteps), sigmas=list(sigmas),
                    denoising_seconds=float(denoising_seconds), output=str(output))
    info = PngInfo()
    info.add_text('generation', json.dumps(metadata))
    image.save(output, pnginfo=info)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved generated image: {output}', flush=True)
