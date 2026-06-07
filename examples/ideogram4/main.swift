import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PythonKit

typealias FloatType = Float
typealias TextFloatType = Float16

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

let env = ProcessInfo.processInfo.environment
let modelRoot = env["IDEOGRAM4_MODEL"] ?? "/slow/Data/ideogram-4-fp8"
let deviceID = Int(env["IDEOGRAM4_DEVICE"] ?? "0") ?? 0
let dtypeName = env["IDEOGRAM4_REFERENCE_DTYPE"] ?? "float16"
let textDtypeName = env["IDEOGRAM4_TEXT_REFERENCE_DTYPE"] ?? "float16"
let textTokenLength = Int(env["IDEOGRAM4_TEXT_TOKENS"] ?? "32") ?? 32
let ditTextLength = Int(env["IDEOGRAM4_DIT_TEXT_TOKENS"] ?? "4") ?? 4
let ditGridHeight = Int(env["IDEOGRAM4_DIT_GRID_H"] ?? "1") ?? 1
let ditGridWidth = Int(env["IDEOGRAM4_DIT_GRID_W"] ?? "1") ?? 1
let ditTimestep = Float(env["IDEOGRAM4_DIT_T"] ?? "0.75") ?? 0.75
let mode = CommandLine.arguments.dropFirst().first ?? "parity"

let site = Python.import("site")
let sys = Python.import("sys")
let osPath = Python.import("os.path")

func movePythonPathToFront(_ path: String) {
  while Bool(sys.path.__contains__(path)) ?? false {
    sys.path.remove(path)
  }
  sys.path.insert(0, path)
}

func movePythonPathToBack(_ path: String) {
  while Bool(sys.path.__contains__(path)) ?? false {
    sys.path.remove(path)
  }
  sys.path.append(path)
}

var insertedVirtualEnvSitePackages = false
var preferredVirtualEnvSitePackages: String? = nil
if let virtualEnv = env["VIRTUAL_ENV"] {
  let libRoot = URL(fileURLWithPath: virtualEnv).appendingPathComponent("lib")
  if let pythonLibDirs = try? FileManager.default.contentsOfDirectory(
    at: libRoot, includingPropertiesForKeys: nil)
  {
    for pythonLibDir in pythonLibDirs.sorted(by: { $0.path < $1.path }) {
      let sitePackagesDir = pythonLibDir.appendingPathComponent("site-packages").path
      if FileManager.default.fileExists(atPath: sitePackagesDir) {
        movePythonPathToFront(sitePackagesDir)
        insertedVirtualEnvSitePackages = true
        preferredVirtualEnvSitePackages = sitePackagesDir
      }
    }
  }
}

let userSitePackages = String(site.getusersitepackages()) ?? ""
if Bool(osPath.isdir(userSitePackages)) ?? false {
  if insertedVirtualEnvSitePackages {
    movePythonPathToBack(userSitePackages)
  } else if (Bool(sys.path.__contains__(userSitePackages)) ?? false) == false {
    movePythonPathToFront(userSitePackages)
  }
}
let systemDistPackages = "/usr/lib/python3/dist-packages"
if insertedVirtualEnvSitePackages {
  movePythonPathToBack(systemDistPackages)
} else if (Bool(sys.path.__contains__(systemDistPackages)) ?? false) == false {
  movePythonPathToFront(systemDistPackages)
}
if let preferredVirtualEnvSitePackages {
  let currentPath = String(sys.path[0]) ?? ""
  if currentPath != preferredVirtualEnvSitePackages {
    movePythonPathToFront(preferredVirtualEnvSitePackages)
  }
}

let builtins = Python.import("builtins")
let types = Python.import("types")
let torch = Python.import("torch")
let numpy = Python.import("numpy")

torch.set_grad_enabled(false)
torch.manual_seed(42)
if !(Bool(torch.cuda.is_available()) ?? false) {
  print("CUDA is not visible to Python. Run this target outside the sandbox for parity.")
  exit(1)
}
torch.cuda.manual_seed_all(42)

let helper = types.ModuleType("ideogram4_swift_reference")
builtins.exec(
  #"""
  import math
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from safetensors import safe_open
  from safetensors.torch import load_file
  from transformers import AutoConfig
  from transformers.masking_utils import create_causal_mask
  from transformers.models.qwen3_vl.modeling_qwen3_vl import (
      Qwen3VLTextDecoderLayer,
      Qwen3VLTextRMSNorm,
      Qwen3VLTextRotaryEmbedding,
  )

  FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
  FP8_SCALE_SUFFIX = ".weight_scale"
  LLM_TOKEN_INDICATOR = 3
  OUTPUT_IMAGE_INDICATOR = 2
  IMAGE_POSITION_OFFSET = 65536
  QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

  def ref_dtype(name):
      if name == "bfloat16":
          return torch.bfloat16
      if name == "float32":
          return torch.float32
      return torch.float16

  class Fp8Linear(nn.Module):
      def __init__(self, in_features, out_features, bias, compute_dtype):
          super().__init__()
          self.in_features = in_features
          self.out_features = out_features
          self.compute_dtype = compute_dtype
          self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE))
          self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
          if bias:
              self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
          else:
              self.bias = None

      def forward(self, x):
          w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
          bias = self.bias.to(x.dtype) if self.bias is not None else None
          return F.linear(x, w, bias)

  def swap_linears_to_fp8(module, state_dict, compute_dtype, prefix=""):
      for name, child in list(module.named_children()):
          child_prefix = f"{prefix}{name}"
          if isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict:
              setattr(
                  module,
                  name,
                  Fp8Linear(
                      child.in_features,
                      child.out_features,
                      child.bias is not None,
                      compute_dtype,
                  ),
              )
          else:
              swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")

  def load_fp8_state_dict(model, state_dict, device, dtype, assign=False, strict=True):
      prepared = {}
      for k, v in state_dict.items():
          if v.dtype == FP8_WEIGHT_DTYPE:
              prepared[k] = v.to(device=device)
          elif k.endswith(FP8_SCALE_SUFFIX):
              prepared[k] = v.to(device=device, dtype=torch.float32)
          elif v.is_floating_point():
              prepared[k] = v.to(device=device, dtype=dtype)
          else:
              prepared[k] = v.to(device=device)
      missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
      if unexpected:
          raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
      if strict and missing:
          raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
      model.to(device)

  def dequant_weight_np(state_dict, key, dtype_name):
      dtype = ref_dtype(dtype_name)
      w = state_dict[key].to(dtype)
      scale_key = key[:-7] + ".weight_scale" if key.endswith(".weight") else key + "_scale"
      if scale_key in state_dict:
          w = w * state_dict[scale_key].to(dtype).view(-1, 1)
      return w.float().cpu().numpy()

  def dequant_interleaved_qk_weight_np(state_dict, key, heads, head_dim, dtype_name):
      dtype = ref_dtype(dtype_name)
      w = state_dict[key].to(dtype)
      scale_key = key[:-7] + ".weight_scale" if key.endswith(".weight") else key + "_scale"
      if scale_key in state_dict:
          w = w * state_dict[scale_key].to(dtype).view(-1, 1)
      return w.float().view(heads, 2, head_dim // 2, -1).transpose(1, 2).cpu().numpy()

  def interleaved_qk_norm_np(state_dict, key, head_dim):
      return state_dict[key].float().view(2, head_dim // 2).transpose(0, 1).cpu().numpy()

  def tensor_np(state_dict, key):
      return state_dict[key].to(torch.float32).cpu().numpy()

  def load_ideogram_text_state(root):
      full_sd = load_file(f"{root}/text_encoder/model.safetensors")
      return {
          k[len("language_model."):]: v
          for k, v in full_sd.items()
          if k.startswith("language_model.")
      }

  def load_original_qwen_text_state(root, layers):
      with open(f"{root}/model.safetensors.index.json") as f:
          weight_map = __import__("json").load(f)["weight_map"]
      keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight"]
      for i in range(int(layers)):
          prefix = f"model.language_model.layers.{i}"
          keys += [
              f"{prefix}.input_layernorm.weight",
              f"{prefix}.post_attention_layernorm.weight",
              f"{prefix}.self_attn.q_proj.weight",
              f"{prefix}.self_attn.k_proj.weight",
              f"{prefix}.self_attn.v_proj.weight",
              f"{prefix}.self_attn.o_proj.weight",
              f"{prefix}.self_attn.q_norm.weight",
              f"{prefix}.self_attn.k_norm.weight",
              f"{prefix}.mlp.gate_proj.weight",
              f"{prefix}.mlp.up_proj.weight",
              f"{prefix}.mlp.down_proj.weight",
          ]
      state = {}
      for filename in sorted(set(weight_map[k] for k in keys)):
          with safe_open(f"{root}/{filename}", framework="pt", device="cpu") as f:
              for key in keys:
                  if weight_map[key] == filename:
                      state[key[len("model.language_model."):]] = f.get_tensor(key)
      return state

  def make_text_token_ids(token_count, vocab_size=151936):
      ids = (torch.arange(token_count, dtype=torch.long) * 7919 + 12345) % vocab_size
      return ids.view(1, -1)

  def _text_config(config_root):
      config = AutoConfig.from_pretrained(config_root, trust_remote_code=True)
      text_config = config.text_config if hasattr(config, "text_config") else config
      if getattr(text_config, "rope_scaling", None) is None:
          text_config.rope_scaling = getattr(config, "rope_scaling", None) or {"mrope_section": [24, 20, 20]}
      text_config._attn_implementation = "eager"
      return text_config

  def _load_text_module(module, state_dict, device, dtype):
      swap_linears_to_fp8(module, state_dict, compute_dtype=dtype)
      load_fp8_state_dict(module, state_dict, device=device, dtype=dtype, assign=True, strict=True)
      module.eval()
      return module

  def _qwen_text_forward(
      config_root, state_dict, token_count, device_index, dtype_name, capture_layers=None, layers_to_run=None
  ):
      dtype = ref_dtype(dtype_name)
      device = torch.device(f"cuda:{device_index}")
      config = _text_config(config_root)
      embed = nn.Embedding(config.vocab_size, config.hidden_size).to(device=device, dtype=dtype)
      embed.weight.data.copy_(state_dict["embed_tokens.weight"].to(device=device, dtype=dtype))
      rotary_emb = Qwen3VLTextRotaryEmbedding(config=config).to(device)
      norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device=device, dtype=dtype)
      norm.weight.data.copy_(state_dict["norm.weight"].to(device=device, dtype=dtype))
      if capture_layers is None:
          layers_to_run = int(layers_to_run or config.num_hidden_layers)
          capture_set = set()
      else:
          capture_set = set(int(x) for x in capture_layers)
          layers_to_run = max(capture_set) + 1
      layers = []
      for i in range(layers_to_run):
          prefix = f"layers.{i}."
          layer_state = {
              key[len(prefix):]: value
              for key, value in state_dict.items()
              if key.startswith(prefix)
          }
          layer = Qwen3VLTextDecoderLayer(config, i).to(device=device, dtype=dtype)
          layers.append(_load_text_module(layer, layer_state, device, dtype))
      token_ids = make_text_token_ids(int(token_count), config.vocab_size).to(device)
      batch, token_count = token_ids.shape
      hidden_states = embed(token_ids)
      cache_position = torch.arange(token_count, device=device)
      position_ids = cache_position.view(1, 1, -1).expand(3, batch, -1)
      text_position_ids = position_ids[0]
      attention_mask_2d = torch.ones((batch, token_count), dtype=torch.long, device=device)
      causal_mask = create_causal_mask(
          config=config,
          input_embeds=hidden_states,
          attention_mask=attention_mask_2d,
          cache_position=cache_position,
          past_key_values=None,
          position_ids=text_position_ids,
      )
      position_embeddings = rotary_emb(hidden_states, position_ids)
      captured = {}
      for layer_idx, decoder_layer in enumerate(layers):
          hidden_states = decoder_layer(
              hidden_states,
              attention_mask=causal_mask,
              position_ids=text_position_ids,
              past_key_values=None,
              cache_position=cache_position,
              position_embeddings=position_embeddings,
          )
          if layer_idx in capture_set:
              captured[layer_idx] = hidden_states
      if capture_layers is None:
          return norm(hidden_states)[0].float().cpu().numpy()
      selected = [captured[int(i)] for i in capture_layers]
      return torch.cat(selected, dim=-1)[0].float().cpu().numpy()

  def run_ideogram_text_reference(root, state_dict, token_count, device_index, dtype_name):
      return _qwen_text_forward(
          f"{root}/text_encoder",
          state_dict,
          token_count,
          device_index,
          dtype_name,
          capture_layers=QWEN3_VL_ACTIVATION_LAYERS,
      )

  def run_original_qwen_text_reference(root, state_dict, token_count, layers, device_index, dtype_name):
      return _qwen_text_forward(
          root,
          state_dict,
          token_count,
          device_index,
          dtype_name,
          capture_layers=None,
          layers_to_run=int(layers),
      )

  def rotate_half(x):
      half = x.shape[-1] // 2
      return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

  def apply_rotary(q, k, cos, sin):
      cos = cos.unsqueeze(1)
      sin = sin.unsqueeze(1)
      return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

  class Ideogram4MRoPE(nn.Module):
      def __init__(self, head_dim=256, base=5000000, mrope_section=(24, 20, 20)):
          super().__init__()
          inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
          self.register_buffer("inv_freq", inv_freq, persistent=False)
          self.mrope_section = tuple(mrope_section)

      @torch.no_grad()
      def forward(self, position_ids):
          batch_size, seq_len, _ = position_ids.shape
          pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
          inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)
          freqs = inv_freq @ pos.unsqueeze(2)
          freqs = freqs.transpose(2, 3)
          freqs_t = freqs[0].clone()
          for axis, offset in ((1, 1), (2, 2)):
              length = self.mrope_section[axis] * 3
              idx = torch.arange(offset, length, 3, device=freqs_t.device)
              freqs_t[..., idx] = freqs[axis][..., idx]
          emb = torch.cat((freqs_t, freqs_t), dim=-1)
          return emb.cos(), emb.sin()

  class RMSNorm(nn.Module):
      def __init__(self, dim, eps=1e-6):
          super().__init__()
          self.weight = nn.Parameter(torch.ones(dim))
          self.eps = eps

      def forward(self, x):
          return F.rms_norm(x, self.weight.shape, self.weight, self.eps)

  class Ideogram4Attention(nn.Module):
      def __init__(self, hidden_size=4608, num_heads=18, eps=1e-5):
          super().__init__()
          self.hidden_size = hidden_size
          self.num_heads = num_heads
          self.head_dim = hidden_size // num_heads
          self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
          self.norm_q = RMSNorm(self.head_dim, eps=eps)
          self.norm_k = RMSNorm(self.head_dim, eps=eps)
          self.o = nn.Linear(hidden_size, hidden_size, bias=False)

      def forward(self, x, segment_ids, cos, sin):
          batch_size, seq_len, _ = x.shape
          qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
          q, k, v = qkv.unbind(dim=2)
          q = self.norm_q(q).transpose(1, 2)
          k = self.norm_k(k).transpose(1, 2)
          v = v.transpose(1, 2)
          q, k = apply_rotary(q, k, cos, sin)
          attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
          out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
          out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
          return self.o(out)

  class Ideogram4MLP(nn.Module):
      def __init__(self, dim=4608, hidden_dim=12288):
          super().__init__()
          self.w1 = nn.Linear(dim, hidden_dim, bias=False)
          self.w2 = nn.Linear(hidden_dim, dim, bias=False)
          self.w3 = nn.Linear(dim, hidden_dim, bias=False)

      def forward(self, x):
          return self.w2(F.silu(self.w1(x)) * self.w3(x))

  class Ideogram4TransformerBlock(nn.Module):
      def __init__(self):
          super().__init__()
          self.attention = Ideogram4Attention()
          self.feed_forward = Ideogram4MLP()
          self.attention_norm1 = RMSNorm(4608, eps=1e-5)
          self.ffn_norm1 = RMSNorm(4608, eps=1e-5)
          self.attention_norm2 = RMSNorm(4608, eps=1e-5)
          self.ffn_norm2 = RMSNorm(4608, eps=1e-5)
          self.adaln_modulation = nn.Linear(512, 4 * 4608, bias=True)

      def forward(self, x, segment_ids, cos, sin, adaln_input):
          mod = self.adaln_modulation(adaln_input)
          scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
          scale_msa = 1.0 + scale_msa
          scale_mlp = 1.0 + scale_mlp
          gate_msa = torch.tanh(gate_msa)
          gate_mlp = torch.tanh(gate_mlp)
          attn_out = self.attention(
              self.attention_norm1(x) * scale_msa,
              segment_ids=segment_ids,
              cos=cos,
              sin=sin,
          )
          x = x + gate_msa * self.attention_norm2(attn_out)
          x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
          return x

  def sinusoidal_embedding(t, dim=4608, scale=1e4):
      t = t.to(torch.float32)
      half = dim // 2
      freq = math.log(scale) / (half - 1)
      freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
      emb = t.unsqueeze(-1) * freq
      return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

  class Ideogram4EmbedScalar(nn.Module):
      def __init__(self):
          super().__init__()
          self.mlp_in = nn.Linear(4608, 4608, bias=True)
          self.mlp_out = nn.Linear(4608, 4608, bias=True)

      def forward(self, x):
          emb = sinusoidal_embedding(1e4 * x, 4608)
          emb = emb.to(getattr(self.mlp_in, "compute_dtype", None) or self.mlp_in.weight.dtype)
          return self.mlp_out(F.silu(self.mlp_in(emb)))

  class Ideogram4FinalLayer(nn.Module):
      def __init__(self):
          super().__init__()
          self.norm_final = nn.LayerNorm(4608, eps=1e-6, elementwise_affine=False)
          self.linear = nn.Linear(4608, 128, bias=True)
          self.adaln_modulation = nn.Linear(512, 4608, bias=True)

      def forward(self, x, c):
          scale = 1.0 + self.adaln_modulation(F.silu(c))
          return self.linear(self.norm_final(x) * scale)

  class Ideogram4Transformer(nn.Module):
      def __init__(self):
          super().__init__()
          self.input_proj = nn.Linear(128, 4608, bias=True)
          self.llm_cond_norm = RMSNorm(53248, eps=1e-6)
          self.llm_cond_proj = nn.Linear(53248, 4608, bias=True)
          self.t_embedding = Ideogram4EmbedScalar()
          self.adaln_proj = nn.Linear(4608, 512, bias=True)
          self.embed_image_indicator = nn.Embedding(2, 4608)
          self.rotary_emb = Ideogram4MRoPE()
          self.layers = nn.ModuleList([Ideogram4TransformerBlock() for _ in range(34)])
          self.final_layer = Ideogram4FinalLayer()

      def forward(self, *, llm_features, x, t, position_ids, segment_ids, indicator):
          param_dtype = getattr(self.input_proj, "compute_dtype", None) or self.input_proj.weight.dtype
          x = x.to(param_dtype)
          t = t.to(param_dtype)
          llm_features = llm_features.to(param_dtype)
          indicator = indicator.to(torch.long)
          llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
          output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)
          llm_features = llm_features * llm_token_mask
          x = x * output_image_mask
          x = self.input_proj(x) * output_image_mask
          t_cond = self.t_embedding(t)
          if t.dim() == 1:
              t_cond = t_cond.unsqueeze(1)
          adaln_input = F.silu(self.adaln_proj(t_cond))
          llm_features = self.llm_cond_proj(self.llm_cond_norm(llm_features)) * llm_token_mask
          h = x + llm_features
          h = h + self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
          cos, sin = self.rotary_emb(position_ids)
          cos = cos.to(h.dtype)
          sin = sin.to(h.dtype)
          for layer in self.layers:
              h = layer(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)
          return self.final_layer(h, c=adaln_input).to(torch.float32)

  def load_transformer_pack(root, subfolder, device_index, dtype_name):
      dtype = ref_dtype(dtype_name)
      device = torch.device(f"cuda:{device_index}")
      state_dict = load_file(f"{root}/{subfolder}/diffusion_pytorch_model.safetensors")
      model = Ideogram4Transformer()
      model.to(dtype)
      swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
      load_fp8_state_dict(model, state_dict, device=device, dtype=dtype)
      model.eval()
      return {"model": model, "state_dict": state_dict}

  def make_position_ids(text_len, grid_h, grid_w):
      image = []
      for y in range(grid_h):
          for x in range(grid_w):
              image.append([IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET + y, IMAGE_POSITION_OFFSET + x])
      text = [[i, i, i] for i in range(text_len)]
      return torch.tensor([text + image], dtype=torch.long)

  def run_transformer_case(model, text_len, grid_h, grid_w, t_value, device_index, dtype_name):
      dtype = ref_dtype(dtype_name)
      device = torch.device(f"cuda:{device_index}")
      torch.manual_seed(1234 + text_len * 17 + grid_h * 31 + grid_w)
      image_len = grid_h * grid_w
      total = text_len + image_len
      text_features = torch.randn((1, text_len, 53248), dtype=torch.float32, device=device) * 0.01
      x_image = torch.randn((1, image_len, 128), dtype=torch.float32, device=device)
      llm_features = torch.zeros((1, total, 53248), dtype=torch.float32, device=device)
      x = torch.zeros((1, total, 128), dtype=torch.float32, device=device)
      if text_len:
          llm_features[:, :text_len] = text_features
      x[:, text_len:] = x_image
      position_ids = make_position_ids(text_len, grid_h, grid_w).to(device)
      segment_ids = torch.ones((1, total), dtype=torch.long, device=device)
      indicator = torch.full((1, total), OUTPUT_IMAGE_INDICATOR, dtype=torch.long, device=device)
      if text_len:
          indicator[:, :text_len] = LLM_TOKEN_INDICATOR
      t = torch.full((1,), float(t_value), dtype=torch.float32, device=device)
      with torch.no_grad():
          param_dtype = getattr(model.input_proj, "compute_dtype", None) or model.input_proj.weight.dtype
          x_model = x.to(param_dtype)
          t_model = t.to(param_dtype)
          llm_model = llm_features.to(param_dtype)
          llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x_model.dtype).unsqueeze(-1)
          output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x_model.dtype).unsqueeze(-1)
          llm_model = llm_model * llm_token_mask
          x_model = x_model * output_image_mask
          x_model = model.input_proj(x_model) * output_image_mask
          t_cond = model.t_embedding(t_model)
          if t.dim() == 1:
              t_cond = t_cond.unsqueeze(1)
          adaln_input = F.silu(model.adaln_proj(t_cond))
          llm_model = model.llm_cond_proj(model.llm_cond_norm(llm_model)) * llm_token_mask
          h = x_model + llm_model
          h = h + model.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
          stem = h.clone()
          cos, sin = model.rotary_emb(position_ids)
          cos = cos.to(h.dtype)
          sin = sin.to(h.dtype)
          layer0 = model.layers[0](h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)
      out = model(
          llm_features=llm_features,
          x=x,
          t=t,
          position_ids=position_ids,
          segment_ids=segment_ids,
          indicator=indicator,
      )
      return {
          "text_features": text_features[0].float().cpu().numpy(),
          "x_image": x_image[0].float().cpu().numpy(),
          "position_ids": position_ids[0].cpu().numpy(),
          "indicator": (indicator[0] == OUTPUT_IMAGE_INDICATOR).long().cpu().numpy(),
          "stem": stem[0].float().cpu().numpy(),
          "layer0": layer0[0].float().cpu().numpy(),
          "reference": out[0, text_len:].float().cpu().numpy(),
      }

  def run_transformer_prefix_case(model, text_len, grid_h, grid_w, t_value, layers_to_run, device_index, dtype_name):
      dtype = ref_dtype(dtype_name)
      device = torch.device(f"cuda:{device_index}")
      torch.manual_seed(1234 + text_len * 17 + grid_h * 31 + grid_w)
      image_len = grid_h * grid_w
      total = text_len + image_len
      text_features = torch.randn((1, text_len, 53248), dtype=torch.float32, device=device) * 0.01
      x_image = torch.randn((1, image_len, 128), dtype=torch.float32, device=device)
      llm_features = torch.zeros((1, total, 53248), dtype=torch.float32, device=device)
      x = torch.zeros((1, total, 128), dtype=torch.float32, device=device)
      if text_len:
          llm_features[:, :text_len] = text_features
      x[:, text_len:] = x_image
      position_ids = make_position_ids(text_len, grid_h, grid_w).to(device)
      segment_ids = torch.ones((1, total), dtype=torch.long, device=device)
      indicator = torch.full((1, total), OUTPUT_IMAGE_INDICATOR, dtype=torch.long, device=device)
      if text_len:
          indicator[:, :text_len] = LLM_TOKEN_INDICATOR
      t = torch.full((1,), float(t_value), dtype=torch.float32, device=device)
      with torch.no_grad():
          param_dtype = getattr(model.input_proj, "compute_dtype", None) or model.input_proj.weight.dtype
          x_model = x.to(param_dtype)
          t_model = t.to(param_dtype)
          llm_model = llm_features.to(param_dtype)
          llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x_model.dtype).unsqueeze(-1)
          output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x_model.dtype).unsqueeze(-1)
          llm_model = llm_model * llm_token_mask
          x_model = x_model * output_image_mask
          x_model = model.input_proj(x_model) * output_image_mask
          t_cond = model.t_embedding(t_model)
          if t.dim() == 1:
              t_cond = t_cond.unsqueeze(1)
          adaln_input = F.silu(model.adaln_proj(t_cond))
          llm_model = model.llm_cond_proj(model.llm_cond_norm(llm_model)) * llm_token_mask
          h = x_model + llm_model
          h = h + model.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
          if layers_to_run > 0:
              cos, sin = model.rotary_emb(position_ids)
              cos = cos.to(h.dtype)
              sin = sin.to(h.dtype)
              for layer in model.layers[:layers_to_run]:
                  h = layer(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)
      return {
          "text_features": text_features[0].float().cpu().numpy(),
          "x_image": x_image[0].float().cpu().numpy(),
          "indicator": (indicator[0] == OUTPUT_IMAGE_INDICATOR).long().cpu().numpy(),
          "reference": h[0].float().cpu().numpy(),
      }
  """#,
  helper.__dict__)

func copiedToCPU(_ tensor: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
  tensor.as(of: Float.self).rawValue.toCPU()
}

func copiedToCPU(_ tensor: DynamicGraph.Tensor<Float16>) -> Tensor<Float> {
  Tensor<Float>(from: tensor.as(of: Float16.self).rawValue.toCPU())
}

func maxAbsDiff2D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var maxDiff: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      maxDiff = max(maxDiff, abs(Float(lhs[i, j]) - Float(rhs[i, j])))
    }
  }
  return maxDiff
}

func maxRelativeDiff2D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  var maxAbsDiff: Float = 0
  var maxAbsRef: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let ref = Float(rhs[i, j])
      maxAbsDiff = max(maxAbsDiff, abs(Float(lhs[i, j]) - ref))
      maxAbsRef = max(maxAbsRef, abs(ref))
    }
  }
  return maxAbsDiff / max(maxAbsRef, 1e-6)
}

func maxAbsDiff2D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>, rhsColumnOffset: Int) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(rhsColumnOffset + lhs.shape[1] <= rhs.shape[1])
  var maxDiff: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      maxDiff = max(maxDiff, abs(Float(lhs[i, j]) - Float(rhs[i, j + rhsColumnOffset])))
    }
  }
  return maxDiff
}

func maxRelativeDiff2D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>, rhsColumnOffset: Int) -> Float {
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(rhsColumnOffset + lhs.shape[1] <= rhs.shape[1])
  var maxAbsDiff: Float = 0
  var maxAbsRef: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      let ref = Float(rhs[i, j + rhsColumnOffset])
      maxAbsDiff = max(maxAbsDiff, abs(Float(lhs[i, j]) - ref))
      maxAbsRef = max(maxAbsRef, abs(ref))
    }
  }
  return maxAbsDiff / max(maxAbsRef, 1e-6)
}

func maxAbsDiffRow(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>, row: Int, rhsColumnOffset: Int)
  -> Float
{
  precondition(lhs.shape.count == 2)
  precondition(rhs.shape.count == 2)
  precondition(row < lhs.shape[0])
  precondition(row < rhs.shape[0])
  precondition(rhsColumnOffset + lhs.shape[1] <= rhs.shape[1])
  var maxDiff: Float = 0
  for j in 0..<lhs.shape[1] {
    maxDiff = max(maxDiff, abs(Float(lhs[row, j]) - Float(rhs[row, j + rhsColumnOffset])))
  }
  return maxDiff
}

func tensorFromPython(_ object: PythonObject) -> Tensor<Float> {
  try! Tensor<Float>(numpy: object)
}

func dequantWeight(_ stateDict: PythonObject, _ key: String) -> Tensor<Float> {
  tensorFromPython(helper.dequant_weight_np(stateDict, key, dtypeName))
}

func dequantInterleavedQKWeight(
  _ stateDict: PythonObject, _ key: String, heads: Int, headDim: Int
) -> Tensor<Float> {
  tensorFromPython(
    helper.dequant_interleaved_qk_weight_np(stateDict, key, heads, headDim, dtypeName))
}

func dequantTextWeight(_ stateDict: PythonObject, _ key: String) -> Tensor<Float> {
  tensorFromPython(helper.dequant_weight_np(stateDict, key, textDtypeName))
}

func dequantTextInterleavedQKWeight(
  _ stateDict: PythonObject, _ key: String, heads: Int, headDim: Int
) -> Tensor<Float> {
  tensorFromPython(
    helper.dequant_interleaved_qk_weight_np(stateDict, key, heads, headDim, textDtypeName))
}

func interleavedQKNorm(_ stateDict: PythonObject, _ key: String, headDim: Int) -> Tensor<Float> {
  tensorFromPython(helper.interleaved_qk_norm_np(stateDict, key, headDim))
}

func tensorValue(_ stateDict: PythonObject, _ key: String) -> Tensor<Float> {
  tensorFromPython(helper.tensor_np(stateDict, key))
}

func copyDenseWeight(_ dense: Model, _ stateDict: PythonObject, _ key: String) {
  dense.weight.copy(from: Tensor<FloatType>(from: dequantWeight(stateDict, key)))
  dense.weight.to(.unifiedMemory)
}

func copyDenseBias(_ dense: Model, _ stateDict: PythonObject, _ key: String) {
  dense.bias.copy(from: Tensor<FloatType>(from: tensorValue(stateDict, key)))
}

func copyDense(_ dense: Model, _ stateDict: PythonObject, weight: String, bias: String? = nil) {
  copyDenseWeight(dense, stateDict, weight)
  if let bias {
    copyDenseBias(dense, stateDict, bias)
  }
}

func copyTextDenseWeight(_ dense: Model, _ stateDict: PythonObject, _ key: String) {
  dense.weight.copy(from: Tensor<TextFloatType>(from: dequantTextWeight(stateDict, key)))
  dense.weight.to(.unifiedMemory)
}

func ideogramTimestepEmbedding(_ timestep: Float, dim: Int = 4_608) -> Tensor<Float> {
  precondition(dim % 2 == 0)
  let half = dim / 2
  var out = Tensor<Float>(.CPU, .WC(1, dim))
  let scaled = 10_000 * timestep
  let freqScale = log(Float(10_000)) / Float(half - 1)
  for i in 0..<half {
    let freq = exp(Float(i) * -freqScale)
    let value = scaled * freq
    out[0, i] = sin(value)
    out[0, i + half] = cos(value)
  }
  return out
}

func mropeAngles(position: (Int, Int, Int), headDim: Int, theta: Double = 5_000_000.0)
  -> [Double]
{
  let half = headDim / 2
  var angles = [Double](repeating: 0, count: half)
  for i in 0..<half {
    angles[i] = Double(position.0) / pow(theta, Double(i * 2) / Double(headDim))
  }
  let positions = [position.0, position.1, position.2]
  let sections = [24, 20, 20]
  for axis in 1...2 {
    let length = sections[axis] * 3
    var i = axis
    while i < length {
      angles[i] = Double(positions[axis]) / pow(theta, Double(i * 2) / Double(headDim))
      i += 3
    }
  }
  return angles
}

func makeRotary(positionIDs: [(Int, Int, Int)], headDim: Int) -> (Tensor<Float>, Tensor<Float>) {
  let tokenLength = positionIDs.count
  let half = headDim / 2
  var cosTensor = Tensor<Float>(.CPU, .NHWC(1, tokenLength, 1, headDim))
  var sinTensor = Tensor<Float>(.CPU, .NHWC(1, tokenLength, 1, headDim))
  for i in 0..<tokenLength {
    let angles = mropeAngles(position: positionIDs[i], headDim: headDim)
    for k in 0..<half {
      let c = Float(cos(angles[k]))
      let s = Float(sin(angles[k]))
      cosTensor[0, i, 0, k] = c
      cosTensor[0, i, 0, k + half] = c
      sinTensor[0, i, 0, k] = s
      sinTensor[0, i, 0, k + half] = s
    }
  }
  return (cosTensor, sinTensor)
}

func makeQwenTextRotary(tokenLength: Int) -> Tensor<Float> {
  let half = 64
  var rotary = Tensor<Float>(.CPU, .NHWC(1, tokenLength, 1, 128))
  for i in 0..<tokenLength {
    let angles = mropeAngles(position: (i, i, i), headDim: 128)
    for k in 0..<half {
      rotary[0, i, 0, k * 2] = Float(cos(angles[k]))
      rotary[0, i, 0, k * 2 + 1] = Float(sin(angles[k]))
    }
  }
  return rotary
}

func makeIdeogramRotary(textLength: Int, gridHeight: Int, gridWidth: Int) -> (
  Tensor<Float>, Tensor<Float>
) {
  var positions = [(Int, Int, Int)]()
  for i in 0..<textLength {
    positions.append((i, i, i))
  }
  for y in 0..<gridHeight {
    for x in 0..<gridWidth {
      positions.append((65_536, 65_536 + y, 65_536 + x))
    }
  }
  return makeRotary(positionIDs: positions, headDim: 256)
}

func applyRotaryHalf(
  _ x: Model.IO, cos: Model.IO, sin: Model.IO, tokenLength: Int, heads: Int, headDim: Int
) -> Model.IO {
  let halfDim = headDim / 2
  let firstHalf = x.reshaped(
    [1, tokenLength, heads, halfDim], offset: [0, 0, 0, 0],
    strides: [tokenLength * heads * headDim, heads * headDim, headDim, 1]
  ).copied()
  let secondHalf = x.reshaped(
    [1, tokenLength, heads, halfDim], offset: [0, 0, 0, halfDim],
    strides: [tokenLength * heads * headDim, heads * headDim, headDim, 1]
  ).copied()
  let rotated = Functional.concat(axis: 3, (-secondHalf).copied(), firstHalf).copied()
  return x .* cos + rotated .* sin
}

func SelfAttention(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = tovalues(x).reshaped([b, t, hk, k])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([b * t, h * k])
  let unifyheads = Dense(count: width, noBias: true, name: "out_proj")
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { stateDict in
    toqueries.weight.copy(
      from: Tensor<TextFloatType>(
        from: dequantTextInterleavedQKWeight(
          stateDict, "\(prefix).self_attn.q_proj.weight", heads: h, headDim: k)))
    let qNormWeight = tensorFromPython(
      helper.interleaved_qk_norm_np(stateDict, "\(prefix).self_attn.q_norm.weight", k))
    normQ.weight.copy(from: Tensor<TextFloatType>(from: qNormWeight))
    tokeys.weight.copy(
      from: Tensor<TextFloatType>(
        from: dequantTextInterleavedQKWeight(
          stateDict, "\(prefix).self_attn.k_proj.weight", heads: hk, headDim: k)))
    let kNormWeight = tensorFromPython(
      helper.interleaved_qk_norm_np(stateDict, "\(prefix).self_attn.k_norm.weight", k))
    normK.weight.copy(from: Tensor<TextFloatType>(from: kNormWeight))
    copyTextDenseWeight(tovalues, stateDict, "\(prefix).self_attn.v_proj.weight")
    copyTextDenseWeight(unifyheads, stateDict, "\(prefix).self_attn.o_proj.weight")
    toqueries.weight.to(.unifiedMemory)
    tokeys.weight.to(.unifiedMemory)
  }
  return (Model([x, rot], [out]), reader)
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

func TransformerBlock(prefix: String, width: Int, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int)
  -> (
    Model, (PythonObject) -> Void
  )
{
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "input_layernorm")
  var out = norm1(x)
  let (attention, attnReader) = SelfAttention(
    prefix: prefix, width: width, k: k, h: h, hk: hk, b: b, t: t)
  out = attention(out, rot) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "post_attention_layernorm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: width, intermediateSize: MLP, name: "mlp")
  out = residual + ffn(out)
  let reader: (PythonObject) -> Void = { stateDict in
    attnReader(stateDict)
    norm1.weight.copy(
      from: Tensor<TextFloatType>(from: tensorValue(stateDict, "\(prefix).input_layernorm.weight")))
    norm2.weight.copy(
      from: Tensor<TextFloatType>(
        from: tensorValue(stateDict, "\(prefix).post_attention_layernorm.weight")))
    copyTextDenseWeight(w1, stateDict, "\(prefix).mlp.gate_proj.weight")
    copyTextDenseWeight(w2, stateDict, "\(prefix).mlp.down_proj.weight")
    copyTextDenseWeight(w3, stateDict, "\(prefix).mlp.up_proj.weight")
  }
  return (Model([x, rot], [out]), reader)
}

func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { stateDict in
    tokenEmbed.parameters.copy(
      from: Tensor<TextFloatType>(from: tensorValue(stateDict, "embed_tokens.weight")))
    tokenEmbed.parameters.to(.unifiedMemory)
  }
  return (Model([tokens], [embedding]), reader)
}

func Transformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, outputHiddenStates: Int?, batchSize: Int
) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embedReader) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(tokens)
  var readers = [(PythonObject) -> Void]()
  var hiddenStates: Model.IO? = nil
  for i in 0..<layers {
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", width: width, k: 128, h: heads, hk: 8, b: batchSize,
      t: tokenLength, MLP: MLP)
    out = layer(out, rot)
    readers.append(reader)
    if let outputHiddenStates = outputHiddenStates, outputHiddenStates == i {
      hiddenStates = out
    }
  }
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm")
  out = norm(out)
  let reader: (PythonObject) -> Void = { stateDict in
    embedReader(stateDict)
    for reader in readers {
      reader(stateDict)
    }
    norm.weight.copy(from: Tensor<TextFloatType>(from: tensorValue(stateDict, "norm.weight")))
  }
  return (Model([tokens, rot], (hiddenStates.map { [$0] } ?? []) + [out]), reader)
}

func QwenTextFeatures(tokenLength: Int) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embedReader) = TextEmbedding(
    TextFloatType.self, batchSize: 1, vocabularySize: 151_936, maxLength: tokenLength,
    embeddingSize: 4_096)
  var out = embedding(tokens)
  let captureLayers: Set<Int> = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]
  var captured = [Model.IO]()
  var readers = [(PythonObject) -> Void]()
  for i in 0..<36 {
    let (layer, reader) = TransformerBlock(
      prefix: "layers.\(i)", width: 4_096, k: 128, h: 32, hk: 8, b: 1,
      t: tokenLength, MLP: 12_288)
    out = layer(out, rot)
    readers.append(reader)
    if captureLayers.contains(i) {
      captured.append(out.to(.Float32))
    }
  }
  let reader: (PythonObject) -> Void = { stateDict in
    embedReader(stateDict)
    for reader in readers {
      reader(stateDict)
    }
  }
  return (Model([tokens, rot], captured), reader)
}

func Ideogram4Attention(prefix: String, tokenLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rotCos = Input()
  let rotSin = Input()
  let q = Dense(count: 4_608, noBias: true, name: "q")
  let k = Dense(count: 4_608, noBias: true, name: "k")
  let v = Dense(count: 4_608, noBias: true, name: "v")
  var queries = q(x).reshaped([1, tokenLength, 18, 256])
  let normQ = RMSNorm(epsilon: 1e-5, axis: [3], name: "norm_q")
  queries = normQ(queries)
  var keys = k(x).reshaped([1, tokenLength, 18, 256])
  let normK = RMSNorm(epsilon: 1e-5, axis: [3], name: "norm_k")
  keys = normK(keys)
  let values = v(x).reshaped([1, tokenLength, 18, 256])
  queries = applyRotaryHalf(
    queries, cos: rotCos, sin: rotSin, tokenLength: tokenLength, heads: 18, headDim: 256)
  keys = applyRotaryHalf(
    keys, cos: rotCos, sin: rotSin, tokenLength: tokenLength, heads: 18, headDim: 256)
  keys = keys.permuted(0, 2, 1, 3)
  queries = ((1.0 / Float(256).squareRoot()) * queries).permuted(0, 2, 1, 3)
  let attentionValues = values.permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([18 * tokenLength, tokenLength])
  dot = dot.softmax()
  dot = dot.reshaped([1, 18, tokenLength, tokenLength])
  var out = dot * attentionValues
  out = out.reshaped([1, 18, tokenLength, 256]).transposed(1, 2).reshaped([tokenLength, 4_608])
  let o = Dense(count: 4_608, noBias: true, name: "o")
  out = o(out)
  let reader: (PythonObject) -> Void = { stateDict in
    let qkv = dequantWeight(stateDict, "\(prefix).attention.qkv.weight")
    q.weight.copy(from: Tensor<FloatType>(from: qkv[0..<4_608, 0..<4_608]))
    k.weight.copy(from: Tensor<FloatType>(from: qkv[4_608..<(4_608 * 2), 0..<4_608]))
    v.weight.copy(from: Tensor<FloatType>(from: qkv[(4_608 * 2)..<(4_608 * 3), 0..<4_608]))
    q.weight.to(.unifiedMemory)
    k.weight.to(.unifiedMemory)
    v.weight.to(.unifiedMemory)
    copyDenseWeight(o, stateDict, "\(prefix).attention.o.weight")
    normQ.weight.copy(
      from: Tensor<FloatType>(from: tensorValue(stateDict, "\(prefix).attention.norm_q.weight")))
    normK.weight.copy(
      from: Tensor<FloatType>(from: tensorValue(stateDict, "\(prefix).attention.norm_k.weight")))
  }
  return (Model([x, rotCos, rotSin], [out]), reader)
}

func Ideogram4MLP(prefix: String) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let w1 = Dense(count: 12_288, noBias: true, name: "w1")
  let w3 = Dense(count: 12_288, noBias: true, name: "w3")
  let w2 = Dense(count: 4_608, noBias: true, name: "w2")
  let out = w2(w1(x).swish() .* w3(x))
  let reader: (PythonObject) -> Void = { stateDict in
    copyDenseWeight(w1, stateDict, "\(prefix).feed_forward.w1.weight")
    copyDenseWeight(w2, stateDict, "\(prefix).feed_forward.w2.weight")
    copyDenseWeight(w3, stateDict, "\(prefix).feed_forward.w3.weight")
  }
  return (Model([x], [out]), reader)
}

func Ideogram4Block(prefix: String, tokenLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rotCos = Input()
  let rotSin = Input()
  let adaln = Input()
  let modulation = Dense(count: 4 * 4_608, name: "adaln_modulation")
  let mod = modulation(adaln)
  let scaleMSA = 1 + mod.reshaped([1, 4_608], offset: [0, 0], strides: [4 * 4_608, 1])
  let gateMSA = mod.reshaped([1, 4_608], offset: [0, 4_608], strides: [4 * 4_608, 1]).tanh()
  let scaleMLP = 1 + mod.reshaped([1, 4_608], offset: [0, 4_608 * 2], strides: [4 * 4_608, 1])
  let gateMLP = mod.reshaped([1, 4_608], offset: [0, 4_608 * 3], strides: [4 * 4_608, 1]).tanh()

  let attentionNorm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm1")
  let attnIn = (attentionNorm1(x) .* scaleMSA).to(FloatType.dataType)
  let (attention, attentionReader) = Ideogram4Attention(prefix: prefix, tokenLength: tokenLength)
  let attentionNorm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm2")
  var out = x + gateMSA.to(of: x) .* attentionNorm2(attention(attnIn, rotCos, rotSin)).to(of: x)
  let ffnNorm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm1")
  let (mlp, mlpReader) = Ideogram4MLP(prefix: prefix)
  let ffnNorm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm2")
  out =
    out
    + gateMLP.to(of: out)
    .* ffnNorm2(mlp((ffnNorm1(out) .* scaleMLP).to(FloatType.dataType))).to(of: out)
  let reader: (PythonObject) -> Void = { stateDict in
    copyDense(
      modulation, stateDict, weight: "\(prefix).adaln_modulation.weight",
      bias: "\(prefix).adaln_modulation.bias")
    attentionReader(stateDict)
    mlpReader(stateDict)
    attentionNorm1.weight.copy(
      from: Tensor<FloatType>(from: tensorValue(stateDict, "\(prefix).attention_norm1.weight")))
    attentionNorm2.weight.copy(
      from: Tensor<FloatType>(from: tensorValue(stateDict, "\(prefix).attention_norm2.weight")))
    ffnNorm1.weight.copy(
      from: Tensor<FloatType>(from: tensorValue(stateDict, "\(prefix).ffn_norm1.weight")))
    ffnNorm2.weight.copy(
      from: Tensor<FloatType>(from: tensorValue(stateDict, "\(prefix).ffn_norm2.weight")))
  }
  return (Model([x, rotCos, rotSin, adaln], [out]), reader)
}

func Ideogram4Transformer(
  textLength: Int, imageLength: Int, layersToRun: Int = 34, outputHidden: Bool = false
) -> (Model, (PythonObject) -> Void) {
  let usesAdaln = layersToRun > 0 || !outputHidden
  let usesRotary = layersToRun > 0
  let xImage = Input()
  let textFeatures: Input? = textLength > 0 ? Input() : nil
  let indicator = Input()
  let rotCos: Input? = usesRotary ? Input() : nil
  let rotSin: Input? = usesRotary ? Input() : nil
  let tEmbed: Input? = usesAdaln ? Input() : nil
  let inputProj = Dense(count: 4_608, name: "input_proj")
  let imageOut = inputProj(xImage)
  let llmNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "llm_cond_norm")
  let llmProj = Dense(count: 4_608, name: "llm_cond_proj")
  let indicatorEmbed = Embedding(
    FloatType.self, vocabularySize: 2, embeddingSize: 4_608, name: "embed_image_indicator")

  let tokenLength = textLength + imageLength
  var out: Model.IO
  if let textFeatures {
    let textOut = llmProj(llmNorm(textFeatures).to(FloatType.dataType))
    out = Functional.concat(axis: 0, textOut, imageOut)
  } else {
    out = imageOut
  }
  out = (out + indicatorEmbed(indicator)).to(FloatType.dataType)

  let tMlpIn: Model?
  let tMlpOut: Model?
  let adalnProj: Model?
  let adaln: Model.IO?
  if let tEmbed {
    let mlpIn = Dense(count: 4_608, name: "t_embedding_mlp_in")
    let mlpOut = Dense(count: 4_608, name: "t_embedding_mlp_out")
    let proj = Dense(count: 512, name: "adaln_proj")
    tMlpIn = mlpIn
    tMlpOut = mlpOut
    adalnProj = proj
    adaln = proj(mlpOut(mlpIn(tEmbed).swish())).swish()
  } else {
    tMlpIn = nil
    tMlpOut = nil
    adalnProj = nil
    adaln = nil
  }

  var readers = [(PythonObject) -> Void]()
  for i in 0..<layersToRun {
    let (block, reader) = Ideogram4Block(prefix: "layers.\(i)", tokenLength: tokenLength)
    out = block(out, rotCos!, rotSin!, adaln!)
    readers.append(reader)
  }

  let finalAdaln: Model?
  let finalLinear: Model?
  let modelOutput: Model.IO
  if outputHidden {
    finalAdaln = nil
    finalLinear = nil
    modelOutput = out.to(.Float32)
  } else {
    let norm = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
    let finalAdalnLayer = Dense(count: 4_608, name: "final_adaln")
    let linear = Dense(count: 128, name: "final_linear")
    out = linear((norm(out) .* (1 + finalAdalnLayer(adaln!.swish()))).to(FloatType.dataType)).to(
      .Float32)
    modelOutput = out.reshaped(
      [imageLength, 128], offset: [textLength, 0], strides: [128, 1]
    ).copied()
    finalAdaln = finalAdalnLayer
    finalLinear = linear
  }

  let reader: (PythonObject) -> Void = { stateDict in
    copyDense(inputProj, stateDict, weight: "input_proj.weight", bias: "input_proj.bias")
    if textLength > 0 {
      copyDense(llmProj, stateDict, weight: "llm_cond_proj.weight", bias: "llm_cond_proj.bias")
      llmNorm.weight.copy(
        from: Tensor<FloatType>(from: tensorValue(stateDict, "llm_cond_norm.weight")))
    }
    indicatorEmbed.parameters.copy(
      from: Tensor<FloatType>(from: tensorValue(stateDict, "embed_image_indicator.weight")))
    indicatorEmbed.parameters.to(.unifiedMemory)
    if let tMlpIn, let tMlpOut, let adalnProj {
      copyDense(
        tMlpIn, stateDict, weight: "t_embedding.mlp_in.weight", bias: "t_embedding.mlp_in.bias")
      copyDense(
        tMlpOut, stateDict, weight: "t_embedding.mlp_out.weight", bias: "t_embedding.mlp_out.bias")
      copyDense(adalnProj, stateDict, weight: "adaln_proj.weight", bias: "adaln_proj.bias")
    }
    for reader in readers {
      reader(stateDict)
    }
    if let finalAdaln, let finalLinear {
      copyDense(
        finalAdaln, stateDict, weight: "final_layer.adaln_modulation.weight",
        bias: "final_layer.adaln_modulation.bias")
      copyDense(
        finalLinear, stateDict, weight: "final_layer.linear.weight", bias: "final_layer.linear.bias"
      )
    }
  }
  var inputs: [Model.IO] = [xImage]
  if let textFeatures {
    inputs.append(textFeatures)
  }
  inputs.append(indicator)
  if let rotCos, let rotSin {
    inputs += [rotCos, rotSin]
  }
  if let tEmbed {
    inputs.append(tEmbed)
  }
  return (Model(inputs, [modelOutput]), reader)
}

func runTextParity() -> Bool {
  print("ideogram4 text parity: start")
  let stateDict = helper.load_ideogram_text_state(modelRoot)
  let tokenIds = helper.make_text_token_ids(textTokenLength)
  let reference = tensorFromPython(
    helper.run_ideogram_text_reference(
      modelRoot, stateDict, textTokenLength, deviceID, textDtypeName))
  torch.cuda.empty_cache()
  let tokenIdsCPU = try! Tensor<Int32>(numpy: tokenIds[0].to(torch.int32).cpu().numpy())
  let tokenCount = tokenIdsCPU.shape[0]
  let rotCPU = makeQwenTextRotary(tokenLength: tokenCount)
  return graph.withNoGrad {
    let tokens = graph.variable(.CPU, format: .NHWC, shape: [tokenCount], of: Int32.self)
    for i in 0..<tokenCount {
      tokens[i] = tokenIdsCPU[i]
    }
    let tokensGPU = tokens.toGPU(deviceID)
    let rotGPU = graph.variable(Tensor<TextFloatType>(from: rotCPU).toGPU(deviceID))
    let (model, reader) = QwenTextFeatures(tokenLength: tokenCount)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: tokensGPU, rotGPU)
    reader(stateDict)
    let outputs = model(inputs: tokensGPU, rotGPU)
    print("ideogram4 text tokens:", tokenCount)
    print("ideogram4 text output count:", outputs.count, "reference shape:", reference.shape)
    var maxAbs: Float = 0
    var maxRel: Float = 0
    for i in 0..<outputs.count {
      let swift = copiedToCPU(outputs[i].as(of: Float.self))
      let absDiff = maxAbsDiff2D(swift, reference, rhsColumnOffset: i * 4_096)
      let relDiff = maxRelativeDiff2D(swift, reference, rhsColumnOffset: i * 4_096)
      print("ideogram4 text tap \(i) shape:", swift.shape)
      print("ideogram4 text tap \(i) max abs diff:", absDiff)
      print("ideogram4 text tap \(i) max rel diff:", relDiff)
      if tokenCount <= 4 {
        for row in 0..<tokenCount {
          print(
            "ideogram4 text tap \(i) token \(row) max abs diff:",
            maxAbsDiffRow(swift, reference, row: row, rhsColumnOffset: i * 4_096))
        }
      }
      maxAbs = max(maxAbs, absDiff)
      maxRel = max(maxRel, relDiff)
    }
    print("ideogram4 text max abs diff:", maxAbs)
    print("ideogram4 text max rel diff:", maxRel)
    return maxRel < 0.02
  }
}

func runOriginalQwenTextParity() -> Bool {
  print("ideogram4 original qwen text parity: start")
  let qwenRoot = env["QWEN_MODEL"] ?? "/slow/Data/Qwen3-VL-8B-Instruct"
  let layerCount = Int(env["IDEOGRAM4_TEXT_LAYERS"] ?? "36") ?? 36
  let stateDict = helper.load_original_qwen_text_state(qwenRoot, layerCount)
  let tokenIds = helper.make_text_token_ids(textTokenLength)
  let reference = tensorFromPython(
    helper.run_original_qwen_text_reference(
      qwenRoot, stateDict, textTokenLength, layerCount, deviceID, textDtypeName))
  torch.cuda.empty_cache()
  let tokenIdsCPU = try! Tensor<Int32>(numpy: tokenIds[0].to(torch.int32).cpu().numpy())
  let tokenCount = tokenIdsCPU.shape[0]
  let rotCPU = makeQwenTextRotary(tokenLength: tokenCount)
  return graph.withNoGrad {
    let tokens = graph.variable(.CPU, format: .NHWC, shape: [tokenCount], of: Int32.self)
    for i in 0..<tokenCount {
      tokens[i] = tokenIdsCPU[i]
    }
    let tokensGPU = tokens.toGPU(deviceID)
    let rotGPU = graph.variable(Tensor<TextFloatType>(from: rotCPU).toGPU(deviceID))
    let (model, reader) = Transformer(
      TextFloatType.self, vocabularySize: 151_936, maxLength: tokenCount, width: 4_096,
      tokenLength: tokenCount, layers: layerCount, MLP: 12_288, heads: 32,
      outputHiddenStates: nil, batchSize: 1)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: tokensGPU, rotGPU)
    reader(stateDict)
    let swift = copiedToCPU(model(inputs: tokensGPU, rotGPU)[0].as(of: TextFloatType.self))
    let maxAbs = maxAbsDiff2D(swift, reference)
    let maxRel = maxRelativeDiff2D(swift, reference)
    print("ideogram4 original qwen text layers:", layerCount)
    print("ideogram4 original qwen text shape:", swift.shape, reference.shape)
    print("ideogram4 original qwen text max abs diff:", maxAbs)
    print("ideogram4 original qwen text max rel diff:", maxRel)
    return maxAbs < 1 && maxRel < 0.02
  }
}

func runTransformerParity(subfolder: String, textLength: Int) -> Bool {
  let label = subfolder == "transformer" ? "conditional transformer" : "unconditional transformer"
  print("ideogram4 \(label) parity: start")
  let pack = helper.load_transformer_pack(modelRoot, subfolder, deviceID, dtypeName)
  let testCase = helper.run_transformer_case(
    pack["model"], textLength, ditGridHeight, ditGridWidth, ditTimestep, deviceID, dtypeName)
  pack["model"].to("cpu")
  torch.cuda.empty_cache()

  let imageLength = ditGridHeight * ditGridWidth
  let xImageCPU = tensorFromPython(testCase["x_image"])
  let textFeaturesCPU = tensorFromPython(testCase["text_features"])
  let reference = tensorFromPython(testCase["reference"])
  let indicatorCPU = try! Tensor<Int32>(numpy: testCase["indicator"].astype(numpy.int32))
  let (cosCPU, sinCPU) = makeIdeogramRotary(
    textLength: textLength, gridHeight: ditGridHeight, gridWidth: ditGridWidth)
  let tEmbedCPU = ideogramTimestepEmbedding(ditTimestep)
  let stateDict = pack["state_dict"]

  return graph.withNoGrad {
    let xImage = graph.variable(Tensor<FloatType>(from: xImageCPU).toGPU(deviceID))
      .reshaped(.WC(imageLength, 128))
    let indicator = graph.variable(
      .CPU, format: .NHWC, shape: [textLength + imageLength], of: Int32.self)
    for i in 0..<(textLength + imageLength) {
      indicator[i] = indicatorCPU[i]
    }
    let indicatorGPU = indicator.toGPU(deviceID)
    let cosGPU = graph.variable(Tensor<FloatType>(from: cosCPU).toGPU(deviceID))
    let sinGPU = graph.variable(Tensor<FloatType>(from: sinCPU).toGPU(deviceID))
    let tEmbedGPU = graph.variable(Tensor<FloatType>(from: tEmbedCPU).toGPU(deviceID))
      .reshaped(.WC(1, 4_608))
    let (model, reader) = Ideogram4Transformer(textLength: textLength, imageLength: imageLength)
    model.maxConcurrency = .limit(1)
    if textLength > 0 {
      let textFeatures = graph.variable(Tensor<FloatType>(from: textFeaturesCPU).toGPU(deviceID))
        .reshaped(.WC(textLength, 53_248))
      model.compile(inputs: xImage, textFeatures, indicatorGPU, cosGPU, sinGPU, tEmbedGPU)
      reader(stateDict)
      let swift = copiedToCPU(
        model(inputs: xImage, textFeatures, indicatorGPU, cosGPU, sinGPU, tEmbedGPU)[0].as(
          of: Float.self))
      print("ideogram4 \(label) shape:", swift.shape, reference.shape)
      let maxAbs = maxAbsDiff2D(swift, reference)
      let maxRel = maxRelativeDiff2D(swift, reference)
      print("ideogram4 \(label) max abs diff:", maxAbs)
      print("ideogram4 \(label) max rel diff:", maxRel)
      return maxAbs < 0.08 && maxRel < 0.02
    } else {
      model.compile(inputs: xImage, indicatorGPU, cosGPU, sinGPU, tEmbedGPU)
      reader(stateDict)
      let swift = copiedToCPU(
        model(inputs: xImage, indicatorGPU, cosGPU, sinGPU, tEmbedGPU)[0].as(of: Float.self))
      print("ideogram4 \(label) shape:", swift.shape, reference.shape)
      let maxAbs = maxAbsDiff2D(swift, reference)
      let maxRel = maxRelativeDiff2D(swift, reference)
      print("ideogram4 \(label) max abs diff:", maxAbs)
      print("ideogram4 \(label) max rel diff:", maxRel)
      return maxAbs < 0.08 && maxRel < 0.02
    }
  }
}

func runTransformerPrefixParity(subfolder: String, textLength: Int, layersToRun: Int) -> Bool {
  let label = "\(subfolder) prefix \(layersToRun)"
  print("ideogram4 \(label) parity: start")
  let pack = helper.load_transformer_pack(modelRoot, subfolder, deviceID, dtypeName)
  let testCase = helper.run_transformer_prefix_case(
    pack["model"], textLength, ditGridHeight, ditGridWidth, ditTimestep, layersToRun, deviceID,
    dtypeName)
  pack["model"].to("cpu")
  torch.cuda.empty_cache()

  let imageLength = ditGridHeight * ditGridWidth
  let xImageCPU = tensorFromPython(testCase["x_image"])
  let textFeaturesCPU = tensorFromPython(testCase["text_features"])
  let reference = tensorFromPython(testCase["reference"])
  let indicatorCPU = try! Tensor<Int32>(numpy: testCase["indicator"].astype(numpy.int32))
  let (cosCPU, sinCPU) = makeIdeogramRotary(
    textLength: textLength, gridHeight: ditGridHeight, gridWidth: ditGridWidth)
  let tEmbedCPU = ideogramTimestepEmbedding(ditTimestep)
  let stateDict = pack["state_dict"]

  return graph.withNoGrad {
    let xImage = graph.variable(Tensor<FloatType>(from: xImageCPU).toGPU(deviceID))
      .reshaped(.WC(imageLength, 128))
    let indicator = graph.variable(
      .CPU, format: .NHWC, shape: [textLength + imageLength], of: Int32.self)
    for i in 0..<(textLength + imageLength) {
      indicator[i] = indicatorCPU[i]
    }
    let indicatorGPU = indicator.toGPU(deviceID)
    let cosGPU = graph.variable(Tensor<FloatType>(from: cosCPU).toGPU(deviceID))
    let sinGPU = graph.variable(Tensor<FloatType>(from: sinCPU).toGPU(deviceID))
    let tEmbedGPU = graph.variable(Tensor<FloatType>(from: tEmbedCPU).toGPU(deviceID))
      .reshaped(.WC(1, 4_608))
    let (model, reader) = Ideogram4Transformer(
      textLength: textLength, imageLength: imageLength, layersToRun: layersToRun,
      outputHidden: true)
    model.maxConcurrency = .limit(1)
    if textLength > 0 {
      let textFeatures = graph.variable(Tensor<FloatType>(from: textFeaturesCPU).toGPU(deviceID))
        .reshaped(.WC(textLength, 53_248))
      if layersToRun > 0 {
        model.compile(inputs: xImage, textFeatures, indicatorGPU, cosGPU, sinGPU, tEmbedGPU)
      } else {
        model.compile(inputs: xImage, textFeatures, indicatorGPU)
      }
      reader(stateDict)
      let swiftOutput =
        layersToRun > 0
        ? model(inputs: xImage, textFeatures, indicatorGPU, cosGPU, sinGPU, tEmbedGPU)[0]
        : model(inputs: xImage, textFeatures, indicatorGPU)[0]
      let swift = copiedToCPU(swiftOutput.as(of: Float.self))
      let maxAbs = maxAbsDiff2D(swift, reference)
      let maxRel = maxRelativeDiff2D(swift, reference)
      print("ideogram4 \(label) shape:", swift.shape, reference.shape)
      print("ideogram4 \(label) max abs diff:", maxAbs)
      print("ideogram4 \(label) max rel diff:", maxRel)
      return maxAbs < 0.08 && maxRel < 0.02
    } else {
      if layersToRun > 0 {
        model.compile(inputs: xImage, indicatorGPU, cosGPU, sinGPU, tEmbedGPU)
      } else {
        model.compile(inputs: xImage, indicatorGPU)
      }
      reader(stateDict)
      let swiftOutput =
        layersToRun > 0
        ? model(inputs: xImage, indicatorGPU, cosGPU, sinGPU, tEmbedGPU)[0]
        : model(inputs: xImage, indicatorGPU)[0]
      let swift = copiedToCPU(swiftOutput.as(of: Float.self))
      let maxAbs = maxAbsDiff2D(swift, reference)
      let maxRel = maxRelativeDiff2D(swift, reference)
      print("ideogram4 \(label) shape:", swift.shape, reference.shape)
      print("ideogram4 \(label) max abs diff:", maxAbs)
      print("ideogram4 \(label) max rel diff:", maxRel)
      return maxAbs < 0.08 && maxRel < 0.02
    }
  }
}

func requireParity(_ ok: Bool, _ message: String) {
  if !ok {
    print(message)
    exit(1)
  }
}

switch mode {
case "parity-text":
  requireParity(runTextParity(), "Ideogram4 text parity failed")
case "qwen-load-text":
  requireParity(runOriginalQwenTextParity(), "Ideogram4 original Qwen text parity failed")
case "parity-transformer":
  requireParity(
    runTransformerParity(subfolder: "transformer", textLength: ditTextLength),
    "Ideogram4 conditional transformer parity failed")
  requireParity(
    runTransformerParity(subfolder: "unconditional_transformer", textLength: 0),
    "Ideogram4 unconditional transformer parity failed")
case "parity":
  requireParity(runTextParity(), "Ideogram4 text parity failed")
  requireParity(
    runTransformerParity(subfolder: "transformer", textLength: ditTextLength),
    "Ideogram4 conditional transformer parity failed")
  requireParity(
    runTransformerParity(subfolder: "unconditional_transformer", textLength: 0),
    "Ideogram4 unconditional transformer parity failed")
case "debug-transformer-prefix":
  requireParity(
    runTransformerPrefixParity(subfolder: "transformer", textLength: ditTextLength, layersToRun: 0),
    "Ideogram4 transformer stem parity failed")
  requireParity(
    runTransformerPrefixParity(subfolder: "transformer", textLength: ditTextLength, layersToRun: 1),
    "Ideogram4 transformer layer0 parity failed")
case "debug-transformer-depth":
  let layers = (env["IDEOGRAM4_DEBUG_LAYERS"] ?? "34").split(separator: ",").map {
    Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 34
  }
  var allDepthsPassed = true
  for layerCount in layers {
    let depthPassed = runTransformerPrefixParity(
      subfolder: "transformer", textLength: ditTextLength, layersToRun: layerCount)
    allDepthsPassed = depthPassed && allDepthsPassed
  }
  requireParity(allDepthsPassed, "Ideogram4 transformer depth parity failed")
default:
  print(
    "Usage: ideogram4 [parity|parity-text|qwen-load-text|parity-transformer|debug-transformer-prefix|debug-transformer-depth]"
  )
  exit(1)
}
