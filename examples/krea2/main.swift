import Diffusion
import Foundation
import Glibc
import NNC
import NNCPythonConversion
import PythonKit

typealias FloatType = Float
let swiftDtypeName = "float32"
typealias TextFloatType = Float16

setbuf(stdout, nil)

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

let modelRoot = "/slow/Data/Krea-2-Turbo"
let deviceID = 0
let ditReferenceDtype = "float32"
let textReferenceDtype = "float16"
let parityTextLength = 256
let parityGridHeight = 32
let parityGridWidth = 32
let parityTimestep: Float = 0.75
let parityMaxRelativeDiff: Float = 0.02
let textTokenLength = 256
let parityMaxTextRelativeDiff: Float = 0.05
let exportTextLength = 256
let exportGridHeight = 64
let exportGridWidth = 64
let textExportPath = "/slow/Data/krea2_turbo_text_model_f16.ckpt"
let ditExportPath = "/slow/Data/krea2_turbo_dit_f32.ckpt"
let mode = CommandLine.arguments.dropFirst().first ?? "parity-dit"

enum Krea2Config {
  static let hiddenSize = 6_144
  static let layers = 28
  static let attentionHeads = 48
  static let keyValueHeads = 12
  static let headDim = 128
  static let inChannels = 64
  static let timestepEmbedDim = 256
  static let intermediateSize = 16_384
  static let textHiddenSize = 2_560
  static let textLayers = 12
  static let textAttentionHeads = 20
  static let textKeyValueHeads = 20
  static let textIntermediateSize = 6_912
  static let textLayerwiseBlocks = 2
  static let textRefinerBlocks = 2
  static let ropeTheta: Double = 1_000
  static let ropeAxes = [32, 48, 48]
  static let normEps: Float = 1e-5
}

enum Krea2QwenTextConfig {
  static let hiddenSize = 2_560
  static let layers = 36
  static let heads = 32
  static let keyValueHeads = 8
  static let headDim = 128
  static let intermediateSize = 9_728
  static let vocabularySize = 151_936
  static let normEps: Float = 1e-6
  static let captureLayers: [Int] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
}

let site = Python.import("site")
let sys = Python.import("sys")
let osPath = Python.import("os.path")

func movePythonPathToFront(_ path: String) {
  while Bool(sys.path.__contains__(path)) ?? false {
    sys.path.remove(path)
  }
  sys.path.insert(0, path)
}

let userSitePackages = String(site.getusersitepackages()) ?? ""
if Bool(osPath.isdir(userSitePackages)) ?? false,
  (Bool(sys.path.__contains__(userSitePackages)) ?? false) == false
{
  movePythonPathToFront(userSitePackages)
}
let systemDistPackages = "/usr/lib/python3/dist-packages"
if (Bool(sys.path.__contains__(systemDistPackages)) ?? false) == false {
  movePythonPathToFront(systemDistPackages)
}

if let virtualEnv = ProcessInfo.processInfo.environment["VIRTUAL_ENV"] {
  let libRoot = URL(fileURLWithPath: virtualEnv).appendingPathComponent("lib")
  if let pythonLibDirs = try? FileManager.default.contentsOfDirectory(
    at: libRoot, includingPropertiesForKeys: nil)
  {
    for pythonLibDir in pythonLibDirs.sorted(by: { $0.path < $1.path }) {
      let sitePackagesDir = pythonLibDir.appendingPathComponent("site-packages").path
      if FileManager.default.fileExists(atPath: sitePackagesDir) {
        movePythonPathToFront(sitePackagesDir)
      }
    }
  }
}

let builtins = Python.import("builtins")
let types = Python.import("types")
let torch = Python.import("torch")

torch.set_grad_enabled(false)
torch.manual_seed(42)
if !(Bool(torch.cuda.is_available()) ?? false) {
  print("CUDA is not visible to Python. Run this target outside the sandbox for parity.")
  exit(1)
}
torch.cuda.manual_seed_all(42)

let helper = types.ModuleType("krea2_swift_reference")
builtins.exec(
  #"""
  import glob
  import math
  import os
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from safetensors.torch import load_file

  HIDDEN = 6144
  HEADS = 48
  KV_HEADS = 12
  HEAD_DIM = 128
  TEXT_HIDDEN = 2560
  TEXT_HEADS = 20
  TEXT_KV_HEADS = 20
  TEXT_LAYERS = 12
  TEXT_INTERMEDIATE = 6912
  INTERMEDIATE = 16384
  IN_CHANNELS = 64
  TIMESTEP_DIM = 256
  ROPE_AXES = (32, 48, 48)
  ROPE_THETA = 1000.0
  NORM_EPS = 1e-5
  QWEN_TEXT_HIDDEN = 2560
  QWEN_TEXT_HEADS = 32
  QWEN_TEXT_KV_HEADS = 8
  QWEN_TEXT_HEAD_DIM = 128
  QWEN_TEXT_INTERMEDIATE = 9728
  QWEN_TEXT_CAPTURE_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

  def ref_dtype(name):
      if name == "float32":
          return torch.float32
      if name == "float16":
          return torch.float16
      return torch.bfloat16

  class Krea2RMSNorm(nn.Module):
      def __init__(self, dim, eps=NORM_EPS):
          super().__init__()
          self.dim = dim
          self.eps = eps
          self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

      def forward(self, x):
          dtype = x.dtype
          x = F.rms_norm(x.float(), (self.dim,), weight=self.weight.float() + 1.0, eps=self.eps)
          return x.to(dtype)

  def apply_pairwise_rotary(x, rotary):
      cos = rotary[..., 0::2].unsqueeze(0).unsqueeze(2)
      sin = rotary[..., 1::2].unsqueeze(0).unsqueeze(2)
      x0 = x[..., 0::2].float()
      x1 = x[..., 1::2].float()
      out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
      return out.flatten(-2).to(x.dtype)

  def attention(q, k, v):
      if q.shape[2] != k.shape[2]:
          # q/k/v are B,L,H,D here.
          repeat = q.shape[2] // k.shape[2]
          k = k.repeat_interleave(repeat, dim=2)
          v = v.repeat_interleave(repeat, dim=2)
      q = q.transpose(1, 2)
      k = k.transpose(1, 2)
      v = v.transpose(1, 2)
      out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]))
      return out.transpose(1, 2).flatten(2, 3)

  class Krea2Attention(nn.Module):
      def __init__(self, hidden_size, heads, kv_heads):
          super().__init__()
          self.hidden_size = hidden_size
          self.heads = heads
          self.kv_heads = kv_heads
          self.head_dim = hidden_size // heads
          self.to_q = nn.Linear(hidden_size, self.head_dim * heads, bias=False)
          self.to_k = nn.Linear(hidden_size, self.head_dim * kv_heads, bias=False)
          self.to_v = nn.Linear(hidden_size, self.head_dim * kv_heads, bias=False)
          self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
          self.norm_q = Krea2RMSNorm(self.head_dim)
          self.norm_k = Krea2RMSNorm(self.head_dim)
          self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)])

      def forward(self, x, rotary=None):
          q = self.to_q(x).unflatten(-1, (self.heads, self.head_dim))
          k = self.to_k(x).unflatten(-1, (self.kv_heads, self.head_dim))
          v = self.to_v(x).unflatten(-1, (self.kv_heads, self.head_dim))
          gate = torch.sigmoid(self.to_gate(x))
          q = self.norm_q(q)
          k = self.norm_k(k)
          if rotary is not None:
              q = apply_pairwise_rotary(q, rotary)
              k = apply_pairwise_rotary(k, rotary)
          return self.to_out[0](attention(q, k, v) * gate)

  class Krea2SwiGLU(nn.Module):
      def __init__(self, dim, hidden_dim):
          super().__init__()
          self.gate = nn.Linear(dim, hidden_dim, bias=False)
          self.up = nn.Linear(dim, hidden_dim, bias=False)
          self.down = nn.Linear(hidden_dim, dim, bias=False)

      def forward(self, x):
          return self.down(F.silu(self.gate(x)) * self.up(x))

  class Krea2TextFusionBlock(nn.Module):
      def __init__(self):
          super().__init__()
          self.norm1 = Krea2RMSNorm(TEXT_HIDDEN)
          self.norm2 = Krea2RMSNorm(TEXT_HIDDEN)
          self.attn = Krea2Attention(TEXT_HIDDEN, TEXT_HEADS, TEXT_KV_HEADS)
          self.ff = Krea2SwiGLU(TEXT_HIDDEN, TEXT_INTERMEDIATE)

      def forward(self, x):
          x = x + self.attn(self.norm1(x))
          x = x + self.ff(self.norm2(x))
          return x

  class Krea2TextFusion(nn.Module):
      def __init__(self):
          super().__init__()
          self.layerwise_blocks = nn.ModuleList([Krea2TextFusionBlock() for _ in range(2)])
          self.projector = nn.Linear(TEXT_LAYERS, 1, bias=False)
          self.refiner_blocks = nn.ModuleList([Krea2TextFusionBlock() for _ in range(2)])

      def forward(self, x):
          b, l, n, d = x.shape
          x = x.reshape(b * l, n, d)
          for block in self.layerwise_blocks:
              x = block(x.contiguous())
          x = x.reshape(b, l, n, d).permute(0, 1, 3, 2)
          x = self.projector(x).squeeze(-1)
          for block in self.refiner_blocks:
              x = block(x)
          return x

  class Krea2TransformerBlock(nn.Module):
      def __init__(self):
          super().__init__()
          self.scale_shift_table = nn.Parameter(torch.zeros(6, HIDDEN))
          self.norm1 = Krea2RMSNorm(HIDDEN)
          self.norm2 = Krea2RMSNorm(HIDDEN)
          self.attn = Krea2Attention(HIDDEN, HEADS, KV_HEADS)
          self.ff = Krea2SwiGLU(HIDDEN, INTERMEDIATE)

      def forward(self, x, temb_mod, rotary):
          modulation = temb_mod.unflatten(-1, (6, HIDDEN)) + self.scale_shift_table
          prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)
          attn_out = self.attn((1.0 + prescale) * self.norm1(x) + preshift, rotary)
          x = x + pregate * attn_out
          ff_out = self.ff((1.0 + postscale) * self.norm2(x) + postshift)
          return x + postgate * ff_out

  class Krea2TimestepEmbedding(nn.Module):
      def __init__(self):
          super().__init__()
          self.linear_1 = nn.Linear(TIMESTEP_DIM, HIDDEN)
          self.linear_2 = nn.Linear(HIDDEN, HIDDEN)

      def forward(self, timestep, dtype):
          half = TIMESTEP_DIM // 2
          freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half)
          args = (timestep.float() * 1e3)[:, None, None] * freqs
          emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
          return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))

  class Krea2TextProjection(nn.Module):
      def __init__(self):
          super().__init__()
          self.norm = Krea2RMSNorm(TEXT_HIDDEN)
          self.linear_1 = nn.Linear(TEXT_HIDDEN, HIDDEN)
          self.linear_2 = nn.Linear(HIDDEN, HIDDEN)

      def forward(self, x):
          x = self.linear_1(self.norm(x))
          return self.linear_2(F.gelu(x, approximate="tanh"))

  class Krea2FinalLayer(nn.Module):
      def __init__(self):
          super().__init__()
          self.scale_shift_table = nn.Parameter(torch.zeros(2, HIDDEN))
          self.norm = Krea2RMSNorm(HIDDEN)
          self.linear = nn.Linear(HIDDEN, IN_CHANNELS)

      def forward(self, x, temb):
          modulation = temb + self.scale_shift_table
          scale, shift = modulation.chunk(2, dim=1)
          return self.linear((1.0 + scale) * self.norm(x) + shift)

  class Krea2Transformer2DModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.img_in = nn.Linear(IN_CHANNELS, HIDDEN)
          self.time_embed = Krea2TimestepEmbedding()
          self.time_mod_proj = nn.Linear(HIDDEN, 6 * HIDDEN)
          self.text_fusion = Krea2TextFusion()
          self.txt_in = Krea2TextProjection()
          self.transformer_blocks = nn.ModuleList([Krea2TransformerBlock() for _ in range(28)])
          self.final_layer = Krea2FinalLayer()

      def forward(self, hidden_states, encoder_hidden_states, timestep, position_ids):
          temb = self.time_embed(timestep, dtype=hidden_states.dtype)
          temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))
          txt = self.txt_in(self.text_fusion(encoder_hidden_states))
          img = self.img_in(hidden_states)
          x = torch.cat([txt, img], dim=1)
          rotary = make_rotary(position_ids, device=x.device, dtype=torch.float32)
          for block in self.transformer_blocks:
              x = block(x, temb_mod, rotary)
          x = x[:, txt.shape[1]:]
          return self.final_layer(x, temb)

  def make_positions(text_len, grid_h, grid_w, device="cpu"):
      rows = []
      for _ in range(int(text_len)):
          rows.append([0, 0, 0])
      for y in range(int(grid_h)):
          for x in range(int(grid_w)):
              rows.append([0, y, x])
      return torch.tensor(rows, dtype=torch.float32, device=device)

  def make_rotary(position_ids, device=None, dtype=torch.float32):
      pos = position_ids.to(device=device, dtype=torch.float32)
      parts = []
      for axis, dim in enumerate(ROPE_AXES):
          freqs = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
          omega = 1.0 / (ROPE_THETA ** freqs)
          angles = pos[:, axis:axis + 1].double() * omega
          cos = torch.cos(angles).float()
          sin = torch.sin(angles).float()
          parts.append(torch.stack([cos, sin], dim=-1).flatten(-2))
      return torch.cat(parts, dim=-1).to(dtype=dtype)

  def load_transformer_state(root):
      state = {}
      pattern = os.path.join(root, "transformer", "diffusion_pytorch_model-*.safetensors")
      for filename in sorted(glob.glob(pattern)):
          state.update(load_file(filename, device="cpu"))
      return state

  def load_krea_qwen_text_state(root):
      full_sd = load_file(f"{root}/text_encoder/model.safetensors", device="cpu")
      return {
          k[len("language_model."):]: v
          for k, v in full_sd.items()
          if k.startswith("language_model.")
      }

  def make_text_token_ids(token_count, vocab_size=151936):
      ids = (torch.arange(token_count, dtype=torch.long) * 7919 + 12345) % vocab_size
      return ids.view(1, -1)

  def _text_config(config_root):
      from transformers import AutoConfig
      config = AutoConfig.from_pretrained(config_root, trust_remote_code=True)
      text_config = config.text_config if hasattr(config, "text_config") else config
      if getattr(text_config, "rope_scaling", None) is None:
          rope_parameters = getattr(text_config, "rope_parameters", None) or getattr(config, "rope_parameters", None)
          if rope_parameters is not None:
              text_config.rope_scaling = {
                  "mrope_section": list(rope_parameters.get("mrope_section", [24, 20, 20]))
              }
          else:
              text_config.rope_scaling = {"mrope_section": [24, 20, 20]}
      text_config._attn_implementation = "eager"
      return text_config

  def _load_text_module(module, state_dict, device, dtype):
      prepared = {
          k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
          for k, v in state_dict.items()
      }
      missing, unexpected = module.load_state_dict(prepared, strict=True, assign=True)
      if missing or unexpected:
          raise RuntimeError(f"missing={missing[:5]} unexpected={unexpected[:5]}")
      module.eval()
      return module

  def run_krea_qwen_text_reference(root, state_dict, token_count, device_index, dtype_name):
      from transformers.masking_utils import create_causal_mask
      from transformers.models.qwen3_vl.modeling_qwen3_vl import (
          Qwen3VLTextDecoderLayer,
          Qwen3VLTextRotaryEmbedding,
      )
      dtype = ref_dtype(dtype_name)
      device = torch.device(f"cuda:{int(device_index)}")
      config = _text_config(f"{root}/text_encoder")
      embed = nn.Embedding(config.vocab_size, config.hidden_size).to(device=device, dtype=dtype)
      embed.weight.data.copy_(state_dict["embed_tokens.weight"].to(device=device, dtype=dtype))
      rotary_emb = Qwen3VLTextRotaryEmbedding(config=config).to(device)
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
      captured = []
      capture_set = set(QWEN_TEXT_CAPTURE_LAYERS)
      layers_to_run = max(capture_set) + 1
      for i in range(layers_to_run):
          prefix = f"layers.{i}."
          layer_state = {
              key[len(prefix):]: value
              for key, value in state_dict.items()
              if key.startswith(prefix)
          }
          layer = Qwen3VLTextDecoderLayer(config, i).to(device=device, dtype=dtype)
          layer = _load_text_module(layer, layer_state, device, dtype)
          hidden_states = layer(
              hidden_states,
              attention_mask=causal_mask,
              position_ids=text_position_ids,
              past_key_values=None,
              cache_position=cache_position,
              position_embeddings=position_embeddings,
          )
          if i in capture_set:
              captured.append(hidden_states)
          del layer
          torch.cuda.empty_cache()
      return torch.cat(captured, dim=-1)[0].float().cpu().numpy()

  def dequant_weight_np(state_dict, key, dtype_name):
      return state_dict[key].to(ref_dtype(dtype_name)).float().cpu().numpy()

  def dequant_interleaved_qk_weight_np(state_dict, key, heads, head_dim, dtype_name):
      w = state_dict[key].to(ref_dtype(dtype_name))
      return w.float().view(heads, 2, head_dim // 2, -1).transpose(1, 2).cpu().numpy()

  def interleaved_qk_norm_np(state_dict, key, head_dim):
      return state_dict[key].float().view(2, head_dim // 2).transpose(0, 1).cpu().numpy()

  def tensor_np(state_dict, key):
      return state_dict[key].float().cpu().numpy()

  def rms_weight_np(state_dict, key):
      return (state_dict[key].float() + 1.0).cpu().numpy()

  def load_transformer_pack(root, device_index, dtype_name):
      dtype = ref_dtype(dtype_name)
      state_dict = load_transformer_state(root)
      model = Krea2Transformer2DModel()
      missing, unexpected = model.load_state_dict(state_dict, strict=True)
      if missing or unexpected:
          raise RuntimeError(f"missing={missing[:5]} unexpected={unexpected[:5]}")
      device = torch.device(f"cuda:{int(device_index)}")
      model.to(device)
      for name, param in model.named_parameters():
          if ".norm" not in name and "norm." not in name:
              param.data = param.data.to(dtype=dtype)
          else:
              param.data = param.data.float()
      model.eval()
      return {"model": model, "state_dict": state_dict}

  def cast_module_for_reference(module, device, dtype):
      module.to(device)
      for name, param in module.named_parameters():
          if ".norm" in name or "norm." in name:
              param.data = param.data.float()
          else:
              param.data = param.data.to(dtype=dtype)
      module.eval()
      return module

  def prefixed_state_dict(state_dict, prefix):
      start = prefix + "."
      return {k[len(start):]: v for k, v in state_dict.items() if k.startswith(start)}

  def load_prefixed_module(module, state_dict, prefix, device, dtype):
      missing, unexpected = module.load_state_dict(prefixed_state_dict(state_dict, prefix), strict=True)
      if missing or unexpected:
          raise RuntimeError(f"{prefix}: missing={missing[:5]} unexpected={unexpected[:5]}")
      return cast_module_for_reference(module, device, dtype)

  def linear_from_state(x, state_dict, prefix, device, dtype):
      weight = state_dict[f"{prefix}.weight"].to(device=device, dtype=dtype)
      bias_key = f"{prefix}.bias"
      bias = state_dict[bias_key].to(device=device, dtype=dtype) if bias_key in state_dict else None
      return F.linear(x, weight, bias)

  def rms_from_state(x, state_dict, key):
      weight = state_dict[key].to(device=x.device, dtype=torch.float32) + 1.0
      return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=NORM_EPS).to(x.dtype)

  def run_transformer_case(state_dict, text_len, grid_h, grid_w, timestep, device_index, dtype_name):
      dtype = ref_dtype(dtype_name)
      device = torch.device(f"cuda:{int(device_index)}")
      image_len = int(grid_h) * int(grid_w)
      torch.manual_seed(42)
      torch.cuda.manual_seed_all(42)
      x = torch.randn((1, image_len, IN_CHANNELS), device=device, dtype=dtype)
      text = torch.randn((1, int(text_len), TEXT_LAYERS, TEXT_HIDDEN), device=device, dtype=dtype)
      t = torch.full((1,), float(timestep), device=device, dtype=torch.float32)
      positions = make_positions(text_len, grid_h, grid_w, device=device)
      with torch.no_grad():
          half = TIMESTEP_DIM // 2
          freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=device) / half)
          args = (t.float() * 1e3)[:, None, None] * freqs
          temb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
          temb = linear_from_state(F.gelu(linear_from_state(temb, state_dict, "time_embed.linear_1", device, dtype), approximate="tanh"), state_dict, "time_embed.linear_2", device, dtype)
          temb_mod = linear_from_state(F.gelu(temb, approximate="tanh"), state_dict, "time_mod_proj", device, dtype)

          b, l, n, d = text.shape
          hidden = text.reshape(b * l, n, d)
          for i in range(2):
              block = load_prefixed_module(Krea2TextFusionBlock(), state_dict, f"text_fusion.layerwise_blocks.{i}", device, dtype)
              hidden = block(hidden.contiguous())
              del block
              torch.cuda.empty_cache()

          hidden = hidden.reshape(b, l, n, d).permute(0, 1, 3, 2)
          projector = state_dict["text_fusion.projector.weight"].to(device=device, dtype=dtype)
          hidden = F.linear(hidden, projector).squeeze(-1)

          for i in range(2):
              block = load_prefixed_module(Krea2TextFusionBlock(), state_dict, f"text_fusion.refiner_blocks.{i}", device, dtype)
              hidden = block(hidden)
              del block
              torch.cuda.empty_cache()

          hidden = linear_from_state(rms_from_state(hidden, state_dict, "txt_in.norm.weight"), state_dict, "txt_in.linear_1", device, dtype)
          hidden = linear_from_state(F.gelu(hidden, approximate="tanh"), state_dict, "txt_in.linear_2", device, dtype)

          img = linear_from_state(x, state_dict, "img_in", device, dtype)
          hidden = torch.cat([hidden, img], dim=1)
          rotary = make_rotary(positions, device=device, dtype=torch.float32)
          for i in range(28):
              block = load_prefixed_module(Krea2TransformerBlock(), state_dict, f"transformer_blocks.{i}", device, dtype)
              hidden = block(hidden, temb_mod, rotary)
              del block
              torch.cuda.empty_cache()

          hidden = hidden[:, int(text_len):]
          scale_shift = state_dict["final_layer.scale_shift_table"].to(device=device, dtype=dtype)
          modulation = temb + scale_shift
          scale, shift = modulation.chunk(2, dim=1)
          hidden = (1.0 + scale) * rms_from_state(hidden, state_dict, "final_layer.norm.weight") + shift
          reference = linear_from_state(hidden, state_dict, "final_layer.linear", device, dtype)
      return {
          "x": x.float().cpu().numpy(),
          "text": text.float().cpu().numpy(),
          "positions": positions.float().cpu().numpy(),
          "reference": reference.float().cpu().numpy(),
      }
  """#,
  helper.__dict__)

func tensorFromPython(_ object: PythonObject) -> Tensor<Float> {
  try! Tensor<Float>(numpy: object)
}

func copiedToCPU(_ tensor: DynamicGraph.Tensor<Float>) -> Tensor<Float> {
  tensor.as(of: Float.self).rawValue.toCPU()
}

func copiedToCPU(_ tensor: DynamicGraph.Tensor<Float16>) -> Tensor<Float> {
  Tensor<Float>(from: tensor.as(of: Float16.self).rawValue.toCPU())
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

func maxAbsAndRelativeDiff3D(_ lhs: Tensor<Float>, _ rhs: Tensor<Float>) -> (Float, Float) {
  if lhs.shape.count == 2 && rhs.shape.count == 3 && rhs.shape[0] == 1
    && lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == rhs.shape[2]
  {
    var maxAbsDiff: Float = 0
    var maxAbsRef: Float = 0
    for i in 0..<lhs.shape[0] {
      for j in 0..<lhs.shape[1] {
        let ref = Float(rhs[0, i, j])
        maxAbsDiff = max(maxAbsDiff, abs(Float(lhs[i, j]) - ref))
        maxAbsRef = max(maxAbsRef, abs(ref))
      }
    }
    return (maxAbsDiff, maxAbsDiff / max(maxAbsRef, 1e-6))
  }
  precondition(lhs.shape.count == 3)
  precondition(rhs.shape.count == 3)
  precondition(lhs.shape[0] == rhs.shape[0])
  precondition(lhs.shape[1] == rhs.shape[1])
  precondition(lhs.shape[2] == rhs.shape[2])
  var maxAbsDiff: Float = 0
  var maxAbsRef: Float = 0
  for i in 0..<lhs.shape[0] {
    for j in 0..<lhs.shape[1] {
      for k in 0..<lhs.shape[2] {
        let ref = Float(rhs[i, j, k])
        maxAbsDiff = max(maxAbsDiff, abs(Float(lhs[i, j, k]) - ref))
        maxAbsRef = max(maxAbsRef, abs(ref))
      }
    }
  }
  return (maxAbsDiff, maxAbsDiff / max(maxAbsRef, 1e-6))
}

func printSamples(_ label: String, _ lhs: Tensor<Float>, _ rhs: Tensor<Float>) {
  if lhs.shape.count == 2 && rhs.shape.count == 3 && rhs.shape[0] == 1 {
    let rows = min(lhs.shape[0], 2)
    let cols = min(lhs.shape[1], 8)
    for i in 0..<rows {
      var swiftValues = [Float]()
      var refValues = [Float]()
      for j in 0..<cols {
        swiftValues.append(Float(lhs[i, j]))
        refValues.append(Float(rhs[0, i, j]))
      }
      print("\(label) token \(i) swift:", swiftValues)
      print("\(label) token \(i) ref:", refValues)
    }
    return
  }
  let rows = min(lhs.shape[1], 2)
  let cols = min(lhs.shape[2], 8)
  for i in 0..<rows {
    var swiftValues = [Float]()
    var refValues = [Float]()
    for j in 0..<cols {
      swiftValues.append(Float(lhs[0, i, j]))
      refValues.append(Float(rhs[0, i, j]))
    }
    print("\(label) token \(i) swift:", swiftValues)
    print("\(label) token \(i) ref:", refValues)
  }
}

func tensorValue(_ stateDict: PythonObject, _ key: String) -> Tensor<Float> {
  tensorFromPython(helper.tensor_np(stateDict, key))
}

func rmsWeight(_ stateDict: PythonObject, _ key: String) -> Tensor<Float> {
  tensorFromPython(helper.rms_weight_np(stateDict, key))
}

func dequantTextWeight(_ stateDict: PythonObject, _ key: String) -> Tensor<Float> {
  tensorFromPython(helper.dequant_weight_np(stateDict, key, textReferenceDtype))
}

func dequantTextInterleavedQKWeight(
  _ stateDict: PythonObject, _ key: String, heads: Int, headDim: Int
) -> Tensor<Float> {
  tensorFromPython(
    helper.dequant_interleaved_qk_weight_np(stateDict, key, heads, headDim, textReferenceDtype))
}

func interleavedTextQKNorm(_ stateDict: PythonObject, _ key: String, headDim: Int) -> Tensor<Float>
{
  tensorFromPython(helper.interleaved_qk_norm_np(stateDict, key, headDim))
}

func copyDenseWeight(_ dense: Model, _ stateDict: PythonObject, _ key: String) {
  dense.weight.copy(from: Tensor<FloatType>(from: tensorValue(stateDict, key)))
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

func copyRMSNorm(_ norm: Model, _ stateDict: PythonObject, _ key: String) {
  norm.weight.copy(from: Tensor<FloatType>(from: rmsWeight(stateDict, key)))
}

func copyParameter(
  _ parameter: Parameter<FloatType>, _ stateDict: PythonObject, _ key: String, row: Int
) {
  let value = tensorValue(stateDict, key)
  parameter.weight.copy(from: Tensor<FloatType>(from: value[row, 0..<value.shape[1]]))
}

func timestepEmbedding(_ timestep: Float, batchSize: Int = 1) -> Tensor<Float> {
  var embedding = Tensor<Float>(.CPU, .HWC(batchSize, 1, Krea2Config.timestepEmbedDim))
  let half = Krea2Config.timestepEmbedDim / 2
  for i in 0..<half {
    let freq = exp(-log(10_000.0) * Float(i) / Float(half)) * timestep * 1_000
    for b in 0..<batchSize {
      embedding[b, 0, i] = cos(freq)
      embedding[b, 0, i + half] = sin(freq)
    }
  }
  return embedding
}

func kreaRotary(textLength: Int, gridHeight: Int, gridWidth: Int) -> Tensor<Float> {
  let tokenLength = textLength + gridHeight * gridWidth
  var rotary = Tensor<Float>(.CPU, .NHWC(1, tokenLength, 1, Krea2Config.headDim))
  for token in 0..<tokenLength {
    let imageIndex = token - textLength
    let positions: [Double]
    if imageIndex < 0 {
      positions = [0, 0, 0]
    } else {
      positions = [0, Double(imageIndex / gridWidth), Double(imageIndex % gridWidth)]
    }
    var offset = 0
    for axis in 0..<3 {
      let dim = Krea2Config.ropeAxes[axis]
      for i in 0..<(dim / 2) {
        let angle = positions[axis] / pow(Krea2Config.ropeTheta, Double(i * 2) / Double(dim))
        rotary[0, token, 0, offset + i * 2] = Float(cos(angle))
        rotary[0, token, 0, offset + i * 2 + 1] = Float(sin(angle))
      }
      offset += dim
    }
  }
  return rotary
}

func qwenMropeAngles(position: (Int, Int, Int), headDim: Int, theta: Double = 5_000_000.0)
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

func makeQwenTextRotary(tokenLength: Int) -> Tensor<Float> {
  let headDim = Krea2QwenTextConfig.headDim
  let half = headDim / 2
  var rotary = Tensor<Float>(.CPU, .NHWC(1, tokenLength, 1, headDim))
  for i in 0..<tokenLength {
    let angles = qwenMropeAngles(position: (i, i, i), headDim: headDim)
    for k in 0..<half {
      rotary[0, i, 0, k * 2] = Float(cos(angles[k]))
      rotary[0, i, 0, k * 2 + 1] = Float(sin(angles[k]))
    }
  }
  return rotary
}

func sliceModulation(_ x: Model.IO, row: Int, hiddenSize: Int = Krea2Config.hiddenSize) -> Model.IO
{
  x.reshaped(
    [1, 1, hiddenSize], offset: [0, 0, row * hiddenSize],
    strides: [6 * hiddenSize, 6 * hiddenSize, 1])
}

func Krea2SwiGLU(prefix: String, hiddenSize: Int, intermediateSize: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let gate = Dense(count: intermediateSize, noBias: true, name: "gate")
  let up = Dense(count: intermediateSize, noBias: true, name: "up")
  let down = Dense(count: hiddenSize, noBias: true, name: "down")
  let out = down(gate(x).swish() .* up(x))
  let reader: (PythonObject) -> Void = { stateDict in
    copyDenseWeight(gate, stateDict, "\(prefix).gate.weight")
    copyDenseWeight(up, stateDict, "\(prefix).up.weight")
    copyDenseWeight(down, stateDict, "\(prefix).down.weight")
  }
  return (Model([x], [out]), reader)
}

func repeatKeyValueHeads(
  _ x: Model.IO, batchSize: Int, tokenLength: Int, keyValueHeads: Int, heads: Int, headDim: Int
) -> Model.IO {
  precondition(heads % keyValueHeads == 0)
  if heads == keyValueHeads {
    return x
  }
  let repeats = heads / keyValueHeads
  var parts = [Model.IO]()
  parts.reserveCapacity(heads)
  for head in 0..<keyValueHeads {
    let slice = x.reshaped(
      [batchSize, tokenLength, 1, headDim], offset: [0, 0, head, 0],
      strides: [tokenLength * keyValueHeads * headDim, keyValueHeads * headDim, headDim, 1]
    ).contiguous()
    for _ in 0..<repeats {
      parts.append(slice)
    }
  }
  return Concat(axis: 2)(parts)
}

func explicitAttentionNHWC(
  queries: Model.IO, keys: Model.IO, values: Model.IO, batchSize: Int, tokenLength: Int,
  heads: Int, keyValueHeads: Int, headDim: Int
) -> Model.IO {
  let repeatedKeys = repeatKeyValueHeads(
    keys, batchSize: batchSize, tokenLength: tokenLength, keyValueHeads: keyValueHeads,
    heads: heads, headDim: headDim)
  let repeatedValues = repeatKeyValueHeads(
    values, batchSize: batchSize, tokenLength: tokenLength, keyValueHeads: keyValueHeads,
    heads: heads, headDim: headDim)
  let scaledQueries = ((1.0 / Float(headDim).squareRoot()) * queries).transposed(1, 2)
    .contiguous()
  let transposedKeys = repeatedKeys.transposed(1, 2).contiguous()
  let transposedValues = repeatedValues.transposed(1, 2).contiguous()
  var dot = Matmul(transposeB: (2, 3))(scaledQueries, transposedKeys)
  dot = dot.reshaped([batchSize * heads * tokenLength, tokenLength]).softmax()
  dot = dot.reshaped([batchSize, heads, tokenLength, tokenLength])
  var out = dot * transposedValues
  out = out.reshaped([batchSize, heads, tokenLength, headDim]).transposed(1, 2)
  return out.reshaped([batchSize, tokenLength, heads * headDim])
}

func KreaQwenTextEmbedding(tokenLength: Int) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    TextFloatType.self, vocabularySize: Krea2QwenTextConfig.vocabularySize,
    embeddingSize: Krea2QwenTextConfig.hiddenSize, name: "tok_embeddings")
  let embedding = tokenEmbed(tokens)
  let reader: (PythonObject) -> Void = { stateDict in
    tokenEmbed.parameters.copy(
      from: Tensor<TextFloatType>(from: tensorValue(stateDict, "embed_tokens.weight")))
    tokenEmbed.parameters.to(.unifiedMemory)
  }
  return (Model([tokens], [embedding]), reader)
}

func KreaQwenTextSelfAttention(prefix: String, tokenLength: Int) -> (Model, (PythonObject) -> Void)
{
  let x = Input()
  let rot = Input()
  let width = Krea2QwenTextConfig.hiddenSize
  let headDim = Krea2QwenTextConfig.headDim
  let heads = Krea2QwenTextConfig.heads
  let keyValueHeads = Krea2QwenTextConfig.keyValueHeads
  let toKeys = Dense(count: headDim * keyValueHeads, noBias: true, name: "k_proj")
  let toQueries = Dense(count: headDim * heads, noBias: true, name: "q_proj")
  let toValues = Dense(count: headDim * keyValueHeads, noBias: true, name: "v_proj")
  var keys = toKeys(x).reshaped([1, tokenLength, keyValueHeads, headDim])
  let normK = RMSNorm(epsilon: Krea2QwenTextConfig.normEps, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries = toQueries(x).reshaped([1, tokenLength, heads, headDim])
  let normQ = RMSNorm(epsilon: Krea2QwenTextConfig.normEps, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = toValues(x).reshaped([1, tokenLength, keyValueHeads, headDim])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  var out = ScaledDotProductAttention(scale: 1.0 / Float(headDim).squareRoot(), isCausal: true)(
    queries, keys, values
  ).reshaped([tokenLength, heads * headDim])
  let unifyHeads = Dense(count: width, noBias: true, name: "out_proj")
  out = unifyHeads(out)
  let reader: (PythonObject) -> Void = { stateDict in
    toQueries.weight.copy(
      from: Tensor<TextFloatType>(
        from: dequantTextInterleavedQKWeight(
          stateDict, "\(prefix).self_attn.q_proj.weight", heads: heads, headDim: headDim)))
    normQ.weight.copy(
      from: Tensor<TextFloatType>(
        from: interleavedTextQKNorm(
          stateDict, "\(prefix).self_attn.q_norm.weight", headDim: headDim)))
    toKeys.weight.copy(
      from: Tensor<TextFloatType>(
        from: dequantTextInterleavedQKWeight(
          stateDict, "\(prefix).self_attn.k_proj.weight", heads: keyValueHeads, headDim: headDim)))
    normK.weight.copy(
      from: Tensor<TextFloatType>(
        from: interleavedTextQKNorm(
          stateDict, "\(prefix).self_attn.k_norm.weight", headDim: headDim)))
    copyTextDenseWeight(toValues, stateDict, "\(prefix).self_attn.v_proj.weight")
    copyTextDenseWeight(unifyHeads, stateDict, "\(prefix).self_attn.o_proj.weight")
    toQueries.weight.to(.unifiedMemory)
    toKeys.weight.to(.unifiedMemory)
  }
  return (Model([x, rot], [out]), reader)
}

func KreaQwenTextFeedForward() -> (Model, Model, Model, Model) {
  let x = Input()
  let gate = Dense(count: Krea2QwenTextConfig.intermediateSize, noBias: true, name: "gate_proj")
  let up = Dense(count: Krea2QwenTextConfig.intermediateSize, noBias: true, name: "up_proj")
  var out = up(x) .* gate(x).swish()
  let down = Dense(count: Krea2QwenTextConfig.hiddenSize, noBias: true, name: "down_proj")
  out = down(out)
  return (gate, down, up, Model([x], [out], name: "mlp"))
}

func KreaQwenTextTransformerBlock(prefix: String, tokenLength: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let rot = Input()
  let norm1 = RMSNorm(epsilon: Krea2QwenTextConfig.normEps, axis: [1], name: "input_layernorm")
  let (attention, attentionReader) = KreaQwenTextSelfAttention(
    prefix: prefix, tokenLength: tokenLength)
  var out = attention(norm1(x).to(TextFloatType.dataType), rot).to(of: x) + x
  let residual = out
  let norm2 = RMSNorm(
    epsilon: Krea2QwenTextConfig.normEps, axis: [1], name: "post_attention_layernorm")
  let (gate, down, up, ff) = KreaQwenTextFeedForward()
  out = residual + ff(norm2(out).to(TextFloatType.dataType)).to(of: residual)
  let reader: (PythonObject) -> Void = { stateDict in
    attentionReader(stateDict)
    norm1.weight.copy(
      from: Tensor<TextFloatType>(from: tensorValue(stateDict, "\(prefix).input_layernorm.weight")))
    norm2.weight.copy(
      from: Tensor<TextFloatType>(
        from: tensorValue(stateDict, "\(prefix).post_attention_layernorm.weight")))
    copyTextDenseWeight(gate, stateDict, "\(prefix).mlp.gate_proj.weight")
    copyTextDenseWeight(down, stateDict, "\(prefix).mlp.down_proj.weight")
    copyTextDenseWeight(up, stateDict, "\(prefix).mlp.up_proj.weight")
  }
  return (Model([x, rot], [out]), reader)
}

func KreaQwenTextFeatures(tokenLength: Int) -> (Model, (PythonObject) -> Void) {
  let tokens = Input()
  let rot = Input()
  let (embedding, embeddingReader) = KreaQwenTextEmbedding(tokenLength: tokenLength)
  var out = embedding(tokens)
  let captureLayers = Set(Krea2QwenTextConfig.captureLayers)
  var captured = [Model.IO]()
  var readers = [(PythonObject) -> Void]()
  for i in 0..<Krea2QwenTextConfig.layers {
    let (layer, reader) = KreaQwenTextTransformerBlock(
      prefix: "layers.\(i)", tokenLength: tokenLength)
    out = layer(out, rot)
    readers.append(reader)
    if captureLayers.contains(i) {
      captured.append(out.to(.Float32))
    }
  }
  let reader: (PythonObject) -> Void = { stateDict in
    embeddingReader(stateDict)
    for reader in readers {
      reader(stateDict)
    }
  }
  return (Model([tokens, rot], captured), reader)
}

func Krea2Attention(
  prefix: String, hiddenSize: Int, heads: Int, keyValueHeads: Int, batchSize: Int,
  tokenLength: Int, rotary: Bool
) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let rot = rotary ? Input() : nil
  let headDim = hiddenSize / heads
  let toQ = Dense(count: headDim * heads, noBias: true, name: "to_q")
  let toK = Dense(count: headDim * keyValueHeads, noBias: true, name: "to_k")
  let toV = Dense(count: headDim * keyValueHeads, noBias: true, name: "to_v")
  let toGate = Dense(count: hiddenSize, noBias: true, name: "to_gate")
  var queries = toQ(x).reshaped([batchSize, tokenLength, heads, headDim])
  let normQ = RMSNorm(epsilon: Krea2Config.normEps, axis: [3], name: "norm_q")
  queries = normQ(queries).to(FloatType.dataType)
  var keys = toK(x).reshaped([batchSize, tokenLength, keyValueHeads, headDim])
  let normK = RMSNorm(epsilon: Krea2Config.normEps, axis: [3], name: "norm_k")
  keys = normK(keys).to(FloatType.dataType)
  let values = toV(x).reshaped([batchSize, tokenLength, keyValueHeads, headDim])
  if let rot {
    queries = Functional.cmul(left: queries, right: rot)
    keys = Functional.cmul(left: keys, right: rot)
  }
  let attention = explicitAttentionNHWC(
    queries: queries, keys: keys, values: values, batchSize: batchSize, tokenLength: tokenLength,
    heads: heads, keyValueHeads: keyValueHeads, headDim: headDim)
  let gate = toGate(x).sigmoid()
  let toOut = Dense(count: hiddenSize, noBias: true, name: "to_out")
  let out = toOut(attention .* gate)
  let reader: (PythonObject) -> Void = { stateDict in
    copyDenseWeight(toQ, stateDict, "\(prefix).to_q.weight")
    copyDenseWeight(toK, stateDict, "\(prefix).to_k.weight")
    copyDenseWeight(toV, stateDict, "\(prefix).to_v.weight")
    copyDenseWeight(toGate, stateDict, "\(prefix).to_gate.weight")
    copyDenseWeight(toOut, stateDict, "\(prefix).to_out.0.weight")
    copyRMSNorm(normQ, stateDict, "\(prefix).norm_q.weight")
    copyRMSNorm(normK, stateDict, "\(prefix).norm_k.weight")
  }
  if let rot {
    return (Model([x, rot], [out]), reader)
  }
  return (Model([x], [out]), reader)
}

func Krea2TextFusionBlock(prefix: String, batchSize: Int, tokenLength: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let norm1 = RMSNorm(epsilon: Krea2Config.normEps, axis: [2], name: "norm1")
  let (attention, attentionReader) = Krea2Attention(
    prefix: "\(prefix).attn", hiddenSize: Krea2Config.textHiddenSize,
    heads: Krea2Config.textAttentionHeads, keyValueHeads: Krea2Config.textKeyValueHeads,
    batchSize: batchSize, tokenLength: tokenLength, rotary: false)
  var out = x + attention(norm1(x).to(FloatType.dataType))
  let norm2 = RMSNorm(epsilon: Krea2Config.normEps, axis: [2], name: "norm2")
  let (ff, ffReader) = Krea2SwiGLU(
    prefix: "\(prefix).ff", hiddenSize: Krea2Config.textHiddenSize,
    intermediateSize: Krea2Config.textIntermediateSize)
  out = out + ff(norm2(out).to(FloatType.dataType))
  let reader: (PythonObject) -> Void = { stateDict in
    copyRMSNorm(norm1, stateDict, "\(prefix).norm1.weight")
    copyRMSNorm(norm2, stateDict, "\(prefix).norm2.weight")
    attentionReader(stateDict)
    ffReader(stateDict)
  }
  return (Model([x], [out]), reader)
}

func Krea2TextFusion(batchSize: Int, textLength: Int) -> (Model, (PythonObject) -> Void) {
  let x = Input()
  var out = x.reshaped([
    batchSize * textLength, Krea2Config.textLayers, Krea2Config.textHiddenSize,
  ])
  var readers = [(PythonObject) -> Void]()
  for i in 0..<Krea2Config.textLayerwiseBlocks {
    let (block, reader) = Krea2TextFusionBlock(
      prefix: "text_fusion.layerwise_blocks.\(i)", batchSize: batchSize * textLength,
      tokenLength: Krea2Config.textLayers)
    out = block(out)
    readers.append(reader)
  }
  out = out.reshaped([
    batchSize, textLength, Krea2Config.textLayers, Krea2Config.textHiddenSize,
  ])
  .permuted(0, 1, 3, 2)
  let projector = Dense(count: 1, noBias: true, name: "projector")
  out = out.reshaped([
    batchSize * textLength * Krea2Config.textHiddenSize, Krea2Config.textLayers,
  ])
  out = projector(out).reshaped([batchSize, textLength, Krea2Config.textHiddenSize])
  for i in 0..<Krea2Config.textRefinerBlocks {
    let (block, reader) = Krea2TextFusionBlock(
      prefix: "text_fusion.refiner_blocks.\(i)", batchSize: batchSize, tokenLength: textLength)
    out = block(out)
    readers.append(reader)
  }
  let reader: (PythonObject) -> Void = { stateDict in
    copyDenseWeight(projector, stateDict, "text_fusion.projector.weight")
    for reader in readers {
      reader(stateDict)
    }
  }
  return (Model([x], [out]), reader)
}

func Krea2TextProjection() -> (Model, (PythonObject) -> Void) {
  let x = Input()
  let norm = RMSNorm(epsilon: Krea2Config.normEps, axis: [2], name: "norm")
  let linear1 = Dense(count: Krea2Config.hiddenSize, name: "linear_1")
  let linear2 = Dense(count: Krea2Config.hiddenSize, name: "linear_2")
  let out = linear2(linear1(norm(x).to(FloatType.dataType)).GELU(approximate: .tanh))
  let reader: (PythonObject) -> Void = { stateDict in
    copyRMSNorm(norm, stateDict, "txt_in.norm.weight")
    copyDense(linear1, stateDict, weight: "txt_in.linear_1.weight", bias: "txt_in.linear_1.bias")
    copyDense(linear2, stateDict, weight: "txt_in.linear_2.weight", bias: "txt_in.linear_2.bias")
  }
  return (Model([x], [out]), reader)
}

func Krea2TransformerBlock(prefix: String, batchSize: Int, tokenLength: Int, deviceID: Int) -> (
  Model, (PythonObject) -> Void
) {
  let x = Input()
  let tembMod = Input()
  let rot = Input()
  let prescaleTable = Parameter<FloatType>(
    .GPU(deviceID), .CHW(1, 1, Krea2Config.hiddenSize), name: "scale_shift_table_0")
  let preshiftTable = Parameter<FloatType>(
    .GPU(deviceID), .CHW(1, 1, Krea2Config.hiddenSize), name: "scale_shift_table_1")
  let pregateTable = Parameter<FloatType>(
    .GPU(deviceID), .CHW(1, 1, Krea2Config.hiddenSize), name: "scale_shift_table_2")
  let postscaleTable = Parameter<FloatType>(
    .GPU(deviceID), .CHW(1, 1, Krea2Config.hiddenSize), name: "scale_shift_table_3")
  let postshiftTable = Parameter<FloatType>(
    .GPU(deviceID), .CHW(1, 1, Krea2Config.hiddenSize), name: "scale_shift_table_4")
  let postgateTable = Parameter<FloatType>(
    .GPU(deviceID), .CHW(1, 1, Krea2Config.hiddenSize), name: "scale_shift_table_5")
  let prescale = sliceModulation(tembMod, row: 0) + prescaleTable
  let preshift = sliceModulation(tembMod, row: 1) + preshiftTable
  let pregate = sliceModulation(tembMod, row: 2) + pregateTable
  let postscale = sliceModulation(tembMod, row: 3) + postscaleTable
  let postshift = sliceModulation(tembMod, row: 4) + postshiftTable
  let postgate = sliceModulation(tembMod, row: 5) + postgateTable
  let norm1 = RMSNorm(epsilon: Krea2Config.normEps, axis: [2], name: "norm1")
  let (attention, attentionReader) = Krea2Attention(
    prefix: "\(prefix).attn", hiddenSize: Krea2Config.hiddenSize, heads: Krea2Config.attentionHeads,
    keyValueHeads: Krea2Config.keyValueHeads, batchSize: batchSize, tokenLength: tokenLength,
    rotary: true)
  var out =
    x + pregate
    .* attention(((1 + prescale) .* norm1(x).to(FloatType.dataType) + preshift), rot)
  let norm2 = RMSNorm(epsilon: Krea2Config.normEps, axis: [2], name: "norm2")
  let (ff, ffReader) = Krea2SwiGLU(
    prefix: "\(prefix).ff", hiddenSize: Krea2Config.hiddenSize,
    intermediateSize: Krea2Config.intermediateSize)
  out = out + postgate .* ff((1 + postscale) .* norm2(out).to(FloatType.dataType) + postshift)
  let reader: (PythonObject) -> Void = { stateDict in
    copyRMSNorm(norm1, stateDict, "\(prefix).norm1.weight")
    copyRMSNorm(norm2, stateDict, "\(prefix).norm2.weight")
    attentionReader(stateDict)
    ffReader(stateDict)
    copyParameter(prescaleTable, stateDict, "\(prefix).scale_shift_table", row: 0)
    copyParameter(preshiftTable, stateDict, "\(prefix).scale_shift_table", row: 1)
    copyParameter(pregateTable, stateDict, "\(prefix).scale_shift_table", row: 2)
    copyParameter(postscaleTable, stateDict, "\(prefix).scale_shift_table", row: 3)
    copyParameter(postshiftTable, stateDict, "\(prefix).scale_shift_table", row: 4)
    copyParameter(postgateTable, stateDict, "\(prefix).scale_shift_table", row: 5)
  }
  return (Model([x, tembMod, rot], [out]), reader)
}

func Krea2DiT(batchSize: Int, textLength: Int, gridHeight: Int, gridWidth: Int, deviceID: Int) -> (
  Model, (PythonObject) -> Void
) {
  let imageLength = gridHeight * gridWidth
  let tokenLength = textLength + imageLength
  let x = Input()
  let text = Input()
  let rot = Input()
  let tEmbed = Input()
  let imgIn = Dense(count: Krea2Config.hiddenSize, name: "img_in")
  let timeLinear1 = Dense(count: Krea2Config.hiddenSize, name: "time_embed_linear_1")
  let timeLinear2 = Dense(count: Krea2Config.hiddenSize, name: "time_embed_linear_2")
  let timeModProj = Dense(count: 6 * Krea2Config.hiddenSize, name: "time_mod_proj")
  let temb = timeLinear2(timeLinear1(tEmbed).GELU(approximate: .tanh))
  let tembMod = timeModProj(temb.GELU(approximate: .tanh))
  let (textFusion, textFusionReader) = Krea2TextFusion(batchSize: batchSize, textLength: textLength)
  let (txtIn, txtInReader) = Krea2TextProjection()
  let textOut = txtIn(textFusion(text))
  let imageOut = imgIn(x)
  var out = Functional.concat(axis: 1, textOut, imageOut)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<Krea2Config.layers {
    let (block, reader) = Krea2TransformerBlock(
      prefix: "transformer_blocks.\(i)", batchSize: batchSize, tokenLength: tokenLength,
      deviceID: deviceID)
    out = block(out, tembMod, rot)
    readers.append(reader)
  }
  out = out.reshaped(
    [batchSize, imageLength, Krea2Config.hiddenSize], offset: [0, textLength, 0],
    strides: [tokenLength * Krea2Config.hiddenSize, Krea2Config.hiddenSize, 1]
  )
  .contiguous()
  let finalScaleTable = Parameter<FloatType>(
    .GPU(deviceID), .CHW(1, 1, Krea2Config.hiddenSize), name: "final_scale_shift_table_0")
  let finalShiftTable = Parameter<FloatType>(
    .GPU(deviceID), .CHW(1, 1, Krea2Config.hiddenSize), name: "final_scale_shift_table_1")
  let finalNorm = RMSNorm(epsilon: Krea2Config.normEps, axis: [2], name: "final_norm")
  let finalLinear = Dense(count: Krea2Config.inChannels, name: "final_linear")
  let finalScale = temb + finalScaleTable
  let finalShift = temb + finalShiftTable
  out = finalLinear((1 + finalScale) .* finalNorm(out).to(FloatType.dataType) + finalShift)
  let reader: (PythonObject) -> Void = { stateDict in
    copyDense(imgIn, stateDict, weight: "img_in.weight", bias: "img_in.bias")
    copyDense(
      timeLinear1, stateDict, weight: "time_embed.linear_1.weight",
      bias: "time_embed.linear_1.bias")
    copyDense(
      timeLinear2, stateDict, weight: "time_embed.linear_2.weight",
      bias: "time_embed.linear_2.bias")
    copyDense(
      timeModProj, stateDict, weight: "time_mod_proj.weight", bias: "time_mod_proj.bias")
    textFusionReader(stateDict)
    txtInReader(stateDict)
    for reader in readers {
      reader(stateDict)
    }
    copyParameter(finalScaleTable, stateDict, "final_layer.scale_shift_table", row: 0)
    copyParameter(finalShiftTable, stateDict, "final_layer.scale_shift_table", row: 1)
    copyRMSNorm(finalNorm, stateDict, "final_layer.norm.weight")
    copyDense(
      finalLinear, stateDict, weight: "final_layer.linear.weight",
      bias: "final_layer.linear.bias")
  }
  return (Model([x, text, rot, tEmbed], [out]), reader)
}

func runTextParity() -> Bool {
  print("krea2 qwen text parity: start")
  print("krea2 model root:", modelRoot)
  print("krea2 text swift dtype: float16")
  print("krea2 text reference dtype:", textReferenceDtype)
  print("krea2 text tokens:", textTokenLength)
  print("krea2 text capture layers:", Krea2QwenTextConfig.captureLayers)
  print("krea2 text max rel threshold:", parityMaxTextRelativeDiff)
  let stateDict = helper.load_krea_qwen_text_state(modelRoot)
  let tokenIds = helper.make_text_token_ids(textTokenLength)
  let reference = tensorFromPython(
    helper.run_krea_qwen_text_reference(
      modelRoot, stateDict, textTokenLength, deviceID, textReferenceDtype))
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
    let (model, reader) = KreaQwenTextFeatures(tokenLength: tokenCount)
    model.maxConcurrency = .limit(1)
    print("krea2 qwen text compile")
    model.compile(inputs: tokensGPU, rotGPU)
    print("krea2 qwen text load weights")
    reader(stateDict)
    print("krea2 qwen text run swift")
    let outputs = model(inputs: tokensGPU, rotGPU)
    print("krea2 qwen text output count:", outputs.count, "reference shape:", reference.shape)
    var maxAbs: Float = 0
    var maxRel: Float = 0
    for i in 0..<outputs.count {
      let swift = copiedToCPU(outputs[i].as(of: Float.self))
      let columnOffset = i * Krea2QwenTextConfig.hiddenSize
      let absDiff = maxAbsDiff2D(swift, reference, rhsColumnOffset: columnOffset)
      let relDiff = maxRelativeDiff2D(swift, reference, rhsColumnOffset: columnOffset)
      print("krea2 qwen text tap \(i) layer:", Krea2QwenTextConfig.captureLayers[i])
      print("krea2 qwen text tap \(i) shape:", swift.shape)
      print("krea2 qwen text tap \(i) max abs diff:", absDiff)
      print("krea2 qwen text tap \(i) max rel diff:", relDiff)
      maxAbs = max(maxAbs, absDiff)
      maxRel = max(maxRel, relDiff)
    }
    print("krea2 qwen text max abs diff:", maxAbs)
    print("krea2 qwen text max rel diff:", maxRel)
    return maxRel <= parityMaxTextRelativeDiff
  }
}

func runDiTParity() -> Bool {
  print("krea2 dit parity: start")
  print("krea2 model root:", modelRoot)
  print("krea2 swift dtype:", swiftDtypeName)
  print("krea2 reference dtype:", ditReferenceDtype)
  print("krea2 parity text tokens:", parityTextLength)
  print("krea2 parity grid:", parityGridHeight, parityGridWidth)
  print("krea2 parity max rel threshold:", parityMaxRelativeDiff)
  let stateDict = helper.load_transformer_state(modelRoot)
  let testCase = helper.run_transformer_case(
    stateDict, parityTextLength, parityGridHeight, parityGridWidth, parityTimestep, deviceID,
    ditReferenceDtype)
  torch.cuda.empty_cache()
  let imageLength = parityGridHeight * parityGridWidth
  let xCPU = tensorFromPython(testCase["x"])
  let textCPU = tensorFromPython(testCase["text"])
  let reference = tensorFromPython(testCase["reference"])
  let rotCPU = kreaRotary(
    textLength: parityTextLength, gridHeight: parityGridHeight, gridWidth: parityGridWidth)
  let tEmbedCPU = timestepEmbedding(parityTimestep)
  return graph.withNoGrad {
    let x = graph.variable(Tensor<FloatType>(from: xCPU).toGPU(deviceID))
      .reshaped(.HWC(1, imageLength, Krea2Config.inChannels))
    let text = graph.variable(Tensor<FloatType>(from: textCPU).toGPU(deviceID))
      .reshaped(
        .NHWC(1, parityTextLength, Krea2Config.textLayers, Krea2Config.textHiddenSize))
    let rot = graph.variable(Tensor<FloatType>(from: rotCPU).toGPU(deviceID))
    let tEmbed = graph.variable(Tensor<FloatType>(from: tEmbedCPU).toGPU(deviceID))
    let (model, reader) = Krea2DiT(
      batchSize: 1, textLength: parityTextLength, gridHeight: parityGridHeight,
      gridWidth: parityGridWidth, deviceID: deviceID)
    model.maxConcurrency = .limit(1)
    print("krea2 dit compile")
    model.compile(inputs: x, text, rot, tEmbed)
    print("krea2 dit load weights")
    reader(stateDict)
    print("krea2 dit run swift")
    let swift = copiedToCPU(model(inputs: x, text, rot, tEmbed)[0].as(of: FloatType.self))
    print("krea2 dit shape:", swift.shape, reference.shape)
    let (maxAbs, maxRel) = maxAbsAndRelativeDiff3D(swift, reference)
    printSamples("krea2 dit", swift, reference)
    print("krea2 dit max abs diff:", maxAbs)
    print("krea2 dit max rel diff:", maxRel)
    return maxRel <= parityMaxRelativeDiff
  }
}

func exportTextModel() {
  print("krea2 text export: start")
  print("krea2 text export tokens:", exportTextLength)
  print("krea2 text export path:", textExportPath)
  let stateDict = helper.load_krea_qwen_text_state(modelRoot)
  let rotCPU = makeQwenTextRotary(tokenLength: exportTextLength)
  graph.withNoGrad {
    let tokens = graph.variable(.CPU, format: .NHWC, shape: [exportTextLength], of: Int32.self)
    for i in 0..<exportTextLength {
      tokens[i] = 0
    }
    let tokensGPU = tokens.toGPU(deviceID)
    let rotGPU = graph.variable(Tensor<TextFloatType>(from: rotCPU).toGPU(deviceID))
    let (model, reader) = KreaQwenTextFeatures(tokenLength: exportTextLength)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: tokensGPU, rotGPU)
    reader(stateDict)
    graph.openStore(textExportPath) {
      $0.write("text_model", model: model)
    }
  }
  print("krea2 text export: done")
}

func exportDiT() {
  print("krea2 dit export: start")
  print("krea2 export text tokens:", exportTextLength)
  print("krea2 export grid:", exportGridHeight, exportGridWidth)
  print("krea2 export path:", ditExportPath)
  let stateDict = helper.load_transformer_state(modelRoot)
  let imageLength = exportGridHeight * exportGridWidth
  let rotCPU = kreaRotary(
    textLength: exportTextLength, gridHeight: exportGridHeight, gridWidth: exportGridWidth)
  let tEmbedCPU = timestepEmbedding(0)
  graph.withNoGrad {
    let x = graph.variable(
      .GPU(deviceID), .HWC(1, imageLength, Krea2Config.inChannels), of: FloatType.self)
    let text = graph.variable(
      .GPU(deviceID),
      .NHWC(1, exportTextLength, Krea2Config.textLayers, Krea2Config.textHiddenSize),
      of: FloatType.self)
    let rot = graph.variable(Tensor<FloatType>(from: rotCPU).toGPU(deviceID))
    let tEmbed = graph.variable(Tensor<FloatType>(from: tEmbedCPU).toGPU(deviceID))
    let (model, reader) = Krea2DiT(
      batchSize: 1, textLength: exportTextLength, gridHeight: exportGridHeight,
      gridWidth: exportGridWidth, deviceID: deviceID)
    model.maxConcurrency = .limit(1)
    model.compile(inputs: x, text, rot, tEmbed)
    reader(stateDict)
    graph.openStore(ditExportPath) {
      $0.write("dit", model: model)
    }
  }
  print("krea2 dit export: done")
}

func requireParity(_ ok: Bool, _ message: String) {
  if !ok {
    print(message)
    exit(1)
  }
}

switch mode {
case "parity-text":
  requireParity(runTextParity(), "Krea2 Qwen text parity failed")
case "parity-dit":
  requireParity(runDiTParity(), "Krea2 DiT parity failed")
case "parity":
  requireParity(runTextParity(), "Krea2 Qwen text parity failed")
  requireParity(runDiTParity(), "Krea2 DiT parity failed")
case "export-text":
  exportTextModel()
case "export-dit":
  exportDiT()
case "export":
  exportTextModel()
  exportDiT()
default:
  print("Usage: krea2 [parity|parity-text|parity-dit|export-text|export-dit|export]")
}
