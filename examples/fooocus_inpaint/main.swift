import Foundation
import NNC
import NNCPythonConversion
import PythonKit

let torch = Python.import("torch")

let state_dict = torch.load(
  "/home/liu/workspace/Fooocus/inpaint_v26.fooocus.patch", map_location: "cpu")

let head_state_dict = torch.load(
  "/home/liu/workspace/Fooocus/fooocus_inpaint_head.pth", map_location: "cpu")

let UNetXLBaseFixed: [String: [String]] = [
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": ["t-0-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": ["t-1-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn2.to_k.weight": ["t-2-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn2.to_v.weight": ["t-3-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": ["t-4-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": ["t-5-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn2.to_k.weight": ["t-6-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn2.to_v.weight": ["t-7-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": ["t-8-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": ["t-9-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn2.to_k.weight": ["t-10-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn2.to_v.weight": ["t-11-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn2.to_k.weight": ["t-12-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn2.to_v.weight": ["t-13-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn2.to_k.weight": ["t-14-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn2.to_v.weight": ["t-15-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn2.to_k.weight": ["t-16-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn2.to_v.weight": ["t-17-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn2.to_k.weight": ["t-18-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn2.to_v.weight": ["t-19-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn2.to_k.weight": ["t-20-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn2.to_v.weight": ["t-21-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn2.to_k.weight": ["t-22-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn2.to_v.weight": ["t-23-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn2.to_k.weight": ["t-24-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn2.to_v.weight": ["t-25-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn2.to_k.weight": ["t-26-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn2.to_v.weight": ["t-27-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": ["t-28-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": ["t-29-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn2.to_k.weight": ["t-30-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn2.to_v.weight": ["t-31-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn2.to_k.weight": ["t-32-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn2.to_v.weight": ["t-33-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn2.to_k.weight": ["t-34-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn2.to_v.weight": ["t-35-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn2.to_k.weight": ["t-36-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn2.to_v.weight": ["t-37-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn2.to_k.weight": ["t-38-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn2.to_v.weight": ["t-39-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn2.to_k.weight": ["t-40-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn2.to_v.weight": ["t-41-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn2.to_k.weight": ["t-42-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn2.to_v.weight": ["t-43-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn2.to_k.weight": ["t-44-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn2.to_v.weight": ["t-45-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn2.to_k.weight": ["t-46-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn2.to_v.weight": ["t-47-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight": ["t-48-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight": ["t-49-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn2.to_k.weight": ["t-50-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn2.to_v.weight": ["t-51-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn2.to_k.weight": ["t-52-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn2.to_v.weight": ["t-53-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn2.to_k.weight": ["t-54-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn2.to_v.weight": ["t-55-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn2.to_k.weight": ["t-56-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn2.to_v.weight": ["t-57-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn2.to_k.weight": ["t-58-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn2.to_v.weight": ["t-59-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn2.to_k.weight": ["t-60-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn2.to_v.weight": ["t-61-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn2.to_k.weight": ["t-62-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn2.to_v.weight": ["t-63-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn2.to_k.weight": ["t-64-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn2.to_v.weight": ["t-65-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn2.to_k.weight": ["t-66-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn2.to_v.weight": ["t-67-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn2.to_k.weight": ["t-68-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn2.to_v.weight": ["t-69-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn2.to_k.weight": ["t-70-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn2.to_v.weight": ["t-71-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn2.to_k.weight": ["t-72-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn2.to_v.weight": ["t-73-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn2.to_k.weight": ["t-74-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn2.to_v.weight": ["t-75-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn2.to_k.weight": ["t-76-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn2.to_v.weight": ["t-77-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn2.to_k.weight": ["t-78-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn2.to_v.weight": ["t-79-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn2.to_k.weight": ["t-80-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn2.to_v.weight": ["t-81-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn2.to_k.weight": ["t-82-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn2.to_v.weight": ["t-83-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn2.to_k.weight": ["t-84-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn2.to_v.weight": ["t-85-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn2.to_k.weight": ["t-86-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn2.to_v.weight": ["t-87-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": ["t-88-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn2.to_v.weight": ["t-89-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn2.to_k.weight": ["t-90-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn2.to_v.weight": ["t-91-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn2.to_k.weight": ["t-92-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn2.to_v.weight": ["t-93-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn2.to_k.weight": ["t-94-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn2.to_v.weight": ["t-95-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn2.to_k.weight": ["t-96-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn2.to_v.weight": ["t-97-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn2.to_k.weight": ["t-98-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn2.to_v.weight": ["t-99-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn2.to_k.weight": ["t-100-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn2.to_v.weight": ["t-101-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn2.to_k.weight": ["t-102-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn2.to_v.weight": ["t-103-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn2.to_k.weight": ["t-104-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn2.to_v.weight": ["t-105-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn2.to_k.weight": ["t-106-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn2.to_v.weight": ["t-107-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": ["t-108-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn2.to_v.weight": ["t-109-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn2.to_k.weight": ["t-110-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn2.to_v.weight": ["t-111-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn2.to_k.weight": ["t-112-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn2.to_v.weight": ["t-113-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn2.to_k.weight": ["t-114-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn2.to_v.weight": ["t-115-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn2.to_k.weight": ["t-116-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn2.to_v.weight": ["t-117-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn2.to_k.weight": ["t-118-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn2.to_v.weight": ["t-119-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn2.to_k.weight": ["t-120-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn2.to_v.weight": ["t-121-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn2.to_k.weight": ["t-122-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn2.to_v.weight": ["t-123-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn2.to_k.weight": ["t-124-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn2.to_v.weight": ["t-125-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn2.to_k.weight": ["t-126-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn2.to_v.weight": ["t-127-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight": ["t-128-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight": ["t-129-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn2.to_k.weight": ["t-130-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn2.to_v.weight": ["t-131-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": ["t-132-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": ["t-133-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn2.to_k.weight": ["t-134-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn2.to_v.weight": ["t-135-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": ["t-136-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": ["t-137-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn2.to_k.weight": ["t-138-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn2.to_v.weight": ["t-139-0"],
]

let UNetXLBase: [String: [String]] = [
  "model.diffusion_model.time_embed.0.weight": ["t-2-0"],
  "model.diffusion_model.time_embed.0.bias": ["t-2-1"],
  "model.diffusion_model.time_embed.2.weight": ["t-3-0"],
  "model.diffusion_model.time_embed.2.bias": ["t-3-1"],
  "model.diffusion_model.label_emb.0.0.weight": ["t-0-0"],
  "model.diffusion_model.label_emb.0.0.bias": ["t-0-1"],
  "model.diffusion_model.label_emb.0.2.weight": ["t-1-0"],
  "model.diffusion_model.label_emb.0.2.bias": ["t-1-1"],
  "model.diffusion_model.input_blocks.0.0.weight": ["t-4-0"],
  "model.diffusion_model.input_blocks.0.0.bias": ["t-4-1"],
  "model.diffusion_model.input_blocks.1.0.in_layers.0.weight": ["t-5-0"],
  "model.diffusion_model.input_blocks.1.0.in_layers.0.bias": ["t-5-1"],
  "model.diffusion_model.input_blocks.1.0.in_layers.2.weight": ["t-7-0"],
  "model.diffusion_model.input_blocks.1.0.in_layers.2.bias": ["t-7-1"],
  "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight": ["t-6-0"],
  "model.diffusion_model.input_blocks.1.0.emb_layers.1.bias": ["t-6-1"],
  "model.diffusion_model.input_blocks.1.0.out_layers.0.weight": ["t-8-0"],
  "model.diffusion_model.input_blocks.1.0.out_layers.0.bias": ["t-8-1"],
  "model.diffusion_model.input_blocks.1.0.out_layers.3.weight": ["t-9-0"],
  "model.diffusion_model.input_blocks.1.0.out_layers.3.bias": ["t-9-1"],
  "model.diffusion_model.input_blocks.2.0.in_layers.0.weight": ["t-10-0"],
  "model.diffusion_model.input_blocks.2.0.in_layers.0.bias": ["t-10-1"],
  "model.diffusion_model.input_blocks.2.0.in_layers.2.weight": ["t-12-0"],
  "model.diffusion_model.input_blocks.2.0.in_layers.2.bias": ["t-12-1"],
  "model.diffusion_model.input_blocks.2.0.emb_layers.1.weight": ["t-11-0"],
  "model.diffusion_model.input_blocks.2.0.emb_layers.1.bias": ["t-11-1"],
  "model.diffusion_model.input_blocks.2.0.out_layers.0.weight": ["t-13-0"],
  "model.diffusion_model.input_blocks.2.0.out_layers.0.bias": ["t-13-1"],
  "model.diffusion_model.input_blocks.2.0.out_layers.3.weight": ["t-14-0"],
  "model.diffusion_model.input_blocks.2.0.out_layers.3.bias": ["t-14-1"],
  "model.diffusion_model.input_blocks.3.0.op.weight": ["t-15-0"],
  "model.diffusion_model.input_blocks.3.0.op.bias": ["t-15-1"],
  "model.diffusion_model.input_blocks.4.0.in_layers.0.weight": ["t-16-0"],
  "model.diffusion_model.input_blocks.4.0.in_layers.0.bias": ["t-16-1"],
  "model.diffusion_model.input_blocks.4.0.in_layers.2.weight": ["t-18-0"],
  "model.diffusion_model.input_blocks.4.0.in_layers.2.bias": ["t-18-1"],
  "model.diffusion_model.input_blocks.4.0.emb_layers.1.weight": ["t-17-0"],
  "model.diffusion_model.input_blocks.4.0.emb_layers.1.bias": ["t-17-1"],
  "model.diffusion_model.input_blocks.4.0.out_layers.0.weight": ["t-19-0"],
  "model.diffusion_model.input_blocks.4.0.out_layers.0.bias": ["t-19-1"],
  "model.diffusion_model.input_blocks.4.0.out_layers.3.weight": ["t-20-0"],
  "model.diffusion_model.input_blocks.4.0.out_layers.3.bias": ["t-20-1"],
  "model.diffusion_model.input_blocks.4.0.skip_connection.weight": ["t-21-0"],
  "model.diffusion_model.input_blocks.4.0.skip_connection.bias": ["t-21-1"],
  "model.diffusion_model.input_blocks.4.1.norm.weight": ["t-22-0"],
  "model.diffusion_model.input_blocks.4.1.norm.bias": ["t-22-1"],
  "model.diffusion_model.input_blocks.4.1.proj_in.weight": ["t-23-0"],
  "model.diffusion_model.input_blocks.4.1.proj_in.bias": ["t-23-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": ["t-26-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": ["t-25-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": ["t-27-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": ["t-28-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-28-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-34-0", "t-33-0",
  ],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-34-1", "t-33-1",
  ],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight": ["t-35-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias": ["t-35-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": ["t-30-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": ["t-31-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-31-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.weight": ["t-24-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.bias": ["t-24-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.weight": ["t-29-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.bias": ["t-29-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.weight": ["t-32-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.bias": ["t-32-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn1.to_k.weight": ["t-38-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn1.to_q.weight": ["t-37-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn1.to_v.weight": ["t-39-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn1.to_out.0.weight": ["t-40-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-40-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-46-0", "t-45-0",
  ],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-46-1", "t-45-1",
  ],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.ff.net.2.weight": ["t-47-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.ff.net.2.bias": ["t-47-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn2.to_q.weight": ["t-42-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn2.to_out.0.weight": ["t-43-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-43-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.norm1.weight": ["t-36-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.norm1.bias": ["t-36-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.norm2.weight": ["t-41-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.norm2.bias": ["t-41-1"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.norm3.weight": ["t-44-0"],
  "model.diffusion_model.input_blocks.4.1.transformer_blocks.1.norm3.bias": ["t-44-1"],
  "model.diffusion_model.input_blocks.4.1.proj_out.weight": ["t-48-0"],
  "model.diffusion_model.input_blocks.4.1.proj_out.bias": ["t-48-1"],
  "model.diffusion_model.input_blocks.5.0.in_layers.0.weight": ["t-49-0"],
  "model.diffusion_model.input_blocks.5.0.in_layers.0.bias": ["t-49-1"],
  "model.diffusion_model.input_blocks.5.0.in_layers.2.weight": ["t-51-0"],
  "model.diffusion_model.input_blocks.5.0.in_layers.2.bias": ["t-51-1"],
  "model.diffusion_model.input_blocks.5.0.emb_layers.1.weight": ["t-50-0"],
  "model.diffusion_model.input_blocks.5.0.emb_layers.1.bias": ["t-50-1"],
  "model.diffusion_model.input_blocks.5.0.out_layers.0.weight": ["t-52-0"],
  "model.diffusion_model.input_blocks.5.0.out_layers.0.bias": ["t-52-1"],
  "model.diffusion_model.input_blocks.5.0.out_layers.3.weight": ["t-53-0"],
  "model.diffusion_model.input_blocks.5.0.out_layers.3.bias": ["t-53-1"],
  "model.diffusion_model.input_blocks.5.1.norm.weight": ["t-54-0"],
  "model.diffusion_model.input_blocks.5.1.norm.bias": ["t-54-1"],
  "model.diffusion_model.input_blocks.5.1.proj_in.weight": ["t-55-0"],
  "model.diffusion_model.input_blocks.5.1.proj_in.bias": ["t-55-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": ["t-58-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": ["t-57-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": ["t-59-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": ["t-60-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-60-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-66-0", "t-65-0",
  ],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-66-1", "t-65-1",
  ],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight": ["t-67-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias": ["t-67-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": ["t-62-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": ["t-63-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-63-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.weight": ["t-56-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.bias": ["t-56-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.weight": ["t-61-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.bias": ["t-61-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.weight": ["t-64-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.bias": ["t-64-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn1.to_k.weight": ["t-70-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn1.to_q.weight": ["t-69-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn1.to_v.weight": ["t-71-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn1.to_out.0.weight": ["t-72-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-72-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-78-0", "t-77-0",
  ],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-78-1", "t-77-1",
  ],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.ff.net.2.weight": ["t-79-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.ff.net.2.bias": ["t-79-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn2.to_q.weight": ["t-74-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn2.to_out.0.weight": ["t-75-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-75-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.norm1.weight": ["t-68-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.norm1.bias": ["t-68-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.norm2.weight": ["t-73-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.norm2.bias": ["t-73-1"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.norm3.weight": ["t-76-0"],
  "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.norm3.bias": ["t-76-1"],
  "model.diffusion_model.input_blocks.5.1.proj_out.weight": ["t-80-0"],
  "model.diffusion_model.input_blocks.5.1.proj_out.bias": ["t-80-1"],
  "model.diffusion_model.input_blocks.6.0.op.weight": ["t-81-0"],
  "model.diffusion_model.input_blocks.6.0.op.bias": ["t-81-1"],
  "model.diffusion_model.input_blocks.7.0.in_layers.0.weight": ["t-82-0"],
  "model.diffusion_model.input_blocks.7.0.in_layers.0.bias": ["t-82-1"],
  "model.diffusion_model.input_blocks.7.0.in_layers.2.weight": ["t-84-0"],
  "model.diffusion_model.input_blocks.7.0.in_layers.2.bias": ["t-84-1"],
  "model.diffusion_model.input_blocks.7.0.emb_layers.1.weight": ["t-83-0"],
  "model.diffusion_model.input_blocks.7.0.emb_layers.1.bias": ["t-83-1"],
  "model.diffusion_model.input_blocks.7.0.out_layers.0.weight": ["t-85-0"],
  "model.diffusion_model.input_blocks.7.0.out_layers.0.bias": ["t-85-1"],
  "model.diffusion_model.input_blocks.7.0.out_layers.3.weight": ["t-86-0"],
  "model.diffusion_model.input_blocks.7.0.out_layers.3.bias": ["t-86-1"],
  "model.diffusion_model.input_blocks.7.0.skip_connection.weight": ["t-87-0"],
  "model.diffusion_model.input_blocks.7.0.skip_connection.bias": ["t-87-1"],
  "model.diffusion_model.input_blocks.7.1.norm.weight": ["t-88-0"],
  "model.diffusion_model.input_blocks.7.1.norm.bias": ["t-88-1"],
  "model.diffusion_model.input_blocks.7.1.proj_in.weight": ["t-89-0"],
  "model.diffusion_model.input_blocks.7.1.proj_in.bias": ["t-89-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight": ["t-92-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight": ["t-91-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight": ["t-93-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight": ["t-94-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-94-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-100-0", "t-99-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-100-1", "t-99-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight": ["t-101-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias": ["t-101-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight": ["t-96-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight": ["t-97-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-97-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.weight": ["t-90-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.bias": ["t-90-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.weight": ["t-95-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.bias": ["t-95-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.weight": ["t-98-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.bias": ["t-98-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn1.to_k.weight": ["t-104-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn1.to_q.weight": ["t-103-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn1.to_v.weight": ["t-105-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn1.to_out.0.weight": [
    "t-106-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-106-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-112-0", "t-111-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-112-1", "t-111-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.ff.net.2.weight": ["t-113-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.ff.net.2.bias": ["t-113-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn2.to_q.weight": ["t-108-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn2.to_out.0.weight": [
    "t-109-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-109-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.norm1.weight": ["t-102-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.norm1.bias": ["t-102-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.norm2.weight": ["t-107-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.norm2.bias": ["t-107-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.norm3.weight": ["t-110-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.norm3.bias": ["t-110-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn1.to_k.weight": ["t-116-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn1.to_q.weight": ["t-115-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn1.to_v.weight": ["t-117-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn1.to_out.0.weight": [
    "t-118-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn1.to_out.0.bias": ["t-118-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.ff.net.0.proj.weight": [
    "t-124-0", "t-123-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.ff.net.0.proj.bias": [
    "t-124-1", "t-123-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.ff.net.2.weight": ["t-125-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.ff.net.2.bias": ["t-125-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn2.to_q.weight": ["t-120-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn2.to_out.0.weight": [
    "t-121-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.attn2.to_out.0.bias": ["t-121-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.norm1.weight": ["t-114-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.norm1.bias": ["t-114-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.norm2.weight": ["t-119-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.norm2.bias": ["t-119-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.norm3.weight": ["t-122-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.2.norm3.bias": ["t-122-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn1.to_k.weight": ["t-128-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn1.to_q.weight": ["t-127-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn1.to_v.weight": ["t-129-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn1.to_out.0.weight": [
    "t-130-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn1.to_out.0.bias": ["t-130-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.ff.net.0.proj.weight": [
    "t-136-0", "t-135-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.ff.net.0.proj.bias": [
    "t-136-1", "t-135-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.ff.net.2.weight": ["t-137-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.ff.net.2.bias": ["t-137-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn2.to_q.weight": ["t-132-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn2.to_out.0.weight": [
    "t-133-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn2.to_out.0.bias": ["t-133-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.norm1.weight": ["t-126-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.norm1.bias": ["t-126-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.norm2.weight": ["t-131-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.norm2.bias": ["t-131-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.norm3.weight": ["t-134-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.norm3.bias": ["t-134-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn1.to_k.weight": ["t-140-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn1.to_q.weight": ["t-139-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn1.to_v.weight": ["t-141-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn1.to_out.0.weight": [
    "t-142-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn1.to_out.0.bias": ["t-142-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.ff.net.0.proj.weight": [
    "t-148-0", "t-147-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.ff.net.0.proj.bias": [
    "t-148-1", "t-147-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.ff.net.2.weight": ["t-149-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.ff.net.2.bias": ["t-149-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn2.to_q.weight": ["t-144-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn2.to_out.0.weight": [
    "t-145-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.attn2.to_out.0.bias": ["t-145-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.norm1.weight": ["t-138-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.norm1.bias": ["t-138-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.norm2.weight": ["t-143-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.norm2.bias": ["t-143-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.norm3.weight": ["t-146-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.4.norm3.bias": ["t-146-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn1.to_k.weight": ["t-152-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn1.to_q.weight": ["t-151-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn1.to_v.weight": ["t-153-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn1.to_out.0.weight": [
    "t-154-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn1.to_out.0.bias": ["t-154-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.ff.net.0.proj.weight": [
    "t-160-0", "t-159-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.ff.net.0.proj.bias": [
    "t-160-1", "t-159-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.ff.net.2.weight": ["t-161-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.ff.net.2.bias": ["t-161-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn2.to_q.weight": ["t-156-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn2.to_out.0.weight": [
    "t-157-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.attn2.to_out.0.bias": ["t-157-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.norm1.weight": ["t-150-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.norm1.bias": ["t-150-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.norm2.weight": ["t-155-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.norm2.bias": ["t-155-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.norm3.weight": ["t-158-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.5.norm3.bias": ["t-158-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn1.to_k.weight": ["t-164-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn1.to_q.weight": ["t-163-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn1.to_v.weight": ["t-165-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn1.to_out.0.weight": [
    "t-166-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn1.to_out.0.bias": ["t-166-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.ff.net.0.proj.weight": [
    "t-172-0", "t-171-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.ff.net.0.proj.bias": [
    "t-172-1", "t-171-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.ff.net.2.weight": ["t-173-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.ff.net.2.bias": ["t-173-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn2.to_q.weight": ["t-168-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn2.to_out.0.weight": [
    "t-169-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.attn2.to_out.0.bias": ["t-169-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.norm1.weight": ["t-162-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.norm1.bias": ["t-162-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.norm2.weight": ["t-167-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.norm2.bias": ["t-167-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.norm3.weight": ["t-170-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.6.norm3.bias": ["t-170-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn1.to_k.weight": ["t-176-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn1.to_q.weight": ["t-175-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn1.to_v.weight": ["t-177-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn1.to_out.0.weight": [
    "t-178-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn1.to_out.0.bias": ["t-178-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.ff.net.0.proj.weight": [
    "t-184-0", "t-183-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.ff.net.0.proj.bias": [
    "t-184-1", "t-183-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.ff.net.2.weight": ["t-185-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.ff.net.2.bias": ["t-185-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn2.to_q.weight": ["t-180-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn2.to_out.0.weight": [
    "t-181-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.attn2.to_out.0.bias": ["t-181-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.norm1.weight": ["t-174-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.norm1.bias": ["t-174-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.norm2.weight": ["t-179-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.norm2.bias": ["t-179-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.norm3.weight": ["t-182-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.7.norm3.bias": ["t-182-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn1.to_k.weight": ["t-188-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn1.to_q.weight": ["t-187-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn1.to_v.weight": ["t-189-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn1.to_out.0.weight": [
    "t-190-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn1.to_out.0.bias": ["t-190-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.ff.net.0.proj.weight": [
    "t-196-0", "t-195-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.ff.net.0.proj.bias": [
    "t-196-1", "t-195-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.ff.net.2.weight": ["t-197-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.ff.net.2.bias": ["t-197-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn2.to_q.weight": ["t-192-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn2.to_out.0.weight": [
    "t-193-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.attn2.to_out.0.bias": ["t-193-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.norm1.weight": ["t-186-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.norm1.bias": ["t-186-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.norm2.weight": ["t-191-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.norm2.bias": ["t-191-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.norm3.weight": ["t-194-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.8.norm3.bias": ["t-194-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn1.to_k.weight": ["t-200-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn1.to_q.weight": ["t-199-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn1.to_v.weight": ["t-201-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn1.to_out.0.weight": [
    "t-202-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn1.to_out.0.bias": ["t-202-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.ff.net.0.proj.weight": [
    "t-208-0", "t-207-0",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.ff.net.0.proj.bias": [
    "t-208-1", "t-207-1",
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.ff.net.2.weight": ["t-209-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.ff.net.2.bias": ["t-209-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn2.to_q.weight": ["t-204-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn2.to_out.0.weight": [
    "t-205-0"
  ],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn2.to_out.0.bias": ["t-205-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.norm1.weight": ["t-198-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.norm1.bias": ["t-198-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.norm2.weight": ["t-203-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.norm2.bias": ["t-203-1"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.norm3.weight": ["t-206-0"],
  "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.norm3.bias": ["t-206-1"],
  "model.diffusion_model.input_blocks.7.1.proj_out.weight": ["t-210-0"],
  "model.diffusion_model.input_blocks.7.1.proj_out.bias": ["t-210-1"],
  "model.diffusion_model.input_blocks.8.0.in_layers.0.weight": ["t-211-0"],
  "model.diffusion_model.input_blocks.8.0.in_layers.0.bias": ["t-211-1"],
  "model.diffusion_model.input_blocks.8.0.in_layers.2.weight": ["t-213-0"],
  "model.diffusion_model.input_blocks.8.0.in_layers.2.bias": ["t-213-1"],
  "model.diffusion_model.input_blocks.8.0.emb_layers.1.weight": ["t-212-0"],
  "model.diffusion_model.input_blocks.8.0.emb_layers.1.bias": ["t-212-1"],
  "model.diffusion_model.input_blocks.8.0.out_layers.0.weight": ["t-214-0"],
  "model.diffusion_model.input_blocks.8.0.out_layers.0.bias": ["t-214-1"],
  "model.diffusion_model.input_blocks.8.0.out_layers.3.weight": ["t-215-0"],
  "model.diffusion_model.input_blocks.8.0.out_layers.3.bias": ["t-215-1"],
  "model.diffusion_model.input_blocks.8.1.norm.weight": ["t-216-0"],
  "model.diffusion_model.input_blocks.8.1.norm.bias": ["t-216-1"],
  "model.diffusion_model.input_blocks.8.1.proj_in.weight": ["t-217-0"],
  "model.diffusion_model.input_blocks.8.1.proj_in.bias": ["t-217-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight": ["t-220-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight": ["t-219-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight": ["t-221-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight": [
    "t-222-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-222-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-228-0", "t-227-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-228-1", "t-227-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight": ["t-229-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias": ["t-229-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight": ["t-224-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight": [
    "t-225-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-225-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.weight": ["t-218-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.bias": ["t-218-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.weight": ["t-223-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.bias": ["t-223-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.weight": ["t-226-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.bias": ["t-226-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn1.to_k.weight": ["t-232-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn1.to_q.weight": ["t-231-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn1.to_v.weight": ["t-233-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn1.to_out.0.weight": [
    "t-234-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-234-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-240-0", "t-239-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-240-1", "t-239-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.ff.net.2.weight": ["t-241-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.ff.net.2.bias": ["t-241-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn2.to_q.weight": ["t-236-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn2.to_out.0.weight": [
    "t-237-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-237-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.norm1.weight": ["t-230-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.norm1.bias": ["t-230-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.norm2.weight": ["t-235-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.norm2.bias": ["t-235-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.norm3.weight": ["t-238-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.1.norm3.bias": ["t-238-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn1.to_k.weight": ["t-244-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn1.to_q.weight": ["t-243-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn1.to_v.weight": ["t-245-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn1.to_out.0.weight": [
    "t-246-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn1.to_out.0.bias": ["t-246-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.ff.net.0.proj.weight": [
    "t-252-0", "t-251-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.ff.net.0.proj.bias": [
    "t-252-1", "t-251-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.ff.net.2.weight": ["t-253-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.ff.net.2.bias": ["t-253-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn2.to_q.weight": ["t-248-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn2.to_out.0.weight": [
    "t-249-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.attn2.to_out.0.bias": ["t-249-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.norm1.weight": ["t-242-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.norm1.bias": ["t-242-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.norm2.weight": ["t-247-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.norm2.bias": ["t-247-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.norm3.weight": ["t-250-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.2.norm3.bias": ["t-250-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn1.to_k.weight": ["t-256-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn1.to_q.weight": ["t-255-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn1.to_v.weight": ["t-257-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn1.to_out.0.weight": [
    "t-258-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn1.to_out.0.bias": ["t-258-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.ff.net.0.proj.weight": [
    "t-264-0", "t-263-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.ff.net.0.proj.bias": [
    "t-264-1", "t-263-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.ff.net.2.weight": ["t-265-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.ff.net.2.bias": ["t-265-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn2.to_q.weight": ["t-260-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn2.to_out.0.weight": [
    "t-261-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.attn2.to_out.0.bias": ["t-261-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.norm1.weight": ["t-254-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.norm1.bias": ["t-254-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.norm2.weight": ["t-259-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.norm2.bias": ["t-259-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.norm3.weight": ["t-262-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.3.norm3.bias": ["t-262-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn1.to_k.weight": ["t-268-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn1.to_q.weight": ["t-267-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn1.to_v.weight": ["t-269-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn1.to_out.0.weight": [
    "t-270-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn1.to_out.0.bias": ["t-270-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.ff.net.0.proj.weight": [
    "t-276-0", "t-275-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.ff.net.0.proj.bias": [
    "t-276-1", "t-275-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.ff.net.2.weight": ["t-277-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.ff.net.2.bias": ["t-277-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn2.to_q.weight": ["t-272-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn2.to_out.0.weight": [
    "t-273-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.attn2.to_out.0.bias": ["t-273-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.norm1.weight": ["t-266-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.norm1.bias": ["t-266-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.norm2.weight": ["t-271-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.norm2.bias": ["t-271-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.norm3.weight": ["t-274-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.4.norm3.bias": ["t-274-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn1.to_k.weight": ["t-280-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn1.to_q.weight": ["t-279-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn1.to_v.weight": ["t-281-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn1.to_out.0.weight": [
    "t-282-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn1.to_out.0.bias": ["t-282-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.ff.net.0.proj.weight": [
    "t-288-0", "t-287-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.ff.net.0.proj.bias": [
    "t-288-1", "t-287-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.ff.net.2.weight": ["t-289-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.ff.net.2.bias": ["t-289-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn2.to_q.weight": ["t-284-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn2.to_out.0.weight": [
    "t-285-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.attn2.to_out.0.bias": ["t-285-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.norm1.weight": ["t-278-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.norm1.bias": ["t-278-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.norm2.weight": ["t-283-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.norm2.bias": ["t-283-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.norm3.weight": ["t-286-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.5.norm3.bias": ["t-286-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn1.to_k.weight": ["t-292-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn1.to_q.weight": ["t-291-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn1.to_v.weight": ["t-293-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn1.to_out.0.weight": [
    "t-294-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn1.to_out.0.bias": ["t-294-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.ff.net.0.proj.weight": [
    "t-300-0", "t-299-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.ff.net.0.proj.bias": [
    "t-300-1", "t-299-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.ff.net.2.weight": ["t-301-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.ff.net.2.bias": ["t-301-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn2.to_q.weight": ["t-296-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn2.to_out.0.weight": [
    "t-297-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.attn2.to_out.0.bias": ["t-297-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.norm1.weight": ["t-290-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.norm1.bias": ["t-290-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.norm2.weight": ["t-295-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.norm2.bias": ["t-295-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.norm3.weight": ["t-298-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.6.norm3.bias": ["t-298-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn1.to_k.weight": ["t-304-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn1.to_q.weight": ["t-303-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn1.to_v.weight": ["t-305-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn1.to_out.0.weight": [
    "t-306-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn1.to_out.0.bias": ["t-306-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.ff.net.0.proj.weight": [
    "t-312-0", "t-311-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.ff.net.0.proj.bias": [
    "t-312-1", "t-311-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.ff.net.2.weight": ["t-313-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.ff.net.2.bias": ["t-313-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn2.to_q.weight": ["t-308-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn2.to_out.0.weight": [
    "t-309-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.attn2.to_out.0.bias": ["t-309-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.norm1.weight": ["t-302-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.norm1.bias": ["t-302-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.norm2.weight": ["t-307-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.norm2.bias": ["t-307-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.norm3.weight": ["t-310-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.7.norm3.bias": ["t-310-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn1.to_k.weight": ["t-316-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn1.to_q.weight": ["t-315-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn1.to_v.weight": ["t-317-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn1.to_out.0.weight": [
    "t-318-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn1.to_out.0.bias": ["t-318-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.ff.net.0.proj.weight": [
    "t-324-0", "t-323-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.ff.net.0.proj.bias": [
    "t-324-1", "t-323-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.ff.net.2.weight": ["t-325-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.ff.net.2.bias": ["t-325-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn2.to_q.weight": ["t-320-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn2.to_out.0.weight": [
    "t-321-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.attn2.to_out.0.bias": ["t-321-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.norm1.weight": ["t-314-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.norm1.bias": ["t-314-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.norm2.weight": ["t-319-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.norm2.bias": ["t-319-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.norm3.weight": ["t-322-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.8.norm3.bias": ["t-322-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn1.to_k.weight": ["t-328-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn1.to_q.weight": ["t-327-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn1.to_v.weight": ["t-329-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn1.to_out.0.weight": [
    "t-330-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn1.to_out.0.bias": ["t-330-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.ff.net.0.proj.weight": [
    "t-336-0", "t-335-0",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.ff.net.0.proj.bias": [
    "t-336-1", "t-335-1",
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.ff.net.2.weight": ["t-337-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.ff.net.2.bias": ["t-337-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn2.to_q.weight": ["t-332-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn2.to_out.0.weight": [
    "t-333-0"
  ],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.attn2.to_out.0.bias": ["t-333-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.norm1.weight": ["t-326-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.norm1.bias": ["t-326-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.norm2.weight": ["t-331-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.norm2.bias": ["t-331-1"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.norm3.weight": ["t-334-0"],
  "model.diffusion_model.input_blocks.8.1.transformer_blocks.9.norm3.bias": ["t-334-1"],
  "model.diffusion_model.input_blocks.8.1.proj_out.weight": ["t-338-0"],
  "model.diffusion_model.input_blocks.8.1.proj_out.bias": ["t-338-1"],
  "model.diffusion_model.middle_block.0.in_layers.0.weight": ["t-339-0"],
  "model.diffusion_model.middle_block.0.in_layers.0.bias": ["t-339-1"],
  "model.diffusion_model.middle_block.0.in_layers.2.weight": ["t-341-0"],
  "model.diffusion_model.middle_block.0.in_layers.2.bias": ["t-341-1"],
  "model.diffusion_model.middle_block.0.emb_layers.1.weight": ["t-340-0"],
  "model.diffusion_model.middle_block.0.emb_layers.1.bias": ["t-340-1"],
  "model.diffusion_model.middle_block.0.out_layers.0.weight": ["t-342-0"],
  "model.diffusion_model.middle_block.0.out_layers.0.bias": ["t-342-1"],
  "model.diffusion_model.middle_block.0.out_layers.3.weight": ["t-343-0"],
  "model.diffusion_model.middle_block.0.out_layers.3.bias": ["t-343-1"],
  "model.diffusion_model.middle_block.1.norm.weight": ["t-344-0"],
  "model.diffusion_model.middle_block.1.norm.bias": ["t-344-1"],
  "model.diffusion_model.middle_block.1.proj_in.weight": ["t-345-0"],
  "model.diffusion_model.middle_block.1.proj_in.bias": ["t-345-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight": ["t-348-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight": ["t-347-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight": ["t-349-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight": ["t-350-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-350-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-356-0", "t-355-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-356-1", "t-355-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight": ["t-357-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias": ["t-357-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight": ["t-352-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight": ["t-353-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-353-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight": ["t-346-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias": ["t-346-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight": ["t-351-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias": ["t-351-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight": ["t-354-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias": ["t-354-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn1.to_k.weight": ["t-360-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn1.to_q.weight": ["t-359-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn1.to_v.weight": ["t-361-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn1.to_out.0.weight": ["t-362-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-362-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-368-0", "t-367-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-368-1", "t-367-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.ff.net.2.weight": ["t-369-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.ff.net.2.bias": ["t-369-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn2.to_q.weight": ["t-364-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn2.to_out.0.weight": ["t-365-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-365-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.norm1.weight": ["t-358-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.norm1.bias": ["t-358-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.norm2.weight": ["t-363-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.norm2.bias": ["t-363-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.norm3.weight": ["t-366-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.1.norm3.bias": ["t-366-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn1.to_k.weight": ["t-372-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn1.to_q.weight": ["t-371-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn1.to_v.weight": ["t-373-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn1.to_out.0.weight": ["t-374-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn1.to_out.0.bias": ["t-374-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.ff.net.0.proj.weight": [
    "t-380-0", "t-379-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.ff.net.0.proj.bias": [
    "t-380-1", "t-379-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.ff.net.2.weight": ["t-381-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.ff.net.2.bias": ["t-381-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn2.to_q.weight": ["t-376-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn2.to_out.0.weight": ["t-377-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.attn2.to_out.0.bias": ["t-377-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.norm1.weight": ["t-370-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.norm1.bias": ["t-370-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.norm2.weight": ["t-375-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.norm2.bias": ["t-375-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.norm3.weight": ["t-378-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.2.norm3.bias": ["t-378-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn1.to_k.weight": ["t-384-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn1.to_q.weight": ["t-383-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn1.to_v.weight": ["t-385-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn1.to_out.0.weight": ["t-386-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn1.to_out.0.bias": ["t-386-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.ff.net.0.proj.weight": [
    "t-392-0", "t-391-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.ff.net.0.proj.bias": [
    "t-392-1", "t-391-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.ff.net.2.weight": ["t-393-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.ff.net.2.bias": ["t-393-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn2.to_q.weight": ["t-388-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn2.to_out.0.weight": ["t-389-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.attn2.to_out.0.bias": ["t-389-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.norm1.weight": ["t-382-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.norm1.bias": ["t-382-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.norm2.weight": ["t-387-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.norm2.bias": ["t-387-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.norm3.weight": ["t-390-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.3.norm3.bias": ["t-390-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn1.to_k.weight": ["t-396-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn1.to_q.weight": ["t-395-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn1.to_v.weight": ["t-397-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn1.to_out.0.weight": ["t-398-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn1.to_out.0.bias": ["t-398-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.ff.net.0.proj.weight": [
    "t-404-0", "t-403-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.ff.net.0.proj.bias": [
    "t-404-1", "t-403-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.ff.net.2.weight": ["t-405-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.ff.net.2.bias": ["t-405-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn2.to_q.weight": ["t-400-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn2.to_out.0.weight": ["t-401-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.attn2.to_out.0.bias": ["t-401-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.norm1.weight": ["t-394-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.norm1.bias": ["t-394-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.norm2.weight": ["t-399-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.norm2.bias": ["t-399-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.norm3.weight": ["t-402-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.4.norm3.bias": ["t-402-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn1.to_k.weight": ["t-408-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn1.to_q.weight": ["t-407-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn1.to_v.weight": ["t-409-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn1.to_out.0.weight": ["t-410-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn1.to_out.0.bias": ["t-410-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.ff.net.0.proj.weight": [
    "t-416-0", "t-415-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.ff.net.0.proj.bias": [
    "t-416-1", "t-415-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.ff.net.2.weight": ["t-417-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.ff.net.2.bias": ["t-417-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn2.to_q.weight": ["t-412-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn2.to_out.0.weight": ["t-413-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.attn2.to_out.0.bias": ["t-413-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.norm1.weight": ["t-406-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.norm1.bias": ["t-406-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.norm2.weight": ["t-411-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.norm2.bias": ["t-411-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.norm3.weight": ["t-414-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.5.norm3.bias": ["t-414-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn1.to_k.weight": ["t-420-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn1.to_q.weight": ["t-419-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn1.to_v.weight": ["t-421-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn1.to_out.0.weight": ["t-422-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn1.to_out.0.bias": ["t-422-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.ff.net.0.proj.weight": [
    "t-428-0", "t-427-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.ff.net.0.proj.bias": [
    "t-428-1", "t-427-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.ff.net.2.weight": ["t-429-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.ff.net.2.bias": ["t-429-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn2.to_q.weight": ["t-424-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn2.to_out.0.weight": ["t-425-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.attn2.to_out.0.bias": ["t-425-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.norm1.weight": ["t-418-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.norm1.bias": ["t-418-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.norm2.weight": ["t-423-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.norm2.bias": ["t-423-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.norm3.weight": ["t-426-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.6.norm3.bias": ["t-426-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn1.to_k.weight": ["t-432-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn1.to_q.weight": ["t-431-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn1.to_v.weight": ["t-433-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn1.to_out.0.weight": ["t-434-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn1.to_out.0.bias": ["t-434-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.ff.net.0.proj.weight": [
    "t-440-0", "t-439-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.ff.net.0.proj.bias": [
    "t-440-1", "t-439-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.ff.net.2.weight": ["t-441-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.ff.net.2.bias": ["t-441-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn2.to_q.weight": ["t-436-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn2.to_out.0.weight": ["t-437-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.attn2.to_out.0.bias": ["t-437-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.norm1.weight": ["t-430-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.norm1.bias": ["t-430-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.norm2.weight": ["t-435-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.norm2.bias": ["t-435-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.norm3.weight": ["t-438-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.7.norm3.bias": ["t-438-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn1.to_k.weight": ["t-444-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn1.to_q.weight": ["t-443-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn1.to_v.weight": ["t-445-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn1.to_out.0.weight": ["t-446-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn1.to_out.0.bias": ["t-446-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.ff.net.0.proj.weight": [
    "t-452-0", "t-451-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.ff.net.0.proj.bias": [
    "t-452-1", "t-451-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.ff.net.2.weight": ["t-453-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.ff.net.2.bias": ["t-453-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn2.to_q.weight": ["t-448-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn2.to_out.0.weight": ["t-449-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.attn2.to_out.0.bias": ["t-449-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.norm1.weight": ["t-442-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.norm1.bias": ["t-442-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.norm2.weight": ["t-447-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.norm2.bias": ["t-447-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.norm3.weight": ["t-450-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.8.norm3.bias": ["t-450-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn1.to_k.weight": ["t-456-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn1.to_q.weight": ["t-455-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn1.to_v.weight": ["t-457-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn1.to_out.0.weight": ["t-458-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn1.to_out.0.bias": ["t-458-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.ff.net.0.proj.weight": [
    "t-464-0", "t-463-0",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.ff.net.0.proj.bias": [
    "t-464-1", "t-463-1",
  ],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.ff.net.2.weight": ["t-465-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.ff.net.2.bias": ["t-465-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn2.to_q.weight": ["t-460-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn2.to_out.0.weight": ["t-461-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.attn2.to_out.0.bias": ["t-461-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.norm1.weight": ["t-454-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.norm1.bias": ["t-454-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.norm2.weight": ["t-459-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.norm2.bias": ["t-459-1"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.norm3.weight": ["t-462-0"],
  "model.diffusion_model.middle_block.1.transformer_blocks.9.norm3.bias": ["t-462-1"],
  "model.diffusion_model.middle_block.1.proj_out.weight": ["t-466-0"],
  "model.diffusion_model.middle_block.1.proj_out.bias": ["t-466-1"],
  "model.diffusion_model.middle_block.2.in_layers.0.weight": ["t-467-0"],
  "model.diffusion_model.middle_block.2.in_layers.0.bias": ["t-467-1"],
  "model.diffusion_model.middle_block.2.in_layers.2.weight": ["t-469-0"],
  "model.diffusion_model.middle_block.2.in_layers.2.bias": ["t-469-1"],
  "model.diffusion_model.middle_block.2.emb_layers.1.weight": ["t-468-0"],
  "model.diffusion_model.middle_block.2.emb_layers.1.bias": ["t-468-1"],
  "model.diffusion_model.middle_block.2.out_layers.0.weight": ["t-470-0"],
  "model.diffusion_model.middle_block.2.out_layers.0.bias": ["t-470-1"],
  "model.diffusion_model.middle_block.2.out_layers.3.weight": ["t-471-0"],
  "model.diffusion_model.middle_block.2.out_layers.3.bias": ["t-471-1"],
  "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": ["t-472-0"],
  "model.diffusion_model.output_blocks.0.0.in_layers.0.bias": ["t-472-1"],
  "model.diffusion_model.output_blocks.0.0.in_layers.2.weight": ["t-474-0"],
  "model.diffusion_model.output_blocks.0.0.in_layers.2.bias": ["t-474-1"],
  "model.diffusion_model.output_blocks.0.0.emb_layers.1.weight": ["t-473-0"],
  "model.diffusion_model.output_blocks.0.0.emb_layers.1.bias": ["t-473-1"],
  "model.diffusion_model.output_blocks.0.0.out_layers.0.weight": ["t-475-0"],
  "model.diffusion_model.output_blocks.0.0.out_layers.0.bias": ["t-475-1"],
  "model.diffusion_model.output_blocks.0.0.out_layers.3.weight": ["t-476-0"],
  "model.diffusion_model.output_blocks.0.0.out_layers.3.bias": ["t-476-1"],
  "model.diffusion_model.output_blocks.0.0.skip_connection.weight": ["t-477-0"],
  "model.diffusion_model.output_blocks.0.0.skip_connection.bias": ["t-477-1"],
  "model.diffusion_model.output_blocks.0.1.norm.weight": ["t-478-0"],
  "model.diffusion_model.output_blocks.0.1.norm.bias": ["t-478-1"],
  "model.diffusion_model.output_blocks.0.1.proj_in.weight": ["t-479-0"],
  "model.diffusion_model.output_blocks.0.1.proj_in.bias": ["t-479-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn1.to_k.weight": ["t-482-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn1.to_q.weight": ["t-481-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn1.to_v.weight": ["t-483-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn1.to_out.0.weight": [
    "t-484-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-484-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-490-0", "t-489-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-490-1", "t-489-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.ff.net.2.weight": ["t-491-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.ff.net.2.bias": ["t-491-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn2.to_q.weight": ["t-486-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn2.to_out.0.weight": [
    "t-487-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-487-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.norm1.weight": ["t-480-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.norm1.bias": ["t-480-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.norm2.weight": ["t-485-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.norm2.bias": ["t-485-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.norm3.weight": ["t-488-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.0.norm3.bias": ["t-488-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn1.to_k.weight": ["t-494-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn1.to_q.weight": ["t-493-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn1.to_v.weight": ["t-495-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn1.to_out.0.weight": [
    "t-496-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-496-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-502-0", "t-501-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-502-1", "t-501-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.ff.net.2.weight": ["t-503-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.ff.net.2.bias": ["t-503-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn2.to_q.weight": ["t-498-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn2.to_out.0.weight": [
    "t-499-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-499-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.norm1.weight": ["t-492-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.norm1.bias": ["t-492-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.norm2.weight": ["t-497-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.norm2.bias": ["t-497-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.norm3.weight": ["t-500-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.1.norm3.bias": ["t-500-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn1.to_k.weight": ["t-506-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn1.to_q.weight": ["t-505-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn1.to_v.weight": ["t-507-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn1.to_out.0.weight": [
    "t-508-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn1.to_out.0.bias": ["t-508-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.ff.net.0.proj.weight": [
    "t-514-0", "t-513-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.ff.net.0.proj.bias": [
    "t-514-1", "t-513-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.ff.net.2.weight": ["t-515-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.ff.net.2.bias": ["t-515-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn2.to_q.weight": ["t-510-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn2.to_out.0.weight": [
    "t-511-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.attn2.to_out.0.bias": ["t-511-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.norm1.weight": ["t-504-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.norm1.bias": ["t-504-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.norm2.weight": ["t-509-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.norm2.bias": ["t-509-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.norm3.weight": ["t-512-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.2.norm3.bias": ["t-512-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn1.to_k.weight": ["t-518-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn1.to_q.weight": ["t-517-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn1.to_v.weight": ["t-519-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn1.to_out.0.weight": [
    "t-520-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn1.to_out.0.bias": ["t-520-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.ff.net.0.proj.weight": [
    "t-526-0", "t-525-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.ff.net.0.proj.bias": [
    "t-526-1", "t-525-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.ff.net.2.weight": ["t-527-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.ff.net.2.bias": ["t-527-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn2.to_q.weight": ["t-522-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn2.to_out.0.weight": [
    "t-523-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.attn2.to_out.0.bias": ["t-523-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.norm1.weight": ["t-516-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.norm1.bias": ["t-516-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.norm2.weight": ["t-521-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.norm2.bias": ["t-521-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.norm3.weight": ["t-524-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.3.norm3.bias": ["t-524-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn1.to_k.weight": ["t-530-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn1.to_q.weight": ["t-529-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn1.to_v.weight": ["t-531-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn1.to_out.0.weight": [
    "t-532-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn1.to_out.0.bias": ["t-532-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.ff.net.0.proj.weight": [
    "t-538-0", "t-537-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.ff.net.0.proj.bias": [
    "t-538-1", "t-537-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.ff.net.2.weight": ["t-539-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.ff.net.2.bias": ["t-539-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn2.to_q.weight": ["t-534-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn2.to_out.0.weight": [
    "t-535-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.attn2.to_out.0.bias": ["t-535-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.norm1.weight": ["t-528-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.norm1.bias": ["t-528-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.norm2.weight": ["t-533-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.norm2.bias": ["t-533-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.norm3.weight": ["t-536-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.4.norm3.bias": ["t-536-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn1.to_k.weight": ["t-542-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn1.to_q.weight": ["t-541-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn1.to_v.weight": ["t-543-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn1.to_out.0.weight": [
    "t-544-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn1.to_out.0.bias": ["t-544-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.ff.net.0.proj.weight": [
    "t-550-0", "t-549-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.ff.net.0.proj.bias": [
    "t-550-1", "t-549-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.ff.net.2.weight": ["t-551-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.ff.net.2.bias": ["t-551-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn2.to_q.weight": ["t-546-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn2.to_out.0.weight": [
    "t-547-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.attn2.to_out.0.bias": ["t-547-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.norm1.weight": ["t-540-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.norm1.bias": ["t-540-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.norm2.weight": ["t-545-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.norm2.bias": ["t-545-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.norm3.weight": ["t-548-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.5.norm3.bias": ["t-548-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn1.to_k.weight": ["t-554-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn1.to_q.weight": ["t-553-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn1.to_v.weight": ["t-555-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn1.to_out.0.weight": [
    "t-556-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn1.to_out.0.bias": ["t-556-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.ff.net.0.proj.weight": [
    "t-562-0", "t-561-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.ff.net.0.proj.bias": [
    "t-562-1", "t-561-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.ff.net.2.weight": ["t-563-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.ff.net.2.bias": ["t-563-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn2.to_q.weight": ["t-558-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn2.to_out.0.weight": [
    "t-559-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.attn2.to_out.0.bias": ["t-559-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.norm1.weight": ["t-552-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.norm1.bias": ["t-552-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.norm2.weight": ["t-557-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.norm2.bias": ["t-557-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.norm3.weight": ["t-560-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.6.norm3.bias": ["t-560-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn1.to_k.weight": ["t-566-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn1.to_q.weight": ["t-565-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn1.to_v.weight": ["t-567-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn1.to_out.0.weight": [
    "t-568-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn1.to_out.0.bias": ["t-568-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.ff.net.0.proj.weight": [
    "t-574-0", "t-573-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.ff.net.0.proj.bias": [
    "t-574-1", "t-573-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.ff.net.2.weight": ["t-575-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.ff.net.2.bias": ["t-575-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn2.to_q.weight": ["t-570-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn2.to_out.0.weight": [
    "t-571-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.attn2.to_out.0.bias": ["t-571-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.norm1.weight": ["t-564-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.norm1.bias": ["t-564-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.norm2.weight": ["t-569-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.norm2.bias": ["t-569-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.norm3.weight": ["t-572-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.7.norm3.bias": ["t-572-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn1.to_k.weight": ["t-578-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn1.to_q.weight": ["t-577-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn1.to_v.weight": ["t-579-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn1.to_out.0.weight": [
    "t-580-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn1.to_out.0.bias": ["t-580-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.ff.net.0.proj.weight": [
    "t-586-0", "t-585-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.ff.net.0.proj.bias": [
    "t-586-1", "t-585-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.ff.net.2.weight": ["t-587-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.ff.net.2.bias": ["t-587-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn2.to_q.weight": ["t-582-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn2.to_out.0.weight": [
    "t-583-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.attn2.to_out.0.bias": ["t-583-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.norm1.weight": ["t-576-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.norm1.bias": ["t-576-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.norm2.weight": ["t-581-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.norm2.bias": ["t-581-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.norm3.weight": ["t-584-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.8.norm3.bias": ["t-584-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn1.to_k.weight": ["t-590-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn1.to_q.weight": ["t-589-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn1.to_v.weight": ["t-591-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn1.to_out.0.weight": [
    "t-592-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn1.to_out.0.bias": ["t-592-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.ff.net.0.proj.weight": [
    "t-598-0", "t-597-0",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.ff.net.0.proj.bias": [
    "t-598-1", "t-597-1",
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.ff.net.2.weight": ["t-599-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.ff.net.2.bias": ["t-599-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn2.to_q.weight": ["t-594-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn2.to_out.0.weight": [
    "t-595-0"
  ],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.attn2.to_out.0.bias": ["t-595-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.norm1.weight": ["t-588-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.norm1.bias": ["t-588-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.norm2.weight": ["t-593-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.norm2.bias": ["t-593-1"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.norm3.weight": ["t-596-0"],
  "model.diffusion_model.output_blocks.0.1.transformer_blocks.9.norm3.bias": ["t-596-1"],
  "model.diffusion_model.output_blocks.0.1.proj_out.weight": ["t-600-0"],
  "model.diffusion_model.output_blocks.0.1.proj_out.bias": ["t-600-1"],
  "model.diffusion_model.output_blocks.1.0.in_layers.0.weight": ["t-601-0"],
  "model.diffusion_model.output_blocks.1.0.in_layers.0.bias": ["t-601-1"],
  "model.diffusion_model.output_blocks.1.0.in_layers.2.weight": ["t-603-0"],
  "model.diffusion_model.output_blocks.1.0.in_layers.2.bias": ["t-603-1"],
  "model.diffusion_model.output_blocks.1.0.emb_layers.1.weight": ["t-602-0"],
  "model.diffusion_model.output_blocks.1.0.emb_layers.1.bias": ["t-602-1"],
  "model.diffusion_model.output_blocks.1.0.out_layers.0.weight": ["t-604-0"],
  "model.diffusion_model.output_blocks.1.0.out_layers.0.bias": ["t-604-1"],
  "model.diffusion_model.output_blocks.1.0.out_layers.3.weight": ["t-605-0"],
  "model.diffusion_model.output_blocks.1.0.out_layers.3.bias": ["t-605-1"],
  "model.diffusion_model.output_blocks.1.0.skip_connection.weight": ["t-606-0"],
  "model.diffusion_model.output_blocks.1.0.skip_connection.bias": ["t-606-1"],
  "model.diffusion_model.output_blocks.1.1.norm.weight": ["t-607-0"],
  "model.diffusion_model.output_blocks.1.1.norm.bias": ["t-607-1"],
  "model.diffusion_model.output_blocks.1.1.proj_in.weight": ["t-608-0"],
  "model.diffusion_model.output_blocks.1.1.proj_in.bias": ["t-608-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn1.to_k.weight": ["t-611-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn1.to_q.weight": ["t-610-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn1.to_v.weight": ["t-612-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight": [
    "t-613-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-613-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-619-0", "t-618-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-619-1", "t-618-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.ff.net.2.weight": ["t-620-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.ff.net.2.bias": ["t-620-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn2.to_q.weight": ["t-615-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight": [
    "t-616-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-616-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.norm1.weight": ["t-609-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.norm1.bias": ["t-609-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.norm2.weight": ["t-614-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.norm2.bias": ["t-614-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.norm3.weight": ["t-617-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.0.norm3.bias": ["t-617-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn1.to_k.weight": ["t-623-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn1.to_q.weight": ["t-622-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn1.to_v.weight": ["t-624-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn1.to_out.0.weight": [
    "t-625-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-625-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-631-0", "t-630-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-631-1", "t-630-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.ff.net.2.weight": ["t-632-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.ff.net.2.bias": ["t-632-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn2.to_q.weight": ["t-627-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn2.to_out.0.weight": [
    "t-628-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-628-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.norm1.weight": ["t-621-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.norm1.bias": ["t-621-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.norm2.weight": ["t-626-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.norm2.bias": ["t-626-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.norm3.weight": ["t-629-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.1.norm3.bias": ["t-629-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn1.to_k.weight": ["t-635-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn1.to_q.weight": ["t-634-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn1.to_v.weight": ["t-636-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn1.to_out.0.weight": [
    "t-637-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn1.to_out.0.bias": ["t-637-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.ff.net.0.proj.weight": [
    "t-643-0", "t-642-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.ff.net.0.proj.bias": [
    "t-643-1", "t-642-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.ff.net.2.weight": ["t-644-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.ff.net.2.bias": ["t-644-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn2.to_q.weight": ["t-639-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn2.to_out.0.weight": [
    "t-640-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.attn2.to_out.0.bias": ["t-640-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.norm1.weight": ["t-633-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.norm1.bias": ["t-633-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.norm2.weight": ["t-638-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.norm2.bias": ["t-638-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.norm3.weight": ["t-641-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.2.norm3.bias": ["t-641-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn1.to_k.weight": ["t-647-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn1.to_q.weight": ["t-646-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn1.to_v.weight": ["t-648-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn1.to_out.0.weight": [
    "t-649-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn1.to_out.0.bias": ["t-649-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.ff.net.0.proj.weight": [
    "t-655-0", "t-654-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.ff.net.0.proj.bias": [
    "t-655-1", "t-654-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.ff.net.2.weight": ["t-656-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.ff.net.2.bias": ["t-656-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn2.to_q.weight": ["t-651-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn2.to_out.0.weight": [
    "t-652-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.attn2.to_out.0.bias": ["t-652-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.norm1.weight": ["t-645-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.norm1.bias": ["t-645-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.norm2.weight": ["t-650-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.norm2.bias": ["t-650-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.norm3.weight": ["t-653-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.3.norm3.bias": ["t-653-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn1.to_k.weight": ["t-659-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn1.to_q.weight": ["t-658-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn1.to_v.weight": ["t-660-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn1.to_out.0.weight": [
    "t-661-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn1.to_out.0.bias": ["t-661-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.ff.net.0.proj.weight": [
    "t-667-0", "t-666-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.ff.net.0.proj.bias": [
    "t-667-1", "t-666-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.ff.net.2.weight": ["t-668-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.ff.net.2.bias": ["t-668-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn2.to_q.weight": ["t-663-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn2.to_out.0.weight": [
    "t-664-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.attn2.to_out.0.bias": ["t-664-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.norm1.weight": ["t-657-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.norm1.bias": ["t-657-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.norm2.weight": ["t-662-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.norm2.bias": ["t-662-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.norm3.weight": ["t-665-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.4.norm3.bias": ["t-665-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn1.to_k.weight": ["t-671-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn1.to_q.weight": ["t-670-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn1.to_v.weight": ["t-672-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn1.to_out.0.weight": [
    "t-673-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn1.to_out.0.bias": ["t-673-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.ff.net.0.proj.weight": [
    "t-679-0", "t-678-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.ff.net.0.proj.bias": [
    "t-679-1", "t-678-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.ff.net.2.weight": ["t-680-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.ff.net.2.bias": ["t-680-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn2.to_q.weight": ["t-675-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn2.to_out.0.weight": [
    "t-676-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.attn2.to_out.0.bias": ["t-676-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.norm1.weight": ["t-669-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.norm1.bias": ["t-669-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.norm2.weight": ["t-674-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.norm2.bias": ["t-674-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.norm3.weight": ["t-677-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.5.norm3.bias": ["t-677-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn1.to_k.weight": ["t-683-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn1.to_q.weight": ["t-682-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn1.to_v.weight": ["t-684-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn1.to_out.0.weight": [
    "t-685-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn1.to_out.0.bias": ["t-685-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.ff.net.0.proj.weight": [
    "t-691-0", "t-690-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.ff.net.0.proj.bias": [
    "t-691-1", "t-690-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.ff.net.2.weight": ["t-692-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.ff.net.2.bias": ["t-692-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn2.to_q.weight": ["t-687-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn2.to_out.0.weight": [
    "t-688-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.attn2.to_out.0.bias": ["t-688-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.norm1.weight": ["t-681-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.norm1.bias": ["t-681-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.norm2.weight": ["t-686-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.norm2.bias": ["t-686-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.norm3.weight": ["t-689-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.6.norm3.bias": ["t-689-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn1.to_k.weight": ["t-695-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn1.to_q.weight": ["t-694-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn1.to_v.weight": ["t-696-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn1.to_out.0.weight": [
    "t-697-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn1.to_out.0.bias": ["t-697-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.ff.net.0.proj.weight": [
    "t-703-0", "t-702-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.ff.net.0.proj.bias": [
    "t-703-1", "t-702-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.ff.net.2.weight": ["t-704-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.ff.net.2.bias": ["t-704-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn2.to_q.weight": ["t-699-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn2.to_out.0.weight": [
    "t-700-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.attn2.to_out.0.bias": ["t-700-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.norm1.weight": ["t-693-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.norm1.bias": ["t-693-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.norm2.weight": ["t-698-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.norm2.bias": ["t-698-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.norm3.weight": ["t-701-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.7.norm3.bias": ["t-701-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn1.to_k.weight": ["t-707-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn1.to_q.weight": ["t-706-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn1.to_v.weight": ["t-708-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn1.to_out.0.weight": [
    "t-709-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn1.to_out.0.bias": ["t-709-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.ff.net.0.proj.weight": [
    "t-715-0", "t-714-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.ff.net.0.proj.bias": [
    "t-715-1", "t-714-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.ff.net.2.weight": ["t-716-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.ff.net.2.bias": ["t-716-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn2.to_q.weight": ["t-711-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn2.to_out.0.weight": [
    "t-712-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.attn2.to_out.0.bias": ["t-712-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.norm1.weight": ["t-705-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.norm1.bias": ["t-705-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.norm2.weight": ["t-710-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.norm2.bias": ["t-710-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.norm3.weight": ["t-713-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.8.norm3.bias": ["t-713-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn1.to_k.weight": ["t-719-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn1.to_q.weight": ["t-718-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn1.to_v.weight": ["t-720-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn1.to_out.0.weight": [
    "t-721-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn1.to_out.0.bias": ["t-721-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.ff.net.0.proj.weight": [
    "t-727-0", "t-726-0",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.ff.net.0.proj.bias": [
    "t-727-1", "t-726-1",
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.ff.net.2.weight": ["t-728-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.ff.net.2.bias": ["t-728-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn2.to_q.weight": ["t-723-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn2.to_out.0.weight": [
    "t-724-0"
  ],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.attn2.to_out.0.bias": ["t-724-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.norm1.weight": ["t-717-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.norm1.bias": ["t-717-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.norm2.weight": ["t-722-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.norm2.bias": ["t-722-1"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.norm3.weight": ["t-725-0"],
  "model.diffusion_model.output_blocks.1.1.transformer_blocks.9.norm3.bias": ["t-725-1"],
  "model.diffusion_model.output_blocks.1.1.proj_out.weight": ["t-729-0"],
  "model.diffusion_model.output_blocks.1.1.proj_out.bias": ["t-729-1"],
  "model.diffusion_model.output_blocks.2.0.in_layers.0.weight": ["t-730-0"],
  "model.diffusion_model.output_blocks.2.0.in_layers.0.bias": ["t-730-1"],
  "model.diffusion_model.output_blocks.2.0.in_layers.2.weight": ["t-732-0"],
  "model.diffusion_model.output_blocks.2.0.in_layers.2.bias": ["t-732-1"],
  "model.diffusion_model.output_blocks.2.0.emb_layers.1.weight": ["t-731-0"],
  "model.diffusion_model.output_blocks.2.0.emb_layers.1.bias": ["t-731-1"],
  "model.diffusion_model.output_blocks.2.0.out_layers.0.weight": ["t-733-0"],
  "model.diffusion_model.output_blocks.2.0.out_layers.0.bias": ["t-733-1"],
  "model.diffusion_model.output_blocks.2.0.out_layers.3.weight": ["t-734-0"],
  "model.diffusion_model.output_blocks.2.0.out_layers.3.bias": ["t-734-1"],
  "model.diffusion_model.output_blocks.2.0.skip_connection.weight": ["t-735-0"],
  "model.diffusion_model.output_blocks.2.0.skip_connection.bias": ["t-735-1"],
  "model.diffusion_model.output_blocks.2.1.norm.weight": ["t-736-0"],
  "model.diffusion_model.output_blocks.2.1.norm.bias": ["t-736-1"],
  "model.diffusion_model.output_blocks.2.1.proj_in.weight": ["t-737-0"],
  "model.diffusion_model.output_blocks.2.1.proj_in.bias": ["t-737-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn1.to_k.weight": ["t-740-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn1.to_q.weight": ["t-739-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn1.to_v.weight": ["t-741-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight": [
    "t-742-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-742-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-748-0", "t-747-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-748-1", "t-747-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.ff.net.2.weight": ["t-749-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.ff.net.2.bias": ["t-749-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn2.to_q.weight": ["t-744-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight": [
    "t-745-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-745-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.norm1.weight": ["t-738-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.norm1.bias": ["t-738-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.norm2.weight": ["t-743-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.norm2.bias": ["t-743-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.norm3.weight": ["t-746-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.0.norm3.bias": ["t-746-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn1.to_k.weight": ["t-752-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn1.to_q.weight": ["t-751-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn1.to_v.weight": ["t-753-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn1.to_out.0.weight": [
    "t-754-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-754-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-760-0", "t-759-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-760-1", "t-759-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.ff.net.2.weight": ["t-761-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.ff.net.2.bias": ["t-761-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn2.to_q.weight": ["t-756-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn2.to_out.0.weight": [
    "t-757-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-757-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.norm1.weight": ["t-750-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.norm1.bias": ["t-750-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.norm2.weight": ["t-755-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.norm2.bias": ["t-755-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.norm3.weight": ["t-758-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.1.norm3.bias": ["t-758-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn1.to_k.weight": ["t-764-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn1.to_q.weight": ["t-763-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn1.to_v.weight": ["t-765-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn1.to_out.0.weight": [
    "t-766-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn1.to_out.0.bias": ["t-766-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.ff.net.0.proj.weight": [
    "t-772-0", "t-771-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.ff.net.0.proj.bias": [
    "t-772-1", "t-771-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.ff.net.2.weight": ["t-773-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.ff.net.2.bias": ["t-773-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn2.to_q.weight": ["t-768-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn2.to_out.0.weight": [
    "t-769-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.attn2.to_out.0.bias": ["t-769-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.norm1.weight": ["t-762-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.norm1.bias": ["t-762-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.norm2.weight": ["t-767-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.norm2.bias": ["t-767-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.norm3.weight": ["t-770-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.2.norm3.bias": ["t-770-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn1.to_k.weight": ["t-776-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn1.to_q.weight": ["t-775-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn1.to_v.weight": ["t-777-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn1.to_out.0.weight": [
    "t-778-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn1.to_out.0.bias": ["t-778-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.ff.net.0.proj.weight": [
    "t-784-0", "t-783-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.ff.net.0.proj.bias": [
    "t-784-1", "t-783-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.ff.net.2.weight": ["t-785-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.ff.net.2.bias": ["t-785-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn2.to_q.weight": ["t-780-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn2.to_out.0.weight": [
    "t-781-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.attn2.to_out.0.bias": ["t-781-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.norm1.weight": ["t-774-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.norm1.bias": ["t-774-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.norm2.weight": ["t-779-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.norm2.bias": ["t-779-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.norm3.weight": ["t-782-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.3.norm3.bias": ["t-782-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn1.to_k.weight": ["t-788-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn1.to_q.weight": ["t-787-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn1.to_v.weight": ["t-789-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn1.to_out.0.weight": [
    "t-790-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn1.to_out.0.bias": ["t-790-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.ff.net.0.proj.weight": [
    "t-796-0", "t-795-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.ff.net.0.proj.bias": [
    "t-796-1", "t-795-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.ff.net.2.weight": ["t-797-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.ff.net.2.bias": ["t-797-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn2.to_q.weight": ["t-792-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn2.to_out.0.weight": [
    "t-793-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.attn2.to_out.0.bias": ["t-793-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.norm1.weight": ["t-786-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.norm1.bias": ["t-786-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.norm2.weight": ["t-791-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.norm2.bias": ["t-791-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.norm3.weight": ["t-794-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.4.norm3.bias": ["t-794-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn1.to_k.weight": ["t-800-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn1.to_q.weight": ["t-799-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn1.to_v.weight": ["t-801-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn1.to_out.0.weight": [
    "t-802-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn1.to_out.0.bias": ["t-802-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.ff.net.0.proj.weight": [
    "t-808-0", "t-807-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.ff.net.0.proj.bias": [
    "t-808-1", "t-807-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.ff.net.2.weight": ["t-809-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.ff.net.2.bias": ["t-809-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn2.to_q.weight": ["t-804-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn2.to_out.0.weight": [
    "t-805-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.attn2.to_out.0.bias": ["t-805-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.norm1.weight": ["t-798-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.norm1.bias": ["t-798-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.norm2.weight": ["t-803-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.norm2.bias": ["t-803-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.norm3.weight": ["t-806-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.5.norm3.bias": ["t-806-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn1.to_k.weight": ["t-812-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn1.to_q.weight": ["t-811-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn1.to_v.weight": ["t-813-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn1.to_out.0.weight": [
    "t-814-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn1.to_out.0.bias": ["t-814-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.ff.net.0.proj.weight": [
    "t-820-0", "t-819-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.ff.net.0.proj.bias": [
    "t-820-1", "t-819-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.ff.net.2.weight": ["t-821-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.ff.net.2.bias": ["t-821-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn2.to_q.weight": ["t-816-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn2.to_out.0.weight": [
    "t-817-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.attn2.to_out.0.bias": ["t-817-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.norm1.weight": ["t-810-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.norm1.bias": ["t-810-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.norm2.weight": ["t-815-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.norm2.bias": ["t-815-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.norm3.weight": ["t-818-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.6.norm3.bias": ["t-818-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn1.to_k.weight": ["t-824-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn1.to_q.weight": ["t-823-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn1.to_v.weight": ["t-825-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn1.to_out.0.weight": [
    "t-826-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn1.to_out.0.bias": ["t-826-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.ff.net.0.proj.weight": [
    "t-832-0", "t-831-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.ff.net.0.proj.bias": [
    "t-832-1", "t-831-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.ff.net.2.weight": ["t-833-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.ff.net.2.bias": ["t-833-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn2.to_q.weight": ["t-828-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn2.to_out.0.weight": [
    "t-829-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.attn2.to_out.0.bias": ["t-829-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.norm1.weight": ["t-822-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.norm1.bias": ["t-822-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.norm2.weight": ["t-827-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.norm2.bias": ["t-827-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.norm3.weight": ["t-830-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.7.norm3.bias": ["t-830-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn1.to_k.weight": ["t-836-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn1.to_q.weight": ["t-835-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn1.to_v.weight": ["t-837-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn1.to_out.0.weight": [
    "t-838-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn1.to_out.0.bias": ["t-838-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.ff.net.0.proj.weight": [
    "t-844-0", "t-843-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.ff.net.0.proj.bias": [
    "t-844-1", "t-843-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.ff.net.2.weight": ["t-845-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.ff.net.2.bias": ["t-845-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn2.to_q.weight": ["t-840-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn2.to_out.0.weight": [
    "t-841-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.attn2.to_out.0.bias": ["t-841-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.norm1.weight": ["t-834-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.norm1.bias": ["t-834-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.norm2.weight": ["t-839-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.norm2.bias": ["t-839-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.norm3.weight": ["t-842-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.8.norm3.bias": ["t-842-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn1.to_k.weight": ["t-848-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn1.to_q.weight": ["t-847-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn1.to_v.weight": ["t-849-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn1.to_out.0.weight": [
    "t-850-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn1.to_out.0.bias": ["t-850-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.ff.net.0.proj.weight": [
    "t-856-0", "t-855-0",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.ff.net.0.proj.bias": [
    "t-856-1", "t-855-1",
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.ff.net.2.weight": ["t-857-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.ff.net.2.bias": ["t-857-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn2.to_q.weight": ["t-852-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn2.to_out.0.weight": [
    "t-853-0"
  ],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.attn2.to_out.0.bias": ["t-853-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.norm1.weight": ["t-846-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.norm1.bias": ["t-846-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.norm2.weight": ["t-851-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.norm2.bias": ["t-851-1"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.norm3.weight": ["t-854-0"],
  "model.diffusion_model.output_blocks.2.1.transformer_blocks.9.norm3.bias": ["t-854-1"],
  "model.diffusion_model.output_blocks.2.1.proj_out.weight": ["t-858-0"],
  "model.diffusion_model.output_blocks.2.1.proj_out.bias": ["t-858-1"],
  "model.diffusion_model.output_blocks.2.2.conv.weight": ["t-859-0"],
  "model.diffusion_model.output_blocks.2.2.conv.bias": ["t-859-1"],
  "model.diffusion_model.output_blocks.3.0.in_layers.0.weight": ["t-860-0"],
  "model.diffusion_model.output_blocks.3.0.in_layers.0.bias": ["t-860-1"],
  "model.diffusion_model.output_blocks.3.0.in_layers.2.weight": ["t-862-0"],
  "model.diffusion_model.output_blocks.3.0.in_layers.2.bias": ["t-862-1"],
  "model.diffusion_model.output_blocks.3.0.emb_layers.1.weight": ["t-861-0"],
  "model.diffusion_model.output_blocks.3.0.emb_layers.1.bias": ["t-861-1"],
  "model.diffusion_model.output_blocks.3.0.out_layers.0.weight": ["t-863-0"],
  "model.diffusion_model.output_blocks.3.0.out_layers.0.bias": ["t-863-1"],
  "model.diffusion_model.output_blocks.3.0.out_layers.3.weight": ["t-864-0"],
  "model.diffusion_model.output_blocks.3.0.out_layers.3.bias": ["t-864-1"],
  "model.diffusion_model.output_blocks.3.0.skip_connection.weight": ["t-865-0"],
  "model.diffusion_model.output_blocks.3.0.skip_connection.bias": ["t-865-1"],
  "model.diffusion_model.output_blocks.3.1.norm.weight": ["t-866-0"],
  "model.diffusion_model.output_blocks.3.1.norm.bias": ["t-866-1"],
  "model.diffusion_model.output_blocks.3.1.proj_in.weight": ["t-867-0"],
  "model.diffusion_model.output_blocks.3.1.proj_in.bias": ["t-867-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight": ["t-870-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q.weight": ["t-869-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v.weight": ["t-871-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.weight": [
    "t-872-0"
  ],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-872-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-878-0", "t-877-0",
  ],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-878-1", "t-877-1",
  ],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.weight": ["t-879-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.bias": ["t-879-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q.weight": ["t-874-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.weight": [
    "t-875-0"
  ],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-875-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.weight": ["t-868-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.bias": ["t-868-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.weight": ["t-873-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.bias": ["t-873-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.weight": ["t-876-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.bias": ["t-876-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn1.to_k.weight": ["t-882-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn1.to_q.weight": ["t-881-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn1.to_v.weight": ["t-883-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn1.to_out.0.weight": [
    "t-884-0"
  ],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-884-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-890-0", "t-889-0",
  ],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-890-1", "t-889-1",
  ],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.ff.net.2.weight": ["t-891-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.ff.net.2.bias": ["t-891-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn2.to_q.weight": ["t-886-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn2.to_out.0.weight": [
    "t-887-0"
  ],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-887-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.norm1.weight": ["t-880-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.norm1.bias": ["t-880-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.norm2.weight": ["t-885-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.norm2.bias": ["t-885-1"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.norm3.weight": ["t-888-0"],
  "model.diffusion_model.output_blocks.3.1.transformer_blocks.1.norm3.bias": ["t-888-1"],
  "model.diffusion_model.output_blocks.3.1.proj_out.weight": ["t-892-0"],
  "model.diffusion_model.output_blocks.3.1.proj_out.bias": ["t-892-1"],
  "model.diffusion_model.output_blocks.4.0.in_layers.0.weight": ["t-893-0"],
  "model.diffusion_model.output_blocks.4.0.in_layers.0.bias": ["t-893-1"],
  "model.diffusion_model.output_blocks.4.0.in_layers.2.weight": ["t-895-0"],
  "model.diffusion_model.output_blocks.4.0.in_layers.2.bias": ["t-895-1"],
  "model.diffusion_model.output_blocks.4.0.emb_layers.1.weight": ["t-894-0"],
  "model.diffusion_model.output_blocks.4.0.emb_layers.1.bias": ["t-894-1"],
  "model.diffusion_model.output_blocks.4.0.out_layers.0.weight": ["t-896-0"],
  "model.diffusion_model.output_blocks.4.0.out_layers.0.bias": ["t-896-1"],
  "model.diffusion_model.output_blocks.4.0.out_layers.3.weight": ["t-897-0"],
  "model.diffusion_model.output_blocks.4.0.out_layers.3.bias": ["t-897-1"],
  "model.diffusion_model.output_blocks.4.0.skip_connection.weight": ["t-898-0"],
  "model.diffusion_model.output_blocks.4.0.skip_connection.bias": ["t-898-1"],
  "model.diffusion_model.output_blocks.4.1.norm.weight": ["t-899-0"],
  "model.diffusion_model.output_blocks.4.1.norm.bias": ["t-899-1"],
  "model.diffusion_model.output_blocks.4.1.proj_in.weight": ["t-900-0"],
  "model.diffusion_model.output_blocks.4.1.proj_in.bias": ["t-900-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": ["t-903-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": ["t-902-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": ["t-904-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": [
    "t-905-0"
  ],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-905-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-911-0", "t-910-0",
  ],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-911-1", "t-910-1",
  ],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.weight": ["t-912-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.bias": ["t-912-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": ["t-907-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": [
    "t-908-0"
  ],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-908-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.weight": ["t-901-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.bias": ["t-901-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.weight": ["t-906-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.bias": ["t-906-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.weight": ["t-909-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.bias": ["t-909-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn1.to_k.weight": ["t-915-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn1.to_q.weight": ["t-914-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn1.to_v.weight": ["t-916-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn1.to_out.0.weight": [
    "t-917-0"
  ],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-917-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-923-0", "t-922-0",
  ],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-923-1", "t-922-1",
  ],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.ff.net.2.weight": ["t-924-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.ff.net.2.bias": ["t-924-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn2.to_q.weight": ["t-919-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn2.to_out.0.weight": [
    "t-920-0"
  ],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-920-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.norm1.weight": ["t-913-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.norm1.bias": ["t-913-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.norm2.weight": ["t-918-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.norm2.bias": ["t-918-1"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.norm3.weight": ["t-921-0"],
  "model.diffusion_model.output_blocks.4.1.transformer_blocks.1.norm3.bias": ["t-921-1"],
  "model.diffusion_model.output_blocks.4.1.proj_out.weight": ["t-925-0"],
  "model.diffusion_model.output_blocks.4.1.proj_out.bias": ["t-925-1"],
  "model.diffusion_model.output_blocks.5.0.in_layers.0.weight": ["t-926-0"],
  "model.diffusion_model.output_blocks.5.0.in_layers.0.bias": ["t-926-1"],
  "model.diffusion_model.output_blocks.5.0.in_layers.2.weight": ["t-928-0"],
  "model.diffusion_model.output_blocks.5.0.in_layers.2.bias": ["t-928-1"],
  "model.diffusion_model.output_blocks.5.0.emb_layers.1.weight": ["t-927-0"],
  "model.diffusion_model.output_blocks.5.0.emb_layers.1.bias": ["t-927-1"],
  "model.diffusion_model.output_blocks.5.0.out_layers.0.weight": ["t-929-0"],
  "model.diffusion_model.output_blocks.5.0.out_layers.0.bias": ["t-929-1"],
  "model.diffusion_model.output_blocks.5.0.out_layers.3.weight": ["t-930-0"],
  "model.diffusion_model.output_blocks.5.0.out_layers.3.bias": ["t-930-1"],
  "model.diffusion_model.output_blocks.5.0.skip_connection.weight": ["t-931-0"],
  "model.diffusion_model.output_blocks.5.0.skip_connection.bias": ["t-931-1"],
  "model.diffusion_model.output_blocks.5.1.norm.weight": ["t-932-0"],
  "model.diffusion_model.output_blocks.5.1.norm.bias": ["t-932-1"],
  "model.diffusion_model.output_blocks.5.1.proj_in.weight": ["t-933-0"],
  "model.diffusion_model.output_blocks.5.1.proj_in.bias": ["t-933-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": ["t-936-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": ["t-935-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": ["t-937-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": [
    "t-938-0"
  ],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": ["t-938-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": [
    "t-944-0", "t-943-0",
  ],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": [
    "t-944-1", "t-943-1",
  ],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight": ["t-945-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.bias": ["t-945-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": ["t-940-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": [
    "t-941-0"
  ],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": ["t-941-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.weight": ["t-934-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.bias": ["t-934-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.weight": ["t-939-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.bias": ["t-939-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.weight": ["t-942-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.bias": ["t-942-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn1.to_k.weight": ["t-948-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn1.to_q.weight": ["t-947-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn1.to_v.weight": ["t-949-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn1.to_out.0.weight": [
    "t-950-0"
  ],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn1.to_out.0.bias": ["t-950-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.ff.net.0.proj.weight": [
    "t-956-0", "t-955-0",
  ],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.ff.net.0.proj.bias": [
    "t-956-1", "t-955-1",
  ],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.ff.net.2.weight": ["t-957-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.ff.net.2.bias": ["t-957-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn2.to_q.weight": ["t-952-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn2.to_out.0.weight": [
    "t-953-0"
  ],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.attn2.to_out.0.bias": ["t-953-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.norm1.weight": ["t-946-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.norm1.bias": ["t-946-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.norm2.weight": ["t-951-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.norm2.bias": ["t-951-1"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.norm3.weight": ["t-954-0"],
  "model.diffusion_model.output_blocks.5.1.transformer_blocks.1.norm3.bias": ["t-954-1"],
  "model.diffusion_model.output_blocks.5.1.proj_out.weight": ["t-958-0"],
  "model.diffusion_model.output_blocks.5.1.proj_out.bias": ["t-958-1"],
  "model.diffusion_model.output_blocks.5.2.conv.weight": ["t-959-0"],
  "model.diffusion_model.output_blocks.5.2.conv.bias": ["t-959-1"],
  "model.diffusion_model.output_blocks.6.0.in_layers.0.weight": ["t-960-0"],
  "model.diffusion_model.output_blocks.6.0.in_layers.0.bias": ["t-960-1"],
  "model.diffusion_model.output_blocks.6.0.in_layers.2.weight": ["t-962-0"],
  "model.diffusion_model.output_blocks.6.0.in_layers.2.bias": ["t-962-1"],
  "model.diffusion_model.output_blocks.6.0.emb_layers.1.weight": ["t-961-0"],
  "model.diffusion_model.output_blocks.6.0.emb_layers.1.bias": ["t-961-1"],
  "model.diffusion_model.output_blocks.6.0.out_layers.0.weight": ["t-963-0"],
  "model.diffusion_model.output_blocks.6.0.out_layers.0.bias": ["t-963-1"],
  "model.diffusion_model.output_blocks.6.0.out_layers.3.weight": ["t-964-0"],
  "model.diffusion_model.output_blocks.6.0.out_layers.3.bias": ["t-964-1"],
  "model.diffusion_model.output_blocks.6.0.skip_connection.weight": ["t-965-0"],
  "model.diffusion_model.output_blocks.6.0.skip_connection.bias": ["t-965-1"],
  "model.diffusion_model.output_blocks.7.0.in_layers.0.weight": ["t-966-0"],
  "model.diffusion_model.output_blocks.7.0.in_layers.0.bias": ["t-966-1"],
  "model.diffusion_model.output_blocks.7.0.in_layers.2.weight": ["t-968-0"],
  "model.diffusion_model.output_blocks.7.0.in_layers.2.bias": ["t-968-1"],
  "model.diffusion_model.output_blocks.7.0.emb_layers.1.weight": ["t-967-0"],
  "model.diffusion_model.output_blocks.7.0.emb_layers.1.bias": ["t-967-1"],
  "model.diffusion_model.output_blocks.7.0.out_layers.0.weight": ["t-969-0"],
  "model.diffusion_model.output_blocks.7.0.out_layers.0.bias": ["t-969-1"],
  "model.diffusion_model.output_blocks.7.0.out_layers.3.weight": ["t-970-0"],
  "model.diffusion_model.output_blocks.7.0.out_layers.3.bias": ["t-970-1"],
  "model.diffusion_model.output_blocks.7.0.skip_connection.weight": ["t-971-0"],
  "model.diffusion_model.output_blocks.7.0.skip_connection.bias": ["t-971-1"],
  "model.diffusion_model.output_blocks.8.0.in_layers.0.weight": ["t-972-0"],
  "model.diffusion_model.output_blocks.8.0.in_layers.0.bias": ["t-972-1"],
  "model.diffusion_model.output_blocks.8.0.in_layers.2.weight": ["t-974-0"],
  "model.diffusion_model.output_blocks.8.0.in_layers.2.bias": ["t-974-1"],
  "model.diffusion_model.output_blocks.8.0.emb_layers.1.weight": ["t-973-0"],
  "model.diffusion_model.output_blocks.8.0.emb_layers.1.bias": ["t-973-1"],
  "model.diffusion_model.output_blocks.8.0.out_layers.0.weight": ["t-975-0"],
  "model.diffusion_model.output_blocks.8.0.out_layers.0.bias": ["t-975-1"],
  "model.diffusion_model.output_blocks.8.0.out_layers.3.weight": ["t-976-0"],
  "model.diffusion_model.output_blocks.8.0.out_layers.3.bias": ["t-976-1"],
  "model.diffusion_model.output_blocks.8.0.skip_connection.weight": ["t-977-0"],
  "model.diffusion_model.output_blocks.8.0.skip_connection.bias": ["t-977-1"],
  "model.diffusion_model.out.0.weight": ["t-978-0"],
  "model.diffusion_model.out.0.bias": ["t-978-1"],
  "model.diffusion_model.out.2.weight": ["t-979-0"],
  "model.diffusion_model.out.2.bias": ["t-979-1"],
]

let graph = DynamicGraph()

let head = try! Tensor<Float>(numpy: head_state_dict["head"].float().cpu().numpy())

graph.openStore("/home/liu/workspace/swift-diffusion/fooocus_inpaint_sd_xl_v2.6_f16.ckpt") {
  store0 in
  graph.openStore("/home/liu/workspace/swift-diffusion/sd_xl_base_1.0_f16.ckpt") { store in
    let keys = state_dict.keys()
    var unhandledKeys = Set(store.keys)
    for key in keys {
      let torchTensor = state_dict[key][0].float()
      let smin = state_dict[key][1].float()
      let smax = state_dict[key][2].float()
      let tensor = ((torchTensor / 255.0) * (smax - smin) + smin).cpu().numpy()
      let extkey = "model.\(key)"
      if let v = UNetXLBase[extkey] {
        let tensor = try! Tensor<Float>(numpy: tensor)
        if v.count == 1, let v = v.first {
          let v = "__unet__[\(v)]"
          let existingTensor = Tensor<Float>(from: store.read(v)!).toCPU()
          let shape = tensor.shape
          if shape[1] == 4 && shape.count == 4 {
            let result = graph.variable(existingTensor) + graph.variable(tensor)
            // This is the dimension that receives the input mask etc. Now concat with head.
            var weight = Tensor<Float>(.CPU, .NCHW(shape[0], 9, shape[2], shape[3]))
            weight[0..<shape[0], 0..<4, 0..<shape[2], 0..<shape[3]] = result.rawValue
            weight[0..<shape[0], 4..<9, 0..<shape[2], 0..<shape[3]] = head
            store0.write(v, tensor: Tensor<Float16>(from: weight))
          } else {
            let result =
              graph.variable(existingTensor)
              + graph.variable(tensor).reshaped(format: .NCHW, shape: existingTensor.shape)
            store0.write(v, tensor: Tensor<Float16>(from: result.rawValue))
          }
          unhandledKeys.remove(v)
        } else {
          let shape = tensor.shape
          let count = shape[0] / v.count
          for (i, v) in v.enumerated() {
            let v = "__unet__[\(v)]"
            let existingTensor = Tensor<Float>(from: store.read(v)!).toCPU()
            if shape.count == 1 {
              let thisTensor = tensor[(i * count)..<((i + 1) * count)].copied()
              let result = graph.variable(existingTensor) + graph.variable(thisTensor)
              store0.write(v, tensor: Tensor<Float16>(from: result.rawValue))
            } else {
              precondition(shape.count == 2)
              let thisTensor = tensor[(i * count)..<((i + 1) * count), 0..<shape[1]].copied()
              let result = graph.variable(existingTensor) + graph.variable(thisTensor)
              store0.write(v, tensor: Tensor<Float16>(from: result.rawValue))
            }
            unhandledKeys.remove(v)
          }
        }
      } else if let v = UNetXLBaseFixed[extkey] {
        precondition(v.count == 1)
        let tensor = try! Tensor<Float>(numpy: tensor)
        let v = "__unet_fixed__[\(v[0])]"
        let existingTensor = Tensor<Float>(from: store.read(v)!).toCPU()
        let result =
          graph.variable(existingTensor)
          + graph.variable(tensor).reshaped(format: .NCHW, shape: existingTensor.shape)
        store0.write(v, tensor: Tensor<Float16>(from: result.rawValue))
        unhandledKeys.remove(v)
      } else {
        print("NO KEY FOUND! \(key)")
      }
    }
    // These are the keys on the decoder block.
    for key in unhandledKeys {
      guard let tensor = store.read(key) else { continue }
      store0.write(key, tensor: tensor)
    }
  }
}

print(head)
