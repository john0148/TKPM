import argparse
import os
from typing import List, Tuple

from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model

def load_pipeline(args: dict, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    from transformers import CLIPProcessor
    pipeline = OmniGen2Pipeline.from_pretrained(
        args["model_path"],
        processor=CLIPProcessor.from_pretrained(
            args["model_path"],
            subfolder="processor",
            use_fast=True
        ),
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    DFloat11Model.from_pretrained(
        "DFloat11/OmniGen2-mllm-DF11",
        device="cpu",
        bfloat16_model=pipeline.mllm,
    )

    cfg = {
        "patch_size": 2,
        "in_channels": 16,
        "out_channels": None,
        "hidden_size": 2520,
        "num_layers": 32,
        "num_refiner_layers": 2,
        "num_attention_heads": 21,
        "num_kv_heads": 7,
        "multiple_of": 256,
        "ffn_dim_multiplier": None,
        "norm_eps": 1e-5,
        "axes_dim_rope": (40, 40, 40),
        "axes_lens": (1024, 1664, 1664),
        "text_feat_dim": 2048,
        "timestep_scale": 1000.0,
    }
    with no_init_weights():
        pipeline.transformer = OmniGen2Transformer2DModel(**cfg).to(torch.bfloat16)

    DFloat11Model.from_pretrained(
        "DFloat11/OmniGen2-transformer-DF11",
        device="cpu",
        bfloat16_model=pipeline.transformer,
    )
    if args["scheduler"] == "dpmsolver":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if args["enable_sequential_cpu_offload"]:
        pipeline.enable_sequential_cpu_offload()
    elif args["enable_model_cpu_offload"]:
        pipeline.enable_model_cpu_offload()
    elif args["enable_group_offload"]:
        apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(accelerator.device)
    return pipeline

def preprocess(input_image_path: List[str] = []) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if input_image_path:
        input_images = []
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                          for f in os.listdir(input_image_path[0])]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images

def run(args: dict, 
        accelerator: Accelerator, 
        pipeline: OmniGen2Pipeline, 
        instruction: str, 
        negative_prompt: str, 
        input_images: List[Image.Image]) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(args["seed"])

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args["width"],
        height=args["height"],
        num_inference_steps=args["num_inference_step"],
        max_sequence_length=1024,
        text_guidance_scale=args["text_guidance_scale"],
        image_guidance_scale=args["image_guidance_scale"],
        cfg_range=(args["cfg_range_start"], args["cfg_range_end"]),
        negative_prompt=negative_prompt,
        num_images_per_prompt=args["num_images_per_prompt"],
        generator=generator,
        output_type="pil",
    )
    return results

def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)

args = {
    "model_path": "OmniGen2/OmniGen2",
    "num_inference_step": 50,
    "height": 1024,
    "width": 1024,
    "text_guidance_scale": 10.0,
    "image_guidance_scale": 2.0,
    "instruction": "Change the background to classroom in |image_1|.",
    "input_image_path": ['character/output_edit_2.png'],
    "output_image_path": "outputs/output_edit.png", # "outputs/output_in_context_generation.png"
    "num_images_per_prompt": 1,
    "scheduler": "euler",  # "euler", "dpmsolver"
    "seed": 0,
    "max_input_image_pixels": 1048576,  # Maximum number of pixels for each input image.
    "dtype": 'bf16',  # 'fp32', 'fp16', 'bf16'
    "cfg_range_start": 0.0,  # "Start of the CFG range."
    "cfg_range_end": 1.0,  # "End of the CFG range."
    "negative_prompt": "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
    "enable_model_cpu_offload": False,
    "enable_sequential_cpu_offload": False,
    "enable_group_offload": False
}

# Initialize accelerator
accelerator = Accelerator(mixed_precision=args["dtype"] if args["dtype"] != 'fp32' else 'no')

# Set weight dtype
weight_dtype = torch.float32
if args["dtype"] == 'fp16':
    weight_dtype = torch.float16
elif args["dtype"] == 'bf16':
    weight_dtype = torch.bfloat16

# Load pipeline and process inputs
pipeline = load_pipeline(args, accelerator, weight_dtype)

input_images = preprocess(args["input_image_path"])

# Generate and save image
results = run(args, accelerator, pipeline, args["instruction"], args["negative_prompt"], input_images)
os.makedirs(os.path.dirname(args["output_image_path"]), exist_ok=True)

if len(results.images) > 1:
    for i, image in enumerate(results.images):
        image_name, ext = os.path.splitext(args["output_image_path"])
        image.save(f"{image_name}_{i}{ext}")

vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
output_image = create_collage(vis_images)

output_image_path = args["output_image_path"]
output_image.save(output_image_path)
print(f"Image saved to {output_image_path}")