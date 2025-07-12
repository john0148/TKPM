"""
OmniGen2 Model Wrapper for FastAPI Backend
"""

import os
import sys
import torch
from typing import List, Optional, Dict, Any
from PIL import Image, ImageOps
import logging
from torchvision.transforms.functional import to_pil_image, to_tensor

# Add OmniGen2 path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'OmniGen2-DFloat11'))

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model

logger = logging.getLogger(__name__)

class OmniGen2Model:
    """OmniGen2 model wrapper with DFloat11 optimization"""
    
    def __init__(self):
        """Initialize OmniGen2 model with DFloat11 compression"""
        try:
            logger.info("Initializing OmniGen2 with DFloat11...")
            
            # Check available devices
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for OmniGen2")
            
            # Always use cuda:1 for OmniGen2
            if torch.cuda.device_count() >= 2:
                target_device = 'cuda:1'
                logger.info("Using cuda:1 for OmniGen2")
            else:
                target_device = 'cuda:0'
                logger.warning("Only one CUDA device available, using cuda:0 for OmniGen2")
            
            self.target_device = target_device
            
            # Default configuration
            self.config = {
                "model_path": "OmniGen2/OmniGen2",
                "dtype": 'bf16',
                "enable_sequential_cpu_offload": False,
                "enable_model_cpu_offload": False,
                "enable_group_offload": False,
                "device": target_device
            }
            
            # Initialize accelerator
            self.accelerator = Accelerator(
                mixed_precision=self.config["dtype"] if self.config["dtype"] != 'fp32' else 'no'
            )
            
            # Set weight dtype
            if self.config["dtype"] == 'fp16':
                self.weight_dtype = torch.float16
            elif self.config["dtype"] == 'bf16':
                self.weight_dtype = torch.bfloat16
            else:
                self.weight_dtype = torch.float32
            
            # Load pipeline
            self.pipeline = self._load_pipeline()
            logger.info(f"OmniGen2 model loaded successfully on {target_device}!")
            
        except Exception as e:
            logger.error(f"Failed to initialize OmniGen2 model: {e}")
            raise e
    
    def _load_pipeline(self) -> OmniGen2Pipeline:
        """Load and configure OmniGen2 pipeline"""
        try:
            from transformers import CLIPProcessor
            
            # Load base pipeline
            pipeline = OmniGen2Pipeline.from_pretrained(
                self.config["model_path"],
                processor=CLIPProcessor.from_pretrained(
                    self.config["model_path"],
                    subfolder="processor",
                    use_fast=True
                ),
                torch_dtype=self.weight_dtype,
                trust_remote_code=True,
            )
            
            # Load DFloat11 compressed models
            DFloat11Model.from_pretrained(
                "DFloat11/OmniGen2-mllm-DF11",
                device="cpu",
                bfloat16_model=pipeline.mllm,
            )

            # Configure transformer
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

            # Configure offloading using target_device
            if self.config["enable_sequential_cpu_offload"]:
                pipeline.enable_sequential_cpu_offload()
            elif self.config["enable_model_cpu_offload"]:
                pipeline.enable_model_cpu_offload()
            elif self.config["enable_group_offload"]:
                apply_group_offloading(
                    pipeline.transformer, 
                    onload_device=self.target_device, 
                    offload_type="block_level", 
                    num_blocks_per_group=2, 
                    use_stream=True
                )
                apply_group_offloading(
                    pipeline.mllm, 
                    onload_device=self.target_device, 
                    offload_type="block_level", 
                    num_blocks_per_group=2, 
                    use_stream=True
                )
                apply_group_offloading(
                    pipeline.vae, 
                    onload_device=self.target_device, 
                    offload_type="block_level", 
                    num_blocks_per_group=2, 
                    use_stream=True
                )
            else:
                pipeline = pipeline.to(self.target_device)
                
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise e
    
    def _preprocess_images(self, input_images: List[Image.Image]) -> List[Image.Image]:
        """Preprocess input images"""
        if not input_images:
            return None
            
        processed_images = []
        for img in input_images:
            if isinstance(img, Image.Image):
                img = ImageOps.exif_transpose(img)
                processed_images.append(img)
            else:
                raise ValueError("All input images must be PIL Image objects")
                
        return processed_images
    
    def _create_collage(self, images: List[torch.Tensor]) -> Image.Image:
        """Create horizontal collage from list of images"""
        if len(images) == 1:
            return to_pil_image((images[0] * 0.5 + 0.5).clamp(0, 1))
            
        max_height = max(img.shape[-2] for img in images)
        total_width = sum(img.shape[-1] for img in images)
        canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
        
        current_x = 0
        for img in images:
            h, w = img.shape[-2:]
            canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
            current_x += w
        
        return to_pil_image(canvas.clamp(0, 1))
    
    def generate_in_context(
        self,
        instruction: str,
        input_images: List[Image.Image],
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 5.0,
        image_guidance_scale: float = 2.0,
        cfg_range_start: float = 0.0,
        cfg_range_end: float = 1.0,
        negative_prompt: str = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        num_images_per_prompt: int = 1,
        max_input_image_side_length: int = 1024,
        max_pixels: int = 1048576,
        seed: int = 0,
        scheduler: str = "euler"
    ) -> Dict[str, Any]:
        """
        Generate image with in-context generation (multiple objects composition)
        
        Args:
            instruction: Text instruction describing the desired composition
            input_images: List of input images to use as context
            width: Output image width
            height: Output image height
            num_inference_steps: Number of inference steps
            text_guidance_scale: Text guidance scale
            image_guidance_scale: Image guidance scale
            cfg_range_start: CFG range start
            cfg_range_end: CFG range end
            negative_prompt: Negative prompt
            num_images_per_prompt: Number of images per prompt
            max_input_image_side_length: Maximum input image side length
            max_pixels: Maximum pixels for input images
            seed: Random seed
            scheduler: Scheduler type ('euler' or 'dpmsolver')
            
        Returns:
            Dictionary with generated images and metadata
        """
        try:
            # Preprocess images
            processed_images = self._preprocess_images(input_images)
            
            # Configure scheduler
            if scheduler == 'dpmsolver':
                self.pipeline.scheduler = DPMSolverMultistepScheduler(
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint",
                    solver_order=2,
                    prediction_type="flow_prediction",
                )
            
            # Set generator
            generator = torch.Generator(device=self.target_device).manual_seed(seed)
            
            logger.info(f"Generating in-context image: '{instruction}'")
            
            # Generate
            results = self.pipeline(
                prompt=instruction,
                input_images=processed_images,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                max_sequence_length=1024,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                cfg_range=(cfg_range_start, cfg_range_end),
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                output_type="pil",
            )
            
            # Create collage if multiple images
            if len(results.images) > 1:
                vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
                collage = self._create_collage(vis_images)
                output_image = collage
            else:
                output_image = results.images[0]
            
            return {
                "image": output_image,
                "individual_images": results.images if len(results.images) > 1 else None,
                "instruction": instruction,
                "seed": seed,
                "num_input_images": len(processed_images) if processed_images else 0
            }
            
        except Exception as e:
            logger.error(f"Error during in-context generation: {e}")
            raise e
    
    def edit_image(
        self,
        instruction: str,
        input_image: Image.Image,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 5.0,
        image_guidance_scale: float = 2.0,
        cfg_range_start: float = 0.0,
        cfg_range_end: float = 1.0,
        negative_prompt: str = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        num_images_per_prompt: int = 1,
        max_input_image_side_length: int = 1024,
        max_pixels: int = 1048576,
        seed: int = 0,
        scheduler: str = "euler"
    ) -> Dict[str, Any]:
        """
        Edit image based on instruction
        
        Args:
            instruction: Text instruction describing the desired edit
            input_image: Input image to edit
            width: Output image width
            height: Output image height
            num_inference_steps: Number of inference steps
            text_guidance_scale: Text guidance scale
            image_guidance_scale: Image guidance scale
            cfg_range_start: CFG range start
            cfg_range_end: CFG range end
            negative_prompt: Negative prompt
            num_images_per_prompt: Number of images per prompt
            max_input_image_side_length: Maximum input image side length
            max_pixels: Maximum pixels for input images
            seed: Random seed
            scheduler: Scheduler type ('euler' or 'dpmsolver')
            
        Returns:
            Dictionary with edited image and metadata
        """
        try:
            # Preprocess single image
            processed_images = self._preprocess_images([input_image])
            
            # Configure scheduler
            if scheduler == 'dpmsolver':
                self.pipeline.scheduler = DPMSolverMultistepScheduler(
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint",
                    solver_order=2,
                    prediction_type="flow_prediction",
                )
            
            # Set generator
            generator = torch.Generator(device=self.target_device).manual_seed(seed)
            
            logger.info(f"Editing image: '{instruction}'")
            
            # Generate
            results = self.pipeline(
                prompt=instruction,
                input_images=processed_images,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                max_sequence_length=1024,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                cfg_range=(cfg_range_start, cfg_range_end),
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                output_type="pil",
            )
            
            output_image = results.images[0]
            
            return {
                "image": output_image,
                "instruction": instruction,
                "seed": seed,
                "original_image": input_image
            }
            
        except Exception as e:
            logger.error(f"Error during image editing: {e}")
            raise e
    
    def health_check(self) -> bool:
        """Check if model is healthy"""
        try:
            return self.pipeline is not None
        except:
            return False
    
    def get_device_info(self) -> str:
        """Get device information"""
        return self.target_device
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0
    ) -> Dict[str, Any]:
        """
        Generate image from text prompt (for training pipeline compatibility)
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            width: Output width
            height: Output height
            seed: Random seed
            
        Returns:
            Dictionary with generated image and metadata
        """
        try:
            # Use generate_in_context with no input images
            result = self.generate_in_context(
                instruction=prompt,
                input_images=[],  # No input images for text-to-image
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                text_guidance_scale=guidance_scale,
                image_guidance_scale=1.0,
                negative_prompt=negative_prompt,
                seed=seed
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            raise e 