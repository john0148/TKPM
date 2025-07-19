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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'OmniGen2'))

from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

logger = logging.getLogger(__name__)

class OmniGen2Model:
    """OmniGen2 model wrapper"""
    
    # --- THAY ĐỔI: Thêm target_device vào hàm khởi tạo ---
    def __init__(self, target_device: str = 'cuda:0'):
        """Initialize OmniGen2 model on a specific device."""
        try:
            logger.info(f"Initializing OmniGen2 with DFloat11 on device: {target_device}")
            
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for OmniGen2")
            
            # --- THAY ĐỔI: Sử dụng trực tiếp target_device được truyền vào ---
            self.target_device = target_device
            logger.info(f"Forcing OmniGen2 to use {self.target_device}")

            # Default configuration
            self.config = {
                "model_path": "OmniGen2/OmniGen2",
                "transformer_path": None,
                "transformer_lora_path": None,
                "num_inference_step": 50,
                "height": 1024,
                "width": 1024,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 3.0,
                "dtype": 'bf16',
                "num_images_per_prompt": 1,
                "scheduler": "euler",  # "euler", "dpmsolver"
                "seed": 0,
                "max_input_image_pixels": 1048576,
                "dtype": 'bf16',
                "cfg_range_start": 0.0,
                "cfg_range_end": 1.0,
                "negative_prompt": "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
                "enable_model_cpu_offload": False,
                "enable_sequential_cpu_offload": False,
                "enable_group_offload": False,
                "enable_teacache": False,
                "teacache_rel_l1_thresh": 0.05,
                "enable_taylorseer": False,
                "device": self.target_device # Sử dụng device đã được gán
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
            logger.info(f"OmniGen2 model loaded successfully on {self.target_device}!")
            
        except Exception as e:
            logger.error(f"Failed to initialize OmniGen2 model: {e}")
            raise e
    
    def _load_pipeline(self) -> OmniGen2Pipeline:
        """Load and configure OmniGen2 pipeline (chuẩn repo gốc)"""
        try:
            pipeline = OmniGen2Pipeline.from_pretrained(
                self.config["model_path"],
                torch_dtype=self.weight_dtype,
                trust_remote_code=True,
            )

            if self.config["transformer_path"]:
                print(f"Transformer weights loaded from {self.config['transformer_path']}")
                pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                    self.config["transformer_path"],
                    torch_dtype=self.weight_dtype,
                )
            else:
                pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                    self.config["model_path"],
                    subfolder="transformer",
                    torch_dtype=self.weight_dtype,
                )

            if self.config["transformer_lora_path"]:
                print(f"LoRA weights loaded from {self.config['transformer_lora_path']}")
                pipeline.load_lora_weights(self.config["transformer_lora_path"])

            if self.config["enable_teacache"] and  self.config["enable_taylorseer"]:
                print("WARNING: enable_teacache and enable_taylorseer are mutually exclusive. enable_teacache will be ignored.")

            if self.config["enable_taylorseer"]:
                pipeline.enable_taylorseer = True
            elif self.config["enable_teacache"]:
                pipeline.transformer.enable_teacache = True
                pipeline.transformer.teacache_rel_l1_thresh = self.config["teacache_rel_l1_thresh"]

            if self.config["scheduler"] == "dpmsolver++":
                from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
                scheduler = DPMSolverMultistepScheduler(
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint",
                    solver_order=2,
                    prediction_type="flow_prediction",
                )
                pipeline.scheduler = scheduler

            if self.config["enable_sequential_cpu_offload"]:
                pipeline.enable_sequential_cpu_offload()
            elif self.config["enable_model_cpu_offload"]:
                pipeline.enable_model_cpu_offload()
            elif self.config["enable_group_offload"]:
                apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
                apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
                apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
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
        negative_prompt: str = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1
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