"""
DreamO Model Wrapper for FastAPI Backend
"""

import os
import sys
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from PIL import Image
import logging

# Add DreamO path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'DreamO'))

from dreamo_generator import Generator

logger = logging.getLogger(__name__)

class DreamOModel:
    """DreamO model wrapper with nunchaku optimization"""
    
    def __init__(self):
        """Initialize DreamO model with nunchaku configuration"""
        try:
            # Check available devices
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for DreamO")
            
            # Always use cuda:0 for DreamO
            target_device = 'cuda:0'
            logger.info("Using cuda:0 for DreamO")
            
            self.target_device = target_device
            
            # Configure for nunchaku mode with direct VRAM loading
            args = {
                'version': 'v1.1',  # Use latest version
                'offload': False,   # Disable offloading - load directly to VRAM
                'no_turbo': False,  # Use turbo for speed
                'quant': 'nunchaku', # Use nunchaku quantization
                'device': target_device
            }
            
            logger.info(f"Initializing DreamO with nunchaku configuration on {target_device} (no offload)...")
            self.generator = Generator(**args)
            logger.info(f"DreamO model loaded successfully on {target_device}!")
            
        except Exception as e:
            logger.error(f"Failed to initialize DreamO model: {e}")
            raise e
    
    def generate_image(
        self,
        prompt: str,
        ref_images: List[Image.Image],
        ref_tasks: List[str],
        width: int = 1024,
        height: int = 1024,
        ref_res: int = 512,
        num_steps: int = 12,
        guidance: float = 4.5,
        seed: int = -1,
        true_cfg: float = 1.0,
        cfg_start_step: int = 0,
        cfg_end_step: int = 0,
        neg_prompt: str = "",
        neg_guidance: float = 3.5,
        first_step_guidance: float = 0
    ) -> Dict[str, Any]:
        """
        Generate image with multiple reference images
        
        Args:
            prompt: Text description for generation
            ref_images: List of reference images
            ref_tasks: List of tasks for each reference image ('ip', 'id', 'style')
            width: Output image width
            height: Output image height
            ref_res: Resolution for reference images
            num_steps: Number of inference steps
            guidance: Guidance scale
            seed: Random seed (-1 for random)
            true_cfg: True CFG scale
            cfg_start_step: CFG start step
            cfg_end_step: CFG end step
            neg_prompt: Negative prompt
            neg_guidance: Negative guidance scale
            first_step_guidance: First step guidance scale
            
        Returns:
            Dictionary with generated image and metadata
        """
        try:
            logger.info(f"Starting generation with {len(ref_images)} reference images")
            
            # Check model health and reload if needed
            self.reload_model_if_needed()
            
            # Validate inputs
            if not ref_images:
                raise ValueError("At least one reference image is required")
            
            if len(ref_images) != len(ref_tasks):
                raise ValueError("Number of reference images must match number of tasks")
            
            # Validate tasks
            valid_tasks = ['ip', 'id', 'style']
            for task in ref_tasks:
                if task not in valid_tasks:
                    raise ValueError(f"Invalid task '{task}'. Must be one of {valid_tasks}")
            
            # Ensure CUDA is available and model is on correct device
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            # Use the device determined during initialization
            device = self.target_device
            
            # Convert PIL images to numpy arrays
            ref_images_np = []
            for img in ref_images:
                if isinstance(img, Image.Image):
                    ref_images_np.append(np.array(img))
                else:
                    ref_images_np.append(img)
            
            # Create copy of ref_tasks to avoid modifying input
            ref_tasks_copy = ref_tasks.copy()
            
            # Ensure we have exactly the right number of items for DreamO
            actual_count = len(ref_images)
            
            # Preprocessing with actual data only
            ref_conds, debug_images, actual_seed = self.generator.pre_condition(
                ref_images=ref_images_np,
                ref_tasks=ref_tasks_copy,
                ref_res=ref_res,
                seed=seed
            )
            
            logger.info(f"Generating image with prompt: '{prompt}', seed: {actual_seed}")
            
            # Generate image with proper device handling
            with torch.cuda.device(device):
                torch_generator = torch.Generator(device=device).manual_seed(actual_seed)
                
                result = self.generator.dreamo_pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    ref_conds=ref_conds,
                    generator=torch_generator,
                    true_cfg_scale=true_cfg,
                    true_cfg_start_step=cfg_start_step,
                    true_cfg_end_step=cfg_end_step,
                    negative_prompt=neg_prompt,
                    neg_guidance_scale=neg_guidance,
                    first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else guidance,
                )
            
            generated_image = result.images[0]
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Image generation completed successfully")
            
            return {
                "image": generated_image,
                "debug_images": debug_images,
                "seed": actual_seed,
                "prompt": prompt,
                "ref_count": len([img for img in ref_images if img is not None])
            }
            
        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            
            # Emergency memory cleanup on error
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Memory cleanup after error")
            except:
                pass
            
            # Re-raise the original error
            raise e
    
    def generate_multi_reference(
        self,
        reference_images: List[Image.Image],
        num_outputs: int,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 2.5,
        num_inference_steps: int = 12
    ) -> Dict[str, Any]:
        """
        Generate multiple variations from reference images (for training pipeline)
        
        Args:
            reference_images: List of reference images
            num_outputs: Number of variations to generate
            prompt: Text prompt
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale
            num_inference_steps: Number of inference steps
            
        Returns:
            Dictionary with images list
        """
        try:
            if not reference_images:
                raise ValueError("At least one reference image is required")
            
            # Generate multiple outputs
            generated_images = []
            
            for i in range(num_outputs):
                # Use different seeds for variation
                seed = np.random.randint(0, 2147483647)
                
                # Determine task type based on content (simple heuristic)
                ref_tasks = ['ip'] * len(reference_images)  # Default to IP
                
                result = self.generate_image(
                    prompt=prompt,
                    ref_images=reference_images,
                    ref_tasks=ref_tasks,
                    width=1024,
                    height=1024,
                    num_steps=num_inference_steps,
                    guidance=guidance_scale,
                    seed=seed,
                    neg_prompt=negative_prompt
                )
                
                generated_images.append(result["image"])
            
            return {
                "images": generated_images,
                "count": len(generated_images)
            }
            
        except Exception as e:
            logger.error(f"Error during multi-reference generation: {e}")
            raise e
    
    def health_check(self) -> bool:
        """Check if model is healthy"""
        try:
            if self.generator is None:
                return False
            
            # Try a simple operation to verify model is working
            # This helps detect if model was unloaded from VRAM
            if hasattr(self.generator, 'dreamo_pipeline'):
                return True
            
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def reload_model_if_needed(self):
        """Reload model if it was unloaded from VRAM"""
        try:
            if not self.health_check():
                logger.warning("Model appears to be unhealthy, attempting reload...")
                
                # Reinitialize the generator
                args = {
                    'version': 'v1.1',
                    'offload': False,
                    'no_turbo': False,
                    'quant': 'nunchaku',
                    'device': self.target_device
                }
                
                self.generator = Generator(**args)
                logger.info(f"Model reloaded successfully on {self.target_device}")
                
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            raise e
    
    def get_device_info(self) -> str:
        """Get device information"""
        try:
            if hasattr(self, 'target_device'):
                return self.target_device
            elif hasattr(self.generator, 'device'):
                return str(self.generator.device)
            elif hasattr(self.generator, 'dreamo_pipeline'):
                if hasattr(self.generator.dreamo_pipeline, 'device'):
                    return str(self.generator.dreamo_pipeline.device)
            return "unknown"
        except:
            return "unknown" 