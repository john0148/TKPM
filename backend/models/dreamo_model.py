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
            # Configure for nunchaku mode with best performance
            args = {
                'version': 'v1.1',  # Use latest version
                'offload': True,    # Enable offloading for memory efficiency
                'no_turbo': False,  # Use turbo for speed
                'quant': 'nunchaku', # Use nunchaku quantization
                'device': 'auto'    # Auto-detect device
            }
            
            logger.info("Initializing DreamO with nunchaku configuration...")
            self.generator = Generator(**args)
            logger.info("DreamO model loaded successfully!")
            
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
            
            # Convert PIL images to numpy arrays
            ref_images_np = []
            for img in ref_images:
                if isinstance(img, Image.Image):
                    ref_images_np.append(np.array(img))
                else:
                    ref_images_np.append(img)
            
            # Pad lists to match generator expectations (max 2 refs in original code)
            # We'll extend this to support more references
            while len(ref_images_np) < 10:  # Support up to 10 refs
                ref_images_np.append(None)
                ref_tasks.append(None)
            
            # Preprocessing
            ref_conds, debug_images, actual_seed = self.generator.pre_condition(
                ref_images=ref_images_np[:len(ref_images)],  # Only pass actual ref images
                ref_tasks=ref_tasks[:len(ref_images)],       # Only pass actual ref tasks
                ref_res=ref_res,
                seed=seed
            )
            
            logger.info(f"Generating image with prompt: '{prompt}', seed: {actual_seed}")
            
            # Generate image
            result = self.generator.dreamo_pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                ref_conds=ref_conds,
                generator=torch.Generator(device="cpu").manual_seed(actual_seed),
                true_cfg_scale=true_cfg,
                true_cfg_start_step=cfg_start_step,
                true_cfg_end_step=cfg_end_step,
                negative_prompt=neg_prompt,
                neg_guidance_scale=neg_guidance,
                first_step_guidance_scale=first_step_guidance if first_step_guidance > 0 else guidance,
            )
            
            generated_image = result.images[0]
            
            return {
                "image": generated_image,
                "debug_images": debug_images,
                "seed": actual_seed,
                "prompt": prompt,
                "ref_count": len([img for img in ref_images if img is not None])
            }
            
        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            raise e
    
    def health_check(self) -> bool:
        """Check if model is healthy"""
        try:
            return self.generator is not None
        except:
            return False 