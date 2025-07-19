"""
Pydantic schemas for OmniGen2 API
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class SchedulerType(str, Enum):
    """Valid scheduler types for OmniGen2"""
    EULER = "euler"
    DPMSOLVER = "dpmsolver"

class OmniGen2InContextRequest(BaseModel):
    """Request schema for OmniGen2 in-context generation"""
    instruction: str = Field(..., description="Text instruction describing the desired composition", min_length=1, max_length=1000)
    input_images: List[str] = Field(..., description="List of base64 encoded input images", min_items=1, max_items=5)
    width: int = Field(1024, description="Output image width", ge=256, le=2048)
    height: int = Field(1024, description="Output image height", ge=256, le=2048)
    num_inference_steps: int = Field(50, description="Number of inference steps", ge=10, le=100)
    text_guidance_scale: float = Field(5.0, description="Text guidance scale", ge=1.0, le=20.0)
    image_guidance_scale: float = Field(2.0, description="Image guidance scale", ge=1.0, le=10.0)
    cfg_range_start: float = Field(0.0, description="CFG range start", ge=0.0, le=1.0)
    cfg_range_end: float = Field(1.0, description="CFG range end", ge=0.0, le=1.0)
    negative_prompt: str = Field("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar", description="Negative prompt", max_length=1000)
    num_images_per_prompt: int = Field(1, description="Number of images per prompt", ge=1, le=4)
    max_input_image_side_length: int = Field(1024, description="Maximum input image side length", ge=256, le=2048)
    max_pixels: int = Field(1048576, description="Maximum pixels for input images", ge=65536, le=4194304)
    seed: int = Field(0, description="Random seed", ge=0)
    scheduler: SchedulerType = Field(SchedulerType.EULER, description="Scheduler type")
    
    @validator('cfg_range_end')
    def validate_cfg_range(cls, v, values):
        if 'cfg_range_start' in values and v < values['cfg_range_start']:
            raise ValueError('cfg_range_end must be greater than or equal to cfg_range_start')
        return v
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "instruction": "Please let the person in image 2 hold the toy from the first image in a parking lot.",
                "input_images": [
                    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
                    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
                ],
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0,
                "seed": 0
            }
        }
    }

class OmniGen2EditRequest(BaseModel):
    """Request schema for OmniGen2 image editing"""
    instruction: str = Field(..., description="Text instruction describing the desired edit", min_length=1, max_length=1000)
    input_image: str = Field(..., description="Base64 encoded input image to edit")
    width: int = Field(1024, description="Output image width", ge=256, le=2048)
    height: int = Field(1024, description="Output image height", ge=256, le=2048)
    num_inference_steps: int = Field(50, description="Number of inference steps", ge=10, le=100)
    text_guidance_scale: float = Field(5.0, description="Text guidance scale", ge=1.0, le=20.0)
    image_guidance_scale: float = Field(2.0, description="Image guidance scale", ge=1.0, le=10.0)
    cfg_range_start: float = Field(0.0, description="CFG range start", ge=0.0, le=1.0)
    cfg_range_end: float = Field(1.0, description="CFG range end", ge=0.0, le=1.0)
    negative_prompt: str = Field("(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar", description="Negative prompt", max_length=1000)
    num_images_per_prompt: int = Field(1, description="Number of images per prompt", ge=1, le=4)
    max_input_image_side_length: int = Field(1024, description="Maximum input image side length", ge=256, le=2048)
    max_pixels: int = Field(1048576, description="Maximum pixels for input images", ge=65536, le=4194304)
    seed: int = Field(0, description="Random seed", ge=0)
    scheduler: SchedulerType = Field(SchedulerType.EULER, description="Scheduler type")
    
    @validator('cfg_range_end')
    def validate_cfg_range(cls, v, values):
        if 'cfg_range_start' in values and v < values['cfg_range_start']:
            raise ValueError('cfg_range_end must be greater than or equal to cfg_range_start')
        return v
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "instruction": "Change the background to classroom",
                "input_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0,
                "seed": 0
            }
        }
    }

class OmniGen2Response(BaseModel):
    """Response schema for OmniGen2 operations"""
    success: bool = Field(..., description="Whether operation was successful")
    image: Optional[str] = Field(None, description="Base64 encoded generated/edited image")
    individual_images: Optional[List[str]] = Field(None, description="Base64 encoded individual images if multiple were generated")
    instruction: Optional[str] = Field(None, description="Instruction used")
    seed: Optional[int] = Field(None, description="Seed used for generation")
    num_input_images: Optional[int] = Field(None, description="Number of input images used")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    generation_time: Optional[float] = Field(None, description="Time taken for generation in seconds")
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "success": True,
                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                "instruction": "Change the background to classroom",
                "seed": 0,
                "num_input_images": 1,
                "generation_time": 25.8
            }
        }
    }

class OmniGen2HealthResponse(BaseModel):
    """Response schema for OmniGen2 health check"""
    healthy: bool = Field(..., description="Whether the model is healthy")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Device being used")
    memory_usage: Optional[str] = Field(None, description="Current memory usage")
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "healthy": True,
                "model_loaded": True,
                "device": "cuda",
                "memory_usage": "14.3GB / 24GB"
            }
        }
    } 