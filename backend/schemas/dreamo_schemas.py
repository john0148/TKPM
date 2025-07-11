"""
Pydantic schemas for DreamO API
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class TaskType(str, Enum):
    """Valid task types for DreamO"""
    IP = "ip"
    ID = "id" 
    STYLE = "style"

class ReferenceImage(BaseModel):
    """Reference image configuration"""
    image_data: str = Field(..., description="Base64 encoded image data")
    task: TaskType = Field(..., description="Task type for this reference image")
    
    class Config:
        schema_extra = {
            "example": {
                "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
                "task": "ip"
            }
        }

class DreamOGenerateRequest(BaseModel):
    """Request schema for DreamO image generation"""
    prompt: str = Field(..., description="Text description for generation", min_length=1, max_length=1000)
    ref_images: List[ReferenceImage] = Field(..., description="List of reference images with their tasks", min_items=1, max_items=10)
    width: int = Field(1024, description="Output image width", ge=256, le=2048)
    height: int = Field(1024, description="Output image height", ge=256, le=2048)
    ref_res: int = Field(512, description="Resolution for reference images", ge=256, le=1024)
    num_steps: int = Field(12, description="Number of inference steps", ge=1, le=50)
    guidance: float = Field(4.5, description="Guidance scale", ge=1.0, le=20.0)
    seed: int = Field(-1, description="Random seed (-1 for random)", ge=-1)
    true_cfg: float = Field(1.0, description="True CFG scale", ge=0.0, le=10.0)
    cfg_start_step: int = Field(0, description="CFG start step", ge=0)
    cfg_end_step: int = Field(0, description="CFG end step", ge=0)
    neg_prompt: str = Field("", description="Negative prompt", max_length=500)
    neg_guidance: float = Field(3.5, description="Negative guidance scale", ge=1.0, le=20.0)
    first_step_guidance: float = Field(0, description="First step guidance scale", ge=0.0, le=20.0)
    
    @validator('cfg_end_step')
    def validate_cfg_steps(cls, v, values):
        if 'cfg_start_step' in values and v > 0 and v <= values['cfg_start_step']:
            raise ValueError('cfg_end_step must be greater than cfg_start_step when both are non-zero')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "a person playing guitar in the street",
                "ref_images": [
                    {
                        "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
                        "task": "ip"
                    }
                ],
                "width": 1024,
                "height": 1024,
                "num_steps": 12,
                "guidance": 4.5,
                "seed": -1
            }
        }

class DreamOGenerateResponse(BaseModel):
    """Response schema for DreamO image generation"""
    success: bool = Field(..., description="Whether generation was successful")
    image: Optional[str] = Field(None, description="Base64 encoded generated image")
    debug_images: Optional[List[str]] = Field(None, description="Base64 encoded debug images showing preprocessing")
    seed: Optional[int] = Field(None, description="Actual seed used for generation")
    prompt: Optional[str] = Field(None, description="Prompt used for generation")
    ref_count: Optional[int] = Field(None, description="Number of reference images used")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    generation_time: Optional[float] = Field(None, description="Time taken for generation in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                "debug_images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
                "seed": 12345,
                "prompt": "a person playing guitar in the street",
                "ref_count": 1,
                "generation_time": 15.2
            }
        }

class DreamOHealthResponse(BaseModel):
    """Response schema for DreamO health check"""
    healthy: bool = Field(..., description="Whether the model is healthy")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="DreamO version")
    device: str = Field(..., description="Device being used")
    
    class Config:
        schema_extra = {
            "example": {
                "healthy": True,
                "model_loaded": True,
                "version": "v1.1",
                "device": "cuda"
            }
        } 