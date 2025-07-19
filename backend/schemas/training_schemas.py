from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import UploadFile

class TrainingRequest(BaseModel):
    """Request schema cho training pipeline"""
    name_object: str = Field(..., description="Tên trigger word cho object cần train")
    description: Optional[str] = Field(None, description="Mô tả chi tiết về object")
    
    model_config = {"protected_namespaces": ()}
    
class TrainingResponse(BaseModel):
    """Response schema cho training pipeline"""
    success: bool
    message: str
    training_id: str
    model_path: Optional[str] = None
    generated_image_path: Optional[str] = None
    variations_count: Optional[int] = None
    
    model_config = {"protected_namespaces": ()}

class TrainingStatus(BaseModel):
    """Schema cho status của training process"""
    training_id: str
    status: str  # "generating_variations", "preparing_dataset", "training", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    generated_image_path: Optional[str] = None
    
    model_config = {"protected_namespaces": ()}

class InferenceRequest(BaseModel):
    """Request schema cho inference với trained model"""
    model_id: str = Field(..., description="ID của model đã train")
    prompt: str = Field(..., description="Prompt để generate image")
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 28
    guidance_scale: float = 4.0
    
    model_config = {"protected_namespaces": ()} 