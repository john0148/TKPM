from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List, Optional
import asyncio
import logging
from PIL import Image
import io

from schemas.training_schemas import (
    TrainingRequest, TrainingResponse, TrainingStatus, InferenceRequest
)
from models.omnigen2_model import OmniGen2Model
from utils.training_utils import training_manager
from utils.image_utils import process_uploaded_image

router = APIRouter(prefix="/training", tags=["training"])
logger = logging.getLogger(__name__)

# Import global model instances từ main module
def get_global_models():
    """Lấy global model instances từ main module"""
    import main
    return main.dreamo_model, main.omnigen2_model

@router.post("/start", response_model=TrainingResponse)
async def start_training_pipeline(
    background_tasks: BackgroundTasks,
    name_object: str = Form(..., description="Tên trigger word cho object"),
    description: Optional[str] = Form(None, description="Mô tả chi tiết về object"),
    reference_images: List[UploadFile] = File(..., description="Danh sách reference images")
):
    """
    Bắt đầu training pipeline:
    1. Generate 15 variations từ reference images bằng DreamO
    2. Chuẩn bị dataset
    3. Train LoRA model với OmniGen2
    4. Generate test image
    """
    try:
        # Validate inputs
        if len(reference_images) < 1 or len(reference_images) > 5:
            raise HTTPException(
                status_code=400, 
                detail="Cần từ 1-5 reference images"
            )
        
        # Validate name_object
        if not name_object or len(name_object.strip()) < 2:
            raise HTTPException(
                status_code=400,
                detail="name_object phải có ít nhất 2 ký tự"
            )
        
        name_object = name_object.strip().lower()
        
        # Process uploaded images
        processed_images = []
        for uploaded_file in reference_images:
            try:
                image = process_uploaded_image(uploaded_file)
                processed_images.append(image)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Lỗi xử lý ảnh {uploaded_file.filename}: {str(e)}"
                )
        
        # Create training session
        training_id = training_manager.create_training_session(name_object, description)
        
        logger.info(f"Started training pipeline for '{name_object}' with {len(processed_images)} reference images")
        
        # Start background task
        background_tasks.add_task(
            run_training_pipeline,
            training_id,
            name_object,
            processed_images
        )
        
        return TrainingResponse(
            success=True,
            message="Training pipeline started successfully",
            training_id=training_id,
            variations_count=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def run_training_pipeline(
    training_id: str,
    name_object: str,
    reference_images: List[Image.Image]
):
    """Background task chạy toàn bộ training pipeline"""
    try:
        # Get global models
        dreamo_model, omnigen2_model = get_global_models()
        
        # Step 1: Generate variations với DreamO
        training_manager.update_status(
            training_id, "generating_variations", 0.1, 
            f"Generating 15 variations từ {len(reference_images)} reference images..."
        )
        
        # Generate multiple variations từ mỗi reference image
        all_variations = []
        variations_per_ref = max(1, 15 // len(reference_images))
        
        for i, ref_image in enumerate(reference_images):
            training_manager.update_status(
                training_id, "generating_variations", 
                0.1 + (0.3 * i / len(reference_images)),
                f"Generating variations từ reference image {i+1}/{len(reference_images)}..."
            )
            
            # Generate variations với DreamO nunchaku mode
            variations = dreamo_model.generate_multi_reference(
                reference_images=[ref_image],
                num_outputs=variations_per_ref,
                prompt=f"high quality photo of {name_object}",
                negative_prompt="blurry, low quality, distorted",
                guidance_scale=2.5,
                num_inference_steps=4
            )
            
            all_variations.extend(variations["images"])
        
        # Đảm bảo có đúng 15 variations
        while len(all_variations) < 15 and len(all_variations) > 0:
            all_variations.extend(all_variations[:15 - len(all_variations)])
        all_variations = all_variations[:15]
        
        # Step 2: Save variations
        training_manager.update_status(
            training_id, "preparing_dataset", 0.4,
            f"Saving {len(all_variations)} variations and preparing dataset..."
        )
        
        variations_count = training_manager.save_variations(training_id, all_variations)
        training_manager.create_dataset_config(training_id, name_object)
        
        # Step 3: Train LoRA model
        training_manager.update_status(
            training_id, "training", 0.5,
            "Starting OmniGen2 LoRA training..."
        )
        
        training_success = await training_manager.run_training(training_id, name_object)
        
        if not training_success:
            training_manager.update_status(
                training_id, "failed", 0.0,
                "Training failed. Check logs for details."
            )
            return
        
        # Step 4: Generate test image
        test_image_path = await training_manager.generate_test_image(training_id, name_object, omnigen2_model)
        
        # Final status update
        training_manager.update_status(
            training_id, "completed", 1.0,
            f"Training completed successfully! Generated {variations_count} variations and trained LoRA model."
        )
        
        # Update final response
        status = training_manager.get_status(training_id)
        status["generated_image_path"] = test_image_path
        status["variations_count"] = variations_count
        
    except Exception as e:
        logger.error(f"Training pipeline error for {training_id}: {str(e)}")
        training_manager.update_status(
            training_id, "failed", 0.0,
            f"Pipeline failed: {str(e)}"
        )

@router.get("/status/{training_id}", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """Lấy status của training process"""
    status = training_manager.get_status(training_id)
    
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return TrainingStatus(
        training_id=training_id,
        status=status["status"],
        progress=status["progress"],
        message=status["message"],
        generated_image_path=status.get("generated_image_path")
    )

@router.post("/inference/{training_id}")
async def inference_with_trained_model(
    training_id: str,
    request: InferenceRequest
):
    """Generate image với trained model"""
    try:
        # Check if model exists
        session_dir = training_manager.base_dir / training_id
        model_path = session_dir / "model"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Trained model not found. Make sure training is completed."
            )
        
        # Get global model and sử dụng nó cho inference với LoRA weights
        _, omnigen2_model = get_global_models()
        omnigen2_model.pipeline.load_lora_weights(str(model_path))
        
        # Generate image
        result = omnigen2_model.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=1024,
            height=1024
        )
        
        # Save result
        output_dir = session_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        import time
        timestamp = int(time.time())
        output_path = output_dir / f"inference_{timestamp}.png"
        result["image"].save(output_path)
        
        return {
            "success": True,
            "message": "Image generated successfully",
            "image_path": str(output_path),
            "training_id": training_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.get("/list")
async def list_training_sessions():
    """Liệt kê tất cả training sessions"""
    sessions = []
    
    if training_manager.base_dir.exists():
        for session_dir in training_manager.base_dir.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        status = training_manager.get_status(metadata["training_id"])
                        
                        sessions.append({
                            "training_id": metadata["training_id"],
                            "name_object": metadata["name_object"],
                            "description": metadata.get("description"),
                            "created_at": metadata["created_at"],
                            "status": status["status"],
                            "progress": status["progress"]
                        })
                    except Exception as e:
                        logger.warning(f"Error reading session {session_dir.name}: {str(e)}")
    
    return {"sessions": sessions}

@router.delete("/delete/{training_id}")
async def delete_training_session(training_id: str):
    """Xóa training session và tất cả files liên quan"""
    try:
        session_dir = training_manager.base_dir / training_id
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Training session not found")
        
        # Remove directory và all contents
        import shutil
        shutil.rmtree(session_dir)
        
        # Remove from memory
        if training_id in training_manager.training_status:
            del training_manager.training_status[training_id]
        
        return {
            "success": True,
            "message": f"Training session {training_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting training session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}") 