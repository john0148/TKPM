from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List, Optional
import asyncio
import logging
from PIL import Image
import gc
# --- THAY ĐỔI: Import app_state, không import main ---
import app_state
import torch
import json
import numpy as np
# --- THAY ĐỔI: Không cần multiprocessing nữa ---
# import multiprocessing as mp
import time

from schemas.training_schemas import TrainingResponse, TrainingStatus, InferenceRequest
from models.omnigen2_model import OmniGen2Model
# DreamOModel chỉ để type hint, không cần thiết
# from models.dreamo_model import DreamOModel
from utils.training_utils import training_manager
from utils.image_utils import process_uploaded_image

router = APIRouter(prefix="/training", tags=["training"])
logger = logging.getLogger(__name__)

def dreamo_generation_task(reference_images, name_object, subject_type, output_queue):
    """Hàm này sẽ chạy trong một process riêng cho DreamO để đảm bảo giải phóng VRAM."""
    try:
        # Tải model DreamO BÊN TRONG process này
        from models.dreamo_model import DreamOModel
        dreamo_model = DreamOModel()
        logger.info("DreamO model loaded in a separate process for variation generation.")
        
        diverse_prompts = create_diverse_prompts(name_object, 15, subject_type)
        all_variations = []
        variations_per_ref = max(1, 15 // len(reference_images))
        
        for i, ref_image in enumerate(reference_images):
            start_idx = i * variations_per_ref
            end_idx = min(start_idx + variations_per_ref, len(diverse_prompts))
            for prompt in diverse_prompts[start_idx:end_idx]:
                result = dreamo_model.generate_image(
                    prompt=prompt, ref_images=[ref_image], ref_tasks=['ip'],
                    width=1024, height=1024, num_steps=12, guidance=4.5, seed=-1,
                    neg_prompt="(((deformed))), blurry, over saturation, bad anatomy"
                )
                all_variations.append(result["image"])
        
        while len(all_variations) < 15 and len(all_variations) > 0:
            all_variations.extend(all_variations[:15 - len(all_variations)])
        
        output_queue.put(all_variations[:15])
        
        logger.info("DreamO process finished. VRAM will be released automatically upon process exit.")
        
    except Exception as e:
        logger.error(f"Error in DreamO generation task: {e}", exc_info=True)
        output_queue.put([])

def create_diverse_prompts(name_object: str, num_images: int, subject_type: str = 'object'):
    """Create diverse prompts for the training dataset"""
    backgrounds = [
        "in a park", "at the beach", "in a classroom", "in a city street", 
        "in a forest", "in a kitchen", "in a futuristic city", "in a library",
        "in a garden", "in a snowy landscape", "in a coffee shop", "in a studio", 
        "in a mountain landscape", "in a night market", "in a science lab"
    ]
    
    while len(backgrounds) < num_images:
        backgrounds = backgrounds * 2
    backgrounds = backgrounds[:num_images]
    
    prompts = []
    for background in backgrounds:
        prompt = f"A photo of {name_object} {background}"
        prompts.append(prompt)
    
    return prompts

def force_gpu_cleanup():
    """Force cleanup of all GPU memory"""
    logger.info("Forcing GPU memory cleanup...")
    
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Reset memory stats
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.reset_accumulated_memory_stats(i)
                
                # Force memory release
                torch.cuda.memory.empty_cache()
                
        logger.info("GPU cache cleared and memory stats reset.")
        
        # Log memory usage after cleanup
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            # Force additional cleanup if still too much memory
            if memory_allocated > 0.5:  # More than 500MB
                logger.warning(f"GPU {i} still has {memory_allocated:.2f}GB allocated. Forcing additional cleanup.")
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Try to allocate and free a small tensor to force memory defragmentation
                    try:
                        temp_tensor = torch.zeros(1, device=f'cuda:{i}')
                        del temp_tensor
                        torch.cuda.empty_cache()
                    except:
                        pass

@router.post("/start", response_model=TrainingResponse)
async def start_training_pipeline(
    background_tasks: BackgroundTasks,
    name_object: str = Form(..., description="Trigger word for the object"),
    description: Optional[str] = Form(None, description="Detailed description of the object"),
    reference_images: List[UploadFile] = File(..., description="List of reference images"),
    subject_type: str = Form("object", description="Subject type: 'object' or 'background'")
):
    """
    Starts the training pipeline:
    1. Generate 15 variations from reference images using the pre-loaded DreamO model.
    2. Unload all models from VRAM.
    3. Prepare the dataset.
    4. Train a LoRA model with OmniGen2 on 2 GPUs.
    5. Generate a test image.
    """
    try:
        if len(reference_images) < 1 or len(reference_images) > 5:
            raise HTTPException(status_code=400, detail="Requires 1-5 reference images")
        
        name_object = name_object.strip().lower()
        
        processed_images = [process_uploaded_image(file) for file in reference_images]
        
        training_id = training_manager.create_training_session(name_object, description)
        
        logger.info(f"Started training pipeline for '{name_object}' with {len(processed_images)} reference images")
        
        background_tasks.add_task(
            run_training_pipeline,
            training_id,
            name_object,
            processed_images,
            subject_type
        )
        
        return TrainingResponse(
            success=True,
            message="Training pipeline started successfully in the background.",
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
    reference_images: List[Image.Image],
    subject_type: str
):
    """
    Runs the entire training pipeline:
    1. Use pre-loaded DreamO.
    2. Unload ALL models.
    3. Train.
    4. Reload OmniGen2 only.
    """
    try:
        # --- Giai đoạn 1: Sử dụng DreamO có sẵn trong VRAM ---
        training_manager.update_status(
            training_id, "generating_variations", 0.1, 
            "Generating variations using pre-loaded DreamO model..."
        )
        
        dreamo_model = app_state.dreamo_model
        if dreamo_model is None:
            raise Exception("DreamO model is not loaded. Cannot start training.")

        diverse_prompts = create_diverse_prompts(name_object, 15, subject_type)
        all_variations = []
        variations_per_ref = max(1, 15 // len(reference_images))

        for i, ref_image in enumerate(reference_images):
            start_idx = i * variations_per_ref
            end_idx = min(start_idx + variations_per_ref, len(diverse_prompts))
            for prompt in diverse_prompts[start_idx:end_idx]:
                result = dreamo_model.generate_image(
                    prompt=prompt, ref_images=[ref_image], ref_tasks=['ip'],
                    width=1024, height=1024, num_steps=12, guidance=4.5, seed=-1
                )
                all_variations.append(result["image"])
        
        while len(all_variations) < 15 and len(all_variations) > 0:
            all_variations.extend(all_variations[:15 - len(all_variations)])
        
        if not all_variations:
            raise Exception("DreamO variation generation failed.")
        
        logger.info(f"Successfully generated {len(all_variations)} variations.")

        # --- Giai đoạn 2: Unload TOÀN BỘ models khỏi VRAM ---
        training_manager.update_status(training_id, "unloading_models", 0.3, "Unloading all models to free up VRAM...")
        
        # Xóa các đối tượng model
        del dreamo_model
        if app_state.omnigen2_model:
            del app_state.omnigen2_model
        
        # Set global references to None
        app_state.dreamo_model = None
        app_state.omnigen2_model = None
        
        force_gpu_cleanup()
        logger.info("All models unloaded and VRAM is cleared for training.")
        
        # Wait a bit more and check memory again
        time.sleep(10)
        
        # Verify memory is free
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"Before training - GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                # Warn if too much memory is still used
                if memory_allocated > 1.0:  # More than 1GB
                    logger.warning(f"GPU {i} still has {memory_allocated:.2f}GB allocated. Training may fail.")
        
        # --- Giai đoạn 3 & 4: Chuẩn bị dataset và Train ---
        training_manager.update_status(training_id, "preparing_dataset", 0.4, "Preparing dataset...")
        training_manager.save_variations(training_id, all_variations)
        training_manager.create_dataset_config(training_id, name_object, subject_type)
        
        training_success = await training_manager.run_training(training_id, name_object)
        if not training_success:
            raise Exception("OmniGen2 training script failed.")
            
        # --- Giai đoạn 5: Tải lại CHỈ OmniGen2 ---
        training_manager.update_status(training_id, "reloading_model", 0.9, "Reloading OmniGen2 model and new LoRA weights...")
        
        app_state.omnigen2_model = OmniGen2Model(target_device='cuda:1')
        
        session_dir = training_manager.base_dir / training_id
        model_path = session_dir / "model"
        if model_path.exists():
            app_state.omnigen2_model.pipeline.load_lora_weights(str(model_path))
            logger.info("✅ LoRA weights loaded into OmniGen2 model.")
        
        test_image_path = await training_manager.generate_test_image(training_id, name_object, app_state.omnigen2_model)
        
        training_manager.update_status(
            training_id, "completed", 1.0,
            "Training completed successfully! OmniGen2 is ready. DreamO will reload on next use."
        )
        status = training_manager.get_status(training_id)
        status["generated_image_path"] = test_image_path
        
    except Exception as e:
        logger.error(f"Training pipeline error for {training_id}: {str(e)}", exc_info=True)
        training_manager.update_status(training_id, "failed", 0.0, f"Pipeline failed: {str(e)}")

@router.get("/status/{training_id}", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """Get status of training process"""
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
    """Generate image with trained model"""
    try:
        # Check if model exists
        session_dir = training_manager.base_dir / training_id
        model_path = session_dir / "model"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Trained model not found. Make sure training is completed."
            )
        
        # Get global model and use it for inference with LoRA weights
        omnigen2_model = app_state.omnigen2_model
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
    """List all training sessions"""
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
    """Delete training session and all related files"""
    try:
        session_dir = training_manager.base_dir / training_id
        
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Training session not found")
        
        # Remove directory and all contents
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

@router.get("/gpu-status")
async def get_gpu_status():
    """Get current GPU memory status and recommendations"""
    try:
        gpu_status = training_manager.get_gpu_status()
        
        # Create detailed message
        message_parts = [f"Found {gpu_status['gpu_count']} GPU(s)"]
        
        if gpu_status.get('recommended_gpu') is not None:
            message_parts.append(f"Recommended single GPU: {gpu_status['recommended_gpu']}")
        
        if gpu_status.get('multi_gpu_feasible'):
            message_parts.append(f"🚀 Multi-GPU FSDP training available: {gpu_status['total_memory_gb']}GB total memory")
        else:
            message_parts.append("Multi-GPU training not recommended (insufficient memory or GPUs)")
        
        return {
            "success": True,
            "gpu_status": gpu_status,
            "message": ". ".join(message_parts)
        }
    except Exception as e:
        logger.error(f"Error getting GPU status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU status: {str(e)}") 