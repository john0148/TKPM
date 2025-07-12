"""
FastAPI routes for DreamO image generation
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from schemas.dreamo_schemas import (
    DreamOGenerateRequest, 
    DreamOGenerateResponse, 
    DreamOHealthResponse
)
from utils.image_utils import (
    decode_base64_list_to_images,
    encode_image_list_to_base64,
    create_error_response,
    create_success_response,
    TimingContext,
    resize_image_if_needed
)

logger = logging.getLogger(__name__)

router = APIRouter()

def get_dreamo_model():
    """Dependency to get DreamO model instance"""
    import main
    if main.dreamo_model is None:
        raise HTTPException(status_code=503, detail="DreamO model not loaded")
    return main.dreamo_model

@router.post("/generate", response_model=DreamOGenerateResponse)
async def generate_image(
    request: DreamOGenerateRequest,
    model = Depends(get_dreamo_model)
):
    """
    Generate image using DreamO with multiple reference images
    
    This endpoint supports up to 10 reference images with different tasks:
    - IP: General image prompting (objects, characters, animals)
    - ID: Face identity preservation 
    - Style: Style transfer
    
    Each reference image can have a different task type, allowing for complex
    multi-modal generation scenarios.
    """
    
    logger.info(f"Received DreamO request - prompt: '{request.prompt}', ref_images: {len(request.ref_images)}")
    
    with TimingContext() as timer:
        try:
            logger.info(f"Generating image with {len(request.ref_images)} reference images")
            
            # Decode reference images
            ref_images_pil = []
            ref_tasks = []
            
            for ref_img_data in request.ref_images:
                try:
                    # Decode base64 to PIL Image
                    img = decode_base64_list_to_images([ref_img_data.image_data])[0]
                    
                    # Resize if needed to prevent memory issues
                    img = resize_image_if_needed(img, max_width=2048, max_height=2048)
                    
                    ref_images_pil.append(img)
                    ref_tasks.append(ref_img_data.task.value)
                    
                except Exception as e:
                    logger.error(f"Failed to decode reference image: {e}")
                    return JSONResponse(
                        status_code=400,
                        content=create_error_response(
                            f"Failed to decode reference image: {e}",
                            timer.elapsed_time
                        )
                    )
            
            logger.info(f"Decoded {len(ref_images_pil)} reference images with tasks: {ref_tasks}")
            
            # Generate image using DreamO
            result = model.generate_image(
                prompt=request.prompt,
                ref_images=ref_images_pil,
                ref_tasks=ref_tasks,
                width=request.width,
                height=request.height,
                ref_res=request.ref_res,
                num_steps=request.num_steps,
                guidance=request.guidance,
                seed=request.seed,
                true_cfg=request.true_cfg,
                cfg_start_step=request.cfg_start_step,
                cfg_end_step=request.cfg_end_step,
                neg_prompt=request.neg_prompt,
                neg_guidance=request.neg_guidance,
                first_step_guidance=request.first_step_guidance
            )
            
            # Encode debug images if present
            debug_images_encoded = None
            if result.get("debug_images"):
                try:
                    debug_images_encoded = encode_image_list_to_base64(
                        result["debug_images"], 
                        format="PNG"
                    )
                except Exception as e:
                    logger.warning(f"Failed to encode debug images: {e}")
            
            # Create success response
            response_data = {
                "success": True,
                "image": create_success_response(
                    result["image"],
                    format="PNG"
                )["image"],
                "debug_images": debug_images_encoded,
                "seed": result["seed"],
                "prompt": result["prompt"],
                "ref_count": result["ref_count"],
                "error": None,
                "generation_time": timer.elapsed_time
            }
            
            logger.info(f"Successfully generated image in {timer.elapsed_time:.2f}s")
            return response_data
            
        except ValueError as e:
            # Handle validation errors
            logger.error(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content=create_error_response(str(e), timer.elapsed_time)
            )
            
        except Exception as e:
            # Handle generation errors
            logger.error(f"Generation error: {e}")
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    f"Image generation failed: {e}",
                    timer.elapsed_time
                )
            )

@router.get("/health", response_model=DreamOHealthResponse)
async def health_check(model = Depends(get_dreamo_model)):
    """
    Check DreamO model health status
    
    Returns information about model status, version, and device.
    """
    try:
        is_healthy = model.health_check()
        
        # Get device info
        device = model.get_device_info()
        
        return DreamOHealthResponse(
            healthy=is_healthy,
            model_loaded=model is not None,
            version="v1.1",  # DreamO version
            device=device
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return DreamOHealthResponse(
            healthy=False,
            model_loaded=False,
            version="unknown",
            device="unknown"
        )

@router.get("/examples")
async def get_examples():
    """
    Get example requests for DreamO API
    
    Returns various example configurations showing different use cases:
    - Single IP reference
    - ID face preservation
    - Style transfer
    - Multi-reference combinations
    """
    examples = {
        "single_ip": {
            "prompt": "a person playing guitar in the street",
            "ref_images": [
                {
                    "image_data": "data:image/jpeg;base64,[BASE64_ENCODED_PERSON_IMAGE]",
                    "task": "ip"
                }
            ],
            "width": 1024,
            "height": 1024,
            "num_steps": 12,
            "guidance": 4.5
        },
        "face_id": {
            "prompt": "portrait, professional headshot",
            "ref_images": [
                {
                    "image_data": "data:image/jpeg;base64,[BASE64_ENCODED_FACE_IMAGE]",
                    "task": "id"
                }
            ],
            "width": 1024,
            "height": 1024,
            "num_steps": 12,
            "guidance": 4.5
        },
        "style_transfer": {
            "prompt": "generate a same style image. A rooster wearing overalls.",
            "ref_images": [
                {
                    "image_data": "data:image/jpeg;base64,[BASE64_ENCODED_STYLE_IMAGE]",
                    "task": "style"
                }
            ],
            "width": 1024,
            "height": 1024,
            "num_steps": 12,
            "guidance": 4.5
        },
        "multi_reference": {
            "prompt": "A girl is wearing a shirt and skirt on the beach",
            "ref_images": [
                {
                    "image_data": "data:image/jpeg;base64,[BASE64_ENCODED_PERSON_IMAGE]",
                    "task": "id"
                },
                {
                    "image_data": "data:image/jpeg;base64,[BASE64_ENCODED_SHIRT_IMAGE]",
                    "task": "ip"
                },
                {
                    "image_data": "data:image/jpeg;base64,[BASE64_ENCODED_SKIRT_IMAGE]",
                    "task": "ip"
                }
            ],
            "width": 1024,
            "height": 1024,
            "num_steps": 12,
            "guidance": 4.5
        }
    }
    
    return {
        "examples": examples,
        "description": "Example requests for different DreamO use cases",
        "supported_tasks": ["ip", "id", "style"],
        "max_reference_images": 10,
        "recommended_settings": {
            "ip_task": {"guidance": 4.5, "num_steps": 12},
            "id_task": {"guidance": 4.5, "num_steps": 12}, 
            "style_task": {"guidance": 4.5, "num_steps": 12}
        }
    }

@router.get("/models/info")
async def get_model_info():
    """
    Get detailed information about DreamO model configuration
    """
    return {
        "model_name": "DreamO",
        "version": "v1.1",
        "description": "Unified Framework for Image Customization",
        "optimization": "Nunchaku quantization for efficient inference",
        "capabilities": [
            "Image Prompting (IP) - General objects, characters, animals",
            "Identity Preservation (ID) - Face identity preservation",
            "Style Transfer - Artistic style application",
            "Multi-condition generation - Combine multiple references"
        ],
        "supported_formats": ["JPEG", "PNG", "WebP"],
        "max_input_resolution": "2048x2048",
        "max_output_resolution": "2048x2048",
        "typical_inference_time": "15-20 seconds on RTX 3080",
        "memory_requirements": {
            "nunchaku_mode": "6.5GB VRAM",
            "int8_mode": "16GB VRAM", 
            "full_precision": "24GB VRAM"
        }
    } 