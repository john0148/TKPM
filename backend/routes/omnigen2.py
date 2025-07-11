"""
FastAPI routes for OmniGen2 image generation and editing
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from schemas.omnigen2_schemas import (
    OmniGen2InContextRequest,
    OmniGen2EditRequest, 
    OmniGen2Response,
    OmniGen2HealthResponse
)
from utils.image_utils import (
    decode_base64_to_image,
    decode_base64_list_to_images,
    encode_image_list_to_base64,
    create_error_response,
    create_success_response,
    TimingContext,
    resize_image_if_needed
)

logger = logging.getLogger(__name__)

router = APIRouter()

def get_omnigen2_model():
    """Dependency to get OmniGen2 model instance"""
    import main
    if main.omnigen2_model is None:
        raise HTTPException(status_code=503, detail="OmniGen2 model not loaded")
    return main.omnigen2_model

@router.post("/in-context-generation", response_model=OmniGen2Response)
async def generate_in_context(
    request: OmniGen2InContextRequest,
    model = Depends(get_omnigen2_model)
):
    """
    Generate image with in-context generation using multiple input images
    
    This endpoint allows you to compose multiple objects from different images
    into a new scene based on the instruction. For example:
    - Combine a person from one image with an object from another
    - Place multiple objects in a new environment
    - Create complex compositions with multiple subjects
    
    The instruction should reference the input images explicitly, e.g.:
    "Let the person in image 2 hold the toy from image 1 in a parking lot"
    """
    
    with TimingContext() as timer:
        try:
            logger.info(f"In-context generation with {len(request.input_images)} input images")
            
            # Decode input images
            input_images_pil = []
            for i, img_data in enumerate(request.input_images):
                try:
                    img = decode_base64_to_image(img_data)
                    # Resize if needed to prevent memory issues
                    img = resize_image_if_needed(
                        img, 
                        max_width=request.max_input_image_side_length,
                        max_height=request.max_input_image_side_length,
                        max_pixels=request.max_pixels
                    )
                    input_images_pil.append(img)
                    
                except Exception as e:
                    logger.error(f"Failed to decode input image {i+1}: {e}")
                    return JSONResponse(
                        status_code=400,
                        content=create_error_response(
                            f"Failed to decode input image {i+1}: {e}",
                            timer.elapsed_time
                        )
                    )
            
            logger.info(f"Decoded {len(input_images_pil)} input images")
            
            # Generate image using OmniGen2
            result = model.generate_in_context(
                instruction=request.instruction,
                input_images=input_images_pil,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                text_guidance_scale=request.text_guidance_scale,
                image_guidance_scale=request.image_guidance_scale,
                cfg_range_start=request.cfg_range_start,
                cfg_range_end=request.cfg_range_end,
                negative_prompt=request.negative_prompt,
                num_images_per_prompt=request.num_images_per_prompt,
                max_input_image_side_length=request.max_input_image_side_length,
                max_pixels=request.max_pixels,
                seed=request.seed,
                scheduler=request.scheduler.value
            )
            
            # Encode individual images if multiple were generated
            individual_images_encoded = None
            if result.get("individual_images"):
                try:
                    individual_images_encoded = encode_image_list_to_base64(
                        result["individual_images"],
                        format="PNG"
                    )
                except Exception as e:
                    logger.warning(f"Failed to encode individual images: {e}")
            
            # Create success response
            response_data = {
                "success": True,
                "image": create_success_response(
                    result["image"],
                    format="PNG"
                )["image"],
                "individual_images": individual_images_encoded,
                "instruction": result["instruction"],
                "seed": result["seed"],
                "num_input_images": result["num_input_images"],
                "error": None,
                "generation_time": timer.elapsed_time
            }
            
            logger.info(f"Successfully generated in-context image in {timer.elapsed_time:.2f}s")
            return response_data
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content=create_error_response(str(e), timer.elapsed_time)
            )
            
        except Exception as e:
            logger.error(f"In-context generation error: {e}")
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    f"In-context generation failed: {e}",
                    timer.elapsed_time
                )
            )

@router.post("/edit", response_model=OmniGen2Response)
async def edit_image(
    request: OmniGen2EditRequest,
    model = Depends(get_omnigen2_model)
):
    """
    Edit image based on instruction
    
    This endpoint allows you to modify an existing image according to 
    text instructions. Examples of edits include:
    - Change background: "Change the background to classroom"
    - Modify objects: "Replace the sword with a hammer"
    - Add elements: "Add a fisherman hat to the woman's head"
    - Remove elements: "Remove the cat"
    - Transform style: "Make it look like a painting"
    
    The instruction should be clear and specific about what changes to make.
    """
    
    with TimingContext() as timer:
        try:
            logger.info(f"Editing image with instruction: '{request.instruction}'")
            
            # Decode input image
            try:
                input_image_pil = decode_base64_to_image(request.input_image)
                # Resize if needed
                input_image_pil = resize_image_if_needed(
                    input_image_pil,
                    max_width=request.max_input_image_side_length,
                    max_height=request.max_input_image_side_length,
                    max_pixels=request.max_pixels
                )
                
            except Exception as e:
                logger.error(f"Failed to decode input image: {e}")
                return JSONResponse(
                    status_code=400,
                    content=create_error_response(
                        f"Failed to decode input image: {e}",
                        timer.elapsed_time
                    )
                )
            
            # Edit image using OmniGen2
            result = model.edit_image(
                instruction=request.instruction,
                input_image=input_image_pil,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                text_guidance_scale=request.text_guidance_scale,
                image_guidance_scale=request.image_guidance_scale,
                cfg_range_start=request.cfg_range_start,
                cfg_range_end=request.cfg_range_end,
                negative_prompt=request.negative_prompt,
                num_images_per_prompt=request.num_images_per_prompt,
                max_input_image_side_length=request.max_input_image_side_length,
                max_pixels=request.max_pixels,
                seed=request.seed,
                scheduler=request.scheduler.value
            )
            
            # Create success response
            response_data = {
                "success": True,
                "image": create_success_response(
                    result["image"],
                    format="PNG"
                )["image"],
                "individual_images": None,  # Single image for edit
                "instruction": result["instruction"],
                "seed": result["seed"],
                "num_input_images": 1,
                "error": None,
                "generation_time": timer.elapsed_time
            }
            
            logger.info(f"Successfully edited image in {timer.elapsed_time:.2f}s")
            return response_data
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content=create_error_response(str(e), timer.elapsed_time)
            )
            
        except Exception as e:
            logger.error(f"Image editing error: {e}")
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    f"Image editing failed: {e}",
                    timer.elapsed_time
                )
            )

@router.get("/health", response_model=OmniGen2HealthResponse)
async def health_check(model = Depends(get_omnigen2_model)):
    """
    Check OmniGen2 model health status
    
    Returns information about model status, device, and memory usage.
    """
    try:
        is_healthy = model.health_check()
        
        # Get device info
        device = "unknown"
        memory_usage = None
        
        try:
            if hasattr(model, 'accelerator'):
                device = str(model.accelerator.device)
                
            # Try to get memory usage
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                cached = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_usage = f"{allocated:.1f}GB allocated, {cached:.1f}GB cached / {total:.1f}GB total"
                
        except Exception as e:
            logger.warning(f"Could not get device/memory info: {e}")
        
        return OmniGen2HealthResponse(
            healthy=is_healthy,
            model_loaded=model is not None,
            device=device,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return OmniGen2HealthResponse(
            healthy=False,
            model_loaded=False,
            device="unknown",
            memory_usage=None
        )

@router.get("/examples")
async def get_examples():
    """
    Get example requests for OmniGen2 API
    
    Returns example configurations for different use cases:
    - In-context generation with multiple objects
    - Image editing with various modifications
    """
    examples = {
        "in_context_generation": {
            "person_with_object": {
                "instruction": "Please let the person in image 2 hold the toy from the first image in a parking lot.",
                "input_images": [
                    "data:image/jpeg;base64,[BASE64_ENCODED_TOY_IMAGE]",
                    "data:image/jpeg;base64,[BASE64_ENCODED_PERSON_IMAGE]"
                ],
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0
            },
            "multiple_objects": {
                "instruction": "The cat is sitting on the table. The bird is perching on the edge of the table.",
                "input_images": [
                    "data:image/jpeg;base64,[BASE64_ENCODED_CAT_IMAGE]",
                    "data:image/jpeg;base64,[BASE64_ENCODED_BIRD_IMAGE]",
                    "data:image/jpeg;base64,[BASE64_ENCODED_TABLE_IMAGE]"
                ],
                "width": 800,
                "height": 512,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0
            },
            "wedding_scene": {
                "instruction": "Create a wedding figure based on the girl in the first image and the man in the second image. Set the background as a wedding hall.",
                "input_images": [
                    "data:image/jpeg;base64,[BASE64_ENCODED_WOMAN_IMAGE]",
                    "data:image/jpeg;base64,[BASE64_ENCODED_MAN_IMAGE]"
                ],
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 3.0
            }
        },
        "image_editing": {
            "background_change": {
                "instruction": "Change the background to classroom",
                "input_image": "data:image/jpeg;base64,[BASE64_ENCODED_IMAGE]",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0
            },
            "object_replacement": {
                "instruction": "Replace the sword with a hammer",
                "input_image": "data:image/jpeg;base64,[BASE64_ENCODED_IMAGE]",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0
            },
            "add_element": {
                "instruction": "Add a fisherman hat to the woman's head",
                "input_image": "data:image/jpeg;base64,[BASE64_ENCODED_IMAGE]",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0
            },
            "remove_element": {
                "instruction": "Remove the cat",
                "input_image": "data:image/jpeg;base64,[BASE64_ENCODED_IMAGE]",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0
            }
        }
    }
    
    return {
        "examples": examples,
        "description": "Example requests for OmniGen2 in-context generation and image editing",
        "supported_schedulers": ["euler", "dpmsolver"],
        "max_input_images": 5,
        "recommended_settings": {
            "in_context_generation": {
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0,
                "num_inference_steps": 50
            },
            "image_editing": {
                "text_guidance_scale": 5.0,
                "image_guidance_scale": 2.0,
                "num_inference_steps": 50
            }
        }
    }

@router.get("/models/info")
async def get_model_info():
    """
    Get detailed information about OmniGen2 model configuration
    """
    return {
        "model_name": "OmniGen2",
        "description": "Advanced multimodal generation with DFloat11 compression",
        "optimization": "DFloat11 lossless compression - 32% smaller with identical outputs",
        "capabilities": [
            "Visual Understanding - Interpret and analyze image content",
            "Text-to-Image Generation - Create images from text descriptions",
            "Image Editing - Modify images based on instructions",
            "In-context Generation - Compose multiple objects into new scenes"
        ],
        "architecture": {
            "text_encoder": "Qwen2.5-VL for robust text and visual understanding",
            "transformer": "Custom architecture with flash attention",
            "vae": "Advanced encoder/decoder for latent space",
            "scheduler": "Flow matching and DPM solver support"
        },
        "supported_formats": ["JPEG", "PNG", "WebP"],
        "max_input_resolution": "2048x2048",
        "max_output_resolution": "2048x2048",
        "typical_inference_time": "25-30 seconds on A100 GPU",
        "memory_requirements": {
            "dfloat11_compressed": "14.3GB VRAM peak usage",
            "original_bfloat16": "18.4GB VRAM peak usage",
            "compression_ratio": "32% size reduction"
        },
        "supported_tasks": [
            "Text-to-image generation",
            "Instruction-guided image editing",
            "Multi-object composition",
            "Scene understanding and modification"
        ]
    } 