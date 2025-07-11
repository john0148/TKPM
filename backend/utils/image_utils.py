"""
Image utility functions for FastAPI backend
"""

import base64
import io
import time
from typing import List, Optional, Tuple, Any
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    try:
        buffer = io.BytesIO()
        
        # Convert RGBA to RGB for JPEG
        if format.upper() == "JPEG" and image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
            image = background
        
        image.save(buffer, format=format, quality=95 if format.upper() == "JPEG" else None)
        buffer.seek(0)
        
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        # Add data URI prefix
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise e

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded string (with or without data URI prefix)
        
    Returns:
        PIL Image object
    """
    try:
        # Remove data URI prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Create PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        logger.error(f"Error decoding base64 to image: {e}")
        raise ValueError(f"Invalid base64 image data: {e}")

def encode_image_list_to_base64(images: List[Image.Image], format: str = "PNG") -> List[str]:
    """
    Encode list of PIL Images to base64 strings
    
    Args:
        images: List of PIL Image objects
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        List of base64 encoded strings
    """
    try:
        encoded_images = []
        for image in images:
            encoded_image = encode_image_to_base64(image, format)
            encoded_images.append(encoded_image)
        return encoded_images
        
    except Exception as e:
        logger.error(f"Error encoding image list to base64: {e}")
        raise e

def decode_base64_list_to_images(base64_strings: List[str]) -> List[Image.Image]:
    """
    Decode list of base64 strings to PIL Images
    
    Args:
        base64_strings: List of base64 encoded strings
        
    Returns:
        List of PIL Image objects
    """
    try:
        images = []
        for base64_string in base64_strings:
            image = decode_base64_to_image(base64_string)
            images.append(image)
        return images
        
    except Exception as e:
        logger.error(f"Error decoding base64 list to images: {e}")
        raise e

def validate_image_size(image: Image.Image, max_width: int = 2048, max_height: int = 2048, max_pixels: int = 4194304) -> bool:
    """
    Validate image dimensions and pixel count
    
    Args:
        image: PIL Image object
        max_width: Maximum allowed width
        max_height: Maximum allowed height  
        max_pixels: Maximum allowed total pixels
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        width, height = image.size
        total_pixels = width * height
        
        if width > max_width or height > max_height:
            logger.warning(f"Image size {width}x{height} exceeds maximum dimensions {max_width}x{max_height}")
            return False
            
        if total_pixels > max_pixels:
            logger.warning(f"Image has {total_pixels} pixels, exceeds maximum {max_pixels}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating image size: {e}")
        return False

def resize_image_if_needed(image: Image.Image, max_width: int = 2048, max_height: int = 2048, max_pixels: int = 4194304) -> Image.Image:
    """
    Resize image if it exceeds limits while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_width: Maximum allowed width
        max_height: Maximum allowed height
        max_pixels: Maximum allowed total pixels
        
    Returns:
        Resized PIL Image object
    """
    try:
        width, height = image.size
        total_pixels = width * height
        
        # Check if resize is needed
        if width <= max_width and height <= max_height and total_pixels <= max_pixels:
            return image
        
        # Calculate resize ratio based on dimensions
        width_ratio = max_width / width if width > max_width else 1.0
        height_ratio = max_height / height if height > max_height else 1.0
        dimension_ratio = min(width_ratio, height_ratio)
        
        # Calculate resize ratio based on pixels
        pixel_ratio = (max_pixels / total_pixels) ** 0.5 if total_pixels > max_pixels else 1.0
        
        # Use the most restrictive ratio
        final_ratio = min(dimension_ratio, pixel_ratio)
        
        # Calculate new dimensions
        new_width = int(width * final_ratio)
        new_height = int(height * final_ratio)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        return resized_image
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image

def create_error_response(error_message: str, generation_time: Optional[float] = None) -> dict:
    """
    Create standardized error response
    
    Args:
        error_message: Error message to include
        generation_time: Optional generation time
        
    Returns:
        Error response dictionary
    """
    response = {
        "success": False,
        "image": None,
        "error": error_message
    }
    
    if generation_time is not None:
        response["generation_time"] = generation_time
        
    return response

def create_success_response(
    image: Image.Image, 
    additional_data: Optional[dict] = None,
    generation_time: Optional[float] = None,
    format: str = "PNG"
) -> dict:
    """
    Create standardized success response
    
    Args:
        image: Generated PIL Image
        additional_data: Additional data to include in response
        generation_time: Time taken for generation
        format: Image format for encoding
        
    Returns:
        Success response dictionary
    """
    try:
        response = {
            "success": True,
            "image": encode_image_to_base64(image, format),
            "error": None
        }
        
        if additional_data:
            response.update(additional_data)
            
        if generation_time is not None:
            response["generation_time"] = generation_time
            
        return response
        
    except Exception as e:
        logger.error(f"Error creating success response: {e}")
        return create_error_response(f"Failed to encode response image: {e}", generation_time)

class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time 