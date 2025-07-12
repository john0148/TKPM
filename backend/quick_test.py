#!/usr/bin/env python3
"""
Quick test script để kiểm tra tất cả components
"""

import logging
import sys
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test tất cả components"""
    logger.info("🔍 Quick System Test")
    
    # Test 1: CUDA
    logger.info("=== CUDA Test ===")
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {device_name} ({memory_gb:.1f} GB)")
    else:
        logger.error("❌ CUDA not available")
        return False
    
    # Test 2: Imports
    logger.info("=== Import Test ===")
    try:
        from models.dreamo_model import DreamOModel
        logger.info("✅ DreamO model import successful")
    except Exception as e:
        logger.error(f"❌ DreamO import failed: {e}")
        return False
    
    try:
        from models.omnigen2_model import OmniGen2Model
        logger.info("✅ OmniGen2 model import successful")
    except Exception as e:
        logger.error(f"❌ OmniGen2 import failed: {e}")
        return False
    
    # Test 3: Model Initialization
    logger.info("=== Model Initialization Test ===")
    try:
        dreamo_model = DreamOModel()
        logger.info(f"✅ DreamO initialized on {dreamo_model.get_device_info()}")
    except Exception as e:
        logger.error(f"❌ DreamO initialization failed: {e}")
        return False
    
    try:
        omnigen2_model = OmniGen2Model()
        logger.info(f"✅ OmniGen2 initialized on {omnigen2_model.get_device_info()}")
    except Exception as e:
        logger.error(f"❌ OmniGen2 initialization failed: {e}")
        return False
    
    # Test 4: Health Checks
    logger.info("=== Health Check Test ===")
    dreamo_healthy = dreamo_model.health_check()
    omnigen2_healthy = omnigen2_model.health_check()
    
    logger.info(f"DreamO health: {'✅ Healthy' if dreamo_healthy else '❌ Unhealthy'}")
    logger.info(f"OmniGen2 health: {'✅ Healthy' if omnigen2_healthy else '❌ Unhealthy'}")
    
    if not (dreamo_healthy and omnigen2_healthy):
        logger.error("❌ Some models are unhealthy")
        return False
    
    # Test 5: Training Components
    logger.info("=== Training Components Test ===")
    try:
        from PIL import Image
        dummy_img = Image.new('RGB', (512, 512), color='red')
        
        # Test DreamO multi-reference
        dreamo_result = dreamo_model.generate_multi_reference(
            reference_images=[dummy_img],
            num_outputs=1,
            prompt="test",
            num_inference_steps=1  # Very fast test
        )
        logger.info("✅ DreamO multi-reference working")
        
        # Test OmniGen2 generate_image
        omnigen2_result = omnigen2_model.generate_image(
            prompt="test",
            num_inference_steps=1,  # Very fast test
            width=512,
            height=512
        )
        logger.info("✅ OmniGen2 generate_image working")
        
    except Exception as e:
        logger.error(f"❌ Training components test failed: {e}")
        return False
    
    logger.info("🎉 All tests passed! System is ready.")
    return True

def main():
    """Main function"""
    success = quick_test()
    
    if success:
        logger.info("✅ System is ready for use!")
        logger.info("💡 You can now run: python start_backend.py")
    else:
        logger.error("❌ System has issues. Please check the errors above.")

if __name__ == "__main__":
    main() 