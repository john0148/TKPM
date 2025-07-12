#!/usr/bin/env python3
"""
Script kiểm tra nhanh cấu hình GPU và models
"""

import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_config():
    """Kiểm tra cấu hình GPU"""
    logger.info("=== GPU Configuration Check ===")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA không khả dụng!")
        return False
    
    device_count = torch.cuda.device_count()
    logger.info(f"✅ CUDA available, {device_count} device(s) found")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  GPU {i}: {device_name} ({memory_gb:.1f} GB)")
    
    if device_count >= 2:
        logger.info("✅ Đủ GPU cho cấu hình tối ưu:")
        logger.info("  - DreamO: cuda:0")
        logger.info("  - OmniGen2: cuda:1")
    else:
        logger.warning("⚠️ Chỉ có 1 GPU, cả hai models sẽ chạy trên cuda:0")
    
    return True

def check_model_paths():
    """Kiểm tra đường dẫn models"""
    logger.info("=== Model Paths Check ===")
    
    import os
    
    paths_to_check = [
        ("../DreamO", "DreamO"),
        ("../OmniGen2-DFloat11", "OmniGen2"),
        ("models", "Backend models"),
        ("routes", "API routes"),
        ("schemas", "Schemas")
    ]
    
    all_good = True
    for path, name in paths_to_check:
        if os.path.exists(path):
            logger.info(f"✅ {name}: {path}")
        else:
            logger.error(f"❌ {name}: {path} not found")
            all_good = False
    
    return all_good

def check_dependencies():
    """Kiểm tra dependencies"""
    logger.info("=== Dependencies Check ===")
    
    required_packages = [
        "torch",
        "transformers", 
        "diffusers",
        "accelerate",
        "fastapi",
        "uvicorn",
        "pillow",
        "numpy"
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} not installed")
            all_good = False
    
    return all_good

def main():
    """Main check function"""
    logger.info("🔍 Quick Configuration Check")
    
    # Check GPU
    gpu_ok = check_gpu_config()
    
    # Check paths
    paths_ok = check_model_paths()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Summary
    logger.info("=== Summary ===")
    if gpu_ok and paths_ok and deps_ok:
        logger.info("🎉 Tất cả kiểm tra đều pass! Có thể chạy backend.")
        logger.info("💡 Chạy: python start_backend.py")
    else:
        logger.error("❌ Có vấn đề với cấu hình. Vui lòng kiểm tra lại.")
        
        if not gpu_ok:
            logger.error("  - GPU/CUDA không khả dụng")
        if not paths_ok:
            logger.error("  - Thiếu đường dẫn models")
        if not deps_ok:
            logger.error("  - Thiếu dependencies")

if __name__ == "__main__":
    main() 