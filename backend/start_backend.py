#!/usr/bin/env python3
"""
Script để chạy backend với cấu hình DreamO trên cuda:0 và OmniGen2 trên cuda:1
"""

import os
import sys
import logging
import torch
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Kiểm tra môi trường trước khi chạy"""
    logger.info("=== Environment Check ===")
    
    # Check CUDA
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {device_name} ({memory_gb:.1f} GB)")
    
    # Check required paths
    required_paths = [
        "../DreamO",
        "../OmniGen2-DFloat11",
        "models",
        "routes",
        "schemas"
    ]
    
    logger.info("Checking required paths...")
    for path in required_paths:
        if os.path.exists(path):
            logger.info(f"  ✅ {path}")
        else:
            logger.warning(f"  ⚠️ {path} not found")
    
    # Check if we have enough GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        logger.warning("⚠️ Only one GPU available. OmniGen2 will use cuda:0 instead of cuda:1")

def start_backend():
    """Chạy backend server"""
    logger.info("=== Starting Backend Server ===")
    
    # Check environment first
    check_environment()
    
    # Start server
    logger.info("🚀 Starting FastAPI server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        log_level="info"
    )

if __name__ == "__main__":
    start_backend() 