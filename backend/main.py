"""
FastAPI Backend for DreamO and OmniGen2 Services
"""

import os
import sys
import logging
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# --- THAY ĐỔI: Import app_state và các routes ---
import app_state 
from routes import dreamo, omnigen2, training

from models.dreamo_model import DreamOModel
from models.omnigen2_model import OmniGen2Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- THAY ĐỔI: Loại bỏ các biến toàn cục ở đây ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifespan - load all models on startup"""
    
    logger.info("Loading all models on startup...")
    
    try:
        # --- THAY ĐỔI: Cập nhật trạng thái trong app_state ---
        logger.info("Loading OmniGen2 model on cuda:1...")
        app_state.omnigen2_model = OmniGen2Model(target_device='cuda:1')
        logger.info("✅ OmniGen2 model loaded successfully on cuda:1")
        
        logger.info("Loading DreamO model on cuda:0...")
        app_state.dreamo_model = DreamOModel()
        logger.info("✅ DreamO model loaded successfully on cuda:0")
        
        logger.info("🎉 All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}", exc_info=True)
        logger.warning("App will start with limited functionality")
        
    yield
    
    logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="AI Image Generation Backend",
    description="Backend services for DreamO and OmniGen2 image generation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dreamo.router, prefix="/api/dreamo", tags=["DreamO"])
app.include_router(omnigen2.router, prefix="/api/omnigen2", tags=["OmniGen2"])
app.include_router(training.router, prefix="/api", tags=["Training"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Image Generation Backend is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    
    dreamo_info = {
        "loaded": app_state.dreamo_model is not None,
        "device": app_state.dreamo_model.get_device_info() if app_state.dreamo_model else None,
        "healthy": app_state.dreamo_model.health_check() if app_state.dreamo_model else False,
        "status": "loaded" if app_state.dreamo_model else "not_loaded"
    }
    
    omnigen2_info = {
        "loaded": app_state.omnigen2_model is not None,
        "device": app_state.omnigen2_model.get_device_info() if app_state.omnigen2_model else None,
        "healthy": app_state.omnigen2_model.health_check() if app_state.omnigen2_model else False
    }
    
    return {
        "status": "healthy" if omnigen2_info["loaded"] and dreamo_info["loaded"] else "degraded",
        "models": {
            "dreamo": dreamo_info,
            "omnigen2": omnigen2_info
        },
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )