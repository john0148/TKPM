"""
FastAPI Backend for DreamO and OmniGen2 Services
"""

import os
import sys
import logging
import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DreamO'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'OmniGen2-DFloat11'))

from models.dreamo_model import DreamOModel
from models.omnigen2_model import OmniGen2Model
from routes import dreamo, omnigen2, training

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
dreamo_model = None
omnigen2_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifespan - load models on startup"""
    global dreamo_model, omnigen2_model
    
    logger.info("Loading models...")
    
    try:
        # Load DreamO model first (cuda:0)
        logger.info("Loading DreamO model on cuda:0...")
        dreamo_model = DreamOModel()
        logger.info("✅ DreamO model loaded successfully on cuda:0")
        
        # Load OmniGen2 model second (cuda:1)
        logger.info("Loading OmniGen2 model on cuda:1...")
        omnigen2_model = OmniGen2Model()
        logger.info("✅ OmniGen2 model loaded successfully on cuda:1")
        
        logger.info("🎉 All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        # Don't raise here, let the app start with partial functionality
        logger.warning("App will start with limited functionality")
        
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")
    if dreamo_model:
        logger.info("Cleaning up DreamO model...")
    if omnigen2_model:
        logger.info("Cleaning up OmniGen2 model...")

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
    global dreamo_model, omnigen2_model
    
    dreamo_info = {
        "loaded": dreamo_model is not None,
        "device": dreamo_model.get_device_info() if dreamo_model else None,
        "healthy": dreamo_model.health_check() if dreamo_model else False
    }
    
    omnigen2_info = {
        "loaded": omnigen2_model is not None,
        "device": omnigen2_model.get_device_info() if omnigen2_model else None,
        "healthy": omnigen2_model.health_check() if omnigen2_model else False
    }
    
    return {
        "status": "healthy" if (dreamo_info["loaded"] or omnigen2_info["loaded"]) else "degraded",
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