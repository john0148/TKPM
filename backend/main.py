"""
FastAPI Backend for DreamO and OmniGen2 Services
"""

import os
import sys
import logging
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
from routes import dreamo, omnigen2

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
        # Load DreamO model with nunchaku
        logger.info("Loading DreamO model...")
        dreamo_model = DreamOModel()
        
        # Load OmniGen2 model  
        logger.info("Loading OmniGen2 model...")
        omnigen2_model = OmniGen2Model()
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise e
        
    yield
    
    # Cleanup on shutdown
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Image Generation Backend is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global dreamo_model, omnigen2_model
    
    return {
        "status": "healthy",
        "models": {
            "dreamo": dreamo_model is not None,
            "omnigen2": omnigen2_model is not None
        }
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