#!/bin/bash

# AI Image Generation Backend Startup Script

echo "🚀 Starting AI Image Generation Backend..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the backend directory."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected. Consider activating one."
fi

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ PyTorch not installed"

# Check if models exist
echo "🔍 Checking model directories..."
if [ ! -d "../DreamO" ]; then
    echo "❌ DreamO directory not found at ../DreamO"
fi

if [ ! -d "../OmniGen2-DFloat11" ]; then
    echo "❌ OmniGen2-DFloat11 directory not found at ../OmniGen2-DFloat11"
fi

# Start the server
echo "🌟 Starting FastAPI server..."
echo "📚 API Documentation will be available at:"
echo "   - Swagger UI: http://localhost:8000/docs"
echo "   - ReDoc: http://localhost:8000/redoc"
echo "   - Health Check: http://localhost:8000/health"
echo ""
echo "⏳ Loading models... This may take a few minutes..."
echo ""

# Start with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload 