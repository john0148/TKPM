#!/bin/bash

# AI Image Generation Backend Setup Script

echo "🔧 Setting up AI Image Generation Backend..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if conda is available
if ! command_exists conda; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'ai-backend'..."
conda create -n ai-backend python=3.10 -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ai-backend

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Install FaceXLib for DreamO
echo "👤 Installing FaceXLib for DreamO..."
pip install git+https://github.com/ToTheBeginning/facexlib.git

# Optional: Install Nunchaku
read -p "🚀 Do you want to install Nunchaku for faster DreamO inference? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Please install Nunchaku manually from:"
    echo "   https://github.com/mit-han-lab/nunchaku/releases/"
    echo "   This provides 2-4x faster inference and reduces VRAM to 6.5GB"
fi

# Check installations
echo "✅ Checking installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

echo ""
echo "✨ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Activate environment: conda activate ai-backend"
echo "2. Run the backend: ./start.sh or python main.py"
echo "3. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "💡 Tips:"
echo "- Make sure DreamO and OmniGen2-DFloat11 directories are in parent folder"
echo "- First model loading may take several minutes"
echo "- Use Nunchaku for DreamO if you have limited VRAM"

# Make scripts executable
chmod +x start.sh 