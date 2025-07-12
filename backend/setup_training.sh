#!/bin/bash

# Setup script for training environment
echo "Setting up training environment..."

# Create necessary directories
mkdir -p training_data
mkdir -p logs
mkdir -p models

# Install additional dependencies for training
echo "Installing training dependencies..."
pip install omegaconf
pip install peft
pip install accelerate
pip install transformers
pip install diffusers
pip install datasets
pip install wandb  # Optional for logging
pip install tensorboard

# Setup accelerate config
echo "Setting up accelerate config..."
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    accelerate config --config_file ~/.cache/huggingface/accelerate/default_config.yaml
fi

# Create experiments directory for OmniGen2
mkdir -p ../OmniGen2-DFloat11/experiments

# Set permissions
chmod +x setup_training.sh

echo "Training environment setup completed!"
echo ""
echo "Usage:"
echo "1. Start training: POST /api/training/start"
echo "2. Check status: GET /api/training/status/{training_id}"
echo "3. Run inference: POST /api/training/inference/{training_id}"
echo "4. List sessions: GET /api/training/list"
echo "5. Delete session: DELETE /api/training/delete/{training_id}" 