#!/bin/bash
"""
Script để cài đặt dependencies cần thiết cho training
"""

echo "🔧 Installing training dependencies..."

# Cài đặt accelerate
pip install accelerate

# Cài đặt transformers với phiên bản mới nhất
pip install transformers

# Cài đặt diffusers
pip install diffusers

# Cài đặt timm cho learning rate scheduler
pip install timm

# Cài đặt PyYAML cho config files
pip install PyYAML

# Cài đặt các dependencies khác
pip install datasets
pip install huggingface_hub

# Cài đặt accelerate config
accelerate config

echo "✅ Dependencies installed successfully!"
echo "💡 You can now run training without DeepSpeed errors." 