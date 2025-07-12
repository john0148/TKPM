#!/usr/bin/env python3
"""
Script để cấu hình accelerate tự động
"""

import subprocess
import sys

def setup_accelerate():
    """Cấu hình accelerate để tránh DeepSpeed"""
    print("🔧 Setting up Accelerate configuration...")
    
    # Tạo accelerate config không sử dụng DeepSpeed
    config = """compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: '0,1'  # Sử dụng GPU 0 và 1
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    # Lưu config vào file
    import os
    config_dir = os.path.expanduser("~/.cache/huggingface/accelerate")
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, "default_config.yaml")
    with open(config_path, "w") as f:
        f.write(config)
    
    print(f"✅ Accelerate config saved to: {config_path}")
    print("💡 Training will now use standard PyTorch without DeepSpeed")
    print("🎯 GPU configuration: Using GPU 0 and 1")

if __name__ == "__main__":
    setup_accelerate() 