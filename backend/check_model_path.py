#!/usr/bin/env python3
"""
Script to check model path and directory structure
"""

import os
import sys
from pathlib import Path

def check_model_path():
    """Check if model path exists and has correct structure"""
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check different possible model paths
    possible_paths = [
        "../OmniGen2-DFloat11/DFloat11__OmniGen2-transformer-DF11",
        "OmniGen2-DFloat11/DFloat11__OmniGen2-transformer-DF11",
        "DFloat11__OmniGen2-transformer-DF11",
        "../OmniGen2-DFloat11/DFloat11__OmniGen2-transformer-DF11/",
        "OmniGen2-DFloat11/DFloat11__OmniGen2-transformer-DF11/",
        "DFloat11__OmniGen2-transformer-DF11/"
    ]
    
    print("\nChecking model paths:")
    for path in possible_paths:
        full_path = current_dir / path
        exists = full_path.exists()
        print(f"  {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        
        if exists:
            print(f"    Full path: {full_path}")
            print(f"    Is directory: {full_path.is_dir()}")
            
            # Check for important files
            config_file = full_path / "config.json"
            print(f"    config.json exists: {config_file.exists()}")
            
            # List some files
            files = list(full_path.glob("*.safetensors"))
            print(f"    Number of .safetensors files: {len(files)}")
            
            if len(files) > 0:
                print(f"    Sample files: {[f.name for f in files[:3]]}")
    
    # Check training config
    print("\nChecking training config:")
    config_path = current_dir / "training_data/06c3c47d-0e97-4e2c-b293-638a4dbef7b3/training_config.yml"
    print(f"Training config exists: {config_path.exists()}")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            content = f.read()
            if "pretrained_model_path" in content:
                print("✅ Training config contains pretrained_model_path")
                # Extract the path
                for line in content.split('\n'):
                    if 'pretrained_model_path:' in line:
                        path = line.split(':')[1].strip()
                        print(f"  Current path in config: {path}")
                        break
            else:
                print("❌ Training config does not contain pretrained_model_path")

def check_omnigen2_structure():
    """Check OmniGen2-DFloat11 directory structure"""
    print("\nChecking OmniGen2-DFloat11 structure:")
    
    omnigen2_dir = Path("../OmniGen2-DFloat11")
    if omnigen2_dir.exists():
        print(f"✅ OmniGen2-DFloat11 directory exists: {omnigen2_dir}")
        
        # List contents
        contents = list(omnigen2_dir.iterdir())
        print(f"Contents of OmniGen2-DFloat11:")
        for item in contents:
            if item.is_dir():
                print(f"  📁 {item.name}/")
            else:
                print(f"  📄 {item.name}")
        
        # Check for transformer model
        transformer_dir = omnigen2_dir / "DFloat11__OmniGen2-transformer-DF11"
        if transformer_dir.exists():
            print(f"\n✅ Transformer model directory exists: {transformer_dir}")
            
            # Check for config.json
            config_file = transformer_dir / "config.json"
            if config_file.exists():
                print(f"✅ config.json exists in transformer directory")
                
                # Read and display config
                try:
                    import json
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    print(f"Config keys: {list(config.keys())}")
                except Exception as e:
                    print(f"❌ Error reading config.json: {e}")
            else:
                print(f"❌ config.json not found in transformer directory")
        else:
            print(f"❌ Transformer model directory not found")
    else:
        print(f"❌ OmniGen2-DFloat11 directory not found")

def main():
    """Main function"""
    print("🔍 Checking model paths and structure")
    print("=" * 50)
    
    check_model_path()
    check_omnigen2_structure()
    
    print("\n" + "=" * 50)
    print("✅ Path checking completed!")

if __name__ == "__main__":
    main() 