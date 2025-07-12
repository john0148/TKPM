#!/usr/bin/env python3
"""
Script để kiểm tra accelerate config
"""

import os
import subprocess
import sys

def check_accelerate_config():
    """Kiểm tra accelerate config hiện tại"""
    print("🔍 Checking Accelerate configuration...")
    
    try:
        # Chạy accelerate env để xem config hiện tại
        result = subprocess.run(
            ["accelerate", "env"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Accelerate config found:")
            print(result.stdout)
        else:
            print("❌ Accelerate config not found or invalid")
            print("Error:", result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ Accelerate not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking accelerate config: {e}")
        return False
    
    return True

def setup_accelerate_config():
    """Setup accelerate config nếu chưa có"""
    print("\n🔧 Setting up Accelerate configuration...")
    
    try:
        # Import và chạy setup function
        from setup_accelerate import setup_accelerate
        setup_accelerate()
        return True
    except Exception as e:
        print(f"❌ Failed to setup accelerate: {e}")
        return False

def main():
    """Main function"""
    print("=== Accelerate Configuration Check ===")
    
    # Kiểm tra config hiện tại
    config_ok = check_accelerate_config()
    
    if not config_ok:
        print("\n🔄 Setting up new accelerate config...")
        if setup_accelerate_config():
            print("✅ Accelerate config setup successful!")
            
            # Kiểm tra lại
            print("\n🔍 Verifying new config...")
            check_accelerate_config()
        else:
            print("❌ Failed to setup accelerate config")
            return False
    else:
        print("✅ Accelerate config is already properly configured!")
    
    print("\n🎉 Accelerate is ready for training!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 