#!/usr/bin/env python3
"""
Script để kiểm tra tất cả đường dẫn cần thiết
"""

import os
from pathlib import Path

def check_paths():
    """Kiểm tra tất cả đường dẫn cần thiết"""
    print("🔍 Checking all required paths...")
    
    # Danh sách đường dẫn cần kiểm tra
    paths_to_check = [
        ("../OmniGen2-DFloat11/train.py", "Training script"),
        ("../OmniGen2-DFloat11/convert_ckpt_to_hf_format.py", "Convert script"),
        ("../OmniGen2-DFloat11/experiments", "Experiments directory (optional)"),
        ("../DreamO", "DreamO directory"),
        ("../OmniGen2-DFloat11", "OmniGen2 directory"),
        ("models", "Backend models"),
        ("routes", "Backend routes"),
        ("schemas", "Backend schemas"),
        ("utils", "Backend utils")
    ]
    
    all_good = True
    
    for path, description in paths_to_check:
        if os.path.exists(path):
            print(f"✅ {description}: {path}")
        else:
            # Experiments directory không bắt buộc
            if "experiments" in path:
                print(f"⚠️ {description}: {path} - Will be created during training")
            else:
                print(f"❌ {description}: {path} - NOT FOUND")
                all_good = False
    
    # Kiểm tra thêm một số file quan trọng
    important_files = [
        ("../OmniGen2-DFloat11/omnigen2", "OmniGen2 package"),
        ("../DreamO/dreamo_generator.py", "DreamO generator"),
        ("models/dreamo_model.py", "DreamO model wrapper"),
        ("models/omnigen2_model.py", "OmniGen2 model wrapper")
    ]
    
    print("\n🔍 Checking important files...")
    for path, description in important_files:
        if os.path.exists(path):
            print(f"✅ {description}: {path}")
        else:
            print(f"❌ {description}: {path} - NOT FOUND")
            all_good = False
    
    # Kiểm tra working directory
    print(f"\n📁 Current working directory: {os.getcwd()}")
    
    return all_good

def check_relative_paths():
    """Kiểm tra đường dẫn tương đối từ backend/"""
    print("\n🔍 Testing relative paths from backend/...")
    
    # Test các đường dẫn tương đối
    test_paths = [
        "../OmniGen2-DFloat11/train.py",
        "../OmniGen2-DFloat11/convert_ckpt_to_hf_format.py",
        "../DreamO/dreamo_generator.py"
    ]
    
    all_good = True
    
    for path in test_paths:
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - NOT FOUND")
            all_good = False
    
    return all_good

def main():
    """Main function"""
    print("=== Path Check ===")
    
    # Kiểm tra tất cả đường dẫn
    paths_ok = check_paths()
    
    # Kiểm tra đường dẫn tương đối
    relative_ok = check_relative_paths()
    
    # Summary
    print("\n=== Summary ===")
    if paths_ok and relative_ok:
        print("🎉 All paths are correct!")
        print("💡 Training should work without path errors.")
    else:
        print("❌ Some paths are missing or incorrect.")
        print("💡 Please check the missing paths above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1) 