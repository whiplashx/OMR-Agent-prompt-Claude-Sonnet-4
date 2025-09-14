#!/usr/bin/env python3
"""
CUDA Installation Helper for OMR Pipeline
=========================================

Helps install CUDA-compatible versions of PyTorch and other dependencies.
"""

import subprocess
import sys
import platform
from pathlib import Path

def check_nvidia_driver():
    """Check if NVIDIA drivers are installed."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA drivers detected")
            # Extract CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"   CUDA Version: {cuda_version}")
                    return cuda_version
            return "unknown"
        else:
            print("❌ NVIDIA drivers not found")
            return None
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers not installed")
        return None

def install_pytorch_cuda(cuda_version="11.8"):
    """Install PyTorch with CUDA support."""
    print(f"\n🔧 Installing PyTorch with CUDA {cuda_version} support...")
    
    # Map CUDA versions to PyTorch index URLs
    cuda_urls = {
        "11.8": "https://download.pytorch.org/whl/cu118",
        "12.1": "https://download.pytorch.org/whl/cu121",
        "12.2": "https://download.pytorch.org/whl/cu121",  # Use 12.1 for 12.2
        "12.3": "https://download.pytorch.org/whl/cu121",  # Use 12.1 for 12.3
    }
    
    # Default to 11.8 if version not supported
    index_url = cuda_urls.get(cuda_version, cuda_urls["11.8"])
    
    commands = [
        [sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "-y"],
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", index_url]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Command completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    return True

def install_tensorflow_cuda():
    """Install TensorFlow with CUDA support."""
    print("\n🔧 Installing TensorFlow with CUDA support...")
    
    commands = [
        [sys.executable, "-m", "pip", "uninstall", "tensorflow", "tensorflow-gpu", "-y"],
        [sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]"]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Command completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    return True

def verify_cuda_installation():
    """Verify that CUDA installations work correctly."""
    print("\n🔍 Verifying CUDA installations...")
    
    # Test PyTorch
    try:
        import torch
        pytorch_cuda = torch.cuda.is_available()
        print(f"PyTorch CUDA: {'✅' if pytorch_cuda else '❌'}")
        if pytorch_cuda:
            print(f"   Device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ PyTorch CUDA error: {e}")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow CUDA: {'✅' if gpus else '❌'}")
        if gpus:
            print(f"   GPU count: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
    except ImportError:
        print("❌ TensorFlow not installed")
    except Exception as e:
        print(f"❌ TensorFlow CUDA error: {e}")
    
    # Test Ultralytics
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        ultralytics_cuda = str(model.device) != 'cpu'
        print(f"Ultralytics CUDA: {'✅' if ultralytics_cuda else '❌'}")
        if ultralytics_cuda:
            print(f"   Device: {model.device}")
    except ImportError:
        print("❌ Ultralytics not installed")
    except Exception as e:
        print(f"❌ Ultralytics CUDA error: {e}")

def main():
    """Main installation function."""
    print("🚀 CUDA Installation Helper for OMR Pipeline")
    print("=" * 60)
    
    # Check system
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Check NVIDIA drivers
    cuda_version = check_nvidia_driver()
    
    if cuda_version is None:
        print("\n❌ NVIDIA drivers not found!")
        print("\n📋 Installation Steps:")
        print("1. Download and install NVIDIA drivers from: https://www.nvidia.com/drivers")
        print("2. Download and install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("3. Restart your computer")
        print("4. Run this script again")
        return 1
    
    # Ask user what to install
    print(f"\n📋 Available installations:")
    print("1. PyTorch with CUDA")
    print("2. TensorFlow with CUDA")
    print("3. Both PyTorch and TensorFlow")
    print("4. Verify current installation")
    print("5. Exit")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled.")
        return 1
    
    if choice == "1":
        install_pytorch_cuda(cuda_version)
    elif choice == "2":
        install_tensorflow_cuda()
    elif choice == "3":
        install_pytorch_cuda(cuda_version)
        install_tensorflow_cuda()
    elif choice == "4":
        pass  # Just verify
    elif choice == "5":
        print("Goodbye!")
        return 0
    else:
        print("Invalid choice!")
        return 1
    
    # Always verify at the end
    verify_cuda_installation()
    
    print("\n🎉 CUDA installation process completed!")
    print("\n💡 Next steps:")
    print("1. Restart your IDE/terminal")
    print("2. Test the OMR pipeline: python cuda_config.py")
    print("3. Launch web interface: python launch_web_ui.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())