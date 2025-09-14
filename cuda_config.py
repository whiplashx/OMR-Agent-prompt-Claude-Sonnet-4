#!/usr/bin/env python3
"""
CUDA Configuration for OMR Pipeline
===================================

Configures and checks CUDA availability for GPU acceleration.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def check_cuda_availability() -> Dict[str, any]:
    """
    Check CUDA availability across different frameworks.
    
    Returns:
        Dictionary with CUDA status for each framework
    """
    cuda_status = {
        'pytorch_available': False,
        'pytorch_cuda': False,
        'pytorch_device_count': 0,
        'pytorch_device_name': None,
        'tensorflow_available': False,
        'tensorflow_cuda': False,
        'tensorflow_gpu_count': 0,
        'opencv_cuda': False,
        'ultralytics_cuda': False,
        'recommended_device': 'cpu'
    }
    
    # Check PyTorch CUDA
    try:
        import torch
        cuda_status['pytorch_available'] = True
        cuda_status['pytorch_cuda'] = torch.cuda.is_available()
        if cuda_status['pytorch_cuda']:
            cuda_status['pytorch_device_count'] = torch.cuda.device_count()
            cuda_status['pytorch_device_name'] = torch.cuda.get_device_name(0)
            cuda_status['recommended_device'] = 'cuda'
            logger.info(f"PyTorch CUDA available: {cuda_status['pytorch_device_count']} devices")
            logger.info(f"Primary GPU: {cuda_status['pytorch_device_name']}")
        else:
            logger.debug("PyTorch is available but CUDA is not")
    except ImportError:
        logger.debug("PyTorch not available")
    except Exception as e:
        logger.error(f"Error checking PyTorch CUDA: {e}")
    
    # Check TensorFlow CUDA
    try:
        import tensorflow as tf
        cuda_status['tensorflow_available'] = True
        
        # Handle different TensorFlow versions
        try:
            # TensorFlow 2.x
            if hasattr(tf.config, 'list_physical_devices'):
                gpus = tf.config.list_physical_devices('GPU')
            elif hasattr(tf.config.experimental, 'list_physical_devices'):
                gpus = tf.config.experimental.list_physical_devices('GPU')
            else:
                # Fallback for older versions
                gpus = []
        except AttributeError:
            # Very old TensorFlow version
            gpus = []
            
        cuda_status['tensorflow_cuda'] = len(gpus) > 0
        cuda_status['tensorflow_gpu_count'] = len(gpus)
        if cuda_status['tensorflow_cuda']:
            logger.info(f"TensorFlow CUDA available: {len(gpus)} GPUs")
            for gpu in gpus:
                logger.info(f"TensorFlow GPU: {gpu}")
        else:
            logger.debug("TensorFlow available but no GPUs detected")
    except ImportError:
        logger.debug("TensorFlow not available")
    except Exception as e:
        logger.debug(f"TensorFlow CUDA check failed: {e}")
    
    # Check OpenCV CUDA
    try:
        import cv2
        cuda_status['opencv_cuda'] = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if cuda_status['opencv_cuda']:
            logger.info(f"OpenCV CUDA available: {cv2.cuda.getCudaEnabledDeviceCount()} devices")
        else:
            logger.debug("OpenCV available but CUDA support not enabled")
    except AttributeError:
        logger.debug("OpenCV available but compiled without CUDA support")
    except ImportError:
        logger.debug("OpenCV not available")
    except Exception as e:
        logger.debug(f"Error checking OpenCV CUDA: {e}")
    
    # Check Ultralytics CUDA (YOLO)
    try:
        from ultralytics import YOLO
        # Try to create a model instance and check device
        model = YOLO('yolov8n.pt')  # Lightweight model for testing
        cuda_status['ultralytics_cuda'] = str(model.device) != 'cpu'
        if cuda_status['ultralytics_cuda']:
            logger.info(f"Ultralytics YOLO CUDA available on device: {model.device}")
        else:
            logger.debug("Ultralytics YOLO available but using CPU")
    except ImportError:
        logger.debug("Ultralytics not available")
    except Exception as e:
        logger.debug(f"Error checking Ultralytics CUDA: {e}")
    
    return cuda_status

def get_optimal_device() -> str:
    """
    Get the optimal device for processing.
    
    Returns:
        Device string: 'cuda', 'cuda:0', 'mps', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            return f'cuda:{torch.cuda.current_device()}'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon
    except ImportError:
        pass
    
    return 'cpu'

def configure_cuda_environment():
    """Configure environment variables for optimal CUDA performance."""
    # Suppress TensorFlow verbose output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors and warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
    
    # CUDA optimizations
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Don't block on CUDA operations
    os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Enable CUDA Device-Side Assertions
    
    # Memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # OpenCV CUDA
    os.environ['OPENCV_DNN_CUDA'] = '1'
    
    # TensorFlow GPU memory growth
    try:
        import tensorflow as tf
        
        # Handle different TensorFlow versions
        if hasattr(tf.config, 'list_physical_devices'):
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Configured TensorFlow GPU memory growth")
        elif hasattr(tf.config.experimental, 'list_physical_devices'):
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Configured TensorFlow GPU memory growth")
    except Exception as e:
        logger.warning(f"Could not configure TensorFlow GPU: {e}")

def get_cuda_config() -> Dict:
    """
    Get CUDA configuration for OMR pipeline components.
    
    Returns:
        Configuration dictionary with CUDA settings
    """
    cuda_status = check_cuda_availability()
    device = get_optimal_device()
    
    config = {
        'device': device,
        'use_cuda': device.startswith('cuda'),
        'symbol_detection': {
            'device': device,
            'half_precision': device.startswith('cuda'),  # Use FP16 on GPU
            'batch_size': 16 if device.startswith('cuda') else 4,
            'workers': 4 if device.startswith('cuda') else 2,
        },
        'preprocessing': {
            'use_cuda': cuda_status.get('opencv_cuda', False),
            'gpu_memory_fraction': 0.3  # Limit GPU memory for preprocessing
        },
        'staff_detection': {
            'use_cuda': cuda_status.get('opencv_cuda', False),
            'parallel_processing': device.startswith('cuda')
        },
        'optimization': {
            'torch_compile': device.startswith('cuda'),  # Use torch.compile on GPU
            'mixed_precision': device.startswith('cuda'),
            'cudnn_benchmark': device.startswith('cuda'),
            'pin_memory': device.startswith('cuda')
        }
    }
    
    return config

def setup_cuda_optimizations():
    """Setup CUDA optimizations for maximum performance."""
    configure_cuda_environment()
    
    try:
        import torch
        if torch.cuda.is_available():
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Clear cache
            torch.cuda.empty_cache()
            
            logger.info("CUDA optimizations enabled")
    except ImportError:
        logger.warning("PyTorch not available for CUDA optimizations")

def print_cuda_info():
    """Print comprehensive CUDA information."""
    print("🚀 CUDA Configuration Report")
    print("=" * 50)
    
    cuda_status = check_cuda_availability()
    config = get_cuda_config()
    
    # System info
    print("\n📊 CUDA Availability:")
    print(f"  PyTorch CUDA: {'✅' if cuda_status['pytorch_cuda'] else '❌'}")
    if cuda_status['pytorch_cuda']:
        print(f"    Devices: {cuda_status['pytorch_device_count']}")
        print(f"    Primary GPU: {cuda_status['pytorch_device_name']}")
    
    print(f"  TensorFlow CUDA: {'✅' if cuda_status['tensorflow_cuda'] else '❌'}")
    if cuda_status['tensorflow_cuda']:
        print(f"    GPU Count: {cuda_status['tensorflow_gpu_count']}")
    
    print(f"  OpenCV CUDA: {'✅' if cuda_status['opencv_cuda'] else '❌'}")
    print(f"  Ultralytics CUDA: {'✅' if cuda_status['ultralytics_cuda'] else '❌'}")
    
    # Configuration
    print(f"\n⚙️ Recommended Configuration:")
    print(f"  Device: {config['device']}")
    print(f"  Use CUDA: {'✅' if config['use_cuda'] else '❌'}")
    print(f"  Symbol Detection Batch Size: {config['symbol_detection']['batch_size']}")
    print(f"  Half Precision: {'✅' if config['symbol_detection']['half_precision'] else '❌'}")
    print(f"  Mixed Precision: {'✅' if config['optimization']['mixed_precision'] else '❌'}")
    
    # Performance tips
    print(f"\n💡 Performance Tips:")
    if config['use_cuda']:
        print("  • CUDA acceleration enabled - expect 3-10x speedup")
        print("  • Consider using larger batch sizes for better GPU utilization")
        print("  • Mixed precision training can provide additional speedup")
    else:
        print("  • CUDA not available - using CPU (consider installing CUDA drivers)")
        print("  • Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("  • Ensure NVIDIA drivers and CUDA toolkit are installed")

def install_cuda_pytorch():
    """Provide instructions for installing PyTorch with CUDA support."""
    print("\n🔧 Installing PyTorch with CUDA Support:")
    print("=" * 50)
    print("\n1. Check your CUDA version:")
    print("   nvidia-smi")
    print("\n2. Install PyTorch with CUDA 11.8 (recommended):")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\n3. For CUDA 12.1:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\n4. Verify installation:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")
    print("\n5. For TensorFlow with CUDA:")
    print("   pip install tensorflow[and-cuda]")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Print CUDA information
    print_cuda_info()
    
    # Setup optimizations if CUDA is available
    setup_cuda_optimizations()
    
    # Show installation instructions
    cuda_status = check_cuda_availability()
    if not cuda_status['pytorch_cuda']:
        install_cuda_pytorch()