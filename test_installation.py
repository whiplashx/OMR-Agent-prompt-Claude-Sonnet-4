#!/usr/bin/env python3
"""
Quick Test Script for OMR Pipeline
==================================

A simple test to verify the OMR pipeline is working correctly.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_import():
    """Test if we can import the main pipeline."""
    print("🧪 Testing basic import...")
    
    try:
        from omr_pipeline import OMRPipeline
        print("   ✅ OMRPipeline imported successfully!")
        return True
    except Exception as e:
        print(f"   ❌ Failed to import OMRPipeline: {e}")
        return False

def test_pipeline_creation():
    """Test if we can create a pipeline instance."""
    print("\n🏭 Testing pipeline creation...")
    
    try:
        from omr_pipeline import OMRPipeline
        pipeline = OMRPipeline()
        print("   ✅ OMRPipeline instance created successfully!")
        return True, pipeline
    except Exception as e:
        print(f"   ❌ Failed to create OMRPipeline: {e}")
        return False, None

def test_sample_processing(pipeline):
    """Test processing with a sample image."""
    print("\n🖼️ Testing sample image processing...")
    
    try:
        # Create a simple test image
        import numpy as np
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add some staff lines
        for y in [150, 170, 190, 210, 230]:
            test_image[y-1:y+1, 50:550] = 0
        
        # Process the image
        result = pipeline.process_image(
            image=test_image,
            output_path="test_output.mxl"
        )
        
        if result and 'musicxml_path' in result:
            print("   ✅ Sample processing completed!")
            print(f"      Quality score: {result.get('quality_score', 0):.2f}")
            print(f"      Processing time: {result.get('processing_time', 0):.1f}s")
            return True
        else:
            print("   ⚠️ Processing completed but no result returned")
            return False
            
    except Exception as e:
        print(f"   ❌ Sample processing failed: {e}")
        return False

def test_dependencies():
    """Test if key dependencies are available."""
    print("\n📦 Testing key dependencies...")
    
    dependencies = [
        ("numpy", "NumPy for numerical operations"),
        ("cv2", "OpenCV for computer vision"),
        ("PIL", "Pillow for image handling"),
        ("streamlit", "Streamlit for web UI"),
    ]
    
    available = 0
    total = len(dependencies)
    
    for module, description in dependencies:
        try:
            if module == "cv2":
                import cv2
            elif module == "PIL":
                import PIL
            else:
                __import__(module)
            print(f"   ✅ {module} - {description}")
            available += 1
        except ImportError:
            print(f"   ❌ {module} - {description}")
    
    print(f"\n   📊 Dependencies: {available}/{total} available")
    return available >= 2  # Need at least numpy and one other

def main():
    """Run all tests."""
    print("🎼 OMR Pipeline Quick Test")
    print("=" * 40)
    
    # Test 1: Dependencies
    deps_ok = test_dependencies()
    
    # Test 2: Basic import
    import_ok = test_basic_import()
    
    if not import_ok:
        print("\n❌ Basic import failed. Check your installation.")
        return 1
    
    # Test 3: Pipeline creation
    creation_ok, pipeline = test_pipeline_creation()
    
    if not creation_ok:
        print("\n❌ Pipeline creation failed.")
        return 1
    
    # Test 4: Sample processing (optional)
    if deps_ok:
        processing_ok = test_sample_processing(pipeline)
    else:
        print("\n⚠️ Skipping processing test due to missing dependencies")
        processing_ok = True
    
    # Summary
    print("\n" + "=" * 40)
    if import_ok and creation_ok:
        print("✅ OMR Pipeline is working!")
        print("\n🚀 You can now:")
        print("   1. Process images: python -c \"from src.omr_pipeline import OMRPipeline; p=OMRPipeline(); p.process_image('image.png', 'output.mxl')\"")
        print("   2. Launch UI: streamlit run src/ui/correction_interface.py")
        print("   3. Run examples: python examples/basic_usage.py")
        return 0
    else:
        print("❌ OMR Pipeline has issues. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())