#!/usr/bin/env python3
"""
OMR Pipeline - Getting Started Script
====================================

This script helps you get started with the OMR Pipeline quickly.
It checks your setup, provides sample usage, and guides you through first steps.
"""

import sys
import os
from pathlib import Path
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
    """Check if core dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ("numpy", "Essential for image processing"),
        ("opencv-python", "Computer vision operations"),
        ("PIL", "Image handling (Pillow)"),
    ]
    
    optional_packages = [
        ("streamlit", "Web UI for manual correction"),
        ("ultralytics", "YOLO object detection"),
        ("music21", "Music analysis tools"),
    ]
    
    available = []
    missing = []
    
    # Check required packages
    for package, description in required_packages:
        try:
            if package == "PIL":
                import PIL
            else:
                __import__(package)
            print(f"   ✓ {package} - {description}")
            available.append(package)
        except ImportError:
            print(f"   ✗ {package} - {description}")
            missing.append(package)
    
    # Check optional packages
    print("\n   Optional packages:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"   ✓ {package} - {description}")
            available.append(package)
        except ImportError:
            print(f"   ○ {package} - {description} (optional)")
    
    return len(missing) == 0, missing

def install_dependencies():
    """Install missing dependencies."""
    print("\n🔧 Installing dependencies...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("   ✓ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Installation failed: {e}")
        print("   Try running manually: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        print("   ✗ requirements.txt not found")
        print("   Make sure you're in the OMR directory")
        return False

def check_omr_pipeline():
    """Check if OMR pipeline can be imported."""
    print("\n🎼 Checking OMR Pipeline...")
    
    try:
        sys.path.insert(0, str(Path.cwd() / "src"))
        from omr_pipeline import OMRPipeline
        print("   ✓ OMR Pipeline imported successfully!")
        
        # Try to create pipeline instance
        pipeline = OMRPipeline()
        print("   ✓ OMR Pipeline instance created!")
        return True, pipeline
        
    except ImportError as e:
        print(f"   ✗ Failed to import OMR Pipeline: {e}")
        return False, None
    except Exception as e:
        print(f"   ✗ Failed to create pipeline: {e}")
        return False, None

def create_sample_image():
    """Create a simple sample sheet music image."""
    print("\n🖼️ Creating sample image...")
    
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        
        # Create a simple sheet music image
        width, height = 800, 600
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw staff lines
        staff_y_positions = [200, 220, 240, 260, 280]
        for y in staff_y_positions:
            draw.line([(50, y), (750, y)], fill='black', width=2)
        
        # Draw a simple treble clef (rough approximation)
        draw.ellipse([80, 180, 120, 220], outline='black', width=3)
        draw.line([100, 200, 100, 160], fill='black', width=3)
        
        # Draw some note heads
        note_positions = [(200, 200), (300, 240), (400, 220), (500, 260)]
        for x, y in note_positions:
            draw.ellipse([x-10, y-6, x+10, y+6], fill='black')
            # Add stems
            if y <= 240:  # Stem up
                draw.line([x+8, y, x+8, y-40], fill='black', width=2)
            else:  # Stem down
                draw.line([x-8, y, x-8, y+40], fill='black', width=2)
        
        # Save the image
        sample_path = Path("sample_sheet_music.png")
        image.save(sample_path)
        print(f"   ✓ Sample image created: {sample_path}")
        return sample_path
        
    except Exception as e:
        print(f"   ✗ Failed to create sample image: {e}")
        return None

def demonstrate_usage(pipeline, sample_image_path):
    """Demonstrate basic pipeline usage."""
    print("\n🚀 Demonstrating OMR Pipeline usage...")
    
    if not sample_image_path or not sample_image_path.exists():
        print("   ✗ No sample image available")
        return
    
    try:
        print(f"   Processing: {sample_image_path}")
        
        # Process the sample image
        result = pipeline.process_image(
            image_path=str(sample_image_path),
            output_path="sample_output.mxl"
        )
        
        if result:
            print("   ✓ Processing completed!")
            print(f"      - MusicXML: {result.get('musicxml_path', 'N/A')}")
            print(f"      - Quality Score: {result.get('quality_score', 0):.2f}")
            print(f"      - Processing Time: {result.get('processing_time', 0):.1f}s")
            
            # Show confidence data
            confidence = result.get('confidence_data', {})
            if confidence:
                print(f"      - Overall Confidence: {confidence.get('overall_confidence', 0):.2f}")
        else:
            print("   ⚠️ Processing completed but returned no result")
            
    except Exception as e:
        print(f"   ⚠️ Processing encountered an issue: {e}")
        print("   This is normal for the first run without trained models")

def show_next_steps():
    """Show next steps for the user."""
    print("\n📋 Next Steps:")
    print()
    print("1. 📖 Read the documentation:")
    print("   - USAGE_GUIDE.md - Comprehensive usage guide")
    print("   - README.md - Project overview and features")
    print("   - examples/ - Working code examples")
    print()
    print("2. 🎼 Try processing your own sheet music:")
    print("   python -c \"from src.omr_pipeline import OMRPipeline; p=OMRPipeline(); p.process_image('your_sheet.png', 'output.mxl')\"")
    print()
    print("3. 🖥️ Launch the web interface:")
    print("   streamlit run src/ui/correction_interface.py")
    print()
    print("4. 🧪 Run the test suite:")
    print("   pytest tests/ -v")
    print()
    print("5. 📊 Try the examples:")
    print("   python examples/basic_usage.py")
    print("   python examples/batch_processing.py")
    print()
    print("6. ⚙️ Customize configuration:")
    print("   - Edit config parameters for your specific use case")
    print("   - See USAGE_GUIDE.md for configuration options")

def main():
    """Main getting started workflow."""
    print("🎼 OMR Pipeline - Getting Started")
    print("=" * 50)
    
    # Check environment
    if not check_python_version():
        print("\n❌ Please upgrade Python to 3.8 or higher")
        return 1
    
    # Check current directory
    if not Path("src").exists() or not Path("requirements.txt").exists():
        print("\n❌ Please run this script from the OMR directory")
        print("   Make sure you can see 'src/' folder and 'requirements.txt'")
        return 1
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\n⚠️ Missing required packages: {', '.join(missing)}")
        install_choice = input("Install dependencies now? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_dependencies():
                print("\n❌ Failed to install dependencies")
                return 1
        else:
            print("\n❌ Dependencies required to continue")
            return 1
    
    # Check OMR pipeline
    pipeline_ok, pipeline = check_omr_pipeline()
    if not pipeline_ok:
        print("\n❌ OMR Pipeline not working properly")
        print("   Try installing dependencies: pip install -r requirements.txt")
        return 1
    
    # Create and process sample
    sample_image = create_sample_image()
    if pipeline and sample_image:
        demonstrate_usage(pipeline, sample_image)
    
    # Show next steps
    show_next_steps()
    
    print("\n✅ Setup complete! You're ready to use the OMR Pipeline.")
    print("   Start with the USAGE_GUIDE.md for detailed instructions.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())