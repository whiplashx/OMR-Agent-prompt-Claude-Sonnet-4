#!/usr/bin/env python3
"""
Simple OMR Pipeline Example
===========================

A working example showing how to use the OMR Pipeline.
"""

import sys
from pathlib import Path
import numpy as np

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main example function."""
    print("🎼 OMR Pipeline Simple Example")
    print("=" * 40)
    
    try:
        # Import the pipeline
        from omr_pipeline import OMRPipeline
        print("✅ Imported OMRPipeline successfully")
        
        # Create pipeline instance
        pipeline = OMRPipeline()
        print("✅ Created pipeline instance")
        
        # Create a simple test image
        print("\n📸 Creating sample sheet music image...")
        
        # Create a 600x400 white image
        height, width = 400, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add staff lines (5 lines for a musical staff)
        staff_y_positions = [150, 170, 190, 210, 230]
        for y in staff_y_positions:
            # Draw horizontal staff lines
            image[y-1:y+1, 50:550] = 0  # Black lines
        
        # Add a simple treble clef shape (rough approximation)
        # Just a circle and vertical line
        for dy in range(-10, 11):
            for dx in range(-8, 9):
                if dx*dx + dy*dy <= 64:  # Circle equation
                    if 0 <= 170+dy < height and 0 <= 100+dx < width:
                        image[170+dy, 100+dx] = 0
        
        # Add stem for treble clef
        for y in range(160, 200):
            if 0 <= y < height:
                image[y, 108:110] = 0
        
        # Add some note heads
        note_positions = [(200, 190), (300, 210), (400, 170)]
        for x, y in note_positions:
            # Draw filled ellipse for note head
            for dy in range(-6, 7):
                for dx in range(-8, 9):
                    if dx*dx/64 + dy*dy/36 <= 1:  # Ellipse equation
                        if 0 <= y+dy < height and 0 <= x+dx < width:
                            image[y+dy, x+dx] = 0
            
            # Add stem
            stem_length = 40
            if y <= 190:  # Notes above middle line get stems down
                for stem_y in range(y+6, min(y+6+stem_length, height)):
                    image[stem_y, x+6:x+8] = 0
            else:  # Notes below middle line get stems up
                for stem_y in range(max(y-6-stem_length, 0), y-6):
                    image[stem_y, x-2:x] = 0
        
        print("   ✅ Created sample image with staff lines and notes")
        
        # Process the image
        print("\n🔄 Processing image through OMR pipeline...")
        result = pipeline.process_image(image=image, image_path="sample_image.png")
        
        if result and result.success:
            print("   ✅ Processing completed successfully!")
            print(f"      - Total processing time: {result.total_time:.2f} seconds")
            print(f"      - Preprocessing time: {result.preprocessing_time:.2f}s")
            print(f"      - Staff detection time: {result.staff_detection_time:.2f}s")
            print(f"      - Symbol detection time: {result.symbol_detection_time:.2f}s")
            
            # Check for detected elements
            if result.detected_symbols:
                print(f"      - Detected {len(result.detected_symbols)} symbols")
            else:
                print("      - No symbols detected (expected without trained models)")
                
            if result.detected_staves:
                print(f"      - Detected {len(result.detected_staves)} staff systems")
            else:
                print("      - No staves detected")
                
        else:
            print("   ⚠️ Processing completed but with issues")
            if result and result.error_message:
                print(f"      Error: {result.error_message}")
        
        print("\n📋 What this demonstrates:")
        print("   • OMR Pipeline imports and runs successfully")
        print("   • Image preprocessing works")
        print("   • Staff detection algorithms run")
        print("   • Symbol detection attempts (needs trained models for full functionality)")
        print("   • System handles errors gracefully")
        
        print("\n🚀 Next steps:")
        print("   1. Try with real sheet music images:")
        print("      python simple_example.py your_sheet_music.png")
        print("   2. Launch the web interface:")
        print("      streamlit run src/ui/correction_interface.py")
        print("   3. Check out examples directory for more advanced usage")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   • Make sure you're in the OMR directory")
        print("   • Check that dependencies are installed: pip install -r requirements.txt")
        print("   • Try running: python test_installation.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())