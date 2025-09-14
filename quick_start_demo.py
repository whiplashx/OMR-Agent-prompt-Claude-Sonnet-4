#!/usr/bin/env python3
"""
OMR Pipeline Quick Start Demo
============================

This script demonstrates the most common ways to use the OMR Pipeline.
Run this to see working examples of different usage patterns.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_basic_usage():
    """Demonstrate basic OMR pipeline usage."""
    print("🎼 Demo 1: Basic Pipeline Usage")
    print("-" * 40)
    
    try:
        from omr_pipeline import OMRPipeline
        
        # Create pipeline
        pipeline = OMRPipeline()
        print("✅ Created OMR pipeline")
        
        # Create sample sheet music image
        image = create_sample_sheet_music()
        print("✅ Created sample sheet music image")
        
        # Process the image
        print("🔄 Processing image...")
        result = pipeline.process_image(image=image, image_path="demo_sheet.png")
        
        if result.success:
            print("✅ Processing successful!")
            print(f"   ⏱️ Total time: {result.total_time:.2f} seconds")
            print(f"   📊 Processing breakdown:")
            print(f"      - Preprocessing: {result.preprocessing_time:.2f}s")
            print(f"      - Staff detection: {result.staff_detection_time:.2f}s")
            print(f"      - Symbol detection: {result.symbol_detection_time:.2f}s")
            print(f"      - Music reconstruction: {result.music_reconstruction_time:.2f}s")
            print(f"      - Output generation: {result.output_generation_time:.2f}s")
        else:
            print(f"⚠️ Processing had issues: {result.error_message}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_custom_configuration():
    """Demonstrate custom configuration options."""
    print("\n🎼 Demo 2: Custom Configuration")
    print("-" * 40)
    
    try:
        from omr_pipeline import OMRPipeline
        
        # Configuration for high-quality processing
        config = {
            'apply_denoising': True,
            'apply_skew_correction': True,
            'confidence_threshold': 0.7,
            'staff_line_thickness': 2,
        }
        
        pipeline = OMRPipeline(config=config)
        print("✅ Created pipeline with custom configuration")
        print(f"   📋 Config: denoising={config['apply_denoising']}, threshold={config['confidence_threshold']}")
        
        # Process with custom settings
        image = create_sample_sheet_music()
        result = pipeline.process_image(image=image, image_path="demo_custom.png")
        
        print(f"✅ Custom processing completed in {result.total_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_result_analysis():
    """Demonstrate how to analyze results."""
    print("\n🎼 Demo 3: Result Analysis")
    print("-" * 40)
    
    try:
        from omr_pipeline import OMRPipeline
        
        pipeline = OMRPipeline()
        image = create_sample_sheet_music()
        result = pipeline.process_image(image=image, image_path="demo_analysis.png")
        
        print("📊 Analyzing OMR Results:")
        
        # Basic success check
        print(f"   ✅ Success: {result.success}")
        if not result.success:
            print(f"   ❌ Error: {result.error_message}")
            return False
        
        # Timing analysis
        print(f"   ⏱️ Performance:")
        print(f"      - Total: {result.total_time:.2f}s")
        print(f"      - Fastest step: {min(result.preprocessing_time, result.staff_detection_time, result.symbol_detection_time):.2f}s")
        print(f"      - Slowest step: {max(result.preprocessing_time, result.staff_detection_time, result.symbol_detection_time):.2f}s")
        
        # Content analysis
        if result.detected_staves:
            print(f"   🎵 Staff Analysis:")
            print(f"      - Detected {len(result.detected_staves)} staff systems")
        
        if result.detected_symbols:
            print(f"   🔍 Symbol Analysis:")
            print(f"      - Detected {len(result.detected_symbols)} symbols")
            
            # Show symbol types
            symbol_types = {}
            for symbol in result.detected_symbols:
                symbol_type = symbol.get('class', 'unknown')
                symbol_types[symbol_type] = symbol_types.get(symbol_type, 0) + 1
            
            print(f"      - Symbol types found:")
            for symbol_type, count in symbol_types.items():
                print(f"        * {symbol_type}: {count}")
        
        if result.musical_elements:
            print(f"   🎼 Musical Content:")
            measures = result.musical_elements.get('measures', [])
            voices = result.musical_elements.get('voices', [])
            print(f"      - Measures: {len(measures)}")
            print(f"      - Voices: {len(voices)}")
        
        # File outputs
        print(f"   📄 Generated Files:")
        if hasattr(result, 'musicxml_path') and result.musicxml_path:
            print(f"      - MusicXML: {result.musicxml_path}")
        if hasattr(result, 'confidence_json_path') and result.confidence_json_path:
            print(f"      - Confidence JSON: {result.confidence_json_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n🎼 Demo 4: Batch Processing")
    print("-" * 40)
    
    try:
        from omr_pipeline import OMRPipeline
        
        pipeline = OMRPipeline()
        
        # Create multiple sample images
        sample_images = []
        for i in range(3):
            image = create_sample_sheet_music(variation=i)
            sample_images.append(image)
        
        print(f"✅ Created {len(sample_images)} sample images")
        
        # Process each image
        results = []
        for i, image in enumerate(sample_images):
            print(f"   🔄 Processing image {i+1}/3...")
            result = pipeline.process_image(image=image, image_path=f"demo_batch_{i}.png")
            results.append(result)
        
        # Analyze batch results
        print("📊 Batch Results Summary:")
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.total_time for r in results)
        avg_time = total_time / len(results)
        
        print(f"   ✅ Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"   ⏱️ Total processing time: {total_time:.2f}s")
        print(f"   📈 Average time per image: {avg_time:.2f}s")
        
        # Individual results
        for i, result in enumerate(results):
            status = "✅" if result.success else "❌"
            symbols = len(result.detected_symbols or [])
            print(f"   {status} Image {i+1}: {result.total_time:.2f}s, {symbols} symbols")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_sample_sheet_music(variation=0):
    """Create a sample sheet music image with optional variations."""
    height, width = 400, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add staff lines
    staff_y = 150 + variation * 10  # Slight variation in staff position
    staff_positions = [staff_y + i*20 for i in range(5)]
    
    for y in staff_positions:
        if 0 <= y < height:
            image[y-1:y+1, 50:550] = 0
    
    # Add treble clef
    clef_x, clef_y = 80, staff_y + 40
    for dy in range(-15, 16):
        for dx in range(-10, 11):
            if dx*dx + dy*dy <= 100:
                if 0 <= clef_y+dy < height and 0 <= clef_x+dx < width:
                    image[clef_y+dy, clef_x+dx] = 0
    
    # Add notes with variation
    note_positions = [
        (180 + variation*20, staff_y + 0),   # Line positions vary by variation
        (250 + variation*15, staff_y + 20),
        (320 + variation*10, staff_y + 40),
        (390 + variation*5,  staff_y + 20),
    ]
    
    for x, y in note_positions:
        if 0 <= y < height and 0 <= x < width-20:
            # Note head
            for dy in range(-6, 7):
                for dx in range(-8, 9):
                    if dx*dx/64 + dy*dy/36 <= 1:
                        if 0 <= y+dy < height and 0 <= x+dx < width:
                            image[y+dy, x+dx] = 0
            
            # Note stem
            stem_height = 35
            if y <= staff_y + 20:  # Stem down
                for stem_y in range(y+6, min(y+6+stem_height, height)):
                    if 0 <= x+6 < width:
                        image[stem_y, x+6:x+8] = 0
            else:  # Stem up  
                for stem_y in range(max(y-6-stem_height, 0), y-6):
                    if 0 <= x-2 < width:
                        image[stem_y, x-2:x] = 0
    
    return image

def main():
    """Run all demos."""
    print("🎼 OMR Pipeline Quick Start Demos")
    print("=" * 50)
    print("This script demonstrates common usage patterns for the OMR Pipeline.")
    print("Each demo shows a different aspect of the system.\n")
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Custom Configuration", demo_custom_configuration),
        ("Result Analysis", demo_result_analysis),
        ("Batch Processing", demo_batch_processing),
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Demo '{name}' failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Demo Results Summary:")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\n🎯 Overall: {successful}/{len(results)} demos successful")
    
    if successful == len(results):
        print("\n🎉 All demos completed successfully!")
        print("🚀 Your OMR Pipeline is ready to use!")
        print("\n📚 Next steps:")
        print("   1. Try with your own sheet music images")
        print("   2. Launch the web interface: streamlit run src/ui/correction_interface.py")
        print("   3. Read HOW_TO_USE.md for comprehensive usage guide")
        print("   4. Explore the examples/ directory for more advanced usage")
    else:
        print("\n⚠️ Some demos had issues - check the error messages above")
        print("💡 Try running: python test_installation.py")
    
    return 0 if successful == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())