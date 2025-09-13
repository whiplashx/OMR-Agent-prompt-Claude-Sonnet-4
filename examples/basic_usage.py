"""
Basic OMR Pipeline Usage Examples
================================

This file demonstrates basic usage of the OMR pipeline for common tasks.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.omr_pipeline import OMRPipeline


def example_1_basic_image_processing():
    """
    Example 1: Basic image processing with minimal configuration.
    """
    print("Example 1: Basic Image Processing")
    print("=" * 50)
    
    # Initialize pipeline with default configuration
    pipeline = OMRPipeline()
    
    # For demonstration, create a synthetic test image
    test_image = create_test_sheet_music()
    
    # Process the image
    results = pipeline.process_image(test_image, "synthetic_test_image")
    
    if results.success:
        print(f"✅ Processing successful!")
        print(f"⏱️  Total time: {results.total_time:.2f} seconds")
        print(f"🎼 Detected {len(results.detected_staves)} staves")
        print(f"🎵 Detected {len(results.detected_symbols)} symbols")
        print(f"📊 Overall confidence: {results.confidence_scores['overall_confidence']:.3f}")
        print(f"⭐ Quality score: {results.quality_assessment['overall_score']:.3f}")
        
        # Save outputs to example directory
        output_dir = Path(__file__).parent / "output" / "example1"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline.save_musicxml(results.musicxml_content, str(output_dir / "basic_output.mxl"))
        pipeline.save_json_report(results.json_report, str(output_dir / "basic_report.json"))
        
        print(f"💾 Outputs saved to: {output_dir}")
    else:
        print(f"❌ Processing failed: {results.error_message}")
    
    print()


def example_2_custom_configuration():
    """
    Example 2: Using custom configuration for specialized processing.
    """
    print("Example 2: Custom Configuration")
    print("=" * 50)
    
    # Define custom configuration for high-quality processing
    custom_config = {
        'preprocessing': {
            'denoise_strength': 7,  # Stronger denoising
            'contrast_enhancement': True,
            'skew_correction': True,
            'binarization_method': 'adaptive'
        },
        'staff_detection': {
            'line_thickness_range': (1, 3),  # Thinner lines
            'min_line_length': 150,           # Longer minimum length
            'angle_tolerance': 1.5            # Stricter angle tolerance
        },
        'symbol_detection': {
            'confidence_threshold': 0.4,     # Higher confidence threshold
            'nms_threshold': 0.3,             # More aggressive NMS
            'model_size': 'large'             # Use larger model if available
        },
        'music_reconstruction': {
            'voice_separation_threshold': 25,
            'measure_grouping': True,
            'rhythm_quantization': True
        },
        'musicxml_generation': {
            'divisions': 480,                 # High-precision timing
            'validate_output': True,
            'include_layout': True            # Include layout information
        }
    }
    
    # Initialize pipeline with custom configuration
    pipeline = OMRPipeline(custom_config)
    
    # Create test image with more complex content
    test_image = create_complex_test_sheet_music()
    
    # Process the image
    results = pipeline.process_image(test_image, "custom_config_test")
    
    if results.success:
        print(f"✅ High-quality processing successful!")
        print(f"⏱️  Total time: {results.total_time:.2f} seconds")
        
        # Detailed timing breakdown
        print("\n📈 Processing stage timings:")
        print(f"   Preprocessing: {results.preprocessing_time:.2f}s")
        print(f"   Staff detection: {results.staff_detection_time:.2f}s") 
        print(f"   Symbol detection: {results.symbol_detection_time:.2f}s")
        print(f"   Music reconstruction: {results.music_reconstruction_time:.2f}s")
        print(f"   Output generation: {results.output_generation_time:.2f}s")
        
        # Confidence analysis
        conf = results.confidence_scores
        print(f"\n📊 Confidence analysis:")
        print(f"   Overall: {conf['overall_confidence']:.3f}")
        print(f"   High confidence symbols: {conf['high_confidence_count']}")
        print(f"   Low confidence symbols: {conf['low_confidence_count']}")
        
        # Save outputs
        output_dir = Path(__file__).parent / "output" / "example2"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline.save_musicxml(results.musicxml_content, str(output_dir / "custom_output.mxl"))
        pipeline.save_json_report(results.json_report, str(output_dir / "custom_report.json"))
        
        # Also save configuration for reference
        with open(output_dir / "config_used.json", 'w') as f:
            json.dump(custom_config, f, indent=2)
        
        print(f"💾 Outputs and config saved to: {output_dir}")
    else:
        print(f"❌ Processing failed: {results.error_message}")
    
    print()


def example_3_quality_assessment():
    """
    Example 3: Demonstrating quality assessment and recommendations.
    """
    print("Example 3: Quality Assessment and Recommendations")
    print("=" * 50)
    
    pipeline = OMRPipeline()
    
    # Process different quality images to show quality assessment
    test_images = [
        ("high_quality", create_high_quality_test_image()),
        ("medium_quality", create_medium_quality_test_image()),
        ("low_quality", create_low_quality_test_image())
    ]
    
    for image_name, test_image in test_images:
        print(f"\n🔍 Processing {image_name} image:")
        
        results = pipeline.process_image(test_image, f"{image_name}_test")
        
        if results.success:
            quality = results.quality_assessment
            
            print(f"   Overall quality score: {quality['overall_score']:.3f}")
            print(f"   Component scores:")
            for component, score in quality['component_scores'].items():
                print(f"     {component}: {score:.3f}")
            
            if quality['issues_detected']:
                print(f"   ⚠️  Issues detected:")
                for issue in quality['issues_detected']:
                    print(f"     • {issue}")
            
            if quality['recommendations']:
                print(f"   💡 Recommendations:")
                for rec in quality['recommendations']:
                    print(f"     • {rec}")
        else:
            print(f"   ❌ Processing failed: {results.error_message}")
    
    print()


def example_4_error_handling():
    """
    Example 4: Robust error handling and graceful degradation.
    """
    print("Example 4: Error Handling and Graceful Degradation")
    print("=" * 50)
    
    pipeline = OMRPipeline()
    
    # Test with various problematic inputs
    test_cases = [
        ("empty_image", np.zeros((100, 100, 3), dtype=np.uint8)),
        ("too_small_image", np.ones((50, 50, 3), dtype=np.uint8) * 255),
        ("too_large_image", np.ones((5000, 5000, 3), dtype=np.uint8) * 255),
        ("corrupted_data", None)
    ]
    
    for test_name, test_image in test_cases:
        print(f"\n🧪 Testing {test_name}:")
        
        try:
            if test_image is not None:
                results = pipeline.process_image(test_image, test_name)
                
                if results.success:
                    print(f"   ✅ Unexpectedly successful (might indicate robustness)")
                    print(f"   Quality score: {results.quality_assessment.get('overall_score', 'N/A')}")
                else:
                    print(f"   ⚠️  Gracefully failed: {results.error_message}")
                    print(f"   Processing time: {results.total_time:.2f}s")
            else:
                # This should raise an exception
                results = pipeline.process_image(test_image, test_name)
                
        except Exception as e:
            print(f"   ❌ Exception handled: {type(e).__name__}: {e}")
    
    print()


def create_test_sheet_music():
    """Create a simple synthetic sheet music image for testing."""
    # Create white background
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add staff lines
    staff_y_positions = [150, 170, 190, 210, 230]
    for y in staff_y_positions:
        cv2.line(image, (50, y), (750, y), (0, 0, 0), 2)
    
    # Add some note heads
    note_positions = [(120, 140), (180, 160), (240, 180), (300, 200), (360, 160)]
    for x, y in note_positions:
        cv2.circle(image, (x, y), 8, (0, 0, 0), -1)
    
    # Add treble clef (simplified)
    cv2.putText(image, "♪", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    return image


def create_complex_test_sheet_music():
    """Create a more complex synthetic sheet music image."""
    # Create larger white background
    image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # Add two staves
    for staff_offset in [0, 300]:
        staff_y_positions = [150 + staff_offset, 170 + staff_offset, 190 + staff_offset, 
                           210 + staff_offset, 230 + staff_offset]
        for y in staff_y_positions:
            cv2.line(image, (80, y), (1100, y), (0, 0, 0), 2)
        
        # Add clef
        cv2.putText(image, "♪", (90, 200 + staff_offset), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Add more notes with varying positions
        note_positions = [(150, 140 + staff_offset), (200, 160 + staff_offset), 
                         (250, 180 + staff_offset), (300, 200 + staff_offset),
                         (350, 180 + staff_offset), (400, 160 + staff_offset)]
        for x, y in note_positions:
            cv2.circle(image, (x, y), 8, (0, 0, 0), -1)
            # Add stems
            cv2.line(image, (x + 8, y), (x + 8, y - 40), (0, 0, 0), 2)
    
    return image


def create_high_quality_test_image():
    """Create a high-quality test image."""
    return create_complex_test_sheet_music()


def create_medium_quality_test_image():
    """Create a medium-quality test image with some noise."""
    image = create_test_sheet_music()
    
    # Add some noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image


def create_low_quality_test_image():
    """Create a low-quality test image with significant issues."""
    image = create_test_sheet_music()
    
    # Add heavy noise
    noise = np.random.normal(0, 50, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Blur the image
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Add some skew
    height, width = image.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[10, 5], [width-5, 10], [5, height-10], [width-10, height-5]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, matrix, (width, height))
    
    return image


def main():
    """Run all examples."""
    print("🎼 OMR Pipeline Basic Usage Examples")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run all examples
        example_1_basic_image_processing()
        example_2_custom_configuration()
        example_3_quality_assessment()
        example_4_error_handling()
        
        print("🎉 All examples completed successfully!")
        print(f"📁 Check the output directory: {output_dir}")
        
    except Exception as e:
        print(f"💥 An error occurred while running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()