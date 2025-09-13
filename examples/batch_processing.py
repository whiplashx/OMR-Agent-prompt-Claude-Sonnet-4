"""
Batch Processing Examples
========================

This file demonstrates batch processing capabilities of the OMR pipeline.
"""

import os
import json
import time
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.omr_pipeline import OMRPipeline

try:
    import cv2
    import numpy as np
except ImportError:
    print("Warning: OpenCV and NumPy not available. Using mock data for demonstration.")
    cv2 = None
    np = None


def create_sample_dataset():
    """Create a sample dataset for batch processing demonstration."""
    dataset_dir = Path(__file__).parent / "sample_dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    if cv2 is None or np is None:
        print("Skipping dataset creation - OpenCV/NumPy not available")
        return dataset_dir
    
    print("Creating sample dataset...")
    
    # Create 5 different synthetic sheet music images
    for i in range(5):
        # Create base image
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Add staff lines with slight variations
        staff_y_start = 150 + (i * 5)  # Slight vertical offset for each image
        staff_y_positions = [staff_y_start + j * 20 for j in range(5)]
        
        for y in staff_y_positions:
            cv2.line(image, (50, y), (750, y), (0, 0, 0), 2)
        
        # Add treble clef
        cv2.putText(image, "♪", (70, staff_y_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Add notes with different patterns for each image
        note_count = 4 + i  # Different number of notes per image
        for j in range(note_count):
            x = 120 + (j * 80)
            y = staff_y_positions[j % len(staff_y_positions)] - 10
            cv2.circle(image, (x, y), 8, (0, 0, 0), -1)
            
            # Add stems
            cv2.line(image, (x + 8, y), (x + 8, y - 40), (0, 0, 0), 2)
        
        # Add some variation to make each image unique
        if i % 2 == 0:
            # Add some sharps
            cv2.putText(image, "#", (100, staff_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save the image
        image_path = dataset_dir / f"sheet_music_{i+1:02d}.png"
        cv2.imwrite(str(image_path), image)
    
    print(f"Created {5} sample images in {dataset_dir}")
    return dataset_dir


def example_1_basic_batch_processing():
    """
    Example 1: Basic batch processing with default settings.
    """
    print("\nExample 1: Basic Batch Processing")
    print("=" * 50)
    
    # Create sample dataset
    dataset_dir = create_sample_dataset()
    output_dir = Path(__file__).parent / "output" / "batch_basic"
    
    # Initialize pipeline
    pipeline = OMRPipeline()
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        str(f) for f in dataset_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print("❌ No image files found in dataset directory")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Process batch
    start_time = time.time()
    results = pipeline.process_batch(image_files, str(output_dir))
    total_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\n📊 Batch Processing Results:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per image: {total_time/len(results):.2f}s")
    
    if successful > 0:
        avg_confidence = sum(
            r.confidence_scores.get('overall_confidence', 0) 
            for r in results if r.success
        ) / successful
        print(f"   Average confidence: {avg_confidence:.3f}")
    
    print(f"💾 Results saved to: {output_dir}")


def example_2_batch_with_intermediate_results():
    """
    Example 2: Batch processing with intermediate result saving.
    """
    print("\nExample 2: Batch Processing with Intermediate Results")
    print("=" * 50)
    
    # Create sample dataset
    dataset_dir = create_sample_dataset()
    output_dir = Path(__file__).parent / "output" / "batch_intermediate"
    
    # Custom configuration for detailed processing
    config = {
        'preprocessing': {
            'denoise_strength': 5,
            'contrast_enhancement': True,
            'skew_correction': True
        },
        'symbol_detection': {
            'confidence_threshold': 0.2,  # Lower threshold for more detections
        },
        'json_export': {
            'include_intermediate_results': True,
            'pretty_print': True
        }
    }
    
    pipeline = OMRPipeline(config)
    
    # Find image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        str(f) for f in dataset_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print("❌ No image files found")
        return
    
    print(f"📁 Processing {len(image_files)} images with intermediate result saving...")
    
    # Process with intermediate results
    results = pipeline.process_batch(
        image_files, 
        str(output_dir), 
        save_intermediate=True
    )
    
    # Analyze intermediate results
    print(f"\n📊 Processing completed with intermediate results:")
    
    for i, result in enumerate(results):
        if result.success:
            base_name = Path(result.image_path).stem
            intermediate_dir = output_dir / "intermediate" / base_name
            
            print(f"\n   📄 {base_name}:")
            print(f"      Processing time: {result.total_time:.2f}s")
            print(f"      Confidence: {result.confidence_scores.get('overall_confidence', 0):.3f}")
            print(f"      Quality: {result.quality_assessment.get('overall_score', 0):.3f}")
            
            if intermediate_dir.exists():
                files = list(intermediate_dir.iterdir())
                print(f"      Intermediate files: {len(files)}")
                for file in files:
                    print(f"        • {file.name}")
    
    print(f"\n💾 All results and intermediate files saved to: {output_dir}")


def example_3_batch_with_custom_processing():
    """
    Example 3: Custom batch processing with filtering and analysis.
    """
    print("\nExample 3: Custom Batch Processing with Analysis")
    print("=" * 50)
    
    # Create dataset
    dataset_dir = create_sample_dataset()
    output_dir = Path(__file__).parent / "output" / "batch_custom"
    
    pipeline = OMRPipeline()
    
    # Find image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        str(f) for f in dataset_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print("❌ No image files found")
        return
    
    # Process images individually for custom analysis
    all_results = []
    confidence_threshold = 0.5
    
    print(f"📁 Processing {len(image_files)} images with custom analysis...")
    
    for i, image_file in enumerate(image_files):
        print(f"\n🔄 Processing {i+1}/{len(image_files)}: {Path(image_file).name}")
        
        if cv2 is not None:
            image = cv2.imread(image_file)
            if image is None:
                print(f"   ❌ Could not load image")
                continue
                
            result = pipeline.process_image(image, image_file)
            all_results.append(result)
            
            if result.success:
                confidence = result.confidence_scores.get('overall_confidence', 0)
                quality = result.quality_assessment.get('overall_score', 0)
                
                print(f"   ✅ Success - Confidence: {confidence:.3f}, Quality: {quality:.3f}")
                
                # Custom analysis
                if confidence < confidence_threshold:
                    print(f"   ⚠️  Low confidence detected!")
                    
                    # Analyze low confidence symbols
                    low_conf_count = result.confidence_scores.get('low_confidence_count', 0)
                    total_symbols = result.confidence_scores.get('total_symbols', 0)
                    
                    if total_symbols > 0:
                        low_conf_ratio = low_conf_count / total_symbols
                        print(f"      Low confidence ratio: {low_conf_ratio:.2%}")
                
                # Save outputs for successful results
                base_name = Path(image_file).stem
                pipeline._save_results(result, str(output_dir), base_name, False)
                
            else:
                print(f"   ❌ Failed: {result.error_message}")
    
    # Generate comprehensive analysis
    print(f"\n📊 Comprehensive Batch Analysis:")
    
    successful_results = [r for r in all_results if r.success]
    failed_results = [r for r in all_results if not r.success]
    
    print(f"   Success rate: {len(successful_results)}/{len(all_results)} ({len(successful_results)/len(all_results)*100:.1f}%)")
    
    if successful_results:
        # Confidence analysis
        confidences = [r.confidence_scores.get('overall_confidence', 0) for r in successful_results]
        avg_confidence = sum(confidences) / len(confidences)
        high_conf_count = sum(1 for c in confidences if c >= 0.8)
        low_conf_count = sum(1 for c in confidences if c < 0.5)
        
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   High confidence (≥0.8): {high_conf_count}/{len(confidences)}")
        print(f"   Low confidence (<0.5): {low_conf_count}/{len(confidences)}")
        
        # Quality analysis
        qualities = [r.quality_assessment.get('overall_score', 0) for r in successful_results]
        avg_quality = sum(qualities) / len(qualities)
        print(f"   Average quality: {avg_quality:.3f}")
        
        # Performance analysis
        times = [r.total_time for r in successful_results]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"   Processing time - Avg: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s")
    
    # Save analysis report
    analysis_report = {
        'batch_summary': {
            'total_images': len(all_results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(all_results) if all_results else 0
        },
        'confidence_analysis': {
            'average': avg_confidence if successful_results else 0,
            'high_confidence_count': high_conf_count if successful_results else 0,
            'low_confidence_count': low_conf_count if successful_results else 0
        } if successful_results else {},
        'quality_analysis': {
            'average': avg_quality if successful_results else 0
        } if successful_results else {},
        'performance_analysis': {
            'average_time': avg_time if successful_results else 0,
            'min_time': min_time if successful_results else 0,
            'max_time': max_time if successful_results else 0
        } if successful_results else {},
        'failed_images': [
            {
                'path': r.image_path,
                'error': r.error_message
            } for r in failed_results
        ]
    }
    
    report_path = output_dir / "batch_analysis_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(analysis_report, f, indent=2)
    
    print(f"\n💾 Analysis report saved to: {report_path}")


def example_4_performance_comparison():
    """
    Example 4: Compare performance of different configurations.
    """
    print("\nExample 4: Performance Comparison")
    print("=" * 50)
    
    # Create dataset
    dataset_dir = create_sample_dataset()
    
    # Define different configurations to compare
    configs = {
        'fast': {
            'preprocessing': {
                'denoise_strength': 3,
                'contrast_enhancement': False,
                'skew_correction': False
            },
            'symbol_detection': {
                'confidence_threshold': 0.5,
                'model_size': 'small'
            }
        },
        'balanced': {
            'preprocessing': {
                'denoise_strength': 5,
                'contrast_enhancement': True,
                'skew_correction': True
            },
            'symbol_detection': {
                'confidence_threshold': 0.3,
                'model_size': 'medium'
            }
        },
        'quality': {
            'preprocessing': {
                'denoise_strength': 7,
                'contrast_enhancement': True,
                'skew_correction': True
            },
            'symbol_detection': {
                'confidence_threshold': 0.2,
                'model_size': 'large'
            },
            'musicxml_generation': {
                'divisions': 960,  # Higher precision
                'validate_output': True
            }
        }
    }
    
    # Find test images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        str(f) for f in dataset_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ][:3]  # Limit to 3 images for comparison
    
    if not image_files:
        print("❌ No image files found")
        return
    
    comparison_results = {}
    
    print(f"🏁 Comparing {len(configs)} configurations on {len(image_files)} images...")
    
    for config_name, config in configs.items():
        print(f"\n⚙️  Testing '{config_name}' configuration...")
        
        pipeline = OMRPipeline(config)
        start_time = time.time()
        
        config_results = []
        for image_file in image_files:
            if cv2 is not None:
                image = cv2.imread(image_file)
                if image is not None:
                    result = pipeline.process_image(image, image_file)
                    config_results.append(result)
        
        total_time = time.time() - start_time
        
        # Analyze results for this configuration
        successful = [r for r in config_results if r.success]
        
        if successful:
            avg_confidence = sum(r.confidence_scores.get('overall_confidence', 0) for r in successful) / len(successful)
            avg_quality = sum(r.quality_assessment.get('overall_score', 0) for r in successful) / len(successful)
            avg_processing_time = sum(r.total_time for r in successful) / len(successful)
        else:
            avg_confidence = 0
            avg_quality = 0
            avg_processing_time = 0
        
        comparison_results[config_name] = {
            'total_time': total_time,
            'successful_count': len(successful),
            'failed_count': len(config_results) - len(successful),
            'avg_confidence': avg_confidence,
            'avg_quality': avg_quality,
            'avg_processing_time': avg_processing_time
        }
        
        print(f"   Success rate: {len(successful)}/{len(config_results)}")
        print(f"   Avg confidence: {avg_confidence:.3f}")
        print(f"   Avg quality: {avg_quality:.3f}")
        print(f"   Avg time per image: {avg_processing_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
    
    # Generate comparison report
    print(f"\n📊 Performance Comparison Summary:")
    print(f"{'Config':<12} {'Success%':<10} {'Confidence':<12} {'Quality':<10} {'Time/img':<10}")
    print("-" * 60)
    
    for config_name, results in comparison_results.items():
        success_rate = results['successful_count'] / (results['successful_count'] + results['failed_count']) * 100
        print(f"{config_name:<12} {success_rate:<10.1f} {results['avg_confidence']:<12.3f} "
              f"{results['avg_quality']:<10.3f} {results['avg_processing_time']:<10.2f}s")
    
    # Save comparison report
    output_dir = Path(__file__).parent / "output" / "performance_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "performance_comparison.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n💾 Comparison report saved to: {output_dir}")


def main():
    """Run all batch processing examples."""
    print("🎼 OMR Pipeline Batch Processing Examples")
    print("=" * 60)
    
    try:
        example_1_basic_batch_processing()
        example_2_batch_with_intermediate_results()
        example_3_batch_with_custom_processing()
        example_4_performance_comparison()
        
        print("\n🎉 All batch processing examples completed!")
        print(f"📁 Check the output directories in: {Path(__file__).parent / 'output'}")
        
    except Exception as e:
        print(f"💥 An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()