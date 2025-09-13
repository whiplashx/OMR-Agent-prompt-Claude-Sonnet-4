"""
Evaluation Examples and Benchmarking
====================================

This file demonstrates evaluation capabilities of the OMR system.
"""

import json
import sys
from pathlib import Path
import time

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import OMREvaluator, EvaluationRunner

try:
    import cv2
    import numpy as np
except ImportError:
    print("Warning: OpenCV and NumPy not available. Using mock data for demonstration.")
    cv2 = None
    np = None


def create_sample_ground_truth():
    """
    Create sample ground truth data for evaluation demonstration.
    """
    return {
        'symbols': [
            {
                'class_name': 'quarter_note',
                'bbox': (120, 135, 16, 20),
                'pitch': 'C4',
                'confidence': 1.0
            },
            {
                'class_name': 'quarter_note',
                'bbox': (180, 155, 16, 20),
                'pitch': 'D4',
                'confidence': 1.0
            },
            {
                'class_name': 'quarter_note',
                'bbox': (240, 175, 16, 20),
                'pitch': 'E4',
                'confidence': 1.0
            },
            {
                'class_name': 'treble_clef',
                'bbox': (70, 180, 25, 50),
                'confidence': 1.0
            }
        ],
        'musical_elements': {
            'staves': [
                {
                    'elements': [
                        {
                            'element_type': 'note',
                            'pitch': 'C4',
                            'duration': 1.0,
                            'x_position': 128,
                            'y_position': 145
                        },
                        {
                            'element_type': 'note',
                            'pitch': 'D4',
                            'duration': 1.0,
                            'x_position': 188,
                            'y_position': 165
                        },
                        {
                            'element_type': 'note',
                            'pitch': 'E4',
                            'duration': 1.0,
                            'x_position': 248,
                            'y_position': 185
                        }
                    ],
                    'key_signature': {'sharps': 0, 'key': 'C'},
                    'time_signature': {'numerator': 4, 'denominator': 4}
                }
            ]
        }
    }


def create_sample_predictions():
    """
    Create sample prediction data with various accuracy levels.
    """
    from src.detection.symbol_detector import DetectedSymbol
    
    # High accuracy predictions
    high_accuracy_symbols = [
        DetectedSymbol(
            class_name='quarter_note',
            confidence=0.95,
            bbox=(118, 133, 18, 22),
            center=(127, 144),
            pitch='C4'
        ),
        DetectedSymbol(
            class_name='quarter_note',
            confidence=0.92,
            bbox=(178, 153, 18, 22),
            center=(187, 164),
            pitch='D4'
        ),
        DetectedSymbol(
            class_name='treble_clef',
            confidence=0.98,
            bbox=(68, 178, 27, 52),
            center=(81, 204),
            pitch=None
        )
    ]
    
    # Medium accuracy predictions (one missing, one misclassified)
    medium_accuracy_symbols = [
        DetectedSymbol(
            class_name='quarter_note',
            confidence=0.85,
            bbox=(115, 130, 20, 25),
            center=(125, 142),
            pitch='C4'
        ),
        DetectedSymbol(
            class_name='eighth_note',  # Misclassified
            confidence=0.72,
            bbox=(175, 150, 20, 25),
            center=(185, 162),
            pitch='D4'
        ),
        DetectedSymbol(
            class_name='treble_clef',
            confidence=0.88,
            bbox=(65, 175, 30, 55),
            center=(80, 202),
            pitch=None
        )
        # Missing the third quarter note
    ]
    
    # Low accuracy predictions (multiple errors)
    low_accuracy_symbols = [
        DetectedSymbol(
            class_name='quarter_note',
            confidence=0.45,
            bbox=(110, 125, 25, 30),
            center=(122, 140),
            pitch='C4'
        ),
        DetectedSymbol(
            class_name='quarter_rest',  # Wrong class
            confidence=0.38,
            bbox=(170, 145, 25, 30),
            center=(182, 160),
            pitch=None
        ),
        DetectedSymbol(
            class_name='treble_clef',
            confidence=0.62,
            bbox=(60, 170, 35, 60),
            center=(77, 200),
            pitch=None
        ),
        DetectedSymbol(
            class_name='quarter_note',  # False positive
            confidence=0.33,
            bbox=(300, 200, 20, 25),
            center=(310, 212),
            pitch='F4'
        )
    ]
    
    return {
        'high_accuracy': high_accuracy_symbols,
        'medium_accuracy': medium_accuracy_symbols,
        'low_accuracy': low_accuracy_symbols
    }


def example_1_basic_evaluation():
    """
    Example 1: Basic symbol-level evaluation.
    """
    print("Example 1: Basic Symbol-Level Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = OMREvaluator()
    
    # Get sample data
    ground_truth = create_sample_ground_truth()
    predictions = create_sample_predictions()
    
    # Evaluate each accuracy level
    for accuracy_level, symbols in predictions.items():
        print(f"\n📊 Evaluating {accuracy_level} predictions:")
        
        # Prepare data for evaluation
        predicted_results = {'symbols': symbols}
        
        # Run evaluation
        results = evaluator.evaluate_full_pipeline(predicted_results, ground_truth)
        
        # Display results
        if 'symbol_level' in results:
            for result in results['symbol_level']:
                print(f"   {result.metric_name}: {result.score:.3f}")
                
                # Show detailed metrics for detection
                if result.metric_name == 'detection' and 'details' in result.details:
                    details = result.details
                    print(f"     Precision: {details.get('precision', 0):.3f}")
                    print(f"     Recall: {details.get('recall', 0):.3f}")
                    print(f"     F1-score: {details.get('f1_score', 0):.3f}")
    
    print()


def example_2_semantic_evaluation():
    """
    Example 2: Semantic-level musical evaluation.
    """
    print("Example 2: Semantic-Level Musical Evaluation")
    print("=" * 50)
    
    evaluator = OMREvaluator()
    ground_truth = create_sample_ground_truth()
    
    # Create different semantic accuracy scenarios
    scenarios = {
        'perfect_semantics': {
            'musical_elements': ground_truth['musical_elements']
        },
        'pitch_errors': {
            'musical_elements': {
                'staves': [
                    {
                        'elements': [
                            {
                                'element_type': 'note',
                                'pitch': 'D4',  # Wrong pitch
                                'duration': 1.0,
                                'x_position': 128,
                                'y_position': 145
                            },
                            {
                                'element_type': 'note',
                                'pitch': 'D4',
                                'duration': 1.0,
                                'x_position': 188,
                                'y_position': 165
                            },
                            {
                                'element_type': 'note',
                                'pitch': 'E4',
                                'duration': 1.0,
                                'x_position': 248,
                                'y_position': 185
                            }
                        ],
                        'key_signature': {'sharps': 0, 'key': 'C'},
                        'time_signature': {'numerator': 4, 'denominator': 4}
                    }
                ]
            }
        },
        'rhythm_errors': {
            'musical_elements': {
                'staves': [
                    {
                        'elements': [
                            {
                                'element_type': 'note',
                                'pitch': 'C4',
                                'duration': 0.5,  # Wrong duration
                                'x_position': 128,
                                'y_position': 145
                            },
                            {
                                'element_type': 'note',
                                'pitch': 'D4',
                                'duration': 1.5,  # Wrong duration
                                'x_position': 188,
                                'y_position': 165
                            },
                            {
                                'element_type': 'note',
                                'pitch': 'E4',
                                'duration': 1.0,
                                'x_position': 248,
                                'y_position': 185
                            }
                        ],
                        'key_signature': {'sharps': 0, 'key': 'C'},
                        'time_signature': {'numerator': 4, 'denominator': 4}
                    }
                ]
            }
        }
    }
    
    print("🎵 Semantic Evaluation Results:")
    
    for scenario_name, predicted_results in scenarios.items():
        print(f"\n📝 Scenario: {scenario_name}")
        
        results = evaluator.evaluate_full_pipeline(predicted_results, ground_truth)
        
        if 'semantic_level' in results:
            for result in results['semantic_level']:
                print(f"   {result.metric_name}: {result.score:.3f}")
    
    print()


def example_3_comprehensive_evaluation_report():
    """
    Example 3: Generate comprehensive evaluation report.
    """
    print("Example 3: Comprehensive Evaluation Report")
    print("=" * 50)
    
    evaluator = OMREvaluator()
    ground_truth = create_sample_ground_truth()
    predictions = create_sample_predictions()
    
    # Use medium accuracy predictions for realistic report
    predicted_results = {
        'symbols': predictions['medium_accuracy'],
        'musical_elements': {
            'staves': [
                {
                    'elements': [
                        {
                            'element_type': 'note',
                            'pitch': 'C4',
                            'duration': 1.0,
                            'x_position': 125,
                            'y_position': 142
                        },
                        {
                            'element_type': 'note',
                            'pitch': 'D4',
                            'duration': 1.0,
                            'x_position': 185,
                            'y_position': 162
                        }
                        # Missing third note
                    ],
                    'key_signature': {'sharps': 0, 'key': 'C'},
                    'time_signature': {'numerator': 4, 'denominator': 4}
                }
            ]
        }
    }
    
    # Run evaluation
    evaluation_results = evaluator.evaluate_full_pipeline(predicted_results, ground_truth)
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(evaluation_results)
    
    print("📋 Comprehensive Evaluation Report:")
    print("=" * 40)
    
    # Summary scores
    if 'summary' in report:
        print("\n📊 Summary Scores:")
        for metric, score in report['summary'].items():
            print(f"   {metric}: {score:.3f}")
    
    # Detailed results
    print("\n📈 Detailed Results:")
    for category, results in evaluation_results.items():
        print(f"\n   {category.replace('_', ' ').title()}:")
        for result in results:
            print(f"     {result.metric_name}: {result.score:.3f}")
            
            # Show per-class scores if available
            if result.per_class_scores:
                print(f"       Per-class performance:")
                for class_name, scores in result.per_class_scores.items():
                    if isinstance(scores, dict) and 'f1_score' in scores:
                        print(f"         {class_name}: F1={scores['f1_score']:.3f}")
    
    # Recommendations
    if 'recommendations' in report:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    # Save report
    output_dir = Path(__file__).parent / "output" / "evaluation_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed report
    with open(output_dir / "comprehensive_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Report saved to: {output_dir / 'comprehensive_report.json'}")
    print()


def example_4_dataset_evaluation():
    """
    Example 4: Dataset-wide evaluation simulation.
    """
    print("Example 4: Dataset-Wide Evaluation")
    print("=" * 50)
    
    # Simulate dataset evaluation
    runner = EvaluationRunner()
    
    # Create mock dataset results
    dataset_results = []
    image_names = ['sheet1.png', 'sheet2.png', 'sheet3.png', 'sheet4.png', 'sheet5.png']
    
    ground_truth = create_sample_ground_truth()
    predictions = create_sample_predictions()
    
    print("🗂️  Simulating dataset evaluation...")
    
    for i, image_name in enumerate(image_names):
        # Vary the prediction quality across the dataset
        if i < 2:
            symbols = predictions['high_accuracy']
            expected_score = 0.9
        elif i < 4:
            symbols = predictions['medium_accuracy']
            expected_score = 0.7
        else:
            symbols = predictions['low_accuracy']
            expected_score = 0.4
        
        predicted_results = {'symbols': symbols}
        
        # Simulate evaluation
        evaluator = OMREvaluator()
        evaluation_results = evaluator.evaluate_full_pipeline(predicted_results, ground_truth)
        
        # Extract scores
        symbol_scores = []
        if 'symbol_level' in evaluation_results:
            for result in evaluation_results['symbol_level']:
                symbol_scores.append(result.score)
        
        avg_score = sum(symbol_scores) / len(symbol_scores) if symbol_scores else 0
        
        dataset_results.append({
            'image_name': image_name,
            'scores': symbol_scores,
            'average_score': avg_score,
            'expected_score': expected_score
        })
        
        print(f"   📄 {image_name}: Score {avg_score:.3f} (expected {expected_score:.3f})")
    
    # Calculate dataset statistics
    all_scores = [r['average_score'] for r in dataset_results]
    
    print(f"\n📊 Dataset Evaluation Summary:")
    print(f"   Total images: {len(dataset_results)}")
    print(f"   Average score: {sum(all_scores)/len(all_scores):.3f}")
    print(f"   Best score: {max(all_scores):.3f}")
    print(f"   Worst score: {min(all_scores):.3f}")
    print(f"   Score variance: {np.var(all_scores) if np is not None else 'N/A'}")
    
    # Identify problematic images
    threshold = 0.6
    problematic_images = [r for r in dataset_results if r['average_score'] < threshold]
    
    if problematic_images:
        print(f"\n⚠️  Images below quality threshold ({threshold}):")
        for result in problematic_images:
            print(f"   📉 {result['image_name']}: {result['average_score']:.3f}")
    
    # Save dataset evaluation results
    output_dir = Path(__file__).parent / "output" / "dataset_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_summary = {
        'dataset_statistics': {
            'total_images': len(dataset_results),
            'average_score': sum(all_scores) / len(all_scores),
            'best_score': max(all_scores),
            'worst_score': min(all_scores),
            'quality_threshold': threshold,
            'images_below_threshold': len(problematic_images)
        },
        'individual_results': dataset_results,
        'problematic_images': [
            {
                'name': r['image_name'],
                'score': r['average_score'],
                'recommendations': [
                    "Review image quality",
                    "Check preprocessing parameters",
                    "Verify ground truth annotations"
                ]
            } for r in problematic_images
        ]
    }
    
    with open(output_dir / "dataset_evaluation_summary.json", 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    
    print(f"\n💾 Dataset evaluation summary saved to: {output_dir}")
    print()


def example_5_custom_metrics():
    """
    Example 5: Implementing and using custom evaluation metrics.
    """
    print("Example 5: Custom Evaluation Metrics")
    print("=" * 50)
    
    # Define a custom metric for note stem detection accuracy
    def evaluate_note_stem_accuracy(predicted_symbols, ground_truth_symbols):
        """Custom metric for evaluating note stem detection."""
        
        # Count notes with stems in ground truth
        gt_notes_with_stems = [
            s for s in ground_truth_symbols 
            if s.get('class_name', '').endswith('_note') and s.get('class_name') != 'whole_note'
        ]
        
        # Count correctly detected note stems in predictions
        correct_stems = 0
        
        for gt_note in gt_notes_with_stems:
            # Find matching predicted note (simplified matching)
            for pred_symbol in predicted_symbols:
                if (hasattr(pred_symbol, 'class_name') and 
                    pred_symbol.class_name == gt_note.get('class_name')):
                    # Check if stem was detected (simplified - assume detected if note detected)
                    correct_stems += 1
                    break
        
        accuracy = correct_stems / len(gt_notes_with_stems) if gt_notes_with_stems else 0
        return accuracy
    
    # Define a custom musical coherence metric
    def evaluate_musical_coherence(musical_elements):
        """Custom metric for evaluating musical coherence."""
        
        coherence_score = 1.0
        issues = []
        
        # Check for basic musical rules
        staves = musical_elements.get('staves', [])
        
        for staff in staves:
            elements = staff.get('elements', [])
            
            # Check time signature consistency
            time_sig = staff.get('time_signature', {})
            if time_sig:
                expected_duration = time_sig.get('numerator', 4)
                
                # Simple coherence check: total duration should be reasonable
                total_duration = sum(
                    elem.get('duration', 1.0) for elem in elements 
                    if elem.get('element_type') == 'note'
                )
                
                if total_duration > expected_duration * 2:  # Allow some flexibility
                    coherence_score *= 0.8
                    issues.append("Excessive note duration in measure")
        
        return {
            'coherence_score': coherence_score,
            'issues': issues
        }
    
    print("🔧 Implementing Custom Metrics:")
    
    # Test custom metrics
    ground_truth = create_sample_ground_truth()
    predictions = create_sample_predictions()
    
    for accuracy_level, symbols in predictions.items():
        print(f"\n📊 {accuracy_level} predictions:")
        
        # Note stem accuracy
        stem_accuracy = evaluate_note_stem_accuracy(symbols, ground_truth['symbols'])
        print(f"   Note stem accuracy: {stem_accuracy:.3f}")
        
        # Musical coherence (using ground truth musical elements as example)
        coherence_result = evaluate_musical_coherence(ground_truth['musical_elements'])
        print(f"   Musical coherence: {coherence_result['coherence_score']:.3f}")
        
        if coherence_result['issues']:
            print(f"   Coherence issues:")
            for issue in coherence_result['issues']:
                print(f"     • {issue}")
    
    # Show how to integrate custom metrics
    print(f"\n🔗 Integration with Main Evaluator:")
    print(f"   1. Extend OMREvaluator class")
    print(f"   2. Add custom metric methods")
    print(f"   3. Include in evaluation pipeline")
    print(f"   4. Generate reports with custom metrics")
    
    # Save custom metrics example
    output_dir = Path(__file__).parent / "output" / "custom_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    custom_metrics_example = {
        'custom_metrics': {
            'note_stem_accuracy': {
                'description': 'Accuracy of note stem detection',
                'implementation': 'evaluate_note_stem_accuracy()',
                'results': {
                    accuracy_level: evaluate_note_stem_accuracy(
                        symbols, ground_truth['symbols']
                    ) for accuracy_level, symbols in predictions.items()
                }
            },
            'musical_coherence': {
                'description': 'Coherence of musical structure',
                'implementation': 'evaluate_musical_coherence()',
                'results': evaluate_musical_coherence(ground_truth['musical_elements'])
            }
        }
    }
    
    with open(output_dir / "custom_metrics_example.json", 'w') as f:
        json.dump(custom_metrics_example, f, indent=2)
    
    print(f"\n💾 Custom metrics example saved to: {output_dir}")
    print()


def main():
    """Run all evaluation examples."""
    print("📊 OMR Evaluation Examples and Benchmarking")
    print("=" * 60)
    
    try:
        example_1_basic_evaluation()
        example_2_semantic_evaluation()
        example_3_comprehensive_evaluation_report()
        example_4_dataset_evaluation()
        example_5_custom_metrics()
        
        print("🎉 All evaluation examples completed!")
        print(f"📁 Check the output directories in: {Path(__file__).parent / 'output'}")
        
        # Provide guidance for real-world usage
        print(f"\n📋 Real-World Evaluation Workflow:")
        print(f"   1. Prepare ground truth annotations")
        print(f"   2. Process images with OMR pipeline")
        print(f"   3. Run evaluation: python -m src.evaluation.evaluation_cli")
        print(f"   4. Analyze results and identify improvements")
        print(f"   5. Iterate on model/pipeline parameters")
        
    except Exception as e:
        print(f"💥 An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()