"""
Evaluation CLI Module
====================

Command-line interface for running OMR evaluation on datasets.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import cv2
    import numpy as np
except ImportError:
    print("Warning: OpenCV and NumPy not installed. Install with: pip install opencv-python numpy")

from .metrics import OMREvaluator
from ..omr_pipeline import OMRPipeline

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Runs evaluation on OMR datasets.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the evaluation runner.
        
        Args:
            config_path: Path to evaluation configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.evaluator = OMREvaluator(self.config.get('evaluation', {}))
        self.omr_pipeline = OMRPipeline()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load evaluation configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def evaluate_dataset(self, dataset_path: str, output_dir: str) -> Dict:
        """
        Evaluate OMR pipeline on a complete dataset.
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Output directory for results
            
        Returns:
            Aggregated evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images and ground truth files
        dataset_files = self._find_dataset_files(dataset_path)
        
        if not dataset_files:
            raise ValueError(f"No valid dataset files found in {dataset_path}")
        
        # Run evaluation on each file
        all_results = []
        individual_results = {}
        
        for image_path, gt_path in dataset_files:
            logger.info(f"Evaluating {image_path}")
            
            try:
                # Run OMR pipeline
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                pipeline_results = self.omr_pipeline.process_image(image)
                
                # Load ground truth
                ground_truth = self._load_ground_truth(gt_path)
                
                # Evaluate
                evaluation_results = self.evaluator.evaluate_full_pipeline(
                    pipeline_results, ground_truth
                )
                
                all_results.append(evaluation_results)
                individual_results[os.path.basename(image_path)] = evaluation_results
                
                # Save individual results
                result_file = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_evaluation.json"
                )
                self.evaluator.save_evaluation_results(evaluation_results, result_file)
                
            except Exception as e:
                logger.error(f"Error evaluating {image_path}: {e}")
                continue
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Generate overall report
        overall_report = self.evaluator.generate_evaluation_report(aggregated_results)
        
        # Save aggregated results
        aggregate_file = os.path.join(output_dir, "aggregated_evaluation.json")
        with open(aggregate_file, 'w') as f:
            json.dump({
                'aggregated_results': aggregated_results,
                'overall_report': overall_report,
                'individual_results': {k: self._serialize_results(v) for k, v in individual_results.items()}
            }, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {output_dir}")
        return overall_report
    
    def _find_dataset_files(self, dataset_path: str) -> List[tuple]:
        """Find image and ground truth file pairs in dataset."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        gt_extensions = {'.json', '.xml', '.txt'}
        
        dataset_files = []
        
        for root, dirs, files in os.walk(dataset_path):
            image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
            
            for image_file in image_files:
                image_path = os.path.join(root, image_file)
                base_name = os.path.splitext(image_file)[0]
                
                # Look for corresponding ground truth file
                gt_file = None
                for ext in gt_extensions:
                    potential_gt = os.path.join(root, f"{base_name}{ext}")
                    if os.path.exists(potential_gt):
                        gt_file = potential_gt
                        break
                
                if gt_file:
                    dataset_files.append((image_path, gt_file))
                else:
                    logger.warning(f"No ground truth found for {image_file}")
        
        return dataset_files
    
    def _load_ground_truth(self, gt_path: str) -> Dict:
        """Load ground truth annotations from file."""
        try:
            with open(gt_path, 'r') as f:
                if gt_path.endswith('.json'):
                    return json.load(f)
                elif gt_path.endswith('.xml'):
                    # Parse MusicXML ground truth
                    return self._parse_musicxml_ground_truth(gt_path)
                else:
                    # Assume plain text format
                    return {'raw_text': f.read()}
        except Exception as e:
            logger.error(f"Error loading ground truth from {gt_path}: {e}")
            return {}
    
    def _parse_musicxml_ground_truth(self, xml_path: str) -> Dict:
        """Parse MusicXML file to extract ground truth information."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract basic musical information
            ground_truth = {
                'symbols': [],
                'musical_elements': {
                    'staves': []
                }
            }
            
            # This is a simplified parser - real implementation would be more comprehensive
            for part in root.findall('.//part'):
                staff_data = {
                    'elements': [],
                    'key_signature': None,
                    'time_signature': None
                }
                
                for measure in part.findall('measure'):
                    for note in measure.findall('note'):
                        pitch_elem = note.find('pitch')
                        duration_elem = note.find('duration')
                        
                        if pitch_elem is not None:
                            step = pitch_elem.find('step')
                            octave = pitch_elem.find('octave')
                            
                            if step is not None and octave is not None:
                                pitch = f"{step.text}{octave.text}"
                                duration = float(duration_elem.text) if duration_elem is not None else 1.0
                                
                                staff_data['elements'].append({
                                    'element_type': 'note',
                                    'pitch': pitch,
                                    'duration': duration,
                                    'x_position': 0,  # Would need layout information
                                    'y_position': 0
                                })
                
                ground_truth['musical_elements']['staves'].append(staff_data)
            
            return ground_truth
            
        except Exception as e:
            logger.error(f"Error parsing MusicXML ground truth: {e}")
            return {}
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results from multiple evaluations."""
        if not all_results:
            return {}
        
        aggregated = {}
        
        # Collect all metric names by category
        for results in all_results:
            for category, result_list in results.items():
                if category not in aggregated:
                    aggregated[category] = {}
                
                for result in result_list:
                    metric_name = result.metric_name
                    if metric_name not in aggregated[category]:
                        aggregated[category][metric_name] = []
                    
                    aggregated[category][metric_name].append(result.score)
        
        # Calculate statistics for each metric
        final_aggregated = {}
        for category, metrics in aggregated.items():
            final_aggregated[category] = []
            
            for metric_name, scores in metrics.items():
                if scores:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    min_score = np.min(scores)
                    max_score = np.max(scores)
                    
                    # Create aggregated result
                    from .metrics import EvaluationResult
                    aggregated_result = EvaluationResult(
                        metric_name=metric_name,
                        score=mean_score,
                        details={
                            'mean': mean_score,
                            'std': std_score,
                            'min': min_score,
                            'max': max_score,
                            'count': len(scores),
                            'individual_scores': scores
                        }
                    )
                    
                    final_aggregated[category].append(aggregated_result)
        
        return final_aggregated
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Convert evaluation results to serializable format."""
        serializable = {}
        
        for category, result_list in results.items():
            serializable[category] = []
            for result in result_list:
                serializable[category].append({
                    'metric_name': result.metric_name,
                    'score': float(result.score),
                    'details': result.details,
                    'per_class_scores': result.per_class_scores
                })
        
        return serializable


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='OMR Evaluation Tool')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('output_dir', help='Output directory for evaluation results')
    parser.add_argument('--config', help='Path to evaluation configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run evaluation
        runner = EvaluationRunner(args.config)
        results = runner.evaluate_dataset(args.dataset_path, args.output_dir)
        
        # Print summary
        print("\nEvaluation Summary:")
        print("==================")
        
        if 'summary' in results:
            for metric, score in results['summary'].items():
                print(f"{metric}: {score:.3f}")
        
        if 'recommendations' in results:
            print("\nRecommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()