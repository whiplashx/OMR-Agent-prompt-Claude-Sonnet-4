"""
Evaluation Metrics Module
========================

Implements comprehensive evaluation metrics for OMR systems including
symbol-level accuracy, semantic-level correctness, and musical meaning evaluation.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import math

# Import OMR components
from ..detection.symbol_detector import DetectedSymbol
from ..reconstruction.music_reconstructor import Note, Measure, MusicalElement

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Dict = field(default_factory=dict)
    per_class_scores: Dict = field(default_factory=dict)


@dataclass
class SymbolMatch:
    """Represents a match between predicted and ground truth symbols."""
    predicted_symbol: DetectedSymbol
    ground_truth_symbol: Dict
    iou: float
    class_match: bool
    confidence: float


class BoundingBoxMatcher:
    """
    Matches predicted bounding boxes with ground truth using IoU.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the matcher.
        
        Args:
            iou_threshold: Minimum IoU for a positive match
        """
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box (x, y, width, height)
            box2: Second bounding box (x, y, width, height)
            
        Returns:
            IoU score between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to (x1, y1, x2, y2) format
        box1_coords = (x1, y1, x1 + w1, y1 + h1)
        box2_coords = (x2, y2, x2 + w2, y2 + h2)
        
        # Calculate intersection
        x_left = max(box1_coords[0], box2_coords[0])
        y_top = max(box1_coords[1], box2_coords[1])
        x_right = min(box1_coords[2], box2_coords[2])
        y_bottom = min(box1_coords[3], box2_coords[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def match_symbols(self, predicted: List[DetectedSymbol], 
                     ground_truth: List[Dict]) -> List[SymbolMatch]:
        """
        Match predicted symbols with ground truth symbols.
        
        Args:
            predicted: List of predicted symbols
            ground_truth: List of ground truth symbol annotations
            
        Returns:
            List of symbol matches
        """
        matches = []
        used_gt_indices = set()
        
        # Sort predictions by confidence (highest first)
        sorted_predictions = sorted(predicted, key=lambda s: s.confidence, reverse=True)
        
        for pred_symbol in sorted_predictions:
            best_match = None
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_symbol in enumerate(ground_truth):
                if gt_idx in used_gt_indices:
                    continue
                
                # Calculate IoU
                if 'bbox' in gt_symbol:
                    iou = self.calculate_iou(pred_symbol.bbox, gt_symbol['bbox'])
                    
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_match = gt_symbol
                        best_gt_idx = gt_idx
            
            if best_match is not None:
                # Check class match
                gt_class = best_match.get('class_name', '')
                class_match = (pred_symbol.class_name == gt_class)
                
                match = SymbolMatch(
                    predicted_symbol=pred_symbol,
                    ground_truth_symbol=best_match,
                    iou=best_iou,
                    class_match=class_match,
                    confidence=pred_symbol.confidence
                )
                matches.append(match)
                used_gt_indices.add(best_gt_idx)
        
        return matches


class SymbolLevelEvaluator:
    """
    Evaluates symbol-level detection and classification performance.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the evaluator.
        
        Args:
            iou_threshold: IoU threshold for positive matches
        """
        self.matcher = BoundingBoxMatcher(iou_threshold)
        self.iou_threshold = iou_threshold
    
    def evaluate(self, predicted: List[DetectedSymbol], 
                ground_truth: List[Dict]) -> List[EvaluationResult]:
        """
        Evaluate symbol-level performance.
        
        Args:
            predicted: Predicted symbols
            ground_truth: Ground truth annotations
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Match symbols
        matches = self.matcher.match_symbols(predicted, ground_truth)
        
        # Calculate detection metrics
        detection_result = self._calculate_detection_metrics(
            matches, len(predicted), len(ground_truth)
        )
        results.append(detection_result)
        
        # Calculate classification metrics
        classification_result = self._calculate_classification_metrics(matches)
        results.append(classification_result)
        
        # Calculate per-class metrics
        per_class_result = self._calculate_per_class_metrics(matches, ground_truth)
        results.append(per_class_result)
        
        # Calculate confidence calibration
        calibration_result = self._calculate_confidence_calibration(matches)
        results.append(calibration_result)
        
        return results
    
    def _calculate_detection_metrics(self, matches: List[SymbolMatch], 
                                   num_predicted: int, num_gt: int) -> EvaluationResult:
        """Calculate detection precision, recall, and F1."""
        true_positives = len(matches)
        false_positives = num_predicted - true_positives
        false_negatives = num_gt - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return EvaluationResult(
            metric_name='detection',
            score=f1,
            details={
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
        )
    
    def _calculate_classification_metrics(self, matches: List[SymbolMatch]) -> EvaluationResult:
        """Calculate classification accuracy among detected symbols."""
        if not matches:
            return EvaluationResult(
                metric_name='classification',
                score=0.0,
                details={'accuracy': 0.0, 'total_matches': 0}
            )
        
        correct_classifications = sum(1 for match in matches if match.class_match)
        accuracy = correct_classifications / len(matches)
        
        return EvaluationResult(
            metric_name='classification',
            score=accuracy,
            details={
                'accuracy': accuracy,
                'correct_classifications': correct_classifications,
                'total_matches': len(matches)
            }
        )
    
    def _calculate_per_class_metrics(self, matches: List[SymbolMatch], 
                                   ground_truth: List[Dict]) -> EvaluationResult:
        """Calculate per-class precision and recall."""
        # Count by class
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Count true positives and false positives from matches
        for match in matches:
            gt_class = match.ground_truth_symbol.get('class_name', 'unknown')
            pred_class = match.predicted_symbol.class_name
            
            if match.class_match:
                class_stats[gt_class]['tp'] += 1
            else:
                class_stats[gt_class]['fn'] += 1
                class_stats[pred_class]['fp'] += 1
        
        # Count false negatives (unmatched ground truth)
        matched_gt = {match.ground_truth_symbol.get('id', id(match.ground_truth_symbol)) for match in matches}
        for gt_symbol in ground_truth:
            gt_id = gt_symbol.get('id', id(gt_symbol))
            if gt_id not in matched_gt:
                gt_class = gt_symbol.get('class_name', 'unknown')
                class_stats[gt_class]['fn'] += 1
        
        # Calculate per-class metrics
        per_class_scores = {}
        total_f1 = 0
        valid_classes = 0
        
        for class_name, stats in class_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_scores[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn
            }
            
            if tp + fn > 0:  # Only count classes that exist in ground truth
                total_f1 += f1
                valid_classes += 1
        
        macro_f1 = total_f1 / valid_classes if valid_classes > 0 else 0
        
        return EvaluationResult(
            metric_name='per_class',
            score=macro_f1,
            details={'macro_f1': macro_f1, 'num_classes': valid_classes},
            per_class_scores=per_class_scores
        )
    
    def _calculate_confidence_calibration(self, matches: List[SymbolMatch]) -> EvaluationResult:
        """Calculate how well confidence scores correlate with accuracy."""
        if not matches:
            return EvaluationResult(
                metric_name='confidence_calibration',
                score=0.0,
                details={'ece': 0.0, 'num_bins': 0}
            )
        
        # Expected Calibration Error (ECE)
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(matches)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find matches in this confidence bin
            in_bin = [
                match for match in matches 
                if bin_lower < match.confidence <= bin_upper
            ]
            
            if len(in_bin) > 0:
                # Calculate accuracy in this bin
                accuracy_in_bin = sum(1 for match in in_bin if match.class_match) / len(in_bin)
                
                # Average confidence in this bin
                avg_confidence_in_bin = sum(match.confidence for match in in_bin) / len(in_bin)
                
                # Add to ECE
                ece += (len(in_bin) / total_samples) * abs(avg_confidence_in_bin - accuracy_in_bin)
        
        return EvaluationResult(
            metric_name='confidence_calibration',
            score=1.0 - ece,  # Higher score is better
            details={
                'ece': ece,
                'num_bins': num_bins,
                'total_samples': total_samples
            }
        )


class SemanticLevelEvaluator:
    """
    Evaluates semantic-level musical correctness.
    """
    
    def __init__(self):
        """Initialize the semantic evaluator."""
        pass
    
    def evaluate(self, predicted_music: Dict, ground_truth_music: Dict) -> List[EvaluationResult]:
        """
        Evaluate semantic-level musical accuracy.
        
        Args:
            predicted_music: Predicted musical elements
            ground_truth_music: Ground truth musical elements
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Pitch accuracy
        pitch_result = self._evaluate_pitch_accuracy(predicted_music, ground_truth_music)
        results.append(pitch_result)
        
        # Rhythm accuracy
        rhythm_result = self._evaluate_rhythm_accuracy(predicted_music, ground_truth_music)
        results.append(rhythm_result)
        
        # Key signature accuracy
        key_result = self._evaluate_key_signature_accuracy(predicted_music, ground_truth_music)
        results.append(key_result)
        
        # Time signature accuracy
        time_result = self._evaluate_time_signature_accuracy(predicted_music, ground_truth_music)
        results.append(time_result)
        
        # Overall musical similarity
        overall_result = self._evaluate_overall_musical_similarity(predicted_music, ground_truth_music)
        results.append(overall_result)
        
        return results
    
    def _evaluate_pitch_accuracy(self, predicted: Dict, ground_truth: Dict) -> EvaluationResult:
        """Evaluate pitch accuracy of detected notes."""
        pred_notes = self._extract_notes(predicted)
        gt_notes = self._extract_notes(ground_truth)
        
        if not gt_notes:
            return EvaluationResult(
                metric_name='pitch_accuracy',
                score=0.0,
                details={'error': 'No ground truth notes'}
            )
        
        # Match notes by position and compare pitches
        correct_pitches = 0
        total_matches = 0
        
        for pred_note in pred_notes:
            # Find closest ground truth note by position
            closest_gt = min(
                gt_notes,
                key=lambda gt: abs(pred_note.x_position - gt['x_position']) + 
                              abs(pred_note.y_position - gt['y_position'])
            )
            
            # Check if positions are close enough
            position_threshold = 20  # pixels
            if (abs(pred_note.x_position - closest_gt['x_position']) < position_threshold and
                abs(pred_note.y_position - closest_gt['y_position']) < position_threshold):
                
                total_matches += 1
                if pred_note.pitch == closest_gt.get('pitch'):
                    correct_pitches += 1
        
        accuracy = correct_pitches / total_matches if total_matches > 0 else 0
        
        return EvaluationResult(
            metric_name='pitch_accuracy',
            score=accuracy,
            details={
                'correct_pitches': correct_pitches,
                'total_matches': total_matches,
                'accuracy': accuracy
            }
        )
    
    def _evaluate_rhythm_accuracy(self, predicted: Dict, ground_truth: Dict) -> EvaluationResult:
        """Evaluate rhythm accuracy of detected notes."""
        pred_notes = self._extract_notes(predicted)
        gt_notes = self._extract_notes(ground_truth)
        
        if not gt_notes:
            return EvaluationResult(
                metric_name='rhythm_accuracy',
                score=0.0,
                details={'error': 'No ground truth notes'}
            )
        
        # Match notes and compare durations
        correct_rhythms = 0
        total_matches = 0
        
        for pred_note in pred_notes:
            # Find closest ground truth note
            closest_gt = min(
                gt_notes,
                key=lambda gt: abs(pred_note.x_position - gt['x_position'])
            )
            
            position_threshold = 20
            if abs(pred_note.x_position - closest_gt['x_position']) < position_threshold:
                total_matches += 1
                
                # Compare durations (with tolerance)
                pred_duration = pred_note.duration
                gt_duration = closest_gt.get('duration', 1.0)
                
                if abs(pred_duration - gt_duration) < 0.1:  # Small tolerance
                    correct_rhythms += 1
        
        accuracy = correct_rhythms / total_matches if total_matches > 0 else 0
        
        return EvaluationResult(
            metric_name='rhythm_accuracy',
            score=accuracy,
            details={
                'correct_rhythms': correct_rhythms,
                'total_matches': total_matches,
                'accuracy': accuracy
            }
        )
    
    def _evaluate_key_signature_accuracy(self, predicted: Dict, ground_truth: Dict) -> EvaluationResult:
        """Evaluate key signature detection accuracy."""
        # Extract key signatures from both predictions and ground truth
        pred_keys = self._extract_key_signatures(predicted)
        gt_keys = self._extract_key_signatures(ground_truth)
        
        if not gt_keys:
            return EvaluationResult(
                metric_name='key_signature_accuracy',
                score=1.0 if not pred_keys else 0.0,
                details={'message': 'No key signatures in ground truth'}
            )
        
        # Compare key signatures
        correct_keys = 0
        for pred_key, gt_key in zip(pred_keys, gt_keys):
            if pred_key.get('sharps', 0) == gt_key.get('sharps', 0):
                correct_keys += 1
        
        accuracy = correct_keys / len(gt_keys) if gt_keys else 0
        
        return EvaluationResult(
            metric_name='key_signature_accuracy',
            score=accuracy,
            details={
                'correct_keys': correct_keys,
                'total_keys': len(gt_keys),
                'accuracy': accuracy
            }
        )
    
    def _evaluate_time_signature_accuracy(self, predicted: Dict, ground_truth: Dict) -> EvaluationResult:
        """Evaluate time signature detection accuracy."""
        pred_times = self._extract_time_signatures(predicted)
        gt_times = self._extract_time_signatures(ground_truth)
        
        if not gt_times:
            return EvaluationResult(
                metric_name='time_signature_accuracy',
                score=1.0 if not pred_times else 0.0,
                details={'message': 'No time signatures in ground truth'}
            )
        
        # Compare time signatures
        correct_times = 0
        for pred_time, gt_time in zip(pred_times, gt_times):
            if (pred_time.get('numerator') == gt_time.get('numerator') and
                pred_time.get('denominator') == gt_time.get('denominator')):
                correct_times += 1
        
        accuracy = correct_times / len(gt_times) if gt_times else 0
        
        return EvaluationResult(
            metric_name='time_signature_accuracy',
            score=accuracy,
            details={
                'correct_times': correct_times,
                'total_times': len(gt_times),
                'accuracy': accuracy
            }
        )
    
    def _evaluate_overall_musical_similarity(self, predicted: Dict, ground_truth: Dict) -> EvaluationResult:
        """Evaluate overall musical similarity using a composite score."""
        # This could implement more sophisticated musical similarity measures
        # For now, we'll use a simple weighted combination of other metrics
        
        # Get individual metric scores
        pitch_score = self._evaluate_pitch_accuracy(predicted, ground_truth).score
        rhythm_score = self._evaluate_rhythm_accuracy(predicted, ground_truth).score
        key_score = self._evaluate_key_signature_accuracy(predicted, ground_truth).score
        time_score = self._evaluate_time_signature_accuracy(predicted, ground_truth).score
        
        # Weighted combination
        weights = {'pitch': 0.4, 'rhythm': 0.4, 'key': 0.1, 'time': 0.1}
        
        overall_score = (
            weights['pitch'] * pitch_score +
            weights['rhythm'] * rhythm_score +
            weights['key'] * key_score +
            weights['time'] * time_score
        )
        
        return EvaluationResult(
            metric_name='overall_musical_similarity',
            score=overall_score,
            details={
                'pitch_score': pitch_score,
                'rhythm_score': rhythm_score,
                'key_score': key_score,
                'time_score': time_score,
                'weights': weights
            }
        )
    
    def _extract_notes(self, music_data: Dict) -> List:
        """Extract note information from musical data."""
        notes = []
        
        if 'staves' in music_data:
            for staff_data in music_data['staves']:
                if 'elements' in staff_data:
                    for element in staff_data['elements']:
                        if isinstance(element, Note) or (isinstance(element, dict) and element.get('element_type') == 'note'):
                            notes.append(element)
        
        return notes
    
    def _extract_key_signatures(self, music_data: Dict) -> List[Dict]:
        """Extract key signature information."""
        key_sigs = []
        
        if 'staves' in music_data:
            for staff_data in music_data['staves']:
                key_sig = staff_data.get('key_signature')
                if key_sig:
                    if hasattr(key_sig, 'sharps'):
                        key_sigs.append({'sharps': key_sig.sharps, 'key': key_sig.key})
                    else:
                        key_sigs.append(key_sig)
        
        return key_sigs
    
    def _extract_time_signatures(self, music_data: Dict) -> List[Dict]:
        """Extract time signature information."""
        time_sigs = []
        
        if 'staves' in music_data:
            for staff_data in music_data['staves']:
                time_sig = staff_data.get('time_signature')
                if time_sig:
                    if hasattr(time_sig, 'numerator'):
                        time_sigs.append({'numerator': time_sig.numerator, 'denominator': time_sig.denominator})
                    else:
                        time_sigs.append(time_sig)
        
        return time_sigs


class OMREvaluator:
    """
    Main OMR evaluation class that combines symbol and semantic level evaluation.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the OMR evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or {}
        self.iou_threshold = self.config.get('iou_threshold', 0.5)
        
        self.symbol_evaluator = SymbolLevelEvaluator(self.iou_threshold)
        self.semantic_evaluator = SemanticLevelEvaluator()
    
    def evaluate_full_pipeline(self, predicted_results: Dict, 
                              ground_truth: Dict) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluate complete OMR pipeline results.
        
        Args:
            predicted_results: Results from OMR pipeline
            ground_truth: Ground truth annotations
            
        Returns:
            Dictionary of evaluation results by category
        """
        evaluation_results = {}
        
        # Symbol-level evaluation
        if 'symbols' in predicted_results and 'symbols' in ground_truth:
            symbol_results = self.symbol_evaluator.evaluate(
                predicted_results['symbols'],
                ground_truth['symbols']
            )
            evaluation_results['symbol_level'] = symbol_results
        
        # Semantic-level evaluation
        if 'musical_elements' in predicted_results and 'musical_elements' in ground_truth:
            semantic_results = self.semantic_evaluator.evaluate(
                predicted_results['musical_elements'],
                ground_truth['musical_elements']
            )
            evaluation_results['semantic_level'] = semantic_results
        
        return evaluation_results
    
    def generate_evaluation_report(self, evaluation_results: Dict) -> Dict:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_full_pipeline
            
        Returns:
            Formatted evaluation report
        """
        report = {
            'summary': {},
            'detailed_results': evaluation_results,
            'recommendations': []
        }
        
        # Summarize scores
        all_scores = []
        for category, results in evaluation_results.items():
            category_scores = []
            for result in results:
                category_scores.append(result.score)
                all_scores.append(result.score)
            
            if category_scores:
                report['summary'][f'{category}_average'] = np.mean(category_scores)
        
        if all_scores:
            report['summary']['overall_average'] = np.mean(all_scores)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(evaluation_results)
        
        return report
    
    def _generate_recommendations(self, evaluation_results: Dict) -> List[str]:
        """Generate improvement recommendations based on evaluation results."""
        recommendations = []
        
        # Check symbol-level performance
        if 'symbol_level' in evaluation_results:
            for result in evaluation_results['symbol_level']:
                if result.metric_name == 'detection' and result.score < 0.8:
                    recommendations.append(
                        f"Detection F1 score is {result.score:.3f}. Consider improving "
                        "the object detection model or adjusting confidence thresholds."
                    )
                
                if result.metric_name == 'classification' and result.score < 0.9:
                    recommendations.append(
                        f"Classification accuracy is {result.score:.3f}. Consider "
                        "adding more training data or improving feature extraction."
                    )
        
        # Check semantic-level performance
        if 'semantic_level' in evaluation_results:
            for result in evaluation_results['semantic_level']:
                if result.metric_name == 'pitch_accuracy' and result.score < 0.8:
                    recommendations.append(
                        f"Pitch accuracy is {result.score:.3f}. Consider improving "
                        "staff line detection or pitch calculation algorithms."
                    )
                
                if result.metric_name == 'rhythm_accuracy' and result.score < 0.8:
                    recommendations.append(
                        f"Rhythm accuracy is {result.score:.3f}. Consider improving "
                        "note duration classification or rhythm quantization."
                    )
        
        if not recommendations:
            recommendations.append("All metrics are performing well. Consider evaluating on more challenging datasets.")
        
        return recommendations
    
    def save_evaluation_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results to save
            output_path: Output file path
        """
        # Convert evaluation results to serializable format
        serializable_results = {}
        
        for category, result_list in results.items():
            serializable_results[category] = []
            for result in result_list:
                serializable_result = {
                    'metric_name': result.metric_name,
                    'score': float(result.score),
                    'details': result.details,
                    'per_class_scores': result.per_class_scores
                }
                serializable_results[category].append(serializable_result)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")


def test_evaluator():
    """Test function for the OMR evaluator."""
    from ..detection.symbol_detector import DetectedSymbol
    
    # Create test data
    predicted_symbols = [
        DetectedSymbol(
            class_name='quarter_note',
            confidence=0.9,
            bbox=(100, 200, 20, 30),
            center=(110, 215),
            pitch='C4'
        ),
        DetectedSymbol(
            class_name='half_note',
            confidence=0.8,
            bbox=(150, 190, 25, 35),
            center=(162, 207),
            pitch='D4'
        )
    ]
    
    ground_truth_symbols = [
        {
            'class_name': 'quarter_note',
            'bbox': (98, 198, 22, 32),
            'pitch': 'C4'
        },
        {
            'class_name': 'half_note',
            'bbox': (148, 188, 27, 37),
            'pitch': 'D4'
        }
    ]
    
    # Test symbol-level evaluation
    evaluator = OMREvaluator()
    
    predicted_results = {'symbols': predicted_symbols}
    ground_truth = {'symbols': ground_truth_symbols}
    
    results = evaluator.evaluate_full_pipeline(predicted_results, ground_truth)
    
    print("Evaluation Results:")
    for category, result_list in results.items():
        print(f"\n{category}:")
        for result in result_list:
            print(f"  {result.metric_name}: {result.score:.3f}")


if __name__ == "__main__":
    test_evaluator()