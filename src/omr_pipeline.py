"""
OMR Pipeline Main Module
=======================

Main orchestrator for the Optical Music Recognition pipeline.
Coordinates all components from image preprocessing to MusicXML output.
Includes CUDA acceleration support for GPU processing.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

# CUDA configuration
try:
    from cuda_config import get_cuda_config, setup_cuda_optimizations, check_cuda_availability
    CUDA_CONFIG = get_cuda_config()
    setup_cuda_optimizations()
except ImportError:
    # Fallback configuration without CUDA
    CUDA_CONFIG = {
        'device': 'cpu',
        'use_cuda': False,
        'symbol_detection': {'device': 'cpu', 'batch_size': 4, 'half_precision': False}
    }

# Image processing
try:
    import cv2
    import numpy as np
except ImportError:
    print("Warning: OpenCV and NumPy not installed. Install with: pip install opencv-python numpy")

# OMR Components
# Component imports with fallback for different import contexts
try:
    # Try absolute imports first (when run as script or from parent directory)
    from preprocessing.image_preprocessor import ImagePreprocessor
    from detection.staff_detector import StaffDetector
    from detection.symbol_detector import SymbolDetector
    from reconstruction.music_reconstructor import MusicReconstructor
    from output.musicxml_generator import MusicXMLGenerator
    from output.json_exporter import JSONExporter
except ImportError:
    try:
        # Try relative imports (when imported as package)
        from .preprocessing.image_preprocessor import ImagePreprocessor
        from .detection.staff_detector import StaffDetector
        from .detection.symbol_detector import SymbolDetector
        from .reconstruction.music_reconstructor import MusicReconstructor
        from .output.musicxml_generator import MusicXMLGenerator
        from .output.json_exporter import JSONExporter
    except ImportError as e:
        print(f"Warning: Could not import OMR components: {e}")
        # Create placeholder classes for testing
        class ImagePreprocessor:
            def __init__(self, config=None): 
                self.config = config or {}
                self.use_cuda = CUDA_CONFIG.get('preprocessing', {}).get('use_cuda', False)
                print(f"ImagePreprocessor initialized (placeholder) - CUDA: {self.use_cuda}")
            
            def preprocess(self, image): 
                processing_time = 0.05 if self.use_cuda else 0.2
                time.sleep(processing_time)
                return {'processed_image': image, 'skew_angle': 0, 'processing_time': processing_time}
        
        class StaffDetector:
            def __init__(self, config=None): 
                self.config = config or {}
                self.use_cuda = CUDA_CONFIG.get('staff_detection', {}).get('use_cuda', False)
                print(f"StaffDetector initialized (placeholder) - CUDA: {self.use_cuda}")
            
            def detect_staves(self, image): 
                processing_time = 0.1 if self.use_cuda else 0.3
                time.sleep(processing_time)
                return {'staff_lines': [], 'staff_systems': []}
            
            def remove_staves(self, image, staff_lines): return image
        
        class SymbolDetector:
            def __init__(self, config=None): 
                self.config = config or {}
                self.device = CUDA_CONFIG.get('device', 'cpu')
                self.use_cuda = CUDA_CONFIG.get('use_cuda', False)
                print(f"SymbolDetector initialized (placeholder) - Device: {self.device}")
            
            def detect_symbols(self, image): 
                # Simulate processing time based on device
                processing_time = 0.1 if self.use_cuda else 0.5
                time.sleep(processing_time)
                return {'detections': []}
        
        class MusicReconstructor:
            def __init__(self, config=None): 
                self.config = config or {}
                print("MusicReconstructor initialized (placeholder)")
            
            def reconstruct_music(self, symbols, staff_info): 
                time.sleep(0.05)
                return {'measures': [], 'voices': []}
        
        class MusicXMLGenerator:
            def __init__(self, config=None): 
                self.config = config or {}
                print("MusicXMLGenerator initialized (placeholder)")
            
            def generate_musicxml(self, music_data, output_path=None): 
                content = '<?xml version="1.0"?><score-partwise version="3.1"></score-partwise>'
                if output_path:
                    Path(output_path).write_text(content)
                    return output_path
                else:
                    return content
        
        class JSONExporter:
            def __init__(self, config=None): 
                self.config = config or {}
                print("JSONExporter initialized (placeholder)")
            
            def export_confidence_data(self, results, output_path):
                data = {'overall_confidence': 0.5, 'low_confidence_symbols': []}
                with open(output_path, 'w') as f:
                    json.dump(data, f)
                return output_path
            
            def export_json(self, musical_elements, detected_symbols, detected_staves):
                return {
                    'musical_elements': musical_elements or {},
                    'detected_symbols': len(detected_symbols) if detected_symbols else 0,
                    'detected_staves': len(detected_staves) if detected_staves else 0,
                    'overall_confidence': 0.5,
                    'processing_info': 'Placeholder processing results'
                }

logger = logging.getLogger(__name__)


@dataclass
class OMRResults:
    """Container for OMR pipeline results."""
    image_path: str
    preprocessing_time: float = 0.0
    staff_detection_time: float = 0.0
    symbol_detection_time: float = 0.0
    music_reconstruction_time: float = 0.0
    output_generation_time: float = 0.0
    total_time: float = 0.0
    
    # Processing status
    success: bool = False
    error_message: str = ""
    
    # Intermediate results
    preprocessed_image: Optional[np.ndarray] = None
    detected_staves: Optional[List] = None
    detected_symbols: Optional[List] = None
    musical_elements: Optional[Dict] = None
    
    # Final outputs
    musicxml_content: Optional[str] = None
    json_report: Optional[Dict] = None
    confidence_scores: Optional[Dict] = None
    
    # Quality metrics
    quality_assessment: Dict = field(default_factory=dict)


class OMRPipeline:
    """
    Main OMR pipeline that orchestrates all processing steps.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the OMR pipeline.
        
        Args:
            config: Configuration dictionary for pipeline components
        """
        self.config = config or self._get_default_config()
        
        # Initialize all components
        try:
            self.preprocessor = ImagePreprocessor(self.config.get('preprocessing', {}))
            self.staff_detector = StaffDetector(self.config.get('staff_detection', {}))
            self.symbol_detector = SymbolDetector(self.config.get('symbol_detection', {}))
            self.music_reconstructor = MusicReconstructor(self.config.get('music_reconstruction', {}))
            self.musicxml_generator = MusicXMLGenerator(self.config.get('musicxml_generation', {}))
            self.json_exporter = JSONExporter(self.config.get('json_export', {}))
            
            logger.info("OMR Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OMR pipeline: {e}")
            raise
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for all pipeline components."""
        base_config = {
            'preprocessing': {
                'denoise_strength': 5,
                'contrast_enhancement': True,
                'skew_correction': True,
                'binarization_method': 'adaptive',
                'use_cuda': CUDA_CONFIG.get('preprocessing', {}).get('use_cuda', False)
            },
            'staff_detection': {
                'line_thickness_range': (1, 5),
                'min_line_length': 100,
                'angle_tolerance': 2.0,
                'use_cuda': CUDA_CONFIG.get('staff_detection', {}).get('use_cuda', False),
                'parallel_processing': CUDA_CONFIG.get('staff_detection', {}).get('parallel_processing', False)
            },
            'symbol_detection': {
                'confidence_threshold': 0.3,
                'nms_threshold': 0.4,
                'model_size': 'medium',
                'device': CUDA_CONFIG.get('device', 'cpu'),
                'batch_size': CUDA_CONFIG.get('symbol_detection', {}).get('batch_size', 4),
                'half_precision': CUDA_CONFIG.get('symbol_detection', {}).get('half_precision', False),
                'workers': CUDA_CONFIG.get('symbol_detection', {}).get('workers', 2)
            },
            'music_reconstruction': {
                'voice_separation_threshold': 30,
                'measure_grouping': True,
                'rhythm_quantization': True
            },
            'musicxml_generation': {
                'divisions': 480,
                'validate_output': True,
                'include_layout': False
            },
            'json_export': {
                'include_confidence_scores': True,
                'include_intermediate_results': True,
                'pretty_print': True
            },
            'cuda': CUDA_CONFIG
        }
        return base_config
    
    def process_image(self, image: np.ndarray, image_path: str = "") -> OMRResults:
        """
        Process a single sheet music image through the complete OMR pipeline.
        
        Args:
            image: Input image as numpy array
            image_path: Optional path to the image file
            
        Returns:
            OMRResults containing all processing results and timings
        """
        start_time = time.time()
        results = OMRResults(image_path=image_path)
        
        try:
            # Validate input image
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            logger.info(f"Starting OMR processing for: {image_path or 'memory image'}")
            logger.info(f"Image dimensions: {image.shape}")
            
            # Step 1: Image Preprocessing
            logger.info("Step 1: Image preprocessing...")
            preprocess_start = time.time()
            
            preprocess_result = self.preprocessor.preprocess(image)
            
            # Handle both dictionary and direct image returns
            if isinstance(preprocess_result, dict):
                preprocessed_image = preprocess_result.get('processed_image', image)
            else:
                preprocessed_image = preprocess_result
                
            results.preprocessed_image = preprocessed_image
            results.preprocessing_time = time.time() - preprocess_start
            
            logger.info(f"Preprocessing completed in {results.preprocessing_time:.2f}s")
            
            # Step 2: Staff Detection and Removal
            logger.info("Step 2: Staff detection and removal...")
            staff_start = time.time()
            
            staff_result = self.staff_detector.detect_staves(preprocessed_image)
            
            # Handle both dictionary and direct list returns
            if isinstance(staff_result, dict):
                detected_staves = staff_result.get('staff_lines', [])
            else:
                detected_staves = staff_result if staff_result else []
                
            staff_removed_image = self.staff_detector.remove_staves(preprocessed_image, detected_staves)
            
            results.detected_staves = detected_staves
            results.staff_detection_time = time.time() - staff_start
            
            logger.info(f"Staff detection completed in {results.staff_detection_time:.2f}s")
            logger.info(f"Detected {len(detected_staves)} staves")
            
            # Step 3: Symbol Detection
            logger.info("Step 3: Symbol detection and classification...")
            symbol_start = time.time()
            
            symbol_result = self.symbol_detector.detect_symbols(staff_removed_image)
            
            # Handle both dictionary and direct list returns
            if isinstance(symbol_result, dict):
                detected_symbols = symbol_result.get('detections', [])
            else:
                detected_symbols = symbol_result if symbol_result else []
            
            # Associate symbols with staves
            detected_symbols = self._associate_symbols_with_staves(detected_symbols, detected_staves)
            
            results.detected_symbols = detected_symbols
            results.symbol_detection_time = time.time() - symbol_start
            
            logger.info(f"Symbol detection completed in {results.symbol_detection_time:.2f}s")
            logger.info(f"Detected {len(detected_symbols)} symbols")
            
            # Step 4: Music Reconstruction
            logger.info("Step 4: Musical meaning reconstruction...")
            reconstruction_start = time.time()
            
            musical_elements = self.music_reconstructor.reconstruct_music(
                detected_symbols, detected_staves
            )
            results.musical_elements = musical_elements
            results.music_reconstruction_time = time.time() - reconstruction_start
            
            logger.info(f"Music reconstruction completed in {results.music_reconstruction_time:.2f}s")
            
            # Step 5: Output Generation
            logger.info("Step 5: Generating outputs...")
            output_start = time.time()
            
            # Generate MusicXML
            musicxml_content = self.musicxml_generator.generate_musicxml(musical_elements)
            results.musicxml_content = musicxml_content
            
            # Generate JSON report
            json_report = self.json_exporter.export_json(
                musical_elements, detected_symbols, detected_staves
            )
            results.json_report = json_report
            
            # Calculate confidence scores and quality assessment
            confidence_scores = self._calculate_confidence_scores(detected_symbols)
            results.confidence_scores = confidence_scores
            
            quality_assessment = self._assess_quality(results)
            results.quality_assessment = quality_assessment
            
            results.output_generation_time = time.time() - output_start
            logger.info(f"Output generation completed in {results.output_generation_time:.2f}s")
            
            # Calculate total time
            results.total_time = time.time() - start_time
            results.success = True
            
            logger.info(f"OMR processing completed successfully in {results.total_time:.2f}s")
            logger.info(f"Overall confidence: {confidence_scores.get('overall_confidence', 0):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in OMR pipeline: {e}")
            results.total_time = time.time() - start_time
            results.success = False
            results.error_message = str(e)
            return results
    
    def process_batch(self, image_paths: List[str], output_dir: str, 
                     save_intermediate: bool = False) -> List[OMRResults]:
        """
        Process multiple images in batch with comprehensive output management.
        
        Args:
            image_paths: List of paths to image files
            output_dir: Directory to save outputs
            save_intermediate: Whether to save intermediate processing results
            
        Returns:
            List of OMRResults for each processed image
        """
        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        batch_stats = {
            'total_images': len(image_paths),
            'successful_processing': 0,
            'failed_processing': 0,
            'total_time': 0,
            'average_time_per_image': 0
        }
        
        batch_start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Could not load image: {image_path}")
                    batch_stats['failed_processing'] += 1
                    continue
                
                # Process image
                results = self.process_image(image, image_path)
                all_results.append(results)
                
                if results.success:
                    batch_stats['successful_processing'] += 1
                    
                    # Save outputs
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    self._save_results(results, output_dir, base_name, save_intermediate)
                    
                    logger.info(f"Successfully processed and saved outputs for {image_path}")
                else:
                    batch_stats['failed_processing'] += 1
                    logger.error(f"Failed to process {image_path}: {results.error_message}")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                batch_stats['failed_processing'] += 1
                continue
        
        # Calculate batch statistics
        batch_stats['total_time'] = time.time() - batch_start_time
        if batch_stats['successful_processing'] > 0:
            total_processing_time = sum(r.total_time for r in all_results if r.success)
            batch_stats['average_time_per_image'] = total_processing_time / batch_stats['successful_processing']
        
        # Save batch summary
        batch_summary = {
            'batch_statistics': batch_stats,
            'individual_results': [
                {
                    'image_path': r.image_path,
                    'success': r.success,
                    'total_time': r.total_time,
                    'error_message': r.error_message,
                    'confidence': r.confidence_scores.get('overall_confidence', 0) if r.confidence_scores else 0
                }
                for r in all_results
            ]
        }
        
        batch_summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(batch_summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed.")
        logger.info(f"Successful: {batch_stats['successful_processing']}/{batch_stats['total_images']}")
        logger.info(f"Total time: {batch_stats['total_time']:.2f}s")
        logger.info(f"Average time per image: {batch_stats['average_time_per_image']:.2f}s")
        
        return all_results
    
    def _save_results(self, results: OMRResults, output_dir: str, base_name: str, 
                     save_intermediate: bool = False):
        """Save all results to appropriate files."""
        try:
            # Save MusicXML
            if results.musicxml_content:
                musicxml_path = os.path.join(output_dir, f"{base_name}.mxl")
                self.save_musicxml(results.musicxml_content, musicxml_path)
            
            # Save JSON report
            if results.json_report:
                json_path = os.path.join(output_dir, f"{base_name}_report.json")
                self.save_json_report(results.json_report, json_path)
            
            # Save intermediate results if requested
            if save_intermediate:
                intermediate_dir = os.path.join(output_dir, 'intermediate', base_name)
                os.makedirs(intermediate_dir, exist_ok=True)
                
                # Save preprocessed image
                if results.preprocessed_image is not None:
                    cv2.imwrite(
                        os.path.join(intermediate_dir, 'preprocessed.png'),
                        results.preprocessed_image
                    )
                
                # Save processing summary
                processing_summary = {
                    'timings': {
                        'preprocessing': results.preprocessing_time,
                        'staff_detection': results.staff_detection_time,
                        'symbol_detection': results.symbol_detection_time,
                        'music_reconstruction': results.music_reconstruction_time,
                        'output_generation': results.output_generation_time,
                        'total': results.total_time
                    },
                    'detection_counts': {
                        'staves': len(results.detected_staves) if results.detected_staves else 0,
                        'symbols': len(results.detected_symbols) if results.detected_symbols else 0
                    },
                    'quality_assessment': results.quality_assessment,
                    'confidence_scores': results.confidence_scores
                }
                
                with open(os.path.join(intermediate_dir, 'processing_summary.json'), 'w') as f:
                    json.dump(processing_summary, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving results for {base_name}: {e}")
    
    def _associate_symbols_with_staves(self, symbols: List, staves: List) -> List:
        """Associate detected symbols with their corresponding staves."""
        if not staves or not symbols:
            return symbols
        
        try:
            for symbol in symbols:
                # Handle different symbol data structures
                if hasattr(symbol, 'center'):
                    symbol_y = symbol.center[1]
                elif hasattr(symbol, 'bbox') and len(symbol.bbox) >= 4:
                    symbol_y = symbol.bbox[1] + symbol.bbox[3]//2
                elif isinstance(symbol, dict):
                    bbox = symbol.get('bbox', [0, 0, 0, 0])
                    if len(bbox) >= 4:
                        symbol_y = bbox[1] + bbox[3]//2
                    else:
                        continue  # Skip this symbol if no valid position
                else:
                    continue  # Skip this symbol if no valid position
                
                closest_staff = None
                min_distance = float('inf')
                
                for staff in staves:
                    # Handle different staff data structures
                    if hasattr(staff, 'center_y'):
                        staff_center_y = staff.center_y
                    elif hasattr(staff, 'y_position'):
                        staff_center_y = staff.y_position
                    elif isinstance(staff, dict):
                        staff_center_y = staff.get('y_position', staff.get('center_y', 0))
                    else:
                        continue  # Skip this staff if no valid position
                    
                    distance = abs(symbol_y - staff_center_y)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_staff = staff
                
                # Associate symbol with closest staff
                if closest_staff:
                    if hasattr(symbol, 'staff_id'):
                        symbol.staff_id = closest_staff.id if hasattr(closest_staff, 'id') else 0
                    elif isinstance(symbol, dict):
                        symbol['staff_id'] = closest_staff.get('id', 0) if isinstance(closest_staff, dict) else 0
        
        except Exception as e:
            logger.warning(f"Error associating symbols with staves: {e}")
        
        return symbols
    
    def save_musicxml(self, musicxml_content: str, output_path: str):
        """Save MusicXML content to file with validation."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(musicxml_content)
            
            # Validate the saved file
            if self._validate_musicxml_file(output_path):
                logger.info(f"Valid MusicXML saved to {output_path}")
            else:
                logger.warning(f"MusicXML saved to {output_path} but validation failed")
                
        except Exception as e:
            logger.error(f"Error saving MusicXML to {output_path}: {e}")
    
    def save_json_report(self, json_report: Dict, output_path: str):
        """Save JSON report to file with validation."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON report to {output_path}: {e}")
    
    def _validate_musicxml_file(self, file_path: str) -> bool:
        """Basic validation of saved MusicXML file."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Basic checks
            if root.tag not in ['score-partwise', 'score-timewise']:
                return False
            
            # Check for required elements
            if not root.find('.//part'):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_confidence_scores(self, detected_symbols: List) -> Dict:
        """Calculate comprehensive confidence statistics."""
        if not detected_symbols:
            return {
                'overall_confidence': 0.0,
                'low_confidence_count': 0,
                'medium_confidence_count': 0,
                'high_confidence_count': 0,
                'confidence_distribution': {},
                'symbol_type_confidence': {},
                'total_symbols': 0
            }
        
        try:
            # Extract confidences from different data structures
            confidences = []
            for symbol in detected_symbols:
                if hasattr(symbol, 'confidence'):
                    confidences.append(symbol.confidence)
                elif isinstance(symbol, dict) and 'confidence' in symbol:
                    confidences.append(symbol['confidence'])
                # If no confidence available, use default medium confidence
                else:
                    confidences.append(0.6)
        
            if not confidences:
                return {'total_symbols': len(detected_symbols), 'overall_confidence': 0.0}
            
            # Calculate statistics
            overall_confidence = sum(confidences) / len(confidences)
            
            # Confidence thresholds
            low_threshold = 0.4
            high_threshold = 0.8
            
            low_confidence_count = sum(1 for c in confidences if c < low_threshold)
            medium_confidence_count = sum(1 for c in confidences 
                                        if low_threshold <= c < high_threshold)
            high_confidence_count = sum(1 for c in confidences if c >= high_threshold)
            
            # Create confidence distribution
            confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            distribution = {}
            
            for i in range(len(confidence_bins) - 1):
                bin_name = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
                count = sum(1 for c in confidences 
                           if confidence_bins[i] <= c < confidence_bins[i+1])
                distribution[bin_name] = count
            
            # Per symbol type confidence
            symbol_type_confidence = {}
            for symbol in detected_symbols:
                symbol_type = None
                confidence_val = 0.6  # Default
                
                if hasattr(symbol, 'class_name'):
                    symbol_type = symbol.class_name
                elif isinstance(symbol, dict) and 'class' in symbol:
                    symbol_type = symbol['class']
                elif isinstance(symbol, dict) and 'class_name' in symbol:
                    symbol_type = symbol['class_name']
                
                if hasattr(symbol, 'confidence'):
                    confidence_val = symbol.confidence
                elif isinstance(symbol, dict) and 'confidence' in symbol:
                    confidence_val = symbol['confidence']
                
                if symbol_type:
                    if symbol_type not in symbol_type_confidence:
                        symbol_type_confidence[symbol_type] = []
                    symbol_type_confidence[symbol_type].append(confidence_val)
            
            # Average confidence per symbol type
            for symbol_type, conf_list in symbol_type_confidence.items():
                if conf_list:
                    symbol_type_confidence[symbol_type] = {
                        'average': sum(conf_list) / len(conf_list),
                        'count': len(conf_list),
                        'min': min(conf_list),
                        'max': max(conf_list)
                    }
            
            return {
                'overall_confidence': overall_confidence,
                'low_confidence_count': low_confidence_count,
                'medium_confidence_count': medium_confidence_count,
                'high_confidence_count': high_confidence_count,
                'confidence_distribution': distribution,
                'symbol_type_confidence': symbol_type_confidence,
                'total_symbols': len(detected_symbols),
                'confidence_statistics': {
                    'mean': overall_confidence,
                    'min': min(confidences),
                    'max': max(confidences),
                    'std': np.std(confidences) if len(confidences) > 1 else 0.0
                }
            }
            
        except Exception as e:
            logger.warning(f"Error calculating confidence scores: {e}")
            return {
                'overall_confidence': 0.5,
                'total_symbols': len(detected_symbols),
                'error': str(e)
            }
    
    def _assess_quality(self, results: OMRResults) -> Dict:
        """Assess the quality of OMR processing results."""
        quality = {
            'overall_score': 0.0,
            'component_scores': {},
            'issues_detected': [],
            'recommendations': []
        }
        
        try:
            # Staff detection quality
            staff_score = 0.8  # Default reasonable score
            if results.detected_staves:
                # Check staff detection metrics
                num_staves = len(results.detected_staves)
                if 1 <= num_staves <= 6:  # Reasonable number of staves
                    staff_score = 0.9
                elif num_staves > 6:
                    staff_score = 0.7
                    quality['issues_detected'].append("High number of detected staves")
                else:
                    staff_score = 0.6
                    quality['issues_detected'].append("No staves detected")
            
            quality['component_scores']['staff_detection'] = staff_score
            
            # Symbol detection quality
            symbol_score = 0.5
            if results.detected_symbols and results.confidence_scores:
                overall_conf = results.confidence_scores.get('overall_confidence', 0)
                high_conf_ratio = (results.confidence_scores.get('high_confidence_count', 0) / 
                                 max(results.confidence_scores.get('total_symbols', 1), 1))
                
                symbol_score = (overall_conf * 0.7) + (high_conf_ratio * 0.3)
                
                if overall_conf < 0.5:
                    quality['issues_detected'].append("Low overall symbol confidence")
                if high_conf_ratio < 0.3:
                    quality['issues_detected'].append("Few high-confidence symbols")
            
            quality['component_scores']['symbol_detection'] = symbol_score
            
            # Processing time quality
            time_score = 1.0
            if results.total_time > 30:  # More than 30 seconds
                time_score = 0.7
                quality['issues_detected'].append("Long processing time")
            elif results.total_time > 60:  # More than 1 minute
                time_score = 0.5
                quality['issues_detected'].append("Very long processing time")
            
            quality['component_scores']['processing_time'] = time_score
            
            # Musical content quality
            music_score = 0.7  # Default
            if results.musical_elements:
                # Check for musical content richness
                staves = results.musical_elements.get('staves', [])
                total_elements = sum(len(staff.get('elements', [])) for staff in staves)
                
                if total_elements > 10:
                    music_score = 0.9
                elif total_elements > 5:
                    music_score = 0.8
                elif total_elements == 0:
                    music_score = 0.3
                    quality['issues_detected'].append("No musical elements detected")
            
            quality['component_scores']['musical_content'] = music_score
            
            # Calculate overall score
            scores = list(quality['component_scores'].values())
            quality['overall_score'] = sum(scores) / len(scores) if scores else 0.0
            
            # Generate recommendations
            if quality['overall_score'] < 0.6:
                quality['recommendations'].append("Consider image quality improvement")
            if symbol_score < 0.6:
                quality['recommendations'].append("Review symbol detection parameters")
            if staff_score < 0.7:
                quality['recommendations'].append("Check staff detection settings")
            if not quality['recommendations']:
                quality['recommendations'].append("Processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            quality['issues_detected'].append("Quality assessment failed")
        
        return quality


def create_cli():
    """Create command-line interface for the OMR pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OMR Pipeline - Convert sheet music images to MusicXML')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')
    parser.add_argument('--intermediate', action='store_true', help='Save intermediate results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration if provided
        config = None
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Initialize pipeline
        pipeline = OMRPipeline(config)
        
        if args.batch or os.path.isdir(args.input):
            # Batch processing
            if os.path.isdir(args.input):
                # Find all image files in directory
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                image_paths = []
                
                for root, dirs, files in os.walk(args.input):
                    for file in files:
                        if Path(file).suffix.lower() in image_extensions:
                            image_paths.append(os.path.join(root, file))
                
                logger.info(f"Found {len(image_paths)} images for batch processing")
            else:
                image_paths = [args.input]
            
            # Process batch
            results = pipeline.process_batch(image_paths, args.output, args.intermediate)
            
            # Summary
            successful = sum(1 for r in results if r.success)
            print(f"\nBatch processing completed: {successful}/{len(results)} images processed successfully")
            
        else:
            # Single image processing
            image = cv2.imread(args.input)
            if image is None:
                raise ValueError(f"Could not load image: {args.input}")
            
            result = pipeline.process_image(image, args.input)
            
            if result.success:
                # Save outputs
                base_name = os.path.splitext(os.path.basename(args.input))[0]
                pipeline._save_results(result, args.output, base_name, args.intermediate)
                
                print(f"\nProcessing completed successfully!")
                print(f"Processing time: {result.total_time:.2f}s")
                print(f"Overall confidence: {result.confidence_scores.get('overall_confidence', 0):.3f}")
                print(f"Quality score: {result.quality_assessment.get('overall_score', 0):.3f}")
            else:
                print(f"Processing failed: {result.error_message}")
    
    except Exception as e:
        logger.error(f"OMR pipeline failed: {e}")
        return 1
    
    return 0


def test_pipeline():
    """Test function for the OMR pipeline."""
    try:
        import numpy as np
        
        # Create a test image (white background with some black lines)
        test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Add some horizontal lines to simulate staff lines
        for y in [200, 220, 240, 260, 280]:
            cv2.line(test_image, (50, y), (550, y), (0, 0, 0), 2)
        
        # Add some simple shapes to simulate notes
        cv2.circle(test_image, (100, 190), 8, (0, 0, 0), -1)  # Note head
        cv2.circle(test_image, (150, 210), 8, (0, 0, 0), -1)  # Note head
        cv2.circle(test_image, (200, 230), 8, (0, 0, 0), -1)  # Note head
        
        # Initialize pipeline
        pipeline = OMRPipeline()
        
        # Process the test image
        results = pipeline.process_image(test_image, "test_image")
        
        print("OMR Pipeline Test Results:")
        print("=" * 40)
        print(f"Success: {results.success}")
        
        if results.success:
            print(f"Total processing time: {results.total_time:.2f}s")
            print(f"  - Preprocessing: {results.preprocessing_time:.2f}s")
            print(f"  - Staff detection: {results.staff_detection_time:.2f}s")
            print(f"  - Symbol detection: {results.symbol_detection_time:.2f}s")
            print(f"  - Music reconstruction: {results.music_reconstruction_time:.2f}s")
            print(f"  - Output generation: {results.output_generation_time:.2f}s")
            
            if results.detected_staves:
                print(f"Detected staves: {len(results.detected_staves)}")
            
            if results.detected_symbols:
                print(f"Detected symbols: {len(results.detected_symbols)}")
            
            if results.confidence_scores:
                print(f"Overall confidence: {results.confidence_scores['overall_confidence']:.3f}")
            
            if results.quality_assessment:
                print(f"Quality score: {results.quality_assessment['overall_score']:.3f}")
            
            print("Pipeline test completed successfully!")
        else:
            print(f"Pipeline test failed: {results.error_message}")
            
    except Exception as e:
        print(f"Pipeline test failed with exception: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        test_pipeline()