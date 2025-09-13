"""
JSON Exporter Module
===================

Exports OMR results and confidence information to JSON format.
Provides detailed analysis and metadata for manual review.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..detection.symbol_detector import DetectedSymbol
from ..detection.staff_detector import Staff
from ..reconstruction.music_reconstructor import MusicalElement, Note, Measure, Voice

logger = logging.getLogger(__name__)


class JSONExporter:
    """
    Exports OMR processing results to JSON format with confidence scores.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the JSON exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.include_confidence = self.config.get('include_confidence', True)
        self.include_bboxes = self.config.get('include_bboxes', True)
        self.include_raw_symbols = self.config.get('include_raw_symbols', True)
        self.include_metadata = self.config.get('include_metadata', True)
        self.pretty_print = self.config.get('pretty_print', True)
    
    def export(self, results: Dict) -> Dict:
        """
        Export OMR results to JSON-serializable format.
        
        Args:
            results: Complete OMR processing results
            
        Returns:
            JSON-serializable dictionary
        """
        logger.info("Exporting results to JSON format")
        
        export_data = {
            'export_info': self._create_export_info(),
            'input_metadata': self._extract_input_metadata(results),
            'processing_summary': self._create_processing_summary(results),
        }
        
        # Add staff information
        if 'staff_info' in results:
            export_data['staff_detection'] = self._export_staff_detection(results['staff_info'])
        
        # Add symbol detection results
        if 'symbols' in results:
            export_data['symbol_detection'] = self._export_symbol_detection(results['symbols'])
        
        # Add musical reconstruction results
        if 'musical_elements' in results:
            export_data['musical_reconstruction'] = self._export_musical_reconstruction(
                results['musical_elements']
            )
        
        # Add confidence analysis
        if self.include_confidence:
            export_data['confidence_analysis'] = self._create_confidence_analysis(results)
        
        # Add processing metadata
        if self.include_metadata:
            export_data['processing_metadata'] = self._extract_processing_metadata(results)
        
        logger.info("JSON export completed")
        return export_data
    
    def _create_export_info(self) -> Dict:
        """Create export information section."""
        return {
            'export_timestamp': datetime.now().isoformat(),
            'export_version': '1.0.0',
            'format_version': '1.0',
            'exporter': 'OMR Pipeline JSON Exporter'
        }
    
    def _extract_input_metadata(self, results: Dict) -> Dict:
        """Extract metadata about the input image."""
        metadata = {
            'input_path': results.get('input_path', 'unknown'),
            'image_dimensions': None,
            'preprocessing_applied': []
        }
        
        # Extract image dimensions if available
        if 'original_image' in results:
            image = results['original_image']
            if hasattr(image, 'shape'):
                metadata['image_dimensions'] = {
                    'height': int(image.shape[0]),
                    'width': int(image.shape[1]),
                    'channels': int(image.shape[2]) if len(image.shape) > 2 else 1
                }
        
        # Extract preprocessing information
        if 'metadata' in results and 'processing_stages' in results['metadata']:
            metadata['preprocessing_applied'] = results['metadata']['processing_stages']
        
        return metadata
    
    def _create_processing_summary(self, results: Dict) -> Dict:
        """Create a summary of processing results."""
        summary = {
            'total_staves_detected': 0,
            'total_symbols_detected': 0,
            'total_notes_reconstructed': 0,
            'total_measures': 0,
            'processing_stages_completed': []
        }
        
        # Count staves
        if 'staff_info' in results:
            summary['total_staves_detected'] = len(results['staff_info'])
        
        # Count symbols
        if 'symbols' in results:
            summary['total_symbols_detected'] = len(results['symbols'])
        
        # Count notes and measures
        if 'musical_elements' in results:
            musical_data = results['musical_elements']
            if 'staves' in musical_data:
                total_notes = 0
                total_measures = 0
                
                for staff_data in musical_data['staves']:
                    if 'measures' in staff_data:
                        total_measures += len(staff_data['measures'])
                        for measure in staff_data['measures']:
                            for element in measure.elements:
                                if isinstance(element, Note):
                                    total_notes += 1
                
                summary['total_notes_reconstructed'] = total_notes
                summary['total_measures'] = total_measures
        
        # Processing stages
        if 'metadata' in results and 'processing_stages' in results['metadata']:
            summary['processing_stages_completed'] = results['metadata']['processing_stages']
        
        return summary
    
    def _export_staff_detection(self, staff_info: List[Staff]) -> Dict:
        """Export staff detection results."""
        staff_data = {
            'num_staves_detected': len(staff_info),
            'staves': []
        }
        
        for i, staff in enumerate(staff_info):
            staff_dict = {
                'staff_index': i,
                'num_lines': len(staff.lines),
                'y_bounds': {
                    'top': float(staff.y_top),
                    'bottom': float(staff.y_bottom)
                },
                'x_bounds': {
                    'start': float(staff.x_start),
                    'end': float(staff.x_end)
                },
                'line_spacing': float(staff.line_spacing),
                'confidence': float(staff.confidence),
                'lines': []
            }
            
            # Add individual line information
            for j, line in enumerate(staff.lines):
                line_dict = {
                    'line_index': j,
                    'y_position': float(line.y_position),
                    'x_start': float(line.x_start),
                    'x_end': float(line.x_end),
                    'thickness': int(line.thickness),
                    'confidence': float(line.confidence)
                }
                staff_dict['lines'].append(line_dict)
            
            staff_data['staves'].append(staff_dict)
        
        return staff_data
    
    def _export_symbol_detection(self, symbols: List[DetectedSymbol]) -> Dict:
        """Export symbol detection results."""
        symbol_data = {
            'num_symbols_detected': len(symbols),
            'symbol_types': {},
            'symbols': []
        }
        
        # Count symbol types
        for symbol in symbols:
            if symbol.class_name not in symbol_data['symbol_types']:
                symbol_data['symbol_types'][symbol.class_name] = 0
            symbol_data['symbol_types'][symbol.class_name] += 1
        
        # Export individual symbols
        for i, symbol in enumerate(symbols):
            symbol_dict = {
                'symbol_index': i,
                'class_name': symbol.class_name,
                'confidence': float(symbol.confidence),
                'center': {
                    'x': float(symbol.center[0]),
                    'y': float(symbol.center[1])
                }
            }
            
            # Add bounding box if requested
            if self.include_bboxes:
                symbol_dict['bbox'] = {
                    'x': int(symbol.bbox[0]),
                    'y': int(symbol.bbox[1]),
                    'width': int(symbol.bbox[2]),
                    'height': int(symbol.bbox[3])
                }
            
            # Add staff association
            if symbol.staff_line is not None:
                symbol_dict['staff_line'] = int(symbol.staff_line)
            
            # Add pitch if available
            if symbol.pitch:
                symbol_dict['pitch'] = symbol.pitch
            
            # Add additional data
            if symbol.additional_data:
                symbol_dict['additional_data'] = self._serialize_additional_data(
                    symbol.additional_data
                )
            
            symbol_data['symbols'].append(symbol_dict)
        
        return symbol_data
    
    def _export_musical_reconstruction(self, musical_elements: Dict) -> Dict:
        """Export musical reconstruction results."""
        reconstruction_data = {
            'num_staves': 0,
            'staves': []
        }
        
        if 'staves' not in musical_elements:
            return reconstruction_data
        
        reconstruction_data['num_staves'] = len(musical_elements['staves'])
        
        for staff_data in musical_elements['staves']:
            staff_dict = {
                'staff_index': staff_data.get('staff_index', 0),
                'clef': self._export_clef(staff_data.get('clef')),
                'time_signature': self._export_time_signature(staff_data.get('time_signature')),
                'key_signature': self._export_key_signature(staff_data.get('key_signature')),
                'measures': self._export_measures(staff_data.get('measures', [])),
                'voices': self._export_voices(staff_data.get('voices', []))
            }
            
            reconstruction_data['staves'].append(staff_dict)
        
        return reconstruction_data
    
    def _export_clef(self, clef) -> Optional[Dict]:
        """Export clef information."""
        if not clef:
            return None
        
        return {
            'type': clef.clef_type,
            'confidence': float(clef.confidence),
            'position': {
                'x': float(clef.x_position),
                'y': float(clef.y_position)
            }
        }
    
    def _export_time_signature(self, time_sig) -> Optional[Dict]:
        """Export time signature information."""
        if not time_sig:
            return None
        
        return {
            'numerator': int(time_sig.numerator),
            'denominator': int(time_sig.denominator),
            'confidence': float(time_sig.confidence),
            'position': {
                'x': float(time_sig.x_position),
                'y': float(time_sig.y_position)
            }
        }
    
    def _export_key_signature(self, key_sig) -> Optional[Dict]:
        """Export key signature information."""
        if not key_sig:
            return None
        
        return {
            'sharps': int(key_sig.sharps),
            'key': key_sig.key,
            'confidence': float(key_sig.confidence),
            'position': {
                'x': float(key_sig.x_position),
                'y': float(key_sig.y_position)
            }
        }
    
    def _export_measures(self, measures: List[Measure]) -> List[Dict]:
        """Export measures information."""
        measures_data = []
        
        for measure in measures:
            measure_dict = {
                'index': int(measure.index),
                'x_bounds': {
                    'start': float(measure.x_start),
                    'end': float(measure.x_end)
                },
                'num_elements': len(measure.elements),
                'elements': self._export_musical_elements(measure.elements)
            }
            
            measures_data.append(measure_dict)
        
        return measures_data
    
    def _export_voices(self, voices: List[Voice]) -> List[Dict]:
        """Export voice information."""
        voices_data = []
        
        for voice in voices:
            voice_dict = {
                'voice_number': int(voice.voice_number),
                'staff_index': int(voice.staff_index),
                'num_elements': len(voice.elements),
                'elements': self._export_musical_elements(voice.elements)
            }
            
            voices_data.append(voice_dict)
        
        return voices_data
    
    def _export_musical_elements(self, elements: List[MusicalElement]) -> List[Dict]:
        """Export musical elements."""
        elements_data = []
        
        for element in elements:
            element_dict = {
                'element_type': element.element_type,
                'confidence': float(element.confidence),
                'position': {
                    'x': float(element.x_position),
                    'y': float(element.y_position)
                },
                'staff_index': int(element.staff_index)
            }
            
            if element.measure_index is not None:
                element_dict['measure_index'] = int(element.measure_index)
            
            # Add type-specific information
            if isinstance(element, Note):
                element_dict.update({
                    'pitch': element.pitch,
                    'duration': float(element.duration),
                    'is_rest': bool(element.is_rest),
                    'dotted': bool(element.dotted),
                    'voice': int(element.voice),
                    'tied_to_next': bool(element.tied_to_next),
                    'tied_from_prev': bool(element.tied_from_prev)
                })
                
                if element.accidental:
                    element_dict['accidental'] = element.accidental
                
                if element.stem_direction:
                    element_dict['stem_direction'] = element.stem_direction
                
                if element.beam_group is not None:
                    element_dict['beam_group'] = int(element.beam_group)
            
            elements_data.append(element_dict)
        
        return elements_data
    
    def _create_confidence_analysis(self, results: Dict) -> Dict:
        """Create confidence analysis section."""
        analysis = {
            'overall_confidence': 0.0,
            'staff_detection_confidence': 0.0,
            'symbol_detection_confidence': 0.0,
            'reconstruction_confidence': 0.0,
            'confidence_distribution': {},
            'low_confidence_items': []
        }
        
        confidences = []
        
        # Collect staff confidences
        if 'staff_info' in results:
            staff_confidences = [staff.confidence for staff in results['staff_info']]
            if staff_confidences:
                analysis['staff_detection_confidence'] = float(np.mean(staff_confidences))
                confidences.extend(staff_confidences)
        
        # Collect symbol confidences
        if 'symbols' in results:
            symbol_confidences = [symbol.confidence for symbol in results['symbols']]
            if symbol_confidences:
                analysis['symbol_detection_confidence'] = float(np.mean(symbol_confidences))
                confidences.extend(symbol_confidences)
                
                # Find low confidence symbols
                for symbol in results['symbols']:
                    if symbol.confidence < 0.7:  # Threshold for low confidence
                        analysis['low_confidence_items'].append({
                            'type': 'symbol',
                            'class_name': symbol.class_name,
                            'confidence': float(symbol.confidence),
                            'position': {
                                'x': float(symbol.center[0]),
                                'y': float(symbol.center[1])
                            }
                        })
        
        # Collect reconstruction confidences
        if 'musical_elements' in results and 'staves' in results['musical_elements']:
            reconstruction_confidences = []
            for staff_data in results['musical_elements']['staves']:
                for element in staff_data.get('elements', []):
                    reconstruction_confidences.append(element.confidence)
            
            if reconstruction_confidences:
                analysis['reconstruction_confidence'] = float(np.mean(reconstruction_confidences))
                confidences.extend(reconstruction_confidences)
        
        # Overall confidence
        if confidences:
            analysis['overall_confidence'] = float(np.mean(confidences))
        
        # Confidence distribution
        if confidences:
            confidence_ranges = {
                'very_high': len([c for c in confidences if c >= 0.9]),
                'high': len([c for c in confidences if 0.8 <= c < 0.9]),
                'medium': len([c for c in confidences if 0.6 <= c < 0.8]),
                'low': len([c for c in confidences if 0.4 <= c < 0.6]),
                'very_low': len([c for c in confidences if c < 0.4])
            }
            analysis['confidence_distribution'] = confidence_ranges
        
        return analysis
    
    def _extract_processing_metadata(self, results: Dict) -> Dict:
        """Extract processing metadata."""
        metadata = {
            'processing_timestamp': datetime.now().isoformat(),
            'config': results.get('metadata', {}).get('config', {}),
            'processing_stages': results.get('metadata', {}).get('processing_stages', []),
            'image_statistics': {}
        }
        
        # Add image statistics if available
        if 'preprocessed_image' in results:
            image = results['preprocessed_image']
            if hasattr(image, 'shape'):
                metadata['image_statistics'] = {
                    'dimensions': list(image.shape),
                    'mean_intensity': float(np.mean(image)) if hasattr(np, 'mean') else 0.0,
                    'std_intensity': float(np.std(image)) if hasattr(np, 'std') else 0.0
                }
        
        return metadata
    
    def _serialize_additional_data(self, data: Dict) -> Dict:
        """Serialize additional data, handling non-JSON types."""
        serialized = {}
        
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, (list, tuple)):
                serialized[key] = list(value)
            elif isinstance(value, dict):
                serialized[key] = self._serialize_additional_data(value)
            elif hasattr(value, '__dict__'):
                # Handle objects with attributes
                serialized[key] = str(value)
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def save_to_file(self, export_data: Dict, file_path: str):
        """
        Save export data to JSON file.
        
        Args:
            export_data: Data to export
            file_path: Output file path
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            if self.pretty_print:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(export_data, f, ensure_ascii=False)
        
        logger.info(f"JSON export saved to {file_path}")


def test_json_exporter():
    """Test function for JSON exporter."""
    from ..detection.symbol_detector import DetectedSymbol
    from ..detection.staff_detector import Staff, StaffLine
    from ..reconstruction.music_reconstructor import Note
    
    # Create test data
    staff_line = StaffLine(
        y_position=200,
        x_start=50,
        x_end=750,
        thickness=3,
        confidence=0.95
    )
    
    staff = Staff(
        lines=[staff_line],
        y_top=180,
        y_bottom=220,
        x_start=50,
        x_end=750,
        line_spacing=20,
        confidence=0.9
    )
    
    symbol = DetectedSymbol(
        class_name='quarter_note',
        confidence=0.85,
        bbox=(100, 190, 20, 30),
        center=(110, 205),
        staff_line=0,
        pitch='C4'
    )
    
    note = Note(
        pitch='C4',
        duration=1.0,
        confidence=0.8,
        x_position=110,
        y_position=205,
        staff_index=0
    )
    
    # Create test results
    results = {
        'input_path': 'test_sheet.png',
        'staff_info': [staff],
        'symbols': [symbol],
        'musical_elements': {
            'staves': [{
                'staff_index': 0,
                'elements': [note],
                'measures': [],
                'voices': []
            }]
        },
        'metadata': {
            'processing_stages': ['preprocessing', 'staff_detection', 'symbol_detection']
        }
    }
    
    # Test export
    exporter = JSONExporter()
    export_data = exporter.export(results)
    
    print("JSON Export Sample:")
    print(json.dumps(export_data, indent=2)[:500] + "...")


if __name__ == "__main__":
    test_json_exporter()