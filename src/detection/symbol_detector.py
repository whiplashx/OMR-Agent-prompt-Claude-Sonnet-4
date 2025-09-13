"""
Symbol Detection Module
======================

Detects and classifies musical symbols using YOLO object detection.
Handles notes, rests, clefs, accidentals, and other musical notation elements.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)


@dataclass
class DetectedSymbol:
    """Represents a detected musical symbol."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[float, float]      # (x, y) center coordinates
    staff_line: Optional[int] = None  # Which staff line this symbol is on/near
    pitch: Optional[str] = None       # Detected pitch (e.g., "C4")
    additional_data: Optional[Dict] = None


class SymbolDetector:
    """
    Detects and classifies musical symbols in sheet music images.
    """
    
    # Define musical symbol classes
    SYMBOL_CLASSES = {
        # Notes
        0: 'whole_note',
        1: 'half_note', 
        2: 'quarter_note',
        3: 'eighth_note',
        4: 'sixteenth_note',
        5: 'thirty_second_note',
        
        # Rests
        6: 'whole_rest',
        7: 'half_rest',
        8: 'quarter_rest', 
        9: 'eighth_rest',
        10: 'sixteenth_rest',
        
        # Clefs
        11: 'treble_clef',
        12: 'bass_clef',
        13: 'alto_clef',
        14: 'tenor_clef',
        
        # Accidentals
        15: 'sharp',
        16: 'flat',
        17: 'natural',
        18: 'double_sharp',
        19: 'double_flat',
        
        # Time signatures
        20: 'time_signature_4_4',
        21: 'time_signature_3_4',
        22: 'time_signature_2_4',
        23: 'time_signature_6_8',
        24: 'time_signature_common',
        25: 'time_signature_cut',
        
        # Key signatures  
        26: 'key_signature',
        
        # Articulations and dynamics
        27: 'dot',
        28: 'tie',
        29: 'slur',
        30: 'beam',
        31: 'accent',
        32: 'staccato',
        33: 'tenuto',
        
        # Dynamics
        34: 'forte',
        35: 'piano',
        36: 'mezzoforte',
        37: 'fortissimo',
        38: 'pianissimo',
        
        # Barlines
        39: 'barline',
        40: 'double_barline',
        41: 'final_barline',
        42: 'repeat_start',
        43: 'repeat_end',
        
        # Other symbols
        44: 'fermata',
        45: 'grace_note',
        46: 'tremolo',
        47: 'turn',
        48: 'mordent',
        49: 'trill'
    }
    
    def __init__(self, config: Dict = None):
        """
        Initialize the symbol detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_path = self.config.get('model_path', 'models/symbol_detector.pt')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.nms_threshold = self.config.get('nms_threshold', 0.4)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model for symbol detection."""
        if YOLO is None:
            logger.error("YOLO not available. Please install ultralytics package.")
            return
        
        model_path = Path(self.model_path)
        
        if model_path.exists():
            logger.info(f"Loading trained model from {self.model_path}")
            self.model = YOLO(self.model_path)
        else:
            logger.warning(f"Model file {self.model_path} not found. Using pre-trained YOLOv8 as base.")
            # Start with a pre-trained model that we can fine-tune
            self.model = YOLO('yolov8n.pt')  # Nano version for faster inference
            
        if self.model:
            logger.info(f"Model loaded successfully on device: {self.device}")
    
    def detect_symbols(self, image: np.ndarray, staff_info: List = None) -> List[DetectedSymbol]:
        """
        Detect musical symbols in the image.
        
        Args:
            image: Preprocessed image with staff lines removed
            staff_info: Information about detected staves
            
        Returns:
            List of detected symbols with their properties
        """
        if self.model is None:
            logger.error("No model available for symbol detection")
            return []
        
        logger.info("Starting symbol detection")
        
        # Prepare image for YOLO (convert to RGB if needed)
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Run inference
        results = self.model(
            rgb_image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            device=self.device
        )
        
        # Process results
        detected_symbols = []
        
        if results and len(results) > 0:
            result = results[0]  # First (and only) image result
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                
                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    
                    # Convert to our bbox format (x, y, width, height)
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Get class name
                    class_name = self.SYMBOL_CLASSES.get(class_id, f'unknown_{class_id}')
                    
                    # Associate with staff line if staff info is available
                    staff_line = None
                    if staff_info:
                        staff_line = self._associate_with_staff_line(center[1], staff_info)
                    
                    # Create detected symbol
                    symbol = DetectedSymbol(
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=bbox,
                        center=center,
                        staff_line=staff_line
                    )
                    
                    detected_symbols.append(symbol)
        
        # Post-process detections
        detected_symbols = self._post_process_detections(detected_symbols, image, staff_info)
        
        logger.info(f"Detected {len(detected_symbols)} symbols")
        return detected_symbols
    
    def _associate_with_staff_line(self, y_position: float, staff_info: List) -> Optional[int]:
        """
        Associate a symbol with the nearest staff line.
        
        Args:
            y_position: Y coordinate of the symbol center
            staff_info: List of detected staves
            
        Returns:
            Index of the nearest staff line (0-4 within each staff)
        """
        min_distance = float('inf')
        closest_line = None
        
        for staff in staff_info:
            for i, line in enumerate(staff.lines):
                distance = abs(y_position - line.y_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_line = i
        
        # Only associate if the symbol is reasonably close to a staff line
        staff_space = getattr(staff_info[0], 'line_spacing', 20) if staff_info else 20
        if min_distance < staff_space * 1.5:  # Within 1.5 staff spaces
            return closest_line
        
        return None
    
    def _post_process_detections(self, symbols: List[DetectedSymbol], 
                               image: np.ndarray, staff_info: List) -> List[DetectedSymbol]:
        """
        Post-process detected symbols to improve accuracy and add metadata.
        
        Args:
            symbols: List of detected symbols
            image: Original image
            staff_info: Staff information
            
        Returns:
            Post-processed list of symbols
        """
        processed_symbols = []
        
        for symbol in symbols:
            # Skip low-confidence detections
            if symbol.confidence < self.confidence_threshold:
                continue
            
            # Add pitch information for note symbols
            if 'note' in symbol.class_name and staff_info:
                symbol.pitch = self._estimate_pitch(symbol, staff_info)
            
            # Enhance symbol with additional analysis
            symbol.additional_data = self._analyze_symbol_context(symbol, image, symbols)
            
            processed_symbols.append(symbol)
        
        # Remove duplicate detections
        processed_symbols = self._remove_duplicate_detections(processed_symbols)
        
        # Sort symbols by horizontal position (reading order)
        processed_symbols.sort(key=lambda s: s.center[0])
        
        return processed_symbols
    
    def _estimate_pitch(self, symbol: DetectedSymbol, staff_info: List) -> Optional[str]:
        """
        Estimate the musical pitch of a note symbol based on its position.
        
        Args:
            symbol: Detected note symbol
            staff_info: Staff information
            
        Returns:
            Estimated pitch string (e.g., "C4", "F#5")
        """
        if not staff_info or symbol.staff_line is None:
            return None
        
        # Find which staff this symbol belongs to
        symbol_y = symbol.center[1]
        staff = None
        
        for s in staff_info:
            if s.y_top <= symbol_y <= s.y_bottom + s.line_spacing * 2:
                staff = s
                break
        
        if not staff:
            return None
        
        # Calculate position relative to staff
        staff_middle = staff.lines[2].y_position  # Middle line (B4 in treble clef)
        line_spacing = staff.line_spacing
        
        # Calculate steps from middle line
        steps_from_middle = round((staff_middle - symbol_y) / (line_spacing / 2))
        
        # Map to pitch (assuming treble clef for now)
        # Middle line is B4, each step is a semitone up/down
        treble_clef_pitches = [
            'C8', 'B7', 'A7', 'G7', 'F7', 'E7', 'D7', 'C7',
            'B6', 'A6', 'G6', 'F6', 'E6', 'D6', 'C6', 'B5',
            'A5', 'G5', 'F5', 'E5', 'D5', 'C5', 'B4', 'A4',
            'G4', 'F4', 'E4', 'D4', 'C4', 'B3', 'A3', 'G3'
        ]
        
        # Adjust index based on steps from middle
        middle_index = 22  # B4 position in the array
        pitch_index = middle_index - steps_from_middle
        
        if 0 <= pitch_index < len(treble_clef_pitches):
            return treble_clef_pitches[pitch_index]
        
        return None
    
    def _analyze_symbol_context(self, symbol: DetectedSymbol, 
                              image: np.ndarray, all_symbols: List[DetectedSymbol]) -> Dict:
        """
        Analyze the context around a symbol for additional information.
        
        Args:
            symbol: Symbol to analyze
            image: Original image
            all_symbols: All detected symbols
            
        Returns:
            Dictionary with additional context information
        """
        context = {}
        
        # Look for nearby symbols that might be related
        nearby_symbols = []
        search_radius = 50  # pixels
        
        for other_symbol in all_symbols:
            if other_symbol == symbol:
                continue
            
            distance = np.sqrt(
                (symbol.center[0] - other_symbol.center[0])**2 + 
                (symbol.center[1] - other_symbol.center[1])**2
            )
            
            if distance < search_radius:
                nearby_symbols.append({
                    'symbol': other_symbol,
                    'distance': distance,
                    'relative_position': (
                        other_symbol.center[0] - symbol.center[0],
                        other_symbol.center[1] - symbol.center[1]
                    )
                })
        
        context['nearby_symbols'] = nearby_symbols
        
        # Look for augmentation dots
        if 'note' in symbol.class_name:
            dots = [s for s in nearby_symbols 
                   if s['symbol'].class_name == 'dot' 
                   and abs(s['relative_position'][1]) < 10  # Same height
                   and s['relative_position'][0] > 0]  # To the right
            context['augmentation_dots'] = len(dots)
        
        # Look for accidentals
        if 'note' in symbol.class_name:
            accidentals = [s for s in nearby_symbols 
                         if s['symbol'].class_name in ['sharp', 'flat', 'natural']
                         and abs(s['relative_position'][1]) < 20  # Similar height
                         and s['relative_position'][0] < 0]  # To the left
            if accidentals:
                context['accidental'] = accidentals[0]['symbol'].class_name
        
        return context
    
    def _remove_duplicate_detections(self, symbols: List[DetectedSymbol]) -> List[DetectedSymbol]:
        """
        Remove duplicate detections using Non-Maximum Suppression.
        
        Args:
            symbols: List of detected symbols
            
        Returns:
            List with duplicates removed
        """
        if len(symbols) <= 1:
            return symbols
        
        # Group symbols by class
        class_groups = {}
        for symbol in symbols:
            if symbol.class_name not in class_groups:
                class_groups[symbol.class_name] = []
            class_groups[symbol.class_name].append(symbol)
        
        # Apply NMS within each class
        filtered_symbols = []
        
        for class_name, class_symbols in class_groups.items():
            if len(class_symbols) == 1:
                filtered_symbols.extend(class_symbols)
                continue
            
            # Prepare data for NMS
            boxes = np.array([
                [s.bbox[0], s.bbox[1], s.bbox[0] + s.bbox[2], s.bbox[1] + s.bbox[3]]
                for s in class_symbols
            ])
            scores = np.array([s.confidence for s in class_symbols])
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.confidence_threshold,
                self.nms_threshold
            )
            
            if len(indices) > 0:
                for i in indices.flatten():
                    filtered_symbols.append(class_symbols[i])
        
        return filtered_symbols
    
    def visualize_detections(self, image: np.ndarray, symbols: List[DetectedSymbol]) -> np.ndarray:
        """
        Create a visualization of detected symbols on the image.
        
        Args:
            image: Original image
            symbols: List of detected symbols
            
        Returns:
            Image with detection overlays
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Define colors for different symbol types
        colors = {
            'note': (0, 255, 0),      # Green
            'rest': (255, 0, 0),      # Red  
            'clef': (0, 0, 255),      # Blue
            'accidental': (255, 255, 0),  # Cyan
            'time_signature': (255, 0, 255),  # Magenta
            'default': (128, 128, 128)  # Gray
        }
        
        for symbol in symbols:
            x, y, w, h = symbol.bbox
            
            # Choose color based on symbol type
            color = colors['default']
            for key in colors:
                if key in symbol.class_name:
                    color = colors[key]
                    break
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw class name and confidence
            label = f"{symbol.class_name}: {symbol.confidence:.2f}"
            if symbol.pitch:
                label += f" ({symbol.pitch})"
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(
                vis_image,
                (x, y - text_height - baseline),
                (x + text_width, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                label,
                (x, y - baseline),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return vis_image
    
    def save_detection_results(self, symbols: List[DetectedSymbol], output_path: str):
        """
        Save detection results to a file.
        
        Args:
            symbols: List of detected symbols
            output_path: Path to save results
        """
        import json
        
        # Convert symbols to serializable format
        results = []
        for symbol in symbols:
            result = {
                'class_name': symbol.class_name,
                'confidence': symbol.confidence,
                'bbox': symbol.bbox,
                'center': symbol.center,
                'staff_line': symbol.staff_line,
                'pitch': symbol.pitch,
                'additional_data': symbol.additional_data
            }
            results.append(result)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detection results saved to {output_path}")


def create_detection_package_init():
    """Create __init__.py for detection package."""
    return '''"""
Detection Package
================

Musical symbol detection and classification.
"""

from .staff_detector import StaffDetector, Staff, StaffLine
from .symbol_detector import SymbolDetector, DetectedSymbol

__all__ = ['StaffDetector', 'Staff', 'StaffLine', 'SymbolDetector', 'DetectedSymbol']
'''


if __name__ == "__main__":
    # Test function
    detector = SymbolDetector()
    print(f"Symbol detector initialized with {len(detector.SYMBOL_CLASSES)} symbol classes")
    print("Available symbol classes:")
    for class_id, class_name in detector.SYMBOL_CLASSES.items():
        print(f"  {class_id}: {class_name}")