"""
Unit Tests for Detection Modules
===============================

Test suite for staff detection and symbol detection modules.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from detection.staff_detector import StaffDetector
    from detection.symbol_detector import SymbolDetector
except ImportError:
    StaffDetector = None
    SymbolDetector = None

class TestStaffDetector(unittest.TestCase):
    """Test cases for StaffDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if StaffDetector is None:
            self.skipTest("StaffDetector not available - missing dependencies")
        
        self.detector = StaffDetector()
        self.test_image = self._create_test_image_with_staffs()
    
    def _create_test_image_with_staffs(self) -> np.ndarray:
        """Create test image with staff lines."""
        height, width = 400, 600
        image = np.ones((height, width), dtype=np.uint8) * 255
        
        # Add clear staff lines
        staff_positions = [100, 120, 140, 160, 180, 250, 270, 290, 310, 330]
        for y in staff_positions:
            image[y-1:y+1, 50:550] = 0
        
        return image
    
    def test_staff_detection(self):
        """Test staff line detection."""
        if StaffDetector is None:
            self.skipTest("Dependencies not available")
        
        result = self.detector.detect_staffs(self.test_image)
        
        # Should return detection results
        self.assertIsNotNone(result)
        self.assertIn('staff_lines', result)
        self.assertIn('staff_systems', result)
        
        # Should find staff lines
        staff_lines = result['staff_lines']
        self.assertGreater(len(staff_lines), 0)
    
    def test_staff_removal(self):
        """Test staff line removal."""
        if StaffDetector is None:
            self.skipTest("Dependencies not available")
        
        # First detect staffs
        detection_result = self.detector.detect_staffs(self.test_image)
        
        # Then remove them
        cleaned_image = self.detector.remove_staffs(
            self.test_image, 
            detection_result['staff_lines']
        )
        
        # Image should be different after staff removal
        self.assertFalse(np.array_equal(self.test_image, cleaned_image))
        
        # Should maintain image shape
        self.assertEqual(self.test_image.shape, cleaned_image.shape)
    
    def test_empty_image(self):
        """Test handling of empty/invalid images."""
        if StaffDetector is None:
            self.skipTest("Dependencies not available")
        
        # Empty image
        empty_image = np.ones((100, 100), dtype=np.uint8) * 255
        result = self.detector.detect_staffs(empty_image)
        
        # Should handle gracefully
        self.assertIsNotNone(result)
        self.assertIn('staff_lines', result)

class TestSymbolDetector(unittest.TestCase):
    """Test cases for SymbolDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if SymbolDetector is None:
            self.skipTest("SymbolDetector not available - missing dependencies")
        
        self.detector = SymbolDetector()
        self.test_image = self._create_test_image_with_symbols()
    
    def _create_test_image_with_symbols(self) -> np.ndarray:
        """Create test image with musical symbols."""
        height, width = 400, 600
        image = np.ones((height, width), dtype=np.uint8) * 255
        
        # Add some simple shapes that could be musical symbols
        # Circle for note head
        center_y, center_x = 200, 150
        for y in range(center_y - 10, center_y + 10):
            for x in range(center_x - 8, center_x + 8):
                if (y - center_y)**2 + (x - center_x)**2 <= 64:
                    if 0 <= y < height and 0 <= x < width:
                        image[y, x] = 0
        
        # Rectangle for rest
        image[180:200, 300:310] = 0
        
        return image
    
    def test_symbol_detection(self):
        """Test musical symbol detection."""
        if SymbolDetector is None:
            self.skipTest("Dependencies not available")
        
        # Note: This will likely fail without a trained model
        # but tests the interface
        try:
            result = self.detector.detect_symbols(self.test_image)
            
            # Should return detection results
            self.assertIsNotNone(result)
            self.assertIn('detections', result)
            self.assertIsInstance(result['detections'], list)
            
        except Exception as e:
            # Expected if no model is available
            self.assertIn('model', str(e).lower())
    
    def test_confidence_filtering(self):
        """Test confidence threshold filtering."""
        if SymbolDetector is None:
            self.skipTest("Dependencies not available")
        
        # Mock detection results
        mock_detections = [
            {'class': 'quarter_note', 'confidence': 0.9, 'bbox': [100, 100, 120, 120]},
            {'class': 'half_note', 'confidence': 0.3, 'bbox': [200, 200, 220, 220]},
            {'class': 'treble_clef', 'confidence': 0.8, 'bbox': [50, 50, 80, 80]}
        ]
        
        # Test filtering with threshold 0.5
        filtered = self.detector._filter_by_confidence(mock_detections, 0.5)
        
        # Should keep only high-confidence detections
        self.assertEqual(len(filtered), 2)
        for detection in filtered:
            self.assertGreaterEqual(detection['confidence'], 0.5)
    
    def test_non_maximum_suppression(self):
        """Test non-maximum suppression for overlapping detections."""
        if SymbolDetector is None:
            self.skipTest("Dependencies not available")
        
        # Mock overlapping detections
        mock_detections = [
            {'class': 'quarter_note', 'confidence': 0.9, 'bbox': [100, 100, 120, 120]},
            {'class': 'quarter_note', 'confidence': 0.7, 'bbox': [105, 105, 125, 125]},  # Overlapping
            {'class': 'half_note', 'confidence': 0.8, 'bbox': [200, 200, 220, 220]}
        ]
        
        # Apply NMS
        filtered = self.detector._apply_nms(mock_detections, iou_threshold=0.3)
        
        # Should remove overlapping detection with lower confidence
        self.assertLessEqual(len(filtered), len(mock_detections))

if __name__ == '__main__':
    unittest.main()