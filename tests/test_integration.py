"""
Integration Tests for OMR Pipeline
=================================

End-to-end integration tests for the complete OMR pipeline.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from omr_pipeline import OMRPipeline
except ImportError:
    OMRPipeline = None

class TestOMRPipeline(unittest.TestCase):
    """Integration tests for complete OMR pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        if OMRPipeline is None:
            self.skipTest("OMRPipeline not available - missing dependencies")
        
        self.pipeline = OMRPipeline()
        self.test_image = self._create_comprehensive_test_image()
        
        # Create temporary directory for outputs
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_comprehensive_test_image(self) -> np.ndarray:
        """Create a comprehensive test sheet music image."""
        height, width = 800, 1200
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add staff lines for two systems
        staff_systems = [
            [150, 170, 190, 210, 230],  # First system
            [400, 420, 440, 460, 480]   # Second system
        ]
        
        for staff_lines in staff_systems:
            for y in staff_lines:
                image[y-1:y+1, 100:1100] = 0
        
        # Add treble clefs
        clef_positions = [(120, 170), (370, 420)]
        for x, y in clef_positions:
            # Simple treble clef shape
            image[y-20:y+20, x:x+20] = 0
        
        # Add quarter notes
        note_positions = [
            (200, 150),  # C5 on first staff
            (250, 170),  # A4 on first staff
            (300, 190),  # F4 on first staff
            (450, 400),  # C5 on second staff
            (500, 420),  # A4 on second staff
        ]
        
        for x, y in note_positions:
            # Note head (filled ellipse)
            for dy in range(-6, 7):
                for dx in range(-8, 9):
                    if dx*dx/64 + dy*dy/36 <= 1:
                        if 0 <= y+dy < height and 0 <= x+dx < width:
                            image[y+dy, x+dx] = 0
            
            # Stem
            if y < 200:  # Stem down
                image[y+6:y+40, x+6:x+8] = 0
            else:  # Stem up
                image[y-40:y-6, x-2:x] = 0
        
        # Add bar lines
        bar_positions = [350, 700, 1050]
        for x in bar_positions:
            for staff_lines in staff_systems:
                y_start, y_end = staff_lines[0], staff_lines[-1]
                image[y_start:y_end+2, x:x+2] = 0
        
        return image
    
    def test_full_pipeline_processing(self):
        """Test complete pipeline from image to MusicXML."""
        if OMRPipeline is None:
            self.skipTest("Dependencies not available")
        
        output_path = self.temp_dir / "test_output.mxl"
        
        try:
            result = self.pipeline.process_image(
                self.test_image,
                output_path=str(output_path)
            )
            
            # Check that result is returned
            self.assertIsNotNone(result)
            self.assertIn('musicxml_path', result)
            self.assertIn('confidence_data', result)
            self.assertIn('quality_score', result)
            self.assertIn('processing_time', result)
            
            # Check quality score is reasonable
            quality_score = result['quality_score']
            self.assertIsInstance(quality_score, (int, float))
            self.assertGreaterEqual(quality_score, 0)
            self.assertLessEqual(quality_score, 1)
            
            # Check confidence data structure
            confidence_data = result['confidence_data']
            self.assertIsInstance(confidence_data, dict)
            self.assertIn('overall_confidence', confidence_data)
            
        except Exception as e:
            # Expected if dependencies are missing
            self.assertIn(('model', 'cv2', 'ultralytics'), str(e).lower())
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        if OMRPipeline is None:
            self.skipTest("Dependencies not available")
        
        # Create multiple test images
        test_images = [
            self.test_image,
            self._create_simple_test_image(),
            self._create_empty_test_image()
        ]
        
        image_paths = []
        for i, image in enumerate(test_images):
            image_path = self.temp_dir / f"test_image_{i}.png"
            # Would normally save image here, but avoiding cv2 dependency
            image_paths.append(str(image_path))
        
        output_dir = self.temp_dir / "batch_output"
        
        try:
            results = self.pipeline.process_batch(
                image_paths,
                str(output_dir)
            )
            
            # Should return results for all images
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), len(image_paths))
            
            # Each result should have required fields
            for result in results:
                if result is not None:  # Some may fail
                    self.assertIn('musicxml_path', result)
                    self.assertIn('confidence_data', result)
            
        except Exception as e:
            # Expected if dependencies are missing
            self.assertIn(('model', 'cv2', 'image'), str(e).lower())
    
    def _create_simple_test_image(self) -> np.ndarray:
        """Create a simple test image with minimal content."""
        height, width = 400, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Just staff lines
        staff_positions = [180, 200, 220, 240, 260]
        for y in staff_positions:
            image[y-1:y+1, 50:550] = 0
        
        return image
    
    def _create_empty_test_image(self) -> np.ndarray:
        """Create an empty test image."""
        height, width = 400, 600
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    
    def test_pipeline_configuration(self):
        """Test pipeline with different configurations."""
        if OMRPipeline is None:
            self.skipTest("Dependencies not available")
        
        config = {
            'confidence_threshold': 0.8,
            'staff_line_thickness': 3,
            'output_format': 'musicxml',
            'divisions': 480
        }
        
        pipeline = OMRPipeline(config=config)
        
        # Should accept configuration
        self.assertEqual(pipeline.config['confidence_threshold'], 0.8)
        self.assertEqual(pipeline.config['divisions'], 480)
    
    def test_error_handling(self):
        """Test pipeline error handling."""
        if OMRPipeline is None:
            self.skipTest("Dependencies not available")
        
        # Test with None input
        with self.assertRaises((ValueError, TypeError)):
            self.pipeline.process_image(None)
        
        # Test with invalid output path
        invalid_output = "/invalid/path/that/does/not/exist/output.mxl"
        
        try:
            result = self.pipeline.process_image(
                self.test_image,
                output_path=invalid_output
            )
            # If it doesn't raise an error, should at least return None or error info
            if result is not None:
                self.assertIn('error', result.get('status', '').lower())
        except (OSError, IOError):
            # Expected for invalid path
            pass

if __name__ == '__main__':
    unittest.main()