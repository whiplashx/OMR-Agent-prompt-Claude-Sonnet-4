"""
Unit Tests for Image Preprocessor
=================================

Test suite for the ImagePreprocessor module.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from preprocessing.image_preprocessor import ImagePreprocessor
except ImportError:
    ImagePreprocessor = None

class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if ImagePreprocessor is None:
            self.skipTest("ImagePreprocessor not available - missing dependencies")
        
        self.preprocessor = ImagePreprocessor()
        
        # Create sample test image
        self.test_image = self._create_test_image()
    
    def _create_test_image(self) -> np.ndarray:
        """Create a sample sheet music image for testing."""
        height, width = 400, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add staff lines
        staff_positions = [100, 120, 140, 160, 180]
        for y in staff_positions:
            image[y-1:y+1, 50:550] = 0
        
        # Add some noise
        noise = np.random.randint(0, 50, (height, width, 3))
        image = np.clip(image.astype(int) - noise, 0, 255).astype(np.uint8)
        
        return image
    
    def test_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        if ImagePreprocessor is None:
            self.skipTest("Dependencies not available")
        
        result = self.preprocessor.preprocess(self.test_image)
        
        # Check that result is returned
        self.assertIsNotNone(result)
        self.assertIn('processed_image', result)
        self.assertIn('skew_angle', result)
        self.assertIn('processing_time', result)
        
        # Check image properties
        processed_image = result['processed_image']
        self.assertEqual(len(processed_image.shape), 2)  # Should be grayscale
        self.assertGreaterEqual(processed_image.min(), 0)
        self.assertLessEqual(processed_image.max(), 255)
    
    def test_noise_reduction(self):
        """Test noise reduction functionality."""
        if ImagePreprocessor is None:
            self.skipTest("Dependencies not available")
        
        # Create noisy image
        noisy_image = self.test_image.copy()
        noise = np.random.randint(0, 100, noisy_image.shape)
        noisy_image = np.clip(noisy_image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        denoised = self.preprocessor._reduce_noise(noisy_image)
        
        # Denoised image should be different from original
        self.assertFalse(np.array_equal(noisy_image, denoised))
        
        # Should maintain image shape
        self.assertEqual(noisy_image.shape, denoised.shape)
    
    def test_binarization(self):
        """Test image binarization."""
        if ImagePreprocessor is None:
            self.skipTest("Dependencies not available")
        
        # Convert to grayscale first
        gray_image = self.preprocessor._convert_to_grayscale(self.test_image)
        binary = self.preprocessor._binarize_image(gray_image)
        
        # Should be binary (only 0 and 255)
        unique_values = np.unique(binary)
        self.assertTrue(len(unique_values) <= 2)
        self.assertIn(0, unique_values)
        
        # Should maintain shape
        self.assertEqual(gray_image.shape, binary.shape)
    
    def test_skew_detection(self):
        """Test skew angle detection."""
        if ImagePreprocessor is None:
            self.skipTest("Dependencies not available")
        
        angle = self.preprocessor._detect_skew(self.test_image)
        
        # Angle should be a number
        self.assertIsInstance(angle, (int, float))
        
        # Should be within reasonable range
        self.assertGreaterEqual(angle, -45)
        self.assertLessEqual(angle, 45)
    
    def test_invalid_input(self):
        """Test handling of invalid inputs."""
        if ImagePreprocessor is None:
            self.skipTest("Dependencies not available")
        
        # Test with None input
        with self.assertRaises((ValueError, TypeError)):
            self.preprocessor.preprocess(None)
        
        # Test with empty array
        empty_array = np.array([])
        with self.assertRaises((ValueError, IndexError)):
            self.preprocessor.preprocess(empty_array)
    
    def test_configuration_options(self):
        """Test different configuration options."""
        if ImagePreprocessor is None:
            self.skipTest("Dependencies not available")
        
        config = {
            'apply_denoising': False,
            'apply_skew_correction': False,
            'binarization_method': 'otsu'
        }
        
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.preprocess(self.test_image)
        
        # Should still return valid result
        self.assertIsNotNone(result)
        self.assertIn('processed_image', result)

if __name__ == '__main__':
    unittest.main()