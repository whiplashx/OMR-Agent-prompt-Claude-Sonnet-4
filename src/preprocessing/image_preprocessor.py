"""
Image Preprocessing Module
=========================

Handles image preprocessing for sheet music including:
- Noise reduction and filtering
- Binarization with adaptive thresholding
- Skew correction using Hough transform
- Image normalization and resizing
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from scipy import ndimage
from skimage import morphology, filters
from skimage.transform import rotate

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesses sheet music images for optimal OMR performance.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the image preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.target_height = self.config.get('target_height', 1200)
        self.deskew = self.config.get('deskew', True)
        self.denoise = self.config.get('denoise', True)
        self.enhance_contrast = self.config.get('enhance_contrast', True)
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed binary image
        """
        logger.info("Starting image preprocessing")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Resize image to target height
        resized = self._resize_image(gray)
        
        # Step 2: Denoise if enabled
        if self.denoise:
            denoised = self._denoise_image(resized)
        else:
            denoised = resized
        
        # Step 3: Enhance contrast if enabled
        if self.enhance_contrast:
            enhanced = self._enhance_contrast(denoised)
        else:
            enhanced = denoised
        
        # Step 4: Correct skew if enabled
        if self.deskew:
            deskewed = self._correct_skew(enhanced)
        else:
            deskewed = enhanced
        
        # Step 5: Binarize the image
        binary = self._binarize_image(deskewed)
        
        # Step 6: Clean up binary image
        cleaned = self._clean_binary_image(binary)
        
        logger.info("Image preprocessing completed")
        return cleaned
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target height while maintaining aspect ratio.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if height == self.target_height:
            return image
        
        # Calculate new width maintaining aspect ratio
        scale_factor = self.target_height / height
        new_width = int(width * scale_factor)
        
        # Resize using cubic interpolation for better quality
        resized = cv2.resize(
            image, 
            (new_width, self.target_height), 
            interpolation=cv2.INTER_CUBIC
        )
        
        logger.debug(f"Resized from {width}x{height} to {new_width}x{self.target_height}")
        return resized
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the image using Non-local Means Denoising.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Denoised image
        """
        # Use Non-local Means Denoising which is effective for preserving edges
        denoised = cv2.fastNlMeansDenoising(
            image,
            None,
            h=10,           # Filter strength
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Alternative: Bilateral filter for edge-preserving smoothing
        # denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        logger.debug("Applied noise reduction")
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=2.0,      # Limit contrast enhancement
            tileGridSize=(8, 8)  # Size of neighborhood for histogram equalization
        )
        
        enhanced = clahe.apply(image)
        
        logger.debug("Applied contrast enhancement")
        return enhanced
    
    def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in the image using Hough line detection.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Deskewed image
        """
        # Create a binary image for line detection
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            binary,
            rho=1,                    # Distance resolution in pixels
            theta=np.pi/180,          # Angular resolution in radians
            threshold=100,            # Minimum votes for line detection
            minLineLength=200,        # Minimum line length
            maxLineGap=20            # Maximum gap between line segments
        )
        
        if lines is None:
            logger.warning("No lines detected for skew correction")
            return image
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle in degrees
            if x2 - x1 != 0:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Focus on nearly horizontal lines (likely staff lines)
                if abs(angle) < 30:
                    angles.append(angle)
        
        if not angles:
            logger.warning("No suitable lines found for skew correction")
            return image
        
        # Use median angle to avoid outliers
        skew_angle = np.median(angles)
        
        # Only correct if skew is significant (> 0.5 degrees)
        if abs(skew_angle) > 0.5:
            logger.info(f"Correcting skew angle: {skew_angle:.2f} degrees")
            
            # Rotate image to correct skew
            corrected = rotate(
                image, 
                -skew_angle,  # Negative to correct the skew
                resize=True,  # Resize canvas to fit rotated image
                preserve_range=True
            ).astype(np.uint8)
            
            return corrected
        else:
            logger.debug(f"Skew angle {skew_angle:.2f} degrees is minimal, no correction needed")
            return image
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert grayscale image to binary using adaptive thresholding.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary image (0 = black, 255 = white)
        """
        # Method 1: Adaptive Gaussian thresholding
        adaptive_binary = cv2.adaptiveThreshold(
            image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,  # Size of neighborhood for threshold calculation
            C=8           # Constant subtracted from mean
        )
        
        # Method 2: Otsu's thresholding as backup
        _, otsu_binary = cv2.threshold(
            image, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Use adaptive thresholding as primary method
        # It handles varying lighting conditions better
        binary = adaptive_binary
        
        logger.debug("Applied adaptive thresholding")
        return binary
    
    def _clean_binary_image(self, binary: np.ndarray) -> np.ndarray:
        """
        Clean up binary image by removing small noise and filling gaps.
        
        Args:
            binary: Input binary image
            
        Returns:
            Cleaned binary image
        """
        # Remove small noise using morphological opening
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small gaps using morphological closing
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
        
        # Remove very small connected components (likely noise)
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            255 - closed,  # Invert for connected components
            connectivity=8
        )
        
        # Create mask for components larger than minimum size
        min_size = 20  # Minimum component size in pixels
        mask = np.zeros_like(closed)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 255
        
        # Invert back to original orientation
        cleaned = 255 - mask
        
        logger.debug("Applied morphological cleaning")
        return cleaned
    
    def get_preprocessing_metadata(self, original_image: np.ndarray, processed_image: np.ndarray) -> Dict:
        """
        Generate metadata about the preprocessing steps applied.
        
        Args:
            original_image: Original input image
            processed_image: Final processed image
            
        Returns:
            Dictionary containing preprocessing metadata
        """
        metadata = {
            'original_size': original_image.shape,
            'processed_size': processed_image.shape,
            'target_height': self.target_height,
            'preprocessing_steps': [],
            'config': self.config
        }
        
        # Track which steps were applied
        metadata['preprocessing_steps'].append('resize')
        
        if self.denoise:
            metadata['preprocessing_steps'].append('denoise')
        
        if self.enhance_contrast:
            metadata['preprocessing_steps'].append('contrast_enhancement')
        
        if self.deskew:
            metadata['preprocessing_steps'].append('skew_correction')
        
        metadata['preprocessing_steps'].extend(['binarization', 'morphological_cleaning'])
        
        return metadata


def test_preprocessor():
    """Test function for the ImagePreprocessor."""
    import matplotlib.pyplot as plt
    
    # Create a test image with some noise and skew
    test_image = np.ones((800, 1000), dtype=np.uint8) * 240
    
    # Add some "staff lines"
    for y in range(200, 700, 80):
        cv2.line(test_image, (50, y), (950, y + 2), 0, 3)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Process with preprocessor
    preprocessor = ImagePreprocessor()
    processed = preprocessor.process(test_image)
    
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Preprocessed')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_preprocessor()