"""
Staff Detection and Removal Module
==================================

Detects staff lines in sheet music and removes them while preserving musical symbols.
Uses Hough line detection and morphological operations.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StaffLine:
    """Represents a single staff line."""
    y_position: int
    x_start: int
    x_end: int
    thickness: int
    confidence: float


@dataclass 
class Staff:
    """Represents a complete staff (5 lines)."""
    lines: List[StaffLine]
    y_top: int
    y_bottom: int
    x_start: int
    x_end: int
    line_spacing: float
    confidence: float


class StaffDetector:
    """
    Detects and removes staff lines from sheet music images.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the staff detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.line_thickness_range = self.config.get('line_thickness_range', (2, 8))
        self.min_line_length = self.config.get('min_line_length', 100)
        self.staff_line_spacing_range = self.config.get('staff_line_spacing_range', (15, 40))
        self.hough_threshold = self.config.get('hough_threshold', 100)
        self.max_line_gap = self.config.get('max_line_gap', 20)
        
    def detect_staves(self, image: np.ndarray) -> List[Staff]:
        """
        Detect all staves in the image.
        
        Args:
            image: Preprocessed binary image
            
        Returns:
            List of detected Staff objects
        """
        logger.info("Starting staff detection")
        
        # Step 1: Detect all horizontal lines
        horizontal_lines = self._detect_horizontal_lines(image)
        
        if not horizontal_lines:
            logger.warning("No horizontal lines detected")
            return []
        
        # Step 2: Filter lines by thickness and length
        staff_lines = self._filter_staff_lines(horizontal_lines, image)
        
        # Step 3: Group lines into staves
        staves = self._group_lines_into_staves(staff_lines)
        
        # Step 4: Validate and refine staves
        validated_staves = self._validate_staves(staves, image)
        
        logger.info(f"Detected {len(validated_staves)} staves")
        return validated_staves
    
    def _detect_horizontal_lines(self, image: np.ndarray) -> List[Tuple]:
        """
        Detect horizontal lines using Hough line detection.
        
        Args:
            image: Binary image
            
        Returns:
            List of line tuples (x1, y1, x2, y2)
        """
        # Ensure we have a binary image (inverted for line detection)
        if np.max(image) > 1:
            binary = 255 - image  # Invert: lines should be white on black
        else:
            binary = 1 - image
            binary = (binary * 255).astype(np.uint8)
        
        # Use morphological operations to enhance horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        enhanced = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            enhanced,
            rho=1,                          # Distance resolution
            theta=np.pi/180,                # Angular resolution
            threshold=self.hough_threshold,  # Minimum votes
            minLineLength=self.min_line_length,  # Minimum line length
            maxLineGap=self.max_line_gap    # Maximum gap between segments
        )
        
        if lines is None:
            return []
        
        # Filter for nearly horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 != 0:
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # Accept lines within 5 degrees of horizontal
                if angle < 5 or angle > 175:
                    horizontal_lines.append((x1, y1, x2, y2))
            else:
                # Vertical line - skip
                continue
        
        logger.debug(f"Found {len(horizontal_lines)} horizontal lines")
        return horizontal_lines
    
    def _filter_staff_lines(self, lines: List[Tuple], image: np.ndarray) -> List[StaffLine]:
        """
        Filter lines to identify potential staff lines based on thickness and length.
        
        Args:
            lines: List of line tuples
            image: Original binary image
            
        Returns:
            List of StaffLine objects
        """
        staff_lines = []
        height, width = image.shape
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Calculate line properties
            y_pos = int((y1 + y2) / 2)  # Average y position
            x_start = min(x1, x2)
            x_end = max(x1, x2)
            line_length = x_end - x_start
            
            # Skip short lines
            if line_length < self.min_line_length:
                continue
            
            # Estimate line thickness by examining pixels around the line
            thickness = self._estimate_line_thickness(image, x_start, x_end, y_pos)
            
            # Filter by thickness
            if not (self.line_thickness_range[0] <= thickness <= self.line_thickness_range[1]):
                continue
            
            # Calculate confidence based on line properties
            length_ratio = min(line_length / (width * 0.3), 1.0)  # Prefer longer lines
            thickness_score = 1.0 - abs(thickness - 3) / 5.0     # Prefer thickness around 3
            confidence = (length_ratio + thickness_score) / 2.0
            
            staff_line = StaffLine(
                y_position=y_pos,
                x_start=x_start,
                x_end=x_end,
                thickness=thickness,
                confidence=confidence
            )
            
            staff_lines.append(staff_line)
        
        # Sort by y position
        staff_lines.sort(key=lambda line: line.y_position)
        
        logger.debug(f"Filtered to {len(staff_lines)} potential staff lines")
        return staff_lines
    
    def _estimate_line_thickness(self, image: np.ndarray, x_start: int, x_end: int, y_pos: int) -> int:
        """
        Estimate the thickness of a horizontal line.
        
        Args:
            image: Binary image
            x_start: Start x coordinate
            x_end: End x coordinate  
            y_pos: Y position of the line
            
        Returns:
            Estimated thickness in pixels
        """
        height, width = image.shape
        
        # Sample multiple points along the line
        sample_points = min(10, x_end - x_start)
        x_positions = np.linspace(x_start, x_end, sample_points, dtype=int)
        
        thickness_estimates = []
        
        for x in x_positions:
            if x < 0 or x >= width:
                continue
            
            # Search vertically around the line position
            search_range = 10
            y_start = max(0, y_pos - search_range)
            y_end = min(height, y_pos + search_range)
            
            # Extract vertical slice
            vertical_slice = image[y_start:y_end, x]
            
            # Find the darkest region (staff line)
            # In binary image, dark = 0, white = 255
            if len(vertical_slice) == 0:
                continue
            
            # Find continuous dark regions
            dark_regions = []
            in_dark_region = False
            region_start = 0
            
            for i, pixel in enumerate(vertical_slice):
                if pixel < 128:  # Dark pixel
                    if not in_dark_region:
                        in_dark_region = True
                        region_start = i
                else:  # Light pixel
                    if in_dark_region:
                        in_dark_region = False
                        dark_regions.append(i - region_start)
            
            # If we ended in a dark region
            if in_dark_region:
                dark_regions.append(len(vertical_slice) - region_start)
            
            # Take the largest dark region as the line thickness
            if dark_regions:
                thickness_estimates.append(max(dark_regions))
        
        if thickness_estimates:
            return int(np.median(thickness_estimates))
        else:
            return 3  # Default thickness
    
    def _group_lines_into_staves(self, staff_lines: List[StaffLine]) -> List[Staff]:
        """
        Group staff lines into complete staves (5 lines each).
        
        Args:
            staff_lines: List of potential staff lines
            
        Returns:
            List of Staff objects
        """
        staves = []
        used_lines = set()
        
        for i, line1 in enumerate(staff_lines):
            if i in used_lines:
                continue
            
            # Try to find 4 more lines to complete a staff
            staff_candidates = [line1]
            candidate_indices = [i]
            
            # Look for the next 4 lines with appropriate spacing
            current_y = line1.y_position
            
            for j in range(i + 1, len(staff_lines)):
                if j in used_lines:
                    continue
                
                line2 = staff_lines[j]
                spacing = line2.y_position - current_y
                
                # Check if spacing is within expected range
                if (self.staff_line_spacing_range[0] <= spacing <= self.staff_line_spacing_range[1]):
                    staff_candidates.append(line2)
                    candidate_indices.append(j)
                    current_y = line2.y_position
                    
                    if len(staff_candidates) == 5:
                        break
                elif spacing > self.staff_line_spacing_range[1]:
                    # Gap too large, stop looking
                    break
            
            # If we found 5 lines, create a staff
            if len(staff_candidates) == 5:
                # Validate spacing consistency
                spacings = []
                for k in range(4):
                    spacing = staff_candidates[k+1].y_position - staff_candidates[k].y_position
                    spacings.append(spacing)
                
                # Check if spacings are reasonably consistent
                spacing_std = np.std(spacings)
                spacing_mean = np.mean(spacings)
                
                if spacing_std < spacing_mean * 0.3:  # Standard deviation < 30% of mean
                    # Create staff object
                    staff = Staff(
                        lines=staff_candidates,
                        y_top=staff_candidates[0].y_position,
                        y_bottom=staff_candidates[4].y_position,
                        x_start=min(line.x_start for line in staff_candidates),
                        x_end=max(line.x_end for line in staff_candidates),
                        line_spacing=spacing_mean,
                        confidence=np.mean([line.confidence for line in staff_candidates])
                    )
                    
                    staves.append(staff)
                    
                    # Mark lines as used
                    for idx in candidate_indices:
                        used_lines.add(idx)
        
        logger.debug(f"Grouped lines into {len(staves)} potential staves")
        return staves
    
    def _validate_staves(self, staves: List[Staff], image: np.ndarray) -> List[Staff]:
        """
        Validate detected staves and remove false positives.
        
        Args:
            staves: List of candidate staves
            image: Original binary image
            
        Returns:
            List of validated staves
        """
        validated_staves = []
        
        for staff in staves:
            # Check minimum requirements
            if len(staff.lines) != 5:
                continue
            
            # Check if staff spans a reasonable width
            staff_width = staff.x_end - staff.x_start
            image_width = image.shape[1]
            
            if staff_width < image_width * 0.3:  # Staff should span at least 30% of image width
                continue
            
            # Check line spacing consistency
            spacings = []
            for i in range(4):
                spacing = staff.lines[i+1].y_position - staff.lines[i].y_position
                spacings.append(spacing)
            
            spacing_cv = np.std(spacings) / np.mean(spacings)  # Coefficient of variation
            
            if spacing_cv > 0.3:  # Reject if spacing is too inconsistent
                continue
            
            # Check if lines are roughly parallel and horizontal
            angles = []
            for line in staff.lines:
                if line.x_end - line.x_start > 0:
                    # Calculate slight angle based on thickness variation
                    # This is a simplified check - in a more advanced version,
                    # you might track actual line slopes
                    angles.append(0)  # Assuming horizontal lines for now
            
            validated_staves.append(staff)
        
        # Sort staves by vertical position
        validated_staves.sort(key=lambda s: s.y_top)
        
        logger.info(f"Validated {len(validated_staves)} staves")
        return validated_staves
    
    def remove_staves(self, image: np.ndarray, staves: List[Staff]) -> np.ndarray:
        """
        Remove staff lines from image while preserving musical symbols.
        
        Args:
            image: Binary image with staff lines
            staves: List of detected staves
            
        Returns:
            Image with staff lines removed
        """
        logger.info("Removing staff lines")
        
        result = image.copy()
        
        for staff in staves:
            for line in staff.lines:
                result = self._remove_single_staff_line(result, line)
        
        # Clean up artifacts from staff removal
        result = self._clean_staff_removal_artifacts(result)
        
        logger.info("Staff line removal completed")
        return result
    
    def _remove_single_staff_line(self, image: np.ndarray, line: StaffLine) -> np.ndarray:
        """
        Remove a single staff line from the image.
        
        Args:
            image: Binary image
            line: StaffLine to remove
            
        Returns:
            Image with the line removed
        """
        result = image.copy()
        height, width = image.shape
        
        # Create a mask for the staff line
        y_start = max(0, line.y_position - line.thickness // 2)
        y_end = min(height, line.y_position + line.thickness // 2 + 1)
        x_start = max(0, line.x_start)
        x_end = min(width, line.x_end)
        
        # Use morphological operations to identify the line more precisely
        line_region = image[y_start:y_end, x_start:x_end]
        
        if line_region.size == 0:
            return result
        
        # Create horizontal kernel for line detection
        kernel_width = min(line_region.shape[1], 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        
        # Find the exact line pixels
        line_pixels = cv2.morphologyEx(255 - line_region, cv2.MORPH_OPEN, kernel)
        
        # Remove line pixels by setting them to white
        mask = line_pixels > 0
        line_region[mask] = 255
        
        # Put the processed region back
        result[y_start:y_end, x_start:x_end] = line_region
        
        return result
    
    def _clean_staff_removal_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Clean up artifacts left from staff line removal.
        
        Args:
            image: Image after staff line removal
            
        Returns:
            Cleaned image
        """
        # Remove small horizontal artifacts
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Fill small gaps that might have been created
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fill_kernel)
        
        return cleaned
    
    def get_staff_regions(self, staves: List[Staff]) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes for each staff region.
        
        Args:
            staves: List of detected staves
            
        Returns:
            List of (x, y, width, height) tuples for each staff
        """
        regions = []
        
        for staff in staves:
            # Add some padding around the staff
            padding_y = int(staff.line_spacing * 2)  # 2 staff spaces above and below
            padding_x = 10
            
            x = max(0, staff.x_start - padding_x)
            y = max(0, staff.y_top - padding_y)
            width = staff.x_end - staff.x_start + 2 * padding_x
            height = staff.y_bottom - staff.y_top + 2 * padding_y
            
            regions.append((x, y, width, height))
        
        return regions


def test_staff_detector():
    """Test function for the StaffDetector."""
    import matplotlib.pyplot as plt
    
    # Create a test image with staff lines
    test_image = np.ones((400, 800), dtype=np.uint8) * 255
    
    # Add 5 staff lines
    staff_y_positions = [100, 120, 140, 160, 180]
    for y in staff_y_positions:
        cv2.line(test_image, (50, y), (750, y), 0, 2)
    
    # Add another staff
    staff_y_positions_2 = [250, 270, 290, 310, 330]
    for y in staff_y_positions_2:
        cv2.line(test_image, (50, y), (750, y), 0, 2)
    
    # Test the detector
    detector = StaffDetector()
    staves = detector.detect_staves(test_image)
    staff_removed = detector.remove_staves(test_image, staves)
    
    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title('Original with Staff Lines')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(staff_removed, cmap='gray')
    plt.title('After Staff Removal')
    plt.axis('off')
    
    # Show detected staves
    plt.subplot(2, 2, 3)
    annotated = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    for i, staff in enumerate(staves):
        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
        for line in staff.lines:
            cv2.line(annotated, (line.x_start, line.y_position), 
                    (line.x_end, line.y_position), color, 3)
    
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Staves: {len(staves)}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Detected {len(staves)} staves")
    for i, staff in enumerate(staves):
        print(f"Staff {i+1}: {len(staff.lines)} lines, "
              f"spacing: {staff.line_spacing:.1f}, "
              f"confidence: {staff.confidence:.3f}")


if __name__ == "__main__":
    test_staff_detector()