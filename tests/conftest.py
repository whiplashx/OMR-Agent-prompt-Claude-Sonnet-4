"""
Test Configuration for OMR Pipeline
===================================

Configuration and fixtures for pytest test suite.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Generator, Dict, Any

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a sample sheet music image for testing."""
    # Create a synthetic sheet music image
    height, width = 800, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add staff lines
    staff_positions = [100, 150, 200, 250, 300]
    for y in staff_positions:
        image[y-2:y+2, :] = 0
    
    # Add some note-like shapes
    note_positions = [(120, 150), (140, 200), (160, 250)]
    for x, y in note_positions:
        # Simple ellipse for note head
        for dy in range(-5, 6):
            for dx in range(-8, 9):
                if dx*dx/64 + dy*dy/25 <= 1:
                    if 0 <= y+dy < height and 0 <= x+dx < width:
                        image[y+dy, x+dx] = 0
    
    return image

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing."""
    return {
        'confidence_threshold': 0.5,
        'staff_line_thickness': 2,
        'min_staff_length': 100,
        'model_path': None,  # Use default for testing
        'output_format': 'musicxml',
        'divisions': 480
    }

@pytest.fixture
def mock_detection_results():
    """Mock symbol detection results for testing."""
    return [
        {
            'class': 'quarter_note',
            'confidence': 0.95,
            'bbox': [100, 120, 120, 140],
            'pitch': 'C4'
        },
        {
            'class': 'half_note', 
            'confidence': 0.88,
            'bbox': [150, 140, 170, 160],
            'pitch': 'E4'
        },
        {
            'class': 'treble_clef',
            'confidence': 0.99,
            'bbox': [50, 80, 80, 120],
            'pitch': None
        }
    ]

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "test_pipeline" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)