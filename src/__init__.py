"""
OMR Package Initialization
==========================

Package-level imports and metadata for the OMR Pipeline.
"""

__version__ = "1.0.0"
__author__ = "OMR Development Team"
__email__ = "omr@example.com"
__license__ = "MIT"

# Import main classes for easy access
try:
    from src.omr_pipeline import OMRPipeline
    from src.preprocessing.image_preprocessor import ImagePreprocessor
    from src.detection.staff_detector import StaffDetector
    from src.detection.symbol_detector import SymbolDetector
    from src.reconstruction.music_reconstructor import MusicReconstructor
    from src.output.musicxml_generator import MusicXMLGenerator
    from src.output.json_exporter import JSONExporter
    from src.evaluation.metrics import OMREvaluator
    
    __all__ = [
        'OMRPipeline',
        'ImagePreprocessor', 
        'StaffDetector',
        'SymbolDetector',
        'MusicReconstructor',
        'MusicXMLGenerator', 
        'JSONExporter',
        'OMREvaluator'
    ]
except ImportError:
    # Allow package to be imported even if dependencies are missing
    __all__ = []

# Package metadata
__package_info__ = {
    'name': 'omr-pipeline',
    'version': __version__,
    'description': 'Comprehensive Optical Music Recognition Pipeline',
    'author': __author__,
    'author_email': __email__,
    'license': __license__,
    'url': 'https://github.com/example/omr-pipeline',
    'keywords': [
        'optical music recognition',
        'omr', 
        'sheet music',
        'musicxml',
        'computer vision',
        'deep learning'
    ]
}