"""
OMR (Optical Music Recognition) Pipeline
=========================================

A comprehensive pipeline for converting sheet music images to MusicXML format.

Features:
- Image preprocessing and staff detection
- Instance segmentation for musical symbols
- Symbol classification and grouping
- Pitch and rhythm reconstruction
- MusicXML generation with MuseScore compatibility
- Manual correction UI
- Comprehensive evaluation metrics

Usage:
    from omr import OMRPipeline
    
    pipeline = OMRPipeline()
    result = pipeline.process_image("sheet_music.png")
    pipeline.save_musicxml(result, "output.mxl")
"""

from .src.omr_pipeline import OMRPipeline
from .src.preprocessing import ImagePreprocessor
from .src.detection import StaffDetector, SymbolDetector
from .src.reconstruction import MusicReconstructor
from .src.output import MusicXMLGenerator

__version__ = "1.0.0"
__author__ = "OMR Pipeline"

__all__ = [
    "OMRPipeline",
    "ImagePreprocessor", 
    "StaffDetector",
    "SymbolDetector",
    "MusicReconstructor",
    "MusicXMLGenerator"
]