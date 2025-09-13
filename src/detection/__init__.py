"""
Detection Package
================

Musical symbol detection and classification.
"""

from .staff_detector import StaffDetector, Staff, StaffLine
from .symbol_detector import SymbolDetector, DetectedSymbol

__all__ = ['StaffDetector', 'Staff', 'StaffLine', 'SymbolDetector', 'DetectedSymbol']