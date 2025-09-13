"""
OMR Evaluation Package
=====================

Comprehensive evaluation tools for Optical Music Recognition systems.
Includes symbol-level and semantic-level evaluation metrics.
"""

from .metrics import (
    OMREvaluator,
    SymbolLevelEvaluator,
    SemanticLevelEvaluator,
    EvaluationResult,
    SymbolMatch,
    BoundingBoxMatcher
)

from .evaluation_cli import EvaluationRunner

__all__ = [
    'OMREvaluator',
    'SymbolLevelEvaluator', 
    'SemanticLevelEvaluator',
    'EvaluationResult',
    'SymbolMatch',
    'BoundingBoxMatcher',
    'EvaluationRunner'
]