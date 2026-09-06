"""
LLM Reasoner module for generating investment views.

Phase 3 of Filing-Aware LLM Views for Portfolio Optimization.
"""

from .schemas import ReasonerInput, ViewOutput, View, FilingTrigger
from .input_builder import InputBuilder
from .reasoner import ViewReasoner
from .calibration import confidence_to_sigma, calibrate_uncertainty

__all__ = [
    "ReasonerInput",
    "ViewOutput",
    "View",
    "FilingTrigger",
    "InputBuilder",
    "ViewReasoner",
    "confidence_to_sigma",
    "calibrate_uncertainty",
]
