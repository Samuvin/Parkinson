"""
Facade pattern for backward compatibility.
Provides simple interface to complex subsystem (Facade Pattern).
"""

import numpy as np
from typing import Dict, List, Optional, Any
from src.container import get_detection_service
from src.core.interfaces import IDetectionService


class DetectionFacade:
    """
    Simplified facade for detection operations.

    Implements Facade pattern to provide backward compatibility
    while using the new SOLID-compliant architecture underneath.
    """

    def __init__(self):
        """Initialize facade with detection service."""
        self._service: IDetectionService = get_detection_service()

    def detect_ensemble(
        self,
        speech_features: Optional[np.ndarray] = None,
        handwriting_features: Optional[np.ndarray] = None,
        gait_features: Optional[np.ndarray] = None,
        voting_method: str = 'soft',
        calibration_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run ensemble detection (backward-compatible return shape).
        """
        context = calibration_context or {}

        return self._service.detect_ensemble(
            speech_features=speech_features,
            handwriting_features=handwriting_features,
            gait_features=gait_features,
            voting_method=voting_method,
            calibration_context=context
        )

    def detect_single_modality(
        self,
        modality: str,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """Run detection for a single modality."""
        return self._service.detect_single_modality(modality, features)

    def get_loaded_modalities(self) -> List[str]:
        """Get list of available modalities."""
        return ['speech', 'handwriting', 'gait']

    def is_model_loaded(self, modality: str) -> bool:
        """Check if model is loaded for modality."""
        return modality in self.get_loaded_modalities()

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'loaded_models': self.get_loaded_modalities(),
            'model_details': {}
        }


def get_model_manager() -> DetectionFacade:
    """
    Get model manager (backward-compatible factory).

    External code can continue using get_model_manager()
    without knowing about the refactored architecture.
    """
    return DetectionFacade()
