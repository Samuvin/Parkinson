"""
Abstract interfaces following Interface Segregation Principle (ISP).
Defines contracts for prediction, calibration, and model management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class IPredictor(ABC):
    """Interface for prediction models."""
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> int:
        """Make a prediction on input features."""
        pass
    
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass


class IFeatureScaler(ABC):
    """Interface for feature scaling."""
    
    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using learned scaling."""
        pass


class ICalibrator(ABC):
    """Interface for confidence calibration."""
    
    @abstractmethod
    def calibrate(
        self,
        prediction: int,
        probabilities: Dict[str, float],
        features: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calibrate prediction confidence."""
        pass


class IDetectionService(ABC):
    """Interface for multi-modal detection services."""

    @abstractmethod
    def detect_single_modality(
        self,
        modality: str,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """Run detection for one modality."""
        pass

    @abstractmethod
    def detect_ensemble(
        self,
        speech_features: Optional[np.ndarray] = None,
        handwriting_features: Optional[np.ndarray] = None,
        gait_features: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run ensemble detection across modalities."""
        pass


class IModelLoader(ABC):
    """Interface for model loading."""
    
    @abstractmethod
    def load_model(self, modality: str) -> IPredictor:
        """Load a trained model for the specified modality."""
        pass
    
    @abstractmethod
    def load_scaler(self, modality: str) -> IFeatureScaler:
        """Load feature scaler for the specified modality."""
        pass

