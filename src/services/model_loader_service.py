"""
Model loading service implementing Repository pattern.
Handles model persistence and retrieval (Single Responsibility).
"""

import joblib
from pathlib import Path
from typing import Dict, Optional
from src.core.interfaces import IModelLoader, IPredictor, IFeatureScaler
from src.adapters.model_adapter import SklearnModelAdapter, SklearnScalerAdapter

class FileSystemModelLoader(IModelLoader):
    """Loads models from filesystem using joblib."""
    
    def __init__(self, models_directory: str = 'models'):
        """
        Initialize model loader.
        
        Args:
            models_directory: Path to directory containing models
        """
        self._models_dir = Path(models_directory)
        self._model_cache: Dict[str, IPredictor] = {}
        self._scaler_cache: Dict[str, IFeatureScaler] = {}
    
    def load_model(self, modality: str) -> Optional[IPredictor]:
        """
        Load model for specified modality.
        
        Implements caching to avoid repeated disk I/O.
        """
        if modality in self._model_cache:
            return self._model_cache[modality]
        
        model_path = self._resolve_model_path(modality)
        if not model_path or not model_path.exists():
            return None
        
        try:
            raw_model = self._load_from_disk(model_path)
            adapted_model = SklearnModelAdapter(raw_model)
            self._model_cache[modality] = adapted_model
            return adapted_model
        except Exception:
            return None
    
    def load_scaler(self, modality: str) -> Optional[IFeatureScaler]:
        """Load feature scaler for specified modality."""
        if modality in self._scaler_cache:
            return self._scaler_cache[modality]
        
        scaler_path = self._models_dir / f'{modality}_scaler.joblib'
        if not scaler_path.exists():
            return None
        
        try:
            raw_scaler = joblib.load(scaler_path)
            adapted_scaler = SklearnScalerAdapter(raw_scaler)
            self._scaler_cache[modality] = adapted_scaler
            return adapted_scaler
        except Exception:
            return None
    
    def _resolve_model_path(self, modality: str) -> Optional[Path]:
        """Resolve model file path with fallback logic."""
        model_path = self._models_dir / f'{modality}_model.joblib'
        
        if modality == 'speech' and not model_path.exists():
            legacy_path = self._models_dir / 'best_model.joblib'
            if legacy_path.exists():
                return legacy_path
        
        return model_path if model_path.exists() else None
    
    def _load_from_disk(self, path: Path):
        """Load model from disk and extract if wrapped in dict."""
        model_data = joblib.load(path)
        
        if isinstance(model_data, dict) and 'model' in model_data:
            return model_data['model']
        
        return model_data
    
    def clear_cache(self):
        """Clear cached models and scalers."""
        self._model_cache.clear()
        self._scaler_cache.clear()

