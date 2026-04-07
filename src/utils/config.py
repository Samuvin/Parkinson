"""Configuration utilities for loading and managing project settings."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for the Parkinson's prediction system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file.
                        If None, uses ``config/project.yaml`` under the project root.
        """
        if config_path is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "project.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration settings.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'data.train_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config = Config()
            >>> train_size = config.get('data.train_size')
            >>> cv_folds = config.get('training.cv_folds', 5)
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.config.get('data', {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model ('logistic_regression' or 'svm')
            
        Returns:
            Model configuration dictionary
        """
        return self.config.get('models', {}).get(model_name, {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.config.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration section."""
        return self.config.get('evaluation', {})
    
    def get_webapp_config(self) -> Dict[str, Any]:
        """Get web application configuration section."""
        return self.config.get('webapp', {})
    
    def get_features_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration section."""
        return self.config.get('features', {})
    
    @property
    def random_state(self) -> int:
        """Get the random state for reproducibility."""
        return self.get('data.random_state', 42)
    
    @property
    def train_size(self) -> float:
        """Get training set size ratio."""
        return self.get('data.train_size', 0.70)
    
    @property
    def val_size(self) -> float:
        """Get validation set size ratio."""
        return self.get('data.val_size', 0.15)
    
    @property
    def test_size(self) -> float:
        """Get test set size ratio."""
        return self.get('data.test_size', 0.15)
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"Config(path={self.config_path})"
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        return yaml.dump(self.config, default_flow_style=False)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to the project root
    """
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path object pointing to the data directory
    """
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """
    Get the raw data directory path.
    
    Returns:
        Path object pointing to the raw data directory
    """
    return get_data_dir() / "raw"


def get_processed_data_dir() -> Path:
    """
    Get the processed data directory path.
    
    Returns:
        Path object pointing to the processed data directory
    """
    return get_data_dir() / "processed"


def get_models_dir() -> Path:
    """
    Get the models directory path.
    
    Returns:
        Path object pointing to the models directory
    """
    return get_project_root() / "models"


def get_reports_dir() -> Path:
    """
    Get the reports directory path for training artifacts (metrics, predictions export).

    Returns:
        Path object pointing to the reports directory
    """
    return get_project_root() / "reports"


def ensure_dir_exists(directory: Path) -> None:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Path to the directory
    """
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration loading
    config = Config()
    print("Configuration loaded successfully!")
    print("\nData Config:", config.get_data_config())
    print("\nTraining Config:", config.get_training_config())
    print("\nRandom State:", config.random_state)

