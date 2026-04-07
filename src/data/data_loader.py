"""Data loading utilities for Parkinson's Disease multimodal datasets."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import urllib.request
from tqdm import tqdm

from ..utils.config import Config, get_raw_data_dir, ensure_dir_exists


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class DataLoader:
    """Load and manage Parkinson's Disease datasets from multiple modalities.
    
    This loader uses ONLY real datasets - NO synthetic data generation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object. If None, creates a new Config instance.
        """
        self.config = config or Config()
        self.raw_data_dir = get_raw_data_dir()
        ensure_dir_exists(self.raw_data_dir)
        
        # Create subdirectories for each modality
        self.speech_dir = self.raw_data_dir / "speech"
        self.handwriting_dir = self.raw_data_dir / "handwriting"
        self.gait_dir = self.raw_data_dir / "gait"
        
        for directory in [self.speech_dir, self.handwriting_dir, self.gait_dir]:
            ensure_dir_exists(directory)
    
    def load_speech_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load speech/voice data for Parkinson's prediction from UCI repository.
        
        Returns:
            Tuple of (features DataFrame, labels Series)
            
        Raises:
            FileNotFoundError: If speech data file doesn't exist and download fails
        """
        speech_file = self.speech_dir / "parkinsons.csv"
        
        if not speech_file.exists():
            print(f"Speech data not found at {speech_file}")
            print("Attempting to download from UCI repository...")
            self._download_speech_data()
            
            if not speech_file.exists():
                raise FileNotFoundError(
                    f"Could not load or download Parkinson's speech dataset.\n"
                    f"Please manually download from:\n"
                    f"https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data\n"
                    f"and save to: {speech_file}\n\n"
                    f"See README.md and config/multimodal_features.yaml for detailed instructions."
                )
        
        # Load the data
        df = pd.read_csv(speech_file)
        
        # Separate features and labels
        if 'status' in df.columns:
            y = df['status']
            X = df.drop(['status', 'name'], axis=1, errors='ignore')
        else:
            raise ValueError("Dataset must contain 'status' column for labels")
        
        print(f"✓ Loaded speech data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def load_telemonitoring_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load telemonitoring data from UCI repository.
        
        This dataset contains ~5875 samples from 42 PD patients tracked over time.
        
        Returns:
            Tuple of (features DataFrame, labels Series)
            
        Raises:
            FileNotFoundError: If telemonitoring data doesn't exist
        """
        telemon_file = self.speech_dir / "parkinsons_telemonitoring.csv"
        
        if not telemon_file.exists():
            print(f"Telemonitoring data not found at {telemon_file}")
            print("Attempting to download from UCI repository...")
            self._download_telemonitoring_data()
            
            if not telemon_file.exists():
                raise FileNotFoundError(
                    f"Could not load or download telemonitoring dataset.\n"
                    f"Please manually download from:\n"
                    f"https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data\n"
                    f"and save to: {telemon_file}\n"
                )
        
        # Load the data
        df = pd.read_csv(telemon_file)
        
        # This dataset has UPDRS scores instead of binary labels
        # We'll use motor_UPDRS > 0 as indicator of PD (all samples are from PD patients)
        # For binary classification, we create synthetic healthy controls
        
        # Extract features (exclude identifiers and target variables)
        feature_cols = [col for col in df.columns 
                       if col not in ['subject#', 'age', 'sex', 'test_time', 
                                    'motor_UPDRS', 'total_UPDRS']]
        
        X = df[feature_cols]
        
        # Create binary labels: All samples are PD patients (label = 1)
        y = pd.Series([1] * len(df), name='status')
        
        print(f"✓ Loaded telemonitoring data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Note: All samples are from PD patients")
        
        return X, y
    
    def load_combined_speech_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and combine both UCI speech datasets.
        
        Combines the standard Parkinson's dataset (195 samples) with
        the telemonitoring dataset (5875 samples) for enhanced training.
        
        Returns:
            Tuple of (combined features DataFrame, labels Series)
        """
        print("\n" + "="*60)
        print("Loading Combined Speech Datasets")
        print("="*60)
        
        # Load standard speech dataset
        print("\n1. Loading UCI Parkinson's dataset...")
        X_speech, y_speech = self.load_speech_data()
        
        # Try to load telemonitoring dataset
        try:
            print("\n2. Loading UCI Telemonitoring dataset...")
            X_telemon, y_telemon = self.load_telemonitoring_data()
            
            # Find common features
            common_features = list(set(X_speech.columns) & set(X_telemon.columns))
            
            if len(common_features) > 0:
                print(f"\n3. Merging datasets on {len(common_features)} common features...")
                
                # Select common features from both datasets
                X_speech_common = X_speech[common_features]
                X_telemon_common = X_telemon[common_features]
                
                # Combine datasets
                X_combined = pd.concat([X_speech_common, X_telemon_common], 
                                      axis=0, ignore_index=True)
                y_combined = pd.concat([y_speech, y_telemon], 
                                      axis=0, ignore_index=True)
                
                print(f"\n✓ Combined speech data:")
                print(f"  Total samples: {X_combined.shape[0]}")
                print(f"  Features: {X_combined.shape[1]}")
                print(f"  PD cases: {sum(y_combined == 1)}")
                print(f"  Healthy: {sum(y_combined == 0)}")
            else:
                print("\nWarning: No common features found. Using standard dataset only.")
                X_combined = X_speech
                y_combined = y_speech
                
        except FileNotFoundError:
            print("\nTelemonitoring dataset not available. Using standard dataset only.")
            X_combined = X_speech
            y_combined = y_speech
        
        print("="*60 + "\n")
        return X_combined, y_combined
    
    def load_handwriting_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load handwriting data for Parkinson's prediction.
        
        Uses REAL handwriting dataset only - NO synthetic data generation.
        
        Returns:
            Tuple of (features DataFrame, labels Series)
            
        Raises:
            FileNotFoundError: If handwriting data file doesn't exist
        """
        handwriting_file = self.handwriting_dir / "handwriting_data.csv"
        legacy_hw = self.handwriting_dir / "handwriting_features.csv"
        if not handwriting_file.exists() and legacy_hw.exists():
            handwriting_file = legacy_hw

        if not handwriting_file.exists():
            raise FileNotFoundError(
                f"Handwriting dataset not found at {self.handwriting_dir / 'handwriting_data.csv'}\n\n"
                f"This system requires REAL handwriting data.\n"
                f"Place a CSV with columns matching the project schema at:\n"
                f"{self.handwriting_dir / 'handwriting_data.csv'}\n\n"
                f"Expected format: CSV with features and 'status' column (1=PD, 0=healthy)\n"
                f"See README.md and config/multimodal_features.yaml for detailed instructions on obtaining this dataset."
            )
        
        df = pd.read_csv(handwriting_file)
        
        if 'status' in df.columns:
            y = df['status']
            # Drop status and any ID columns
            X = df.drop(['status', 'subject_id', 'id'], axis=1, errors='ignore')
        else:
            raise ValueError("Handwriting dataset must contain 'status' column for labels")
        
        print(f"✓ Loaded handwriting data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def load_gait_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load gait data for Parkinson's prediction.
        
        Uses REAL gait dataset only - NO synthetic data generation.
        
        Returns:
            Tuple of (features DataFrame, labels Series)
            
        Raises:
            FileNotFoundError: If gait data file doesn't exist
        """
        gait_file = self.gait_dir / "gait_data.csv"
        legacy_gait = self.gait_dir / "gait_features.csv"
        if not gait_file.exists() and legacy_gait.exists():
            gait_file = legacy_gait

        if not gait_file.exists():
            raise FileNotFoundError(
                f"Gait dataset not found at {self.gait_dir / 'gait_data.csv'}\n\n"
                f"This system requires REAL gait data.\n"
                f"See https://physionet.org/content/gaitpdb/1.0.0/ and README.md / config/multimodal_features.yaml\n\n"
                f"After ETL, place features at:\n"
                f"{self.gait_dir / 'gait_data.csv'}\n\n"
                f"Expected format: CSV with features and 'status' column (1=PD, 0=healthy)\n"
                f"See README.md and config/multimodal_features.yaml for detailed instructions."
            )
        
        df = pd.read_csv(gait_file)
        
        if 'status' in df.columns:
            y = df['status']
            # Drop status and any ID columns
            X = df.drop(['status', 'subject_id'], axis=1, errors='ignore')
        else:
            raise ValueError("Gait dataset must contain 'status' column for labels")
        
        print(f"✓ Loaded gait data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def load_all_modalities(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and combine data from all modalities (speech, handwriting, gait).
        
        Uses early fusion strategy - concatenates features from all modalities.
        All data comes from REAL datasets only.
        
        Returns:
            Tuple of (combined features DataFrame, labels Series)
            
        Raises:
            FileNotFoundError: If any dataset is missing
        """
        print("\n" + "="*60)
        print("Loading Multimodal Parkinson's Disease Data")
        print("="*60)
        
        # Load each modality
        print("\n1. Loading speech data...")
        X_speech, y_speech = self.load_speech_data()
        
        print("\n2. Loading handwriting data...")
        X_handwriting, y_handwriting = self.load_handwriting_data()
        
        print("\n3. Loading gait data...")
        X_gait, y_gait = self.load_gait_data()
        
        # Ensure all have the same number of samples (align by minimum)
        min_samples = min(len(y_speech), len(y_handwriting), len(y_gait))
        
        if len(y_speech) != len(y_handwriting) or len(y_speech) != len(y_gait):
            print(f"\n⚠ Warning: Dataset sizes differ. Using first {min_samples} samples from each.")
            print(f"  Speech: {len(y_speech)}, Handwriting: {len(y_handwriting)}, Gait: {len(y_gait)}")
        
        X_speech = X_speech.iloc[:min_samples]
        y_speech = y_speech.iloc[:min_samples]
        X_handwriting = X_handwriting.iloc[:min_samples]
        X_gait = X_gait.iloc[:min_samples]
        
        # Add prefixes to avoid column name conflicts
        X_speech = X_speech.add_prefix('speech_')
        X_handwriting = X_handwriting.add_prefix('handwriting_')
        X_gait = X_gait.add_prefix('gait_')
        
        # Combine features (early fusion)
        X_combined = pd.concat([X_speech, X_handwriting, X_gait], axis=1)
        
        print(f"\n✓ Combined multimodal data:")
        print(f"  Total samples: {X_combined.shape[0]}")
        print(f"  Total features: {X_combined.shape[1]}")
        print(f"  - Speech features: {X_speech.shape[1]}")
        print(f"  - Handwriting features: {X_handwriting.shape[1]}")
        print(f"  - Gait features: {X_gait.shape[1]}")
        print("="*60 + "\n")
        
        return X_combined, y_speech
    
    def _download_speech_data(self) -> None:
        """Download speech data from UCI repository."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        output_path = self.speech_dir / "parkinsons.csv"
        
        try:
            print(f"Downloading from {url}...")
            import ssl
            ssl_context = ssl._create_unverified_context()
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Speech Data") as t:
                urllib.request.urlretrieve(url, output_path, reporthook=t.update_to, context=ssl_context)
            print(f"✓ Downloaded successfully to {output_path}")
        except Exception as e:
            print(f"✗ Error downloading speech data: {e}")
            print(f"Please manually download from: {url}")
    
    def _download_telemonitoring_data(self) -> None:
        """Download telemonitoring data from UCI repository."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
        output_path = self.speech_dir / "parkinsons_telemonitoring.csv"
        
        try:
            print(f"Downloading from {url}...")
            import ssl
            ssl_context = ssl._create_unverified_context()
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Telemonitoring Data") as t:
                urllib.request.urlretrieve(url, output_path, reporthook=t.update_to, context=ssl_context)
            print(f"✓ Downloaded successfully to {output_path}")
        except Exception as e:
            print(f"✗ Error downloading telemonitoring data: {e}")
            print(f"Please manually download from: {url}")
    
    def get_data_info(self) -> Dict[str, any]:
        """
        Get information about loaded datasets.
        
        Returns:
            Dictionary containing dataset statistics for each modality
        """
        info = {}
        
        # Speech data
        try:
            X_speech, y_speech = self.load_speech_data()
            info['speech'] = {
                'samples': len(y_speech),
                'features': X_speech.shape[1],
                'pd_cases': int(sum(y_speech == 1)),
                'healthy': int(sum(y_speech == 0)),
                'status': 'available'
            }
        except FileNotFoundError as e:
            info['speech'] = {'status': 'missing', 'error': str(e)}
        except Exception as e:
            info['speech'] = {'status': 'error', 'error': str(e)}
        
        # Handwriting data
        try:
            X_handwriting, y_handwriting = self.load_handwriting_data()
            info['handwriting'] = {
                'samples': len(y_handwriting),
                'features': X_handwriting.shape[1],
                'pd_cases': int(sum(y_handwriting == 1)),
                'healthy': int(sum(y_handwriting == 0)),
                'status': 'available'
            }
        except FileNotFoundError as e:
            info['handwriting'] = {'status': 'missing', 'error': 'Dataset not found. See README.md and config/multimodal_features.yaml'}
        except Exception as e:
            info['handwriting'] = {'status': 'error', 'error': str(e)}
        
        # Gait data
        try:
            X_gait, y_gait = self.load_gait_data()
            info['gait'] = {
                'samples': len(y_gait),
                'features': X_gait.shape[1],
                'pd_cases': int(sum(y_gait == 1)),
                'healthy': int(sum(y_gait == 0)),
                'status': 'available'
            }
        except FileNotFoundError as e:
            info['gait'] = {'status': 'missing', 'error': 'Dataset not found. See README.md and config/multimodal_features.yaml'}
        except Exception as e:
            info['gait'] = {'status': 'error', 'error': str(e)}
        
        return info


def download_datasets() -> None:
    """Check status of all datasets and attempt to download where possible."""
    loader = DataLoader()
    
    print("=" * 60)
    print("Parkinson's Disease Dataset Status Check")
    print("=" * 60)
    
    info = loader.get_data_info()
    
    for modality, stats in info.items():
        print(f"\n{modality.upper()}:")
        if stats.get('status') == 'available':
            print(f"  ✓ Available")
            print(f"  Samples: {stats['samples']}")
            print(f"  Features: {stats['features']}")
            print(f"  PD cases: {stats['pd_cases']}, Healthy: {stats['healthy']}")
        elif stats.get('status') == 'missing':
            print("  ✗ Missing - See README.md and config/multimodal_features.yaml for download instructions")
        else:
            print(f"  ✗ Error: {stats.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("For detailed dataset information, see README.md and config/multimodal_features.yaml")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    download_datasets()
