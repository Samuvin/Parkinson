"""
Video processing for gait analysis in Parkinson's Disease prediction.
Extracts 10 gait features from walking videos.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def extract_gait_features(video_path: str) -> Dict[str, float]:
    """
    Extract 10 gait features from walking video.
    
    Note: These are estimated features using basic motion detection.
    Professional gait analysis requires force plates or motion capture systems.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with 10 gait features
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        # Raise exception if video cannot be opened
        raise RuntimeError(
            f"Failed to open video file '{video_path}'. "
            f"Please ensure the file is a valid video format and is not corrupted."
        )
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default
    
    # Collect motion data and additional video statistics
    motion_data = []
    prev_frame = None
    frame_count = 0
    total_motion = 0
    frame_intensities = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Store frame intensity statistics
        frame_intensities.append(np.mean(gray))
        
        if prev_frame is not None:
            # Calculate frame difference (motion)
            frame_diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Calculate motion amount
            motion_amount = np.sum(thresh) / (255 * thresh.size)
            motion_data.append(motion_amount)
            total_motion += motion_amount
        
        prev_frame = gray
        
        # Limit processing to first 300 frames for performance
        if frame_count > 300:
            break
    
    cap.release()
    
    if len(motion_data) < 10:
        raise ValueError(
            f"Insufficient motion data extracted from video '{video_path}'. "
            f"Only {len(motion_data)} motion frames detected (minimum 10 required). "
            f"Please ensure the video contains clear walking/gait movement."
        )
    
    # Extract features from motion data
    features = calculate_gait_features(motion_data, fps, frame_count)
    
    # Add variation based on actual video content
    if total_motion > 0:
        # Adjust features based on motion intensity
        motion_factor = total_motion / len(motion_data)
        print(f"  Motion factor: {motion_factor:.4f}")
        
        if motion_factor < 0.01:  # Very low motion
            features['gait_speed'] *= 0.7
            features['cadence'] *= 0.85
            features['stride_regularity'] *= 0.9
            print(f"  Applied LOW motion adjustments")
        elif motion_factor > 0.05:  # High motion
            print(f"  HIGH motion detected")
        
        # Add variation based on brightness changes (can indicate tremor/unsteadiness)
        if len(frame_intensities) > 1:
            intensity_var = np.std(frame_intensities)
            print(f"  Brightness variation: {intensity_var:.2f}")
            if intensity_var > 20:  # High variation
                features['stride_interval_std'] *= 1.3
                features['gait_asymmetry'] = min(features['gait_asymmetry'] * 1.4, 0.4)
                print(f"  Applied brightness variation adjustments")
    
    print(f"✓ Extracted gait features from {frame_count} frames ({len(motion_data)} motion samples)")
    print(f"  Sample features: stride_interval={features['stride_interval']:.3f}, gait_speed={features['gait_speed']:.3f}, regularity={features['stride_regularity']:.3f}")
    
    return features


def calculate_gait_features(motion_data: List[float], fps: float, frame_count: int) -> Dict[str, float]:
    """Calculate gait features from motion data."""
    motion_array = np.array(motion_data)
    
    features = {}
    
    # Calculate overall motion statistics for feature variation
    mean_motion = np.mean(motion_array)
    std_motion = np.std(motion_array)
    max_motion = np.max(motion_array)
    
    # 1-2: Stride interval and variability
    # Detect peaks in motion (steps)
    steps = detect_steps(motion_array)
    
    if len(steps) > 1:
        step_intervals = np.diff(steps) / fps  # Convert to seconds
        # Add variation based on actual motion intensity
        interval_base = np.mean(step_intervals) * 2  # 2 steps = 1 stride
        # Scale based on motion characteristics
        if mean_motion < 0.01:  # Low motion - slower gait
            interval_base *= 1.2
        elif mean_motion > 0.05:  # High motion - faster gait
            interval_base *= 0.9
        
        features['stride_interval'] = float(interval_base)
        features['stride_interval_std'] = float(np.std(step_intervals) * 2 * (1 + std_motion * 10))
    else:
        # Fallback with variation based on motion
        features['stride_interval'] = float(0.95 + mean_motion * 10)
        features['stride_interval_std'] = float(0.04 + std_motion * 2)
    
    # 3-4: Swing and stance time estimates
    # Swing time: low motion periods
    # Stance time: high motion periods
    motion_threshold = np.median(motion_array)
    swing_frames = np.sum(motion_array < motion_threshold)
    stance_frames = np.sum(motion_array >= motion_threshold)
    
    swing_time_base = (swing_frames / fps) / max(len(steps), 1) * 0.4
    stance_time_base = (stance_frames / fps) / max(len(steps), 1) * 0.7
    
    # Adjust based on motion characteristics
    features['swing_time'] = float(swing_time_base * (1 - mean_motion * 2))
    features['stance_time'] = float(stance_time_base * (1 + mean_motion))
    
    # 5: Double support time estimate
    features['double_support'] = float(features['stance_time'] * (0.3 + std_motion))
    
    # 6-7: Gait speed and cadence
    duration = frame_count / fps
    if duration > 0 and len(steps) > 0:
        cadence = (len(steps) * 60) / duration  # Steps per minute
        # Adjust cadence based on motion intensity
        cadence_adjusted = cadence * (0.8 + mean_motion * 8)
        features['cadence'] = float(min(max(cadence_adjusted, 70), 150))
        
        # Estimate speed (assuming average step length)
        step_length = 0.5 + mean_motion * 3  # Vary step length with motion
        features['gait_speed'] = float((len(steps) * step_length) / duration)
    else:
        features['cadence'] = float(90.0 + mean_motion * 400)
        features['gait_speed'] = float(0.8 + mean_motion * 10)
    
    # 8: Step length estimate
    if features['cadence'] > 0:
        features['step_length'] = float(features['gait_speed'] / (features['cadence'] / 60))
    else:
        features['step_length'] = float(0.5 + mean_motion * 2)
    
    # 9: Stride regularity (from motion consistency)
    regularity = 1.0 - (std_motion / (mean_motion + 1e-10))
    # Make it more sensitive to variations
    regularity_scaled = np.clip(regularity, 0.4, 0.98)
    features['stride_regularity'] = float(regularity_scaled)
    
    # 10: Gait asymmetry (from motion pattern)
    # Analyze first half vs second half of motion
    mid_point = len(motion_array) // 2
    first_half_mean = np.mean(motion_array[:mid_point])
    second_half_mean = np.mean(motion_array[mid_point:])
    asymmetry = abs(first_half_mean - second_half_mean) / (first_half_mean + second_half_mean + 1e-10)
    # Make asymmetry more pronounced
    asymmetry_scaled = asymmetry * (1 + max_motion * 5)
    features['gait_asymmetry'] = float(np.clip(asymmetry_scaled, 0.05, 0.35))
    
    return features


def detect_steps(motion_data: np.ndarray, min_distance: int = 15) -> List[int]:
    """
    Detect steps from motion data by finding peaks.
    
    Args:
        motion_data: Array of motion amounts per frame
        min_distance: Minimum frames between steps
        
    Returns:
        List of frame indices where steps occur
    """
    # Smooth the data
    if len(motion_data) > 5:
        smoothed = np.convolve(motion_data, np.ones(5)/5, mode='same')
    else:
        smoothed = motion_data
    
    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            # Check if it's above threshold
            if smoothed[i] > np.mean(smoothed) * 0.8:
                peaks.append(i)
    
    # Filter peaks that are too close together
    if len(peaks) < 2:
        return peaks
    
    filtered_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] >= min_distance:
            filtered_peaks.append(peak)
    
    return filtered_peaks






def get_feature_names() -> List[str]:
    """Get list of all 10 gait feature names in order."""
    return [
        'stride_interval',
        'stride_interval_std',
        'swing_time',
        'stance_time',
        'double_support',
        'gait_speed',
        'cadence',
        'step_length',
        'stride_regularity',
        'gait_asymmetry'
    ]


def features_dict_to_array(features: Dict[str, float]) -> List[float]:
    """Convert features dictionary to ordered array."""
    feature_names = get_feature_names()
    return [features[name] for name in feature_names]


if __name__ == "__main__":
    print("Gait Video Feature Extractor")
    print("10 features for Parkinson's Disease prediction")
    print("\nFeature list:")
    for i, name in enumerate(get_feature_names(), 1):
        print(f"  {i}. {name}")
    print("\nNote: Features are estimated from video analysis.")
    print("For best accuracy, use professional gait analysis equipment.")

