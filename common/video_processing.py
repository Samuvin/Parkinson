"""
Video processing for gait analysis in Parkinson's Disease prediction.
Extracts 10 gait features from walking videos using MediaPipe Pose estimation.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
    _mp_pose = mp.solutions.pose
except (ImportError, AttributeError):
    _MP_AVAILABLE = False
    _mp_pose = None

# Landmark indices in MediaPipe Pose (33-keypoint model)
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_HEEL = 29
_RIGHT_HEEL = 30


def extract_gait_features(video_path: str) -> Dict[str, float]:
    """
    Extract 10 gait features from a walking video.

    Uses MediaPipe Pose landmarks when available. Falls back to frame-
    differencing motion analysis when MediaPipe is not installed.

    Args:
        video_path: Path to walking video file.

    Returns:
        Dictionary with 10 gait features whose keys match
        ``config/multimodal_features.yaml`` ``gait_features``.
    """
    if _MP_AVAILABLE:
        return _extract_with_mediapipe(video_path)
    return _extract_with_motion(video_path)


# ---------------------------------------------------------------------------
# MediaPipe pose-based extraction
# ---------------------------------------------------------------------------

def _extract_with_mediapipe(video_path: str) -> Dict[str, float]:
    """Extract pose-based gait features using MediaPipe Pose."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video file '{video_path}'. "
            "Please ensure the file is a valid video format."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    left_ankle_y: List[float] = []
    right_ankle_y: List[float] = []
    hip_x: List[float] = []
    frame_count = 0

    with _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                h, w = frame.shape[:2]

                left_ankle_y.append(lm[_LEFT_ANKLE].y * h)
                right_ankle_y.append(lm[_RIGHT_ANKLE].y * h)

                mid_hip_x = (lm[_LEFT_HIP].x + lm[_RIGHT_HIP].x) / 2.0
                hip_x.append(mid_hip_x * w)

            if frame_count >= 600:
                break

    cap.release()

    if len(left_ankle_y) < 20:
        raise ValueError(
            f"Insufficient pose data from '{video_path}'. "
            f"Only {len(left_ankle_y)} frames with detected landmarks "
            "(minimum 20 required). Ensure the full body is visible."
        )

    return _compute_pose_features(
        np.array(left_ankle_y),
        np.array(right_ankle_y),
        np.array(hip_x),
        fps,
    )


def _compute_pose_features(
    la_y: np.ndarray,
    ra_y: np.ndarray,
    hip_x: np.ndarray,
    fps: float,
) -> Dict[str, float]:
    """Compute 10 gait features from ankle/hip landmark time-series."""
    feats: Dict[str, float] = {}

    # --- Heel-strike detection (ankle Y local maxima — foot lowest = highest Y) ---
    left_strikes = _find_peaks(la_y)
    right_strikes = _find_peaks(ra_y)
    all_strikes = sorted(left_strikes + right_strikes)

    if len(all_strikes) >= 2:
        intervals = np.diff(all_strikes) / fps  # seconds per step
        stride_intervals = intervals[::2] * 2 if len(intervals) >= 2 else intervals
        feats['stride_interval'] = float(np.mean(stride_intervals))
        feats['stride_variability'] = float(np.std(stride_intervals))
        feats['cadence'] = float(60.0 / feats['stride_interval']) if feats['stride_interval'] > 0 else 90.0
    else:
        feats['stride_interval'] = 1.1
        feats['stride_variability'] = 0.08
        feats['cadence'] = 90.0

    # --- Swing / stance / double-support times ---
    n = len(la_y)
    threshold = np.percentile(la_y, 40)
    left_stance = la_y >= threshold
    right_stance = ra_y >= np.percentile(ra_y, 40)

    n_left_swing = np.sum(~left_stance)
    n_left_stance = np.sum(left_stance)
    n_double = np.sum(left_stance & right_stance)
    steps = max(len(all_strikes), 1)

    feats['swing_time'] = float((n_left_swing / fps) / steps)
    feats['stance_time'] = float((n_left_stance / fps) / steps)
    feats['double_support_time'] = float((n_double / fps) / steps)

    # --- Gait speed (hip displacement per second) ---
    if len(hip_x) >= 2:
        total_disp = float(np.sum(np.abs(np.diff(hip_x))))
        duration = len(hip_x) / fps
        feats['gait_speed'] = total_disp / duration / 100.0  # normalise to m/s proxy
    else:
        feats['gait_speed'] = 1.0

    # --- Step length proxy (ankle horizontal distance at heel-strike) ---
    if left_strikes and right_strikes:
        paired = min(len(left_strikes), len(right_strikes))
        step_lengths = []
        for i in range(paired):
            # Use Y difference as step length proxy (pixels, normalised)
            diff = abs(la_y[left_strikes[i]] - ra_y[right_strikes[i]])
            step_lengths.append(diff / 100.0)
        feats['step_length'] = float(np.mean(step_lengths))
    else:
        feats['step_length'] = 0.6

    # --- Stride regularity (autocorrelation of left ankle Y) ---
    if len(la_y) >= 40:
        norm = la_y - np.mean(la_y)
        autocorr = np.correlate(norm, norm, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr /= (autocorr[0] + 1e-10)
        stride_lag = max(int(feats['stride_interval'] * fps), 1)
        if stride_lag < len(autocorr):
            feats['stride_regularity'] = float(np.clip(autocorr[stride_lag], 0.0, 1.0))
        else:
            feats['stride_regularity'] = 0.7
    else:
        feats['stride_regularity'] = 0.7

    # --- Gait asymmetry (left vs right step time difference) ---
    if len(left_strikes) >= 2 and len(right_strikes) >= 2:
        left_step_t = float(np.mean(np.diff(left_strikes))) / fps
        right_step_t = float(np.mean(np.diff(right_strikes))) / fps
        total = left_step_t + right_step_t
        feats['gait_asymmetry'] = abs(left_step_t - right_step_t) / (total + 1e-10)
    else:
        feats['gait_asymmetry'] = 0.05

    return feats


def _find_peaks(signal: np.ndarray, min_dist: int = 10) -> List[int]:
    """Find local maxima in a 1-D signal with minimum separation."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
    return peaks


# ---------------------------------------------------------------------------
# Fallback: frame-differencing motion analysis
# ---------------------------------------------------------------------------

def _extract_with_motion(video_path: str) -> Dict[str, float]:
    """Fallback gait extractor using frame-difference motion signals."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video file '{video_path}'. "
            "Please ensure the file is a valid video format."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    motion_data: List[float] = []
    prev_frame: Optional[np.ndarray] = None
    frame_count = 0
    frame_intensities: List[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        frame_intensities.append(float(np.mean(gray)))

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_data.append(float(np.sum(thresh) / (255 * thresh.size)))

        prev_frame = gray
        if frame_count >= 300:
            break

    cap.release()

    if len(motion_data) < 10:
        raise ValueError(
            f"Insufficient motion data from '{video_path}' "
            f"({len(motion_data)} frames, minimum 10)."
        )

    return _motion_features(motion_data, fps, frame_count, frame_intensities)


def _motion_features(
    motion_data: List[float],
    fps: float,
    frame_count: int,
    frame_intensities: List[float],
) -> Dict[str, float]:
    """Compute 10 gait features from frame-difference motion signal."""
    arr = np.array(motion_data)
    mean_m = float(np.mean(arr))
    std_m = float(np.std(arr))
    max_m = float(np.max(arr))

    steps = _find_peaks(arr)

    if len(steps) > 1:
        step_ints = np.diff(steps) / fps
        stride_int = float(np.mean(step_ints) * 2)
        stride_var = float(np.std(step_ints) * 2)
    else:
        stride_int = 0.95 + mean_m * 10
        stride_var = 0.04 + std_m * 2

    threshold = float(np.median(arr))
    swing_frames = int(np.sum(arr < threshold))
    stance_frames = int(np.sum(arr >= threshold))

    feats: Dict[str, float] = {
        'stride_interval': stride_int,
        'stride_variability': stride_var,
        'swing_time': float(swing_frames / fps / max(len(steps), 1) * 0.4),
        'stance_time': float(stance_frames / fps / max(len(steps), 1) * 0.7),
        'double_support_time': float(stance_frames / fps / max(len(steps), 1) * 0.7 * (0.3 + std_m)),
        'gait_speed': float(len(steps) * (0.5 + mean_m * 3) / (frame_count / fps))
        if frame_count > 0 else 1.0,
        'cadence': float(min(max(len(steps) * 60 / (frame_count / fps) * (0.8 + mean_m * 8), 70), 150))
        if frame_count > 0 else 90.0,
        'step_length': 0.0,
        'stride_regularity': float(np.clip(1.0 - std_m / (mean_m + 1e-10), 0.4, 0.98)),
        'gait_asymmetry': 0.0,
    }

    if feats['cadence'] > 0:
        feats['step_length'] = feats['gait_speed'] / (feats['cadence'] / 60.0)

    mid = len(arr) // 2
    asym = abs(np.mean(arr[:mid]) - np.mean(arr[mid:])) / (np.mean(arr) + 1e-10)
    feats['gait_asymmetry'] = float(np.clip(asym * (1 + max_m * 5), 0.05, 0.35))

    # Adjust for brightness variation (lighting flicker can indicate tremor)
    if len(frame_intensities) > 1 and np.std(frame_intensities) > 20:
        feats['stride_variability'] = min(feats['stride_variability'] * 1.3, 0.5)
        feats['gait_asymmetry'] = min(feats['gait_asymmetry'] * 1.4, 0.4)

    return feats


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_feature_names() -> List[str]:
    """Return the 10 gait feature names in training-column order."""
    return [
        'stride_interval',
        'stride_variability',
        'swing_time',
        'stance_time',
        'double_support_time',
        'gait_speed',
        'cadence',
        'step_length',
        'stride_regularity',
        'gait_asymmetry',
    ]


def features_dict_to_array(features: Dict[str, float]) -> List[float]:
    """Convert a features dict to a list ordered by ``get_feature_names()``."""
    return [features[name] for name in get_feature_names()]


if __name__ == "__main__":
    backend = "MediaPipe Pose" if _MP_AVAILABLE else "frame-differencing (install mediapipe for better accuracy)"
    print(f"Gait Feature Extractor — backend: {backend}")
    print("\n10 gait features (match config/multimodal_features.yaml):")
    for i, name in enumerate(get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
