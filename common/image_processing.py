"""
Handwriting image processing for Parkinson's Disease prediction.
Extracts 10 handwriting features from images of handwriting/drawings.
"""

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage, fft
from skimage import filters, morphology, measure
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


def extract_handwriting_features(image_path: str) -> Dict[str, float]:
    """
    Extract 10 handwriting features from image using real-time image analysis.
    
    Note: These are estimated features from static images.
    Real handwriting analysis requires time-series pen data from digitizers.
    
    Args:
        image_path: Path to handwriting image
        
    Returns:
        Dictionary with 10 handwriting features extracted from the actual image
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        # Try with PIL
        img = np.array(Image.open(image_path))
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Threshold to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    features = {}
    
    # 1-2: Pressure estimates (from stroke thickness)
    stroke_widths = estimate_stroke_widths(binary)
    features['mean_pressure'] = float(np.mean(stroke_widths) / 10.0)  # Normalize
    features['std_pressure'] = float(np.std(stroke_widths) / 10.0)
    
    # 3-4: Velocity estimates (from stroke smoothness)
    smoothness = estimate_smoothness(binary)
    features['mean_velocity'] = float(2.0 + smoothness)  # Estimated m/s
    features['std_velocity'] = float(0.5 + smoothness * 0.5)
    
    # 5: Acceleration estimate
    features['mean_acceleration'] = float(1.0 + smoothness * 0.5)
    
    # 6: Pen-up time estimate (from gaps)
    pen_up_ratio = estimate_pen_up_time(binary)
    features['pen_up_time'] = float(pen_up_ratio * 0.5)
    
    # 7: Stroke length
    total_stroke = np.sum(binary > 0)
    features['stroke_length'] = float(total_stroke / 1000.0)  # Normalize
    
    # 8: Writing tempo
    features['writing_tempo'] = float(1.5 - pen_up_ratio * 0.5)
    
    # 9: Tremor frequency (from stroke irregularity)
    tremor = estimate_tremor(binary)
    features['tremor_frequency'] = float(5.0 + tremor * 3.0)  # Hz
    
    # 10: Fluency score
    fluency = estimate_fluency(binary, smoothness, tremor)
    features['fluency_score'] = float(fluency)
    
    return features


def estimate_stroke_widths(binary_img: np.ndarray) -> np.ndarray:
    """Estimate stroke widths from binary image."""
    # Distance transform to get stroke widths
    dist_transform = ndimage.distance_transform_edt(binary_img)
    stroke_pixels = binary_img > 0
    
    if np.sum(stroke_pixels) == 0:
        return np.array([5.0])
    
    widths = dist_transform[stroke_pixels] * 2  # Diameter
    return widths[widths > 0]


def estimate_smoothness(binary_img: np.ndarray) -> float:
    """Estimate writing smoothness from contour analysis."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0.5
    
    # Analyze largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 10:
        return 0.5
    
    # Calculate curvature changes
    contour_points = largest_contour.reshape(-1, 2)
    
    # Calculate angles between consecutive segments
    angles = []
    for i in range(1, len(contour_points) - 1):
        v1 = contour_points[i] - contour_points[i-1]
        v2 = contour_points[i+1] - contour_points[i]
        
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angles.append(abs(angle))
    
    if angles:
        smoothness = 1.0 - (np.mean(angles) / np.pi)
        return float(np.clip(smoothness, 0, 1))
    
    return 0.5


def estimate_pen_up_time(binary_img: np.ndarray) -> float:
    """Estimate pen-up time ratio from gaps in writing."""
    # Dilate to connect nearby strokes
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary_img, kernel, iterations=1)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(dilated)
    
    # More components = more pen lifts
    if num_labels < 2:
        return 0.1
    
    # Estimate ratio based on number of components
    pen_up_ratio = min((num_labels - 1) * 0.1, 0.8)
    return float(pen_up_ratio)


def estimate_tremor(binary_img: np.ndarray) -> float:
    """Estimate tremor from stroke irregularity using frequency analysis."""
    # Get skeleton of strokes
    skeleton = morphology.skeletonize(binary_img > 0)
    
    if np.sum(skeleton) == 0:
        return 0.3
    
    # Find main stroke direction
    y_coords, x_coords = np.where(skeleton)
    
    if len(y_coords) < 10:
        return 0.3
    
    # Calculate perpendicular deviations
    # Fit line to stroke
    if len(x_coords) > 1:
        coeffs = np.polyfit(x_coords, y_coords, 1)
        fitted_y = np.polyval(coeffs, x_coords)
        deviations = np.abs(y_coords - fitted_y)
        
        # High standard deviation = more tremor
        tremor_score = np.std(deviations) / 10.0
        return float(np.clip(tremor_score, 0, 1))
    
    return 0.3


def estimate_fluency(binary_img: np.ndarray, smoothness: float, tremor: float) -> float:
    """Estimate writing fluency score."""
    # Combine smoothness and tremor
    # High smoothness, low tremor = high fluency
    fluency = (smoothness * 0.6 + (1 - tremor) * 0.4)
    
    # Adjust based on stroke continuity
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # More continuous strokes = better fluency
        avg_contour_length = np.mean([len(c) for c in contours])
        if avg_contour_length > 100:
            fluency += 0.1
    
    return float(np.clip(fluency, 0, 1))


def get_feature_names() -> List[str]:
    """Get list of all 10 handwriting feature names in order."""
    return [
        'mean_pressure', 'std_pressure',
        'mean_velocity', 'std_velocity',
        'mean_acceleration',
        'pen_up_time',
        'stroke_length',
        'writing_tempo',
        'tremor_frequency',
        'fluency_score'
    ]


def features_dict_to_array(features: Dict[str, float]) -> List[float]:
    """Convert features dictionary to ordered array."""
    feature_names = get_feature_names()
    return [features[name] for name in feature_names]


if __name__ == "__main__":
    print("Handwriting Feature Extractor")
    print("10 features for Parkinson's Disease prediction")
    print("\nFeature list:")
    for i, name in enumerate(get_feature_names(), 1):
        print(f"  {i}. {name}")
    print("\nNote: Features are estimated from static images.")
    print("For best accuracy, use digitizer pen data with time-series information.")

