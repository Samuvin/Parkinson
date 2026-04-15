"""
Handwriting image processing for Parkinson's Disease prediction.
Extracts 10 image-based features from spiral or word drawings.

Feature names are aligned with ``config/multimodal_features.yaml``
``handwriting_features`` and with the training CSV column names.
"""

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from skimage import morphology, filters
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Skeleton turn count used when approximating stroke-width variance from
# a raster image via coefficient-of-variation of distance-transform values.
_SPIRAL_TURNS = 3


def extract_handwriting_features(image_path: str) -> Dict[str, float]:
    """
    Extract 10 image-based handwriting features from a spiral/word drawing.

    Args:
        image_path: Path to PNG/JPEG image of a handwriting sample.

    Returns:
        Dictionary with 10 features whose keys match
        ``config/multimodal_features.yaml`` ``handwriting_features``.
    """
    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Binarise: ink = 255 (dark strokes on light background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    feats: Dict[str, float] = {}

    skeleton = morphology.skeletonize(binary > 0).astype(np.uint8) * 255

    # 1. stroke_width_variance — coeff. of variation of stroke widths
    feats['stroke_width_variance'] = _calc_stroke_width_variance(binary)

    # 2. edge_roughness — ink contour perimeter / convex-hull perimeter ratio
    feats['edge_roughness'] = _calc_edge_roughness(binary)

    # 3. stroke_smoothness — mean absolute curvature of the skeletonised
    #    contour (lower = smoother)
    feats['stroke_smoothness'] = _stroke_smoothness(skeleton)

    # 4. contour_complexity — isoperimetric quotient of the ink region
    #    (perimeter² / area); higher = more complex / irregular
    feats['contour_complexity'] = _contour_complexity(binary)

    # 5. stroke_inflection_count — curvature-sign reversals along skeleton
    feats['stroke_inflection_count'] = _calc_stroke_inflection_count(binary)

    # 6. fragment_ratio — normalised count of disconnected ink components
    feats['fragment_ratio'] = _calc_fragment_ratio(binary)

    # 7. stroke_width_mean — mean stroke width via distance transform
    feats['stroke_width_mean'] = _calc_stroke_width_mean(binary)

    # 8. ink_hull_ratio — ink pixel area / convex-hull area ratio
    feats['ink_hull_ratio'] = _calc_ink_hull_ratio(binary)

    # 9. line_waviness — RMS perpendicular deviation of each stroke segment
    #    from its principal axis
    feats['line_waviness'] = _line_waviness(skeleton)

    # 10. ink_coverage — fraction of bounding box covered by ink
    feats['ink_coverage'] = _calc_ink_coverage(binary)

    return feats


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def _calc_stroke_width_variance(binary: np.ndarray) -> float:
    """
    Coefficient of variation of stroke widths (std / mean of distance-transform
    values on ink pixels).

    Higher values indicate more irregular stroke widths, which is a direct
    image-based tremor indicator valid for any handwriting sample.
    """
    dist = ndimage.distance_transform_edt(binary)
    ink = binary > 0
    if not np.any(ink):
        return 0.5
    vals = dist[ink]
    mean_w = float(np.mean(vals))
    if mean_w < 1e-10:
        return 0.5
    return float(np.clip(np.std(vals) / mean_w, 0.0, 1.0))


def _calc_edge_roughness(binary: np.ndarray) -> float:
    """
    Ratio of total ink contour perimeter to the convex-hull perimeter.

    A value of 1.0 means perfectly convex strokes; higher values indicate
    rougher, more irregular stroke edges — a valid tremor indicator for any
    handwriting image without requiring a spiral drawing.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0
    total_perim = sum(cv2.arcLength(c, closed=True) for c in contours)
    ys, xs = np.where(binary > 0)
    if len(xs) < 3:
        return 1.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    hull_perim = cv2.arcLength(hull, closed=True)
    if hull_perim < 1:
        return 1.0
    return float(np.clip(total_perim / hull_perim, 1.0, 10.0))


def _stroke_smoothness(skeleton: np.ndarray) -> float:
    """
    Mean absolute curvature of the skeletonised contour.

    Lower values → smoother strokes (healthy); higher → more curvature (PD).
    """
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.5

    curvatures = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(float)
        if len(pts) < 5:
            continue
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            angle = abs(np.arctan2(
                v1[0] * v2[1] - v1[1] * v2[0],
                v1[0] * v2[0] + v1[1] * v2[1],
            ))
            curvatures.append(angle)

    if not curvatures:
        return 0.5

    return float(np.clip(np.mean(curvatures) / np.pi, 0.0, 1.0))


def _contour_complexity(binary: np.ndarray) -> float:
    """
    Isoperimetric quotient: perimeter² / (4π × area).

    Circle = 1.0; more complex / irregular shapes > 1.0.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0

    total_area = sum(cv2.contourArea(c) for c in contours)
    total_perim = sum(cv2.arcLength(c, closed=False) for c in contours)

    if total_area < 1:
        return 1.0

    iq = (total_perim ** 2) / (4 * np.pi * total_area + 1e-10)
    return float(np.clip(iq, 1.0, 20.0))


def _calc_stroke_inflection_count(binary: np.ndarray) -> float:
    """
    Count of curvature-sign reversals (direction changes) in the skeleton contour.

    Mirrors the PaHaW ``pressure_Number_of_changing_point`` feature: a higher
    count indicates more irregular, tremor-affected strokes.  The raw count is
    returned so its scale is comparable to the tablet-derived training values.
    """
    skel = morphology.skeletonize(binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    total_changes = 0
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(float)
        if len(pts) < 5:
            continue
        # Signed cross-product at each interior point → curvature sign
        signs = []
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross != 0.0:
                signs.append(1 if cross > 0 else -1)
        # Count sign changes (inflections)
        for j in range(1, len(signs)):
            if signs[j] != signs[j - 1]:
                total_changes += 1

    return float(total_changes)


def _calc_fragment_ratio(binary: np.ndarray) -> float:
    """
    Normalised count of disconnected ink fragments (fragment_ratio).

    Each connected component is an isolated stroke segment.  A higher ratio
    indicates more broken, fragmented writing — a PD-relevant signal directly
    extractable from any static handwriting image.

    Formula: (n_components - 1) / sqrt(ink_pixels + 1)  clipped to [0, 1].
    """
    ink = (binary > 0).astype(np.uint8)
    n_labels, _ = cv2.connectedComponents(ink)
    n_fragments = max(n_labels - 1, 0)  # subtract background label
    ink_pixels = float(np.sum(ink))
    if ink_pixels < 1:
        return 1.0
    return float(np.clip(n_fragments / (np.sqrt(ink_pixels) + 1e-10), 0.0, 1.0))


def _calc_stroke_width_mean(binary: np.ndarray) -> float:
    """
    Mean stroke width (stroke_width_mean) via Euclidean distance transform.

    The distance transform assigns each ink pixel its distance to the nearest
    background pixel; doubling gives the local stroke diameter.  This is the
    standard image-based stroke-width estimator and is directly computable
    from any binary handwriting image.
    """
    dist = ndimage.distance_transform_edt(binary)
    ink = binary > 0
    if not np.any(ink):
        return 1.0

    widths = dist[ink] * 2.0  # diameter = 2 × inradius
    return float(np.mean(widths[widths > 0]) if np.any(widths > 0) else 1.0)


def _calc_ink_hull_ratio(binary: np.ndarray) -> float:
    """
    Ratio of ink pixels to convex-hull area of the drawing.

    Confident, fast drawing fills the convex hull efficiently (ratio → 1).
    Hesitant or tremor-affected drawing leaves internal gaps and detours
    (ratio < 1).  Produces a 0–1 value comparable to the training-data
    ``velocity_mean`` range.
    """
    ys, xs = np.where(binary > 0)
    if len(xs) < 10:
        return 0.5

    pts = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)

    if hull_area < 1:
        return 0.5

    ink_area = float(np.sum(binary > 0))
    return float(np.clip(ink_area / hull_area, 0.0, 1.0))


def _line_waviness(skeleton: np.ndarray) -> float:
    """
    RMS perpendicular deviation of skeleton pixels from their principal axis.

    Computed per connected segment; averaged across all segments.
    """
    num_labels, labels = cv2.connectedComponents(skeleton)
    waviness_scores = []

    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) < 5:
            continue

        pts = np.column_stack([xs.astype(float), ys.astype(float)])
        mean_pt = pts.mean(axis=0)
        centred = pts - mean_pt

        # PCA: principal axis from SVD
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        axis = vt[0]

        # Perpendicular distance from each point to the principal axis
        perp = centred - (centred @ axis)[:, None] * axis
        rms_dev = float(np.sqrt(np.mean(np.sum(perp ** 2, axis=1))))
        waviness_scores.append(rms_dev)

    if not waviness_scores:
        return 0.3

    return float(np.clip(np.mean(waviness_scores) / 10.0, 0.0, 1.0))


def _calc_ink_coverage(binary: np.ndarray) -> float:
    """
    Ink coverage (ink_coverage): fraction of the tight bounding box filled by ink.

    Directly measures how densely the writer covers the page.  Parkinson's
    patients tend to produce smaller, more compressed writing (micrographia),
    so both the bounding box and coverage density are affected.  The value is
    in [0, 1] and is purely image-based — no dynamic signals required.
    """
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0.0

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bbox_area = max((y1 - y0 + 1) * (x1 - x0 + 1), 1)
    return float(np.clip(float(np.sum(binary > 0)) / bbox_area, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_feature_names() -> List[str]:
    """Return the 10 handwriting feature names in training-column order."""
    return [
        'stroke_width_variance',
        'edge_roughness',
        'stroke_smoothness',
        'contour_complexity',
        'stroke_inflection_count',
        'fragment_ratio',
        'stroke_width_mean',
        'ink_hull_ratio',
        'line_waviness',
        'ink_coverage',
    ]


def features_dict_to_array(features: Dict[str, float]) -> List[float]:
    """Convert a features dict to a list ordered by ``get_feature_names()``."""
    return [features[name] for name in get_feature_names()]


if __name__ == "__main__":
    print("Handwriting Feature Extractor (image-based)")
    print("\n10 features (match config/multimodal_features.yaml):")
    for i, name in enumerate(get_feature_names(), 1):
        print(f"  {i:2d}. {name}")
