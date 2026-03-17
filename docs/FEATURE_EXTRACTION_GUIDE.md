# Feature Extraction Guide

**Parkinson's Disease Early Prediction System**

This document explains **what data we accept**, **what we extract from it**, and **why each feature matters** for prediction. It is written so that anyone reading it can understand exactly what the system measures from **Video Gait** and **Handwriting** (and briefly, **Voice**) data.

---

## Table of Contents

1. [Overview: What We Extract](#overview-what-we-extract)
2. [Video Gait Features (10 features)](#video-gait-features-10-features)
3. [Handwriting Features (10 features)](#handwriting-features-10-features)
4. [Voice/Speech Features (22 features) — Summary](#voice-speech-features-22-features--summary)
5. [How the Extracted Features Are Used](#how-the-extracted-features-are-used)

---

## Overview: What We Extract

The system supports **three modalities** of input:

| Modality        | Input type        | Number of features | Purpose |
|----------------|-------------------|--------------------|--------|
| **Video Gait** | Walking video     | **10**             | How the person walks (stride, speed, symmetry) |
| **Handwriting**| Image of writing   | **10**             | How the person writes (pressure, tremor, fluency) |
| **Voice**      | Audio (speech)    | **22**             | How the person speaks (pitch, jitter, shimmer) |

When you **upload**:

- A **video** → we can extract **gait** (walking) features and, optionally, **voice** (if the video has audio) and **handwriting** (from a single frame).
- An **image** (handwriting/drawing) → we extract **handwriting** features only.
- An **audio** file → we extract **voice/speech** features only.

The rest of this guide focuses in detail on **Video Gait** and **Handwriting** features.

---

## Video Gait Features (10 features)

**Source:** A **walking video** (e.g. person walking toward or across the camera).  
**Method:** We analyze **motion** between consecutive frames: frame differencing, motion amount per frame, and step detection. We do **not** use force plates or motion capture; these are **estimated** gait features suitable for screening, not clinical diagnosis.

### What We Extract

| # | Feature name           | What it is (plain English) | How we get it |
|---|------------------------|----------------------------|---------------|
| 1 | **stride_interval**    | Time (seconds) for one full stride (left step + right step). | Peaks in motion over time → step times → average step interval × 2. |
| 2 | **stride_interval_std**| How much stride time varies (consistency of walking rhythm). | Standard deviation of step intervals (scaled). Higher = less regular. |
| 3 | **swing_time**        | Time the foot is in the air (not on the ground) per step. | Low-motion frames vs high-motion frames; proportion assigned to “swing”. |
| 4 | **stance_time**       | Time the foot is on the ground per step. | Same motion split; proportion assigned to “stance”. |
| 5 | **double_support**    | Time when both feet are on the ground (between steps). | Derived from stance time and motion variability. |
| 6 | **gait_speed**        | Walking speed (e.g. m/s or proportional units). | (Number of steps × estimated step length) / video duration. |
| 7 | **cadence**           | Steps per minute. | (Steps counted in video × 60) / duration in seconds; clipped to a realistic range (e.g. 70–150). |
| 8 | **step_length**       | Estimated length of one step. | Gait speed / (cadence/60), or from motion intensity. |
| 9 | **stride_regularity** | How consistent the walking pattern is (0–1 style). | 1 minus (motion variability / mean motion). Higher = more regular. |
| 10| **gait_asymmetry**    | Difference between left and right (or first vs second half of video). | Compare mean motion in first half vs second half of the sequence; scaled to 0.05–0.35. |

### Summary for readers

- **Timing:** stride_interval, stride_interval_std, swing_time, stance_time, double_support.  
- **Speed & rhythm:** gait_speed, cadence, step_length.  
- **Pattern quality:** stride_regularity, gait_asymmetry.

These 10 numbers are what we **extract from the video** and feed into the model. In Parkinson’s, gait often becomes slower, less regular, and more asymmetric; these features help capture that.

---

## Handwriting Features (10 features)

**Source:** A **single image** of handwriting or a drawing (e.g. spiral, sentence).  
**Method:** We do **image analysis** only: binary strokes, contours, skeletons, and simple frequency-style measures. We do **not** use pen digitizers or time-series pen data; these are **estimated** features from the static image.

### What We Extract

| # | Feature name            | What it is (plain English) | How we get it |
|---|-------------------------|----------------------------|---------------|
| 1 | **mean_pressure**       | Average “pressure” (stroke thickness). | Distance transform on strokes → mean stroke width, normalized (e.g. /10). |
| 2 | **std_pressure**        | Variation in pressure (thick vs thin strokes). | Standard deviation of stroke widths. |
| 3 | **mean_velocity**       | Estimated writing speed (arbitrary units). | From stroke smoothness (fewer sharp turns → “faster”). |
| 4 | **std_velocity**       | Variation in writing speed. | From smoothness variation. |
| 5 | **mean_acceleration**   | How much “acceleration” changes (smooth vs jerky). | Derived from smoothness (e.g. 1.0 + smoothness×0.5). |
| 6 | **pen_up_time**         | Proportion of time the pen is off the paper (gaps between strokes). | Connected components after light dilation → more separate strokes → higher pen-up ratio; scaled to time-like value. |
| 7 | **stroke_length**       | Total inked length (stroke pixel count). | Sum of binary stroke pixels, normalized (e.g. /1000). |
| 8 | **writing_tempo**       | Overall “tempo” of writing (strokes per time, conceptually). | Inverse relationship with pen_up_ratio (e.g. 1.5 − pen_up×0.5). |
| 9 | **tremor_frequency**    | How much high-frequency “wobble” is in the stroke (Hz-like). | Skeleton of strokes → fit line → perpendicular deviations → higher deviation std → higher tremor; scaled to ~5–8 Hz. |
| 10| **fluency_score**      | How smooth and continuous the writing is (0–1). | Combination of smoothness (contour angles) and tremor; longer continuous contours add a small bonus. |

### Summary for readers

- **Pressure (from thickness):** mean_pressure, std_pressure.  
- **Motion (from shape):** mean_velocity, std_velocity, mean_acceleration, writing_tempo.  
- **Stroke structure:** pen_up_time, stroke_length.  
- **Tremor & quality:** tremor_frequency, fluency_score.

These 10 numbers are what we **extract from the handwriting image** and feed into the model. In Parkinson’s, handwriting often shows micrographia, tremor, and reduced fluency; these features aim to capture that from a single image.

---

## Voice/Speech Features (22 features) — Summary

When **audio** (or video with audio) is processed, we extract **22 speech features** aligned with the UCI Parkinson’s speech dataset:

- **Pitch:** MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)  
- **Jitter (pitch variation):** MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP  
- **Shimmer (amplitude variation):** MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA  
- **Noise/quality:** NHR, HNR  
- **Nonlinear dynamics:** RPDE, DFA, spread1, spread2, D2, PPE  

These are computed with **Praat (Parselmouth)** and **librosa** from the audio signal. They are not described in full detail here; the focus of this guide is **Video Gait** and **Handwriting**.

---

## How the Extracted Features Are Used

1. **Upload**  
   - User uploads a **video** (gait and/or combined voice/handwriting) or an **image** (handwriting) or **audio** (voice).

2. **Extraction**  
   - **Video:** We extract **gait** from the full video (motion); optionally **voice** from extracted audio and **handwriting** from one extracted frame.  
   - **Image:** We extract only **handwriting** features.  
   - **Audio:** We extract only **voice** features.

3. **Feature vectors**  
   - Gait: 10 numbers (ordered list as in the table).  
   - Handwriting: 10 numbers (ordered list as in the table).  
   - Voice: 22 numbers.

4. **Prediction**  
   - These vectors are passed to the **multimodal model** (e.g. SE-ResNet + Attention Fusion or the ML fallback). The model outputs a **risk/score** (and optionally explainability). No raw video or image is stored for prediction; only the extracted feature vectors are used.

5. **Interpretation**  
   - **Gait:** Low speed, low regularity, high asymmetry and stride variability can be associated with gait impairment.  
   - **Handwriting:** Low fluency, high tremor frequency, high pen-up time, and irregular pressure can be associated with motor impairment.  
   - The system is for **research and screening** only; it is **not** a substitute for clinical diagnosis.

---

## Quick Reference: Feature Lists

**Video Gait (10):**  
`stride_interval`, `stride_interval_std`, `swing_time`, `stance_time`, `double_support`, `gait_speed`, `cadence`, `step_length`, `stride_regularity`, `gait_asymmetry`

**Handwriting (10):**  
`mean_pressure`, `std_pressure`, `mean_velocity`, `std_velocity`, `mean_acceleration`, `pen_up_time`, `stroke_length`, `writing_tempo`, `tremor_frequency`, `fluency_score`

**Voice (22):**  
See “Voice/Speech Features” above or the code in `utils/audio_processing.py` (`get_feature_names()`).

---

*This guide reflects the feature extraction logic in `utils/video_processing.py` (gait), `utils/image_processing.py` (handwriting), and `utils/audio_processing.py` (speech).*
