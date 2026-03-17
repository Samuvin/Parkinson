# Feature Extraction and Model Input — Descriptive Guide

This document describes **how we extract features** from each modality (which libraries we use and what we get), and **what we do when we have all three values vs when we do not have some values**.

For the full pipeline diagram (feature extraction → encoder → attention → classifier → output), see **ALGORITHM_INTERNALS.puml**.

---

## 1. Audio (Speech) — We use soundfile/librosa and Parselmouth, and we get 22 numbers

**What we use**

- **Load audio:** We use **soundfile** (preferred, fast) or **librosa** to load the audio file. We convert to mono and float32 in the range [-1, 1].
- **Analyse:** We use **Parselmouth** (Python interface to Praat) for voice analysis, and **numpy** for nonlinear measures.

**What we do**

1. Load the audio (soundfile or librosa).
2. Build a Praat Sound object (Parselmouth).
3. **Pitch:** Praat “To Pitch” → we get fundamental frequency (Fo, Fhi, Flo).
4. **Jitter:** Praat “To PointProcess” + jitter measures → local %, Abs, RAP, PPQ, DDP.
5. **Shimmer:** Praat shimmer (local, dB, APQ3, APQ5, APQ11, DDA).
6. **HNR/NHR:** Praat “To Harmonicity” → mean HNR; we derive NHR.
7. **Nonlinear (numpy):** We compute RPDE, DFA, spread1, spread2, D2, PPE from the pitch/f0 signal.

**Output we get (22 numbers, in this order)**

| # | Feature name     | Meaning (short)        |
|---|------------------|------------------------|
| 1 | MDVP:Fo(Hz)      | Mean fundamental frequency |
| 2 | MDVP:Fhi(Hz)     | Max fundamental frequency   |
| 3 | MDVP:Flo(Hz)     | Min fundamental frequency   |
| 4 | MDVP:Jitter(%)   | Jitter (local %)           |
| 5 | MDVP:Jitter(Abs) | Jitter (absolute)          |
| 6 | MDVP:RAP         | Relative Average Perturbation |
| 7 | MDVP:PPQ         | Pitch Perturbation Quotient   |
| 8 | Jitter:DDP       | Jitter DDP                   |
| 9 | MDVP:Shimmer     | Shimmer (local)              |
| 10 | MDVP:Shimmer(dB) | Shimmer (dB)                 |
| 11 | Shimmer:APQ3     | Shimmer APQ3                 |
| 12 | Shimmer:APQ5     | Shimmer APQ5                 |
| 13 | MDVP:APQ         | Shimmer APQ11                |
| 14 | Shimmer:DDA      | Shimmer DDA                  |
| 15 | NHR              | Noise-to-Harmonics Ratio     |
| 16 | HNR              | Harmonics-to-Noise Ratio    |
| 17 | RPDE             | Recurrence Period Density Entropy |
| 18 | DFA              | Detrended Fluctuation Analysis   |
| 19 | spread1          | Nonlinear spread 1             |
| 20 | spread2          | Nonlinear spread 2             |
| 21 | D2               | Correlation dimension          |
| 22 | PPE              | Pitch Period Entropy            |

So: **we use soundfile/librosa to load and Parselmouth + numpy to analyse, and we get these 22 outputs** (UCI order).

---

## 2. Handwriting — We use OpenCV, scipy, and skimage, and we get 10 numbers

**What we use**

- **Load image:** We use **OpenCV (cv2)** or **PIL** to load the image.
- **Preprocess:** Grayscale, then **binary** with **Otsu** threshold (OpenCV).
- **Analyse:** **OpenCV** (contours, connected components), **scipy** (distance_transform_edt, ndimage), **skimage** (morphology, skeleton).

**What we do**

1. Load image → grayscale → binary (Otsu).
2. **Stroke width** (scipy distance transform) → we derive mean and std of “pressure”.
3. **Smoothness** (OpenCV contours, curvature along contour) → we derive mean and std of “velocity”.
4. **Acceleration** from smoothness.
5. **Pen-up time** from connected components (OpenCV) — ratio of “gaps” in writing.
6. **Stroke length** from number of ink pixels (normalised).
7. **Writing tempo** from pen-up and stroke info.
8. **Tremor** from skeleton (skimage) and deviations from a fitted line.
9. **Fluency** as a function of smoothness and tremor.

**Output we get (10 numbers, in this order)**

| # | Feature name     | Meaning (short)           |
|---|------------------|---------------------------|
| 1 | mean_pressure    | Mean stroke width (pressure) |
| 2 | std_pressure     | Std of stroke width         |
| 3 | mean_velocity    | Mean velocity estimate      |
| 4 | std_velocity     | Std of velocity             |
| 5 | mean_acceleration | Mean acceleration estimate |
| 6 | pen_up_time      | Pen-up time ratio            |
| 7 | stroke_length    | Normalised stroke length     |
| 8 | writing_tempo    | Writing tempo                |
| 9 | tremor_frequency | Tremor estimate (Hz)         |
| 10 | fluency_score    | Fluency (smoothness, tremor) |

So: **we use OpenCV, scipy, and skimage to load and analyse the image, and we get these 10 outputs**.

---

## 3. Gait — We use OpenCV (video), and we get 10 numbers

**What we use**

- **Open video:** We use **OpenCV (cv2)** — **VideoCapture** — to open the video and read frames.
- **Preprocess:** Each frame → **grayscale**, **GaussianBlur** (OpenCV).
- **Motion:** **absdiff(prev, curr)** (OpenCV) → threshold → **motion per frame**.
- **Steps:** Peak detection in the motion signal → step intervals; we derive swing, stance, cadence, speed, etc.

**What we do**

1. Open video with OpenCV; read frames (grayscale, blur).
2. **Motion:** Absolute difference between consecutive frames → threshold → motion amount per frame.
3. **Peak detection** in motion array → step indices → step intervals (in time).
4. From step intervals and motion we derive: stride interval, stride variability, swing time, stance time, double support, gait speed, cadence, step length, stride regularity, gait asymmetry.

**Output we get (10 numbers, in this order)**

| # | Feature name        | Meaning (short)        |
|---|---------------------|------------------------|
| 1 | stride_interval     | Mean stride interval   |
| 2 | stride_interval_std| Std of stride interval |
| 3 | swing_time          | Swing phase time       |
| 4 | stance_time         | Stance phase time      |
| 5 | double_support      | Double support time    |
| 6 | gait_speed          | Gait speed              |
| 7 | cadence             | Steps per minute        |
| 8 | step_length         | Step length             |
| 9 | stride_regularity   | Stride regularity      |
| 10 | gait_asymmetry      | Left–right asymmetry   |

So: **we use OpenCV to open the video and compute motion and steps, and we get these 10 outputs**.

---

## 4. What we do if we have all three values

When the user provides **audio, handwriting image, and gait video**:

1. We run **all three** feature extractors (as above) and get:
   - **Speech:** 22 numbers  
   - **Handwriting:** 10 numbers  
   - **Gait:** 10 numbers  

2. We **scale** each vector with the saved scalers (trained on the same feature order).

3. We pass **all three** vectors into the model:
   - Each encoder (speech, handwriting, gait) gets its **real** 22 or 10 values.
   - The model produces three 64-d embeddings, then **attention fusion** assigns three weights (e.g. 0.4, 0.35, 0.25) that sum to 1.
   - The **fused** vector is a weighted sum of the three embeddings; the classifier predicts from this.

4. The user gets: **prediction** (Parkinson’s / Healthy), **confidence**, and **attention_weights** (how much the model used speech vs handwriting vs gait).

So: **if we have all three values, we use all three in the same model and attention decides how much to trust each.**

---

## 5. What we do if we do not have some values

When the user provides **only one or two** modalities (e.g. only gait, or only speech and handwriting):

1. We run the extractors **only for the modalities the user sent** and get the corresponding lists (e.g. only 10 gait numbers).

2. For **missing** modalities we **do not** run extraction. Instead we **fill with zeros**:
   - Missing **speech** → we pass a vector of **22 zeros** (after scaling, the scaler is still applied to this zero vector).
   - Missing **handwriting** → **10 zeros**.
   - Missing **gait** → **10 zeros**.

3. The model **always** receives **three** inputs:
   - Real data for the modality(ies) provided.
   - Zero-filled vectors for the missing one(s).

4. Inside the model:
   - Encoders that get **real** data produce **strong** 64-d embeddings.
   - Encoders that get **zeros** produce **weak** / nearly constant embeddings.
   - **Attention fusion** scores all three embeddings and applies softmax. The model has learned to put **high weight** on embeddings with real signal and **low weight** on the zero-filled ones (e.g. only gait → weights like [0.02, 0.03, **0.95**]).
   - The **fused** vector is then effectively the embedding of the modality(ies) that had real data; the rest are ignored.

5. The user still gets: **prediction**, **confidence**, and **attention_weights** (e.g. “95% gait, 3% speech, 2% handwriting” when only gait was sent).

So: **if we do not have some values, we zero-fill the missing modalities and run the same model; attention automatically relies on the modalities that were provided and ignores the rest.**

---

## Summary table

| Scenario              | What we pass to the model                    | What attention does                          |
|-----------------------|----------------------------------------------|---------------------------------------------|
| All three provided    | 22 + 10 + 10 real values                     | Weights e.g. 0.4, 0.35, 0.25 (learned)      |
| Only speech           | 22 real, 10 zeros, 10 zeros                  | High weight on speech, ~0 on others          |
| Only handwriting      | 22 zeros, 10 real, 10 zeros                  | High weight on handwriting, ~0 on others    |
| Only gait             | 22 zeros, 10 zeros, 10 real                  | High weight on gait, ~0 on others           |
| Speech + gait         | 22 real, 10 zeros, 10 real                   | High weights on speech and gait, ~0 on hand  |

The same model and the same code path are used in all cases; only the **inputs** (real vs zero-filled) change, and **attention** adapts which modality to trust.
