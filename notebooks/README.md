# Jupyter: multimodal staff demo

The notebook **[`parkinson_multimodal_staff_demo.ipynb`](parkinson_multimodal_staff_demo.ipynb)** walks through the same ideas as production code: aligned **speech + handwriting + gait** tabular features (from CSVs under `data/raw/`), optional **raw** demos (WAV, image, video), and feature extraction via [`common/`](../common/). Paths and column names follow [`config/multimodal_features.yaml`](../config/multimodal_features.yaml) and [`config/project.yaml`](../config/project.yaml).

## Environment

- Use a virtual environment with the project’s `requirements.txt` (or your notebook kernel pointed at that env).
- **Working directory:** open or run the notebook from the **repository root** so imports resolve (`common/`, `dl_models/`, `config/`).

## Demo media (`notebooks/sample/`)

Put **local** demo files here for the notebook cells that resolve sample paths. **Do not commit** large binaries (they are gitignored under `notebooks/sample/**`).

| You can use | Used for |
|-------------|----------|
| `parkinson_example.wav`, `healthy_example.wav`, `speech_*_figshare.wav`, `speech_demo.wav` | Speech waveform / UCI-style features |
| `Handwriting.jpg`, `handwriting_demo.jpg`, or any single `.jpg`/`.png` in this folder | Handwriting image features |
| Any `.mp4`/`.avi`/… (e.g. a walking clip), or `gait_demo.avi` | Gait video features |

Bundled **synthetic** WAVs for quick tests may also live under [`data/examples/speech/`](../data/examples/speech/); regenerate with `python scripts/generate_example_speech_wavs.py` if needed.

## Web app

How URLs and `POST /api/detect` flow through Flask is documented in the **[root `README.md`](../README.md)**.
