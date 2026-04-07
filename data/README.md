# Data directory

## Voice-first (default)

For **[`train.py`](../train.py)** (sklearn speech models), you only need:

- [`raw/speech/parkinsons.csv`](raw/speech/parkinsons.csv) — 22 voice features + `status`

Refresh speech files:

```bash
bash data/raw/speech/download_speech_csvs.sh
# or
python scripts/train_voice_pipeline.py --download-speech
```

**Demo audio (Jupyter only):** [`examples/speech/`](examples/speech/) — synthetic `healthy_example.wav` / `parkinson_example.wav` (run `python scripts/generate_example_speech_wavs.py` to recreate). Training still uses `raw/speech/parkinsons.csv`.

---

## Folder layout

```
data/raw/
  speech/       # voice / tabular speech features  ← start here
  handwriting/  # optional: multimodal DL only
  gait/         # optional: multimodal DL only
data/examples/
  speech/       # optional short WAVs for notebooks (not training at scale)
```

## Multimodal training (`train_dl.py` — optional)

[`load_all_modalities`](../dl_models/dataset.py) also reads handwriting and gait:

| Path | Description |
|------|-------------|
| [`raw/speech/parkinsons.csv`](raw/speech/parkinsons.csv) | 22 speech features + `status` (same schema as [UCI 174](https://archive.ics.uci.edu/dataset/174/parkinsons)). |
| [`raw/handwriting/handwriting_data.csv`](raw/handwriting/handwriting_data.csv) | Ten handwriting features + `status`. |
| [`raw/gait/gait_data.csv`](raw/gait/gait_data.csv) | Ten gait features + `status`. |

## Other speech CSVs

Extra public speech tables (different schemas) are in [`raw/speech/`](raw/speech/) with provenance in [`raw/speech/SPEECH_DATASETS_SOURCES.txt`](raw/speech/SPEECH_DATASETS_SOURCES.txt). Refresh with:

```bash
bash data/raw/speech/download_speech_csvs.sh
```

See **[../DATASETS.md](../DATASETS.md)** for column names, licenses, Kaggle/UCI notes, and row-alignment for multimodal training.
