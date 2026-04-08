# Scripts: what happens when you run them

Run these from your machine with the repo root as the working directory unless a path says otherwise. Paths below are relative to the repository root.

---

## Entry points at the repo root (not under `scripts/`)

| Command | What happens |
|--------|----------------|
| `python wsgi.py` | Loads `.env` from the project root, builds the Flask app (`create_app()`), and starts **Waitress** on `0.0.0.0` (port from `PORT`, default **8000**). Serves the web UI and `/api/*` routes. |
| `python train_dl.py` | Reads `config/project.yaml` and `config/multimodal_features.yaml`, loads aligned speech / handwriting / gait CSVs from `data/raw/`, trains **MultimodalPDNet**, and writes under `models/`: `multimodal_pdnet.pt`, `dl_scalers.joblib`, `dl_model_metrics.json`, and training plots. Optional CLI: `--epochs`, `--batch-size`, `--device`, `--no-smote`, `--config`. |

---

## Shell helpers in `scripts/`

Use these from **bash** (macOS/Linux terminal, **VS Code integrated terminal** with a bash profile, or **Git Bash** on Windows).

| Command | What happens |
|--------|----------------|
| `./scripts/run_app.sh` | Creates `venv/` if missing, activates it, installs `requirements.txt` + `requirements-ml.txt`, then runs **`python wsgi.py`** (same end result as a manual venv + wsgi, with deps refreshed). |
| `./scripts/run_notebook.sh` | Same venv setup, installs `requirements.txt` + `requirements-ml.txt` + `requirements-notebooks.txt`, then starts **Jupyter Notebook** opening `notebooks/parkinson_multimodal_staff_demo.ipynb`. |
| `./scripts/train_model.sh` | Same venv setup with ML deps, then runs **`python train_dl.py`**, forwarding any extra arguments (e.g. `./scripts/train_model.sh --epochs 50 --device cuda`). |

---

## Python utilities in `scripts/`

| Command | What happens |
|--------|----------------|
| `python scripts/train_voice_pipeline.py` | With **no flags**, prints a short message and reminds you to use `train_dl.py`; it does not download by default. |
| `python scripts/train_voice_pipeline.py --download-speech` | Downloads fixed public **speech CSVs** (HTTPS URLs baked into the script) into `data/raw/speech/` (same idea as the bash downloader below). |
| `python scripts/generate_example_speech_wavs.py` | Writes **short synthetic** WAV files under `data/examples/speech/` for notebooks and smoke tests—not real patient audio. |
| `python scripts/build_handwriting_gait_from_public_sources.py` | Downloads/builds **handwriting** and **gait** tabular CSVs from fixed public sources (PaHaW-derived data, PhysioNet gaitpdb) into paths compatible with multimodal training; uses argparse (e.g. `--n-rows`) to align row counts with your speech CSV. |
| `python scripts/uci_spiral_traces_to_csv.py` | Reads **UCI 395** spiral trace text files (after you unpack them) and writes a summary CSV; optional flags approximate project handwriting column names for exploration—it is **not** a drop-in replacement for the main `handwriting_data.csv` pipeline. |

---

## Shell downloaders under `data/raw/`

| Command | What happens |
|--------|----------------|
| `bash data/raw/speech/download_speech_csvs.sh` | Uses `curl` to download several **fixed-URL** speech-related CSVs into `data/raw/speech/` (parkinsons, telemonitoring, extra voice feature tables). |
| `bash data/raw/handwriting/download_handwriting_sources.sh` | Downloads the **UCI 395** spiral archive zip into `data/raw/handwriting/uci_spiral/` and unpacks it. Optionally, with `DOWNLOAD_KAGGLE_HANDWRITING=1`, also fetches Kaggle spiral data if the **kaggle** CLI is configured. |
| `bash data/raw/gait/download_gait_sources.sh` | By default, **does nothing** except print hints (Kaggle gait download is opt-in). With `DOWNLOAD_KAGGLE_GAIT=1`, downloads/unzips a Kaggle gait dataset mirror into `data/raw/gait/kaggle_mirror/` (requires **kaggle** CLI). For the ten-feature `gait_data.csv` used by `train_dl.py`, use `build_handwriting_gait_from_public_sources.py` instead. |

---

## `start.sh` (repo root)

| Command | What happens |
|--------|----------------|
| `./start.sh` | Ensures `venv/` exists, activates it, installs `requirements.txt` + `requirements-ml.txt`, tries to stop prior Waitress/wsgi processes, then runs **`python wsgi.py`**. Same general outcome as `./scripts/run_app.sh`, with extra “kill old server” steps. |
