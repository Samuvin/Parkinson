# Parkinson's Disease Early Prediction System

A Parkinson's Disease prediction system with a **voice-first** training path (tabular speech / UCI-style features) and an optional **multimodal** deep-learning path (speech + handwriting + gait).

## Project Overview

Parkinson's Disease is the second most common neurodegenerative disorder, causing tremors, stiffness, and slow movement. Early prediction is crucial for improving patient outcomes. **By default, focus on voice:** [`train.py`](train.py) trains classical models on **22 speech features** from [`data/raw/speech/parkinsons.csv`](data/raw/speech/parkinsons.csv). When you are ready, [`train_dl.py`](train_dl.py) can use **SE-ResNet with attention fusion** across speech, handwriting, and gait with explainability (Grad-CAM).

## Methodology

- **Voice-first**: sklearn pipeline in [`train.py`](train.py) — speech only (22 features)
- **Multimodal (optional)**: SE-ResNet 1D + attention fusion in [`train_dl.py`](train_dl.py) — speech (22) + handwriting (10) + gait (10)
- **Explainability**: Grad-CAM and attention (multimodal DL path)
- **Class balancing**: preprocessor options in [`train.py`](train.py); SMOTE in [`train_dl.py`](train_dl.py)
- **Framework**: PyTorch for multimodal DL; sklearn for the voice-first path
- **Fallback**: sklearn ensemble (SVM + LR) when DL model is not available
- **Deployment**: Flask + Waitress WSGI server (Windows/Linux/Mac) with JWT authentication

## Datasets

See **[DATASETS.md](DATASETS.md)** for exact CSV paths, column names, row-alignment behavior, and download scripts.

### 1. Speech (voice) — start here
- **File**: `data/raw/speech/parkinsons.csv`
- **Source**: [UCI Parkinsons (174)](https://archive.ics.uci.edu/dataset/174/parkinsons) and common mirrors (e.g. [GitHub `parkinsons.csv`](https://raw.githubusercontent.com/SagarBapodara/Parkison-Disease-Detection-using-Machine-Learning/main/Data/parkinsons.csv))
- **Features**: 22 acoustic measurements (jitter, shimmer, HNR, pitch, nonlinear dynamics)
- **Used by**: [`train.py`](train.py) (and multimodal [`train_dl.py`](train_dl.py))

### 2. Handwriting & gait (optional — multimodal DL only)
- Needed only for [`train_dl.py`](train_dl.py). See **DATASETS.md** for paths, provenance, and how they differ from off-the-shelf Kaggle/UCI CSVs.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
cd Parkinson   # repository root (use your actual clone path)

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your MONGODB_URI and JWT_SECRET_KEY
```

## Project Structure

```
.
├── dl_models/            # Deep learning modules
│   ├── networks.py       # SE-ResNet + Attention Fusion architecture
│   ├── dataset.py        # PyTorch dataset for multimodal data
│   ├── trainer.py        # Training loop with early stopping
│   ├── inference.py      # DL predictor for production inference
│   └── gradcam.py        # Grad-CAM explainability
├── common/               # Shared multimodal feature extraction (speech / handwriting / gait)
├── notebooks/            # Jupyter staff demo (multimodal: tabular + raw WAV/image/video samples)
├── src/                  # sklearn fallback and utilities
│   ├── core/             # Model manager, predictor
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # sklearn model implementations
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Configuration utilities
├── webapp/               # Flask web application
│   ├── app.py            # Application factory
│   ├── api/              # REST API (predict, auth, upload)
│   ├── middleware/        # JWT authentication middleware
│   ├── models/           # User model (MongoDB)
│   ├── templates/        # Jinja2 HTML templates
│   └── static/           # CSS, JS, images
├── models/               # Saved trained models (.joblib, .pt)
├── data/                 # Datasets
├── scripts/              # Maintenance and training helpers
│   └── train_voice_pipeline.py  # Optional CSV download + voice train + reports/
├── train.py              # sklearn training pipeline
├── train_dl.py           # Deep learning training pipeline
├── wsgi.py               # WSGI entry point (Waitress)
├── config.yaml           # Hyperparameters and paths
└── requirements.txt      # Python dependencies
```

## Usage

### 1. Train the Deep Learning Model

```bash
python train_dl.py
```

This trains the SE-ResNet + Attention Fusion model and saves:
- Model weights to `models/multimodal_pdnet.pt`
- Feature scalers to `models/dl_*_scaler.joblib`
- Training plots and metrics

### 2. Run the Application

- **Full mode (default for this project):** all ML libraries, uploads, and file processing — **Linux/macOS:** `./start-full.sh`
- **Light mode:** minimal dependencies and custom prediction logic only — **Linux/macOS:** `./start.sh` · **Windows:** `start.bat`

Or manually (any platform), after activating `venv` and installing `requirements.txt`:

```bash
export USE_LIGHT_MODE=0   # full mode; omit or set to 1 for light mode
python wsgi.py
```

The server listens on port **8000** by default (override with the `PORT` environment variable). Visit `http://localhost:8000` in your browser.

### 3. Stop the Application

```bash
./stop.sh
```

### 4. Staff demo: voice pipeline + Jupyter

1. Train and export reports under `reports/`:
   ```bash
   python scripts/train_voice_pipeline.py
   ```
   Optional: `--download-speech` refreshes CSVs under `data/raw/speech/` from the same fixed HTTPS sources as `data/raw/speech/download_speech_csvs.sh`.

2. Install notebook dependencies: `pip install -r requirements-notebooks.txt`

3. Example WAVs for the staff notebook are **synthetic** and live under `data/examples/speech/`; regenerate with `python scripts/generate_example_speech_wavs.py` if needed (see that folder’s README).

4. From the repository root, start Jupyter and open `notebooks/parkinson_multimodal_staff_demo.ipynb` (speech + handwriting + gait; demo media lives in `notebooks/sample/`).

5. Optional headless check (after installing `requirements-notebooks.txt`; needs network for OpenCV sample downloads):  
   `MPLBACKEND=Agg jupyter nbconvert --to notebook --execute notebooks/parkinson_multimodal_staff_demo.ipynb --output /tmp/parkinson_multimodal_staff_demo-executed.ipynb --ExecutePreprocessor.timeout=600`

Upload-time feature extraction and this notebook both use the shared **`common/`** package (speech, handwriting, gait), separate from **`webapp/`** and **`src/utils/`** (config).

## Architecture

### SE-ResNet 1D + Attention Fusion

Each modality is processed by its own SE-ResNet branch:
1. **1D Convolution** - Extracts local patterns from feature vectors
2. **Residual SE Blocks** - Skip connections + Squeeze-and-Excitation channel attention
3. **Attention Fusion** - Learned weights combine modality embeddings
4. **Dense Classifier** - Final prediction with dropout regularization

### Explainability

- **Grad-CAM**: Per-feature importance scores showing which inputs drive the prediction
- **Attention Weights**: How much each modality (speech, handwriting, gait) contributes
- **SE Channel Weights**: Internal channel attention within each modality branch

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/health` | GET | No | Health check and model status |
| `/api/predict` | POST | Yes | Single prediction |
| `/api/predict_batch` | POST | Yes | Batch predictions |
| `/api/model_info` | GET | Yes | Model information |
| `/api/auth/register` | POST | No | User registration |
| `/api/auth/login` | POST | No | User login |
| `/api/auth/logout` | POST | Yes | User logout |
| `/api/upload/audio` | POST | Yes | Upload audio for speech features |
| `/api/upload/handwriting` | POST | Yes | Upload image for handwriting features |
| `/api/upload/gait` | POST | Yes | Upload video for gait features |

## Configuration

Edit `config.yaml` to adjust:
- Deep learning hyperparameters (learning rate, epochs, architecture)
- Data split ratios
- Feature extraction parameters
- Server settings

Environment variables (`.env`):
- `MONGODB_URI` - MongoDB connection string
- `JWT_SECRET_KEY` - Secret key for JWT token signing

## Testing

Run lightweight checks (e.g. voice pipeline script URL policy):

```bash
python -m unittest tests.test_voice_pipeline
```

## Tech Stack

- **Backend**: Flask, Waitress, PyTorch
- **Frontend**: Jinja2, Bootstrap 5, Chart.js
- **Database**: MongoDB (user auth)
- **Auth**: JWT (PyJWT + bcrypt)
- **ML Fallback**: scikit-learn, XGBoost, LightGBM

---

**Disclaimer**: This system is for research and educational purposes only. It is not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.
