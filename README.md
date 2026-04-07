# Parkinson’s disease detection (multimodal DL + web app)

This repo trains a **multimodal** model on tabular speech, handwriting, and gait features and serves a **Flask** UI plus JSON APIs. Feature extraction from raw media lives under [`common/`](common/).

---

## How the web app runs when you open a URL

1. **Process start** — You run `python wsgi.py` (or Waitress/`start.sh`/`start.bat`). [`wsgi.py`](wsgi.py) loads [`.env`](.env.example) from the project root, then imports `app = create_app()` from [`webapp/app.py`](webapp/app.py).

2. **`create_app()`** — Builds the Flask app, reads config via [`src/utils/config.py`](src/utils/config.py), registers blueprints under `/api` (detect, auth, upload, combined processing, results), and runs [`get_manager()`](webapp/api/detect.py) so legacy sklearn models load at startup if present. It attaches [`enforce_auth`](webapp/middleware/auth.py) as a **`before_request`** hook.

3. **Browser navigation (pages)** — For paths like `/`, `/login`, `/detect`, `/model-report`, `/about`, `/results`, the request path does **not** start with `/api/`. [`enforce_auth`](webapp/middleware/auth.py) returns immediately for those routes, so no JWT is required. The matching view in [`webapp/app.py`](webapp/app.py) runs (for example `index()` → `render_template('index.html')`, `detect_page()` → `detect.html`).

4. **Static assets** — Normal Flask `static` handling; auth middleware skips enforcement for the static endpoint.

---

## What runs when the user provides input (API / detection)

Typical flow: the front end (or a client) calls **`POST /api/detect`** with JSON after the user signs in and obtains a token.

1. **`before_request`** — Because the path is `/api/detect`, [`enforce_auth`](webapp/middleware/auth.py) runs. It allows only the documented public API routes without a token (for example `GET /api/health`, `GET /api/training_report`, `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`). **`POST /api/detect` requires a valid `Authorization: Bearer <jwt>`** header.

2. **Detection handler** — [`run_detection`](webapp/api/detect.py) in [`webapp/api/detect.py`](webapp/api/detect.py) runs:
   - Parses JSON from the body.
   - Validates optional `speech_features` (22 floats), `handwriting_features` (10), `gait_features` (10); at least one modality must be present.
   - Calls **`_build_detect_json_payload(...)`** (same module) to assemble the response (including presentation fields aligned with the multimodal network).
   - If the user is authenticated, tries to persist via [`save_detection`](webapp/models/detection_result.py).
   - Returns JSON to the client.

Other API routes (upload, batch detect, combined processing, etc.) are registered in [`webapp/app.py`](webapp/app.py) from [`webapp/api/`](webapp/api/); they follow the same **`/api/*` + JWT** rule unless listed as public in the middleware.

---

## Quick setup

```bash
cd Parkinson   # your clone path
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # set MONGODB_URI, JWT_SECRET_KEY, etc.
python wsgi.py             # http://0.0.0.0:8000 by default (PORT env overrides)
```

---

## Training and data layout

- **Train multimodal DL:** [`train_dl.py`](train_dl.py) — CSV paths and feature column names come from [`config/multimodal_features.yaml`](config/multimodal_features.yaml) (referenced by [`config/project.yaml`](config/project.yaml)).
- **Aligned CSVs:** under `data/raw/speech/`, `data/raw/handwriting/`, `data/raw/gait/`. Loaders truncate to the **minimum row count** across modalities and use **speech** for labels when training that path; see [`dl_models/data/dataset.py`](dl_models/data/dataset.py).
- **Refresh speech CSVs:** `bash data/raw/speech/download_speech_csvs.sh` or `python scripts/train_voice_pipeline.py --download-speech`.
- **Jupyter staff demo:** [`notebooks/parkinson_multimodal_staff_demo.ipynb`](notebooks/parkinson_multimodal_staff_demo.ipynb) — see [`notebooks/README.md`](notebooks/README.md).

### Config → data → train → view model scores

1. **Features (config)** — Speech, handwriting, and gait **column names** and **CSV paths** (relative to `data.raw_dir`) are defined in [`config/multimodal_features.yaml`](config/multimodal_features.yaml), referenced by `data.multimodal_features_config` in [`config/project.yaml`](config/project.yaml).
2. **Data** — Put aligned tabular CSVs under `data/raw/` as in that YAML, **or** derive matching vectors from raw WAV / images / video using [`common/`](common/) so the numbers line up with those columns.
3. **Train** — Run [`train_dl.py`](train_dl.py). It writes `models/multimodal_pdnet.pt`, `models/dl_scalers.joblib`, `models/dl_model_metrics.json`, and plots (`dl_roc_curve.png`, etc.).
4. **View scores in the browser** — With the app running, open **`/model-report`** (nav: **Model scores**). Same data as the public JSON **`GET /api/training_report`** (no JWT). Plots load from **`/model_images/...`**.

---

## Layout (short)

| Area | Role |
|------|------|
| [`webapp/`](webapp/) | Flask app, templates, API blueprints |
| [`dl_models/`](dl_models/) | Networks, dataset, training, `DLDetector` inference |
| [`common/`](common/) | Raw speech / handwriting / gait feature extraction |
| [`src/`](src/) | Config, optional sklearn joblib loaders, utilities |
| [`models/`](models/) | Saved checkpoints (`.pt`, optional `.joblib`) |
| [`data/raw/`](data/raw/) | Training CSVs |
