# Dataset layout and contracts

## Voice-first (current focus)

The simplest path is **speech / voice only**:

| What | Details |
|------|---------|
| **Primary file** | [`data/raw/speech/parkinsons.csv`](data/raw/speech/parkinsons.csv) — 22 voice features + `status` (same schema as [UCI Parkinsons (174)](https://archive.ics.uci.edu/dataset/174/parkinsons)) |
| **Training** | [`train.py`](train.py) loads **only** speech via [`DataLoader.load_speech_data`](src/data/data_loader.py) and trains sklearn models (LR + SVM). **You do not need handwriting or gait CSVs for this path.** |
| **Refresh speech CSVs** | `bash data/raw/speech/download_speech_csvs.sh` **or** `python scripts/train_voice_pipeline.py --download-speech` (same fixed HTTPS URLs; writes under `data/raw/speech/`) |
| **Staff multimodal notebook** | [`notebooks/parkinson_multimodal_staff_demo.ipynb`](notebooks/parkinson_multimodal_staff_demo.ipynb) uses aligned speech + handwriting + gait CSVs and optional raw demos under [`notebooks/sample/`](notebooks/sample/) (see [`notebooks/sample/README.md`](notebooks/sample/README.md)). Optional auto-downloads: [Figshare voice 23849127](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127), [Wikimedia handwriting](https://commons.wikimedia.org/wiki/File:Writing_by_a_Parkinson%27s_disease_patient.jpg), OpenCV video; details in [`notebooks/samples/README.md`](notebooks/samples/README.md); real gait MC is linked from there ([Springer Nature Figshare 29371769](https://springernature.figshare.com/articles/dataset/Dataset_on_gait_analysis_of_parkinsonian_subjects_effect_of_Nordic_walking/29371769)). |
| **Example WAVs (demo only)** | [`data/examples/speech/`](data/examples/speech/) — bundled **synthetic** `healthy_example.wav` / `parkinson_example.wav` for Jupyter; regenerate via `python scripts/generate_example_speech_wavs.py`. Not used for bulk training. |

**Shared feature extraction:** Raw uploads and notebooks use [`common/`](common/) (speech via librosa/Praat, handwriting images, gait video). That layer is separate from tabular CSV training in [`train.py`](train.py).

Optional: [`data/raw/speech/parkinsons_telemonitoring.csv`](data/raw/speech/parkinsons_telemonitoring.csv) for telemonitoring experiments in [`DataLoader`](src/data/data_loader.py).

---

## Multimodal (optional — `train_dl.py`)

[`train_dl.py`](train_dl.py) / [`load_all_modalities`](dl_models/dataset.py) expect **three** tabular CSVs under [`data/raw/`](data/raw/). **Column names and relative CSV paths** are defined in [`config/multimodal_features.yaml`](config/multimodal_features.yaml) (see `data.multimodal_features_config` in [`config.yaml`](config.yaml)). The notebook [`notebooks/parkinson_multimodal_staff_demo.ipynb`](notebooks/parkinson_multimodal_staff_demo.ipynb) uses the same YAML. Handwriting and gait can be ignored until you are ready for this path.

## File layout

| Path | Purpose |
|------|---------|
| `data/raw/speech/parkinsons.csv` | Voice biomarkers (22 features + `status`) — sourced from [SagarBapodara Kaggle notebook / `parkinson-csv`](https://www.kaggle.com/code/sagarbapodara/parkinson-detection-testing-predictive-system) mirror ([GitHub raw CSV](https://raw.githubusercontent.com/SagarBapodara/Parkison-Disease-Detection-using-Machine-Learning/main/Data/parkinsons.csv)); same schema as [UCI Parkinsons (174)](https://archive.ics.uci.edu/dataset/174/parkinsons). |
| `data/raw/speech/parkinsons_telemonitoring.csv` | Optional: UCI telemonitoring (used by [`src/data/data_loader.py`](src/data/data_loader.py), not by `train_dl.py`) |
| `data/raw/handwriting/handwriting_data.csv` | **Multimodal only:** ten handwriting kinematic features + `status` |
| `data/raw/gait/gait_data.csv` | **Multimodal only:** ten gait summary features + `status` |

## Column contract (must match exactly)

**Speech** — 22 feature columns (order in CSV does not matter; loaders select by name):

- `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`, `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`, `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`, `NHR`, `HNR`, `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`
- Label: `status` (0 = healthy, 1 = Parkinson’s)
- Optional id: `name`

**Handwriting** — [`HANDWRITING_FEATURE_NAMES`](dl_models/dataset.py):

- `mean_pressure`, `pressure_variation`, `mean_velocity`, `velocity_variation`, `mean_acceleration`, `penup_time_ratio`, `mean_stroke_length`, `writing_tempo`, `tremor_power`, `fluency_score`
- Label: `status` (ignored by `train_dl.py`; see below)

**Gait** — [`GAIT_FEATURE_NAMES`](dl_models/dataset.py):

- `stride_interval`, `stride_variability`, `swing_time`, `stance_time`, `double_support_time`, `gait_speed`, `cadence`, `step_length`, `stride_regularity`, `gait_asymmetry`
- Label: `status` (ignored by `train_dl.py`; see below)

## Row alignment caveat (important)

[`load_all_modalities`](dl_models/dataset.py) **truncates** all three modalities to the **minimum row count** and uses **labels from the speech file only**. Handwriting and gait label columns are **not** used for training in that path.

So row *i* in each CSV is treated as one training example across modalities. **Independent** public datasets (different subjects per row) should not be concatenated naively if the goal is subject-level multimodal learning—use matched cohorts or change the loading code.

## Authoritative public sources (UCI, PhysioNet)

These are the usual **primary** citations for academic work. Several are **mirrored or republished** on Kaggle (same or derived content; always check the license on the Kaggle page).

| Modality | Recommended source | Access |
|----------|-------------------|--------|
| Speech | [UCI Parkinsons (ID 174)](https://archive.ics.uci.edu/dataset/174/parkinsons) — `parkinsons.data` | Public |
| Speech (longitudinal) | [UCI Parkinson Telemonitoring (ID 214)](https://archive.ics.uci.edu/dataset/214/parkinsons+telemonitoring) | Public |
| Speech (extended recordings) | [UCI Parkinson speech, multiple sound types (ID 301)](https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings) | Public (26 features per clip; **not** the same 22 columns as ID 174 — requires mapping or model changes) |
| Gait (tabular / signals) | [PhysioNet Gait in Parkinson’s Disease (gaitpdb)](https://physionet.org/content/gaitpdb/1.0.0/) | Public index of per-trial `.txt` signals; **bulk zip** may require [PhysioNet credentialing](https://physionet.org/about/credentialing/) |
| Handwriting (images / traces) | [UCI Parkinson spiral / tablet (ID 395)](https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet) | Semicolon-separated **pen traces** (and readme); use [`download_handwriting_sources.sh`](data/raw/handwriting/download_handwriting_sources.sh) + [`scripts/uci_spiral_traces_to_csv.py`](scripts/uci_spiral_traces_to_csv.py) — not the ten `HANDWRITING_FEATURE_NAMES` without extra mapping |

## Handwriting and gait: can you get CSV from Kaggle or UCI?

**Yes, you can get Parkinson-related handwriting and gait data online** (UCI, Kaggle, PhysioNet, Zenodo, etc.). **What you usually do *not* get is a single CSV that already uses this project’s exact ten handwriting and ten gait column names**—those names are a fixed contract in [`dl_models/dataset.py`](dl_models/dataset.py). Speech is the straightforward case (UCI 174 and many Kaggle mirrors match the 22 voice columns).

You typically do one of the following:

1. **Download a public dataset**, then **rename/map columns** or **compute features** until the CSV matches [`HANDWRITING_FEATURE_NAMES`](dl_models/dataset.py) and [`GAIT_FEATURE_NAMES`](dl_models/dataset.py) (and add `status` plus optional `sample_id`).
2. **Regenerate the bundled CSVs** from public sources using [`scripts/build_handwriting_gait_from_public_sources.py`](scripts/build_handwriting_gait_from_public_sources.py) (PaHaW-derived task features on GitHub + PhysioNet gaitpdb walking trials).

### Handwriting (UCI / Kaggle)

| Source | Typical format | Ready-made match for `handwriting_data.csv`? |
|--------|----------------|---------------------------------------------|
| **UCI** | Spiral / tablet Parkinson datasets are usually **drawings, images, or high-dimensional trace exports** | Rarely — expect **feature extraction** or column mapping |
| **Kaggle** | e.g. [Parkinson disease spiral drawings](https://www.kaggle.com/datasets/team-ai/parkinson-disease-spiral-drawings) — often **images**; notebooks may build their own tables | Only if you (or a notebook) **emit** the ten project columns |
| **PaHaW** (research) | Gold-standard handwriting DB; access often **by application** | Community mirrors (e.g. GitHub) may ship **wide** feature CSVs — still need **aggregation/mapping** to this repo’s ten columns unless you change the model |

### Gait (UCI / Kaggle / PhysioNet)

| Source | Typical format | Ready-made match for `gait_data.csv`? |
|--------|----------------|--------------------------------------|
| **PhysioNet gaitpdb** | **Force-plate time series** per trial (`.txt`), not one row per subject with ten summary stats | No — use **ETL** (see script above) or your own aggregation |
| **Kaggle** | e.g. [Gait in Parkinson’s Disease](https://www.kaggle.com/datasets/zarif98sjs/gait-in-parkinsons-disease) — may be **tabular** depending on version | **Maybe** — open the CSV, check columns, then **rename or map** to [`GAIT_FEATURE_NAMES`](dl_models/dataset.py) |
| **UCI** | There is **no** standard UCI entry that is universally used as “the” Parkinson gait CSV with these ten fields | Expect **external papers / supplements** or **PhysioNet** instead |

### Example: Kaggle CLI (after configuring `~/.kaggle/kaggle.json`)

```bash
mkdir -p data/raw/gait/kaggle_mirror
kaggle datasets download -d zarif98sjs/gait-in-parkinsons-disease -p data/raw/gait/kaggle_mirror --unzip
# Inspect the extracted CSV(s), map columns, then save as data/raw/gait/gait_data.csv or extend dataset.py
```

Always check the **dataset license** on the Kaggle or UCI page before redistributing or committing files.

## Handwriting traces (UCI 395) and gait mirrors

| Step | Command | Notes |
|------|---------|--------|
| **Handwriting — fetch UCI spiral zip** | `bash data/raw/handwriting/download_handwriting_sources.sh` | Fixed URL → [`data/raw/handwriting/uci_spiral/`](data/raw/handwriting/uci_spiral/) (see [`HANDWRITING_DATASETS_SOURCES.txt`](data/raw/handwriting/HANDWRITING_DATASETS_SOURCES.txt)). |
| **Handwriting — one CSV summary** | `python scripts/uci_spiral_traces_to_csv.py` | Optional `--approx-project-cols` adds rough analogues of [`HANDWRITING_FEATURE_NAMES`](dl_models/dataset.py). |
| **Handwriting — optional Kaggle images** | `DOWNLOAD_KAGGLE_HANDWRITING=1 bash data/raw/handwriting/download_handwriting_sources.sh` | Requires [Kaggle API credentials](https://github.com/Kaggle/kaggle-api). |
| **Gait — optional Kaggle tabular** | `DOWNLOAD_KAGGLE_GAIT=1 bash data/raw/gait/download_gait_sources.sh` | Inspect CSV schema against [`GAIT_FEATURE_NAMES`](dl_models/dataset.py); see [`GAIT_DATASETS_SOURCES.txt`](data/raw/gait/GAIT_DATASETS_SOURCES.txt). |
| **Handwriting + gait — `train_dl` contract** | `python scripts/build_handwriting_gait_from_public_sources.py` | Emits [`handwriting_data.csv`](data/raw/handwriting/handwriting_data.csv) + [`gait_data.csv`](data/raw/gait/gait_data.csv) (PaHaW GitHub + PhysioNet gaitpdb). |

## Extra speech CSVs (`data/raw/speech/`)

Besides **`parkinsons.csv`** and **`parkinsons_telemonitoring.csv`**, the speech folder may contain additional **plain CSV** mirrors for research and schema comparison. Only **`parkinsons.csv`** matches the 22-feature + `status` contract used by [`train_dl.py`](train_dl.py); the others need mapping or different models.

| File | Content | Drop-in for `train_dl`? |
|------|---------|-------------------------|
| `parkinsons.csv` | Oxford / UCI 174 voice biomarkers | **Yes** (verify vs [`SPEECH_FEATURE_NAMES`](dl_models/dataset.py)) |
| `parkinsons_telemonitoring.csv` | Telemonitoring voice + UPDRS targets | No (different task; optional for `DataLoader`) |
| `voice_uci489_replicated_acoustic.csv` | UCI 489 replicated acoustic features | No (non-i.i.d. rows) |
| `voice_uci301_multiple_sound_train.csv` | UCI 301 training table (26 features / clip) | No |
| `voice_uci470_pd_classification_features.csv` | UCI 470 high-dimensional speech features | No (two header rows; many columns) |

**Provenance and licenses:** [`data/raw/speech/SPEECH_DATASETS_SOURCES.txt`](data/raw/speech/SPEECH_DATASETS_SOURCES.txt).

**Refresh:**

```bash
bash data/raw/speech/download_speech_csvs.sh
```

## Kaggle and other community hubs

Community mirrors can be convenient, but **terms, versions, and preprocessing differ**. Prefer **pinning a dataset revision** and recording the hash. Anything below must still be converted into this repo’s **fixed 22 + 10 + 10 feature names** (or you change [`dl_models/dataset.py`](dl_models/dataset.py) and the network input sizes).

| Source | Examples | Fit for *this* codebase |
|--------|----------|-------------------------|
| **Kaggle — speech / tabular** | [UCI ML Parkinson’s (mirror)](https://www.kaggle.com/datasets/elnazalikarami/uci-ml-parkinsons-dataset) — search Kaggle for “Parkinson UCI” for other mirrors | **Good** if the mirror preserves the **same 22 biomarkers + label** as UCI 174. Verify column names against [`SPEECH_FEATURE_NAMES`](dl_models/dataset.py). |
| **Kaggle — handwriting / drawings** | [Parkinson disease spiral drawings](https://www.kaggle.com/datasets/team-ai/parkinson-disease-spiral-drawings), related “spiral” / “drawing” PD datasets | Usually **images or trajectories**. Useful after you implement feature extraction into the **10 handwriting** columns, or you replace those column definitions. |
| **Kaggle — biomarkers / mixed** | e.g. “Early biomarkers of Parkinson’s disease” (search on Kaggle) | **Variable schema** — inspect notebooks; often needs **manual column mapping** and label harmonization. |
| **Kaggle — gait competitions** | e.g. [Parkinson’s freezing of gait prediction](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-of-gait-prediction) | **Wearable sensor time series** and FoG labels — strong for gait-ML research but **not** drop-in for the current **10 summary gait** fields; needs aggregation/ETL. |
| **OpenML** | Search [openml.org](https://www.openml.org) for “Parkinson” (includes UCI 174 exports) | Same caveat as mirrors: confirm **exact feature list** matches this project. |
| **Synapse / Sage Bionetworks** | [mPower](https://www.synapse.org/mpower) (mobile voice + tapping) | **Large, real-world** cohort; **different modalities** than handwriting+gait here — major integration effort. |
| **Zenodo / Figshare / Dryad** | Search for “Parkinson gait”, “Parkinson handwriting features” | Often **supplementary data for papers**; licenses vary; almost always need a **small ETL script** to your CSV contract. |

**Practical rule:** A dataset is “useful” only after you can answer: (1) Do the columns match or can I derive them? (2) Are rows **aligned** across modalities if I use multimodal training? (3) What license or competition rules apply (Kaggle competitions often restrict redistribution)?

## Refreshing speech CSV (`parkinsons.csv`)

**Same file as the SagarBapodara Kaggle notebook** (via their GitHub `Data` folder — no Kaggle API key):

```bash
mkdir -p data/raw/speech
curl -fsSL "https://raw.githubusercontent.com/SagarBapodara/Parkison-Disease-Detection-using-Machine-Learning/main/Data/parkinsons.csv" \
  -o data/raw/speech/parkinsons.csv
```

**Official UCI mirror** (usually byte-equivalent schema):

```bash
curl -fsSL "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data" \
  -o data/raw/speech/parkinsons.csv
```

**Via Kaggle CLI** (requires `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d sagarbapodara/parkinson-csv --unzip -p data/raw/speech/
# Then rename/move the CSV to parkinsons.csv if the archive uses another name.
```

**Telemonitoring** (optional, for `DataLoader` — not `train_dl` by default):

```bash
curl -fsSL "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data" \
  -o data/raw/speech/parkinsons_telemonitoring.csv
```

(Older docs sometimes used a `.../parkinsons/telemonitoring/...` path without the extra `parkinsons/` segment; the URL above matches the file listing on [UCI dataset 174](https://archive.ics.uci.edu/dataset/174/parkinsons).)

**Gait from PhysioNet:** after you obtain a local copy of [gaitpdb](https://physionet.org/content/gaitpdb/1.0.0/), use your own ETL to emit `data/raw/gait/gait_data.csv` with [`GAIT_FEATURE_NAMES`](dl_models/dataset.py), or run [`scripts/build_handwriting_gait_from_public_sources.py`](scripts/build_handwriting_gait_from_public_sources.py) (pulls individual trial files over HTTPS).

**Handwriting:** [UCI spiral / graphics tablet (ID 395)](https://archive.ics.uci.edu/dataset/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet) and Kaggle spiral mirrors typically require **feature engineering** before they match `HANDWRITING_FEATURE_NAMES` (see [`scripts/uci_spiral_traces_to_csv.py`](scripts/uci_spiral_traces_to_csv.py) for a trace-level summary CSV).

## Reproducibility

Record the download date and file checksum when refreshing data. Do not commit secrets or credentialed PhysioNet archives; keep raw gaitpdb outside the repo or under `.gitignore` if large.
