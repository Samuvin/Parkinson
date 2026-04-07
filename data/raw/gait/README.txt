Place gait_data.csv here for multimodal train_dl.py (ten gait columns + status).

Provenance and Kaggle CLI examples:
  data/raw/gait/GAIT_DATASETS_SOURCES.txt

Optional Kaggle tabular mirror (requires kaggle CLI + ~/.kaggle/kaggle.json):
  DOWNLOAD_KAGGLE_GAIT=1 bash data/raw/gait/download_gait_sources.sh

Project-shaped gait rows from PhysioNet gait trials (HTTPS inside script):
  python scripts/build_handwriting_gait_from_public_sources.py
