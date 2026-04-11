"""
Parkinson's Disease multimodal feature extraction pipelines.

Each pipeline reads from ``data/raw/<modality>/`` and writes a clean CSV
to ``data/processed/<modality>/``.

Usage (standalone):
    python -m pipelines.speech_pipeline
    python -m pipelines.gait_pipeline
    python -m pipelines.handwriting_pipeline

Or run all at once:
    python -m pipelines
"""
from pathlib import Path
from .speech_pipeline import run as run_speech
from .gait_pipeline import run as run_gait
from .handwriting_pipeline import run as run_handwriting

__all__ = ["run_speech", "run_gait", "run_handwriting", "run_all"]


def run_all(raw_dir: str | Path = "data/raw",
            processed_dir: str | Path = "data/processed",
            force: bool = False) -> None:
    """
    Run all three feature-extraction pipelines.

    Parameters
    ----------
    raw_dir : path to raw dataset root (e.g. ``data/raw``)
    processed_dir : path where processed CSVs will be saved
    force : re-run even if processed files already exist
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    targets = {
        "speech":      (processed_dir / "speech"      / "parkinsons.csv",     run_speech),
        "gait":        (processed_dir / "gait"         / "gait_data.csv",      run_gait),
        "handwriting": (processed_dir / "handwriting"  / "handwriting_data.csv", run_handwriting),
    }

    for modality, (target_path, pipeline_fn) in targets.items():
        if not force and target_path.is_file():
            print(f"[{modality.capitalize()} Pipeline] Already processed → {target_path}")
        else:
            pipeline_fn(raw_dir, processed_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run all feature-extraction pipelines")
    p.add_argument("--raw-dir",       default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if processed files already exist")
    args = p.parse_args()
    run_all(args.raw_dir, args.processed_dir, force=args.force)
