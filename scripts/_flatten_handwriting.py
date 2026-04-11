"""
Flatten data/raw/handwriting/ so all files sit directly in that folder.

Moves:
  PaHaW_extracted_files/1_1.csv .. 8_1.csv  → handwriting/
  PaHaW_extracted_files/000XX/000XX__N_1.csv → handwriting/
  PaHaW_extracted_files/000XX/feature_name_list.csv → handwriting/ (first copy only)
  spiral_images/train_set.npz               → handwriting/
  spiral_images/test_set.npz                → handwriting/

Then removes the now-empty subdirectory trees.

Run from any directory:
    python3 scripts/_flatten_handwriting.py
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "handwriting"

moved = 0
skipped = 0
feature_name_copied = False


def move(src: Path, dst_name: str = None) -> None:
    global moved, skipped, feature_name_copied
    dst = ROOT / (dst_name or src.name)
    if dst.exists():
        skipped += 1
        return
    shutil.move(str(src), str(dst))
    moved += 1


# --- PaHaW root-level files (1_1.csv .. 8_1.csv) ---
pahaw_root = ROOT / "PaHaW_extracted_files"
if pahaw_root.is_dir():
    for f in pahaw_root.iterdir():
        if f.is_file():
            move(f)

    # --- subject subfolders ---
    for subject_dir in sorted(pahaw_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for f in subject_dir.iterdir():
            if not f.is_file():
                continue
            if f.name == "feature_name_list.csv":
                if not feature_name_copied and not (ROOT / "feature_name_list.csv").exists():
                    move(f)
                    feature_name_copied = True
                else:
                    f.unlink()   # duplicate — delete
            else:
                move(f)
        try:
            subject_dir.rmdir()
        except OSError:
            pass  # not empty yet

    try:
        pahaw_root.rmdir()
    except OSError:
        pass

# --- spiral_images ---
spiral_dir = ROOT / "spiral_images"
if spiral_dir.is_dir():
    for f in spiral_dir.iterdir():
        if f.is_file():
            move(f)
    try:
        spiral_dir.rmdir()
    except OSError:
        pass

# --- summary ---
all_files = list(ROOT.iterdir())
csvs = [x for x in all_files if x.suffix == ".csv"]
npzs = [x for x in all_files if x.suffix == ".npz"]
dirs = [x for x in all_files if x.is_dir()]

print(f"Moved  : {moved} files")
print(f"Skipped: {skipped} (already existed)")
print(f"handwriting/ now contains:")
print(f"  {len(csvs)} CSV files")
print(f"  {len(npzs)} NPZ files")
if dirs:
    print(f"  Remaining subdirs: {[d.name for d in dirs]}")
else:
    print(f"  No subdirectories remaining")
