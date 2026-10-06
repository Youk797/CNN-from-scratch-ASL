

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.Entrainement.variables import CLASS_DIRS


def read_split_list(path: Path) -> set[str]:
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def place_file(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            print(f"[WARN] Symlink failed for {src}; falling back to copy.")
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Speech Commands v0.02 root directory")
    parser.add_argument("--target", type=Path, default=Path("data"))
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()

    source = args.source.resolve()
    validation_file = source / "validation_list.txt"
    testing_file = source / "testing_list.txt"
    if not validation_file.is_file() or not testing_file.is_file():
        raise SystemExit(
            "validation_list.txt/testing_list.txt not found. Use a Speech Commands "
            "v0.02 source directory containing the official split lists."
        )

    validation = read_split_list(validation_file)
    testing = read_split_list(testing_file)
    counts = {"train": 0, "validate": 0, "test": 0}

    for cls in CLASS_DIRS:
        class_dir = source / cls
        if not class_dir.is_dir():
            print(f"[WARN] Missing class directory: {class_dir}")
            continue

        for wav in class_dir.glob("*.wav"):
            rel = f"{cls}/{wav.name}"
            if rel in validation:
                split = "validate"
            elif rel in testing:
                split = "test"
            else:
                split = "train"

            dst = args.target / split / cls / wav.name
            place_file(wav, dst, args.mode)
            counts[split] += 1

    print("Prepared dataset:")
    for split, count in counts.items():
        print(f"  {split:<8} {count} WAV files")


if __name__ == "__main__":
    main()
