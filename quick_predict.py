
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.Entrainement.sauvegarde import load_model
from src.Entrainement.prediction import predict_on_wav
from src.Entrainement.variables import CLASS_NAMES


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = REPO_ROOT / "models" / "snapshot.npz"


def main():
    parser = argparse.ArgumentParser(
        description="Predict one of the 31 supported English keywords from a WAV file."
    )
    parser.add_argument("wav", type=Path, help="Path to a WAV file containing one spoken word")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to snapshot.npz")
    parser.add_argument("--top-k", type=int, default=5, help="Number of probabilities to display")
    args = parser.parse_args()

    if not args.wav.is_file():
        raise SystemExit(f"WAV file not found: {args.wav}")
    if not args.model.is_file():
        raise SystemExit(f"Model snapshot not found: {args.model}")

    model, mean, std = load_model(args.model)
    label, probs = predict_on_wav(str(args.wav), model, mean, std)
    probs = np.asarray(probs).reshape(-1)

    order = np.argsort(probs)[::-1][: max(1, args.top_k)]
    print(f"Prediction: {label}")
    for idx in order:
        print(f"  {CLASS_NAMES[int(idx)]:<10} {float(probs[idx]):.4f}")


if __name__ == "__main__":
    main()
