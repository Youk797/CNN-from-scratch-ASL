import os
import shutil
import numpy as np


def audio_quality_metrics(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return {
            "peak": 0.0,
            "rms": 0.0,
            "near_zero": 1.0,
            "clip_frac": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    near_zero = float(np.mean(np.abs(x) < 1e-4))
    clip_frac = float(np.mean(np.abs(x) > 0.99))
    mean = float(np.mean(x))
    std = float(np.std(x))
    return {
        "peak": peak,
        "rms": rms,
        "near_zero": near_zero,
        "clip_frac": clip_frac,
        "mean": mean,
        "std": std,
    }


def is_bad_audio(
    x,
    min_rms=1e-4,
    min_peak=1e-3,
    max_near_zero=0.98,
    max_clip_frac=0.02,
    max_dc_over_std=0.5,
):
    m = audio_quality_metrics(x)
    reasons = []

    if m["rms"] < min_rms or m["peak"] < min_peak:
        reasons.append("silence_or_too_low")
    if m["near_zero"] > max_near_zero:
        reasons.append("mostly_zero")
    if m["clip_frac"] > max_clip_frac:
        reasons.append("clipping")
    if m["std"] > 0 and abs(m["mean"]) / m["std"] > max_dc_over_std:
        reasons.append("dc_offset")

    return (len(reasons) > 0), reasons, m


def quarantine_file(src_path, quarantine_root, class_name):
    """Copy an unusable audio file into quarantine without deleting the source.

    Earlier versions moved the original file. That made a later reset of the
    quarantine directory destructive. Copying is safer for source datasets.
    """
    dst_dir = os.path.join(quarantine_root, class_name)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)
    return dst_path


def reset_quarantine_dir(quarantine_root):
    if os.path.isdir(quarantine_root):
        shutil.rmtree(quarantine_root, ignore_errors=True)
    os.makedirs(quarantine_root, exist_ok=True)


def remove_quarantine_dir(quarantine_root):
    if os.path.isdir(quarantine_root):
        shutil.rmtree(quarantine_root, ignore_errors=True)
