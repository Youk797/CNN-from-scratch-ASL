from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path
    train_dir: Path
    validate_dir: Path
    test_dir: Path
    asl_videos_dir: Path


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def validate_config(config: dict) -> None:
    data = config.get("data", {})
    required = ["data_dir", "data_train_dir", "data_validate_dir", "data_test_dir"]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise RuntimeError(
            "Missing configuration key(s) in [data]: " + ", ".join(missing)
        )


def build_config(raw: dict, base_dir: Path) -> Config:
    data = raw["data"]
    return Config(
        data_dir=resolve_path(data["data_dir"], base_dir),
        train_dir=resolve_path(data["data_train_dir"], base_dir),
        validate_dir=resolve_path(data["data_validate_dir"], base_dir),
        test_dir=resolve_path(data["data_test_dir"], base_dir),
        asl_videos_dir=resolve_path(
            data.get("asl_videos_dir", "data/illustration"), base_dir
        ),
    )
