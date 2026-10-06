import tkinter as tk
from pathlib import Path

from .app import App
from src.Config.config import load_config, validate_config, build_config


def main():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config.toml"

    raw = load_config(config_path)
    validate_config(raw)
    data_config = build_config(raw, repo_root)

    root = tk.Tk()
    App(root, data_config)
    root.mainloop()


if __name__ == "__main__":
    main()
