from __future__ import annotations

from pathlib import Path
import numpy as np

from ..Cnn.model import CNN


def save_model(path, model, mean, std):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sd = model.state_dict_numpy()
    input_shape = getattr(model, "input_shape", None)
    if input_shape is None:
        raise ValueError("model.input_shape manquant.")

    np.savez(
        path,
        **sd,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        num_classes=np.int64(model.num_classes),
        input_shape=np.asarray(input_shape, dtype=np.int64),
    )
    print(f"[SAVE] Modèle sauvegardé dans {path}")


def load_model(path, input_shape=None, num_classes=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "input_shape" in data:
            input_shape = tuple(int(x) for x in data["input_shape"].tolist())
        if "num_classes" in data:
            num_classes = int(data["num_classes"])

        if input_shape is None or num_classes is None:
            raise ValueError("input_shape/num_classes introuvables dans le snapshot.")

        model = CNN(num_classes=num_classes, input_shape=input_shape)
        sd = {
            "conv1_W": data["conv1_W"],
            "conv1_b": data["conv1_b"],
            "conv2_W": data["conv2_W"],
            "conv2_b": data["conv2_b"],
            "fc_W": data["fc_W"],
            "fc_b": data["fc_b"],
        }
        model.load_state_dict_numpy(sd)
        mean = np.array(data["mean"], dtype=np.float32, copy=True)
        std = np.array(data["std"], dtype=np.float32, copy=True)

    print(
        f"[LOAD] Modèle chargé depuis {path} "
        f"(num_classes={num_classes}, input_shape={input_shape})"
    )
    return model, mean, std
