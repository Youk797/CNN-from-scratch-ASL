from __future__ import annotations

import os
import numpy as np

from ..Cnn.numpy_cupy import xp
from ..Cnn.model import CNN
from ..Cnn.losses import SoftmaxM
from ..Cnn.optimisation import Optimisation
from ..Cnn.utils import softmax

from .nettoyage import reset_quarantine_dir
from .variables import CLASS_NAMES, CLASS_DIRS, INPUT_SHAPE
from .entrainement_utils import build_dataset
from .evaluation import accuracy


def _stratified_train_val_split(X, y, val_split, seed):
    if not 0 < val_split < 1:
        raise ValueError("val_split doit être strictement compris entre 0 et 1.")

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_idx.extend(idx.tolist())
            continue
        n_val = max(1, int(round(len(idx) * val_split)))
        n_val = min(n_val, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    if not train_idx or not val_idx:
        raise RuntimeError("Impossible de créer un split train/validation non vide.")

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_model(
    data_root=None,
    validation_root=None,
    max_files_per_class=None,
    num_epochs=20,
    batch_size=32,
    lr=1e-3,
    val_split=0.2,
    seed=0,
    on_epoch_end=None,
):

    if data_root is None:
        raise ValueError("data_root est requis pour l'entraînement.")

    # Reproducible data order and weight initialization (NumPy or CuPy).
    rng = np.random.default_rng(seed)
    try:
        xp.random.seed(seed)
    except Exception:
        pass

    train_quarantine = os.path.join(str(data_root), "quarantaine")
    reset_quarantine_dir(train_quarantine)

    X, y = build_dataset(
        data_root=data_root,
        max_files_per_class=max_files_per_class,
        seed=seed,
        quarantine_root=train_quarantine,
    )

    use_external_validation = (
        validation_root is not None
        and os.path.isdir(str(validation_root))
        and all(os.path.isdir(os.path.join(str(validation_root), d)) for d in CLASS_DIRS)
    )

    if use_external_validation:
        validation_quarantine = os.path.join(str(validation_root), "quarantaine")
        reset_quarantine_dir(validation_quarantine)
        Xva, yva = build_dataset(
            data_root=validation_root,
            max_files_per_class=None,
            seed=seed,
            quarantine_root=validation_quarantine,
        )
        Xtr, ytr = X, y
    else:
        Xtr, ytr, Xva, yva = _stratified_train_val_split(X, y, val_split, seed)

    mean = Xtr.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = Xtr.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)

    model = CNN(num_classes=len(CLASS_NAMES), input_shape=INPUT_SHAPE)
    soft = SoftmaxM()
    opt = Optimisation(lr=lr)

    def _iter_minibatches(Xsplit, ysplit, shuffle=False):
        n = len(Xsplit)
        order = np.arange(n)
        if shuffle:
            rng.shuffle(order)
        for i in range(0, n, batch_size):
            sel = order[i : min(i + batch_size, n)]
            yield Xsplit[sel], ysplit[sel]

    def _eval_split(Xsplit, ysplit):
        losses, accs = [], []
        for xb, yb in _iter_minibatches(Xsplit, ysplit, shuffle=False):
            xb_n = (xb - mean) / std
            xb_b = xp.asarray(xb_n).astype(xp.float32)
            yb_b = xp.asarray(yb).astype(xp.int64)

            logits = model.forward(xb_b)
            loss = soft.forward(logits, yb_b)
            probs = softmax(logits, axis=1)
            acc = accuracy(probs, yb_b)

            losses.append(float(loss))
            accs.append(float(acc))
        return float(np.mean(losses)), float(np.mean(accs))

    for epoch in range(1, num_epochs + 1):
        batch_losses, batch_accs = [], []
        for xb, yb in _iter_minibatches(Xtr, ytr, shuffle=True):
            xb_n = (xb - mean) / std
            xb_b = xp.asarray(xb_n).astype(xp.float32)
            yb_b = xp.asarray(yb).astype(xp.int64)

            logits = model.forward(xb_b)
            loss = soft.forward(logits, yb_b)

            dlogits = soft.backward()
            model.backward(dlogits)
            opt.step(model.params_and_grads())

            probs = softmax(logits, axis=1)
            acc = accuracy(probs, yb_b)
            batch_losses.append(float(loss))
            batch_accs.append(float(acc))

        train_loss = float(np.mean(batch_losses))
        train_acc = float(np.mean(batch_accs))
        val_loss, val_acc = _eval_split(Xva, yva)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, num_epochs)

    return model, mean, std

