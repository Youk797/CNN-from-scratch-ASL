import numpy as np

from ..Cnn.numpy_cupy import xp
from ..Cnn.utils import softmax
from .variables import CLASS_NAMES
from .entrainement_utils import wavtofeatures3ch, signaltofeatures3ch


def predict_from_features(feat, model, mean, std):
    # Prédit une classe à partir de features déjà calculées.
    # feat doit avoir la forme (3, F, T).

    x = feat[None, ...]
    x_n = (x - mean) / std
    xb = xp.asarray(x_n).astype(xp.float32)

    logits = model.forward(xb)
    probs_b = softmax(logits, axis=1)[0]
    probs = xp.asnumpy(probs_b) if xp.__name__ == "cupy" else probs_b

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    return pred_label, probs


def predict_on_signal(wav, model, mean, std):
    # Prédit un mot à partir d'un morceau de signal audio brut.

    feat = signaltofeatures3ch(wav, check_quality=False)
    return predict_from_features(feat, model, mean, std)


def predict_on_wav(wav_path, model, mean, std, quarantine_root=None):
    # Prédit un mot à partir d'un fichier wav contenant normalement un seul mot.

    feat = wavtofeatures3ch(wav_path, quarantine_root=quarantine_root)
    return predict_from_features(feat, model, mean, std)

