import unittest
from pathlib import Path
import numpy as np

from src.Cnn.model import CNN
from src.Cnn.utils import softmax
from src.Entrainement.entrainement_utils import signaltofeatures3ch
from src.Entrainement.nettoyage import is_bad_audio
from src.Entrainement.sauvegarde import load_model
from src.Entrainement.variables import INPUT_SHAPE, SAMPLE_RATE, CLASS_NAMES


REPO_ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]], dtype=np.float32)
        probs = np.asarray(softmax(logits, axis=1))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-6)

    def test_feature_shape(self):
        t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        wav = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        feat = signaltofeatures3ch(wav)
        self.assertEqual(feat.shape, INPUT_SHAPE)
        self.assertTrue(np.isfinite(feat).all())

    def test_silence_is_rejected_by_quality_check(self):
        bad, reasons, _ = is_bad_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.assertTrue(bad)
        self.assertIn("silence_or_too_low", reasons)

    def test_cnn_output_shape(self):
        model = CNN(num_classes=len(CLASS_NAMES), input_shape=INPUT_SHAPE)
        x = np.zeros((2,) + INPUT_SHAPE, dtype=np.float32)
        logits = np.asarray(model.forward(x))
        self.assertEqual(logits.shape, (2, len(CLASS_NAMES)))

    def test_demo_snapshot_loads(self):
        model_path = REPO_ROOT / "models" / "snapshot.npz"
        model, mean, std = load_model(model_path)
        self.assertEqual(model.num_classes, len(CLASS_NAMES))
        self.assertEqual(tuple(model.input_shape), INPUT_SHAPE)
        self.assertEqual(mean.shape, (1, 3, 1, 1))
        self.assertEqual(std.shape, (1, 3, 1, 1))


if __name__ == "__main__":
    unittest.main()
