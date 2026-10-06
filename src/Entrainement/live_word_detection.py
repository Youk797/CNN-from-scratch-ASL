import threading
import time
import numpy as np

from .prediction import predict_on_signal
from .variables import SAMPLE_RATE


class LiveWordDetector:
    """
    Écoute le micro en direct et signale les mots reconnus par le CNN.

    Le CNN n'est pas modifié : on prend simplement le dernier morceau audio
    d'environ 1 seconde, on le transforme en features, puis on le classe.
    """

    def __init__(
        self,
        model,
        mean,
        std,
        on_detection,
        on_status=None,
        sample_rate=SAMPLE_RATE,
        window_seconds=1.0,
        hop_seconds=0.35,
        confidence_threshold=0.50,
        silence_rms_threshold=0.004,
        repeat_cooldown_seconds=1.0,
        device=None,
    ):
        self.model = model
        self.mean = mean
        self.std = std
        self.on_detection = on_detection
        self.on_status = on_status

        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.confidence_threshold = float(confidence_threshold)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.repeat_cooldown_seconds = float(repeat_cooldown_seconds)
        self.device = device

        self._stop_event = threading.Event()
        self._thread = None
        self.running = False

    def start(self):
        if self.running:
            return

        self._stop_event.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.running = False

    def _status(self, msg):
        if self.on_status is not None:
            self.on_status(msg)

    def _rms(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x)))

    def _run(self):
        try:
            import sounddevice as sd
        except ImportError:
            self.running = False
            self._status(
                "[LIVE] Module 'sounddevice' manquant. Installe-le avec : pip install sounddevice"
            )
            return

        window_samples = max(1, int(self.window_seconds * self.sample_rate))
        block_samples = max(1, int(self.hop_seconds * self.sample_rate))

        audio_buffer = np.zeros((0,), dtype=np.float32)
        last_word = None
        last_emit_time = 0.0

        self._status("[LIVE] Micro activé. Parle clairement, avec de petits silences entre les mots.")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_samples,
                device=self.device,
            ) as stream:
                while not self._stop_event.is_set():
                    block, overflowed = stream.read(block_samples)
                    if overflowed:
                        self._status("[LIVE] Attention : le flux micro a débordé, prédiction peut-être moins fiable.")

                    chunk = np.asarray(block, dtype=np.float32).reshape(-1)
                    audio_buffer = np.concatenate([audio_buffer, chunk])

                    # On garde seulement la dernière seconde environ.
                    if audio_buffer.size > window_samples:
                        audio_buffer = audio_buffer[-window_samples:]

                    # Tant qu'on n'a pas une fenêtre complète, on attend.
                    if audio_buffer.size < window_samples:
                        continue

                    segment = audio_buffer.copy()
                    rms = self._rms(segment)

                    # Évite de classer le silence et le bruit très faible.
                    if rms < self.silence_rms_threshold:
                        continue

                    try:
                        label, probs = predict_on_signal(segment, self.model, self.mean, self.std)
                    except Exception as e:
                        self._status(f"[LIVE] Erreur pendant la prédiction : {e}")
                        continue

                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])

                    if confidence < self.confidence_threshold:
                        continue

                    now = time.time()
                    same_word_too_soon = (
                        label == last_word
                        and (now - last_emit_time) < self.repeat_cooldown_seconds
                    )
                    if same_word_too_soon:
                        continue

                    last_word = label
                    last_emit_time = now

                    self.on_detection({
                        "word": label,
                        "confidence": confidence,
                        "rms": rms,
                        "time": now,
                    })

        except Exception as e:
            self._status(f"[LIVE] Impossible d'utiliser le micro : {e}")
        finally:
            self.running = False
            self._status("[LIVE] Micro arrêté.")
