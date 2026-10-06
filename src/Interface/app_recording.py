
import os
import tempfile
import threading
import wave
from tkinter import messagebox

import numpy as np

from ..Entrainement.prediction import predict_on_signal
from ..Entrainement.phrase_prediction import detect_words_in_wav_segmented, format_phrase_result
from ..Entrainement.variables import SAMPLE_RATE


class Recording:
    def _set_record_status(self, text):
        # Met à jour le label d'enregistrement depuis n'importe quel thread.
        self.master.after(0, lambda: self.label_record.config(text=text))

    def _check_recording_ready(self):
        # Vérifie le modèle et évite deux enregistrements simultanés.
        if not self._model_ready():
            messagebox.showwarning(
                "Modèle manquant",
                "Charge ou entraîne un modèle avant d'enregistrer au micro."
            )
            return False

        if self.recording_micro:
            self.log("[REC] Un enregistrement micro est déjà en cours.")
            return False

        # Le live utilise déjà le micro. On l'arrête pour éviter un conflit avec sounddevice.
        if self.live_detector is not None and self.live_detector.running:
            self.live_detector.stop()
            self.log("[REC] Live arrêté automatiquement pour libérer le micro.")

        return True

    def on_record_word_clicked(self):
        # Enregistre un seul mot puis le classe.
        if not self._check_recording_ready():
            return

        self.recording_micro = True
        t = threading.Thread(target=self.record_word_thread, daemon=True)
        t.start()

    def on_record_phrase_clicked(self):
        # Enregistre une phrase courte puis lance la segmentation audio.
        if not self._check_recording_ready():
            return

        self.recording_micro = True
        t = threading.Thread(target=self.record_phrase_thread, daemon=True)
        t.start()

    def _record_micro_seconds(self, seconds):
        """
        Enregistre le micro pendant une durée fixe et renvoie un signal float32 mono.
        Dépendance nécessaire : pip install sounddevice
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("Module 'sounddevice' manquant. Installe-le avec : pip install sounddevice")

        seconds = float(seconds)
        n_samples = max(1, int(seconds * SAMPLE_RATE))

        self._set_record_status(f"Enregistrement : parle maintenant ({seconds:.1f} s)...")
        self.log(f"[REC] Enregistrement micro pendant {seconds:.1f} s...")
        self.master.after(0, self.master.bell)

        audio = sd.rec(
            n_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size > 0:
            wav = wav - float(np.mean(wav))  # réduit un éventuel offset DC du micro
            wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        return wav

    def _write_temp_wav(self, wav):
        # Écrit un signal float32 [-1, 1] dans un WAV temporaire mono 16-bit.
        fd, path = tempfile.mkstemp(prefix="cnn_micro_", suffix=".wav")
        os.close(fd)

        wav = np.asarray(wav, dtype=np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        wav_i16 = (wav * 32767.0).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(SAMPLE_RATE))
            wf.writeframes(wav_i16.tobytes())

        return path

    def record_word_thread(self):
        tmp_path = None
        try:
            wav = self._record_micro_seconds(1.6)
            tmp_path = self._write_temp_wav(wav)
            self._set_record_status("Enregistrement : analyse du mot...")

            detections, phrase = detect_words_in_wav_segmented(
                tmp_path,
                self.model,
                self.mean,
                self.std,
                confidence_threshold=0.30,
                silence_rms_threshold=0.003,
                relative_threshold=0.14,
                low_relative_threshold=0.07,
                min_speech_seconds=0.08,
                min_silence_seconds=0.12,
                padding_seconds=0.08,
                max_segment_seconds=1.20,
                classification_window_seconds=1.0,
            )

            if detections:
                best = max(detections, key=lambda d: d["confidence"])
                word = str(best["word"]).upper()
                confidence = float(best["confidence"])
                result = f"Mot reconnu : {word}\nConfiance : {confidence:.3f}"
                self.log(f"[REC-MOT] {word} | confiance={confidence:.3f}")
                self.master.after(0, lambda w=word: self.video_player.play_word(w))
                self.master.after(0, lambda r=result: messagebox.showinfo("Mot reconnu", r))
            else:
                # Secours : si la segmentation ne trouve rien, on classe directement tout l'enregistrement.
                label, probs = predict_on_signal(wav, self.model, self.mean, self.std)
                probs = np.asarray(probs).reshape(-1)
                pred_idx = int(np.argmax(probs))
                word = str(label).upper()
                confidence = float(probs[pred_idx])
                result = (
                    "Aucun segment net détecté, donc classification directe de l'enregistrement.\n\n"
                    f"Mot proposé : {word}\nConfiance : {confidence:.3f}"
                )
                self.log(f"[REC-MOT] Fallback direct : {word} | confiance={confidence:.3f}")
                self.master.after(0, lambda w=word: self.video_player.play_word(w))
                self.master.after(0, lambda r=result: messagebox.showinfo("Mot reconnu", r))

        except Exception as e:
            err_msg = str(e)
            self.log(f"[REC-MOT] Erreur : {err_msg}")
            self.master.after(0, lambda: messagebox.showerror("Erreur enregistrement mot", err_msg))
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            self.recording_micro = False
            self._set_record_status("Enregistrement : prêt")

    def record_phrase_thread(self):
        tmp_path = None
        try:
            wav = self._record_micro_seconds(5.0)
            tmp_path = self._write_temp_wav(wav)
            self._set_record_status("Enregistrement : analyse de la phrase...")

            detections, phrase = detect_words_in_wav_segmented(
                tmp_path,
                self.model,
                self.mean,
                self.std,
                confidence_threshold=0.35,
                silence_rms_threshold=0.003,
                relative_threshold=0.14,
                low_relative_threshold=0.07,
                min_speech_seconds=0.08,
                min_silence_seconds=0.12,
                padding_seconds=0.06,
                max_segment_seconds=1.20,
                classification_window_seconds=1.0,
            )

            result = format_phrase_result(detections, phrase)
            self.log("[REC-PHRASE] " + result.replace("\n", "\n[REC-PHRASE] "))

            if detections:
                ordered_detections = sorted(detections, key=lambda d: d.get("start", 0.0))
                words = [det["word"] for det in ordered_detections]
                self.master.after(0, lambda ws=words: self.video_player.play_phrase(ws))

            self.master.after(0, lambda r=result: messagebox.showinfo("Phrase reconnue au micro", r))

        except Exception as e:
            err_msg = str(e)
            self.log(f"[REC-PHRASE] Erreur : {err_msg}")
            self.master.after(0, lambda: messagebox.showerror("Erreur enregistrement phrase", err_msg))
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            self.recording_micro = False
            self._set_record_status("Enregistrement : prêt")
