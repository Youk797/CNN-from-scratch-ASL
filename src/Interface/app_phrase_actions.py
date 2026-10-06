
import threading
from tkinter import messagebox

from ..Entrainement.phrase_prediction import detect_words_in_wav_segmented, format_phrase_result
from ..Entrainement.generated_phrase_test import run_generated_phrase_tests, format_generated_phrase_report


class PhraseActions:
    def on_phrase_clicked(self):
        # Vérifie qu'un audio a été choisi
        if self.audio_path is None:
            messagebox.showwarning("Audio manquant", "Choisis d'abord un audio.")
            return

        # Vérifie qu'un modèle est disponible
        if self.model is None or self.mean is None or self.std is None:
            messagebox.showwarning(
                "Modèle manquant",
                "Aucun modèle en mémoire. Clique d'abord sur 'Entraîner un nouveau modèle'."
            )
            return

        try:
            detections, phrase = detect_words_in_wav_segmented(
                self.audio_path,
                self.model,
                self.mean,
                self.std,
                confidence_threshold=0.40,
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
            self.log("[PHRASE] " + result.replace("\n", "\n[PHRASE] "))

            # Affiche tous les signes reconnus dans l'ordre, puis recommence en boucle
            if detections:
                ordered_detections = sorted(detections, key=lambda d: d.get("start", 0.0))
                words = [det["word"] for det in ordered_detections]
                self.video_player.play_phrase(words)

            messagebox.showinfo("Phrase reconnue", result)

        except Exception as e:
            self.log(f"Erreur pendant la reconnaissance de phrase : {e}")
            messagebox.showerror("Erreur", str(e))

    def on_generated_phrase_test_clicked(self):
        if not self._model_ready():
            messagebox.showwarning(
                "Modèle manquant",
                "Charge ou entraîne un modèle avant de tester des phrases générées."
            )
            return

        t = threading.Thread(target=self.generated_phrase_test_thread)
        t.daemon = True
        t.start()

    def generated_phrase_test_thread(self):
        try:


            data_root = str(self.config.test_dir)

            self.log(f"[PHRASE-GEN] Génération temporaire de phrases depuis le jeu de test : {data_root}")
            self.log("[PHRASE-GEN] 1000 phrases seront générées et testées. Cela peut prendre un peu de temps.")

            def on_progress(done, total):
                self.log(f"[PHRASE-GEN] Progression : {done}/{total} phrases testées")

            summary = run_generated_phrase_tests(
                data_root=data_root,
                model=self.model,
                mean=self.mean,
                std=self.std,
                num_phrases=1000,
                seed=0,
                silence_seconds=0.45,
                word_seconds=1.0,
                detection_mode="segmentation_audio",
                window_seconds=1.0,
                hop_seconds=0.25,
                confidence_threshold=0.35,
                silence_rms_threshold=0.003,
                merge_same_word_seconds=0.75,
                on_progress=on_progress,
                progress_every=50,
            )

            report = format_generated_phrase_report(summary, max_details=30)
            self.log("[PHRASE-GEN] " + report.replace("\n", "\n[PHRASE-GEN] "))
            self.master.after(0, lambda: messagebox.showinfo("Test phrases générées", report))

        except Exception as e:
            err_msg = str(e)
            self.log(f"[PHRASE-GEN] Erreur : {err_msg}")
            self.master.after(0, lambda: messagebox.showerror("Erreur test phrases générées", err_msg))
