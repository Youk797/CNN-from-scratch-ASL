
from tkinter import messagebox

from ..Entrainement.live_word_detection import LiveWordDetector


class LiveDetection:
    def on_live_start_clicked(self):
        # Vérifie qu'un modèle est disponible avant d'activer le micro.
        if not self._model_ready():
            messagebox.showwarning(
                "Modèle manquant",
                "Charge ou entraîne un modèle avant de démarrer l'écoute micro."
            )
            return

        if self.live_detector is not None and self.live_detector.running:
            self.log("[LIVE] L'écoute micro est déjà active.")
            return

        def on_detection(det):
            self.master.after(0, lambda: self._handle_live_detection(det))

        def on_status(msg):
            self.log(msg)
            if "Micro arrêté" in msg:
                self.master.after(
                    0,
                    lambda: self.label_live.config(
                        text="Micro : arrêté | Dernier mot détecté : aucun"
                    )
                )

        self.live_detector = LiveWordDetector(
            model=self.model,
            mean=self.mean,
            std=self.std,
            on_detection=on_detection,
            on_status=on_status,
            window_seconds=1.0,
            hop_seconds=0.35,
            confidence_threshold=0.70,
            silence_rms_threshold=0.004,
            repeat_cooldown_seconds=1.0,
        )

        self.live_detector.start()
        self.label_live.config(text="Micro : démarrage... | Dernier mot détecté : aucun")
        self.log("[LIVE] Démarrage de l'écoute micro...")

    def on_live_stop_clicked(self):
        if self.live_detector is None or not self.live_detector.running:
            self.log("[LIVE] L'écoute micro n'est pas active.")
            self.label_live.config(text="Micro : arrêté | Dernier mot détecté : aucun")
            return

        self.live_detector.stop()
        self.label_live.config(text="Micro : arrêt demandé... | Dernier mot détecté : aucun")
        self.log("[LIVE] Arrêt de l'écoute micro demandé.")

    def _handle_live_detection(self, det):
        word = det["word"]
        confidence = float(det["confidence"])
        rms = float(det["rms"])

        self.label_live.config(
            text=f"Micro : actif | Dernier mot détecté : {word} ({confidence:.2f})"
        )
        self.log(f"[LIVE] Mot reconnu : {word} | confiance={confidence:.3f} | rms={rms:.4f}")

        # Affiche immédiatement la vidéo ASL correspondant au mot détecté
        self.video_player.play_word(word)
        self.master.bell()
