
from tkinter import messagebox

import numpy as np

from ..Entrainement.prediction import predict_on_wav
from ..Entrainement.variables import CLASS_NAMES


class Prediction:
    def on_predict_clicked(self):
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
            label, probs = predict_on_wav(self.audio_path, self.model, self.mean, self.std)
            probs = np.asarray(probs).reshape(-1)

            self.log(f"Prédiction pour {self.audio_path} : {label}")

            # Affiche toutes les probabilités, triées par ordre décroissant
            order = np.argsort(probs)[::-1]
            lines = [f"Classe prédite : {label}", ""]
            for i in order:
                lines.append(f"{CLASS_NAMES[i]} : {float(probs[i]):.3f}")
                self.log(f"Proba -> {CLASS_NAMES[i]}={float(probs[i]):.3f}")

            # Affiche la vidéo ASL du mot le plus probable
            self.video_player.play_word(CLASS_NAMES[order[0]])

            messagebox.showinfo("Résultat", "\n".join(lines))

        except Exception as e:
            self.log(f"Erreur pendant la prédiction : {e}")
            messagebox.showerror("Erreur", str(e))
