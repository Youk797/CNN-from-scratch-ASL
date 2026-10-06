import os
import threading
from tkinter import filedialog, messagebox

from ..Entrainement.train_loop import train_model
from ..Entrainement.sauvegarde import save_model, load_model
from .snapshot import MODEL_SNAPSHOT


class Training:
    def try_load_model(self):
        if os.path.exists(MODEL_SNAPSHOT):
            try:
                self.model, self.mean, self.std = load_model(MODEL_SNAPSHOT)
                self.log(f"Modèle chargé depuis {MODEL_SNAPSHOT}.")
            except Exception as e:
                self.log(f"Impossible de charger {MODEL_SNAPSHOT} : {e}")
        else:
            self.log(f"Aucun snapshot trouvé ({MODEL_SNAPSHOT}).")

    def choose_audio(self):
        path = filedialog.askopenfilename(
            title="Choisir un audio",
            filetypes=[("Audio WAV", "*.wav"), ("Tous les fichiers", "*.*")],
        )
        if path:
            self.audio_path = path
            self.label_audio.config(text=os.path.basename(path))
            self.log(f"Audio sélectionné : {path}")

    def on_train_clicked(self):
        current = getattr(self, "_training_thread", None)
        if current is not None and current.is_alive():
            self.log("Un entraînement est déjà en cours.")
            return

        self._training_thread = threading.Thread(
            target=self.train_model_thread, daemon=True
        )
        self._training_thread.start()

    def train_model_thread(self):
        try:
            train_root = str(self.config.train_dir)
            validation_root = str(self.config.validate_dir)
            self.log(f"Chargement du dataset d'entraînement : {train_root}")
            self.log(f"Validation : {validation_root}")

            model, mean, std = train_model(
                data_root=train_root,
                validation_root=validation_root,
                max_files_per_class=1800,
                num_epochs=4,
                batch_size=32,
                lr=1e-3,
                val_split=0.25,  # fallback only if validation_root is unavailable
                seed=0,
                on_epoch_end=self._on_epoch_end,
            )

            self.model = model
            self.mean = mean
            self.std = std

            try:
                save_model(MODEL_SNAPSHOT, self.model, self.mean, self.std)
                self.log(f"Modèle sauvegardé dans {MODEL_SNAPSHOT}.")
            except Exception as e:
                self.log(f"Erreur lors de la sauvegarde du modèle : {e}")

            self.log("Entraînement terminé. Modèle prêt pour la prédiction.")
            self.master.after(
                0,
                lambda: messagebox.showinfo(
                    "Entraînement", "Entraînement terminé."
                ),
            )
        except Exception as e:
            err_msg = str(e)
            self.log(f"Erreur pendant l'entraînement : {err_msg}")
            self.master.after(
                0, lambda: messagebox.showerror("Erreur", err_msg)
            )
