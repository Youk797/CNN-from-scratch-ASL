
from ..Config.config import Config
from ..Cnn.numpy_cupy import xp
from ..Entrainement.nettoyage import remove_quarantine_dir

from .app_widgets import WidgetBuilder
from .app_training import Training
from .app_testing import Testing
from .app_live import LiveDetection
from .app_recording import Recording
from .app_phrase_actions import PhraseActions
from .app_prediction import Prediction


class App(
    WidgetBuilder,
    Training,
    Testing,
    LiveDetection,
    Recording,
    PhraseActions,
    Prediction,
):
    def __init__(self, master, data_config : Config):
        # Fenêtre principale Tkinter
        self.master = master
        self.config = data_config
        master.title("IA CNN") #titre

        # Mode plein écran au lancement.
        # Échap permet de quitter le plein écran, F11 permet de basculer plein écran / fenêtre.
        self.fullscreen = True
        self.master.attributes("-fullscreen", True)
        self.master.bind("<Escape>", self.exit_fullscreen)
        self.master.bind("<F11>", self.toggle_fullscreen)

        # Si l'utilisateur quitte le plein écran, on garde une taille de fenêtre correcte.
        self.master.geometry("1440x820")

        #indique si on utilise cupy ou numpy
        self.backend_name = "CUPY (GPU)" if xp.__name__ == "cupy" else "NUMPY (CPU)"

        # Variables pour stocker le modèle et les stats de normalisation
        self.model = None
        # mean et std servent à normaliser l'audio pour le test et l'entrainement
        self.mean = None
        self.std = None

        # Chemin du fichier audio sélectionnée
        self.audio_path = None

        # Détecteur micro en direct
        self.live_detector = None

        # Empêche de lancer deux enregistrements micro en même temps
        self.recording_micro = False

        # Création des widgets
        self.create_widgets()
        self.log("Bienvenue !\n")
        # Tentative de chargement d'un modèle déjà sauvegardé
        self.try_load_model()
        self._quarantine_dirs = set()
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def set_fullscreen(self, enabled=True):
        # Active ou désactive le plein écran.
        self.fullscreen = bool(enabled)
        self.master.attributes("-fullscreen", self.fullscreen)

    def toggle_fullscreen(self, event=None):
        # F11 : bascule entre plein écran et mode fenêtre.
        self.set_fullscreen(not getattr(self, "fullscreen", False))
        return "break"

    def exit_fullscreen(self, event=None):
        # Échap : quitte le plein écran, mais ne ferme pas l'application.
        self.set_fullscreen(False)
        return "break"

    def on_close(self):

        # Arrête proprement le micro si la détection live est active.
        if getattr(self, "live_detector", None) is not None:
            try:
                self.live_detector.stop()
            except Exception:
                pass

        # Arrête la boucle de lecture vidéo
        if getattr(self, "video_player", None) is not None:
            try:
                self.video_player.stop()
            except Exception:
                pass

        for q in list(self._quarantine_dirs):
            try:
                remove_quarantine_dir(q)
            except Exception:
                pass
        self.master.destroy()

    def log(self, msg):

        def _append():
            # Insere le texte et scrolle à la fin
            self.text_log.insert("end", msg + "\n")
            self.text_log.see("end")
        self.master.after(0, _append)

    def _on_epoch_end(self, epoch, total_epochs):
        self.log(f"Epoch {epoch}/{total_epochs} terminée")

    def _model_ready(self):
        return self.model is not None and self.mean is not None and self.std is not None
