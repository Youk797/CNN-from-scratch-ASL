import tkinter as tk
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageTk


class AslVideoPlayer:
    """
    Lecteur vidéo simple intégré dans Tkinter.

    - play_word(word) : lit un seul signe en boucle.
    - play_phrase(words) : lit tous les signes reconnus dans l'ordre,
      puis recommence la phrase en boucle.
    """

    def __init__(self, parent, videos_dir, width=320, height=240, fps=24):
        self.parent = parent
        self.videos_dir = Path(videos_dir)
        self.width = width
        self.height = height
        self.frame_delay_ms = int(1000 / fps)

        self.container = tk.Frame(parent, bd=1, relief="solid")
        tk.Label(self.container, text="Illustration ASL",
                 font=("TkDefaultFont", 10, "bold")).pack(pady=(6, 2))

        self.word_label = tk.Label(self.container, text="-",
                                   font=("TkDefaultFont", 12))
        self.word_label.pack(pady=(0, 4))

        self.canvas = tk.Label(self.container, width=width, height=height,
                               bg="#202020", fg="#cccccc", text="(aucune vidéo)")
        self.canvas.pack(padx=6, pady=6)

        self.frames = []
        self.frame_index = 0
        self.after_id = None
        self.photo = None  # référence pour éviter le garbage collector

        # Mode phrase
        self.playlist = []
        self.playlist_index = 0
        self.playlist_mode = False

        # Cache pour éviter de relire les mêmes .mp4 à chaque boucle
        self._frames_cache = {}

    def pack(self, **kwargs):
        self.container.pack(**kwargs)

    def play_word(self, word):
        """
        Affiche un seul mot en boucle.
        Utilisé pour la prédiction d'un audio qui contient un seul mot.
        """
        self.stop()
        self.playlist_mode = False
        self.playlist = []
        self.playlist_index = 0

        word = str(word).strip().upper()
        self.word_label.config(text=word)

        self.frames = self._load_frames_for_word(word)
        if not self.frames:
            self.canvas.config(image="", text=f"Pas d'illustration pour « {word} »")
            return

        self.frame_index = 0
        self._show_next_frame()

    def play_phrase(self, words):
        """
        Affiche plusieurs signes dans l'ordre, puis recommence en boucle.

        Exemple :
            play_phrase(["YES", "GO", "LEFT"])
        affiche YES -> GO -> LEFT -> YES -> GO -> LEFT -> ...
        """
        self.stop()

        cleaned_words = []
        for word in words:
            word = str(word).strip().upper()
            if word:
                cleaned_words.append(word)

        if not cleaned_words:
            self.playlist_mode = False
            self.playlist = []
            self.frames = []
            self.word_label.config(text="-")
            self.canvas.config(image="", text="(aucune vidéo)")
            return

        self.playlist_mode = True
        self.playlist = cleaned_words
        self.playlist_index = 0
        self._start_current_phrase_word()

    def stop(self):
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
            self.after_id = None

    def _load_frames_for_word(self, word):
        """
        Charge les frames PIL d'une vidéo correspondant à un mot.
        Les fichiers doivent être nommés en minuscules : yes.mp4, go.mp4, etc.
        """
        key = str(word).strip().lower()
        if not key:
            return []

        if key in self._frames_cache:
            return self._frames_cache[key]

        path = self.videos_dir / f"{key}.mp4"
        if not path.is_file():
            self._frames_cache[key] = []
            return []

        frames = []
        for frame in iio.imiter(str(path)):
            img = Image.fromarray(frame).convert("RGB")
            img.thumbnail((self.width, self.height), Image.LANCZOS)
            frames.append(img)

        self._frames_cache[key] = frames
        return frames

    def _start_current_phrase_word(self):
        if not self.playlist:
            return

        if self.playlist_index >= len(self.playlist):
            self.playlist_index = 0

        word = self.playlist[self.playlist_index]
        self.word_label.config(
            text=f"{word} ({self.playlist_index + 1}/{len(self.playlist)})"
        )

        self.frames = self._load_frames_for_word(word)
        self.frame_index = 0

        if not self.frames:
            self.canvas.config(image="", text=f"Pas d'illustration pour « {word} »")
            # On passe au signe suivant après un court délai.
            self.after_id = self.parent.after(800, self._go_to_next_phrase_word)
            return

        self._show_next_frame()

    def _go_to_next_phrase_word(self):
        if not self.playlist_mode or not self.playlist:
            return

        self.playlist_index = (self.playlist_index + 1) % len(self.playlist)
        self._start_current_phrase_word()

    def _show_next_frame(self):
        if not self.frames:
            return

        self.photo = ImageTk.PhotoImage(self.frames[self.frame_index])
        self.canvas.config(image=self.photo, text="")

        self.frame_index += 1

        if self.frame_index >= len(self.frames):
            if self.playlist_mode:
                # En mode phrase, une vidéo est jouée une seule fois,
                # puis on passe au signe suivant.
                self.after_id = self.parent.after(
                    self.frame_delay_ms,
                    self._go_to_next_phrase_word
                )
            else:
                # En mode mot seul, la même vidéo tourne en boucle.
                self.frame_index = 0
                self.after_id = self.parent.after(
                    self.frame_delay_ms,
                    self._show_next_frame
                )
        else:
            self.after_id = self.parent.after(
                self.frame_delay_ms,
                self._show_next_frame
            )
