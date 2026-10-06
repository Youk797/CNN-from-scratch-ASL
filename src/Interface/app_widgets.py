
import tkinter as tk

from .video_player import AslVideoPlayer


class WidgetBuilder:
    def create_widgets(self):
        # Layout deux colonnes : controls/log à gauche, illustration ASL à droite
        frame_root = tk.Frame(self.master)
        frame_root.pack(fill="both", expand=True)

        frame_left = tk.Frame(frame_root)
        frame_left.pack(side="left", fill="both", expand=True)

        frame_right = tk.Frame(frame_root)
        frame_right.pack(side="right", fill="y", padx=(4, 8), pady=8)

        # Panneau d'illustration ASL à droite
        self.video_player = AslVideoPlayer(
            self.master,
            videos_dir=self.config.asl_videos_dir,
            width=340,
            height=260,
            fps=24,
        )
        self.video_player.pack(in_=frame_right, fill="y")

        # affiche si on utilise CuPy ou NumPy
        frame_backend = tk.Frame(frame_left)
        frame_backend.pack(padx=10, pady=5, fill="x")

        lbl_backend = tk.Label(
            frame_backend,
            text=f"Backend de calcul utilisé par Cnn.py : {self.backend_name}",
            anchor="w",
        )
        lbl_backend.pack(side="left")

        btn_fullscreen = tk.Button(
            frame_backend,
            text="Plein écran / fenêtre (F11)",
            command=self.toggle_fullscreen,
        )
        btn_fullscreen.pack(side="right", padx=5)

        btn_exit_fullscreen = tk.Button(
            frame_backend,
            text="Quitter plein écran (Échap)",
            command=self.exit_fullscreen,
        )
        btn_exit_fullscreen.pack(side="right", padx=5)

        # bouton pour choisir un audio
        frame_top = tk.Frame(frame_left)
        frame_top.pack(pady=10, fill="x")

        btn_choose = tk.Button(frame_top, text="Choisir un audio...", command=self.choose_audio)
        btn_choose.pack(side="left", padx=5)
    
        # Label qui affichera le chemin de l'audio sélectionné
        self.label_audio = tk.Label(frame_top, text="Aucun audio sélectionné")
        self.label_audio.pack(side="left", padx=5)

        #Zone avec les boutons d'actions (entrainer, predire, tester)
        frame_actions = tk.Frame(frame_left)
        frame_actions.pack(pady=10)

        # bouton pour entraine le modele
        btn_train = tk.Button(frame_actions, text="Entraîner un nouveau modèle", command=self.on_train_clicked)
        btn_train.grid(row=0, column=0, padx=10)

        # bouton pour utiliser le modele
        btn_predict = tk.Button(frame_actions, text="Utiliser le modèle\nsur l'audio", command=self.on_predict_clicked)
        btn_predict.grid(row=0, column=1, padx=10)

        # bouton pour tester le modele sur le repertoire test
        btn_test = tk.Button(
            frame_actions,
            text="Tester sur dossier",
            command=self.on_test_clicked
        )
        btn_test.grid(row=0, column=2, padx=10)

        # bouton pour reconnaitre plusieurs mots dans un audio long
        btn_phrase = tk.Button(
            frame_actions,
            text="Reconnaître\nune phrase",
            command=self.on_phrase_clicked
        )
        btn_phrase.grid(row=0, column=3, padx=10)

        # bouton pour générer automatiquement des phrases audio temporaires
        btn_generated_phrase_test = tk.Button(
            frame_actions,
            text="Tester phrases\ngénérées",
            command=self.on_generated_phrase_test_clicked
        )
        btn_generated_phrase_test.grid(row=0, column=4, padx=10)

        # Zone pour la reconnaissance micro en direct
        frame_live = tk.Frame(frame_left)
        frame_live.pack(pady=5, fill="x")

        btn_live_start = tk.Button(
            frame_live,
            text="Démarrer écoute micro",
            command=self.on_live_start_clicked
        )
        btn_live_start.pack(side="left", padx=5)

        btn_live_stop = tk.Button(
            frame_live,
            text="Arrêter écoute micro",
            command=self.on_live_stop_clicked
        )
        btn_live_stop.pack(side="left", padx=5)

        self.label_live = tk.Label(
            frame_live,
            text="Micro : arrêté | Dernier mot détecté : aucun",
            anchor="w"
        )
        self.label_live.pack(side="left", padx=10, fill="x", expand=True)

        # Zone pour un micro enregistré, plus fiable que le live continu
        frame_record = tk.LabelFrame(frame_left, text="Micro enregistré (plus fiable que le live)")
        frame_record.pack(pady=5, padx=10, fill="x")

        btn_record_word = tk.Button(
            frame_record,
            text="Enregistrer un mot\n(1,6 s)",
            command=self.on_record_word_clicked
        )
        btn_record_word.pack(side="left", padx=5, pady=5)

        btn_record_phrase = tk.Button(
            frame_record,
            text="Enregistrer une phrase\n(5 s)",
            command=self.on_record_phrase_clicked
        )
        btn_record_phrase.pack(side="left", padx=5, pady=5)

        self.label_record = tk.Label(
            frame_record,
            text="Enregistrement : prêt",
            anchor="w"
        )
        self.label_record.pack(side="left", padx=10, fill="x", expand=True)

        # Zone de texte log
        self.text_log = tk.Text(frame_left, height=15)
        self.text_log.pack(padx=10, pady=10, fill="both", expand=True)
