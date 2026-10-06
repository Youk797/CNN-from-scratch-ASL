
import glob
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from ..Entrainement.prediction import predict_on_wav
from ..Entrainement.variables import CLASS_DIRS
from ..Entrainement.nettoyage import reset_quarantine_dir


class Testing:
    def on_test_clicked(self):
        # Vérifie si le model existe
        if getattr(self, "model", None) is None or getattr(self, "mean", None) is None or getattr(self, "std", None) is None:
            messagebox.showwarning("Modèle non chargé", "Charge/entraîne un modèle avant de tester.")
            return

        # Dossier test par défaut
        default_test_root = self.config.test_dir

        required_ok = (
            os.path.isdir(default_test_root) and
            all(os.path.isdir(os.path.join(default_test_root, d)) for d in CLASS_DIRS)
        )

        if not required_ok:
            test_root = filedialog.askdirectory(
                title=f"Choisir le dossier test (doit contenir: {', '.join(CLASS_DIRS)})"
            )
            if not test_root:
                return
        else:
            test_root = default_test_root


        # Thread pour ne pas bloquer l'UI
        t = threading.Thread(target=self.test_folder_thread, args=(test_root,))
        #ferme le programme si l'utilisateur ferme l'appli, meme si un processus est en cours
        t.daemon = True
        t.start()

    def test_folder_thread(self, test_root):
            try:
                self.log(f"[TEST] Dossier: {test_root}")
                test_quarantine = os.path.join(test_root, "quarantaine")
                reset_quarantine_dir(test_quarantine)
                self._quarantine_dirs.add(test_quarantine)


                #Récupère tous les fichiers .wav 
                files = []
                for cls in CLASS_DIRS:  
                    cls_dir = os.path.join(test_root, cls)                 # dossier de la classe
                    wavs = sorted(glob.glob(os.path.join(cls_dir, "*.wav")))  # liste des wav
                    self.log(f"[TEST] {cls}: {len(wavs)} fichiers")        # combien de wav trouvés
                    for fp in wavs:
                        files.append((cls, fp))                           

                if not files:
                    raise RuntimeError("Aucun .wav trouvé dans test")

                # Matrice pour compter les bons et les mauvais résultats
                # lignes = vraie classe, colonnes = classe prédite
                cm = np.zeros((len(CLASS_DIRS), len(CLASS_DIRS)), dtype=np.int64)

                # Associe un indice à chaque classe
                dir_to_idx = {cls: i for i, cls in enumerate(CLASS_DIRS)}

                # Teste chaque fichier wav 
                n = 0
                for true_cls, fp in files:
                    true_idx = dir_to_idx[true_cls]  # index de la vraie classe

                    # Prédit la classe et les probabilités
                    try:
                        pred_label, probs = predict_on_wav(fp, self.model, self.mean, self.std, quarantine_root=test_quarantine)
                    except Exception as e:
                        self.log(f"[TEST] {fp} ignoré: {e}")
                        continue

                    pred_idx = int(np.argmax(probs))  # index de la classe la plus probable

                    # Ajoute 1 dans la matrice 
                    cm[true_idx, pred_idx] += 1
                    n += 1

                    # message toutes les 25 prédictions pour suivre l'avancement
                    if n % 25 == 0:
                        self.log(f"[TEST] {n}/{len(files)}...")

          
                total = int(cm.sum())          # nb total de fichiers testés
                correct = int(np.trace(cm))    # nb de fichiers bien classés (diagonale)
                acc = correct / max(total, 1)  

                # Calcule l'accuracy pour chaque classe 
                per_class_acc = []
                for i, cls in enumerate(CLASS_DIRS):
                    denom = int(cm[i, :].sum())                 # nb d'exemples de cette vraie classe
                    a = (int(cm[i, i]) / denom) if denom > 0 else 0.0  # proportion bien classée
                    per_class_acc.append((cls, a, denom))

                # Affiche les résultats dans le log 
                self.log("[TEST] Résultats:")
                self.log(f"  - Accuracy globale: {acc:.3f} ({correct}/{total})")
                for cls, a, denom in per_class_acc:
                    self.log(f"  - Acc {cls}: {a:.3f} (n={denom})")

                # La matrice de confusion n'est plus affichée dans le log principal.
                # Elle est uniquement affichée dans la fenêtre dédiée scrollable
                # ouverte juste après le test.
                self.log("[TEST] Matrice de confusion affichée dans une fenêtre séparée.")

                msg_lines = [
                    f"Test sur: {test_root}",
                    "",
                    f"Accuracy globale: {acc:.3f} ({correct}/{total})",
                    "",
                    "Accuracy par classe:"
                ]
                for cls, a, denom in per_class_acc:
                    msg_lines.append(f" - {cls}: {a:.3f} (n={denom})")

                report_text = "\n".join(msg_lines)

                # L'affichage Tkinter doit se faire dans le thread principal.
                # On ouvre une fenêtre dédiée, avec barres de défilement,
                # pour rendre la matrice de confusion beaucoup plus lisible.
                self.master.after(
                    0,
                    lambda cm=cm.copy(), per_class_acc=list(per_class_acc), report_text=report_text:
                        self.show_confusion_matrix_window(
                            test_root=test_root,
                            cm=cm,
                            per_class_acc=per_class_acc,
                            accuracy=acc,
                            correct=correct,
                            total=total,
                            report_text=report_text,
                        )
                )


            except Exception as e:
                err_msg = str(e)
                self.log(f"[TEST] Erreur: {err_msg}")
                self.master.after(0, lambda: messagebox.showerror("Erreur test", err_msg))

    def show_confusion_matrix_window(self, test_root, cm, per_class_acc, accuracy, correct, total, report_text=None):
        """Affiche les résultats de test dans une fenêtre scrollable.

        La matrice peut être très large quand il y a beaucoup de classes, donc on
        utilise un widget Text avec scroll vertical + horizontal.
        """
        win = tk.Toplevel(self.master)
        win.title("Résultats du test - Matrice de confusion")
        win.geometry("1150x720")
        win.minsize(850, 500)

        header_frame = tk.Frame(win)
        header_frame.pack(fill="x", padx=10, pady=(10, 4))

        title = tk.Label(
            header_frame,
            text=f"Matrice de confusion - Accuracy {accuracy:.3f} ({correct}/{total})",
            font=("TkDefaultFont", 12, "bold"),
            anchor="w",
        )
        title.pack(side="left", fill="x", expand=True)

        def copy_to_clipboard():
            text = txt.get("1.0", "end-1c")
            win.clipboard_clear()
            win.clipboard_append(text)

        btn_copy = tk.Button(header_frame, text="Copier", command=copy_to_clipboard)
        btn_copy.pack(side="right", padx=(5, 0))

        btn_close = tk.Button(header_frame, text="Fermer", command=win.destroy)
        btn_close.pack(side="right", padx=(5, 0))

        body = tk.Frame(win)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        yscroll = tk.Scrollbar(body, orient="vertical")
        xscroll = tk.Scrollbar(body, orient="horizontal")

        txt = tk.Text(
            body,
            wrap="none",
            font=("Consolas", 10),
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
        )
        yscroll.config(command=txt.yview)
        xscroll.config(command=txt.xview)

        txt.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

        labels = list(CLASS_DIRS)
        cm = np.asarray(cm, dtype=np.int64)

        # Largeur fixe pour que les colonnes restent alignées.
        col_w = max(7, max(len(str(x)) for x in labels) + 1)
        row_name_w = max(10, max(len(str(x)) for x in labels) + 2)

        lines = []
        lines.append(f"Dossier testé : {test_root}")
        lines.append(f"Accuracy globale : {accuracy:.3f} ({correct}/{total})")
        lines.append("")
        lines.append("Lecture : lignes = vraie classe, colonnes = classe prédite")
        lines.append("")

        lines.append("Accuracy par classe :")
        for cls, a, denom in per_class_acc:
            lines.append(f"  - {cls:<12} : {a:.3f}   n={denom}")
        lines.append("")

        lines.append("Matrice de confusion :")
        header = "Vrai / Prédit".ljust(row_name_w) + "".join(str(name).upper().rjust(col_w) for name in labels)
        lines.append(header)
        lines.append("-" * len(header))

        for i, true_name in enumerate(labels):
            row_values = []
            for j in range(len(labels)):
                v = int(cm[i, j])
                # On marque légèrement la diagonale, sans casser l'alignement.
                cell = f"[{v}]" if i == j else str(v)
                row_values.append(cell.rjust(col_w))
            lines.append(str(true_name).upper().ljust(row_name_w) + "".join(row_values))

        lines.append("")
        lines.append("Les valeurs entre crochets correspondent à la diagonale : prédictions correctes.")

        full_text = "\n".join(lines)
        txt.insert("1.0", full_text)
        txt.config(state="disabled")

        # On garde aussi un résumé court dans le log principal.
        if report_text:
            self.log("[TEST] Fenêtre de matrice de confusion ouverte.")
