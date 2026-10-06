# Reconnaissance vocale & illustration ASL

Projet de Master 1 à Université Paris Cité : reconnaissance de mots anglais à partir d'audio, avec affichage de la vidéo ASL correspondante.

Le cœur du projet est un CNN 2D implémenté from scratch en NumPy/CuPy, sans PyTorch ni TensorFlow.

## Fonctionnalités

- Reconnaissance de 31 classes de mots anglais à partir de fichiers WAV d'environ 1 seconde.
- Conversion de l'audio en caractéristiques log-Mel à 3 canaux : log-Mel, Δ et ΔΔ.
- Inférence avec un CNN personnalisé fonctionnant sur NumPy (CPU) ou CuPy (GPU).
- Détection de plusieurs mots dans un audio plus long grâce à une segmentation de la parole.
- Reconnaissance en direct via microphone et enregistrement audio de durée fixe.
- Affichage d'une vidéo ASL correspondant à chaque mot reconnu lorsque la vidéo est disponible.
- Évaluation des mots isolés avec accuracy globale, accuracy par classe et matrice de confusion.
- Évaluation de séquences de mots générées avec distance d'édition, précision, rappel et F1-score.

## Lancer le projet

```bash
pip install -r requirements.txt
python run.py
```

Un modèle préentraîné est fourni dans `models/snapshot.npz`, donc le dataset complet n'est pas nécessaire pour tester le modèle.

Pour tester directement un fichier audio :

```bash
python quick_predict.py chemin/vers/audio.wav
```
Vous pouvez aussi tester l'audio vous meme en vous enregistrant via l'interface graphique.

Liste des mots reconnus: yes, no, cat, happy, stop, on, off, one, two, go, backward, bed, bird, dog, eight, five, follow, forward, four, house, learn, left, nine, right, seven, six, three, up, visual, wow, zero.

## Dataset

L'entraînement utilise Google Speech Commands v0.02.

Le dataset complet (~2,5 Go) n'est pas inclus dans le dépôt.  
Les instructions de préparation sont disponibles dans `scripts/prepare_speech_commands.py`.


