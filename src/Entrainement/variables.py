# Noms des dossiers où se trouvent les fichiers audio
CLASS_DIRS = ["yes", "no", "cat", "happy", "stop", "on", "off", "one", "two", "go", "backward", "bed", "bird", "dog","eight","five","follow","forward","four", "house", "learn", "left", "nine", "right","seven", "six", "three", "up", "visual", "wow", "zero"]

# Noms affichés des classes 
CLASS_NAMES = ["YES", "NO", "CAT", "HAPPY", "STOP", "ON", "OFF", "ONE", "TWO", "GO", "BACKWARD", "BED", "BIRD", "DOG", "EIGHT", "FIVE", "FOLLOW", "FORWARD", "FOUR", "HOUSE", "LEARN", "LEFT", "NINE", "RIGHT", "SEVEN", "SIX", "THREE", "UP", "VISUAL", "WOW", "ZERO"]
# fréquence d'échantillonnage 16 000 Hz 
SAMPLE_RATE = 16000

# Durée de chaque extrait audio qu'on utilise 
CLIP_SECONDS = 1.0

# Nombre d'échantillons correspondant à 1 seconde
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# Paramètres pour transformer l'audio en "image" 
# N_FFT : taille de la FFT (définit la résolution fréquentielle)
N_FFT = 512

# taille de la fenêtre (ici 400 échantillons ≈ 25 ms à 16 kHz)
WIN_LENGTH = 400

# pas entre deux fenêtres 
HOP_LENGTH = 160

# N_MELS : nombre de bandes Mel (hauteur de l'image en "fréquences")
N_MELS = 64

#  fréquences min et max gardées 
FMIN = 20.0
FMAX = 8000.0

# Nombre de frames après découpage 
# Avec 16000 samples, une fenêtre de 400 et un pas de 160, on obtient ~98 colonnes de temps
N_FRAMES = 1 + (CLIP_SAMPLES - WIN_LENGTH) // HOP_LENGTH

# Forme attendue par le CNN 
# Option "3 canaux"  entrée de forme (3, N_MELS, N_FRAMES)
# N_MELS = hauteur (fréquences)
# N_FRAMES = largeur (temps)
INPUT_SHAPE = (3, N_MELS, N_FRAMES)

