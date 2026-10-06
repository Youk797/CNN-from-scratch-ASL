import os
import wave
from pathlib import Path
import numpy as np

from .variables import (
    CLASS_DIRS,
    SAMPLE_RATE, CLIP_SAMPLES,
    N_FFT, WIN_LENGTH, HOP_LENGTH,
    N_MELS, FMIN, FMAX,
)

from .nettoyage import quarantine_file, is_bad_audio
EPS = 1e-6  # constante pour éviter certaines divisions par zéro 
WINDOW = np.hanning(WIN_LENGTH).astype(np.float32)

def resample(x, orig_sr, target_sr):
    # Change le sample rate d'un signal audio (ex: 44100 -> 16000)

  
    if orig_sr == target_sr:
        return x.astype(np.float32)

    x = x.astype(np.float32)

    ratio = target_sr / orig_sr

    # Nouvelle longueur du signal
    new_len = int(round(len(x) * ratio))

    # si on obtient 0, on renvoie un signal vide
    if new_len <= 0:
        return np.zeros((0,), dtype=np.float32)

    # Indices de l'ancien signal
    old_idx = np.arange(len(x), dtype=np.float32)

    # Indices du nouveau signal 
    new_idx = np.linspace(0, len(x) - 1, new_len, dtype=np.float32)

    # on reconstruit le signal à la nouvelle taille
    y = np.interp(new_idx, old_idx, x).astype(np.float32)

    return y


def trouverdata():
    # Cherche un dossier data à plusieurs endroits possibles
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "data",
        here.parents[1] / "data",
        here.parents[0] / "data",
        Path.cwd() / "data",
    ]
    # On prend le premier qui existe
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)
    return "data"


def convertisseurwav(path, expected_sr=16000):
    # Lit un fichier .wav, le met en mono, et le met au bon sample rate (16kHz)

    # Lecture du wav 
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()      # nb de canaux 
        sr = wf.getframerate()       # sample rate du fichier
        sampwidth = wf.getsampwidth()# taille d'un échantillon 
        nframes = wf.getnframes()    # nb d'échantillons
        raw = wf.readframes(nframes) # données brutes

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"WAV non supporté (sampwidth={sampwidth}) pour {path}")

    # Si stéréo , on fait la moyenne des canaux -> mono
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)

    # Si pas en 16kHz, on resample vers 16kHz
    if sr != expected_sr:
        x = resample(x, sr, expected_sr)

    return x


def resizesignal(x, target_len=CLIP_SAMPLES):
    # Force un signal à faire exactement target_len échantillons
    if len(x) < target_len:
        # si trop court, on ajoute des zéros à la fin
        pad = target_len - len(x)
        return np.pad(x, (0, pad), mode="constant")
    # si trop long, on coupe
    return x[:target_len]


def hztomel(hz):
    # Convertit une fréquence en Hz vers l'échelle Mel
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def meltohz(mel):
    # Convertit une valeur Mel vers Hz
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def melfilterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):

    n_freqs = n_fft // 2 + 1  

    mel_min = hztomel(fmin)
    mel_max = hztomel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)  
    hz = meltohz(mels)

    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        center = min(center, n_freqs - 1)
        right = min(right, n_freqs - 1)

        for f in range(left, center):
            fb[i, f] = (f - left) / max(center - left, 1)
        for f in range(center, right):
            fb[i, f] = (right - f) / max(right - center, 1)

    return fb


FB = melfilterbank()


def _stft_power(x, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
    x = np.asarray(x, dtype=np.float32)

    L = x.shape[0]
    if L < win_length:
        x = np.pad(x, (0, win_length - L), mode="constant")
        L = x.shape[0]

    T = 1 + (L - win_length) // hop_length
    if T <= 0:
        n_freqs = n_fft // 2 + 1
        return np.zeros((n_freqs, 0), dtype=np.float32)

    stride = x.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(T, win_length),
        strides=(hop_length * stride, stride),
        writeable=False,
    )

    window = WINDOW if win_length == WIN_LENGTH else np.hanning(win_length).astype(np.float32)
    frames = frames * window[None, :]

    if win_length < n_fft:
        frames = np.pad(frames, ((0, 0), (0, n_fft - win_length)), mode="constant")

    X = np.fft.rfft(frames, n=n_fft, axis=1)              
    power = (np.abs(X) ** 2).astype(np.float32).T          
    return power



def logmelspectrogram(x):
    p = _stft_power(x)       
    mel = FB @ p             
    return np.log(mel + EPS).astype(np.float32) 


def delta(S):
    # Approximation de la variation dans le temps 
    F, T = S.shape
    D = np.zeros_like(S, dtype=np.float32)
    for t in range(T):
        t0 = max(0, t - 1)
        t1 = min(T - 1, t + 1)
        D[:, t] = 0.5 * (S[:, t1] - S[:, t0])
    return D


def signaltofeatures3ch(wav, check_quality=False):
    # Convertit directement un signal audio déjà chargé en 3 canaux :
    # [log-mel, delta, delta-delta].
    # Utile pour analyser des morceaux extraits d'une phrase audio plus longue.

    wav = np.asarray(wav, dtype=np.float32)
    wav = resizesignal(wav, CLIP_SAMPLES)

    if check_quality:
        bad, reasons, metrics = is_bad_audio(wav)
        if bad:
            raise ValueError(f"Audio inutilisable: {reasons}")

    S = logmelspectrogram(wav)
    dS = delta(S)
    ddS = delta(dS)

    X = np.stack([S, dS, ddS], axis=0)
    return X.astype(np.float32)


def wavtofeatures3ch(path, quarantine_root=None):
    # Convertit un wav en 3 canaux : [log-mel, delta, delta-delta]

    wav = convertisseurwav(path)            # lit wav -> mono -> 16kHz
    wav = resizesignal(wav, CLIP_SAMPLES)   # force à 1 seconde

    bad, reasons, metrics = is_bad_audio(wav)
    if bad:
        if quarantine_root:
            cls = Path(path).parent.name
            quarantine_file(path, quarantine_root, cls)
        # Do not let rejected audio leak back into the training/test tensors.
        raise ValueError(f"Audio inutilisable: {reasons}")

    return signaltofeatures3ch(wav, check_quality=False)



def build_dataset(
    data_root=None,
    max_files_per_class=None,
    seed=0,
    quarantine_root=None,
):
    # Charge les WAV des classes disponibles et construit X + y (labels)

    # Si data_root pas donné, on essaye de trouver le dossier "data"
    if data_root is None:
        data_root = trouverdata()

    rng = np.random.default_rng(seed)  # pour mélanger de façon reproductible

    X_list, y_list = [], []

    for label, dir_name in enumerate(CLASS_DIRS):
        class_dir = os.path.join(data_root, dir_name)

        # Si dossier absent, on ignore
        if not os.path.isdir(class_dir):
            print(f"[AVERTISSEMENT] Dossier {class_dir} introuvable, ignoré.")
            continue

        # Liste des fichiers wav
        files = sorted([str(p) for p in Path(class_dir).glob("*.wav")])

        # Si on veut limiter le nb de fichiers, on mélange puis on coupe
        if max_files_per_class is not None:
            rng.shuffle(files)
            files = files[:max_files_per_class]

        for fp in files:
            try:
                feat = wavtofeatures3ch(fp, quarantine_root=quarantine_root)
                X_list.append(feat)
                y_list.append(label)
            except Exception as e:
                print(f"  - {fp} ignoré: {e}")

    if not X_list:
        raise RuntimeError("Dataset vide : vérifie les dossiers de classes WAV configurés.")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 3, F, T)
    y = np.asarray(y_list, dtype=np.int64)           # (N,)
    return X, y
