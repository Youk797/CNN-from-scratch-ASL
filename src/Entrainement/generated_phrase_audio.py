
import wave
from pathlib import Path

import numpy as np

from .entrainement_utils import convertisseurwav, resizesignal
from .prediction import predict_on_signal
from .variables import CLASS_DIRS, CLASS_NAMES, SAMPLE_RATE


DIR_TO_LABEL = {d: n for d, n in zip(CLASS_DIRS, CLASS_NAMES)}


def _collect_word_files(data_root):
    """
    Récupère les fichiers WAV disponibles dans data/<mot>/.
    Renvoie un dictionnaire : {"yes": ["...wav", ...], ...}.
    """
    data_root = Path(data_root)
    word_files = {}

    for word in CLASS_DIRS:
        class_dir = data_root / word
        if not class_dir.is_dir():
            continue

        files = sorted(str(p) for p in class_dir.glob("*.wav"))
        if files:
            word_files[word] = files

    return word_files


class AudioPicker:
    """
    Choisit les WAV source en rotation aléatoire.

    Pour un mot donné, on mélange tous ses fichiers WAV, puis on les consomme
    un par un. On ne réutilise un même fichier qu'après avoir épuisé tous les
    fichiers disponibles pour ce mot. Si un mot n'a qu'un seul fichier, la
    réutilisation est forcément inévitable.
    """

    def __init__(self, word_files, rng):
        self.rng = rng
        self.pools = {}
        self.pos = {}
        self.reuse_cycles = {}
        self.total_uses = {}
        self.unique_used = {}

        for word, files in word_files.items():
            files = list(files)
            self.rng.shuffle(files)
            self.pools[word] = files
            self.pos[word] = 0
            self.reuse_cycles[word] = 0
            self.total_uses[word] = 0
            self.unique_used[word] = set()

    def choose(self, word):
        files = self.pools.get(word, [])
        if not files:
            raise RuntimeError(f"Aucun WAV disponible pour le mot '{word}' dans data/{word}/")

        if self.pos[word] >= len(files):
            self.rng.shuffle(files)
            self.pos[word] = 0
            self.reuse_cycles[word] += 1

        wav_path = files[self.pos[word]]
        self.pos[word] += 1
        self.total_uses[word] += 1
        self.unique_used[word].add(wav_path)
        return wav_path

    def stats(self):
        total_source_files = sum(len(files) for files in self.pools.values())
        total_uses = sum(self.total_uses.values())
        total_unique_used = sum(len(s) for s in self.unique_used.values())
        words_reused = sorted([w for w, c in self.reuse_cycles.items() if c > 0])
        max_reuse_cycle = max(self.reuse_cycles.values(), default=0)
        return {
            "total_source_files": int(total_source_files),
            "total_audio_uses": int(total_uses),
            "total_unique_source_files_used": int(total_unique_used),
            "words_reused_after_exhaustion": words_reused,
            "max_reuse_cycle": int(max_reuse_cycle),
        }


def _normalise_peak(x, target_peak=0.85):
    """Normalise un extrait pour éviter les phrases trop faibles ou saturées."""
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-6:
        return x
    x = x / peak * target_peak
    return np.clip(x, -0.99, 0.99).astype(np.float32)


def _write_wav_int16(path, x, sample_rate=SAMPLE_RATE):
    """Écrit un signal float32 [-1, 1] dans un fichier WAV 16-bit mono."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    x_i16 = (x * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(x_i16.tobytes())


def create_phrase_wav(
    words,
    audio_picker,
    output_path,
    silence_seconds=0.45,
    word_seconds=1.0,
):

    silence_len = max(0, int(float(silence_seconds) * SAMPLE_RATE))
    word_len = max(1, int(float(word_seconds) * SAMPLE_RATE))
    silence = np.zeros((silence_len,), dtype=np.float32)

    parts = [silence]
    cursor = silence_len
    used_sources = []
    word_intervals = []

    for word in words:
        wav_path = audio_picker.choose(word)
        used_sources.append(wav_path)
        wav = convertisseurwav(wav_path, expected_sr=SAMPLE_RATE)
        wav = resizesignal(wav, target_len=word_len)
        wav = _normalise_peak(wav)

        start_sample = cursor
        end_sample = cursor + len(wav)
        word_intervals.append({
            "word": DIR_TO_LABEL[word],
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "start": float(start_sample / SAMPLE_RATE),
            "end": float(end_sample / SAMPLE_RATE),
            "source_wav": wav_path,
        })

        parts.append(wav)
        cursor = end_sample
        parts.append(silence)
        cursor += silence_len

    phrase_wav = np.concatenate(parts).astype(np.float32)
    _write_wav_int16(output_path, phrase_wav, sample_rate=SAMPLE_RATE)
    return str(output_path), used_sources, word_intervals


def detect_generated_words_by_known_intervals(
    wav_path,
    word_intervals,
    model,
    mean,
    std,
    confidence_threshold=0.0,
):
    """
    Détecte les mots d'une phrase générée en testant uniquement les morceaux
    où un mot a été inséré.

    """
    wav = convertisseurwav(wav_path, expected_sr=SAMPLE_RATE)
    detections = []

    for interval in word_intervals:
        start = int(interval["start_sample"])
        end = int(interval["end_sample"])
        segment = wav[start:end]

        if segment.size == 0:
            continue

        label, probs = predict_on_signal(segment, model, mean, std)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        if confidence < confidence_threshold:
            continue

        detections.append({
            "word": str(label).upper(),
            "confidence": confidence,
            "start": float(interval["start"]),
            "end": float(interval["end"]),
            "center": float((interval["start"] + interval["end"]) / 2.0),
        })

    phrase = " ".join(det["word"] for det in detections)
    return detections, phrase


def _frame_rms(x, frame_len, hop_len):
    """RMS par petite fenêtre, utilisé pour la segmentation par silence."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size), mode="constant")

    starts = list(range(0, max(1, x.size - frame_len + 1), hop_len))
    if not starts:
        starts = [0]

    rms_values = []
    for s in starts:
        frame = x[s:s + frame_len]
        if frame.size < frame_len:
            frame = np.pad(frame, (0, frame_len - frame.size), mode="constant")
        rms_values.append(float(np.sqrt(np.mean(frame * frame))))

    return np.asarray(rms_values, dtype=np.float32), starts


def detect_words_by_silence_in_wav(
    wav_path,
    model,
    mean,
    std,
    confidence_threshold=0.35,
    silence_rms_threshold=0.003,
    frame_seconds=0.025,
    hop_seconds=0.010,
    relative_threshold=0.18,
    min_speech_seconds=0.10,
    min_silence_seconds=0.18,
    padding_seconds=0.12,
):

    wav = convertisseurwav(wav_path, expected_sr=SAMPLE_RATE)
    if wav.size == 0:
        return [], ""

    frame_len = max(1, int(frame_seconds * SAMPLE_RATE))
    hop_len = max(1, int(hop_seconds * SAMPLE_RATE))
    rms_values, starts = _frame_rms(wav, frame_len, hop_len)

    if rms_values.size == 0 or float(np.max(rms_values)) < silence_rms_threshold:
        return [], ""

    threshold = max(float(silence_rms_threshold), float(relative_threshold) * float(np.max(rms_values)))
    speech = rms_values >= threshold

    min_silence_frames = max(1, int(min_silence_seconds / hop_seconds))
    min_speech_frames = max(1, int(min_speech_seconds / hop_seconds))

    i = 0
    while i < len(speech):
        if speech[i]:
            i += 1
            continue
        j = i
        while j < len(speech) and not speech[j]:
            j += 1
        gap_len = j - i
        if i > 0 and j < len(speech) and gap_len < min_silence_frames:
            speech[i:j] = True
        i = j

    segments = []
    i = 0
    while i < len(speech):
        if not speech[i]:
            i += 1
            continue
        j = i
        while j < len(speech) and speech[j]:
            j += 1
        if j - i >= min_speech_frames:
            start_sample = starts[i]
            end_sample = min(wav.size, starts[j - 1] + frame_len)
            pad = int(padding_seconds * SAMPLE_RATE)
            start_sample = max(0, start_sample - pad)
            end_sample = min(wav.size, end_sample + pad)
            segments.append((start_sample, end_sample))
        i = j

    detections = []
    for start_sample, end_sample in segments:
        segment = wav[start_sample:end_sample]
        if segment.size == 0:
            continue

        label, probs = predict_on_signal(segment, model, mean, std)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        if confidence < confidence_threshold:
            continue

        start_s = start_sample / SAMPLE_RATE
        end_s = end_sample / SAMPLE_RATE
        detections.append({
            "word": str(label).upper(),
            "confidence": confidence,
            "start": float(start_s),
            "end": float(end_s),
            "center": float((start_s + end_s) / 2.0),
        })

    phrase = " ".join(det["word"] for det in detections)
    return detections, phrase
