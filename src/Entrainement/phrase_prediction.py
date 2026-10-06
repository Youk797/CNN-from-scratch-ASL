
import numpy as np

from .entrainement_utils import convertisseurwav
from .prediction import predict_on_signal
from .variables import SAMPLE_RATE
from .audio_segmentation import (
    _moving_average,
    _frame_rms_with_starts,
    _fill_short_silences,
    _remove_short_speech,
    _segments_from_mask,
    _merge_close_segments,
    _hysteresis_speech_mask,
    _split_long_segment_by_valleys,
    segment_speech_by_energy,
)


def _rms(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def _merge_detections(detections, merge_same_word_seconds=0.75):
    # Fusionne les prédictions identiques proches dans le temps.
    # Sans ça, un seul mot peut apparaître 3 ou 4 fois à cause de la fenêtre glissante.
    merged = []

    for det in detections:
        if not merged:
            merged.append(det.copy())
            continue

        last = merged[-1]
        same_word = det["word"] == last["word"]
        close_enough = det["start"] <= last["end"] + merge_same_word_seconds

        if same_word and close_enough:
            last["end"] = max(last["end"], det["end"])
            if det["confidence"] > last["confidence"]:
                last["confidence"] = det["confidence"]
                last["center"] = det["center"]
        else:
            merged.append(det.copy())

    return merged


def detect_words_in_wav(
    wav_path,
    model,
    mean,
    std,
    window_seconds=1.0,
    hop_seconds=0.25,
    confidence_threshold=0.65,
    silence_rms_threshold=0.003,
    merge_same_word_seconds=0.75,
):

    wav = convertisseurwav(wav_path, expected_sr=SAMPLE_RATE)

    if wav.size == 0:
        return [], ""

    window_samples = max(1, int(window_seconds * SAMPLE_RATE))
    hop_samples = max(1, int(hop_seconds * SAMPLE_RATE))

    detections = []

    if len(wav) <= window_samples:
        starts = [0]
    else:
        starts = list(range(0, len(wav) - window_samples + 1, hop_samples))
        last_start = len(wav) - window_samples
        if starts[-1] != last_start:
            starts.append(last_start)

    for start in starts:
        end = min(start + window_samples, len(wav))
        segment = wav[start:end]

        if _rms(segment) < silence_rms_threshold:
            continue

        label, probs = predict_on_signal(segment, model, mean, std)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        if confidence < confidence_threshold:
            continue

        start_s = start / SAMPLE_RATE
        end_s = end / SAMPLE_RATE
        detections.append({
            "word": str(label).upper(),
            "confidence": confidence,
            "start": float(start_s),
            "end": float(end_s),
            "center": float((start_s + end_s) / 2.0),
            "method": "sliding_window",
        })

    detections = _merge_detections(detections, merge_same_word_seconds=merge_same_word_seconds)
    phrase = " ".join(det["word"] for det in detections)
    return detections, phrase


def _extract_centered_classification_window(wav, start_sample, end_sample, window_seconds=1.0):
    wav = np.asarray(wav, dtype=np.float32)
    wanted = max(1, int(float(window_seconds) * SAMPLE_RATE))
    start_sample = int(start_sample)
    end_sample = int(end_sample)

    center = (start_sample + end_sample) // 2
    win_start = center - wanted // 2
    win_end = win_start + wanted

    src_start = max(0, win_start)
    src_end = min(int(wav.size), win_end)
    out = wav[src_start:src_end]

    pad_left = max(0, -win_start)
    pad_right = max(0, win_end - int(wav.size))
    if pad_left or pad_right or out.size < wanted:
        out = np.pad(out, (pad_left, max(pad_right, wanted - out.size - pad_left)), mode="constant")

    if out.size > wanted:
        out = out[:wanted]
    elif out.size < wanted:
        out = np.pad(out, (0, wanted - out.size), mode="constant")

    return out.astype(np.float32), int(src_start), int(src_end)


def detect_words_in_wav_segmented(
    wav_path,
    model,
    mean,
    std,
    confidence_threshold=0.35,
    silence_rms_threshold=0.003,
    relative_threshold=0.14,
    low_relative_threshold=0.07,
    min_speech_seconds=0.08,
    min_silence_seconds=0.12,
    padding_seconds=0.06,
    max_segment_seconds=1.20,
    classification_window_seconds=1.0,
):

    wav = convertisseurwav(wav_path, expected_sr=SAMPLE_RATE)
    if wav.size == 0:
        return [], ""

    segments = segment_speech_by_energy(
        wav,
        silence_rms_threshold=silence_rms_threshold,
        relative_threshold=relative_threshold,
        low_relative_threshold=low_relative_threshold,
        min_speech_seconds=min_speech_seconds,
        min_silence_seconds=min_silence_seconds,
        padding_seconds=padding_seconds,
        max_segment_seconds=max_segment_seconds,
    )

    detections = []
    for start_sample, end_sample in segments:
        segment = wav[start_sample:end_sample]
        if segment.size == 0:
            continue


        segment_for_cnn, class_start_sample, class_end_sample = _extract_centered_classification_window(
            wav,
            start_sample,
            end_sample,
            window_seconds=classification_window_seconds,
        )

        label, probs = predict_on_signal(segment_for_cnn, model, mean, std)
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
            "method": "energy_segmentation_centered_window",
            "class_start": float(class_start_sample / SAMPLE_RATE),
            "class_end": float(class_end_sample / SAMPLE_RATE),
        })

    phrase = " ".join(det["word"] for det in detections)
    return detections, phrase


def format_phrase_result(detections, phrase):
    if not detections:
        return "Aucun mot reconnu avec assez de confiance."

    method = detections[0].get("method", "unknown")
    lines = [f"Phrase reconnue : {phrase}", f"Méthode : {method}", "", "Détails :"]
    for det in detections:
        lines.append(
            f"- {det['word']} "
            f"[{det['start']:.2f}s - {det['end']:.2f}s] "
            f"confiance={det['confidence']:.3f}"
        )
    return "\n".join(lines)
