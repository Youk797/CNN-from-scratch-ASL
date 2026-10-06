
import os
import tempfile
from pathlib import Path

import numpy as np

from .phrase_prediction import detect_words_in_wav, detect_words_in_wav_segmented
from .generated_phrase_templates import (
    BASE_PHRASE_TEMPLATES,
    _build_candidate_phrases,
    _choose_phrases,
)
from .generated_phrase_audio import (
    DIR_TO_LABEL,
    AudioPicker,
    _collect_word_files,
    _normalise_peak,
    _write_wav_int16,
    create_phrase_wav,
    detect_generated_words_by_known_intervals,
    _frame_rms,
    detect_words_by_silence_in_wav,
)
from .generated_phrase_scoring import (
    _levenshtein_distance,
    _lcs_length,
    score_word_sequence,
)


def run_generated_phrase_tests(
    data_root,
    model,
    mean,
    std,
    num_phrases=1000,
    seed=None,
    silence_seconds=0.45,
    word_seconds=1.0,
    detection_mode="segmentation_audio",
    window_seconds=1.0,
    hop_seconds=0.25,
    confidence_threshold=0.35,
    silence_rms_threshold=0.003,
    merge_same_word_seconds=0.75,
    on_progress=None,
    progress_every=50,
):
    """
    Génère des phrases audio temporaires, les teste avec detect_words_in_wav,
    puis renvoie un résumé avec un score moyen.
    """
    rng = np.random.default_rng(seed)
    word_files = _collect_word_files(data_root)

    if not word_files:
        raise RuntimeError(
            f"Aucun fichier WAV trouvé dans {data_root}. "
        )

    phrases, candidate_count = _choose_phrases(word_files.keys(), num_phrases=num_phrases, rng=rng)
    audio_picker = AudioPicker(word_files, rng)
    results = []

    # Le dossier temporaire est automatiquement supprimé à la fin du test.
    with tempfile.TemporaryDirectory(prefix="generated_phrase_test_") as tmp_dir:
        tmp_dir = Path(tmp_dir)

        for i, words in enumerate(phrases, start=1):
            safe_name = "_".join(words)
            wav_path = tmp_dir / f"phrase_{i:04d}_{safe_name}.wav"

            try:
                _, used_sources, word_intervals = create_phrase_wav(
                    words=words,
                    audio_picker=audio_picker,
                    output_path=wav_path,
                    silence_seconds=silence_seconds,
                    word_seconds=word_seconds,
                )

                if detection_mode == "known_intervals":

                    detections, recognized_phrase = detect_generated_words_by_known_intervals(
                        str(wav_path),
                        word_intervals,
                        model,
                        mean,
                        std,
                        confidence_threshold=0.0,
                    )
                elif detection_mode in ("segmentation_audio", "segmentation", "vad", "silence"):

                    detections, recognized_phrase = detect_words_in_wav_segmented(
                        str(wav_path),
                        model,
                        mean,
                        std,
                        confidence_threshold=confidence_threshold,
                        silence_rms_threshold=silence_rms_threshold,
                        relative_threshold=0.14,
                        low_relative_threshold=0.07,
                        min_speech_seconds=0.08,
                        min_silence_seconds=0.12,
                        padding_seconds=0.06,
                        max_segment_seconds=1.20,
                        classification_window_seconds=1.0,
                    )
                else:

                    detections, recognized_phrase = detect_words_in_wav(
                        str(wav_path),
                        model,
                        mean,
                        std,
                        window_seconds=window_seconds,
                        hop_seconds=hop_seconds,
                        confidence_threshold=confidence_threshold,
                        silence_rms_threshold=silence_rms_threshold,
                        merge_same_word_seconds=merge_same_word_seconds,
                    )
            finally:

                try:
                    if wav_path.exists():
                        wav_path.unlink()
                except Exception:
                    pass

            expected_labels = [DIR_TO_LABEL[w] for w in words]
            predicted_labels = [str(det["word"]).upper() for det in detections]
            metrics = score_word_sequence(expected_labels, predicted_labels)

            results.append({
                "index": i,
                "expected": expected_labels,
                "predicted": predicted_labels,
                "recognized_phrase": recognized_phrase,
                "detections": detections,
                "metrics": metrics,
                "source_wavs": [os.path.basename(p) for p in used_sources],
            })

            if on_progress is not None and (i == 1 or i % progress_every == 0 or i == len(phrases)):
                on_progress(i, len(phrases))

    if results:
        avg_score = float(np.mean([r["metrics"]["score_sequence"] for r in results]))
        avg_recall = float(np.mean([r["metrics"]["recall_words"] for r in results]))
        avg_precision = float(np.mean([r["metrics"]["precision_words"] for r in results]))
        avg_f1 = float(np.mean([r["metrics"]["f1_words"] for r in results]))
    else:
        avg_score = avg_recall = avg_precision = avg_f1 = 0.0

    return {
        "data_root": str(data_root),
        "num_phrases": len(results),
        "detection_mode": str(detection_mode),
        "candidate_phrase_count": int(candidate_count),
        "available_words": sorted(word_files.keys()),
        "audio_picker_stats": audio_picker.stats(),
        "avg_score_sequence": avg_score,
        "avg_recall_words": avg_recall,
        "avg_precision_words": avg_precision,
        "avg_f1_words": avg_f1,
        "results": results,
    }


def format_generated_phrase_report(summary, max_details=20):
    """Prépare un rapport clair et sobre pour le test automatique de phrases."""
    audio_stats = summary.get("audio_picker_stats", {})
    results = summary.get("results", [])
    words_reused = audio_stats.get("words_reused_after_exhaustion", [])

    def pct(key):
        return f"{float(summary.get(key, 0.0)) * 100:.1f}%"

    lines = [
        "Évaluation automatique des phrases",
        "",
        "Configuration du test",
        f"- Dossier audio : {summary.get('data_root', '?')}",
        f"- Phrases testées : {summary.get('num_phrases', 0)}",
        f"- Phrases différentes possibles : {summary.get('candidate_phrase_count', '?')}",
        "- Méthode : segmentation par énergie, puis classification CNN sur une fenêtre centrée de 1 seconde",
        "",
        "Résultats globaux",
        f"- Score de séquence : {pct('avg_score_sequence')}",
        f"- Rappel mot-à-mot : {pct('avg_recall_words')}",
        f"- Précision mot-à-mot : {pct('avg_precision_words')}",
        f"- Score équilibré mot-à-mot : {pct('avg_f1_words')}",
        "",
        "Lecture des métriques",
        "- Le rappel mesure la part des mots attendus qui ont été retrouvés.",
        "- La précision mesure la part des mots reconnus qui sont corrects.",
        "- Le score équilibré résume le compromis entre les mots retrouvés et les fausses reconnaissances.",
        "",
        "Données audio utilisées",
        f"- Fichiers WAV disponibles : {audio_stats.get('total_source_files', 0)}",
        f"- Fichiers WAV utilisés pendant le test : {audio_stats.get('total_audio_uses', 0)}",
        f"- Fichiers WAV distincts utilisés : {audio_stats.get('total_unique_source_files_used', 0)}",
    ]

    if words_reused:
        sample = ", ".join(str(w) for w in words_reused[:12])
        suffix = "..." if len(words_reused) > 12 else ""
        lines.append(f"- Réutilisation après épuisement pour : {sample}{suffix}")
    else:
        lines.append("- Aucune réutilisation avant épuisement des fichiers disponibles par mot.")

    lines.extend([
        "",
        f"Exemples de prédiction ({min(max_details, len(results))} affichés)",
    ])

    for r in results[:max_details]:
        m = r.get("metrics", {})
        expected = " ".join(r.get("expected", []))
        predicted = " ".join(r.get("predicted", [])) if r.get("predicted") else "<aucun mot>"
        lines.append(
            f"- #{int(r.get('index', 0)):04d} | attendu : {expected} | reconnu : {predicted} "
            f"| score : {float(m.get('score_sequence', 0.0)) * 100:.1f}% "
            f"| mots corrects : {int(m.get('correct_ordered_words', 0))}/{int(m.get('expected_count', 0))}"
        )

    remaining = len(results) - max_details
    if remaining > 0:
        lines.append(f"- {remaining} autres phrases non affichées.")

    lines.extend([
        "",
        "Les fichiers audio temporaires ont été supprimés automatiquement après le test.",
    ])
    return "\n".join(lines)
