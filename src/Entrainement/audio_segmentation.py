
import numpy as np

from .variables import SAMPLE_RATE


def _moving_average(x, width):
    x = np.asarray(x, dtype=np.float32)
    width = int(width)
    if width <= 1 or x.size == 0:
        return x
    kernel = np.ones((width,), dtype=np.float32) / float(width)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _frame_rms_with_starts(x, frame_len, hop_len):
    x = np.asarray(x, dtype=np.float32)
    frame_len = max(1, int(frame_len))
    hop_len = max(1, int(hop_len))

    if x.size == 0:
        return np.zeros((0,), dtype=np.float32), []

    if x.size < frame_len:
        starts = [0]
    else:
        starts = list(range(0, x.size - frame_len + 1, hop_len))
        last = x.size - frame_len
        if starts[-1] != last:
            starts.append(last)

    values = []
    for start in starts:
        frame = x[start:start + frame_len]
        if frame.size < frame_len:
            frame = np.pad(frame, (0, frame_len - frame.size), mode="constant")
        values.append(float(np.sqrt(np.mean(frame * frame))))

    return np.asarray(values, dtype=np.float32), starts


def _fill_short_silences(mask, max_gap_frames):
    mask = np.asarray(mask, dtype=bool).copy()
    i = 0
    while i < len(mask):
        if mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and not mask[j]:
            j += 1
        if i > 0 and j < len(mask) and (j - i) <= max_gap_frames:
            mask[i:j] = True
        i = j
    return mask


def _remove_short_speech(mask, min_speech_frames):
    mask = np.asarray(mask, dtype=bool).copy()
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if (j - i) < min_speech_frames:
            mask[i:j] = False
        i = j
    return mask


def _segments_from_mask(mask, starts, frame_len, wav_len, padding_samples):
    segments = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1

        start = starts[i]
        end = min(wav_len, starts[j - 1] + frame_len)
        start = max(0, start - padding_samples)
        end = min(wav_len, end + padding_samples)
        if end > start:
            segments.append((int(start), int(end)))
        i = j
    return segments


def _merge_close_segments(segments, max_gap_samples):
    if not segments:
        return []
    segments = sorted(segments)
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        last = merged[-1]
        if start - last[1] <= max_gap_samples:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(int(a), int(b)) for a, b in merged]


def _hysteresis_speech_mask(rms_values, high_threshold, low_threshold):

    rms_values = np.asarray(rms_values, dtype=np.float32)
    n = int(rms_values.size)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    high = rms_values >= float(high_threshold)
    low = rms_values >= float(low_threshold)
    mask = np.zeros((n,), dtype=bool)

    i = 0
    while i < n:
        if not high[i]:
            i += 1
            continue

        a = i
        while a > 0 and low[a - 1]:
            a -= 1

        b = i
        while b + 1 < n and low[b + 1]:
            b += 1

        mask[a:b + 1] = True
        i = b + 1

    return mask


def _split_long_segment_by_valleys(wav, segment, max_segment_seconds, target_segment_seconds, min_piece_seconds):

    start, end = int(segment[0]), int(segment[1])
    max_len = int(float(max_segment_seconds) * SAMPLE_RATE)
    target_len = int(float(target_segment_seconds) * SAMPLE_RATE)
    min_len = int(float(min_piece_seconds) * SAMPLE_RATE)

    duration = end - start
    if duration <= max_len:
        return [(start, end)]


    n_pieces = int(round(duration / max(target_len, 1)))
    n_pieces = max(2, n_pieces)
    n_pieces = min(n_pieces, max(2, duration // max(min_len, 1)))

    if n_pieces <= 1:
        return [(start, end)]

    frame_len = max(1, int(0.025 * SAMPLE_RATE))
    hop_len = max(1, int(0.010 * SAMPLE_RATE))
    rms_values, frame_starts_local = _frame_rms_with_starts(wav[start:end], frame_len, hop_len)
    rms_values = _moving_average(rms_values, 7)

    boundaries = [start]
    last_cut = start

    for k in range(1, n_pieces):
        ideal = start + int(round(k * duration / n_pieces))


        remaining_pieces = n_pieces - k
        search_a = max(last_cut + min_len, ideal - int(0.28 * SAMPLE_RATE))
        search_b = min(end - remaining_pieces * min_len, ideal + int(0.28 * SAMPLE_RATE))

        if search_b <= search_a:
            cut = min(end - remaining_pieces * min_len, max(last_cut + min_len, ideal))
        else:
            candidates = []
            for idx, fs in enumerate(frame_starts_local):
                global_pos = start + int(fs)
                if search_a <= global_pos <= search_b:
                    candidates.append((float(rms_values[idx]), global_pos))
            if candidates:
                _, cut = min(candidates, key=lambda t: t[0])
            else:
                cut = ideal

        cut = int(cut)
        if cut <= last_cut + min_len:
            cut = last_cut + min_len
        if end - cut < remaining_pieces * min_len:
            cut = end - remaining_pieces * min_len

        boundaries.append(cut)
        last_cut = cut

    boundaries.append(end)

    pieces = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a >= min_len:
            pieces.append((int(a), int(b)))

    return pieces if pieces else [(start, end)]


def segment_speech_by_energy(
    wav,
    frame_seconds=0.020,
    hop_seconds=0.010,
    silence_rms_threshold=0.003,
    relative_threshold=0.14,
    low_relative_threshold=0.07,
    min_speech_seconds=0.08,
    min_silence_seconds=0.12,
    padding_seconds=0.06,
    merge_gap_seconds=0.035,
    max_segment_seconds=1.20,
    target_segment_seconds=0.95,
    min_piece_seconds=0.16,
):

    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return []

    frame_len = max(1, int(frame_seconds * SAMPLE_RATE))
    hop_len = max(1, int(hop_seconds * SAMPLE_RATE))
    rms_values, starts = _frame_rms_with_starts(wav, frame_len, hop_len)

    if rms_values.size == 0:
        return []

    rms_smooth = _moving_average(rms_values, 5)
    speech_level = float(np.percentile(rms_smooth, 95))
    noise_level = float(np.percentile(rms_smooth, 20))

    if speech_level < silence_rms_threshold:
        return []

    # Hystérésis : un seuil haut pour trouver la parole, puis un seuil bas pour récupérer les débuts/fins de mots.
    high_threshold = noise_level + relative_threshold * max(speech_level - noise_level, 0.0)
    high_threshold = max(float(silence_rms_threshold), high_threshold, 0.045 * speech_level)

    low_threshold = noise_level + low_relative_threshold * max(speech_level - noise_level, 0.0)
    low_threshold = max(float(silence_rms_threshold) * 0.55, low_threshold, 0.020 * speech_level)
    low_threshold = min(low_threshold, high_threshold)

    speech_mask = _hysteresis_speech_mask(rms_smooth, high_threshold, low_threshold)

    min_silence_frames = max(1, int(min_silence_seconds / hop_seconds))
    min_speech_frames = max(1, int(min_speech_seconds / hop_seconds))

    speech_mask = _fill_short_silences(speech_mask, max_gap_frames=min_silence_frames)
    speech_mask = _remove_short_speech(speech_mask, min_speech_frames=min_speech_frames)

    padding_samples = int(padding_seconds * SAMPLE_RATE)
    segments = _segments_from_mask(speech_mask, starts, frame_len, wav.size, padding_samples)
    segments = _merge_close_segments(segments, int(merge_gap_seconds * SAMPLE_RATE))

    split_segments = []
    for seg in segments:
        split_segments.extend(_split_long_segment_by_valleys(
            wav,
            seg,
            max_segment_seconds=max_segment_seconds,
            target_segment_seconds=target_segment_seconds,
            min_piece_seconds=min_piece_seconds,
        ))

    #  ordre + suppression des segments ridiculement courts.
    min_len = int(min_piece_seconds * SAMPLE_RATE)
    split_segments = [(a, b) for a, b in sorted(split_segments) if b - a >= min_len]
    return split_segments
