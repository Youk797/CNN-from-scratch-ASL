def _levenshtein_distance(a, b):
    """Distance d'édition entre deux listes de mots."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # suppression
                dp[i][j - 1] + 1,          # insertion
                dp[i - 1][j - 1] + cost,   # substitution
            )

    return dp[n][m]

def _lcs_length(a, b):
    """Nombre de mots correctement retrouvés dans le bon ordre."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]

def score_word_sequence(expected_words, predicted_words):
    """
    Calcule un score entre la phrase attendue et la phrase reconnue.

    - score_sequence : pénalise les mots manquants, en trop et les substitutions.
    - recall_words   : proportion de mots attendus retrouvés dans le bon ordre.
    - precision_words: proportion des mots prédits qui sont corrects.
    """
    expected_words = [w.upper() for w in expected_words]
    predicted_words = [w.upper() for w in predicted_words]

    edit_dist = _levenshtein_distance(expected_words, predicted_words)
    denom = max(len(expected_words), len(predicted_words), 1)
    score_sequence = max(0.0, 1.0 - edit_dist / denom)

    good_ordered = _lcs_length(expected_words, predicted_words)
    recall_words = good_ordered / max(len(expected_words), 1)
    precision_words = good_ordered / max(len(predicted_words), 1) if predicted_words else 0.0

    if precision_words + recall_words > 0:
        f1_words = 2 * precision_words * recall_words / (precision_words + recall_words)
    else:
        f1_words = 0.0

    return {
        "score_sequence": float(score_sequence),
        "recall_words": float(recall_words),
        "precision_words": float(precision_words),
        "f1_words": float(f1_words),
        "correct_ordered_words": int(good_ordered),
        "expected_count": int(len(expected_words)),
        "predicted_count": int(len(predicted_words)),
        "edit_distance": int(edit_dist),
    }
