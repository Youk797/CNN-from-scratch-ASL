BASE_PHRASE_TEMPLATES = [
    ["go", "left"],
    ["go", "right"],
    ["go", "forward"],
    ["go", "backward"],
    ["go", "up"],
    ["yes", "go", "left"],
    ["yes", "go", "right"],
    ["yes", "go", "forward"],
    ["yes", "go", "backward"],
    ["no", "stop"],
    ["follow", "cat"],
    ["follow", "dog"],
    ["follow", "bird"],
    ["happy", "cat"],
    ["happy", "dog"],
    ["happy", "bird"],
    ["wow", "happy", "cat"],
    ["wow", "happy", "dog"],
    ["wow", "happy", "bird"],
    ["one", "cat"],
    ["one", "dog"],
    ["one", "bird"],
    ["one", "house"],
    ["one", "bed"],
    ["one", "two", "three"],
    ["four", "five", "six"],
    ["seven", "eight", "nine"],
    ["zero", "one", "two"],
]


def _build_candidate_phrases(available_words):
    """
    Construit beaucoup de phrases possibles à partir du vocabulaire disponible.

    Comme le modèle ne connaît que certains mots, on évite les mots grammaticaux
    inconnus comme "the", "a", "to", etc. Les phrases restent donc simples,
    mais elles gardent une structure compréhensible : commandes, comptage,
    petits groupes nom/adjectif, suites de nombres.
    """
    available = set(available_words)
    candidates = []
    seen = set()

    def add(words):
        words = [str(w).lower() for w in words]
        if not words:
            return
        if not all(w in available for w in words):
            return
        key = tuple(words)
        if key in seen:
            return
        seen.add(key)
        candidates.append(words)

    directions = [w for w in ["left", "right", "forward", "backward", "up"] if w in available]
    animals = [w for w in ["cat", "dog", "bird"] if w in available]
    objects = [w for w in ["house", "bed"] if w in available]
    nouns = animals + objects
    numbers = [w for w in ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"] if w in available]

    for template in BASE_PHRASE_TEMPLATES:
        add(template)

    for direction in directions:
        add(["go", direction])
        add(["yes", "go", direction])
        add(["no", "go", direction])

    add(["stop"])
    add(["no", "stop"])
    add(["yes", "go"])
    add(["go", "on"])
    add(["go", "off"])

    for noun in nouns:
        add(["happy", noun])
        add(["wow", "happy", noun])
        add(["follow", noun])
        add(["yes", "follow", noun])
        add(["no", "follow", noun])
        add(["one", noun])
        add(["two", noun])
        add(["three", noun])


    for animal in animals:
        for direction in directions:
            add([animal, "go", direction])
            add(["follow", animal, direction])

    for number in numbers:
        for noun in nouns:
            add([number, noun])
            add(["happy", number, noun])
            add(["follow", number, noun])

    if len(numbers) >= 2:
        for a in numbers:
            for b in numbers:
                add([a, b])

    if len(numbers) >= 3:
        for a in numbers:
            for b in numbers:
                for c in numbers:
                    add([a, b, c])

    for number in numbers:
        for direction in directions:
            add([number, "go", direction])
            add(["go", direction, number])

    if not candidates:
        raise RuntimeError(
            "Impossible de générer des phrases : aucun template ne correspond "
            "aux mots disponibles dans data/."
        )

    return candidates


def _choose_phrases(available_words, num_phrases, rng):
    """
    Choisit num_phrases phrases. Si on a assez de phrases candidates,
    on évite les doublons de texte. Sinon, on recycle les phrases dans un
    nouvel ordre aléatoire.
    """
    candidates = _build_candidate_phrases(available_words)
    rng.shuffle(candidates)

    selected = []
    while len(selected) < num_phrases:
        batch = candidates.copy()
        rng.shuffle(batch)
        selected.extend(batch)

    return selected[:num_phrases], len(candidates)
