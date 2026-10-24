
import random
import string


# Illustrative confusion rules. These are hypotheses, not measured
# error frequencies from a particular handwriting recognizer.
CONFUSIONS = [
    ("cl", "d"),
    ("d", "cl"),
    ("rn", "m"),
    ("m", "rn"),
    ("vv", "w"),
    ("w", "vv"),
    ("li", "h"),
    ("h", "li"),
    ("0", "o"),
    ("o", "0"),
    ("1", "l"),
    ("l", "1"),
]

ALPHABET = string.ascii_lowercase


def preserve_case(original, replacement):
    """Preserve all-uppercase or initial-uppercase formatting."""
    if original.isupper():
        return replacement.upper()

    if original[:1].isupper():
        return replacement.capitalize()

    return replacement


def apply_one_error(text, rng):
    """Apply one randomly selected recognition-like error."""
    if not text:
        return text

    # Find occurrences of configured visual-confusion patterns.
    confusion_matches = []

    lower_text = text.lower()

    for source, target in CONFUSIONS:
        start = 0

        while True:
            index = lower_text.find(source, start)

            if index == -1:
                break

            confusion_matches.append((index, source, target))
            start = index + 1

    # Choose the error family. If a confusion is unavailable,
    # fall back to ordinary character noise.
    operation = rng.choices(
        ["confusion", "substitute", "delete", "insert", "boundary"],
        weights=[3, 3, 2, 2, 1],
        k=1,
    )[0]

    if operation == "confusion" and confusion_matches:
        index, source, target = rng.choice(confusion_matches)

        original = text[index:index + len(source)]
        replacement = preserve_case(original, target)

        return (
            text[:index]
            + replacement
            + text[index + len(source):]
        )

    # Character positions exclude whitespace for ordinary
    # substitutions and deletions.
    positions = [
        i for i, char in enumerate(text)
        if not char.isspace()
    ]

    if operation == "substitute" and positions:
        index = rng.choice(positions)
        old = text[index]

        if old.isalpha():
            choices = [
                c for c in ALPHABET
                if c != old.lower()
            ]
            new = rng.choice(choices)

            if old.isupper():
                new = new.upper()

        elif old.isdigit():
            new = rng.choice("0123456789".replace(old, ""))

        else:
            new = rng.choice(ALPHABET)

        return text[:index] + new + text[index + 1:]

    if operation == "delete" and positions:
        index = rng.choice(positions)
        return text[:index] + text[index + 1:]

    if operation == "insert":
        # Insert next to a character rather than at an arbitrary
        # position in the middle of a word boundary.
        if not positions:
            return text

        index = rng.choice(positions)
        insert_at = index + 1
        char = rng.choice(ALPHABET)

        return text[:insert_at] + char + text[insert_at:]

    if operation == "boundary":
        words = [
            (i, char) for i, char in enumerate(text)
            if char.isspace()
        ]

        # Either split a word or merge two adjacent words.
        if words and rng.random() < 0.5:
            index, _ = rng.choice(words)
            return text[:index] + text[index + 1:]

        word_positions = [
            i for i in range(1, len(text))
            if text[i - 1].isalpha()
            and text[i].isalpha()
        ]

        if word_positions:
            index = rng.choice(word_positions)
            return text[:index] + " " + text[index:]

    # Fallback for short text or unavailable operation types.
    if positions:
        index = rng.choice(positions)
        return text[:index] + text[index + 1:]

    return text


def inject_noise(text, cer, seed=42):
    """
    Apply approximately cer * len(text) recognition-like edits.

    The achieved character error rate may differ from cer because
    multi-character substitutions can require multiple edit-distance
    operations. Measure CER after corruption for every experiment.
    """
    if not 0 <= cer <= 1:
        raise ValueError("CER must be between 0 and 1.")

    if not text or cer == 0:
        return text

    rng = random.Random(seed)
    result = text

    edit_budget = round(cer * len(text))

    for _ in range(edit_budget):
        result = apply_one_error(result, rng)

    return result
