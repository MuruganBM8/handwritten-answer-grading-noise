
import random
import re

from noise import inject_noise


# Common function words that are less important for answer meaning.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "from",
    "with", "as", "it", "its", "this", "that", "these",
    "those", "they", "them", "their", "we", "you", "he",
    "she", "his", "her", "do", "does", "did", "has", "have",
    "had", "can", "could", "will", "would", "should"
}

NEGATIONS = {"not", "no", "never", "without", "neither", "nor"}


def corrupt_content_words(text, cer, seed=42):
    """
    Corrupt characters specifically inside content words.

    cer controls the approximate number of character edits relative
    to the original text length. Actual CER is measured separately.
    """
    if not text:
        return text

    rng = random.Random(seed)
    matches = list(re.finditer(r"[A-Za-z]+", text))

    candidates = [
        i for i, match in enumerate(matches)
        if len(match.group()) >= 3
        and match.group().lower() not in STOPWORDS
    ]

    if not candidates:
        return text

    words = [match.group() for match in matches]
    edit_budget = round(cer * len(text))

    for _ in range(edit_budget):
        idx = rng.choice(candidates)
        word = words[idx]

        # Replace one letter with a different letter.
        pos = rng.randrange(len(word))
        original_char = word[pos]
        replacement = rng.choice(
            [ch for ch in "abcdefghijklmnopqrstuvwxyz"
             if ch != original_char.lower()]
        )

        if original_char.isupper():
            replacement = replacement.upper()

        words[idx] = word[:pos] + replacement + word[pos + 1:]

    # Replace matched words from right to left so original positions
    # remain valid even if future corruption methods change word length.
    result = text
    for match, word in reversed(list(zip(matches, words))):
        result = result[:match.start()] + word + result[match.end():]

    return result


def corrupt_negation(text):
    """
    Stress test: remove or reverse one negation expression if present.
    Returns (corrupted_text, changed).
    """
    patterns = [
        (r"\bwithout\b", "with"),
        (r"\bneither\b", "either"),
        (r"\bnor\b", "or"),
        (r"\bnever\b", ""),
        (r"\bnot\b", ""),
        (r"\bno\b", ""),
    ]

    for pattern, replacement in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            corrupted = (
                text[:match.start()]
                + replacement
                + text[match.end():]
            )
            corrupted = re.sub(r"\s+", " ", corrupted).strip()
            return corrupted, True

    return text, False


def corrupt_text(text, condition, level, seed=42):
    """
    Apply one corruption condition.

    random: existing general-purpose noise simulator
    content: character edits focused on content words
    """
    if condition == "random":
        return inject_noise(text, level, seed=seed)

    if condition == "content":
        return corrupt_content_words(text, level, seed=seed)

    raise ValueError(f"Unknown corruption condition: {condition}")
