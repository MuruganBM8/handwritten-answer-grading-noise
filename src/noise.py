import random
import string


def inject_noise(text, cer, seed=42):
    """
    Inject approximately cer * len(text) character edits.

    cer: target character error rate, e.g. 0.10 for 10%.
    """
    if not 0 <= cer <= 1:
        raise ValueError("CER must be between 0 and 1.")

    if not text or cer == 0:
        return text

    rng = random.Random(seed)
    chars = list(text)
    edits = round(cer * len(text))
    alphabet = string.ascii_lowercase

    for _ in range(edits):
        operation = rng.choice(
            ["substitute", "delete", "insert"]
        )

        if operation == "substitute" and chars:
            pos = rng.randrange(len(chars))
            old = chars[pos]
            choices = [c for c in alphabet if c != old.lower()]
            replacement = rng.choice(choices)

            if old.isupper():
                replacement = replacement.upper()

            chars[pos] = replacement

        elif operation == "delete" and chars:
            pos = rng.randrange(len(chars))
            chars.pop(pos)

        else:
            pos = rng.randrange(len(chars) + 1)
            chars.insert(pos, rng.choice(alphabet))

    return "".join(chars)