
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

from noise import inject_noise


ROOT = Path(__file__).resolve().parents[1]

LABELS_PATH = ROOT / "data" / "handwriting" / "labels.csv"
TROCR_PATH = ROOT / "results" / "trocr_predictions.csv"
OUTPUT_PATH = ROOT / "results" / "noise_comparison.csv"


def error_counts(reference, prediction):
    substitutions = deletions = insertions = 0

    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(
        reference, prediction
    ):
        ref_len = i2 - i1
        pred_len = j2 - j1

        if tag == "replace":
            substitutions += min(ref_len, pred_len)
            deletions += max(0, ref_len - pred_len)
            insertions += max(0, pred_len - ref_len)

        elif tag == "delete":
            deletions += ref_len

        elif tag == "insert":
            insertions += pred_len

    return substitutions, deletions, insertions


def evaluate(reference, prediction):
    distance = Levenshtein.distance(reference, prediction)
    cer = distance / max(len(reference), 1)
    subs, dels, ins = error_counts(reference, prediction)

    return {
        "cer": cer,
        "edit_distance": distance,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "reference_characters": len(reference),
    }


def main():
    labels = pd.read_csv(LABELS_PATH)
    trocr = pd.read_csv(TROCR_PATH)

    df = labels.merge(
        trocr[["image", "recognized_text"]],
        on="image",
        how="inner",
        validate="one_to_one",
    )

    if df.empty:
        raise ValueError("No matching images found in the two CSV files.")

    rows = []

    for row in df.itertuples(index=False):
        reference = str(row.ground_truth)
        actual_prediction = str(row.recognized_text)

        actual = evaluate(reference, actual_prediction)

        rows.append({
            "image": row.image,
            "method": "real_trocr",
            "target_cer": None,
            **actual,
        })

        for target in [0.05, 0.10, 0.20, 0.30]:
            synthetic_prediction = inject_noise(
                reference,
                cer=target,
                seed=42 + sum(map(ord, row.image)),
            )

            synthetic = evaluate(reference, synthetic_prediction)

            rows.append({
                "image": row.image,
                "method": "synthetic",
                "target_cer": target,
                **synthetic,
            })

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Real vs synthetic noise ---")

    for (method, target), group in results.groupby(
        ["method", "target_cer"], dropna=False
    ):
        total_errors = group["edit_distance"].sum()
        total_chars = group["reference_characters"].sum()

        print(f"\nMethod: {method}, target CER: {target}")
        print(f"Samples: {len(group)}")
        print(f"Overall CER: {total_errors / max(total_chars, 1):.4f}")
        print(f"Mean line CER: {group['cer'].mean():.4f}")
        print(f"Substitutions: {group['substitutions'].sum()}")
        print(f"Deletions: {group['deletions'].sum()}")
        print(f"Insertions: {group['insertions'].sum()}")

    print(f"\nSaved comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
