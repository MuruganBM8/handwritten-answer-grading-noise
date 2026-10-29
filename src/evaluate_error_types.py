
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error

from features import extract_features
from targeted_noise import corrupt_text, corrupt_negation


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "grading_models.joblib"
RESULT_DIR = ROOT / "results"

LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
SEEDS = [11, 22, 33, 44, 55]


def edit_distance(a, b):
    """Levenshtein edit distance using dynamic programming."""
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,                  # insertion
                    previous[j] + 1,                  # deletion
                    previous[j - 1] + (char_a != char_b)
                )
            )

        previous = current

    return previous[-1]


def actual_cer(originals, corrupted):
    total_errors = 0
    total_chars = 0

    for original, noisy in zip(originals, corrupted):
        total_errors += edit_distance(original, noisy)
        total_chars += len(original)

    return total_errors / max(total_chars, 1)



def score_categories(values):
    """Convert scores from 0–5 into integer half-point categories 0–10."""
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0, 5)
    return np.clip(np.rint(values * 2), 0, 10).astype(int)


def calculate_metrics(y_true, predictions):
    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    qwk = cohen_kappa_score(
        score_categories(y_true),
        score_categories(predictions),
        weights="quadratic",
        labels=list(range(11))
    )

    return rmse, qwk



def main():
    test = pd.read_csv(DATA_DIR / "test.csv")
    bundle = joblib.load(MODEL_PATH)

    # Use the shared vectorizer saved during training.
    vectorizer = bundle["ordinary"]["vectorizer"]

    # Evaluate both training strategies and all available models.
    records = []

    conditions = ["random", "content"]

    for condition in conditions:
        for level in LEVELS:
            for seed in SEEDS:
                noisy_answers = [
                    corrupt_text(
                        str(answer),
                        condition=condition,
                        level=level,
                        seed=seed * 100000 + row_idx
                    )
                    for row_idx, answer in enumerate(
                        test["student_answer"]
                    )
                ]

                noisy_test = test.copy()
                noisy_test["student_answer"] = noisy_answers

                cer = actual_cer(
                    test["student_answer"].astype(str).tolist(),
                    noisy_answers
                )

                X = extract_features(noisy_test, vectorizer)
                y_true = test["score_avg"].to_numpy(dtype=float)

                for strategy in ["ordinary", "noise_aware"]:
                    models = bundle[strategy]["models"]

                    for model_name, model in models.items():
                        predictions = model.predict(X)
                        rmse, qwk = calculate_metrics(
                            y_true, predictions
                        )

                        records.append({
                            "condition": condition,
                            "level": level,
                            "seed": seed,
                            "training_strategy": strategy,
                            "model": model_name,
                            "actual_cer": cer,
                            "rmse": rmse,
                            "qwk": qwk
                        })

    # Negation stress test: corrupt the first available negation
    # in each answer, if present. This is not a CER-controlled test.
    negated_answers = []
    changed_count = 0

    for answer in test["student_answer"].astype(str):
        corrupted, changed = corrupt_negation(answer)
        negated_answers.append(corrupted)
        changed_count += int(changed)

    negation_test = test.copy()
    negation_test["student_answer"] = negated_answers

    negation_cer = actual_cer(
        test["student_answer"].astype(str).tolist(),
        negated_answers
    )

    X_negation = extract_features(negation_test, vectorizer)
    y_true = test["score_avg"].to_numpy(dtype=float)

    for strategy in ["ordinary", "noise_aware"]:
        for model_name, model in bundle[strategy]["models"].items():
            predictions = model.predict(X_negation)
            rmse, qwk = calculate_metrics(y_true, predictions)

            records.append({
                "condition": "negation_stress",
                "level": np.nan,
                "seed": -1,
                "training_strategy": strategy,
                "model": model_name,
                "actual_cer": negation_cer,
                "rmse": rmse,
                "qwk": qwk
            })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame(records)
    runs.to_csv(RESULT_DIR / "error_type_runs.csv", index=False)

    summary = (
        runs.groupby(
            ["condition", "level", "training_strategy", "model"],
            dropna=False
        )
        .agg(
            mean_actual_cer=("actual_cer", "mean"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_qwk=("qwk", "mean"),
            std_qwk=("qwk", "std"),
            runs=("rmse", "count")
        )
        .reset_index()
    )

    summary.to_csv(
        RESULT_DIR / "error_type_summary.csv",
        index=False
    )

    print(f"Test answers: {len(test)}")
    print(
        f"Answers changed by negation stress test: "
        f"{changed_count}/{len(test)}"
    )
    print(f"Negation stress-test actual CER: {negation_cer:.4f}")
    print("\nResults saved:")
    print("results/error_type_runs.csv")
    print("results/error_type_summary.csv")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
