from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    cohen_kappa_score,
    mean_squared_error,
)

from features import extract_features
from noise import inject_noise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

test = pd.read_csv(DATA / "test.csv")
bundle = joblib.load(MODEL_DIR / "grading_models.joblib")

vectorizer = bundle["vectorizer"]
models = bundle["models"]


def edit_distance(a, b):
    """Levenshtein edit distance using O(len(b)) memory."""
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )

        previous = current

    return previous[-1]


def calculate_cer(original, corrupted):
    if not original:
        return 0.0 if not corrupted else 1.0

    return edit_distance(original, corrupted) / len(original)


def evaluate_predictions(y_true, y_pred):
    # Marks are on a 0-5 scale.
    y_pred = np.clip(y_pred, 0, 5)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Compare ordinal half-mark categories.
    true_categories = np.rint(y_true * 2).astype(int)
    pred_categories = np.rint(y_pred * 2).astype(int)

    qwk = cohen_kappa_score(
        true_categories,
        pred_categories,
        weights="quadratic",
    )

    return rmse, qwk


rates = [0.00, 0.05, 0.10, 0.20, 0.30]
records = []

original_answers = test["student_answer"].astype(str).tolist()

for rate in rates:
    noisy_test = test.copy()

    noisy_answers = [
        inject_noise(answer, rate, seed=1000 + i)
        for i, answer in enumerate(original_answers)
    ]

    noisy_test["student_answer"] = noisy_answers

    actual_cer = np.mean([
        calculate_cer(original, corrupted)
        for original, corrupted
        in zip(original_answers, noisy_answers)
    ])

    X_test = extract_features(noisy_test, vectorizer)
    y_test = noisy_test["score_avg"].to_numpy(dtype=float)

    for name, model in models.items():
        predictions = model.predict(X_test)

        rmse, qwk = evaluate_predictions(y_test, predictions)

        records.append({
            "model": name,
            "target_CER": rate,
            "actual_CER": actual_cer,
            "RMSE": rmse,
            "QWK": qwk,
        })

        print(
            f"{name:12s} | target CER={rate:.0%} "
            f"| actual CER={actual_cer:.3f} "
            f"| RMSE={rmse:.4f} | QWK={qwk:.4f}"
        )

results = pd.DataFrame(records)
results.to_csv(RESULT_DIR / "noise_evaluation.csv", index=False)

print("\nResults saved to results/noise_evaluation.csv")