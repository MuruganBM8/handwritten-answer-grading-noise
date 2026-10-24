
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from features import extract_features
from noise import inject_noise


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "grading_models.joblib"
RESULT_DIR = ROOT / "results"

RATES = [0.00, 0.05, 0.10, 0.20, 0.30]
SEEDS = [11, 22, 33, 44, 55]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))

        previous = current

    return previous[-1]


def calculate_cer(original, corrupted):
    if not original:
        return 0.0 if not corrupted else 1.0

    return edit_distance(original, corrupted) / len(original)


def evaluate_predictions(y_true, predictions):
    predictions = np.clip(predictions, 0, 5)

    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    true_categories = np.rint(y_true * 2).astype(int)
    pred_categories = np.rint(predictions * 2).astype(int)

    qwk = cohen_kappa_score(
        true_categories,
        pred_categories,
        weights="quadratic",
    )

    return rmse, qwk


def main():
    test = pd.read_csv(DATA_DIR / "test.csv")
    saved = joblib.load(MODEL_PATH)

    strategies = {
        "ordinary": saved["ordinary"],
        "noise_aware": saved["noise_aware"],
    }

    original_answers = test["student_answer"].astype(str).tolist()
    y_test = test["score_avg"].to_numpy(dtype=float)

    records = []

    for seed in SEEDS:
        for rate in RATES:
            noisy_test = test.copy()

            noisy_answers = [
                inject_noise(
                    answer,
                    rate,
                    seed=seed * 10000 + i,
                )
                for i, answer in enumerate(original_answers)
            ]

            noisy_test["student_answer"] = noisy_answers

            actual_cer = np.mean([
                calculate_cer(original, corrupted)
                for original, corrupted in zip(
                    original_answers, noisy_answers
                )
            ])

            for strategy_name, bundle in strategies.items():
                X_test = extract_features(
                    noisy_test,
                    bundle["vectorizer"],
                )

                for model_name, model in bundle["models"].items():
                    predictions = model.predict(X_test)

                    rmse, qwk = evaluate_predictions(
                        y_test, predictions
                    )

                    records.append({
                        "seed": seed,
                        "training_strategy": strategy_name,
                        "model": model_name,
                        "target_cer": rate,
                        "actual_cer": actual_cer,
                        "rmse": rmse,
                        "qwk": qwk,
                    })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(records)
    results.to_csv(
        RESULT_DIR / "robustness_runs.csv",
        index=False,
    )

    summary = (
        results.groupby(
            ["training_strategy", "model", "target_cer"]
        )
        .agg(
            mean_actual_cer=("actual_cer", "mean"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_qwk=("qwk", "mean"),
            std_qwk=("qwk", "std"),
        )
        .reset_index()
    )

    summary.to_csv(
        RESULT_DIR / "robustness_summary.csv",
        index=False,
    )

    print("\nRobustness comparison:")
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print(RESULT_DIR / "robustness_runs.csv")
    print(RESULT_DIR / "robustness_summary.csv")


if __name__ == "__main__":
    main()
