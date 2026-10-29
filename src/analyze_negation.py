
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error

from features import extract_features
from targeted_noise import corrupt_negation


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "grading_models.joblib"
RESULT_DIR = ROOT / "results"


def score_categories(values):
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0, 5)
    return np.clip(np.rint(values * 2), 0, 10).astype(int)


def metrics(y_true, predictions):
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

    vectorizer = bundle["ordinary"]["vectorizer"]

    # Keep only answers whose text is actually changed.
    changed_indices = []
    corrupted_answers = []

    for i, answer in enumerate(test["student_answer"].astype(str)):
        corrupted, changed = corrupt_negation(answer)

        if changed:
            changed_indices.append(i)
            corrupted_answers.append(corrupted)

    if not changed_indices:
        print("No test answers contained a supported negation expression.")
        return

    subset = test.iloc[changed_indices].copy().reset_index(drop=True)
    corrupted_subset = subset.copy()
    corrupted_subset["student_answer"] = corrupted_answers

    X_clean = extract_features(subset, vectorizer)
    X_corrupted = extract_features(corrupted_subset, vectorizer)
    y_true = subset["score_avg"].to_numpy(dtype=float)

    details = []
    summary = []

    for strategy in ["ordinary", "noise_aware"]:
        for model_name, model in bundle[strategy]["models"].items():
            pred_clean = model.predict(X_clean)
            pred_corrupted = model.predict(X_corrupted)

            rmse_clean, qwk_clean = metrics(y_true, pred_clean)
            rmse_corrupted, qwk_corrupted = metrics(
                y_true, pred_corrupted
            )

            for i in range(len(subset)):
                details.append({
                    "test_row": changed_indices[i],
                    "question": subset.iloc[i]["question"],
                    "student_answer_original":
                        subset.iloc[i]["student_answer"],
                    "student_answer_negation_changed":
                        corrupted_answers[i],
                    "true_score": y_true[i],
                    "training_strategy": strategy,
                    "model": model_name,
                    "prediction_clean": pred_clean[i],
                    "prediction_corrupted": pred_corrupted[i],
                    "prediction_change":
                        pred_corrupted[i] - pred_clean[i],
                    "absolute_prediction_change":
                        abs(pred_corrupted[i] - pred_clean[i]),
                    "absolute_error_clean":
                        abs(pred_clean[i] - y_true[i]),
                    "absolute_error_corrupted":
                        abs(pred_corrupted[i] - y_true[i])
                })

            summary.append({
                "training_strategy": strategy,
                "model": model_name,
                "number_of_answers": len(subset),
                "rmse_clean": rmse_clean,
                "rmse_negation_changed": rmse_corrupted,
                "rmse_change": rmse_corrupted - rmse_clean,
                "qwk_clean": qwk_clean,
                "qwk_negation_changed": qwk_corrupted,
                "qwk_change": qwk_corrupted - qwk_clean,
                "mean_absolute_prediction_change": np.mean(
                    np.abs(pred_corrupted - pred_clean)
                ),
                "answers_with_changed_prediction": int(
                    np.sum(
                        ~np.isclose(pred_clean, pred_corrupted)
                    )
                )
            })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    detail_df = pd.DataFrame(details)
    summary_df = pd.DataFrame(summary)

    detail_df.to_csv(
        RESULT_DIR / "negation_per_answer.csv",
        index=False
    )
    summary_df.to_csv(
        RESULT_DIR / "negation_focused_summary.csv",
        index=False
    )

    print(f"Negation-affected answers: {len(subset)}")
    print("\nModel comparison on those answers:")
    print(summary_df.round(4).to_string(index=False))

    print("\nPer-answer predictions saved to:")
    print("results/negation_per_answer.csv")
    print("Summary saved to:")
    print("results/negation_focused_summary.csv")


if __name__ == "__main__":
    main()
