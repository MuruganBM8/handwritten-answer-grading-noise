
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features import FEATURE_NAMES, fit_vectorizer, extract_features
from noise import inject_noise


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "grading_models.joblib"

AUGMENTATION_RATES = [0.05, 0.10, 0.20]


def create_models():
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "SVR": make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=10, epsilon=0.1),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


def make_noise_aware_data(train):
    variants = [train.copy()]

    for rate in AUGMENTATION_RATES:
        noisy = train.copy()

        noisy["student_answer"] = [
            inject_noise(
                answer,
                rate,
                seed=100000 * int(rate * 100) + i,
            )
            for i, answer in enumerate(
                train["student_answer"].astype(str)
            )
        ]

        variants.append(noisy)

    return pd.concat(variants, ignore_index=True)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")

    # Create augmented training examples using training data only.
    augmented_train = make_noise_aware_data(train)

    # Fit ONE vectorizer and reuse it for both strategies.
    vectorizer = fit_vectorizer(augmented_train)

    X_clean = extract_features(train, vectorizer)
    y_clean = train["score_avg"].to_numpy(dtype=float)

    X_augmented = extract_features(augmented_train, vectorizer)
    y_augmented = augmented_train["score_avg"].to_numpy(dtype=float)

    bundles = {}

    for strategy, X, y in [
        ("ordinary", X_clean, y_clean),
        ("noise_aware", X_augmented, y_augmented),
    ]:
        models = create_models()

        print(f"\nTraining strategy: {strategy}")
        print("Training examples:", len(y))

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X, y)

        bundles[strategy] = {
            "vectorizer": vectorizer,
            "models": models,
            "feature_names": FEATURE_NAMES,
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            **bundles,
            "augmentation_rates": AUGMENTATION_RATES,
        },
        MODEL_PATH,
    )

    print("\nBoth strategies trained with the same vectorizer.")
    print("Saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
