
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features import FEATURE_NAMES, fit_vectorizer, extract_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "train.csv"
MODEL_PATH = MODEL_DIR / "grading_models.joblib"


def main():
    train = pd.read_csv(TRAIN_PATH)

    vectorizer = fit_vectorizer(train)
    X_train = extract_features(train, vectorizer)
    y_train = train["score_avg"].to_numpy(dtype=float)

    models = {
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

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        print("Training completed.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "vectorizer": vectorizer,
        "models": models,
        "feature_names": FEATURE_NAMES,
    }

    joblib.dump(bundle, MODEL_PATH)

    print("\nTraining complete.")
    print("Training rows:", len(train))
    print("Number of features:", X_train.shape[1])
    print("Saved models to:", MODEL_PATH)


if __name__ == "__main__":
    main()
