from pathlib import Path
import joblib
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features import fit_vectorizer, extract_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA / "train.csv")

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
    print(f"Training {name}...")
    model.fit(X_train, y_train)

bundle = {
    "vectorizer": vectorizer,
    "models": models,
    "feature_names": [
        "tfidf_cosine",
        "reference_word_overlap",
        "jaccard_similarity",
        "answer_length_ratio",
        "word_count_difference",
        "character_similarity",
    ],
}

output = MODEL_DIR / "grading_models.joblib"
joblib.dump(bundle, output)

print(f"\nModels saved to: {output}")
print("Training complete.")