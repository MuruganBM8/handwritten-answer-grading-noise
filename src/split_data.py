from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
df = pd.read_csv(DATA / "clean_answers.csv")

if df["question"].nunique() < 2:
    raise ValueError("Need at least two distinct questions to split by question.")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["question"])
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

train.to_csv(DATA / "train.csv", index=False)
test.to_csv(DATA / "test.csv", index=False)

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training questions:", train["question"].nunique())
print("Testing questions:", test["question"].nunique())
print(
    "Question overlap:",
    set(train["question"]) & set(test["question"])
)