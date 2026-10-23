
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

INPUT = DATA_DIR / "clean_answers.csv"
TRAIN_OUTPUT = DATA_DIR / "train.csv"
TEST_OUTPUT = DATA_DIR / "test.csv"


def main():
    df = pd.read_csv(INPUT)

    # Every row belonging to a question stays in one split.
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

    train_questions = set(train["question"])
    test_questions = set(test["question"])
    overlap = train_questions & test_questions

    if overlap:
        raise RuntimeError(
            f"Question leakage detected: {overlap}"
        )

    if train.empty or test.empty:
        raise RuntimeError("Training or test split is empty.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUTPUT, index=False)
    test.to_csv(TEST_OUTPUT, index=False)

    print("Dataset split completed")
    print("-" * 35)
    print(f"Total responses: {len(df)}")
    print(f"Training responses: {len(train)}")
    print(f"Testing responses: {len(test)}")
    print(f"Training questions: {len(train_questions)}")
    print(f"Testing questions: {len(test_questions)}")
    print(f"Question overlap: {len(overlap)}")

    print("\nTraining score distribution:")
    print(train["score_avg"].value_counts().sort_index())

    print("\nTesting score distribution:")
    print(test["score_avg"].value_counts().sort_index())

    print("\nSaved:")
    print(TRAIN_OUTPUT)
    print(TEST_OUTPUT)


if __name__ == "__main__":
    main()
