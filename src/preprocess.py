
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "mohler_open_ended.csv"
OUTPUT = ROOT / "data" / "processed" / "clean_answers.csv"

REQUIRED_COLUMNS = [
    "question",
    "instructor_answer",
    "student_answer",
    "score_avg",
]


def clean_text(value):
    """Normalize whitespace without changing word order or case."""
    value = str(value)
    return re.sub(r"\s+", " ", value).strip()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)
    print("Original dataset shape:", df.shape)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print("\nMissing values before cleaning:")
    print(df[REQUIRED_COLUMNS].isna().sum())

    original_rows = len(df)

    # Convert scores to numeric; invalid values become NaN.
    df["score_avg"] = pd.to_numeric(
        df["score_avg"], errors="coerce"
    )

    # Drop records missing essential fields.
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    # Normalize text fields.
    for column in [
        "question",
        "instructor_answer",
        "student_answer",
    ]:
        df[column] = df[column].map(clean_text)

    # Remove empty questions, references, and student answers.
    for column in [
        "question",
        "instructor_answer",
        "student_answer",
    ]:
        df = df[df[column].str.len() > 0]

    if df.empty:
        raise ValueError("No valid records remain after cleaning.")

    if not df["score_avg"].map(
        lambda x: pd.notna(x) and float("-inf") < x < float("inf")
    ).all():
        raise ValueError("Non-finite scores detected.")

    # Report the observed score range instead of assuming it.
    print("\nScore statistics:")
    print(df["score_avg"].describe())

    print("\nObserved score values:")
    print(sorted(df["score_avg"].unique()))

    print("\nUnique questions:", df["question"].nunique())
    print("Duplicate complete rows:", df.duplicated().sum())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("\nRows before cleaning:", original_rows)
    print("Rows after cleaning:", len(df))
    print("Rows removed:", original_rows - len(df))
    print("Saved cleaned dataset to:", OUTPUT)


if __name__ == "__main__":
    main()
