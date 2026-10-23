from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "mohler_open_ended.csv"
OUTPUT = ROOT / "data" / "processed" / "clean_answers.csv"

df = pd.read_csv(INPUT)

required = [
    "question",
    "instructor_answer",
    "student_answer",
    "score_avg",
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Preserve a copy of the original data.
df = df.copy()

# Remove rows without essential text or target values.
df = df.dropna(subset=required)

# Normalize whitespace without changing the meaning of answers.
def clean_text(value):
    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

for column in ["question", "instructor_answer", "student_answer"]:
    df[column] = df[column].map(clean_text)

df["score_avg"] = pd.to_numeric(df["score_avg"], errors="coerce")
df = df.dropna(subset=["score_avg"])

# Keep only usable non-empty answers and references.
df = df[
    (df["student_answer"].str.len() > 0)
    & (df["instructor_answer"].str.len() > 0)
].copy()

if df.empty:
    raise ValueError("No usable rows remain after cleaning.")

if not df["score_avg"].between(0, 5).all():
    raise ValueError(
        "Some scores are outside 0-5. Inspect the score scale before proceeding."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Clean dataset shape:", df.shape)
print("Unique questions:", df["question"].nunique())
print("Score distribution:")
print(df["score_avg"].value_counts().sort_index())
print(f"\nSaved to: {OUTPUT}")