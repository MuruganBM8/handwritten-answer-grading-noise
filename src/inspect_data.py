
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "raw" / "mohler_open_ended.csv"

df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nScore distribution:")
print(df["score_avg"].describe())

print("\nUnique questions:", df["question"].nunique())

print("\nSample record:")
print(df.iloc[0].to_dict())
