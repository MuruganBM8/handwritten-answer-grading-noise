
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading Mohler ASAG dataset...")

dataset = load_dataset(
    "nkazi/MohlerASAG",
    name="raw",
    split="open_ended"
)

df = dataset.to_pandas()

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst five rows:")
print(df.head())

output_path = DATA_DIR / "mohler_open_ended.csv"
df.to_csv(output_path, index=False)

print(f"\nDataset saved to: {output_path}")
