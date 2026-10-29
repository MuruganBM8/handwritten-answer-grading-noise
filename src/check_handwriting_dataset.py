
from datasets import load_dataset

dataset = load_dataset("Teklia/IAM-line")

print(dataset)
print("\nFirst example:")
print("Ground truth:", dataset["test"][0]["text"])
print("Image size:", dataset["test"][0]["image"].size)

# Save 50 test examples locally for the recognition experiment.
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
image_dir = root / "data" / "handwriting" / "images"
image_dir.mkdir(parents=True, exist_ok=True)

rows = []
sample = dataset["test"].select(range(min(300, len(dataset["test"]))))

for i, item in enumerate(sample):
    filename = f"line_{i:03d}.png"
    item["image"].convert("RGB").save(image_dir / filename)

    rows.append({
        "image": filename,
        "ground_truth": item["text"]
    })

labels_path = root / "data" / "handwriting" / "labels.csv"
pd.DataFrame(rows).to_csv(labels_path, index=False)

print(f"\nSaved {len(rows)} images to {image_dir}")
print(f"Saved labels to {labels_path}")
