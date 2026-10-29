
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "handwriting" / "images"
LABELS_PATH = ROOT / "data" / "handwriting" / "labels.csv"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "microsoft/trocr-base-handwritten"


def count_edit_types(reference, prediction):
    """
    Count approximate character substitutions, deletions,
    and insertions using Levenshtein alignment.
    """
    substitutions = 0
    deletions = 0
    insertions = 0

    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(
        reference, prediction
    ):
        ref_len = i2 - i1
        pred_len = j2 - j1

        if tag == "replace":
            substitutions += min(ref_len, pred_len)
            if ref_len > pred_len:
                deletions += ref_len - pred_len
            elif pred_len > ref_len:
                insertions += pred_len - ref_len

        elif tag == "delete":
            deletions += ref_len

        elif tag == "insert":
            insertions += pred_len

    return substitutions, deletions, insertions


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    labels = pd.read_csv(LABELS_PATH)

    required_columns = {"image", "ground_truth"}
    if not required_columns.issubset(labels.columns):
        raise ValueError(
            "labels.csv must contain image and ground_truth columns."
        )

    results = []

    for row in labels.itertuples(index=False):
        image_path = IMAGE_DIR / row.image
        reference = str(row.ground_truth)

        if not image_path.exists():
            print(f"Missing image, skipping: {image_path}")
            continue

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixel_values = processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=256
            )

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Keep spaces and punctuation: they can be OCR errors too.
        errors = Levenshtein.distance(reference, prediction)
        cer = errors / max(len(reference), 1)

        subs, dels, ins = count_edit_types(reference, prediction)

        results.append({
            "image": row.image,
            "ground_truth": reference,
            "recognized_text": prediction,
            "reference_characters": len(reference),
            "edit_distance": errors,
            "cer": cer,
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
        })

        print(f"\nImage: {row.image}")
        print(f"Ground truth: {reference}")
        print(f"Recognized:   {prediction}")
        print(f"CER: {cer:.3f}")

    if not results:
        print("No images were successfully processed.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "trocr_predictions.csv"

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    total_errors = result_df["edit_distance"].sum()
    total_chars = result_df["reference_characters"].sum()
    overall_cer = total_errors / max(total_chars, 1)

    print("\n--- Recognition evaluation ---")
    print(f"Images processed: {len(result_df)}")
    print(f"Overall character error rate: {overall_cer:.4f}")
    print(f"Mean per-image CER: {result_df['cer'].mean():.4f}")
    print(f"Substitutions: {result_df['substitutions'].sum()}")
    print(f"Deletions: {result_df['deletions'].sum()}")
    print(f"Insertions: {result_df['insertions'].sum()}")
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
