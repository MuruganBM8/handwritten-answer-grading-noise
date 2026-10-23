
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "noise_evaluation.csv")

for metric, ylabel, title, filename in [
    ("rmse", "RMSE (lower is better)",
     "Grading Error vs Recognition Noise", "rmse_vs_cer.png"),
    ("qwk", "Quadratic Weighted Kappa (higher is better)",
     "Grading Agreement vs Recognition Noise", "qwk_vs_cer.png"),
]:
    for model_name, group in df.groupby("model"):
        group = group.sort_values("actual_cer")
        plt.plot(
            group["actual_cer"],
            group[metric],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Measured Character Error Rate")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=200)
    plt.show()
    plt.close()

print("Plots saved in:", RESULTS)
