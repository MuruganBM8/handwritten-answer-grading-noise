from pathlib import Path
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Run the evaluation script first.
subprocess.run(
    [sys.executable, str(ROOT / "src" / "evaluate.py")],
    check=True,
)

df = pd.read_csv(RESULTS / "noise_evaluation.csv")

# Plot 1: RMSE vs measured CER.
for model_name, group in df.groupby("model"):
    group = group.sort_values("actual_CER")
    plt.plot(
        group["actual_CER"],
        group["RMSE"],
        marker="o",
        label=model_name,
    )

plt.xlabel("Measured Character Error Rate")
plt.ylabel("RMSE (lower is better)")
plt.title("Grading Error vs Recognition Noise")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "rmse_vs_cer.png", dpi=200)
plt.show()
plt.close()

# Plot 2: QWK vs measured CER.
for model_name, group in df.groupby("model"):
    group = group.sort_values("actual_CER")
    plt.plot(
        group["actual_CER"],
        group["QWK"],
        marker="o",
        label=model_name,
    )

plt.xlabel("Measured Character Error Rate")
plt.ylabel("Quadratic Weighted Kappa (higher is better)")
plt.title("Grading Agreement vs Recognition Noise")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "qwk_vs_cer.png", dpi=200)
plt.show()
plt.close()

print("\nPlots saved:")
print(RESULTS / "rmse_vs_cer.png")
print(RESULTS / "qwk_vs_cer.png")