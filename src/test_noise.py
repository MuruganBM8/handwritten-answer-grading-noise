
from noise import inject_noise


examples = [
    "The chloride ion has a negative charge.",
    "The cell membrane controls movement of substances.",
    "Photosynthesis converts light energy into chemical energy.",
    "The solution is not stable under these conditions.",
]

for text in examples:
    print("\nOriginal:", text)

    for rate in [0.05, 0.10, 0.20]:
        corrupted = inject_noise(text, rate, seed=42)
        print(f"CER setting {rate:.0%}: {corrupted}")
