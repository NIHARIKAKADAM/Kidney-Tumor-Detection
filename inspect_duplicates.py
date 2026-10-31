
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

report_path = Path("near_duplicate_report.json")

with report_path.open(encoding="utf-8") as file:
    pairs = json.load(file)

if not pairs:
    print("No potential near-duplicate pairs found.")
    raise SystemExit(0)

# Show examples with the smallest perceptual-hash distances.
pairs = sorted(pairs, key=lambda pair: int(pair["distance"]))

# Select up to five different image pairs.
selected = []
seen = set()

for pair in pairs:
    key = (pair["image1"], pair["image2"])

    if key not in seen:
        selected.append(pair)
        seen.add(key)

    if len(selected) == 5:
        break

fig, axes = plt.subplots(
    len(selected), 2,
    figsize=(10, 3 * len(selected)),
    squeeze=False
)

for row, pair in enumerate(selected):
    for col, field in enumerate(["image1", "image2"]):
        path = Path(pair[field])

        with Image.open(path) as image:
            axes[row, col].imshow(image.convert("RGB"))

        axes[row, col].set_title(
            f"{path.name}\n"
            f"pHash distance: {pair['distance']}",
            fontsize=10
        )
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("duplicate_comparison.png", dpi=150)
plt.show()

print("Saved: duplicate_comparison.png")
