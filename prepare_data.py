
from pathlib import Path
from collections import Counter
import json

from sklearn.model_selection import train_test_split

DATASET = Path("dataset")
OUTPUT = Path("splits.json")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

images = []
labels = []

for category in ["Normal", "Tumor"]:
    folder = DATASET / category

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )

    print(f"{category}: {len(files)} images")

    for path in files:
        images.append(path.as_posix())
        labels.append(category)

if not images:
    raise RuntimeError("No images found.")

# 70% training, 15% validation, 15% testing
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels,
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels,
)

splits = {
    "train": [
        {"path": p, "label": l}
        for p, l in zip(train_paths, train_labels)
    ],
    "validation": [
        {"path": p, "label": l}
        for p, l in zip(val_paths, val_labels)
    ],
    "test": [
        {"path": p, "label": l}
        for p, l in zip(test_paths, test_labels)
    ],
}

with OUTPUT.open("w", encoding="utf-8") as file:
    json.dump(splits, file, indent=2)

for name, records in splits.items():
    counts = Counter(item["label"] for item in records)
    print(
        f"{name}: {len(records)} images "
        f"(Normal={counts['Normal']}, Tumor={counts['Tumor']})"
    )

print(f"\nSaved dataset splits to {OUTPUT}")
