
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

DATASET = Path("dataset")
OUTPUT = Path("splits_clean.json")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

splits = {
    "train": [],
    "validation": [],
    "test": [],
}

for label in ["Normal", "Tumor"]:
    groups = defaultdict(list)

    for path in sorted((DATASET / label).glob("*")):
        if not path.is_file():
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append(str(path))

    unique_groups = list(groups.values())
    random.shuffle(unique_groups)

    total = len(unique_groups)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    partitions = {
        "train": unique_groups[:train_end],
        "validation": unique_groups[train_end:val_end],
        "test": unique_groups[val_end:],
    }

    for split_name, group_list in partitions.items():
        for group in group_list:
            for path in group:
                splits[split_name].append({
                    "path": path,
                    "label": label,
                })

for name, records in splits.items():
    random.shuffle(records)

OUTPUT.write_text(json.dumps(splits, indent=2))

for name, records in splits.items():
    normal = sum(x["label"] == "Normal" for x in records)
    tumor = sum(x["label"] == "Tumor" for x in records)

    print(
        f"{name}: {len(records)} images "
        f"(Normal: {normal}, Tumor: {tumor})"
    )

print(f"\nClean splits saved to {OUTPUT}")
print("Identical files are grouped into the same split.")
