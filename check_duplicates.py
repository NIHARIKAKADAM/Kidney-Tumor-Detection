
import hashlib
import json
from pathlib import Path

with open("splits.json", encoding="utf-8") as f:
    splits = json.load(f)

hashes = {}
duplicates = []

for split_name in ["train", "validation", "test"]:
    for record in splits[split_name]:
        path = Path(record["path"])

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        if digest in hashes:
            previous_split, previous_path = hashes[digest]

            if previous_split != split_name:
                duplicates.append({
                    "first_split": previous_split,
                    "second_split": split_name,
                    "first_path": previous_path,
                    "second_path": str(path),
                })
        else:
            hashes[digest] = (split_name, str(path))

print("Cross-split exact duplicate pairs:", len(duplicates))

for item in duplicates[:10]:
    print(item)

if duplicates:
    print(
        "\nWARNING: Identical image files exist across "
        "different dataset splits."
    )
else:
    print(
        "\nNo exact cross-split duplicates detected. "
        "This does not rule out near-duplicates or patient overlap."
    )
