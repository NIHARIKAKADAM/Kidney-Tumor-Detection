
import json
from pathlib import Path

import imagehash
from PIL import Image

with open("splits_clean.json") as f:
    splits = json.load(f)

hashes = {}

for split in ["train", "validation", "test"]:
    hashes[split] = []

    for record in splits[split]:
        path = Path(record["path"])

        with Image.open(path) as image:
            fingerprint = imagehash.phash(
                image.convert("L"),
                hash_size=8
            )

        hashes[split].append(
            (str(path), fingerprint)
        )

    print(f"Hashed {len(hashes[split])} {split} images")

print("\nChecking cross-split near-duplicates...")

pairs = []
threshold = 4

for first, second in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:
    for path1, hash1 in hashes[first]:
        for path2, hash2 in hashes[second]:
            distance = hash1 - hash2

            if distance <= threshold:
                pairs.append({
                    "split1": first,
                    "split2": second,
                    "image1": path1,
                    "image2": path2,
                    "distance": int(distance),
                })

Path("near_duplicate_report.json").write_text(
    json.dumps(pairs, indent=2)
)

print("Potential near-duplicate pairs:", len(pairs))
print("Report saved: near_duplicate_report.json")

for pair in pairs[:10]:
    print(pair)
