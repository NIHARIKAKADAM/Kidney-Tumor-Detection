import json
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

SOURCE = Path("splits_clean.json")
OUTPUT = Path("splits_grouped.json")
THRESHOLD = 4
SEED = 42

data = json.loads(SOURCE.read_text())

records = [
    dict(record)
    for split in ("train", "validation", "test")
    for record in data[split]
]

paths = [r["path"] for r in records]
labels = [r["label"] for r in records]

if len(paths) != len(set(paths)):
    raise ValueError("Repeated image paths found.")

hashes = []

for i, path in enumerate(paths):
    with Image.open(path) as img:
        hashes.append(
            int(str(imagehash.phash(img.convert("L"))), 16)
        )

    if (i + 1) % 1000 == 0:
        print(f"Hashed {i + 1}/{len(paths)}", flush=True)

parent = list(range(len(paths)))
sizes = [1] * len(paths)


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    a, b = find(a), find(b)

    if a == b:
        return

    if sizes[a] < sizes[b]:
        a, b = b, a

    parent[b] = a
    sizes[a] += sizes[b]


# Compare all pairs within each class.
for label in ("Normal", "Tumor"):
    indices = [
        i for i, value in enumerate(labels)
        if value == label
    ]

    print(f"Grouping {label}: {len(indices)} images", flush=True)

    for pos, i in enumerate(indices):
        for j in indices[pos + 1:]:
            if (hashes[i] ^ hashes[j]).bit_count() <= THRESHOLD:
                union(i, j)

roots = [find(i) for i in range(len(records))]
groups = defaultdict(list)

for i, root in enumerate(roots):
    groups[root].append(i)

print("Total similarity groups:", len(groups))
print(
    "Largest group:",
    max(len(members) for members in groups.values())
)

# Split groups separately within each class.
rng = np.random.default_rng(SEED)
result = {"train": [], "validation": [], "test": []}

for label in ("Normal", "Tumor"):
    class_groups = [
        members for members in groups.values()
        if labels[members[0]] == label
    ]

    rng.shuffle(class_groups)
    class_groups.sort(key=len, reverse=True)

    total = sum(len(g) for g in class_groups)
    targets = {
        "train": total * 0.70,
        "validation": total * 0.15,
        "test": total * 0.15,
    }

    counts = {name: 0 for name in result}
    assigned = {name: [] for name in result}

    # Greedily balance image counts while keeping groups intact.
    for group in class_groups:
        destination = max(
            result,
            key=lambda name: (
                targets[name] - counts[name]
            ) / targets[name]
        )

        assigned[destination].extend(group)
        counts[destination] += len(group)

    for split, indices in assigned.items():
        result[split].extend(records[i] for i in indices)

    print(f"\n{label} distribution:")
    for split in result:
        count = sum(
            r["label"] == label
            for r in result[split]
        )
        print(f"  {split}: {count}")

# Validate that no group crosses splits.
location = {
    record["path"]: split
    for split, items in result.items()
    for record in items
}

crossing = 0

for members in groups.values():
    assigned_splits = {
        location[paths[i]]
        for i in members
    }
    if len(assigned_splits) > 1:
        crossing += 1

print("\nGroups crossing splits:", crossing)

if crossing != 0:
    raise RuntimeError("Similarity group leakage detected.")

if any(len(items) == 0 for items in result.values()):
    raise RuntimeError("An output split is empty.")

if sum(map(len, result.values())) != len(records):
    raise RuntimeError("Image count mismatch.")

OUTPUT.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print("\nNew split saved:", OUTPUT)
print("Final image counts:")
for split, items in result.items():
    print(f"  {split}: {len(items)}")

print(
    "\nNOTE: pHash grouping reduces visual-similarity "
    "overlap but does not establish patient independence."
)
