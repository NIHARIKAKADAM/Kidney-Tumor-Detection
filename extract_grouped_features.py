
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from feature_extraction import load_models, extract_features


SPLITS_FILE = Path("splits_grouped.json")
OUTPUT_DIR = Path("features_grouped")
CHECKPOINT_EVERY = 50
FEATURE_DIM = 3072

LABEL_MAP = {"Normal": 0, "Tumor": 1}


def process_split(name, records, densenet, resnet):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = OUTPUT_DIR / f"{name}_features.npy"
    labels_path = OUTPUT_DIR / f"{name}_labels.npy"
    progress_path = OUTPUT_DIR / f"{name}_progress.json"

    total = len(records)

    # Memory-mapped files avoid keeping the entire feature
    # matrix in RAM and support resuming after interruption.
    if features_path.exists() or labels_path.exists():
        if not (
            features_path.exists()
            and labels_path.exists()
            and progress_path.exists()
        ):
            raise RuntimeError(
                f"Incomplete checkpoint files for {name}. "
                "Do not delete them; inspect before restarting."
            )

        X = np.lib.format.open_memmap(
            features_path, mode="r+"
        )
        y = np.lib.format.open_memmap(
            labels_path, mode="r+"
        )

        if X.shape != (total, FEATURE_DIM) or y.shape != (total,):
            raise RuntimeError(
                f"Checkpoint shape mismatch for {name}."
            )

        progress = json.loads(progress_path.read_text())
        start = progress["completed"]

        print(f"Resuming {name} from image {start}/{total}")

    else:
        X = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=(total, FEATURE_DIM),
        )
        y = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype=np.int64,
            shape=(total,),
        )
        start = 0

    started_at = time.time()

    for index in range(start, total):
        item = records[index]
        path = Path(item["path"])
        label = item["label"]

        if label not in LABEL_MAP:
            raise ValueError(f"Unknown label: {label}")

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")

            vector = extract_features(image, densenet, resnet)

            if vector.shape != (FEATURE_DIM,):
                raise ValueError(
                    f"Unexpected feature shape: {vector.shape}"
                )

            if not np.isfinite(vector).all():
                raise ValueError("Non-finite features detected")

            X[index] = vector.astype(np.float32)
            y[index] = LABEL_MAP[label]

        except (OSError, UnidentifiedImageError, ValueError) as exc:
            # Stop instead of silently changing the dataset.
            X.flush()
            y.flush()
            raise RuntimeError(
                f"Failed at {name} image {index}: {path}"
            ) from exc

        completed = index + 1

        if completed % CHECKPOINT_EVERY == 0 or completed == total:
            X.flush()
            y.flush()

            progress_path.write_text(
                json.dumps({
                    "completed": completed,
                    "total": total,
                    "split": name,
                }, indent=2)
            )

            elapsed = time.time() - started_at
            print(
                f"{name}: {completed}/{total} completed "
                f"({elapsed:.1f}s elapsed this session)",
                flush=True,
            )

    print(f"Finished {name}: {total} images")


def main():
    if not SPLITS_FILE.exists():
        raise FileNotFoundError("splits.json not found.")

    splits = json.loads(SPLITS_FILE.read_text())

    densenet, resnet = load_models()

    for name in ["train", "validation", "test"]:
        if name not in splits:
            raise KeyError(f"Missing split: {name}")

        process_split(name, splits[name], densenet, resnet)

    print("\nAll feature extraction completed successfully!")


if __name__ == "__main__":
    main()
