
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

from preprocessing import enhance_kidney_image


files = sorted(
    p for p in Path("dataset/Normal").rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not files:
    raise RuntimeError("No Normal CT images found.")

with Image.open(files[0]) as img:
    original = img.convert("RGB")

enhanced = enhance_kidney_image(original)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(original)
axes[0].set_title("Original CT Image")
axes[0].axis("off")

axes[1].imshow(enhanced)
axes[1].set_title("Fuzzy-Enhanced CT Image (Prototype)")
axes[1].axis("off")

plt.tight_layout()
plt.show()
