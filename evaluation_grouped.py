from pathlib import Path
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

FEATURE_DIR = Path("features_grouped")
MODEL_DIR = Path("models_grouped")
OUTPUT_DIR = Path("evaluation_grouped")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load actual held-out test data.
X = np.load(FEATURE_DIR / "test_features.npy")
y = np.load(FEATURE_DIR / "test_labels.npy")

svm = joblib.load(MODEL_DIR / "svm_model.joblib")
rf = joblib.load(MODEL_DIR / "rf_model.joblib")
config = joblib.load(MODEL_DIR / "ensemble_config.joblib")

assert X.shape == (1110, 3072)
assert len(y) == len(X)
assert set(np.unique(y)) == {0, 1}

# Align probabilities with the Tumor class.
svm_index = list(svm.classes_).index(1)
rf_index = list(rf.classes_).index(1)

svm_prob = svm.predict_proba(X)[:, svm_index]
rf_prob = rf.predict_proba(X)[:, rf_index]

tumor_prob = (
    config["svm_weight"] * svm_prob
    + config["rf_weight"] * rf_prob
)

threshold = float(config["threshold"])
pred = (tumor_prob >= threshold).astype(int)

accuracy = accuracy_score(y, pred)
cm = confusion_matrix(y, pred, labels=[0, 1])

report = classification_report(
    y,
    pred,
    labels=[0, 1],
    target_names=["Normal", "Tumor"],
    digits=4,
    zero_division=0,
)

print("Test accuracy:", f"{accuracy:.2%}")
print("Confusion matrix:")
print(cm)
print(report)

# Save actual predictions for reproducibility.
np.save(OUTPUT_DIR / "test_predictions.npy", pred)
np.save(OUTPUT_DIR / "test_tumor_scores.npy", tumor_prob)

# Chart 1: Confusion matrix.
fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Tumor"],
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title("Similarity-Grouped Test Confusion Matrix")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# Chart 2: Class-specific recall.
class_recall = [
    cm[0, 0] / cm[0].sum() * 100,
    cm[1, 1] / cm[1].sum() * 100,
]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(["Normal", "Tumor"], class_recall)

ax.set_ylim(0, 110)
ax.set_ylabel("Recall (%)")
ax.set_title("Test Recall by Class")

for bar, value in zip(bars, class_recall):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
    )

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "class_recall.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# Chart 3: Precision, recall and F1.
precision, recall, f1, _ = precision_recall_fscore_support(
    y, pred, labels=[0, 1], zero_division=0
)

x = np.arange(2)
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width, precision * 100, width, label="Precision")
ax.bar(x, recall * 100, width, label="Recall")
ax.bar(x + width, f1 * 100, width, label="F1-score")

ax.set_xticks(x)
ax.set_xticklabels(["Normal", "Tumor"])
ax.set_ylim(0, 110)
ax.set_ylabel("Score (%)")
ax.set_title("Grouped Test Classification Metrics")
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "classification_metrics.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

summary = (
    "Kidney Tumor Detection - Similarity-Grouped Test Evaluation\n"
    "===========================================================\n"
    f"Test images: {len(y)}\n"
    f"Classification threshold: {threshold:.2f}\n"
    f"SVM weight: {config['svm_weight']}\n"
    f"RF weight: {config['rf_weight']}\n"
    f"Test accuracy: {accuracy:.4%}\n\n"
    f"Confusion matrix:\n{cm}\n\n"
    f"{report}\n"
    "LIMITATIONS:\n"
    "Predictions were generated from the saved trained models.\n"
    "The split groups images using perceptual-hash similarity.\n"
    "Patient-independent evaluation has NOT been established.\n"
    "This research prototype is NOT suitable for clinical diagnosis.\n"
)

(OUTPUT_DIR / "evaluation_summary.txt").write_text(
    summary, encoding="utf-8"
)

print("\nCharts and evaluation saved to:", OUTPUT_DIR.resolve())
