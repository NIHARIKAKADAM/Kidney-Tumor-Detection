
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

FEATURE_DIR = Path("features_clean")
MODEL_DIR = Path("models_clean")

print("Loading cleaned test features...")
X_test = np.load(FEATURE_DIR / "test_features.npy")
y_test = np.load(FEATURE_DIR / "test_labels.npy")

assert X_test.shape == (1101, 3072)
assert y_test.shape == (1101,)

print("Loading trained models...")
svm = joblib.load(MODEL_DIR / "svm_model.joblib")
rf = joblib.load(MODEL_DIR / "rf_model.joblib")
config = joblib.load(MODEL_DIR / "ensemble_config.joblib")

print("Generating test predictions...")

svm_prob = svm.predict_proba(X_test)[:, 1]
rf_prob = rf.predict_proba(X_test)[:, 1]

ensemble_prob = (
    config["svm_weight"] * svm_prob
    + config["rf_weight"] * rf_prob
)

predictions = (
    ensemble_prob >= config["threshold"]
).astype(int)

print("\nTEST ACCURACY:")
print(f"{accuracy_score(y_test, predictions):.4f}")

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, predictions))

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Tumor"],
        zero_division=0,
    )
)

print("\nTest evaluation completed successfully!")
