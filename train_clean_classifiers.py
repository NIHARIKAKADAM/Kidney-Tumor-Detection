
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURE_DIR = Path("features_clean")
MODEL_DIR = Path("models_clean")
MODEL_DIR.mkdir(exist_ok=True)


def load_split(name):
    X = np.load(FEATURE_DIR / f"{name}_features.npy")
    y = np.load(FEATURE_DIR / f"{name}_labels.npy")

    if X.ndim != 2 or X.shape[1] != 3072:
        raise ValueError(f"Invalid feature shape for {name}: {X.shape}")

    if len(X) != len(y):
        raise ValueError(f"Feature/label mismatch for {name}")

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features in {name}")

    return X, y


print("Loading training and validation features...")
X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

print("Training SVM...")
svm = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=42,
    ),
)
svm.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=2,
)
rf.fit(X_train, y_train)

print("Evaluating on validation dataset...")

svm_prob = svm.predict_proba(X_val)[:, 1]
rf_prob = rf.predict_proba(X_val)[:, 1]

svm_acc = accuracy_score(y_val, svm_prob >= 0.5)
rf_acc = accuracy_score(y_val, rf_prob >= 0.5)

print(f"SVM validation accuracy: {svm_acc:.4f}")
print(f"Random Forest validation accuracy: {rf_acc:.4f}")

best_accuracy = -1
best_weight = None

for svm_weight in np.linspace(0, 1, 21):
    ensemble_prob = (
        svm_weight * svm_prob
        + (1 - svm_weight) * rf_prob
    )

    predictions = (ensemble_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weight = float(svm_weight)

print(f"Best SVM weight: {best_weight:.2f}")
print(f"Best RF weight: {1 - best_weight:.2f}")
print(f"Ensemble validation accuracy: {best_accuracy:.4f}")

final_prob = (
    best_weight * svm_prob
    + (1 - best_weight) * rf_prob
)

print("\nValidation classification report:")
print(
    classification_report(
        y_val,
        (final_prob >= 0.5).astype(int),
        target_names=["Normal", "Tumor"],
        zero_division=0,
    )
)

joblib.dump(svm, MODEL_DIR / "svm_model.joblib")
joblib.dump(rf, MODEL_DIR / "rf_model.joblib")
joblib.dump(
    {
        "svm_weight": best_weight,
        "rf_weight": 1 - best_weight,
        "threshold": 0.5,
        "class_names": ["Normal", "Tumor"],
    },
    MODEL_DIR / "ensemble_config.joblib",
)

print("\nModels saved successfully!")
