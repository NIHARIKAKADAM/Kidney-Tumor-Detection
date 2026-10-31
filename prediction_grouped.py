
from pathlib import Path

import joblib
import numpy as np

from feature_extraction import load_models, extract_features


MODEL_DIR = Path("models_grouped")


class KidneyTumorPredictor:
    def __init__(self):
        self.densenet, self.resnet = load_models()

        self.svm = joblib.load(
            MODEL_DIR / "svm_model.joblib"
        )

        self.rf = joblib.load(
            MODEL_DIR / "rf_model.joblib"
        )

        self.config = joblib.load(
            MODEL_DIR / "ensemble_config.joblib"
        )

    def predict(self, image):
        features = extract_features(
            image,
            self.densenet,
            self.resnet
        )

        features = np.asarray(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        svm_prob = self.svm.predict_proba(features)[0]
        rf_prob = self.rf.predict_proba(features)[0]

        probabilities = (
            self.config["svm_weight"] * svm_prob
            + self.config["rf_weight"] * rf_prob
        )

        tumor_probability = float(probabilities[1])
        threshold = float(self.config["threshold"])

        prediction = (
            "Tumor"
            if tumor_probability >= threshold
            else "Normal"
        )

        return {
            "prediction": prediction,
            "normal_probability": float(probabilities[0]),
            "tumor_probability": tumor_probability,
        }
