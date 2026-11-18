# APARTMENT PRICE PREDICTION - PREDICT

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = MODELS_DIR / "results"

FEATURES = [
    "Nr Camere",
    "Suprafata",
    "Etaj",
    "Total Etaje",
    "Sector",
    "Scor"
]

class ApartmentPredictor:

    def __init__(self):

        # Citim modelul selectat pe validare
        with open(
            RESULTS_DIR / "model_summary.json",
            "r",
            encoding="utf-8"
        ) as f:
            summary = json.load(f)

        self.model_name = summary["selected_model"]
        self.is_deep_learning = (
            self.model_name == "Deep Neural Network"
        )

        if self.is_deep_learning:

            # Importam TensorFlow doar daca este necesar
            import tensorflow as tf

            self.model = tf.keras.models.load_model(
                MODELS_DIR / "deep_neural_network.keras"
            )

            preprocessing = joblib.load(
                MODELS_DIR / "deep_learning_preprocessing.pkl"
            )

            self.preprocessor = preprocessing["preprocessor"]
            self.target_scaler = preprocessing["target_scaler"]

        else:
            self.model = joblib.load(
                MODELS_DIR / "price_model.pkl"
            )

    def predict(
        self,
        suprafata,
        camere,
        etaj,
        total_etaje,
        sector,
        scor
    ):

        # Construim un apartament nou
        apartment = pd.DataFrame([{
            "Nr Camere": int(camere),
            "Suprafata": float(suprafata),
            "Etaj": int(etaj),
            "Total Etaje": int(total_etaje),
            "Sector": int(sector),
            "Scor": float(scor)
        }], columns=FEATURES)

        if self.is_deep_learning:

            X = self.preprocessor.transform(apartment)
            X = np.asarray(X, dtype=np.float32)

            prediction_scaled = self.model.predict(
                X,
                verbose=0
            )

            prediction = self.target_scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            )[0, 0]

        else:
            prediction = self.model.predict(apartment)[0]

        return float(prediction)


# Test direct, optional
if __name__ == "__main__":

    predictor = ApartmentPredictor()

    pret = predictor.predict(
        suprafata=75,
        camere=3,
        etaj=4,
        total_etaje=10,
        sector=3,
        scor=5
    )

    print("Model utilizat:", predictor.model_name)
    print(f"Pret estimat: {pret:,.0f} EUR")