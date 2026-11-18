
# ============================================================
# APARTMENT PRICE PREDICTION
# Antrenare, optimizare si evaluare pentru 8 modele ML
# ============================================================

from pathlib import Path
import json
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    ParameterSampler
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURARE
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "apartments_clean.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = MODELS_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Nr Camere",
    "Suprafata",
    "Etaj",
    "Total Etaje",
    "Sector",
    "Scor"
]

NUMERIC_FEATURES = [
    "Nr Camere",
    "Suprafata",
    "Etaj",
    "Total Etaje",
    "Scor"
]

CATEGORICAL_FEATURES = ["Sector"]

# Numar de configuratii pentru fiecare model clasic.
# Creste la 15-20 pentru o cautare mai ampla.
N_SEARCH_ITER = 8

# Categorii pentru matricea de confuzie
PRICE_BINS = [
    -np.inf, 50000, 100000,
    150000, 200000, 250000, np.inf
]

PRICE_LABELS = [
    "Sub 50k",
    "50k-100k",
    "100k-150k",
    "150k-200k",
    "200k-250k",
    "Peste 250k"
]

# ============================================================
# FUNCTII UTILE
# ============================================================

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    relative_error = np.abs(y_true - y_pred) / np.maximum(
        np.abs(y_true), 1
    )

    accuracy_10 = np.mean(relative_error <= 0.10) * 100
    accuracy_20 = np.mean(relative_error <= 0.20) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Within_10pct": float(accuracy_10),
        "Within_20pct": float(accuracy_20)
    }


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "sector",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )


def save_json(data, filename):
    with open(
        RESULTS_DIR / filename,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_predictions(model_name, split_name, y_true, y_pred):
    result = pd.DataFrame({
        "Pret_real": np.asarray(y_true),
        "Pret_prezis": np.asarray(y_pred).reshape(-1)
    })

    result["Eroare_absoluta"] = np.abs(
        result["Pret_real"] - result["Pret_prezis"]
    )

    result["Eroare_procentuala"] = (
        result["Eroare_absoluta"]
        / result["Pret_real"].abs().clip(lower=1)
        * 100
    )

    result["Model"] = model_name
    result["Set"] = split_name

    return result


def save_confusion_matrix(model_name, y_true, y_pred):
    true_bins = pd.cut(
        y_true,
        bins=PRICE_BINS,
        labels=PRICE_LABELS
    )

    pred_bins = pd.cut(
        y_pred,
        bins=PRICE_BINS,
        labels=PRICE_LABELS
    )

    cm = confusion_matrix(
        true_bins,
        pred_bins,
        labels=PRICE_LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=PRICE_LABELS,
        columns=PRICE_LABELS
    )

    safe_name = model_name.lower().replace(" ", "_")

    cm_df.to_csv(
        RESULTS_DIR / f"confusion_{safe_name}.csv"
    )


def display_metrics(model_name, metrics):
    print(
        f"{model_name:<24} | "
        f"MAE: {metrics['MAE']:>9,.0f} EUR | "
        f"RMSE: {metrics['RMSE']:>9,.0f} EUR | "
        f"R2: {metrics['R2']:.4f} | "
        f"±10%: {metrics['Within_10pct']:.1f}%"
    )


# ============================================================
# INCARCAREA DATASETULUI
# ============================================================

print("\n" + "=" * 75)
print("APARTMENT PRICE PREDICTION - 8 MODELE")
print("=" * 75)

df = pd.read_csv(DATA_FILE)

X = df[FEATURES]
y = df["Pret"]

# 70% antrenare, 30% temporar
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=SEED
)

# 15% validare, 15% testare
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=SEED
)

print("\nIMPARTIREA DATASETULUI")
print("-" * 55)
print(f"Total:      {len(df)}")
print(f"Antrenare:  {len(X_train)}")
print(f"Validare:   {len(X_val)}")
print(f"Testare:    {len(X_test)}")


# ============================================================
# DEFINIREA CELOR 7 MODELE SCIKIT-LEARN
# ============================================================

# Modelul 8 va fi construit separat cu TensorFlow/Keras.

model_configs = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {
            "model__fit_intercept": [True, False]
        }
    },

    "Decision Tree": {
        "model": DecisionTreeRegressor(
            random_state=SEED
        ),
        "params": {
            "model__max_depth": [5, 10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8]
        }
    },

    "Random Forest": {
        "model": RandomForestRegressor(
            random_state=SEED,
            n_jobs=-1
        ),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingRegressor(
            random_state=SEED
        ),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0]
        }
    },

    "KNN": {
        "model": KNeighborsRegressor(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }
    },

    "Extra Trees": {
        "model": ExtraTreesRegressor(
            random_state=SEED,
            n_jobs=-1
        ),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }
    },

    "MLP Neural Network": {
        "model": MLPRegressor(
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            max_iter=1000,
            random_state=SEED
        ),
        "params": {
            "model__hidden_layer_sizes": [
                (64, 32),
                (128, 64),
                (128, 64, 32),
                (256, 128, 64)
            ],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [
                0.0005, 0.001, 0.005
            ]
        }
    }
}


# ============================================================
# OPTIMIZAREA CELOR 7 MODELE
# ============================================================

trained_models = {}
validation_results = []
training_results = []
prediction_frames = []
training_histories = {}

print("\nOPTIMIZAREA MODELELOR SCIKIT-LEARN")
print("=" * 75)

for model_name, config in model_configs.items():

    print(f"\nModel: {model_name}")
    start_time = time.time()

    param_list = list(
        ParameterSampler(
            config["params"],
            n_iter=N_SEARCH_ITER,
            random_state=SEED
        )
    )

    best_model = None
    best_params = None
    best_mae = float("inf")

    for i, params in enumerate(param_list, start=1):

        pipeline = Pipeline([
            ("preprocesare", make_preprocessor()),
            ("model", config["model"])
        ])

        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)

        val_pred = pipeline.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)

        print(
            f"  Configuratia {i}/{len(param_list)} "
            f"| MAE validare: {val_mae:,.0f} EUR"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_model = pipeline
            best_params = params

    trained_models[model_name] = best_model

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)

    train_metrics = evaluate(y_train, train_pred)
    val_metrics = evaluate(y_val, val_pred)

    training_results.append({
        "Model": model_name,
        **train_metrics
    })

    validation_results.append({
        "Model": model_name,
        **val_metrics,
        "Training_time_sec": round(
            time.time() - start_time, 2
        )
    })

    prediction_frames.append(
        save_predictions(
            model_name, "Train", y_train, train_pred
        )
    )

    prediction_frames.append(
        save_predictions(
            model_name, "Validation", y_val, val_pred
        )
    )

    safe_name = model_name.lower().replace(" ", "_")

    joblib.dump(
        best_model,
        MODELS_DIR / f"{safe_name}.pkl"
    )

    save_json(
        {
            "model": model_name,
            "best_params": best_params,
            "train": train_metrics,
            "validation": val_metrics
        },
        f"params_{safe_name}.json"
    )

    # Salvam evolutia MLP pentru graficele din Notebook
    if model_name == "MLP Neural Network":
        mlp = best_model.named_steps["model"]

        training_histories["MLP Neural Network"] = {
            "loss": [
                float(v) for v in mlp.loss_curve_
            ],
            "validation_score": [
                float(v) for v in getattr(
                    mlp, "validation_scores_", []
                )
            ]
        }

    display_metrics(model_name, val_metrics)


# ============================================================
# PREGATIREA DATELOR PENTRU DEEP LEARNING
# ============================================================

print("\nPREGATIREA RETELEI DEEP LEARNING")
print("=" * 75)

deep_preprocessor = make_preprocessor()

X_train_nn = deep_preprocessor.fit_transform(X_train)
X_val_nn = deep_preprocessor.transform(X_val)
X_test_nn = deep_preprocessor.transform(X_test)

X_train_nn = np.asarray(X_train_nn, dtype=np.float32)
X_val_nn = np.asarray(X_val_nn, dtype=np.float32)
X_test_nn = np.asarray(X_test_nn, dtype=np.float32)

# Normalizam si pretul pentru o antrenare mai stabila
target_scaler = StandardScaler()

y_train_nn = target_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()

y_val_nn = target_scaler.transform(
    y_val.to_numpy().reshape(-1, 1)
).ravel()

joblib.dump(
    {
        "preprocessor": deep_preprocessor,
        "target_scaler": target_scaler,
        "features": FEATURES
    },
    MODELS_DIR / "deep_learning_preprocessing.pkl"
)


# ============================================================
# MODELUL 8 - DEEP NEURAL NETWORK
# ============================================================

def build_deep_model(hp):

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Input(
            shape=(X_train_nn.shape[1],)
        )
    )

    num_layers = hp.Int(
        "num_layers",
        min_value=2,
        max_value=4
    )

    for i in range(num_layers):

        units = hp.Choice(
            f"units_{i}",
            values=[32, 64, 128, 256]
        )

        model.add(
            tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(
                    hp.Choice(
                        "l2",
                        values=[0.0001, 0.001, 0.01]
                    )
                )
            )
        )

        dropout = hp.Float(
            f"dropout_{i}",
            min_value=0.0,
            max_value=0.4,
            step=0.1
        )

        model.add(
            tf.keras.layers.Dropout(dropout)
        )

    model.add(tf.keras.layers.Dense(1))

    learning_rate = hp.Choice(
        "learning_rate",
        values=[0.0001, 0.0005, 0.001]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


print("\nMODELUL 8 - DEEP NEURAL NETWORK")
print("=" * 75)

start_time = time.time()

tuner = kt.RandomSearch(
    hypermodel=build_deep_model,
    objective="val_loss",
    max_trials=12,
    executions_per_trial=1,
    directory=str(MODELS_DIR / "keras_tuner"),
    project_name="apartment_dnn",
    overwrite=True,
    seed=SEED
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

tuner.search(
    X_train_nn,
    y_train_nn,
    validation_data=(X_val_nn, y_val_nn),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

best_hp = tuner.get_best_hyperparameters(1)[0]

deep_model = tuner.hypermodel.build(best_hp)

history = deep_model.fit(
    X_train_nn,
    y_train_nn,
    validation_data=(X_val_nn, y_val_nn),
    epochs=300,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True
        )
    ],
    verbose=0
)

deep_model.save(
    MODELS_DIR / "deep_neural_network.keras"
)

training_histories["Deep Neural Network"] = {
    key: [float(v) for v in values]
    for key, values in history.history.items()
}

def predict_deep(X_data):
    pred_scaled = deep_model.predict(
        X_data, verbose=0
    ).reshape(-1, 1)

    return target_scaler.inverse_transform(
        pred_scaled
    ).ravel()


deep_train_pred = predict_deep(X_train_nn)
deep_val_pred = predict_deep(X_val_nn)

deep_train_metrics = evaluate(
    y_train, deep_train_pred
)

deep_val_metrics = evaluate(
    y_val, deep_val_pred
)

training_results.append({
    "Model": "Deep Neural Network",
    **deep_train_metrics
})

validation_results.append({
    "Model": "Deep Neural Network",
    **deep_val_metrics,
    "Training_time_sec": round(
        time.time() - start_time, 2
    )
})

prediction_frames.append(
    save_predictions(
        "Deep Neural Network",
        "Train",
        y_train,
        deep_train_pred
    )
)

prediction_frames.append(
    save_predictions(
        "Deep Neural Network",
        "Validation",
        y_val,
        deep_val_pred
    )
)

save_json(
    {
        "model": "Deep Neural Network",
        "best_params": best_hp.values,
        "train": deep_train_metrics,
        "validation": deep_val_metrics
    },
    "params_deep_neural_network.json"
)

display_metrics(
    "Deep Neural Network",
    deep_val_metrics
)

# ============================================================
# CLASAMENTUL PE VALIDARE
# ============================================================

validation_df = pd.DataFrame(
    validation_results
).sort_values("MAE").reset_index(drop=True)

training_df = pd.DataFrame(training_results)

validation_df.to_csv(
    RESULTS_DIR / "validation_metrics.csv",
    index=False
)

training_df.to_csv(
    RESULTS_DIR / "training_metrics.csv",
    index=False
)

best_model_name = validation_df.iloc[0]["Model"]

print("\n" + "=" * 75)
print("CLASAMENT FINAL PE VALIDARE")
print("=" * 75)

print(
    validation_df[
        ["Model", "MAE", "RMSE", "R2", "Within_10pct"]
    ].round(2).to_string(index=False)
)

print(f"\nMODEL SELECTAT: {best_model_name}")


# ============================================================
# EVALUAREA FINALA PE SETUL DE TESTARE
# ============================================================

# Setul de testare nu a fost folosit pentru optimizare.
# Evaluam toate cele 8 modele o singura data.

print("\n" + "=" * 75)
print("EVALUAREA FINALA PE SETUL DE TESTARE")
print("=" * 75)

test_results = []

for model_name, model in trained_models.items():

    test_pred = model.predict(X_test)

    metrics = evaluate(y_test, test_pred)

    test_results.append({
        "Model": model_name,
        **metrics
    })

    prediction_frames.append(
        save_predictions(
            model_name, "Test", y_test, test_pred
        )
    )

    save_confusion_matrix(
        model_name, y_test, test_pred
    )

    display_metrics(model_name, metrics)


# Evaluarea modelului Deep Neural Network
deep_test_pred = predict_deep(X_test_nn)

deep_test_metrics = evaluate(
    y_test, deep_test_pred
)

test_results.append({
    "Model": "Deep Neural Network",
    **deep_test_metrics
})

prediction_frames.append(
    save_predictions(
        "Deep Neural Network",
        "Test",
        y_test,
        deep_test_pred
    )
)

save_confusion_matrix(
    "Deep Neural Network",
    y_test,
    deep_test_pred
)

display_metrics(
    "Deep Neural Network",
    deep_test_metrics
)

test_df = pd.DataFrame(test_results)

# Afisam rezultatele in ordinea stabilita pe validare
test_df["Model"] = pd.Categorical(
    test_df["Model"],
    categories=validation_df["Model"].tolist(),
    ordered=True
)

test_df = test_df.sort_values("Model")

test_df.to_csv(
    RESULTS_DIR / "test_metrics.csv",
    index=False
)

# ============================================================
# SALVAREA REZULTATELOR PENTRU NOTEBOOK
# ============================================================

all_predictions = pd.concat(
    prediction_frames,
    ignore_index=True
)

all_predictions.to_csv(
    RESULTS_DIR / "all_predictions.csv",
    index=False
)

save_json(
    training_histories,
    "training_histories.json"
)

save_json(
    {
        "selected_model": best_model_name,
        "selection_metric": "Validation MAE",
        "features": FEATURES,
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "test_size": len(X_test)
    },
    "model_summary.json"
)

# Salvam modelul castigator intr-un format comun
if best_model_name == "Deep Neural Network":
    # Modelul Keras si preprocesarea sunt deja salvate.
    selected_model_file = "deep_neural_network.keras"
else:
    joblib.dump(
        trained_models[best_model_name],
        MODELS_DIR / "price_model.pkl"
    )
    selected_model_file = "price_model.pkl"

print("\n" + "=" * 75)
print("ANTRENAREA SI EVALUAREA AU FOST FINALIZATE")
print("=" * 75)

print(f"Model selectat: {best_model_name}")
print(f"Fisier model: {selected_model_file}")
print(f"Rezultate salvate in: {RESULTS_DIR}")