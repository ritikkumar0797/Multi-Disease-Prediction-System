import streamlit as st
import os
import joblib
import sqlite3
import json
import math
from pathlib import Path
import pandas as pd

# ===============================
# Paths & Configurations
# ===============================

MODEL_DIR = Path("models")
DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "predictions.db"

DISEASE_CONFIG = {
    "diabetes": {
        "label": "Outcome",
        "features": ["Glucose", "BloodPressure", "BMI", "Age"],
        "dataset_tag": "diabetes_sample"
    },
    "heart": {
        "label": "target",
        "features": ["age", "sex", "trestbps", "chol", "thalach", "exang"],
        "dataset_tag": "heart_sample"
    },
    "flu": {
        "label": "label",
        "features": [
            "fever", "cough", "sore_throat",
            "body_ache", "fatigue", "loss_of_smell"
        ],
        "dataset_tag": "flu_sample"
    }
}

CLASSIFIERS = ["RandomForest", "DecisionTree", "GaussianNB"]

# ===============================
# Database Handling
# ===============================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            classifier TEXT,
            features TEXT,
            outcome INTEGER,
            probability REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(disease, classifier, features_dict, outcome, probability):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (disease, classifier, features, outcome, probability) VALUES (?, ?, ?, ?, ?)",
        (disease, classifier, json.dumps(features_dict), int(outcome), float(probability))
    )
    conn.commit()
    conn.close()


def load_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

# ===============================
# Model Loading
# ===============================

def load_models():
    models = {}
    scalers = {}

    for disease, cfg in DISEASE_CONFIG.items():
        dataset_tag = cfg["dataset_tag"]

        models[disease] = {}

        # Load scaler
        scaler_path = MODEL_DIR / f"{dataset_tag}_scaler.pkl"
        scalers[disease] = joblib.load(scaler_path) if scaler_path.exists() else None

        # Load each classifier
        for clf in CLASSIFIERS:
            model_path = MODEL_DIR / f"{dataset_tag}_{clf}_model.pkl"
            models[disease][clf] = joblib.load(model_path) if model_path.exists() else None

    return models, scalers

# ===============================
# Prediction Logic
# ===============================

def predict_with_model(model, scaler, vals):
    arr = [vals]

    if scaler is not None:
        try:
            arr = scaler.transform(arr)
        except:
            pass

    pred = int(model.predict(arr)[0])

    try:
        proba_arr = model.predict_proba(arr)[0]
        proba = float(proba_arr[1]) if len(proba_arr) > 1 else float(proba_arr.max())
    except:
        proba = 0.0

    return pred, proba

# ===============================
# Streamlit UI
# ===============================

st.set_page_config(page_title="Disease Prediction System", layout="wide")
st.title("🩺 Multi-Disease Prediction")

st.write("Enter patient values, choose a classifier, and get predictions instantly.")

# Load Models
models, scalers = load_models()

# Disease Selection
disease = st.selectbox("Select Disease", list(DISEASE_CONFIG.keys()))
cfg = DISEASE_CONFIG[disease]
features = cfg["features"]

st.header(f"Input Features for {disease.capitalize()}")

# Collect input feature values
user_vals = {}
col1, col2 = st.columns(2)

for i, feat in enumerate(features):
    with (col1 if i % 2 == 0 else col2):

        # SEX (Heart disease only)
        if feat == "sex":
            sex_input = st.selectbox("Sex", ["Male", "Female"])
            user_vals[feat] = 1 if sex_input == "Male" else 0

        # FLU → YES/NO
        elif disease == "flu":
            yes_no = st.selectbox(f"{feat.replace('_',' ').title()}", ["Yes", "No"])
            user_vals[feat] = 1 if yes_no == "Yes" else 0

        # DEFAULT numeric inputs
        else:
            user_vals[feat] = st.number_input(f"{feat}", step=0.1, format="%.2f")

vals_list = list(user_vals.values())

# Prediction Mode
mode = st.radio("Choose Mode", ["Single Classifier Prediction", "Compare All Classifiers"])

if mode == "Single Classifier Prediction":
    st.subheader("Choose Classifier")
    clf_choice = st.selectbox("Classifier", CLASSIFIERS)

    if st.button("Predict"):
        model = models[disease][clf_choice]
        scaler = scalers[disease]

        if model is None:
            st.error(f"Model for {clf_choice} is missing.")
        else:
            pred, proba = predict_with_model(model, scaler, vals_list)

            st.success(f"Prediction: **{'Yes' if pred == 1 else 'No'}**")
            st.info(f"Probability: **{proba:.4f}**")

            save_prediction(disease, clf_choice, user_vals, pred, proba)

else:
    st.subheader("Classifier Comparison")

    if st.button("Compare"):
        results = []
        for clf in CLASSIFIERS:
            model = models[disease][clf]
            scaler = scalers[disease]

            if model is None:
                results.append([clf, "Model Missing", "—"])
                continue

            pred, proba = predict_with_model(model, scaler, vals_list)

            results.append([clf, "Yes" if pred == 1 else "No", round(proba, 4)])

            save_prediction(disease, clf, user_vals, pred, proba)

        df = pd.DataFrame(results, columns=["Classifier", "Prediction", "Probability"])
        st.table(df)

# ===============================
# Show Prediction History
# ===============================

st.header("📊 Prediction History")

history = load_predictions()

if history.empty:
    st.info("No predictions made yet.")
else:
    # Convert 0/1 outcome → Yes/No
    history['outcome'] = history['outcome'].apply(lambda x: "Yes" if x == 1 else "No")

    # Convert "sex" inside JSON features
    def convert_features(fjson):
        data = json.loads(fjson)
        if "sex" in data:
            data["sex"] = "Male" if data["sex"] == 1 else "Female"
        return json.dumps(data)

    history['features'] = history['features'].apply(convert_features)

    st.dataframe(history)

# Initialize DB
init_db()

st.markdown("##### 2025 Ritik Kumar | Multi Disease prediction ")