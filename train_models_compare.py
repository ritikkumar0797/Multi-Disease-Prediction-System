import os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "datasets")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

EXPERIMENTS = {
    "diabetes_sample.csv": (["Glucose","BloodPressure","BMI","Age"], "Outcome"),
    "heart_sample.csv": (["age","sex","trestbps","chol","thalach","exang"], "target"),
    "flu_sample.csv": (["fever","cough","sore_throat","body_ache","fatigue","loss_of_smell"], "label"),
}

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "GaussianNB": GaussianNB()
}

results = []

def load_dataset(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type: " + path)

for ds_file, (features, target) in EXPERIMENTS.items():
    ds_path = os.path.join(DATA_DIR, ds_file)
    if not os.path.exists(ds_path):
        print(f"Dataset not found: {ds_path}. Skipping.")
        continue
    print(f"\\n=== Processing {ds_file} ===")
    df = load_dataset(ds_path)
    # Basic checks and fill missing
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in {ds_file}. Please check the dataset.")
    df = df.dropna().reset_index(drop=True)
    X = df[features].astype(float)
    y = df[target].astype(int)

    # Train/test split
    strat = y if len(np.unique(y))>1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    # Scaling (fit on train, apply to both)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    for clf_name, clf in CLASSIFIERS.items():
        print(f" Training {clf_name}...")
        model = clf
        model.fit(X_train_s, y_train)
        # Predictions
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"  -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        # Save model and scaler
        model_fname = f"{ds_file.replace('.csv','')}_{clf_name}_model.pkl"
        scaler_fname = f"{ds_file.replace('.csv','')}_scaler.pkl"
        joblib.dump(model, os.path.join(MODELS_DIR, model_fname))
        joblib.dump(scaler, os.path.join(MODELS_DIR, scaler_fname))
        results.append({
            "dataset": ds_file,
            "classifier": clf_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "model_file": model_fname
        })

# Save results to CSV
report_df = pd.DataFrame(results)
report_csv = os.path.join(ROOT, "training_report.csv")
report_df.to_csv(report_csv, index=False)
print(f"\\nTraining finished. Report saved to {report_csv}")
print(report_df.to_string(index=False))
