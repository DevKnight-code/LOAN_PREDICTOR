"""
Enhanced Model Training Script — Random Forest + XGBoost
Run this script ONCE to train the model and save it.
Usage: python train_model.py [--model rf|xgb]
Requires: credit_risk_dataset.csv in the same folder
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Loan Default Predictor")
parser.add_argument(
    "--model", choices=["rf", "xgb"], default="xgb",
    help="Model to train: 'rf' = Random Forest, 'xgb' = XGBoost (default: xgb)"
)
args = parser.parse_args()

# ── Load data ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_dataset.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {CSV_PATH}.\n"
        "Please place 'credit_risk_dataset.csv' in the same directory as this script."
    )

df = pd.read_csv(CSV_PATH)
print(f"✅  Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Preprocessing ──────────────────────────────────────────────────────────────
df["Income"] = df["Income"].fillna(df["Income"].mean())

edu_order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
df["Education_Level"] = df["Education_Level"].map(edu_order)

df = pd.get_dummies(df, columns=["Housing_Status"], drop_first=True, dtype=int)

FEATURE_COLS = [
    "Age", "Income", "Loan_Amount", "Credit_Score",
    "Employment_Years", "Education_Level",
    "Housing_Status_Own", "Housing_Status_Rent",
]

X = df[FEATURE_COLS]
y = df["Default"]

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / Test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=10, stratify=y
)

# ── Select & Train model ───────────────────────────────────────────────────────
if args.model == "xgb":
    try:
        from xgboost import XGBClassifier
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model_name = "XGBoost"
    except ImportError:
        print("⚠️  xgboost not installed — falling back to Random Forest.")
        print("   Install with: pip install xgboost")
        args.model = "rf"

if args.model == "rf":
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model_name = "Random Forest"

print(f"\n🚀  Training {model_name}...")
clf.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print(f"\n{'='*50}")
print(f"  {model_name} — Model Performance")
print(f"{'='*50}")
print(f"  Accuracy  : {clf.score(X_test, y_test):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
print(f"\nTop Feature Importances:")
importances = clf.feature_importances_
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<22} {bar} {imp:.4f}")

# ── Save artifacts ─────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(clf,    os.path.join(out_dir, "model.pkl"))
joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

# Save model metadata for the UI
import json
meta = {
    "model_type": model_name,
    "model_flag": args.model,
    "features": FEATURE_COLS,
    "accuracy": round(clf.score(X_test, y_test), 4),
    "roc_auc":  round(roc_auc_score(y_test, y_proba), 4),
}
with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅  Saved model.pkl, scaler.pkl, model_meta.json  [{model_name}]")
