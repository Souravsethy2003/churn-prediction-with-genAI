import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE / "model"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR = BASE / "static" / "charts"; CHART_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Clean TotalCharges (some blanks)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    # Target -> 0/1
    df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)

    # Normalize service columns: "No internet/phone service" -> "No"
    svc_replace = {"No internet service": "No", "No phone service": "No"}
    for col in [
        "MultipleLines","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"
    ]:
        if col in df.columns:
            df[col] = df[col].replace(svc_replace)

    return df

def evaluation(y_true, probs, preds):
    return dict(
        roc_auc=float(roc_auc_score(y_true, probs)),
        precision=float(precision_score(y_true, preds)),
        recall=float(recall_score(y_true, preds)),
        f1=float(f1_score(y_true, preds)),
    )

def main():
    df = load_and_clean(DATA)

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn", "customerID"])  # don't use identifier

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    lr = Pipeline([("pre", pre),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    rf = Pipeline([("pre", pre),
                   ("clf", RandomForestClassifier(
                        n_estimators=500, random_state=42, n_jobs=-1,
                        class_weight="balanced_subsample"))])

    models = [("LogisticRegression", lr), ("RandomForest", rf)]
    if HAS_XGB:
        xgb = Pipeline([("pre", pre),
                        ("clf", XGBClassifier(
                            n_estimators=600, max_depth=5, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, random_state=42,
                            eval_metric="auc", n_jobs=-1, tree_method="hist"))])
        models.append(("XGBoost", xgb))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    probs_map = {}
    for name, pipe in models:
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        m = evaluation(y_test, probs, preds)
        results.append((name, m))
        probs_map[name] = probs
        print(f"{name}: {m}")

    # pick best by ROC-AUC
    best_name, best_metrics = max(results, key=lambda t: t[1]["roc_auc"])
    best_model = dict(models)[best_name]
    joblib.dump(best_model, MODEL_DIR / "telco_best_model.pkl")

    # charts
    # ROC
    plt.figure()
    for name in probs_map:
        fpr, tpr, _ = roc_curve(y_test, probs_map[name])
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test, probs_map[name]):.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(CHART_DIR/"roc_curve.png", dpi=150); plt.close()

    # PR
    plt.figure()
    for name in probs_map:
        prec, rec, _ = precision_recall_curve(y_test, probs_map[name])
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(CHART_DIR/"pr_curve.png", dpi=150); plt.close()

    # Class distribution
    plt.figure()
    vals = df["Churn"].value_counts().sort_index()
    plt.bar(["Not Churn","Churn"], vals.values); plt.title("Churn Class Distribution")
    plt.tight_layout(); plt.savefig(CHART_DIR/"churn_distribution.png", dpi=150); plt.close()

    # Save metrics JSON
    with open(BASE/"metrics.json","w") as f:
        json.dump({name:m for name,m in results} | {"best_model": best_name}, f, indent=2)

    print("\nBest:", best_name)
    print("Saved pickle ->", MODEL_DIR/"telco_best_model.pkl")
    print("Charts ->", CHART_DIR)
    print("Metrics ->", BASE/"metrics.json")

if __name__ == "__main__":
    main()


