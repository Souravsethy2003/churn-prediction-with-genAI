import argparse
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

def clean_like_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])
    svc_replace = {"No internet service": "No", "No phone service": "No"}
    for col in [
        "MultipleLines","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"
    ]:
        if col in df.columns:
            df[col] = df[col].replace(svc_replace)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/telco_best_model.pkl")
    parser.add_argument("--input", required=True)   # CSV with Telco columns
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)
    df = clean_like_training(df)

    X = df.drop(columns=[c for c in ["Churn"] if c in df.columns])
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = pd.DataFrame({
        "customerID": df["customerID"] if "customerID" in df.columns else np.arange(len(df)),
        "churn_probability": probs.round(4),
        "prediction": preds
    })
    out.to_csv(args.out, index=False)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
