# app/app.py
from flask import Flask, request, jsonify, render_template, url_for
from pathlib import Path
from io import BytesIO
import pandas as pd, joblib
import time
import matplotlib.pyplot as plt
from collections import Counter
import sys
import os

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "model" / "telco_best_model.pkl"
STATIC_DIR = BASE / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"
RESULTS_DIR = STATIC_DIR / "results"
CHART_DIR = STATIC_DIR / "charts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__,
            static_folder=str(STATIC_DIR),
            template_folder=str(TEMPLATES_DIR)
)

# import genai function (assumes file app/genai_gemini.py exists)
try:
    from genai_google import generate_aggregated_recommendations
except Exception:
    # fallback stub
    def generate_aggregated_recommendations(top_reasons_list, counts_str, summary_stats):
        return "GenAI is not available (server-side). Using fallback suggestions.\n\n" + \
               "Short-term: Run targeted emails to customers missing security.\n" + \
               "Mid-term: Incentivize conversion from month-to-month to annual.\n" + \
               "Long-term: Improve onboarding & tech-support SLAs."

# cleaning function (same as training)
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

def recommend(row, prob):
    recs = []
    if prob >= 0.8:
        recs += ["Immediate retention call + limited-time 30% discount",
                 "Priority support review of recent issues"]
    elif prob >= 0.6:
        recs += ["Targeted re-engagement email + 14-day add-on trial",
                 "Offer annual plan with incentive"]
    elif prob >= 0.4:
        recs += ["Education-series emails (how-to tips)"]
    else:
        recs += ["Standard engagement; invite referral program"]
    if row.get("tenure", 999) < 6: recs.append("Welcome concierge for new users")
    if row.get("MonthlyCharges", 0) > 90: recs.append("Price-value messaging")
    if row.get("InternetService", "") == "Fiber optic": recs.append("Promote speed-feature education")
    return recs

def load_model_or_exit(path: Path):
    if not path.exists():
        print(f"[ERROR] Model file not found at: {path}\nRun training script to create it.", file=sys.stderr)
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)

# Load model once (fail fast)
model = None
try:
    model = load_model_or_exit(MODEL_PATH)
except Exception:
    model = None
    print("Warning: model not loaded; bulk predict will fail until model is available.", file=sys.stderr)

# helper functions
def drivers(row):
    d = []
    if row.get("Contract","") == "Month-to-month": d.append("Short contract")
    pm = str(row.get("PaymentMethod",""))
    if pm.lower().startswith("electronic"): d.append("Electronic check")
    if row.get("OnlineSecurity","No") == "No": d.append("No online security")
    if row.get("TechSupport","No") == "No": d.append("No tech support")
    if row.get("tenure", 999) < 6: d.append("Very new customer")
    return ", ".join(d[:3]) or "—"

def compute_top_drivers(out_rows, top_k=5):
    c = Counter()
    for r in out_rows:
        td = r.get("top_drivers","")
        parts = [p.strip() for p in td.split(",") if p.strip()]
        c.update(parts)
    top = [(k, v) for k, v in c.most_common(top_k) if k and k != "—"]
    return top

def make_churn_bar_chart(out_df: pd.DataFrame, chart_path: Path):
    counts = out_df['prediction'].value_counts().sort_index()
    labels = ["Not Churn", "Churn"]
    vals = [int(counts.get(0,0)), int(counts.get(1,0))]
    plt.figure(figsize=(4.2,3))
    plt.bar(labels, vals)
    plt.title("Predicted Churn Distribution")
    plt.ylabel("Customers")
    for i, v in enumerate(vals):
        plt.text(i, v + max(vals)*0.01, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()

# routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/model_evaluation")
def model_evaluation():
    # separate page to show static charts
    # templates/evaluation.html will load charts lazily from static/charts/
    return render_template("evaluation.html")

@app.route("/api/bulk_predict", methods=["POST"])
def bulk_predict():
    if model is None:
        return jsonify({"error":"Model not available on server"}), 500

    if "file" in request.files:
        df_in = pd.read_csv(request.files["file"])
    else:
        payload = request.get_json(force=True)
        df_in = pd.DataFrame(payload.get("records", payload))

    df = clean_like_training(df_in)
    X = df.drop(columns=[c for c in ["Churn"] if c in df.columns], errors="ignore")

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        return jsonify({"error":"Prediction failed","details":str(e)}), 500

    preds = (probs >= 0.5).astype(int)
    out_rows = []
    for r, p, y in zip(X.to_dict("records"), probs, preds):
        out_rows.append({
            "customerID": r.get("customerID",""),
            "churn_probability": round(float(p), 4),
            "prediction": int(y),
            "top_drivers": drivers(r),
            "recommendations": " | ".join(recommend(r, float(p)))
        })

    # save CSV
    ts = int(time.time())
    results_filename = f"predictions_{ts}.csv"
    results_path = RESULTS_DIR / results_filename
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(results_path, index=False)

    # churn chart
    chart_filename = f"churn_bar_{ts}.png"
    chart_path = CHART_DIR / chart_filename
    make_churn_bar_chart(out_df, chart_path)

    # aggregate reasons
    top_reasons = compute_top_drivers(out_rows, top_k=5)
    preview = out_rows[:10]

    download_url = url_for('static', filename=f"results/{results_filename}")
    chart_url = url_for('static', filename=f"charts/{chart_filename}")

    return jsonify({
        "download_url": download_url,
        "chart_url": chart_url,
        "top_reasons": top_reasons,
        "preview": preview,
        "rows_scored": len(out_rows)
    })

@app.route("/api/genai_recs", methods=["POST"])
def genai_recs():
    payload = request.get_json(force=True)
    top_reasons = payload.get("top_reasons", [])
    top_list = [r[0] for r in top_reasons]
    counts_str = ", ".join([f"{r[0]} — {r[1]}" for r in top_reasons])
    summary_stats = f"Rows scored: {payload.get('rows_scored','?')}; estimated churn rate: {payload.get('churn_rate_est','?')}"
    text = generate_aggregated_recommendations(top_list, counts_str, summary_stats)
    return jsonify({"text": text})

# simple health
@app.route("/health")
def health():
    return jsonify({"status":"ok", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
