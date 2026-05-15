import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Helpers ---------------------------------------------------------------

def choose(pairs, size):
    """choose from list of (value, prob). probs need not sum exactly 1."""
    values, probs = zip(*pairs)
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return np.random.choice(values, p=probs, size=size)

def make_customer_ids(n, start=100000):
    return [f"{np.random.randint(1000,9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}"
            for _ in range(n)]

# --- Main synthetic generator ---------------------------------------------

def generate_telco_like(n=1000, include_churn=False, seed=7):
    rng = np.random.default_rng(seed)

    # Distributions approximate the public Telco dataset
    gender = choose([("Female", 0.49), ("Male", 0.51)], n)
    senior = rng.choice([0,1], size=n, p=[0.84, 0.16])  # ~16% seniors
    partner = choose([("Yes", 0.49), ("No", 0.51)], n)
    dependents = choose([("Yes", 0.30), ("No", 0.70)], n)

    # Contract & paperless
    contract = choose([("Month-to-month", 0.55), ("One year", 0.25), ("Two year", 0.20)], n)
    paperless = choose([("Yes", 0.60), ("No", 0.40)], n)

    # Payment methods
    payment_method = choose([
        ("Electronic check", 0.34),
        ("Mailed check", 0.16),
        ("Bank transfer (automatic)", 0.25),
        ("Credit card (automatic)", 0.25)
    ], n)

    # Phone service & multiple lines
    phone_service = choose([("Yes", 0.90), ("No", 0.10)], n)
    multiple_lines = np.empty(n, dtype=object)
    for i in range(n):
        if phone_service[i] == "No":
            multiple_lines[i] = "No phone service"
        else:
            multiple_lines[i] = choose([("No", 0.65), ("Yes", 0.35)], 1)[0]

    # Internet service & dependent value columns
    internet_service = choose([("DSL", 0.34), ("Fiber optic", 0.44), ("No", 0.22)], n)

    def internet_dependent_col():
        out = np.empty(n, dtype=object)
        for i in range(n):
            if internet_service[i] == "No":
                out[i] = "No internet service"
            else:
                out[i] = choose([("Yes", 0.25), ("No", 0.75)], 1)[0]
        return out

    online_security   = internet_dependent_col()
    online_backup     = internet_dependent_col()
    device_protection = internet_dependent_col()
    tech_support      = internet_dependent_col()
    streaming_tv      = internet_dependent_col()
    streaming_movies  = internet_dependent_col()

    # Tenure (months) — skewed right, clipped 0..72
    tenure = rng.exponential(scale=20, size=n).astype(int)
    tenure = np.clip(tenure, 0, 72)

    # MonthlyCharges depend on services
    base = rng.normal(20, 5, size=n)  # base
    net_add = np.array([0 if s == "No" else (20 if s == "DSL" else 30) for s in internet_service])
    add_ons = (
        (streaming_tv == "Yes").astype(int) * 7 +
        (streaming_movies == "Yes").astype(int) * 7 +
        (device_protection == "Yes").astype(int) * 5 +
        (online_security == "Yes").astype(int) * 6 +
        (online_backup == "Yes").astype(int) * 5 +
        (tech_support == "Yes").astype(int) * 6
    )
    monthly = base + net_add + add_ons
    # small adjustment for seniors & contracts
    monthly += (senior == 1) * 2
    monthly += (contract == "Two year") * (-3)
    monthly = np.clip(monthly, 18, 140).round(2)

    # TotalCharges ~ tenure * monthly (+ noise); if tenure==0, set ~ first bill or 0
    noise = rng.normal(0, 30, size=n)
    total = (tenure * monthly + noise).round(2)
    total = np.where(tenure == 0, np.clip(monthly + rng.normal(0,5,size=n), 0, None).round(2), total)
    total = np.clip(total, 0, None)

    # Churn (optional): generate a realistic label using simple risk logic
    if include_churn:
        risk = (
            (contract == "Month-to-month").astype(int) * 0.9 +
            (payment_method == "Electronic check").astype(int) * 0.5 +
            (internet_service == "Fiber optic").astype(int) * 0.2 +
            (tenure < 6).astype(int) * 0.7 +
            (monthly > 95).astype(int) * 0.3
        ) + rng.normal(0, 0.25, size=n)
        # logistic
        prob = 1 / (1 + np.exp(-(risk - 0.6)))
        churn = (rng.random(size=n) < prob).astype(int)
        churn_text = np.where(churn == 1, "Yes", "No")
    else:
        churn_text = None

    # Customer IDs
    customer_id = make_customer_ids(n)

    # Assemble DataFrame with exact Telco column order
    cols = [
        "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract",
        "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"
    ]
    data = {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    if include_churn:
        cols.append("Churn")
        data["Churn"] = churn_text

    df = pd.DataFrame(data, columns=cols)
    return df

# --- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="number of synthetic customers")
    parser.add_argument("--out", type=str, default="synthetic_telco_1000.csv", help="output CSV path")
    parser.add_argument("--include-churn", type=int, default=0, help="1 to include Churn column, 0 to omit")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = generate_telco_like(n=args.n, include_churn=bool(args.include_churn), seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote synthetic Telco CSV: {out_path.resolve()}  (rows={len(df)})")
