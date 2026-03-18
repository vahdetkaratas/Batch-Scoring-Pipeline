"""
Generate a sample batch CSV (Telco Customer Churn input schema).

From repo root:
  python scripts/generate_sample_batch.py [--rows 120]
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import pandas as pd

random.seed(42)

GENDERS = ["Male", "Female"]
YESNO = ["Yes", "No"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
SERVICE_COLS_DEFAULT = ["No", "Yes"]


def _customer_id() -> str:
    a = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    b = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{a}-{b}"


def _row() -> dict:
    phone = random.choice(YESNO)
    if phone == "No":
        multiline = "No phone service"
    else:
        multiline = random.choice(["No", "Yes"])

    tenure = random.randint(0, 72)
    monthly = round(random.uniform(18.0, 118.5), 2)
    if tenure <= 0:
        total = monthly
    else:
        total = round(monthly * tenure * random.uniform(0.85, 1.05), 2)

    def svc():
        return random.choice(SERVICE_COLS_DEFAULT)

    return {
        "customerID": _customer_id(),
        "gender": random.choice(GENDERS),
        "SeniorCitizen": random.choices([0, 1], weights=[0.84, 0.16])[0],
        "Partner": random.choice(YESNO),
        "Dependents": random.choice(YESNO),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiline,
        "InternetService": random.choice(INTERNET),
        "OnlineSecurity": svc(),
        "OnlineBackup": svc(),
        "DeviceProtection": svc(),
        "TechSupport": svc(),
        "StreamingTV": svc(),
        "StreamingMovies": svc(),
        "Contract": random.choices(CONTRACTS, weights=[0.55, 0.25, 0.20])[0],
        "PaperlessBilling": random.choice(YESNO),
        "PaymentMethod": random.choice(PAYMENT),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate sample batch_001.csv")
    ap.add_argument("--rows", type=int, default=120, help="Number of rows (default 120)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/input_batches/batch_001.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    ids = set()
    rows = []
    while len(rows) < args.rows:
        r = _row()
        if r["customerID"] in ids:
            continue
        ids.add(r["customerID"])
        rows.append(r)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
