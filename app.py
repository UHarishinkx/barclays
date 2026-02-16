import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Pre-Delinquency Intervention Engine", layout="wide")
st.title("Pre-Delinquency Intervention Engine")
st.caption(
    "Early-warning signals, clear drivers, and proactive playbooks before delinquency."
)

model = joblib.load("model.pkl")
artifacts = joblib.load("artifacts.pkl")

RENAME_MAP = artifacts["rename_map"]
STRESS_COMPONENTS = artifacts["stress_components"]


def _minmax_scale(series, min_val, max_val):
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return (series - min_val) / denom


def clean_and_engineer(df, stats):
    """Apply the same feature logic used during training."""
    df = df.copy()
    df.rename(columns=RENAME_MAP, inplace=True)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.fillna(df.median(numeric_only=True), inplace=True)

    df["late_total"] = (
        df["late_30_59_days"] + df["late_60_89_days"] + df["late_90_days"]
    )
    df["late_severity"] = (
        df["late_30_59_days"]
        + 2 * df["late_60_89_days"]
        + 3 * df["late_90_days"]
    )
    df["payment_miss_ratio"] = df["late_total"] / (df["active_credit_accounts"] + 1)
    df["income_fragility"] = (df["dependents"] + 1) / (df["monthly_income"] + 1)
    df["utilization_spike"] = (df["credit_utilization"] > 0.9).astype(int)
    df["thin_file"] = (df["active_credit_accounts"] <= 3).astype(int)

    scaled = {}
    for col in STRESS_COMPONENTS:
        scaled[col] = _minmax_scale(
            df[col], stats["minmax"][col]["min"], stats["minmax"][col]["max"]
        )

    df["stress_index"] = (
        0.35 * scaled["credit_utilization"]
        + 0.25 * scaled["emi_income_ratio"]
        + 0.25 * scaled["late_severity"]
        + 0.15 * scaled["income_fragility"]
    )

    return df


# Load and score the full dataset.
data = pd.read_csv("data/cs-training.csv")
data = clean_and_engineer(data, artifacts)
X = data[artifacts["features"]]
data["risk_score"] = model.predict_proba(X)[:, 1]


def categorize(r):
    if r > artifacts["thresholds"]["high"]:
        return "High Risk"
    if r > artifacts["thresholds"]["medium"]:
        return "Medium Risk"
    return "Low Risk"


data["risk_category"] = data["risk_score"].apply(categorize)

st.subheader("Portfolio View")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Customers", f"{len(data):,}")
col_b.metric("High Risk", int((data["risk_category"] == "High Risk").sum()))
col_c.metric("Medium Risk", int((data["risk_category"] == "Medium Risk").sum()))
col_d.metric("Low Risk", int((data["risk_category"] == "Low Risk").sum()))

chart_col, dist_col = st.columns([2, 1])
with chart_col:
    st.subheader("Risk Distribution")
    st.bar_chart(data["risk_category"].value_counts())
with dist_col:
    st.subheader("Stress Signal Heat")
    signal_rates = {
        "Utilization Spike": data["utilization_spike"].mean(),
        "Thin File": data["thin_file"].mean(),
        "Payment Miss Ratio": data["payment_miss_ratio"].mean(),
        "Income Fragility": data["income_fragility"].mean(),
    }
    st.dataframe(
        pd.DataFrame(
            {
                "Signal": list(signal_rates.keys()),
                "Rate": [round(v, 3) for v in signal_rates.values()],
            }
        )
    )

st.subheader("Customer Risk Explorer")
index = st.slider("Customer Index", 0, len(data) - 1, 0)
row = data.iloc[index]

col1, col2, col3 = st.columns([1, 1, 2])
col1.metric("Risk Score", f"{row['risk_score']:.3f}")
col2.metric("Risk Tier", row["risk_category"])
col3.metric("Stress Index", f"{row['stress_index']:.3f}")

st.subheader("Early Warning Signals")
signals = {
    "Utilization Spike": int(row["utilization_spike"]),
    "Thin File": int(row["thin_file"]),
    "Payment Miss Ratio": round(float(row["payment_miss_ratio"]), 3),
    "Late Severity": float(row["late_severity"]),
    "Income Fragility": round(float(row["income_fragility"]), 3),
}
st.dataframe(pd.DataFrame([signals]))

st.subheader("Explainable Drivers")
feature_importance = pd.Series(
    model.feature_importances_, index=artifacts["features"]
).sort_values(ascending=False)
top_global = feature_importance.head(8)

z_scores = (X - X.mean()) / X.std(ddof=0)
local_contrib = (z_scores.iloc[index] * feature_importance).sort_values(ascending=False)
top_local = local_contrib.head(5)

col_g, col_l = st.columns(2)
with col_g:
    st.caption("Global top drivers")
    st.dataframe(top_global.reset_index().rename(columns={"index": "feature", 0: "importance"}))
with col_l:
    st.caption("Local top drivers (proxy)")
    st.dataframe(top_local.reset_index().rename(columns={"index": "feature", 0: "score"}))

st.subheader("Intervention Playbook")
if row["risk_category"] == "High Risk":
    st.success(
        "Offer payment holiday or restructure, set proactive call, and trigger savings buffer plan."
    )
elif row["risk_category"] == "Medium Risk":
    st.info(
        "Send gentle nudge, recommend budgeting tools, and monitor for 2-week trend shifts."
    )
else:
    st.write("Maintain normal servicing; no intervention required.")

st.subheader("Customer Risk Overview")
st.dataframe(
    data[
        [
            "risk_score",
            "risk_category",
            "stress_index",
            "utilization_spike",
            "payment_miss_ratio",
        ]
    ].head(50)
)
