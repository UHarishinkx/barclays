import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Pre-Delinquency Intervention Engine", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:wght@500&display=swap');
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    :root {
        --bank-navy: #0b1f3b;
        --bank-teal: #0a7b83;
        --bank-gold: #c9a227;
        --bank-ice: #f5f7fa;
        --bank-slate: #5a6a7a;
    }
    .stApp {
        background: radial-gradient(circle at 15% 20%, #f2f6fb 0%, #ffffff 45%, #f7f9fc 100%);
    }
    .hero {
        padding: 1rem 1.2rem;
        background: linear-gradient(120deg, var(--bank-navy), #142a4f);
        border-radius: 14px;
        color: #ffffff;
    }
    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 0.25rem;
        font-family: 'Source Serif 4', serif;
    }
    .hero p {
        color: #d6e1f0;
        margin-top: 0;
    }
    .section-title {
        margin-top: 0.5rem;
        font-weight: 600;
        color: var(--bank-navy);
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--bank-teal);
        border-bottom: 3px solid var(--bank-teal);
    }
    .stDataFrame, .stTable {
        border-radius: 12px;
        border: 1px solid #e3e8ef;
        background: #ffffff;
    }
    .stDataFrame [data-testid="stDataFrame"],
    .stTable [data-testid="stTable"] {
        background: #ffffff;
    }
    .stDataFrame thead tr th,
    .stTable thead tr th {
        background: #eef3f8;
        color: var(--bank-navy);
    }
    .stDataFrame tbody tr td,
    .stTable tbody tr td {
        background: #ffffff;
        color: var(--bank-navy);
    }
    .stDataFrame tbody tr:nth-child(even) td,
    .stTable tbody tr:nth-child(even) td {
        background: #f7f9fc;
    }
    .stDataFrame tbody tr:hover td,
    .stTable tbody tr:hover td {
        background: #eef6f7;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        padding: 0.75rem;
        box-shadow: 0 6px 18px rgba(11, 31, 59, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: var(--bank-slate);
    }
    div[data-testid="stMetric"] div {
        color: var(--bank-navy);
    }
    div[data-testid="stSidebar"] {
        background: #f3f6fb;
        border-right: 1px solid #e4e9f0;
    }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] p {
        color: var(--bank-navy);
    }
    div[data-baseweb="slider"] > div > div {
        background-color: #d6e3ef;
    }
    div[data-baseweb="slider"] [role="slider"] {
        background-color: var(--bank-teal);
        border: 2px solid #ffffff;
        box-shadow: 0 0 0 2px rgba(10, 123, 131, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Pre-Delinquency Intervention Engine</h1>
        <p>Early warning signals, clear drivers, and simple playbooks before a payment is missed.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = joblib.load("model.pkl")
artifacts = joblib.load("artifacts.pkl")

RENAME_MAP = artifacts["rename_map"]
STRESS_COMPONENTS = artifacts["stress_components"]
METRICS = artifacts.get("metrics", {})


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


def categorize(r):
    if r > artifacts["thresholds"]["high"]:
        return "High Risk"
    if r > artifacts["thresholds"]["medium"]:
        return "Medium Risk"
    return "Low Risk"


# Load and score the full dataset.
data = pd.read_csv("data/cs-training.csv")
data = clean_and_engineer(data, artifacts)
X = data[artifacts["features"]]
data["risk_score"] = model.predict_proba(X)[:, 1]
data["risk_category"] = data["risk_score"].apply(categorize)

st.sidebar.header("Controls")
risk_filter = st.sidebar.multiselect(
    "Risk tiers",
    ["High Risk", "Medium Risk", "Low Risk"],
    default=["High Risk", "Medium Risk", "Low Risk"],
)
min_score = st.sidebar.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.01)

if METRICS:
    st.sidebar.subheader("Model snapshot")
    st.sidebar.write(f"PR-AUC: {METRICS.get('pr_auc', 0):.3f}")
    st.sidebar.write(f"ROC-AUC: {METRICS.get('roc_auc', 0):.3f}")
    st.sidebar.write(f"Target recall: {METRICS.get('target_recall', 0):.2f}")
    st.sidebar.write(f"Threshold: {METRICS.get('threshold', 0):.3f}")

filtered = data[
    data["risk_category"].isin(risk_filter) & (data["risk_score"] >= min_score)
]

portfolio_tab, customer_tab, drivers_tab = st.tabs(
    ["Portfolio", "Customer", "Drivers"]
)

with portfolio_tab:
    st.markdown("<div class='section-title'>Portfolio view</div>", unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Customers", f"{len(filtered):,}")
    col_b.metric("High Risk", int((filtered["risk_category"] == "High Risk").sum()))
    col_c.metric("Medium Risk", int((filtered["risk_category"] == "Medium Risk").sum()))
    col_d.metric("Low Risk", int((filtered["risk_category"] == "Low Risk").sum()))

    chart_col, dist_col = st.columns([2, 1])
    with chart_col:
        st.subheader("Risk distribution")
        if len(filtered) == 0:
            st.info("No customers match the current filters.")
        else:
            st.bar_chart(filtered["risk_category"].value_counts())
    with dist_col:
        st.subheader("Stress signal heat")
        if len(filtered) == 0:
            st.info("No data to summarize.")
        else:
            signal_rates = {
                "Utilization Spike": filtered["utilization_spike"].mean(),
                "Thin File": filtered["thin_file"].mean(),
                "Payment Miss Ratio": filtered["payment_miss_ratio"].mean(),
                "Income Fragility": filtered["income_fragility"].mean(),
            }
            st.dataframe(
                pd.DataFrame(
                    {
                        "Signal": list(signal_rates.keys()),
                        "Rate": [round(v, 3) for v in signal_rates.values()],
                    }
                )
            )

    with st.expander("Preview customers"):
        st.dataframe(
            filtered[
                [
                    "risk_score",
                    "risk_category",
                    "stress_index",
                    "utilization_spike",
                    "payment_miss_ratio",
                ]
            ].head(50)
        )

with customer_tab:
    st.markdown("<div class='section-title'>Customer risk explorer</div>", unsafe_allow_html=True)
    if len(filtered) == 0:
        st.info("Adjust filters to load customers.")
    else:
        index = st.slider("Customer index", 0, len(filtered) - 1, 0)
        row = filtered.iloc[index]

        col1, col2, col3 = st.columns([1, 1, 2])
        col1.metric("Risk Score", f"{row['risk_score']:.3f}")
        col2.metric("Risk Tier", row["risk_category"])
        col3.metric("Stress Index", f"{row['stress_index']:.3f}")

        st.subheader("Early warning signals")
        signals = {
            "Utilization Spike": int(row["utilization_spike"]),
            "Thin File": int(row["thin_file"]),
            "Payment Miss Ratio": round(float(row["payment_miss_ratio"]), 3),
            "Late Severity": float(row["late_severity"]),
            "Income Fragility": round(float(row["income_fragility"]), 3),
        }
        st.dataframe(pd.DataFrame([signals]))

        st.subheader("Intervention playbook")
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

with drivers_tab:
    st.markdown("<div class='section-title'>Explainable drivers</div>", unsafe_allow_html=True)
    feature_importance = pd.Series(
        model.feature_importances_, index=artifacts["features"]
    ).sort_values(ascending=False)
    top_global = feature_importance.head(8)

    col_g, col_l = st.columns(2)
    with col_g:
        st.caption("Global top drivers")
        st.dataframe(
            top_global.reset_index().rename(columns={"index": "feature", 0: "importance"})
        )

    if len(filtered) == 0:
        with col_l:
            st.caption("Local top drivers")
            st.info("Select customers to view local drivers.")
    else:
        row = filtered.iloc[0]
        z_scores = (X - X.mean()) / X.std(ddof=0)
        local_contrib = (z_scores.loc[row.name] * feature_importance).sort_values(
            ascending=False
        )
        top_local = local_contrib.head(5)
        with col_l:
            st.caption("Local top drivers (proxy)")
            st.dataframe(
                top_local.reset_index().rename(columns={"index": "feature", 0: "score"})
            )
