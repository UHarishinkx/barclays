import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

RENAME_MAP = {
    "RevolvingUtilizationOfUnsecuredLines": "credit_utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59_days",
    "DebtRatio": "emi_income_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "active_credit_accounts",
    "NumberOfTimes90DaysLate": "late_90_days",
    "NumberRealEstateLoansOrLines": "property_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89_days",
    "NumberOfDependents": "dependents",
}

STRESS_COMPONENTS = [
    "credit_utilization",
    "emi_income_ratio",
    "late_severity",
    "income_fragility",
]

STRESS_WEIGHTS = {
    "credit_utilization": 0.35,
    "emi_income_ratio": 0.25,
    "late_severity": 0.25,
    "income_fragility": 0.15,
}

# Target recall keeps the model aligned to outreach capacity.
TARGET_RECALL = 0.7


def _minmax_scale(series, min_val, max_val):
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return (series - min_val) / denom


def clean_and_engineer(df, stats=None):
    """Clean the data and build human-readable stress features."""
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

    if stats is None:
        minmax = {}
        for col in STRESS_COMPONENTS:
            minmax[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        stats = {"minmax": minmax}

    scaled = {}
    for col in STRESS_COMPONENTS:
        scaled[col] = _minmax_scale(
            df[col], stats["minmax"][col]["min"], stats["minmax"][col]["max"]
        )

    df["stress_index"] = (
        STRESS_WEIGHTS["credit_utilization"] * scaled["credit_utilization"]
        + STRESS_WEIGHTS["emi_income_ratio"] * scaled["emi_income_ratio"]
        + STRESS_WEIGHTS["late_severity"] * scaled["late_severity"]
        + STRESS_WEIGHTS["income_fragility"] * scaled["income_fragility"]
    )

    return df, stats


def recall_at_k(y_true, y_score, k_percent):
    """Measure how many delinquents appear in the top K percent risk list."""
    k_count = max(1, int(len(y_score) * k_percent / 100))
    top_idx = np.argsort(y_score)[-k_count:]
    captured = y_true.iloc[top_idx].sum()
    total = y_true.sum()
    return float(captured / total) if total > 0 else 0.0


def find_threshold_for_recall(y_true, y_score, target_recall):
    """Pick the lowest threshold that still meets target recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return 0.5
    return float(thresholds[valid[-1]])


def tune_params(X_train_sm, y_train_sm, X_val, y_val):
    """Lightweight hyperparameter search focused on PR-AUC."""
    grid = [
        {"max_depth": 3, "n_estimators": 250, "learning_rate": 0.05},
        {"max_depth": 4, "n_estimators": 300, "learning_rate": 0.05},
        {"max_depth": 5, "n_estimators": 350, "learning_rate": 0.05},
        {"max_depth": 4, "n_estimators": 400, "learning_rate": 0.03},
    ]
    best_score = -1.0
    best_params = None
    for params in grid:
        model = XGBClassifier(
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train_sm, y_train_sm)
        proba_val = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, proba_val)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params


# Load dataset and engineer explainable signals.
data = pd.read_csv("data/cs-training.csv")
data, stats = clean_and_engineer(data)

X = data.drop("SeriousDlqin2yrs", axis=1)
y = data["SeriousDlqin2yrs"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Handle class imbalance for rare delinquency events.
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

best_params = tune_params(X_train_sm, y_train_sm, X_val, y_val)

X_train_final = pd.concat([X_train, X_val], ignore_index=True)
y_train_final = pd.concat([y_train, y_val], ignore_index=True)
X_train_final_sm, y_train_final_sm = smote.fit_resample(X_train_final, y_train_final)

model = XGBClassifier(
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    **best_params,
)
model.fit(X_train_final_sm, y_train_final_sm)

# Core evaluation metrics.
proba_test = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, proba_test)
roc_auc = roc_auc_score(y_test, proba_test)

threshold = find_threshold_for_recall(y_test, proba_test, TARGET_RECALL)
if threshold <= 0:
    threshold = 0.5

high_threshold = max(0.7, threshold)
medium_threshold = max(0.4, min(0.6, high_threshold * 0.7))

preds = (proba_test >= threshold).astype(int)
print(classification_report(y_test, preds))
print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

cm = confusion_matrix(y_test, preds)
print("Confusion Matrix:\n", cm)

print("Recall@Top5%:", round(recall_at_k(y_test, proba_test, 5) * 100, 2), "%")
print("Recall@Top10%:", round(recall_at_k(y_test, proba_test, 10) * 100, 2), "%")

artifacts = {
    "features": X.columns.tolist(),
    "minmax": stats["minmax"],
    "thresholds": {"high": high_threshold, "medium": medium_threshold},
    "rename_map": RENAME_MAP,
    "stress_components": STRESS_COMPONENTS,
    "metrics": {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "target_recall": TARGET_RECALL,
        "threshold": float(threshold),
    },
    "best_params": best_params,
}

joblib.dump(model, "model.pkl")
joblib.dump(artifacts, "artifacts.pkl")

sample = X_test.iloc[0].values.reshape(1, -1)
risk_prob = model.predict_proba(sample)[0][1]
print("\nCustomer Risk Score:", round(risk_prob * 100, 2), "%")

if risk_prob > high_threshold:
    print("Recommended Action: Offer payment holiday / restructure")
elif risk_prob > medium_threshold:
    print("Recommended Action: Send reminder and monitoring")
else:
    print("Recommended Action: Low risk")
