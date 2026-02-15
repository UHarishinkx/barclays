import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Pre-Delinquency Early Risk Monitoring Dashboard")

model = joblib.load("model.pkl")

data = pd.read_csv("data/cs-training.csv")

data.rename(columns={
    "RevolvingUtilizationOfUnsecuredLines": "credit_utilization",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59_days",
    "DebtRatio": "emi_income_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "active_credit_accounts",
    "NumberOfTimes90DaysLate": "late_90_days",
    "NumberRealEstateLoansOrLines": "property_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89_days",
    "NumberOfDependents": "dependents"
}, inplace=True)

data["balance_drop_signal"] = 0
data["late_payment_signal"] = 0
data["salary_delay_signal"] = 0

X = data.drop(["SeriousDlqin2yrs","Unnamed: 0"], axis=1)

data["risk_score"] = model.predict_proba(X)[:,1]

def categorize(r):
    if r > 0.7:
        return "High Risk"
    elif r > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

data["risk_category"] = data["risk_score"].apply(categorize)

st.subheader("Customer Risk Overview")
st.dataframe(data[["risk_score","risk_category"]].head(50))

st.subheader("Risk Distribution")
st.bar_chart(data["risk_category"].value_counts())
