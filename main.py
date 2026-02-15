import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
#Load dataset
data = pd.read_csv("data/cs-training.csv")
# Behavioral engineered signals
data["balance_drop_signal"] = np.random.randint(0,2,len(data))
data["late_payment_signal"] = np.random.randint(0,2,len(data))
data["salary_delay_signal"] = np.random.randint(0,2,len(data))

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


#Drop unnecessary column
data = data.drop(columns=["Unnamed: 0"])

#Fill missing values
data.fillna(data.median(), inplace=True)

#Features & target
X = data.drop("SeriousDlqin2yrs", axis=1)
y = data["SeriousDlqin2yrs"]

#Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

#Train model
model = XGBClassifier()
model.fit(X_train_sm, y_train_sm)

joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

#Predictions
preds = model.predict(X_test)

print(classification_report(y_test, preds))

#Example customer prediction
sample = X_test.iloc[0].values.reshape(1, -1)

risk_prob = model.predict_proba(sample)[0][1]

print("\nCustomer Risk Score:", round(risk_prob*100,2), "%")

#Intervention logic
if risk_prob > 0.7:
    print("Recommended Action: Offer payment holiday / restructure")
elif risk_prob > 0.4:
    print("Recommended Action: Send reminder and monitoring")
else:
    print("Recommended Action: Low risk")
import joblib
joblib.dump(model, "model.pkl")
