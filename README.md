# Pre-Delinquency Intervention Engine

## Overview
Banks usually intervene only after a missed payment, when recovery is expensive and customer trust is already damaged. This project predicts delinquency risk 2 to 4 weeks earlier by detecting subtle stress signals in customer credit behavior. It produces explainable risk drivers and triggers proactive, empathetic interventions (payment holiday, nudges, monitoring).

## What the System Does
- Detects early stress signals before delinquency.
- Scores customers with a predictive model built on engineered behavioral features.
- Uses data-driven thresholds to balance recall and precision.
- Surfaces explainable drivers and an intervention playbook in a dashboard.

## Key Signals and Features
The model derives structured indicators from the source credit dataset, including:
- Late severity and missed payment ratios
- Income fragility (dependents vs income)
- Utilization spike and thin credit file flags
- A single stress index combining normalized components

## Model and Metrics
- Model: XGBoost (well-suited for tabular credit risk data)
- Class imbalance: SMOTE during training
- Thresholding: chosen to meet a target recall for early detection
- Reported metrics:
  - Precision, recall, F1
  - PR-AUC and ROC-AUC
  - Confusion matrix
  - Recall at top K percent (capacity-based outreach)

## Repository Structure
- app.py: Streamlit dashboard for risk insights and interventions
- main.py: Training pipeline, feature engineering, evaluation, and artifact export
- requirements.txt: Python dependencies
- data/cs-training.csv: Source dataset

## How to Run
Create a virtual environment, install dependencies, train the model, then run the dashboard.

```powershell
Set-Location -Path f:/barclays_hackathon/barclays
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe main.py
.venv/Scripts/python.exe -m streamlit run app.py
```

## Dashboard Highlights
- Portfolio KPIs: high, medium, low risk counts
- Risk distribution and stress signal heat
- Customer explorer with local driver explanation
- Intervention playbook by risk tier

## Why It Is Industry-Ready
- Explainable, transparent risk signals for compliance
- Early intervention focus to reduce losses and collections cost
- Scalable design for real-time scoring and outreach
- Metrics aligned to business constraints (top-K recall)

## Notes
- Model artifacts are generated during training and intentionally excluded from version control.
- Update the target recall in main.py to match operational capacity.
