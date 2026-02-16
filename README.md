# Pre-Delinquency Intervention Engine

## Latest Model Snapshot
```
precision    recall  f1-score   support

			  0       0.97      0.84      0.90     27995
			  1       0.24      0.70      0.35      2005

	 accuracy                           0.83     30000
	macro avg       0.61      0.77      0.63     30000
weighted avg       0.93      0.83      0.86     30000

PR-AUC: 0.3597
ROC-AUC: 0.8548
Confusion Matrix:
 [[23427  4568]
 [  601  1404]]
Recall@Top5%: 33.97 %
Recall@Top10%: 51.77 %

Customer Risk Score: 2.53 %
Recommended Action: Low risk
```

## Why This Exists
Collections start too late. By the time a payment is missed, recovery is harder, more expensive, and trust is already eroded. This project moves intervention upstream by predicting delinquency risk **2 to 4 weeks early**, using explainable behavioral signals and a simple, action-ready playbook.

## What You Get
- Early warning risk scores for every customer
- Explainable stress signals (not a black box)
- Data-driven thresholds aligned to outreach capacity
- A dashboard that connects risk, reason, and action

## How It Works (Plain English)
1. **Clean + standardize** the credit dataset.
2. **Engineer stress signals** like late severity, utilization spikes, and income fragility.
3. **Train an XGBoost model** with imbalance handling (SMOTE).
4. **Pick a threshold** that hits a target recall so the team catches most at-risk customers.
5. **Deploy into Streamlit** with explainable drivers and intervention playbooks.

## Signals We Surface
- Late severity and missed payment ratios
- Income fragility (dependents vs income)
- Utilization spike and thin credit file flags
- A single **stress index** that blends normalized signals

## Model Trust and Evaluation
- **PR-AUC and ROC-AUC** for ranking quality
- **Recall@Top-K** to match outreach capacity
- **Confusion matrix** for transparency on false positives
- **Target recall** ensures early detection is prioritized

## Repository Structure
- app.py: Streamlit dashboard for risk insights and interventions
- main.py: Training pipeline, feature engineering, evaluation, and artifact export
- requirements.txt: Python dependencies
- data/cs-training.csv: Source dataset

## Quickstart
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

## Why This Is Industry-Ready
- Explainable, transparent signals support compliance and fairness
- Early intervention reduces losses and collections cost
- Scalable approach for real-time scoring and outreach
- Metrics align with real-world capacity constraints

## Notes
- Model artifacts are generated during training and intentionally excluded from version control.
- Update the target recall in main.py to match operational capacity.
