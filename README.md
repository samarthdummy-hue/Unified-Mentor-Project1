# Bank Customer Churn — Predictive Risk Intelligence

A churn-prediction model and live Streamlit dashboard for the European_Bank
dataset (10,000 retail-banking customers, 20.4% churn).

## What it does
- **Risk calculator & what-if simulator** — enter a customer, get a churn
  probability, sweep any variable to see how risk moves.
- **Portfolio analytics** — scores the entire book live, shows the probability
  distribution, churn by geography, and the highest-risk customers.
- **Explainability** — permutation importance + SHAP summary.
- **Model card** — ROC-AUC, recall/precision at the chosen threshold, confusion
  matrix, and the imbalance baseline.

## Project layout
```
churn_app/
  European_Bank.csv      dataset
  features.py            shared feature engineering (used by training AND app)
  train.py               trains/compares models, tunes threshold, saves artifacts
  explain.py             generates the SHAP summary figure
  app.py                 the Streamlit app
  requirements.txt       pinned, validated versions
  artifacts/             generated: pipeline.pkl, metadata.json, *.png
```

## Run locally
```bash
pip install -r requirements.txt
python train.py        # produces artifacts/pipeline.pkl + metadata.json + importances.png
python explain.py      # produces artifacts/shap_summary.png   (optional but recommended)
streamlit run app.py
```
`artifacts/` is already included, so you can skip straight to `streamlit run app.py`.
Re-run `train.py` only if you change the data or modeling.

## Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo (include `artifacts/`).
2. On share.streamlit.io, create an app pointing at `app.py`.
3. Set the Python version to **3.12** in advanced settings.

## Key findings (use these in your writeup — they're what the model actually shows)
- Best model: **Gradient Boosting**, ROC-AUC **0.867**.
- **Accuracy is misleading here.** Predicting "nobody churns" scores ~79.6%
  on this imbalanced data, so the headline metrics are ROC-AUC and recall.
- Top churn drivers: **Age, NumOfProducts, Geography (Germany), engagement,
  Balance.** Age dominates — i.e. churn is driven substantially by demographics,
  which *contradicts* the original brief's stated conclusion. Report what the
  data shows, not the brief's assumption.
- The decision threshold (default F1-optimal ≈ 0.60) is adjustable in the app.
  Lower it to raise recall (catch more churners at the cost of more false alarms)
  — that tradeoff is the real retention-budget decision.

## Design notes
- One feature-engineering function (`features.add_features`) is used in both
  training and inference, so live predictions transform inputs identically to
  training. This is the main thing people get wrong; it's deliberately central here.
- SHAP is computed once at training time and saved as a PNG, so the deployed app
  doesn't need `shap` at runtime and won't time out.
- Class imbalance handled with class weights / sample weights, not SMOTE, to keep
  the inference path clean.

## Engineered feature definitions (brief was vague; these are the choices made)
- `BalanceSalaryRatio = Balance / (EstimatedSalary + 1)`
- `ProductDensity = NumOfProducts / (Tenure + 1)`  (products per year held)
- `EngagementProduct = IsActiveMember * NumOfProducts`
- `AgeTenureInteraction = Age * Tenure`
- `ZeroBalanceFlag = 1 if Balance == 0` (zero balance is a real state, not missing)
