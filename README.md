# Customer-churn-risk
# Customer Churn Prediction & Retention System
### SQL + Python + Machine Learning + Power BI + GenAI

**Business question:** Of 7,043 telecom customers, which are likely to leave, why, and what should the business do about it?

## Architecture

```
Raw customer data (IBM Telco Customer Churn dataset)
      |
SQL Database (SQLite) — business views answer "why" before any ML
      |
Data Cleaning (Python / Pandas)
      |
Feature Engineering (19 features: tenure, charges, contract, services, etc.)
      |
XGBoost Classifier -> Churn Probability per customer
      |
SHAP Explainability -> per-customer "top churn driver"
      |
Risk Segmentation (High / Medium / Low)
      |
Power BI Dashboard  <-->  GenAI Explanation & Recommendations
      |
Business Decision
```

## Key results

| Metric | Value |
|---|---|
| Dataset | IBM Telco Customer Churn (7,043 customers, 33 columns incl. CLTV, Churn Reason) |
| Model | XGBoost Classifier |
| Test ROC-AUC | **0.854** |
| Churn rate — Month-to-month contracts | **42.7%** |
| Churn rate — Two-year contracts | **2.8%** |
| High Risk customers | **1,126** (16%) |
| CLTV at risk (High Risk segment) | **₹4.50M** |
| Top churn drivers (SHAP) | Contract type (Month-to-month), Tenure, Payment Method (Electronic check) |

**Headline finding (pure SQL, before any ML):** Month-to-month + Fiber-optic customers who churned represent ₹4.73M in lifetime value and ₹100,482/month in lost recurring revenue — the single biggest lever in the dataset.

## Project structure

```
sql/            -- SQLite schema + business views (churn by contract, revenue at risk)
python/          -- data cleaning, feature engineering, XGBoost training, SHAP explainability, risk segmentation
genai/           -- Claude API script that turns segment stats into plain-English "why" + recommendation
powerbi/         -- exported CSVs (scored_customers.csv, powerbi_segment_summary.csv) + dashboard screenshot
```

## Power BI Dashboard

Single-page dashboard built on the model's output:
- **KPI cards:** Total Customers, High Risk Customers, CLTV at Risk
- **Risk Segment donut** (color-coded red/amber/green)
- **Churn Rate by Contract Type** bar chart (the clearest single visual in the project — 42.7% vs 2.8%)
- **Top Churn Drivers by Risk Segment** — SHAP output visualized
- **Customer Drilldown table** — sortable by churn probability, for a retention team to work from directly
- **Risk Segment slicer** — filters every visual on the page interactively

![Dashboard](powerbi/dashboard_screenshot.png)

## Why this design

- **SQL first, not last:** business views answer "why are they leaving" in plain SQL before any model is trained — a stakeholder gets a real answer even if the ML step were removed entirely.
- **Explainability over accuracy alone:** SHAP gives a per-customer top driver, not just a probability score — a retention team needs to know *why* a specific customer is at risk to design the right offer for them.
- **GenAI does explanation, not prediction:** the LLM never guesses who will churn — the XGBoost model does that. GenAI's role is turning segment statistics into a narrative and a concrete recommendation a non-technical stakeholder can act on immediately.
- **Handled real data quality issues, not a toy dataset:** `Total Charges` had 11 blank values (new customers with zero tenure) — handled with domain reasoning (filled with first month's charge) rather than a blind median/zero fill.

## How to run it

```bash
pip install pandas scikit-learn xgboost shap anthropic
python python/01_load_to_sql.py
python python/02_train_and_score.py
export ANTHROPIC_API_KEY=sk-ant-...
python genai/03_generate_insights.py
```
