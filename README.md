# Lending Club Loan Default Risk Prediction

An end-to-end machine learning project that predicts whether a Lending Club loan will be **charged off**, built on ~1.3M historical loans. It covers data cleaning, feature engineering, leak-free preprocessing, model comparison with hyperparameter tuning, a Streamlit web app, and a FastAPI service packaged with Docker.

**[Live Demo (Streamlit)](https://lending-club-default-predictor-vqdwh43hr7eyqgxilhf8or.streamlit.app/)**

---

## Highlights

- **1.3M+ loans** from the Kaggle Lending Club dataset (2007-2018), reduced to clear final outcomes (Fully Paid vs Charged Off).
- **Leakage-aware feature selection:** only origination-time fields are used. Post-outcome columns such as `last_fico_range_*` are deliberately excluded.
- **Leak-free preprocessing:** a custom `OutlierCapper` (1st/99th percentile winsorizing) is the first step of the sklearn `Pipeline`, so caps are learned from training data only and reused unchanged at inference.
- **Four models compared** (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) with `RandomizedSearchCV` and stratified 5-fold CV. XGBoost selected.
- **Evaluated on the natural class distribution** (~80/20), not a rebalanced test set.
- **Two ways to serve the model:** an interactive Streamlit app and a FastAPI `/predict` endpoint containerized with Docker.

---

## Problem Statement

Lenders need to identify risky borrowers before approving or pricing a loan. This project builds a binary classifier that estimates the probability that a loan ends up charged off.

| Original `loan_status` | Target (`loan_default`) |
|---|---|
| Fully Paid | 0 = Good Loan |
| Charged Off | 1 = Bad Loan |

All other statuses (Current, Late, In Grace Period, etc.) are removed because they are not final outcomes.

**Class balance:** roughly 80% good / 20% bad, so the problem is imbalanced and accuracy alone is misleading.

---

## Dataset

Kaggle: [All Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

This project uses `accepted_2007_to_2018Q4.csv`. The raw file is large and is **not included** in this repository.

---

## Features

23 model inputs across borrower, loan, and credit-profile groups.

| Group | Features |
|---|---|
| Loan | `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `sub_grade`, `purpose` |
| Borrower | `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `application_type` |
| Credit profile | `dti`, `delinq_2yrs`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `credit_history_years` |
| Engineered | `fico_score` (mean of `fico_range_low/high`), `loan_to_income`, `installment_to_income` |

`credit_history_years` is derived from `earliest_cr_line` relative to the loan issue date.

**Leakage note:** `last_fico_range_low/high` and any post-issuance fields are never used because they reflect information from after the loan was issued.

---

## Methodology

### 1. Data preparation (`src/data_preparation.py`)
- Filters to Fully Paid / Charged Off and builds the binary target.
- Cleans percentage strings (`int_rate`, `revol_util`) and converts `emp_length` to numeric years.
- Builds `credit_history_years`, `fico_score`, `loan_to_income`, and `installment_to_income`. Zero income is treated as missing so ratios never become infinite.

### 2. Preprocessing (`src/preprocessing.py`, `src/train.py`)
1. **`OutlierCapper`** - winsorizes 11 continuous columns at the 1st/99th percentile. Sparse or near-binary columns (`term`, `delinq_2yrs`, `pub_rec`, `revol_util`) are excluded because their IQR is not meaningful.
2. **Numeric:** median imputation, then `StandardScaler`.
3. **Categorical:** most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`.
4. **Model.**

All steps live in a single `Pipeline`, so training, evaluation, the API, and the app share identical logic.

### 3. Train/test split
Stratified 80/20 split (`random_state=42`). The test set keeps the natural class distribution.

### 4. Modeling and tuning
Models were compared in `notebooks/00_experiment.ipynb` using `RandomizedSearchCV` with stratified 5-fold CV. Imbalance was handled with `class_weight="balanced"` (LR, RF) and `scale_pos_weight` (XGBoost) instead of discarding rows.

`src/train.py` bakes in the winning hyperparameters, so retraining is fast and reproducible.

---

## Results

Evaluated on the held-out test set (53,712 bad loans / 215,350 good loans):

| Metric (test) | XGBoost (selected) |
|---|---|
| ROC-AUC | **0.724** |
| Recall (bad loans) | 0.681 |
| Precision (bad loans) | 0.323 |
| F1 (bad loans) | 0.438 |
| Accuracy | 0.651 |

Train ROC-AUC is 0.733 vs 0.724 on test, so overfitting is minimal.

**How to read these numbers:**
- The model **catches about 68% of loans that actually default**, at the cost of flagging many good loans (precision 0.32). This is a deliberate recall-first trade-off, since a missed default usually costs more than a rejected good loan.
- Precision is naturally low because only ~20% of loans default. Accuracy is not a useful headline metric here.
- ROC-AUC around 0.72 is realistic for Lending Club using only origination-time features.

Full metrics, per-class reports, and confusion matrices are in `reports/model_metrics.json` and `reports/figures/`.

### Model comparison from the experiment notebook

Tuned models on the held-out test set:

| Model | ROC-AUC | Recall (bad) | F1 (bad) |
|---|---|---|---|
| **XGBoost** | **0.7223** | 0.683 | 0.437 |
| Logistic Regression | 0.7106 | 0.674 | 0.428 |
| Random Forest | 0.7015 | 0.707 | 0.418 |

> These figures come from the experiment run that trained on a **50/50 undersampled training set** and tested on the untouched 80/20 test set. The production pipeline (`train.py`) trains on the full data with `scale_pos_weight`. Results are within about 0.002 ROC-AUC of each other, so undersampling gave no meaningful benefit and the full-data approach was kept.

---

## Model Interpretability (SHAP)

SHAP was used to explain the tuned XGBoost model, both globally and per loan.

**Top drivers of default risk:**
1. `int_rate` - higher rate, higher risk
2. `term` - 60-month loans are riskier than 36-month
3. `fico_score` - higher score, lower risk
4. `dti`, `open_acc`, `loan_to_income` - higher values raise risk
5. Loan `grade` (A and B strongly lower risk), `home_ownership` (MORTGAGE lowers risk, RENT raises it)

**Caveat:** `grade` and `sub_grade` are assigned by Lending Club using inputs such as interest rate, FICO, and DTI, so they partly summarize other features. They are known at origination (not leakage), but the model partly reflects Lending Club's own risk scoring.

---

## Project Structure

```text
lending_club_default_risk_project/
├── app/
│   └── streamlit_app.py          # Interactive prediction UI
├── data/
│   ├── raw/                      # Place Kaggle CSV here (not tracked)
│   └── processed/                # Cleaned CSV (generated, not tracked)
├── models/
│   └── xgboost.joblib            # Trained pipeline
├── notebooks/
│   ├── 00_experiment.ipynb       # Model comparison, tuning, SHAP
│   ├── 01_lending_club_project_walkthrough.ipynb
│   └── 02_EDA.ipynb              # Exploratory analysis
├── reports/
│   ├── figures/                  # Confusion matrices and plots
│   └── model_metrics.json        # Train/test metrics
├── src/
│   ├── config.py                 # Paths, features, model registry
│   ├── data_preparation.py       # Cleaning + feature engineering
│   ├── preprocessing.py          # Custom OutlierCapper transformer
│   ├── eda.py
│   ├── train.py                  # Trains the selected model
│   ├── evaluate.py               # Metrics + plots
│   ├── prediction.py             # Inference helper
│   └── api.py                    # FastAPI service
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Setup

```bash
git clone <your-github-repo-link>
cd lending_club_default_risk_project

python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### 2. Get the data
Download `accepted_2007_to_2018Q4.csv` from the Kaggle link above and place it in `data/raw/`.

### 3. Run the pipeline

```bash
python src/data_preparation.py   # creates data/processed/lending_club_cleaned.csv
python src/eda.py                # saves EDA charts to reports/figures/
python src/train.py              # trains and saves models/xgboost.joblib
python src/evaluate.py           # writes reports/model_metrics.json and plots
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 5. Run the API

```bash
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

Interactive docs: `http://localhost:8000/docs`

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 10000, "term": "36 months", "int_rate": 13.5,
    "installment": 339.31, "grade": "B", "sub_grade": "B2",
    "emp_length": 3, "home_ownership": "RENT", "annual_inc": 60000,
    "verification_status": "Verified", "purpose": "debt_consolidation",
    "dti": 18.5, "delinq_2yrs": 0, "open_acc": 10, "pub_rec": 0,
    "revol_bal": 8000, "revol_util": 45.2, "total_acc": 25,
    "application_type": "Individual", "credit_history_years": 8.0,
    "fico_range_low": 690, "fico_range_high": 694
  }'
```

### 6. Run with Docker

```bash
docker build -t lending-club-api .
docker run -p 8000:8000 lending-club-api
```

---

## Streamlit App

Enter applicant and loan details (or load a built-in demo profile) to get:

- Predicted class (good / bad loan)
- Default probability
- Risk bucket

| Probability | Risk Level |
|---|---|
| < 0.30 | Low Risk |
| 0.30 - 0.60 | Medium Risk |
| > 0.60 | High Risk |

---

## Business Interpretation

A higher predicted probability means the borrower is more likely to be charged off. Possible actions:

- Decline very high-risk applications
- Price medium-risk loans at a higher interest rate
- Reduce approved loan amount
- Request additional documents
- Route borderline cases to manual underwriting

---

## Limitations and Future Work

- **Hyperparameter tuning is limited.** The current search uses `RandomizedSearchCV` with a small number of iterations. A planned experiment will use Bayesian optimization (e.g., **Optuna**) to search the parameter space more efficiently, and its results will be compared against the current XGBoost configuration.
- **Threshold is not tuned.** The default 0.5 cutoff is used. A cost-based threshold (weighing the cost of a missed default vs a rejected good loan) would improve real-world usefulness.
- **No out-of-time validation.** The split is random, not chronological, so performance on future loan vintages is not measured.
- **No probability calibration.** Class weighting shifts predicted probabilities upward, so the app's probabilities are best read as risk scores rather than true default rates.
- **Not production-ready:** fairness/bias testing, monitoring, and drift detection are not implemented.
This project is for educational purposes only and is not a credit decision tool.

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, FastAPI, Docker, Matplotlib, Seaborn

---

## Author

**Souradip Dey**