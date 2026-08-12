# Lending Club Loan Default Risk Prediction
![alt text](<Screenshot 2026-08-08 220002.png>)

## Project Overview

This is an end-to-end machine learning project built on the Kaggle **All Lending Club Loan Data** dataset.

The business objective is to predict whether an accepted loan will become a **bad loan / default-risk loan** using borrower, loan, and credit-profile information available at or near loan origination.

The project includes:

- Data loading and cleaning
- Exploratory Data Analysis
- Feature engineering
- Train/test split
- Preprocessing pipeline
- Baseline and tuned ML models
- Model evaluation
- Model serialization
- Streamlit prediction app
- GitHub-ready project structure

---

## Dataset

Dataset link:

```text
https://www.kaggle.com/datasets/wordsforthewise/lending-club
```

The Kaggle dataset contains Lending Club accepted and rejected loan data. For this project, we use the accepted loan file:

```text
accepted_2007_to_2018Q4.csv
```

Because the dataset is very large, the raw CSV is **not included** in this repository.

---

## Problem Statement

Lenders need to identify risky borrowers before approving or pricing loans. This project builds a classification model that predicts whether a loan is likely to be risky.

### Target Definition

The original `loan_status` column is converted into a binary target:

| Original Status | Target |
|---|---|
| Fully Paid | 0 = Good Loan |
| Charged Off | 1 = Bad Loan |

Rows with other loan statuses are removed to avoid ambiguous labels.

---

## Machine Learning Task

```text
Binary Classification
```

Target variable:

```text
loan_default
```

Positive class:

```text
1 = Charged Off / Bad Loan
```

---

## Project Structure

```text
lending_club_default_risk_project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── accepted_2007_to_2018Q4.csv      # Add manually
│   └── processed/
│
├── models/
│
├── notebooks/
│   └── 01_lending_club_project_walkthrough.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── config.py
│   ├── data_preparation.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run This Project

### 1. Clone the repository

```bash
git clone <your-github-repo-link>
cd lending_club_default_risk_project
```

### 2. Create virtual environment

For Windows CMD:

```bash
python -m venv venv
venv\Scripts\activate
```

For PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

For Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset from Kaggle

Download the dataset from:

```text
https://www.kaggle.com/datasets/wordsforthewise/lending-club
```

Place this file inside `data/raw/`:

```text
data/raw/accepted_2007_to_2018Q4.csv
```

### 5. Run data preparation

```bash
python src/data_preparation.py
```

This creates:

```text
data/processed/lending_club_cleaned.csv
```

### 6. Run EDA

```bash
python src/eda.py
```

EDA charts will be saved in:

```text
reports/figures/
```

### 7. Train model

```bash
python src/train.py
```

This saves the trained model pipeline in:

```text
models/lending_club_model.joblib
```

### 8. Evaluate model

```bash
python src/evaluate.py
```

This saves model metrics in:

```text
reports/model_metrics.json
```

### 9. Run Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## Features Used

The model uses selected borrower and loan-related variables such as:

- loan amount
- interest rate
- installment
- grade
- sub grade
- employment length
- home ownership
- annual income
- verification status
- loan purpose
- debt-to-income ratio
- delinquencies
- earliest credit line age
- open accounts
- public records
- revolving balance
- revolving utilization
- total accounts
- application type
- term

The feature list is intentionally controlled to reduce leakage and make the project easier to explain in viva/interview.

---

## Models Used

This project trains and compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

The best model is selected using ROC-AUC score.

---

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

For lending problems, **recall and ROC-AUC** are important because missing risky borrowers can create financial losses.

---

## Business Interpretation

A higher default-risk probability means the borrower is more likely to become a charged-off loan.

Possible business actions:

- Reject very high-risk applications
- Increase interest rate for medium-risk applications
- Request additional documents
- Reduce approved loan amount
- Send applications for manual underwriting

---

## Streamlit App

The Streamlit app allows a user to enter applicant/loan details and returns:

- Predicted class
- Probability of default
- Risk bucket

Risk buckets:

| Probability | Risk Level |
|---|---|
| < 0.30 | Low Risk |
| 0.30 - 0.60 | Medium Risk |
| > 0.60 | High Risk |

---

## Important Notes

This project is for educational purposes only. In a real lending system, additional work is required:

- Fair lending compliance
- Bias and fairness testing
- Model monitoring
- Explainability
- Data drift detection
- Out-of-time validation
- Human review workflow

---

## Author

```text
Souradip Dey
```

