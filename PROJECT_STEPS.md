# Project Execution Steps

Use this file as your submission explanation or viva preparation.

## Step 1: Business Understanding

The lending industry faces credit risk when borrowers fail to repay loans. The goal is to build a machine learning model that predicts whether a loan will become risky.

## Step 2: Dataset Understanding

Dataset: Lending Club accepted loan data.

Important columns:

- loan amount
- term
- interest rate
- installment
- grade and sub-grade
- employment length
- annual income
- home ownership
- debt-to-income ratio
- revolving utilization
- loan status

## Step 3: Target Variable Creation

Original column:

```text
loan_status
```

Selected statuses:

```text
Fully Paid -> Good Loan -> 0
Charged Off -> Bad Loan -> 1
```

Other statuses are removed because they are not final outcomes.

## Step 4: Data Cleaning

Cleaning performed:

- Removed ambiguous loan statuses
- Converted interest rate from percentage string to numeric
- Converted revolving utilization from percentage string to numeric
- Converted employment length into numeric years
- Created credit history length from earliest credit line
- Handled missing values inside ML pipeline

## Step 5: Exploratory Data Analysis

EDA includes:

- target distribution
- loan amount distribution
- interest rate vs default
- grade vs default
- correlation heatmap

## Step 6: Feature Engineering

Main engineered feature:

```text
credit_history_years
```

This represents how long the borrower has had credit history.

## Step 7: Model Building

Models trained:

- Logistic Regression
- Random Forest
- Gradient Boosting

Preprocessing:

- median imputation for numeric features
- most frequent imputation for categorical features
- one-hot encoding for categorical features
- standard scaling for numeric features

## Step 8: Model Selection

The best model is selected using ROC-AUC.

ROC-AUC is suitable because the target may be imbalanced and ranking risky borrowers is important.

## Step 9: Model Evaluation

Metrics used:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix

## Step 10: Deployment

A Streamlit web app is provided.

Run:

```bash
streamlit run app/streamlit_app.py
```

The app takes borrower and loan details as input and gives default probability.

## Step 11: GitHub Submission

Upload everything except:

- raw dataset CSV
- processed CSV
- trained model if too large
- virtual environment folder

These are ignored by `.gitignore`.

