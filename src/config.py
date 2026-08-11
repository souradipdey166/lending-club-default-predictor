from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lending_club_cleaned.csv"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Use a sample by default because the Kaggle file is very large.
# Set SAMPLE_N = None if your machine has enough RAM.
SAMPLE_N = 120000

FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "fico_score",              # engineered: mean of fico_range_low/high
    "loan_to_income",          # engineered: loan_amnt / annual_inc
    "installment_to_income",   # engineered: installment / annual_inc
]

# Raw columns needed only to build engineered features above - not used
# directly as model inputs. fico_range_low/high are origination-time
# FICO band (safe to use). last_fico_range_low/high are POST-outcome
# and must never be used - they leak information from after the loan
# was issued.
FICO_SOURCE_COLUMNS = ["fico_range_low", "fico_range_high"]

TARGET = "loan_default"

GOOD_STATUS = "Fully Paid"
BAD_STATUS = "Charged Off"

# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------
# Every name here is trained + saved separately by train.py, and every
# saved model is evaluated separately by evaluate.py. Nothing is
# auto-picked as "best" - you review reports/model_metrics.json and
# the confusion matrix / ROC curve plots in reports/figures/, then set
# SELECTED_MODEL_NAME below by hand to whichever one you judge best.
MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}

MODEL_PATH_TEMPLATE = str(MODELS_DIR / "{model_name}.joblib")

# Class labels in [0, 1] order, used on confusion matrix axes etc.
CLASS_NAMES = ["Good Loan (0)", "Bad Loan (1)"]

CONFUSION_MATRIX_TEMPLATE = str(FIGURES_DIR / "confusion_matrix_{model_name}_{split}.png")
ROC_CURVE_TEMPLATE = str(FIGURES_DIR / "roc_curve_{model_name}_{split}.png")

# ---------------------------------------------------------------------
# Manually chosen model (set this yourself after reviewing evaluate.py
# output - reports/model_metrics.json and reports/figures/*.png).
# prediction.py loads whichever model is named here.
# ---------------------------------------------------------------------
SELECTED_MODEL_NAME = "xgboost"
SELECTED_MODEL_PATH = MODELS_DIR / f"{SELECTED_MODEL_NAME}.joblib"