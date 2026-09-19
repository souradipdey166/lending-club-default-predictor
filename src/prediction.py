import joblib
import pandas as pd

from config import SELECTED_MODEL_NAME, SELECTED_MODEL_PATH


def build_engineered_features(input_data: dict) -> dict:
    """
    Takes raw applicant fields and derives the same engineered features
    data_preparation.py builds during training:
      - fico_score: mean of fico_range_low / fico_range_high
      - loan_to_income: loan_amnt / annual_inc
      - installment_to_income: installment / annual_inc

    This keeps training and inference in sync - if data_preparation.py's
    feature engineering ever changes, this needs to change with it.
    """
    data = dict(input_data)

    fico_low = data.pop("fico_range_low", None)
    fico_high = data.pop("fico_range_high", None)
    if fico_low is not None and fico_high is not None:
        data["fico_score"] = (fico_low + fico_high) / 2

    annual_inc = data.get("annual_inc")
    safe_income = annual_inc if annual_inc not in (None, 0) else None

    loan_amnt = data.get("loan_amnt")
    installment = data.get("installment")

    data["loan_to_income"] = (
        loan_amnt / safe_income if loan_amnt is not None and safe_income else None
    )
    data["installment_to_income"] = (
        installment / safe_income if installment is not None and safe_income else None
    )

    return data


def predict_default(input_data: dict):
    """
    input_data: raw applicant fields (see sample below for the full set),
    including fico_range_low / fico_range_high (NOT last_fico_range_* -
    those are post-outcome and must never be used).
    """
    if not SELECTED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {SELECTED_MODEL_PATH}.\n"
            f"Run 'python src/train.py' first, then review 'python src/evaluate.py' "
            f"output and confirm config.SELECTED_MODEL_NAME (currently "
            f"'{SELECTED_MODEL_NAME}') points at a model you've actually trained."
        )

    model = joblib.load(SELECTED_MODEL_PATH)

    engineered = build_engineered_features(input_data)
    df = pd.DataFrame([engineered])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[:, 1][0])

    return {
        "model_used": SELECTED_MODEL_NAME,
        "prediction": prediction,
        "prediction_label": "Bad Loan (Likely Default)" if prediction == 1 else "Good Loan (Likely Repaid)",
        "default_probability": probability,
    }


if __name__ == "__main__":
    sample = {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": 13.5,
        "installment": 339.31,
        "grade": "B",
        "sub_grade": "B2",
        "emp_length": 3,
        "home_ownership": "RENT",
        "annual_inc": 60000,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 18.5,
        "delinq_2yrs": 0,
        "open_acc": 10,
        "pub_rec": 0,
        "revol_bal": 8000,
        "revol_util": 45.2,
        "total_acc": 25,
        "application_type": "Individual",
        "credit_history_years": 8,
        "fico_range_low": 690,
        "fico_range_high": 694,
    }

    print(predict_default(sample))
    