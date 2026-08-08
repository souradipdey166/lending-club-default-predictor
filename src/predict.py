import joblib
import pandas as pd

from config import MODEL_PATH


def predict_default(input_data: dict):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train the model first using: python src/train.py")

    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([input_data])

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[:, 1][0])

    return {
        "prediction": prediction,
        "default_probability": probability
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
        "credit_history_years": 8
    }

    print(predict_default(sample))
