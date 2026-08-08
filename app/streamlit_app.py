import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from config import MODEL_PATH


st.set_page_config(
    page_title="Lending Club Default Risk Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Lending Club Loan Default Risk Predictor")
st.write(
    "This app predicts the probability that a loan may become a bad loan / charged-off loan."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("Applicant & Loan Details")

loan_amnt = st.sidebar.number_input("Loan Amount", min_value=500, max_value=40000, value=10000, step=500)
term = st.sidebar.selectbox("Term", ["36 months", "60 months"])
int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=13.5, step=0.1)
installment = st.sidebar.number_input("Installment", min_value=10.0, max_value=2000.0, value=339.31, step=10.0)

grade = st.sidebar.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
sub_grade = st.sidebar.selectbox(
    "Sub Grade",
    [f"{g}{i}" for g in ["A", "B", "C", "D", "E", "F", "G"] for i in range(1, 6)]
)

emp_length = st.sidebar.slider("Employment Length (Years)", 0, 10, 3)
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
annual_inc = st.sidebar.number_input("Annual Income", min_value=0, max_value=1000000, value=60000, step=1000)
verification_status = st.sidebar.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])

purpose = st.sidebar.selectbox(
    "Purpose",
    [
        "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
        "small_business", "car", "medical", "moving", "vacation", "other"
    ]
)

dti = st.sidebar.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=18.5, step=0.5)
delinq_2yrs = st.sidebar.number_input("Delinquencies in Last 2 Years", min_value=0, max_value=20, value=0)
open_acc = st.sidebar.number_input("Open Accounts", min_value=0, max_value=100, value=10)
pub_rec = st.sidebar.number_input("Public Records", min_value=0, max_value=20, value=0)
revol_bal = st.sidebar.number_input("Revolving Balance", min_value=0, max_value=500000, value=8000, step=500)
revol_util = st.sidebar.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=45.2, step=0.5)
total_acc = st.sidebar.number_input("Total Accounts", min_value=0, max_value=200, value=25)
application_type = st.sidebar.selectbox("Application Type", ["Individual", "Joint App"])
credit_history_years = st.sidebar.number_input("Credit History Length (Years)", min_value=0.0, max_value=80.0, value=8.0, step=0.5)

input_data = {
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "installment": installment,
    "grade": grade,
    "sub_grade": sub_grade,
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc": annual_inc,
    "verification_status": verification_status,
    "purpose": purpose,
    "dti": dti,
    "delinq_2yrs": delinq_2yrs,
    "open_acc": open_acc,
    "pub_rec": pub_rec,
    "revol_bal": revol_bal,
    "revol_util": revol_util,
    "total_acc": total_acc,
    "application_type": application_type,
    "credit_history_years": credit_history_years
}

if st.button("Predict Default Risk"):
    input_df = pd.DataFrame([input_data])
    probability = float(model.predict_proba(input_df)[:, 1][0])
    prediction = int(model.predict(input_df)[0])

    st.subheader("Prediction Result")

    if probability < 0.30:
        risk_level = "Low Risk"
    elif probability < 0.60:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    st.metric("Default Probability", f"{probability:.2%}")
    st.metric("Risk Level", risk_level)

    if prediction == 1:
        st.error("Model Prediction: Bad Loan / Possible Default")
    else:
        st.success("Model Prediction: Good Loan / Lower Default Risk")
'''
st.info(
    "Educational project only. Real lending models require fairness testing, explainability, compliance checks, and out-of-time validation."
)
'''
