import sys
from pathlib import Path
import joblib
import streamlit as st

st.set_page_config(
    page_title="Lending Club Default Risk",
    page_icon="\U0001F3E6",
    layout="wide",
)

# Allow importing from src/ regardless of where Streamlit is launched from.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

# Temporary diagnostic
try:
    import joblib
    st.success("JOBLIB IMPORT: SUCCESS")
except Exception as e:
    st.error(f"JOBLIB IMPORT FAILED: {repr(e)}")
    raise

from config import SELECTED_MODEL_NAME, SELECTED_MODEL_PATH  # noqa: E402
from prediction import predict_default  # noqa: E402


# ---------------------------------------------------------------------
# Demo examples
# ---------------------------------------------------------------------
# Illustrative applicant profiles reflecting the kinds of patterns
# typically associated with each outcome in the Lending Club dataset
# (e.g. grade, interest rate, FICO band, DTI, utilization) - not
# literal historical records, but representative of each class.
DEMO_EXAMPLES = {
    "-- Custom (fill in yourself) --": None,
    "Example: Low risk (No Default)": {
        "loan_amnt": 8000, "term": "36 months", "int_rate": 7.2, "installment": 248.11,
        "grade": "A", "sub_grade": "A2", "emp_length": 9, "home_ownership": "MORTGAGE",
        "annual_inc": 95000, "verification_status": "Verified", "purpose": "debt_consolidation",
        "dti": 11.4, "delinq_2yrs": 0, "open_acc": 11, "pub_rec": 0, "revol_bal": 6200,
        "revol_util": 22.5, "total_acc": 28, "application_type": "Individual",
        "credit_history_years": 16.0, "fico_range_low": 780, "fico_range_high": 784,
    },
    "Example: High risk (Default)": {
        "loan_amnt": 22000, "term": "60 months", "int_rate": 27.8, "installment": 683.45,
        "grade": "G", "sub_grade": "G3", "emp_length": 1, "home_ownership": "RENT",
        "annual_inc": 32000, "verification_status": "Not Verified", "purpose": "small_business",
        "dti": 34.9, "delinq_2yrs": 2, "open_acc": 14, "pub_rec": 1, "revol_bal": 18500,
        "revol_util": 88.3, "total_acc": 19, "application_type": "Individual",
        "credit_history_years": 4.0, "fico_range_low": 660, "fico_range_high": 664,
    },
    "Example: Borderline (Moderate risk)": {
        "loan_amnt": 15000, "term": "36 months", "int_rate": 16.9, "installment": 533.72,
        "grade": "C", "sub_grade": "C4", "emp_length": 4, "home_ownership": "RENT",
        "annual_inc": 52000, "verification_status": "Source Verified", "purpose": "credit_card",
        "dti": 22.7, "delinq_2yrs": 0, "open_acc": 9, "pub_rec": 0, "revol_bal": 11000,
        "revol_util": 58.0, "total_acc": 22, "application_type": "Individual",
        "credit_history_years": 8.0, "fico_range_low": 695, "fico_range_high": 699,
    },
}

FIELD_KEYS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "dti", "delinq_2yrs",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "application_type",
    "credit_history_years", "fico_range_low", "fico_range_high",
]

DEFAULTS = {
    "loan_amnt": 10000, "term": "36 months", "int_rate": 13.5, "installment": 339.31,
    "grade": "B", "sub_grade": "B2", "emp_length": 3, "home_ownership": "RENT",
    "annual_inc": 60000, "verification_status": "Verified", "purpose": "debt_consolidation",
    "dti": 18.5, "delinq_2yrs": 0, "open_acc": 10, "pub_rec": 0, "revol_bal": 8000,
    "revol_util": 45.2, "total_acc": 25, "application_type": "Individual",
    "credit_history_years": 8.0, "fico_range_low": 690, "fico_range_high": 694,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_demo(name: str):
    example = DEMO_EXAMPLES.get(name)
    if example:
        for k, v in example.items():
            st.session_state[k] = v


# ---------------------------------------------------------------------
# Sidebar - inputs
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Applicant & Loan Details")

    demo_choice = st.selectbox(
        "Load a demo example",
        list(DEMO_EXAMPLES.keys()),
        key="demo_choice",
        on_change=lambda: load_demo(st.session_state["demo_choice"]),
    )
    st.caption("Loads representative field values - edit anything below afterward.")
    st.divider()

    st.number_input("Loan Amount", min_value=500, max_value=100000, step=500, key="loan_amnt")
    st.selectbox("Term", ["36 months", "60 months"], key="term")
    st.number_input("Interest Rate (%)", min_value=3.0, max_value=35.0, step=0.1, key="int_rate")
    st.number_input("Installment", min_value=10.0, step=1.0, key="installment")
    st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"], key="grade")
    st.selectbox("Sub Grade", [f"{st.session_state['grade']}{i}" for i in range(1, 6)], key="sub_grade")

    st.divider()
    st.number_input("Annual Income", min_value=0, step=1000, key="annual_inc")
    st.number_input("Employment Length (years)", min_value=0, max_value=10, key="emp_length")
    st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], key="home_ownership")
    st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"], key="verification_status")
    st.selectbox(
        "Purpose",
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
         "small_business", "car", "medical", "moving", "vacation", "house",
         "wedding", "renewable_energy", "educational", "other"],
        key="purpose",
    )
    st.selectbox("Application Type", ["Individual", "Joint App"], key="application_type")

    st.divider()
    st.number_input("Debt-to-Income (DTI)", min_value=0.0, max_value=60.0, step=0.5, key="dti")
    st.number_input("Delinquencies (2 yrs)", min_value=0, key="delinq_2yrs")
    st.number_input("Open Credit Lines", min_value=0, key="open_acc")
    st.number_input("Public Records", min_value=0, key="pub_rec")
    st.number_input("Total Credit Lines", min_value=0, key="total_acc")
    st.number_input("Revolving Balance", min_value=0, step=100, key="revol_bal")
    st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, step=0.5, key="revol_util")
    st.number_input("Credit History Length (years)", min_value=0.0, step=0.5, key="credit_history_years")

    st.divider()
    st.caption("Origination-time FICO band")
    st.number_input("FICO Range Low", min_value=300, max_value=850, key="fico_range_low")
    st.number_input("FICO Range High", min_value=300, max_value=850, key="fico_range_high")

# ---------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------
st.title("\U0001F3E6 Lending Club Loan Default Risk Predictor")
st.write("This app predicts the probability that a loan may become a bad loan / charged-off loan.")

predict_clicked = st.button("Predict Default Risk", type="primary")

if not SELECTED_MODEL_PATH.exists():
    st.error(
        f"No trained model found at `{SELECTED_MODEL_PATH}`.\n\n"
        f"Run `python src/train.py` first to train and save the "
        f"'{SELECTED_MODEL_NAME}' model, then reload this app."
    )
    st.stop()

if predict_clicked:
    input_data = {k: st.session_state[k] for k in FIELD_KEYS}

    try:
        result = predict_default(input_data)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    prob = result["default_probability"]
    is_bad = result["prediction"] == 1

    if prob < 0.30:
        risk_level, risk_color = "Low Risk", "green"
    elif prob < 0.60:
        risk_level, risk_color = "Medium Risk", "orange"
    else:
        risk_level, risk_color = "High Risk", "red"

    st.header("Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Default Probability", f"{prob:.2%}")
    with col2:
        st.markdown(f"**Risk Level**")
        st.markdown(f":{risk_color}[## {risk_level}]")

    st.progress(min(max(prob, 0.0), 1.0))

    if is_bad:
        st.error(f"**Model Prediction: Bad Loan / Higher Default Risk**")
    else:
        st.success(f"**Model Prediction: Good Loan / Lower Default Risk**")

    st.caption(
        f"Model used: {result['model_used']}. This is a statistical estimate based on "
        f"historical Lending Club data, not a guarantee or credit decision."
    )