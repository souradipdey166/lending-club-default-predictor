"""
api.py

FastAPI service exposing the trained model as a prediction endpoint.
Reuses prediction.py's predict_default() - same engineered-feature
logic, same model, no duplicated code.

Run locally (from project root):
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Interactive docs once running:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction import predict_default

app = FastAPI(
    title="Lending Club Default Risk API",
    description="Predicts the probability that a loan will default, based on applicant and loan details.",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., example=10000)
    term: str = Field(..., example="36 months")
    int_rate: float = Field(..., example=13.5)
    installment: float = Field(..., example=339.31)
    grade: str = Field(..., example="B")
    sub_grade: str = Field(..., example="B2")
    emp_length: float = Field(..., example=3)
    home_ownership: str = Field(..., example="RENT")
    annual_inc: float = Field(..., example=60000)
    verification_status: str = Field(..., example="Verified")
    purpose: str = Field(..., example="debt_consolidation")
    dti: float = Field(..., example=18.5)
    delinq_2yrs: int = Field(..., example=0)
    open_acc: int = Field(..., example=10)
    pub_rec: int = Field(..., example=0)
    revol_bal: float = Field(..., example=8000)
    revol_util: float = Field(..., example=45.2)
    total_acc: int = Field(..., example=25)
    application_type: str = Field(..., example="Individual")
    credit_history_years: float = Field(..., example=8.0)
    fico_range_low: int = Field(..., example=690)
    fico_range_high: int = Field(..., example=694)


class PredictionResponse(BaseModel):
    model_used: str
    prediction: int
    prediction_label: str
    default_probability: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Lending Club Default Risk API. POST to /predict."}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        result = predict_default(application.dict())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    return result