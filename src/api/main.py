from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerData, PredictionResponse
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Credit Risk API")

# ------------------------
# 1️⃣ Load the Best Model from MLflow
# ------------------------
MODEL_NAME = "CreditRiskModel"

try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    print(f"Model {MODEL_NAME} loaded successfully!")
except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    model = None

# ------------------------
# 2️⃣ Define /predict Endpoint
# ------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(customer_data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_df = pd.DataFrame([customer_data.dict()])

    try:
        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = int(model.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionResponse(probability=probability, prediction=prediction)

