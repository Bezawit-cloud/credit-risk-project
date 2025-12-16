# src/api/pydantic_models.py
from pydantic import BaseModel

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: float
    std_amount: float

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
