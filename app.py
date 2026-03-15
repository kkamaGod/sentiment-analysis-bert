"""FastAPI service for sentiment analysis inference."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(payload: PredictRequest):
    return predict(payload.text, model_dir="artifacts/model")
