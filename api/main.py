"""TelcoCare AI REST API.

Run:
    uvicorn api.main:app --reload
Docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.agent.llm import LLMError, OllamaChat
from src.serving.predictor import get_predictor, get_store

log = logging.getLogger("telcocare.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="TelcoCare AI",
    description="Churn prediction with explanations, plus a RAG-powered retention assistant.",
    version="0.1.0",
)


# --- Schemas -------------------------------------------------------------------
class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: Literal["Yes", "No"] = "No"
    Partner: Literal["Yes", "No"] = "No"
    Dependents: Literal["Yes", "No"] = "No"
    tenure: int = Field(3, ge=0, le=120)
    PhoneService: Literal["Yes", "No"] = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = Field(85.0, ge=0)
    TotalCharges: float = Field(255.0, ge=0)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatMessage] = []


# --- Lazy agent (heavy: loads embedding model + FAISS index) ------------------
@lru_cache(maxsize=1)
def get_agent():
    from src.agent.agent import build_default_agent
    return build_default_agent()


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.0f}"
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, ms)
    return response


# --- Endpoints -----------------------------------------------------------------
@app.get("/health")
def health():
    try:
        predictor = get_predictor()
        model = {"loaded": True, "type": type(predictor.clf).__name__, "threshold": predictor.threshold}
    except FileNotFoundError as e:
        model = {"loaded": False, "error": str(e)}
    return {"status": "ok", "model": model, "llm": OllamaChat().health()}


@app.get("/customers")
def list_customers(n: int = 20, seed: int = 0):
    return {"customer_ids": get_store().sample_ids(n=min(n, 200), seed=seed)}


@app.get("/customers/{customer_id}")
def customer_profile(customer_id: str):
    profile = get_store().profile(customer_id)
    if profile is None:
        raise HTTPException(404, f"Customer '{customer_id}' not found")
    return profile


@app.get("/predict/{customer_id}")
def predict_customer(customer_id: str, top_k: int = 5):
    row = get_store().get(customer_id)
    if row is None:
        raise HTTPException(404, f"Customer '{customer_id}' not found")
    return get_predictor().predict_frame(row, customer_id=customer_id, top_k=top_k).to_dict()


@app.post("/predict")
def predict_features(features: CustomerFeatures, top_k: int = 5):
    return get_predictor().predict_frame(pd.DataFrame([features.model_dump()]), top_k=top_k).to_dict()


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        agent = get_agent()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    try:
        result = agent.run(req.message, history=[m.model_dump() for m in req.history])
    except LLMError as e:
        raise HTTPException(503, str(e))
    return result.to_dict()
