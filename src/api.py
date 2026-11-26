
"""FastAPI service for keyword extraction and verified feedback."""

import logging
import os
import time
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pathlib import Path

from src.config import (
    INFERENCE_LOG,
    FEEDBACK_PATH,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "production_model"
MODEL_URI = "models/production_model"

from src.data import append_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("keyword-api")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = mlflow.pyfunc.load_model(str(MODEL_PATH))
        logger.info("Loaded model from: %s", MODEL_PATH)

    except Exception:
        logger.exception("Could not load model from: %s", MODEL_PATH)

    yield
    state.clear()


app = FastAPI(
    title="Keyword Extraction API",
    description=(
        "BERT-based keyword extraction with MLflow, "
        "verified feedback and continuous learning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://keyword-extractor-36r4.onrender.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    top_n: int = Field(default=5, ge=1, le=50)


class ExtractResponse(BaseModel):
    keywords: list[str]
    model_uri: str


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    keywords: list[str] = Field(..., min_length=1, max_length=30)


class FeedbackResponse(BaseModel):
    status: str
    message: str
    keywords_saved: int


@app.get("/")
def home():
    return {
        "message": "Keyword Extraction API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "model_uri": MODEL_URI,
    }


def run_extraction(req: ExtractRequest):
    model = state.get("model")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Production model is not loaded. Check MLflow.",
        )

    try:
        prediction = model.predict(
            pd.DataFrame([{
                "text": req.text,
                "top_n": req.top_n,
            }])
        )[0]

        keywords = list(prediction)

        # Prediction logs are for monitoring, not training labels.
        if os.getenv("LOG_REQUESTS", "1") == "1":
            append_jsonl(
                [{
                    "ts": time.time(),
                    "text": req.text,
                    "predicted_keywords": keywords,
                }],
                INFERENCE_LOG,
            )

        return ExtractResponse(
            keywords=keywords,
            model_uri=MODEL_URI,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Keyword extraction failed")
        raise HTTPException(
            status_code=500,
            detail="Keyword extraction failed",
        ) from exc


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    return run_extraction(req)


@app.post("/predict", response_model=ExtractResponse)
def predict(req: ExtractRequest):
    return run_extraction(req)


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    # Clean and validate the manually supplied labels.
    verified_keywords = list(dict.fromkeys(
        keyword.strip()
        for keyword in req.keywords
        if keyword and keyword.strip()
    ))

    if not verified_keywords:
        raise HTTPException(
            status_code=400,
            detail="At least one valid keyword is required.",
        )

    record = {
        "ts": time.time(),
        "text": req.text.strip(),
        "keywords": verified_keywords,
        "verified": True,
    }

    try:
        append_jsonl([record], FEEDBACK_PATH)

    except Exception as exc:
        logger.exception("Could not save feedback")
        raise HTTPException(
            status_code=500,
            detail="Could not save feedback.",
        ) from exc

    return FeedbackResponse(
        status="success",
        message="Verified feedback saved for monitoring and future retraining.",
        keywords_saved=len(verified_keywords),
    )
