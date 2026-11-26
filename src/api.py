
"""FastAPI service for keyword extraction and verified feedback."""

import logging
import os
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import INFERENCE_LOG, FEEDBACK_PATH
from src.data import append_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("keyword-api")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "production_model"
MODEL_URI = "models/production_model"

state = {
    "model": None,
    "loading": False,
    "error": None,
}


def load_model_background():
    """Load the model without blocking the web server startup."""
    state["loading"] = True
    state["error"] = None

    try:
        logger.info("Loading model from: %s", MODEL_PATH)
        model = mlflow.pyfunc.load_model(str(MODEL_PATH))
        state["model"] = model
        logger.info("Production model loaded successfully.")

    except Exception as exc:
        state["error"] = str(exc)
        logger.exception("Could not load production model.")

    finally:
        state["loading"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(
        target=load_model_background,
        daemon=True,
    )
    thread.start()

    yield

    state["model"] = None


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
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
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
        "model_loaded": state["model"] is not None,
        "model_loading": state["loading"],
        "model_error": state["error"],
        "model_uri": MODEL_URI,
    }


def run_extraction(req: ExtractRequest):
    model = state["model"]

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is still loading."
                if state["loading"]
                else "Model is unavailable. Check Render logs."
            ),
        )

    try:
        prediction = model.predict(
            pd.DataFrame([{
                "text": req.text,
                "top_n": req.top_n,
            }])
        )[0]

        keywords = list(prediction)

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

    except Exception:
        logger.exception("Could not save feedback")
        raise HTTPException(
            status_code=500,
            detail="Could not save feedback.",
        )

    return FeedbackResponse(
        status="success",
        message="Verified feedback saved for monitoring and future retraining.",
        keywords_saved=len(verified_keywords),
    )
