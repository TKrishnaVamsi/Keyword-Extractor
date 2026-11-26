
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Original KP20k dataset
DATASET_DIR = ROOT / "data" / "processed"
SOURCE_TRAIN_PATH = DATASET_DIR / "train.jsonl"
SOURCE_VAL_PATH = DATASET_DIR / "validation.jsonl"
SOURCE_TEST_PATH = DATASET_DIR / "test.jsonl"

# DVC-prepared datasets
PREPARED_DIR = ROOT / "data" / "prepared"
TRAIN_PATH = PREPARED_DIR / "train.jsonl"
VAL_PATH = PREPARED_DIR / "validation.jsonl"
TEST_PATH = PREPARED_DIR / "test.jsonl"

# Incoming documents and verified feedback
INCOMING_PATH = ROOT / "data" / "incoming" / "documents.jsonl"
FEEDBACK_PATH = ROOT / "data" / "incoming" / "feedback.jsonl"
INFERENCE_LOG = ROOT / "data" / "inference" / "requests.jsonl"

# Model and reports
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "monitoring"
METRICS_PATH = ROOT / "results" / "metrics.json"

# MLflow
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"file:{ROOT / 'mlruns'}"
)
MODEL_NAME = os.getenv("MODEL_NAME", "keyword-extractor")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@production")
EXPERIMENT_NAME = "keyword-extraction"
