# Keyword Extraction Pipeline (MLOps)

An NLP system that extracts key phrases from documents using BERT-style sentence embeddings (KeyBERT approach with MMR diversity). It includes dataset versioning, experiment tracking, a model registry, a FastAPI service, Docker, CI/CD, drift monitoring, and automatic retraining.

## Stack

| Concern | Tool | Where |
|---|---|---|
| Keyword extraction | Sentence-BERT embeddings + MMR | `src/extractor.py` |
| Dataset versioning | DVC | `dvc.yaml`, `data/raw/*.dvc` |
| Experiment tracking & model registry | MLflow | `src/train.py`, `src/keyword_model.py` |
| Serving | FastAPI | `src/api.py` |
| Containerization | Docker / docker-compose | `Dockerfile`, `docker-compose.yml` |
| CI/CD | GitHub Actions | `.github/workflows/` |
| Drift monitoring | Evidently AI | `src/monitor.py`, `monitoring/drift_report.html` |
| Automatic updates | `src/retrain.py` + scheduled workflow | `.github/workflows/retrain.yml` |

## How it works

1. **Data**: `data/raw/documents.jsonl` holds labelled docs (`{"text": ..., "keywords": [...]}`). DVC versions it.
2. **Prepare** (`dvc repro prepare`): splits the data into train and validation sets.
3. **Train** (`dvc repro train`): grid-searches n-gram range and MMR diversity against validation F1. Every run is logged to MLflow. The best model is registered as `keyword-extractor` and gets the `production` alias only if it beats the current production model by `min_improvement`.
4. **Serve**: the API loads `models:/keyword-extractor@production` at startup and logs incoming requests to `data/inference/`.
5. **Monitor and update**: `src/retrain.py` runs Evidently drift checks between training data and incoming documents. If drift is detected and the incoming docs are labelled, it merges them into the dataset, re-runs the DVC pipeline, and the promotion gate decides whether the new model goes live. A GitHub Action runs this weekly.

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

git init
dvc init
dvc add data/raw/documents.jsonl
# Configure a remote for data storage, for example:
# dvc remote add -d storage /path/to/dvc-storage   (or s3://bucket/path)

# Terminal 1: tracking server
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

# Terminal 2: run the pipeline
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
dvc repro

# Serve the production model
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uvicorn src.api:app --port 8000
```

Test it:

```bash
curl -X POST localhost:8000/extract -H "Content-Type: application/json" \
  -d '{"text": "Solar panels convert sunlight into electricity. Batteries store solar power.", "top_n": 4}'
```

Run the tests (they use a fake encoder, so no model download is needed):

```bash
pytest -q
```

## Docker

```bash
docker compose up --build
```

This starts MLflow on port 5000 and the API on port 8000. Run the training pipeline against the MLflow container first (`MLFLOW_TRACKING_URI=http://localhost:5000 dvc repro`), so a production model exists for the API to load.

## Automatic updates

- Put newly labelled documents in `data/incoming/documents.jsonl` (each line needs a `keywords` list for retraining).
- `python -m src.retrain` checks drift and retrains if needed. Use `--force` to retrain regardless.
- The `retrain.yml` workflow does this every Monday. It needs these GitHub secrets: `MLFLOW_TRACKING_URI` (a reachable, hosted MLflow server), plus credentials for your DVC remote (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` if you use S3).

## Notes and limits

- The "training" step tunes extraction settings on labelled data. It does not fine-tune the BERT encoder. Fine-tuning is a natural next step: swap the training stage for a sentence-transformers fine-tuning loop and log it to the same registry.
- The sample dataset has 12 documents, so the metrics are only a smoke test. Use a few hundred labelled documents for meaningful results.
- Reaching a shared MLflow server and DVC remote from GitHub Actions requires hosting them somewhere the runner can reach. A laptop-only MLflow server won't work for the scheduled workflow.
- Request logging stores input text. Set `LOG_REQUESTS=0` if the documents are sensitive.
