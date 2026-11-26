
"""DVC Stage 2: tune keyword extraction, evaluate the winner,
register it in MLflow, and promote it when it improves production."""

import itertools
import json
import random

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import (
    ROOT, TRAIN_PATH, VAL_PATH, MODEL_DIR, METRICS_PATH,
    MLFLOW_TRACKING_URI, MODEL_NAME, EXPERIMENT_NAME,
)
from src.data import load_jsonl
from src.extractor import extract_keywords, get_encoder
from src.keyword_model import KeywordExtractorModel, save_config


def f1(pred, gold):
    p = {k.strip().casefold() for k in pred if k.strip()}
    g = {k.strip().casefold() for k in gold if k.strip()}

    if not p or not g:
        return 0.0

    tp = len(p & g)
    precision = tp / len(p)
    recall = tp / len(g)

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return 2 * precision * recall / denominator

def evaluate(docs, encoder, cfg):
    scores = []

    for index, document in enumerate(docs, start=1):
        keywords = extract_keywords(
            document["text"],
            encoder,
            top_n=cfg["top_n"],
            ngram_range=tuple(cfg["ngram_range"]),
            diversity=cfg["diversity"],
        )
        scores.append(f1(keywords, document["keywords"]))

        if index % 100 == 0:
            print(f"  Evaluated {index}/{len(docs)} documents")

    return sum(scores) / max(len(scores), 1)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = yaml.safe_load(
        (ROOT / "params.yaml").read_text(encoding="utf-8")
    )["train"]

    train_docs = load_jsonl(TRAIN_PATH)
    val_docs = load_jsonl(VAL_PATH)

    if not train_docs or not val_docs:
        raise SystemExit("Missing prepared data. Run `dvc repro prepare` first.")

    # Use a reproducible subset for configuration selection.
    sample_size = min(params.get("tuning_samples", 500), len(val_docs))
    tuning_docs = random.Random(42).sample(val_docs, sample_size)

    print("=" * 60)
    print("KEYWORD EXTRACTION MODEL TRAINING")
    print("=" * 60)
    print(f"Training records available: {len(train_docs)}")
    print(f"Tuning sample: {len(tuning_docs)}")
    print(f"Full validation set: {len(val_docs)}")

    encoder = get_encoder(params["embedding_model"])

    grid = [
        {
            "embedding_model": params["embedding_model"],
            "top_n": params["top_n"],
            "ngram_range": list(ngram),
            "diversity": diversity,
        }
        for ngram, diversity in itertools.product(
            params["ngram_ranges"],
            params["diversities"],
        )
    ]

    print(f"Configurations to evaluate: {len(grid)}")

    best_cfg = None
    best_tuning_f1 = -1.0

    # Grid search: compare all configurations on the same sample.
    with mlflow.start_run(run_name="grid-search"):
        mlflow.log_params({
            "n_train_available": len(train_docs),
            "n_validation": len(val_docs),
            "n_tuning_samples": len(tuning_docs),
            "grid_size": len(grid),
        })

        for index, cfg in enumerate(grid, start=1):
            print(f"\nConfiguration {index}/{len(grid)}: {cfg}")

            with mlflow.start_run(
                run_name=f"config-{index}",
                nested=True,
            ):
                tuning_f1 = evaluate(tuning_docs, encoder, cfg)

                mlflow.log_params({
                    key: str(value) for key, value in cfg.items()
                })
                mlflow.log_metric("tuning_f1", tuning_f1)

                print(f"Tuning F1: {tuning_f1:.4f}")

            if tuning_f1 > best_tuning_f1:
                best_tuning_f1 = tuning_f1
                best_cfg = cfg.copy()

    if best_cfg is None:
        raise RuntimeError("Grid search did not produce a configuration.")

    # Evaluate only the winning configuration on the complete validation set.
    print("\n" + "=" * 60)
    print("FULL VALIDATION OF BEST CONFIGURATION")
    print("=" * 60)
    print(f"Selected configuration: {best_cfg}")

    best_val_f1 = evaluate(val_docs, encoder, best_cfg)
    print(f"Full validation F1: {best_val_f1:.4f}")

    # Register the selected model.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="best-model") as run:
        mlflow.log_params({
            f"best_{key}": str(value)
            for key, value in best_cfg.items()
        })
        mlflow.log_metric("tuning_f1", best_tuning_f1)
        mlflow.log_metric("val_f1", best_val_f1)

        cfg_path = save_config(best_cfg, MODEL_DIR / "config.json")

        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=KeywordExtractorModel(),
            artifacts={"config": str(cfg_path)},
        )

        registered_model = mlflow.register_model(
            info.model_uri,
            MODEL_NAME,
        )
        version = registered_model.version

        print(f"\nRegistered {MODEL_NAME} version {version}")
        print(f"MLflow run ID: {run.info.run_id}")

    # Compare with the current production model.
    client = MlflowClient()
    prod_f1 = None

    try:
        production_model = client.get_model_version_by_alias(
            MODEL_NAME, "production"
        )
        prod_f1 = client.get_run(
            production_model.run_id
        ).data.metrics.get("val_f1")
    except MlflowException:
        print("\nNo existing production model found.")

    min_improvement = params["min_improvement"]
    promote = (
        prod_f1 is None
        or best_val_f1 >= prod_f1 + min_improvement
    )

    if promote:
        client.set_registered_model_alias(
            MODEL_NAME, "production", version
        )
        print(f"\nPromoted version {version} to production.")
    else:
        print("\nExisting production model retained.")
        print(f"Production F1: {prod_f1:.4f}")
        print(f"New model F1: {best_val_f1:.4f}")

    # Save DVC metrics.
    metrics = {
        "val_f1": best_val_f1,
        "tuning_f1": best_tuning_f1,
        "model_version": version,
        "promoted": promote,
        "production_f1_before": prod_f1,
        "config": best_cfg,
        "n_train": len(train_docs),
        "n_tuning": len(tuning_docs),
        "n_val": len(val_docs),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
