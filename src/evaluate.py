
"""DVC Stage 3: evaluate the selected keyword extractor on the
untouched KP20k test split.
"""

import json

from src.config import ROOT, TEST_PATH
from src.data import load_jsonl
from src.extractor import extract_keywords, get_encoder


CONFIG_PATH = ROOT / "models" / "config.json"
METRICS_PATH = ROOT / "results" / "test_metrics.json"


def normalize(keywords):
    return {
        str(keyword).strip().casefold()
        for keyword in keywords
        if str(keyword).strip()
    }


def main():
    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    # Load the selected configuration.
    if not CONFIG_PATH.exists():
        raise SystemExit(
            "Model configuration not found. Run training first."
        )

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    # Load the untouched test split.
    test_docs = load_jsonl(TEST_PATH)

    if not test_docs:
        raise SystemExit(
            "Test dataset is empty. Run `dvc repro prepare` first."
        )

    print(f"Test records: {len(test_docs)}")
    print(f"Model: {cfg['embedding_model']}")
    print(f"Top N: {cfg['top_n']}")
    print(f"N-gram range: {cfg['ngram_range']}")
    print(f"Diversity: {cfg['diversity']}")

    encoder = get_encoder(cfg["embedding_model"])

    total_tp = 0
    total_fp = 0
    total_fn = 0
    document_f1_scores = []

    for index, document in enumerate(test_docs, start=1):

        predictions = extract_keywords(
            document["text"],
            encoder,
            top_n=cfg["top_n"],
            ngram_range=tuple(cfg["ngram_range"]),
            diversity=cfg["diversity"],
        )

        predicted = normalize(predictions)
        actual = normalize(document["keywords"])

        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        denominator = precision + recall
        doc_f1 = (
            2 * precision * recall / denominator
            if denominator else 0.0
        )

        document_f1_scores.append(doc_f1)

        if index % 100 == 0:
            print(f"Evaluated {index}/{len(test_docs)} documents")

    # Calculate micro-averaged metrics over all keyword predictions.
    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0.0
    )

    denominator = precision + recall

    micro_f1 = (
        2 * precision * recall / denominator
        if denominator else 0.0
    )

    macro_f1 = (
        sum(document_f1_scores) / len(document_f1_scores)
        if document_f1_scores else 0.0
    )

    metrics = {
        "test_precision_micro": precision,
        "test_recall_micro": recall,
        "test_f1_micro": micro_f1,
        "test_f1_macro": macro_f1,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "n_test": len(test_docs),
        "config": cfg,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Micro Precision: {precision:.4f}")
    print(f"Micro Recall:    {recall:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")

    print("\nSaved results:", METRICS_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
