
"""Data drift detection using Evidently AI."""

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import REPORT_DIR


def text_features(docs):
    rows = []

    for document in docs:
        text = str(document.get("text", "")).strip()
        words = text.split()

        if not text:
            continue

        rows.append({
            "n_words": len(words),
            "n_chars": len(text),
            "avg_word_len": (
                sum(len(word) for word in words) / max(len(words), 1)
            ),
            "n_sentences": (
                text.count(".") + text.count("!") + text.count("?")
            ),
        })

    return pd.DataFrame(
        rows,
        columns=["n_words", "n_chars", "avg_word_len", "n_sentences"],
    )


def check_drift(
    reference_docs,
    current_docs,
    report_name="drift_report.html",
):
    reference = text_features(reference_docs)
    current = text_features(current_docs)

    if len(reference) < 2:
        raise ValueError("At least two valid reference documents are required.")

    if len(current) < 2:
        raise ValueError("At least two valid current documents are required.")

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference,
        current_data=current,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / report_name
    report.save_html(str(report_path))

    result = report.as_dict()["metrics"][0]["result"]

    dataset_drift = bool(result["dataset_drift"])

    print(f"Drift report saved to: {report_path}")
    print(f"Dataset drift detected: {dataset_drift}")

    return dataset_drift, result
