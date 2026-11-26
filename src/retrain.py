
"""Controlled continuous-learning workflow."""

import argparse
import shutil
import subprocess
from datetime import datetime

from src.config import (
    ROOT,
    SOURCE_TRAIN_PATH,
    FEEDBACK_PATH,
)

from src.data import (
    append_jsonl,
    load_jsonl,
    save_jsonl,
)

from src.monitor import check_drift


def main(force=False, min_feedback=20):
    feedback = load_jsonl(FEEDBACK_PATH)

    # Only accept verified feedback with valid text and keyword labels.
    labeled = [
        row for row in feedback
        if row.get("verified") is True
        and isinstance(row.get("text"), str)
        and row["text"].strip()
        and isinstance(row.get("keywords"), list)
        and any(
            isinstance(keyword, str) and keyword.strip()
            for keyword in row["keywords"]
        )
    ]

    print(f"Verified feedback records: {len(labeled)}")

    if len(labeled) < min_feedback:
        print(
            f"At least {min_feedback} verified records are required. "
            "Collect more feedback before retraining."
        )
        return 0

    reference = load_jsonl(SOURCE_TRAIN_PATH)

    if not reference:
        print("Training reference dataset is empty.")
        return 1

    try:
        drifted, _ = check_drift(
            reference,
            labeled,
            report_name="feedback_drift_report.html",
        )
    except Exception as exc:
        print(f"Drift detection failed: {exc}")
        return 1

    print(f"Dataset drift detected: {drifted}")

    if not drifted and not force:
        print("No drift detected. Skipping retraining.")
        print("Use --force to retrain anyway.")
        return 0

    print("Preparing verified feedback for retraining...")

    # Save original bytes so we can roll back if DVC fails.
    original_data = SOURCE_TRAIN_PATH.read_bytes()

    try:
        append_jsonl(labeled, SOURCE_TRAIN_PATH)

        print("Verified feedback added to training data.")
        print("Running the complete DVC pipeline...")

        subprocess.run(
            ["dvc", "repro"],
            check=True,
            cwd=ROOT,
        )

    except Exception as exc:
        SOURCE_TRAIN_PATH.write_bytes(original_data)

        print("Retraining failed.")
        print("Original training data has been restored.")
        print(f"Reason: {exc}")

        return 1

    # Archive feedback only after successful DVC execution.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = ROOT / "data" / "incoming" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_path = archive_dir / f"feedback-{stamp}.jsonl"

    shutil.copy2(FEEDBACK_PATH, archive_path)
    save_jsonl([], FEEDBACK_PATH)

    print("Retraining pipeline completed successfully.")
    print(f"Feedback archived at: {archive_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when no data drift is detected.",
    )

    parser.add_argument(
        "--min-feedback",
        type=int,
        default=20,
        help="Minimum verified feedback records required.",
    )

    args = parser.parse_args()

    raise SystemExit(
        main(
            force=args.force,
            min_feedback=args.min_feedback,
        )
    )
