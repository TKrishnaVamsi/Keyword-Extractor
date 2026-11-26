
import json
import random
import time
from pathlib import Path

import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

# Correct Hugging Face repository root
BASE_URL = (
    "https://huggingface.co/datasets/"
    "midas/kp20k/resolve/main"
)

SPLITS = {
    "train": {
        "source": "train.jsonl",
        "output": "train.jsonl",
        "limit": 10000,
    },
    "validation": {
        "source": "valid.jsonl",
        "output": "validation.jsonl",
        "limit": 2000,
    },
    "test": {
        "source": "test.jsonl",
        "output": "test.jsonl",
        "limit": 2000,
    },
}

RANDOM_SEED = 42
REQUEST_TIMEOUT = (30, 180)


# --------------------------------------------------
# HTTP SESSION WITH RETRIES
# --------------------------------------------------

def create_session():

    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "KeywordExtractionMLOps/1.0"
    })

    return session


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def normalize_document(document):
    """Convert token lists or strings into plain text."""

    if isinstance(document, list):

        document = " ".join(
            str(token) for token in document
        )

    if not isinstance(document, str):
        return ""

    return " ".join(document.split())


def normalize_keywords(keywords):
    """Clean and deduplicate extractive keyphrases."""

    if not isinstance(keywords, list):
        return []

    cleaned = []
    seen = set()

    for keyword in keywords:

        if not isinstance(keyword, str):
            continue

        keyword = " ".join(
            keyword.split()
        ).strip()

        if not keyword:
            continue

        normalized = keyword.casefold()

        if normalized not in seen:

            cleaned.append(keyword)
            seen.add(normalized)

    return cleaned


# --------------------------------------------------
# DOWNLOAD AND PROCESS ONE SPLIT
# --------------------------------------------------

def download_split(split_name, config, session):

    url = f"{BASE_URL}/{config['source']}"

    output_path = OUTPUT_DIR / config["output"]
    temp_path = output_path.with_suffix(".tmp")

    print("\n" + "=" * 60)
    print(f"DOWNLOADING: {split_name.upper()}")
    print("=" * 60)

    print(f"Source: {url}")
    print(f"Target usable records: {config['limit']}")

    processed = 0
    skipped = 0
    total_lines = 0

    try:

        with session.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        ) as response:

            response.raise_for_status()

            with open(
                temp_path,
                "w",
                encoding="utf-8",
            ) as output:

                progress = tqdm(
                    desc=f"Processing {split_name}",
                    unit=" records",
                )

                try:

                    for line in response.iter_lines(
                        chunk_size=1024,
                        decode_unicode=True,
                    ):

                        if processed >= config["limit"]:
                            break

                        if not line:
                            continue

                        total_lines += 1

                        try:

                            record = json.loads(line)

                            document = normalize_document(
                                record.get("document")
                            )

                            keywords = normalize_keywords(
                                record.get(
                                    "extractive_keyphrases"
                                )
                            )

                            # Reject unusable examples.
                            if len(document.split()) < 10:
                                skipped += 1
                                continue

                            if not keywords:
                                skipped += 1
                                continue

                            cleaned_record = {
                                "text": document,
                                "keywords": keywords,
                            }

                            output.write(
                                json.dumps(
                                    cleaned_record,
                                    ensure_ascii=False,
                                ) + "\n"
                            )

                            processed += 1

                            progress.update(1)

                        except (
                            json.JSONDecodeError,
                            TypeError,
                            ValueError,
                        ):

                            skipped += 1

                finally:

                    progress.close()

        if processed == 0:

            raise RuntimeError(
                f"No usable records found for {split_name}."
            )

        # Replace destination only after successful processing.
        temp_path.replace(output_path)

        print(f"\nSaved: {output_path}")
        print(f"Usable records: {processed}")
        print(f"Skipped records: {skipped}")
        print(f"Total lines inspected: {total_lines}")

        return {
            "records": processed,
            "skipped": skipped,
            "lines_inspected": total_lines,
            "file": str(
                output_path.relative_to(PROJECT_ROOT)
            ),
        }

    except Exception:

        temp_path.unlink(missing_ok=True)

        raise


# --------------------------------------------------
# DATASET VALIDATION
# --------------------------------------------------

def inspect_dataset():

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    for split_name, config in SPLITS.items():

        path = OUTPUT_DIR / config["output"]

        print(f"\n{split_name.upper()}")

        if not path.exists():

            raise FileNotFoundError(
                f"Dataset split missing: {path}"
            )

        record_count = 0

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            first_record = None

            for line in file:

                if not line.strip():
                    continue

                record = json.loads(line)

                if not isinstance(record.get("text"), str):
                    raise ValueError(
                        f"Invalid text in {path.name}"
                    )

                if not isinstance(record.get("keywords"), list):
                    raise ValueError(
                        f"Invalid keywords in {path.name}"
                    )

                record_count += 1

                if first_record is None:
                    first_record = record

        if record_count == 0:

            raise ValueError(
                f"Empty dataset split: {path.name}"
            )

        print(f"Records: {record_count}")

        print("\nText preview:")

        print(
            first_record["text"][:350]
        )

        print("\nReference keywords:")

        print(
            first_record["keywords"]
        )

        print("\nFile:", path.name)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "dataset": "KP20k",
        "source": "midas/kp20k",
        "configuration": "raw",
        "sampling": "First N usable records from each official split",
        "random_seed": RANDOM_SEED,
        "splits": {},
    }

    session = create_session()

    try:

        for split_name, config in SPLITS.items():

            result = download_split(
                split_name,
                config,
                session,
            )

            summary["splits"][split_name] = result

            time.sleep(1)

    finally:

        session.close()

    # Save dataset summary.
    report_path = (
        REPORT_DIR / "dataset_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # Validate downloaded files.
    inspect_dataset()

    print("\n" + "=" * 60)
    print("PHASE 1 DATASET DOWNLOAD COMPLETED")
    print("=" * 60)

    print(
        json.dumps(
            summary,
            indent=4,
        )
    )

    print(f"\nSummary report: {report_path}")


if __name__ == "__main__":

    main()
