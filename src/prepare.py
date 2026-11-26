"""Stage 1 (DVC): validate and prepare official KP20k splits."""

from src.config import (
    SOURCE_TRAIN_PATH,
    SOURCE_VAL_PATH,
    SOURCE_TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
)

from src.data import load_jsonl, save_jsonl


def validate_records(records, split_name):

    if not records:
        raise ValueError(
            f"{split_name} dataset is empty."
        )

    for index, record in enumerate(records):

        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid record {index} in {split_name}."
            )

        text = record.get("text")
        keywords = record.get("keywords")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Missing or invalid text at record {index} "
                f"in {split_name}."
            )

        if not isinstance(keywords, list) or not keywords:
            raise ValueError(
                f"Missing or invalid keywords at record {index} "
                f"in {split_name}."
            )

        if not all(
            isinstance(keyword, str) and keyword.strip()
            for keyword in keywords
        ):
            raise ValueError(
                f"Invalid keyword at record {index} "
                f"in {split_name}."
            )

    return len(records)


def prepare_split(source, destination, split_name):

    if not source.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source}"
        )

    records = load_jsonl(source)

    count = validate_records(records, split_name)

    save_jsonl(records, destination)

    print(
        f"{split_name}: {count} records prepared"
    )

    return count


def main():

    print("=" * 60)
    print("KP20K DATASET PREPARATION")
    print("=" * 60)

    train_count = prepare_split(
        SOURCE_TRAIN_PATH,
        TRAIN_PATH,
        "Training"
    )

    val_count = prepare_split(
        SOURCE_VAL_PATH,
        VAL_PATH,
        "Validation"
    )

    test_count = prepare_split(
        SOURCE_TEST_PATH,
        TEST_PATH,
        "Testing"
    )

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(f"Training records:   {train_count}")
    print(f"Validation records: {val_count}")
    print(f"Testing records:    {test_count}")

    print(
        f"Total records:      "
        f"{train_count + val_count + test_count}"
    )


if __name__ == "__main__":
    main()
