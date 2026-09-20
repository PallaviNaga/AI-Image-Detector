import csv
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError


# Project locations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "cifake"
SPLITS_FOLDER = PROJECT_ROOT / "data" / "splits"

# Dataset settings
RANDOM_SEED = 42
VALIDATION_RATIO = 0.20
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_valid_images(folder):
    """Collect supported images and ignore corrupted files."""

    valid_images = []
    invalid_images = []

    for image_path in sorted(folder.iterdir()):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            try:
                with Image.open(image_path) as image:
                    image.verify()

                valid_images.append(image_path)

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):
                invalid_images.append(image_path)

    return valid_images, invalid_images


def make_record(image_path, label, class_name):
    """Create one CSV record using a project-relative file path."""

    relative_path = image_path.relative_to(PROJECT_ROOT).as_posix()

    return {
        "filepath": relative_path,
        "label": label,
        "class_name": class_name
    }


def write_csv(filename, records):
    """Write dataset records into a CSV file."""

    output_path = SPLITS_FOLDER / filename

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filepath", "label", "class_name"]
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Created: {output_path}")
    print(f"Number of images: {len(records)}")


def main():
    random.seed(RANDOM_SEED)
    SPLITS_FOLDER.mkdir(parents=True, exist_ok=True)

    train_records = []
    validation_records = []
    test_records = []
    all_invalid_images = []

    classes = {
        "REAL": {
            "label": 0,
            "class_name": "real"
        },
        "FAKE": {
            "label": 1,
            "class_name": "ai_generated"
        }
    }

    print("Checking CIFAKE dataset...")
    print("This may take a few minutes.")
    print()

    # Divide the original training set into training and validation.
    for source_class, class_information in classes.items():
        source_folder = SOURCE_ROOT / "train" / source_class

        if not source_folder.exists():
            raise FileNotFoundError(
                f"Required folder was not found: {source_folder}"
            )

        images, invalid_images = collect_valid_images(source_folder)
        all_invalid_images.extend(invalid_images)

        random.shuffle(images)

        validation_count = int(
            len(images) * VALIDATION_RATIO
        )

        validation_images = images[:validation_count]
        training_images = images[validation_count:]

        for image_path in training_images:
            train_records.append(
                make_record(
                    image_path,
                    class_information["label"],
                    class_information["class_name"]
                )
            )

        for image_path in validation_images:
            validation_records.append(
                make_record(
                    image_path,
                    class_information["label"],
                    class_information["class_name"]
                )
            )

    # Keep the original test set unchanged.
    for source_class, class_information in classes.items():
        source_folder = SOURCE_ROOT / "test" / source_class

        if not source_folder.exists():
            raise FileNotFoundError(
                f"Required folder was not found: {source_folder}"
            )

        images, invalid_images = collect_valid_images(source_folder)
        all_invalid_images.extend(invalid_images)

        for image_path in images:
            test_records.append(
                make_record(
                    image_path,
                    class_information["label"],
                    class_information["class_name"]
                )
            )

    # Shuffle the final records so classes are mixed.
    random.shuffle(train_records)
    random.shuffle(validation_records)
    random.shuffle(test_records)

    write_csv("train.csv", train_records)
    write_csv("validation.csv", validation_records)
    write_csv("test.csv", test_records)

    print()
    print("Dataset preparation completed.")
    print(f"Training images: {len(train_records)}")
    print(f"Validation images: {len(validation_records)}")
    print(f"Testing images: {len(test_records)}")
    print(f"Invalid images ignored: {len(all_invalid_images)}")

    if all_invalid_images:
        invalid_file = SPLITS_FOLDER / "invalid_images.txt"

        with invalid_file.open("w", encoding="utf-8") as text_file:
            for image_path in all_invalid_images:
                text_file.write(str(image_path) + "\n")

        print(f"Invalid-image list: {invalid_file}")


if __name__ == "__main__":
    main()