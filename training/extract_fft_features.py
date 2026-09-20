from pathlib import Path
import argparse
import sys
import time

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fft_analysis import FEATURE_NAMES, analyze_fft


SPLITS_DIRECTORY = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "fft_features"

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# FIND CSV COLUMNS
# ---------------------------------------------------------

def find_column(dataframe, possible_names):
    """
    Find a column using a list of possible column names.
    """

    lower_column_mapping = {
        column.lower(): column
        for column in dataframe.columns
    }

    for possible_name in possible_names:
        if possible_name.lower() in lower_column_mapping:
            return lower_column_mapping[
                possible_name.lower()
            ]

    return None


# ---------------------------------------------------------
# NORMALIZE LABEL
# ---------------------------------------------------------

def normalize_label(label):
    """
    Convert dataset labels into:
    0 = Real
    1 = AI-generated
    """

    if isinstance(label, str):
        normalized_label = label.strip().lower()

        if normalized_label in {
            "real",
            "0",
            "camera",
            "authentic",
        }:
            return 0

        if normalized_label in {
            "fake",
            "ai",
            "ai_generated",
            "ai-generated",
            "generated",
            "1",
        }:
            return 1

    numeric_label = int(label)

    if numeric_label not in {0, 1}:
        raise ValueError(
            f"Unsupported label value: {label}"
        )

    return numeric_label


# ---------------------------------------------------------
# RESOLVE IMAGE PATH
# ---------------------------------------------------------

def resolve_image_path(image_path_value):
    """
    Convert a CSV image path into a valid absolute path.
    """

    image_path = Path(str(image_path_value))

    if image_path.is_absolute():
        return image_path

    possible_paths = [
        PROJECT_ROOT / image_path,
        SPLITS_DIRECTORY / image_path,
    ]

    for possible_path in possible_paths:
        if possible_path.exists():
            return possible_path

    return PROJECT_ROOT / image_path


# ---------------------------------------------------------
# CREATE BALANCED SUBSET
# ---------------------------------------------------------

def create_balanced_subset(
    dataframe,
    label_column,
    limit,
    random_seed=42,
):
    """
    Select approximately the same number of real and AI images.
    """

    if limit is None or limit <= 0:
        return dataframe.sample(
            frac=1,
            random_state=random_seed,
        ).reset_index(drop=True)

    if limit >= len(dataframe):
        return dataframe.sample(
            frac=1,
            random_state=random_seed,
        ).reset_index(drop=True)

    dataframe = dataframe.copy()

    dataframe["_normalized_label"] = dataframe[
        label_column
    ].apply(normalize_label)

    number_of_classes = dataframe[
        "_normalized_label"
    ].nunique()

    images_per_class = limit // number_of_classes

    selected_groups = []

    for label_value, group in dataframe.groupby(
        "_normalized_label"
    ):
        sample_size = min(
            images_per_class,
            len(group),
        )

        selected_group = group.sample(
            n=sample_size,
            random_state=random_seed,
        )

        selected_groups.append(
            selected_group
        )

    balanced_dataframe = pd.concat(
        selected_groups,
        ignore_index=True,
    )

    balanced_dataframe = balanced_dataframe.sample(
        frac=1,
        random_state=random_seed,
    ).reset_index(drop=True)

    return balanced_dataframe


# ---------------------------------------------------------
# EXTRACT FEATURES FROM ONE SPLIT
# ---------------------------------------------------------

def extract_features_from_split(
    split_name,
    limit,
):
    csv_path = SPLITS_DIRECTORY / f"{split_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset split file was not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    path_column = find_column(
        dataframe,
        [
            "image_path",
            "filepath",
            "file_path",
            "path",
            "filename",
        ],
    )

    label_column = find_column(
        dataframe,
        [
            "label",
            "class",
            "target",
            "category",
        ],
    )

    if path_column is None:
        raise ValueError(
            f"No image-path column was found in {csv_path}.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    if label_column is None:
        raise ValueError(
            f"No label column was found in {csv_path}.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    selected_dataframe = create_balanced_subset(
        dataframe=dataframe,
        label_column=label_column,
        limit=limit,
    )

    print()
    print(f"Processing {split_name} split")
    print(f"CSV file: {csv_path}")
    print(f"Images selected: {len(selected_dataframe)}")
    print(f"Path column: {path_column}")
    print(f"Label column: {label_column}")

    extracted_rows = []
    failed_images = []

    start_time = time.time()

    iterator = selected_dataframe.iterrows()

    for _, row in tqdm(
        iterator,
        total=len(selected_dataframe),
        desc=f"Extracting {split_name} FFT features",
        unit="image",
    ):
        image_path = resolve_image_path(
            row[path_column]
        )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                fft_result = analyze_fft(image)

            feature_row = {
                "image_path": str(image_path),
                "label": normalize_label(
                    row[label_column]
                ),
            }

            for feature_name in FEATURE_NAMES:
                feature_row[feature_name] = float(
                    fft_result["features"][feature_name]
                )

            extracted_rows.append(
                feature_row
            )

        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:
            failed_images.append(
                {
                    "image_path": str(image_path),
                    "error": str(error),
                }
            )

    elapsed_time = time.time() - start_time

    output_dataframe = pd.DataFrame(
        extracted_rows
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"{split_name}_fft_features.csv"
    )

    output_dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"{split_name.capitalize()} extraction completed.")
    print(f"Successful images: {len(extracted_rows)}")
    print(f"Failed images: {len(failed_images)}")
    print(
        f"Extraction time: {elapsed_time / 60:.2f} minutes"
    )
    print(f"Saved to: {output_path}")

    if failed_images:
        failed_dataframe = pd.DataFrame(
            failed_images
        )

        failed_path = (
            OUTPUT_DIRECTORY
            / f"{split_name}_failed_images.csv"
        )

        failed_dataframe.to_csv(
            failed_path,
            index=False,
        )

        print(
            f"Failed-image list saved to: {failed_path}"
        )

    return output_path


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract FFT features from CIFAKE images."
        )
    )

    parser.add_argument(
        "--train-limit",
        type=int,
        default=10000,
        help="Maximum number of training images.",
    )

    parser.add_argument(
        "--validation-limit",
        type=int,
        default=2000,
        help="Maximum number of validation images.",
    )

    parser.add_argument(
        "--test-limit",
        type=int,
        default=2000,
        help="Maximum number of testing images.",
    )

    arguments = parser.parse_args()

    print()
    print("AI Image Detector - FFT Feature Extraction")
    print("------------------------------------------")
    print(f"Training limit: {arguments.train_limit}")
    print(
        f"Validation limit: {arguments.validation_limit}"
    )
    print(f"Testing limit: {arguments.test_limit}")

    complete_start_time = time.time()

    extract_features_from_split(
        split_name="train",
        limit=arguments.train_limit,
    )

    extract_features_from_split(
        split_name="validation",
        limit=arguments.validation_limit,
    )

    extract_features_from_split(
        split_name="test",
        limit=arguments.test_limit,
    )

    complete_time = time.time() - complete_start_time

    print()
    print("All FFT features extracted successfully.")
    print(
        f"Total time: {complete_time / 60:.2f} minutes"
    )
    print(f"Feature files are located in:")
    print(OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()