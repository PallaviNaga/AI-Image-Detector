import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_DEVIATION = [0.229, 0.224, 0.225]


class ImageCsvDataset(Dataset):
    """Load image paths and labels from one of our CSV split files."""

    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.records = []

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV file was not found: {self.csv_path}"
            )

        with self.csv_path.open(
            "r",
            encoding="utf-8"
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                self.records.append(
                    {
                        "filepath": row["filepath"],
                        "label": int(row["label"]),
                        "class_name": row["class_name"]
                    }
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = PROJECT_ROOT / record["filepath"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = record["label"]

        return image, label


def get_training_transform():
    """Preprocessing and augmentation used during CNN training."""

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STANDARD_DEVIATION
            )
        ]
    )


def get_evaluation_transform():
    """Preprocessing used for validation, testing and prediction."""

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STANDARD_DEVIATION
            )
        ]
    )


def create_datasets():
    """Create training, validation and testing dataset objects."""

    splits_folder = PROJECT_ROOT / "data" / "splits"

    training_dataset = ImageCsvDataset(
        csv_path=splits_folder / "train.csv",
        transform=get_training_transform()
    )

    validation_dataset = ImageCsvDataset(
        csv_path=splits_folder / "validation.csv",
        transform=get_evaluation_transform()
    )

    testing_dataset = ImageCsvDataset(
        csv_path=splits_folder / "test.csv",
        transform=get_evaluation_transform()
    )

    return training_dataset, validation_dataset, testing_dataset