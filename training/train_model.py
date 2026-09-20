import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.cnn_model import (
    CLASS_NAMES,
    count_trainable_parameters,
    create_resnet18_model
)
from src.preprocessing import IMAGE_SIZE, create_datasets


MODELS_FOLDER = PROJECT_ROOT / "models"
OUTPUTS_FOLDER = PROJECT_ROOT / "outputs"

BEST_MODEL_PATH = MODELS_FOLDER / "cnn_model.pth"
TRAINING_HISTORY_PATH = OUTPUTS_FOLDER / "training_history.csv"


def create_balanced_subset(dataset, maximum_images):
    """Select approximately equal numbers of real and AI images."""

    if maximum_images is None or maximum_images >= len(dataset):
        return dataset

    images_per_class = maximum_images // 2

    selected_indices = []
    class_counts = {
        0: 0,
        1: 0
    }

    for index, record in enumerate(dataset.records):
        label = record["label"]

        if class_counts[label] < images_per_class:
            selected_indices.append(index)
            class_counts[label] += 1

        if (
            class_counts[0] >= images_per_class
            and class_counts[1] >= images_per_class
        ):
            break

    return Subset(dataset, selected_indices)


def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    progress_bar = tqdm(
        data_loader,
        desc="Training",
        unit="batch"
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        correct_predictions += (
            predictions == labels
        ).sum().item()
        total_images += batch_size

        current_loss = running_loss / total_images
        current_accuracy = (
            100.0 * correct_predictions / total_images
        )

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            accuracy=f"{current_accuracy:.2f}%"
        )

    epoch_loss = running_loss / total_images
    epoch_accuracy = (
        100.0 * correct_predictions / total_images
    )

    return epoch_loss, epoch_accuracy


def validate_model(
    model,
    data_loader,
    loss_function,
    device
):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_images = 0

    progress_bar = tqdm(
        data_loader,
        desc="Validation",
        unit="batch"
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            correct_predictions += (
                predictions == labels
            ).sum().item()
            total_images += batch_size

    epoch_loss = running_loss / total_images
    epoch_accuracy = (
        100.0 * correct_predictions / total_images
    )

    return epoch_loss, epoch_accuracy


def save_checkpoint(
    model,
    validation_accuracy,
    epoch
):
    checkpoint = {
        "model_name": "resnet18",
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "validation_accuracy": validation_accuracy,
        "epoch": epoch
    }

    torch.save(checkpoint, BEST_MODEL_PATH)


def save_training_history(history):
    with TRAINING_HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "training_loss",
                "training_accuracy",
                "validation_loss",
                "validation_accuracy"
            ]
        )

        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(
        description="Train the AI-image ResNet18 classifier."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--train-limit",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--validation-limit",
        type=int,
        default=2000
    )

    arguments = parser.parse_args()

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUTS_FOLDER.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("AI Image Detector - CNN Training")
    print("--------------------------------")
    print(f"Device: {device}")

    training_dataset, validation_dataset, _ = create_datasets()

    training_subset = create_balanced_subset(
        training_dataset,
        arguments.train_limit
    )

    validation_subset = create_balanced_subset(
        validation_dataset,
        arguments.validation_limit
    )

    print(f"Training images used: {len(training_subset)}")
    print(f"Validation images used: {len(validation_subset)}")
    print(f"Epochs: {arguments.epochs}")
    print(f"Batch size: {arguments.batch_size}")

    training_loader = DataLoader(
        training_subset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = create_resnet18_model(
        pretrained=True,
        fine_tune=False
    )

    model = model.to(device)

    print(
        "Trainable parameters:",
        count_trainable_parameters(model)
    )
    print()

    loss_function = nn.CrossEntropyLoss()

    optimizer = Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=arguments.learning_rate
    )

    best_validation_accuracy = 0.0
    history = []

    training_start_time = time.time()

    for epoch in range(1, arguments.epochs + 1):
        print(f"Epoch {epoch}/{arguments.epochs}")

        training_loss, training_accuracy = train_one_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
            device
        )

        validation_loss, validation_accuracy = validate_model(
            model,
            validation_loader,
            loss_function,
            device
        )

        print(
            f"Training loss: {training_loss:.4f} | "
            f"Training accuracy: {training_accuracy:.2f}%"
        )

        print(
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": f"{training_loss:.6f}",
                "training_accuracy": f"{training_accuracy:.4f}",
                "validation_loss": f"{validation_loss:.6f}",
                "validation_accuracy": f"{validation_accuracy:.4f}"
            }
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            save_checkpoint(
                model,
                validation_accuracy,
                epoch
            )

            print(
                f"Best model saved to: {BEST_MODEL_PATH}"
            )

        save_training_history(history)
        print()

    elapsed_minutes = (
        time.time() - training_start_time
    ) / 60

    print("Training completed.")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.2f}%"
    )
    print(
        f"Total training time: "
        f"{elapsed_minutes:.2f} minutes"
    )
    print(f"Saved model: {BEST_MODEL_PATH}")
    print(f"Training history: {TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    main()