from pathlib import Path
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnn_model import create_resnet18_model
from src.preprocessing import create_datasets


MODEL_PATH = PROJECT_ROOT / "models" / "cnn_model.pth"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
VISUALIZATION_DIRECTORY = OUTPUT_DIRECTORY / "visualizations"

METRICS_PATH = OUTPUT_DIRECTORY / "cnn_test_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIRECTORY / "cnn_test_predictions.csv"
CONFUSION_MATRIX_PATH = (
    VISUALIZATION_DIRECTORY / "cnn_confusion_matrix.png"
)
ROC_CURVE_PATH = VISUALIZATION_DIRECTORY / "cnn_roc_curve.png"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 0

CLASS_NAMES = [
    "Real",
    "AI-Generated",
]


# ---------------------------------------------------------
# CREATE REQUIRED OUTPUT FOLDERS
# ---------------------------------------------------------

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD MODEL CHECKPOINT
# ---------------------------------------------------------

def load_trained_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model was not found at:\n{MODEL_PATH}"
        )

    print(f"Loading trained model from:\n{MODEL_PATH}")

    model = create_resnet18_model(pretrained=False)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    # Supports different checkpoint-saving formats.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------
# SAVE CONFUSION MATRIX
# ---------------------------------------------------------

def save_confusion_matrix(true_labels, predicted_labels):
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )

    plt.title("CNN Test Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    return matrix


# ---------------------------------------------------------
# SAVE ROC CURVE
# ---------------------------------------------------------

def save_roc_curve(true_labels, ai_probabilities, roc_auc):
    false_positive_rate, true_positive_rate, _ = roc_curve(
        true_labels,
        ai_probabilities,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        linewidth=2,
        label=f"ResNet18 AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        linestyle="--",
        linewidth=2,
        label="Random classifier",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CNN Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=300)
    plt.close()


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model():
    print()
    print("AI Image Detector - CNN Testing")
    print("--------------------------------")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    print("Loading test dataset...")

    _, _, test_dataset = create_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Testing images: {len(test_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")

    model = load_trained_model(device)

    true_labels = []
    predicted_labels = []
    ai_probabilities = []

    start_time = time.time()

    print()
    print("Testing the trained CNN model...")

    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc="Testing",
            unit="batch",
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            ai_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )

    evaluation_time = time.time() - start_time

    # -----------------------------------------------------
    # CALCULATE PERFORMANCE METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        true_labels,
        ai_probabilities,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = save_confusion_matrix(
        true_labels,
        predicted_labels,
    )

    save_roc_curve(
        true_labels,
        ai_probabilities,
        roc_auc,
    )

    # -----------------------------------------------------
    # SAVE INDIVIDUAL PREDICTIONS
    # -----------------------------------------------------

    predictions_dataframe = pd.DataFrame(
        {
            "true_label_number": true_labels,
            "true_label": [
                CLASS_NAMES[label] for label in true_labels
            ],
            "predicted_label_number": predicted_labels,
            "predicted_label": [
                CLASS_NAMES[label] for label in predicted_labels
            ],
            "ai_probability": ai_probabilities,
            "correct_prediction": [
                true_label == predicted_label
                for true_label, predicted_label in zip(
                    true_labels,
                    predicted_labels,
                )
            ],
        }
    )

    predictions_dataframe.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics = {
        "model": "ResNet18",
        "test_images": len(test_dataset),
        "device": str(device),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "evaluation_time_minutes": float(evaluation_time / 60),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print()
    print("CNN TEST RESULTS")
    print("----------------")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(
        f"Evaluation time: {evaluation_time / 60:.2f} minutes"
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    print()
    print("Testing completed successfully.")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(
        f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}"
    )
    print(f"ROC curve saved to: {ROC_CURVE_PATH}")


if __name__ == "__main__":
    evaluate_model()