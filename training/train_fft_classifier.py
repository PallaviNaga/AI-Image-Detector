from pathlib import Path
import json
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fft_analysis import FEATURE_NAMES


FFT_FEATURE_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "fft_features"
)

TRAIN_FEATURE_PATH = (
    FFT_FEATURE_DIRECTORY / "train_fft_features.csv"
)

VALIDATION_FEATURE_PATH = (
    FFT_FEATURE_DIRECTORY
    / "validation_fft_features.csv"
)

TEST_FEATURE_PATH = (
    FFT_FEATURE_DIRECTORY / "test_fft_features.csv"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

VISUALIZATION_DIRECTORY = (
    OUTPUT_DIRECTORY / "visualizations"
)

FFT_MODEL_PATH = (
    MODEL_DIRECTORY / "fft_classifier.pkl"
)

METRICS_PATH = (
    OUTPUT_DIRECTORY / "fft_classifier_metrics.json"
)

PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY / "fft_test_predictions.csv"
)

CONFUSION_MATRIX_PATH = (
    VISUALIZATION_DIRECTORY
    / "fft_confusion_matrix.png"
)

ROC_CURVE_PATH = (
    VISUALIZATION_DIRECTORY
    / "fft_roc_curve.png"
)

FEATURE_IMPORTANCE_PATH = (
    VISUALIZATION_DIRECTORY
    / "fft_feature_importance.png"
)


CLASS_NAMES = [
    "Real",
    "AI-Generated",
]


# ---------------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# ---------------------------------------------------------

MODEL_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

VISUALIZATION_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# LOAD FFT FEATURE DATA
# ---------------------------------------------------------

def load_feature_data(csv_path):
    """
    Load extracted FFT features and labels.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"FFT feature file was not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    missing_features = [
        feature_name
        for feature_name in FEATURE_NAMES
        if feature_name not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "The following FFT features are missing from "
            f"{csv_path.name}: {missing_features}"
        )

    if "label" not in dataframe.columns:
        raise ValueError(
            f"The label column is missing from {csv_path}"
        )

    features = dataframe[FEATURE_NAMES]
    labels = dataframe["label"].astype(int)

    return dataframe, features, labels


# ---------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------

def calculate_metrics(
    true_labels,
    predicted_labels,
    ai_probabilities,
):
    """
    Calculate classification performance metrics.
    """

    return {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                true_labels,
                ai_probabilities,
            )
        ),
        "classification_report": (
            classification_report(
                true_labels,
                predicted_labels,
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                true_labels,
                predicted_labels,
                labels=[0, 1],
            ).tolist()
        ),
    }


# ---------------------------------------------------------
# DISPLAY METRICS
# ---------------------------------------------------------

def print_metrics(title, metrics):
    print()
    print(title)
    print("-" * len(title))

    print(
        f"Accuracy:  {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: {metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:    {metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1-score:  {metrics['f1_score'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC:   {metrics['roc_auc']:.4f}"
    )

    print(
        "Confusion matrix:",
        metrics["confusion_matrix"],
    )


# ---------------------------------------------------------
# SAVE CONFUSION MATRIX
# ---------------------------------------------------------

def save_confusion_matrix(matrix):
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )

    plt.title("FFT Random Forest Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------
# SAVE ROC CURVE
# ---------------------------------------------------------

def save_roc_curve(
    true_labels,
    ai_probabilities,
    roc_auc,
):
    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            true_labels,
            ai_probabilities,
        )
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        linewidth=2,
        label=f"Random Forest AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="navy",
        linewidth=2,
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("FFT Classifier ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_PATH,
        dpi=300,
    )

    plt.close()


# ---------------------------------------------------------
# SAVE FEATURE IMPORTANCE
# ---------------------------------------------------------

def save_feature_importance(model):
    importance_dataframe = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": model.feature_importances_,
        }
    )

    importance_dataframe = (
        importance_dataframe.sort_values(
            by="importance",
            ascending=True,
        )
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance_dataframe["feature"],
        importance_dataframe["importance"],
        color="teal",
    )

    plt.xlabel("Importance")
    plt.ylabel("FFT Feature")
    plt.title("Random Forest FFT Feature Importance")
    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_PATH,
        dpi=300,
    )

    plt.close()

    feature_importance_csv_path = (
        OUTPUT_DIRECTORY
        / "fft_feature_importance.csv"
    )

    importance_dataframe.sort_values(
        by="importance",
        ascending=False,
    ).to_csv(
        feature_importance_csv_path,
        index=False,
    )


# ---------------------------------------------------------
# MAIN TRAINING PROGRAM
# ---------------------------------------------------------

def main():
    print()
    print("AI Image Detector - FFT Classifier Training")
    print("-------------------------------------------")

    print("Loading FFT feature files...")

    (
        train_dataframe,
        training_features,
        training_labels,
    ) = load_feature_data(
        TRAIN_FEATURE_PATH
    )

    (
        validation_dataframe,
        validation_features,
        validation_labels,
    ) = load_feature_data(
        VALIDATION_FEATURE_PATH
    )

    (
        test_dataframe,
        test_features,
        test_labels,
    ) = load_feature_data(
        TEST_FEATURE_PATH
    )

    print(
        f"Training samples: {len(training_features)}"
    )

    print(
        f"Validation samples: {len(validation_features)}"
    )

    print(
        f"Testing samples: {len(test_features)}"
    )

    print(
        f"Number of FFT features: {len(FEATURE_NAMES)}"
    )

    # -----------------------------------------------------
    # CREATE RANDOM FOREST
    # -----------------------------------------------------

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print()
    print("Training Random Forest classifier...")

    training_start_time = time.time()

    classifier.fit(
        training_features,
        training_labels,
    )

    training_time = (
        time.time() - training_start_time
    )

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds."
    )

    # -----------------------------------------------------
    # VALIDATION RESULTS
    # -----------------------------------------------------

    validation_predictions = classifier.predict(
        validation_features
    )

    validation_probabilities = (
        classifier.predict_proba(
            validation_features
        )[:, 1]
    )

    validation_metrics = calculate_metrics(
        true_labels=validation_labels,
        predicted_labels=validation_predictions,
        ai_probabilities=validation_probabilities,
    )

    print_metrics(
        "VALIDATION RESULTS",
        validation_metrics,
    )

    # -----------------------------------------------------
    # TEST RESULTS
    # -----------------------------------------------------

    test_predictions = classifier.predict(
        test_features
    )

    test_probabilities = classifier.predict_proba(
        test_features
    )[:, 1]

    test_metrics = calculate_metrics(
        true_labels=test_labels,
        predicted_labels=test_predictions,
        ai_probabilities=test_probabilities,
    )

    print_metrics(
        "TEST RESULTS",
        test_metrics,
    )

    # -----------------------------------------------------
    # SAVE CLASSIFIER
    # -----------------------------------------------------

    model_package = {
        "model": classifier,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "model_type": "RandomForestClassifier",
    }

    joblib.dump(
        model_package,
        FFT_MODEL_PATH,
    )

    # -----------------------------------------------------
    # SAVE TEST PREDICTIONS
    # -----------------------------------------------------

    prediction_dataframe = pd.DataFrame(
        {
            "image_path": (
                test_dataframe["image_path"]
            ),
            "true_label": test_labels,
            "predicted_label": test_predictions,
            "ai_probability": test_probabilities,
            "correct_prediction": (
                test_labels.to_numpy()
                == test_predictions
            ),
        }
    )

    prediction_dataframe.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    all_metrics = {
        "model_type": "RandomForestClassifier",
        "number_of_trees": 300,
        "number_of_features": len(FEATURE_NAMES),
        "training_samples": len(training_features),
        "validation_samples": len(
            validation_features
        ),
        "testing_samples": len(test_features),
        "training_time_seconds": float(
            training_time
        ),
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            all_metrics,
            metrics_file,
            indent=4,
        )

    # -----------------------------------------------------
    # SAVE VISUALIZATIONS
    # -----------------------------------------------------

    save_confusion_matrix(
        test_metrics["confusion_matrix"]
    )

    save_roc_curve(
        true_labels=test_labels,
        ai_probabilities=test_probabilities,
        roc_auc=test_metrics["roc_auc"],
    )

    save_feature_importance(
        classifier
    )

    # -----------------------------------------------------
    # COMPLETION MESSAGE
    # -----------------------------------------------------

    print()
    print("FFT classifier training completed.")
    print(f"Classifier saved to: {FFT_MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(
        f"Predictions saved to: {PREDICTIONS_PATH}"
    )
    print(
        "Confusion matrix saved to: "
        f"{CONFUSION_MATRIX_PATH}"
    )
    print(
        f"ROC curve saved to: {ROC_CURVE_PATH}"
    )
    print(
        "Feature-importance graph saved to: "
        f"{FEATURE_IMPORTANCE_PATH}"
    )


if __name__ == "__main__":
    main()