from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.cnn_model import create_resnet18_model
from src.fft_analysis import FEATURE_NAMES, analyze_fft


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CNN_MODEL_PATH = (
    PROJECT_ROOT / "models" / "cnn_model.pth"
)

FFT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "fft_classifier.pkl"
)


CLASS_NAMES = {
    0: "Likely Real",
    1: "Potentially AI-Generated",
}


# ---------------------------------------------------------
# CNN IMAGE TRANSFORM
# ---------------------------------------------------------

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ---------------------------------------------------------
# LOAD CNN MODEL
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_cnn_model():
    """
    Load the trained ResNet18 CNN model.
    """

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained CNN model was not found at: "
            f"{CNN_MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_resnet18_model(
        pretrained=False
    )

    checkpoint = torch.load(
        CNN_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    return model, device


# ---------------------------------------------------------
# LOAD FFT CLASSIFIER
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_fft_classifier():
    """
    Load the trained Random Forest FFT classifier.
    """

    if not FFT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained FFT classifier was not found at: "
            f"{FFT_MODEL_PATH}"
        )

    model_package = joblib.load(
        FFT_MODEL_PATH
    )

    if (
        isinstance(model_package, dict)
        and "model" in model_package
    ):
        classifier = model_package["model"]

        feature_names = model_package.get(
            "feature_names",
            FEATURE_NAMES,
        )

    else:
        classifier = model_package
        feature_names = FEATURE_NAMES

    return classifier, feature_names


# ---------------------------------------------------------
# PREPARE IMAGE FOR CNN
# ---------------------------------------------------------

def prepare_image_for_cnn(image):
    """
    Convert a PIL image into a normalized CNN tensor.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "The supplied image must be a PIL image."
        )

    rgb_image = image.convert("RGB")

    image_tensor = INFERENCE_TRANSFORM(
        rgb_image
    )

    # Add batch dimension:
    # [3, 224, 224] becomes [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# ---------------------------------------------------------
# CNN PREDICTION
# ---------------------------------------------------------

def predict_cnn(image):
    """
    Use the ResNet18 model to calculate the AI probability.
    """

    model, device = load_cnn_model()

    image_tensor = prepare_image_for_cnn(
        image
    ).to(device)

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

        real_probability = float(
            probabilities[0].item()
        )

        ai_probability = float(
            probabilities[1].item()
        )

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

    confidence = max(
        real_probability,
        ai_probability,
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": (
            CLASS_NAMES[predicted_class]
        ),
        "confidence": confidence,
        "real_probability": real_probability,
        "ai_probability": ai_probability,
        "device": str(device),
    }


# ---------------------------------------------------------
# FFT PREDICTION
# ---------------------------------------------------------

def predict_fft(image):
    """
    Analyse image frequencies and calculate the FFT AI probability.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "The supplied image must be a PIL image."
        )

    classifier, saved_feature_names = (
        load_fft_classifier()
    )

    fft_result = analyze_fft(
        image.convert("RGB")
    )

    feature_values = fft_result[
        "features"
    ]

    # Use the same feature order used during training.
    ordered_features = {
        feature_name: feature_values[feature_name]
        for feature_name in saved_feature_names
    }

    feature_dataframe = pd.DataFrame(
        [ordered_features],
        columns=saved_feature_names,
    )

    probabilities = classifier.predict_proba(
        feature_dataframe
    )[0]

    real_probability = float(
        probabilities[0]
    )

    ai_probability = float(
        probabilities[1]
    )

    predicted_class = int(
        classifier.predict(
            feature_dataframe
        )[0]
    )

    confidence = max(
        real_probability,
        ai_probability,
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": (
            CLASS_NAMES[predicted_class]
        ),
        "confidence": confidence,
        "real_probability": real_probability,
        "ai_probability": ai_probability,
        "spectrum_image": fft_result[
            "spectrum_image"
        ],
        "features": feature_values,
        "feature_vector": fft_result[
            "feature_vector"
        ],
    }


# ---------------------------------------------------------
# COMPLETE CNN AND FFT ANALYSIS
# ---------------------------------------------------------

def analyze_image(image):
    """
    Run both CNN and FFT analysis on the same image.
    """

    cnn_result = predict_cnn(image)

    fft_result = predict_fft(image)

    return {
        "cnn": cnn_result,
        "fft": fft_result,
    }


# ---------------------------------------------------------
# BACKWARD-COMPATIBLE FUNCTION
# ---------------------------------------------------------

def predict_image(image):
    """
    Retain the original function name used by app.py.

    This currently returns only the CNN result.
    """

    return predict_cnn(image)