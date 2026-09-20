# ---------------------------------------------------------
# HYBRID MODEL SETTINGS
# ---------------------------------------------------------

CNN_WEIGHT = 0.70
FFT_WEIGHT = 0.30

LIKELY_REAL_THRESHOLD = 0.35
AI_GENERATED_THRESHOLD = 0.65


# ---------------------------------------------------------
# VALIDATE PROBABILITY
# ---------------------------------------------------------

def validate_probability(
    probability,
    probability_name,
):
    """
    Ensure a probability is a numeric value from 0 to 1.
    """

    try:
        probability = float(probability)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{probability_name} must be a number."
        ) from error

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{probability_name} must be between 0 and 1."
        )

    return probability


# ---------------------------------------------------------
# CLASSIFY FINAL SCORE
# ---------------------------------------------------------

def classify_final_score(final_score):
    """
    Convert the combined AI probability into a three-way result.
    """

    if final_score < LIKELY_REAL_THRESHOLD:
        return {
            "label": "Likely Real",
            "classification_code": "likely_real",
            "risk_level": "Low",
            "colour": "green",
        }

    if final_score < AI_GENERATED_THRESHOLD:
        return {
            "label": "Inconclusive",
            "classification_code": "inconclusive",
            "risk_level": "Moderate",
            "colour": "orange",
        }

    return {
        "label": "Potentially AI-Generated",
        "classification_code": "potentially_ai_generated",
        "risk_level": "High",
        "colour": "red",
    }


# ---------------------------------------------------------
# CHECK MODEL AGREEMENT
# ---------------------------------------------------------

def check_model_agreement(
    cnn_ai_probability,
    fft_ai_probability,
):
    """
    Describe whether the CNN and FFT branches agree.
    """

    cnn_class = (
        1 if cnn_ai_probability >= 0.50 else 0
    )

    fft_class = (
        1 if fft_ai_probability >= 0.50 else 0
    )

    probability_difference = abs(
        cnn_ai_probability
        - fft_ai_probability
    )

    if cnn_class == fft_class:
        agreement_status = "Models agree"

        if cnn_class == 1:
            agreement_message = (
                "Both CNN and FFT detected indicators "
                "consistent with AI-generated content."
            )
        else:
            agreement_message = (
                "Both CNN and FFT produced results "
                "more consistent with a real image."
            )

    else:
        agreement_status = "Models disagree"

        agreement_message = (
            "The CNN and FFT branches produced different "
            "classifications. The final result should be "
            "interpreted cautiously."
        )

    return {
        "agreement": cnn_class == fft_class,
        "agreement_status": agreement_status,
        "agreement_message": agreement_message,
        "probability_difference": (
            probability_difference
        ),
    }


# ---------------------------------------------------------
# COMBINE CNN AND FFT SCORES
# ---------------------------------------------------------

def combine_scores(
    cnn_ai_probability,
    fft_ai_probability,
    cnn_weight=CNN_WEIGHT,
    fft_weight=FFT_WEIGHT,
):
    """
    Combine CNN and FFT AI probabilities.

    Default formula:

    Final score = 0.70(CNN score) + 0.30(FFT score)
    """

    cnn_ai_probability = validate_probability(
        cnn_ai_probability,
        "CNN AI probability",
    )

    fft_ai_probability = validate_probability(
        fft_ai_probability,
        "FFT AI probability",
    )

    cnn_weight = float(cnn_weight)
    fft_weight = float(fft_weight)

    if cnn_weight < 0 or fft_weight < 0:
        raise ValueError(
            "Model weights cannot be negative."
        )

    total_weight = cnn_weight + fft_weight

    if total_weight <= 0:
        raise ValueError(
            "The total model weight must be greater than zero."
        )

    # Normalize weights in case they do not add exactly to 1.
    normalized_cnn_weight = (
        cnn_weight / total_weight
    )

    normalized_fft_weight = (
        fft_weight / total_weight
    )

    final_ai_probability = (
        normalized_cnn_weight
        * cnn_ai_probability
        + normalized_fft_weight
        * fft_ai_probability
    )

    final_real_probability = (
        1.0 - final_ai_probability
    )

    classification = classify_final_score(
        final_ai_probability
    )

    agreement_information = check_model_agreement(
        cnn_ai_probability,
        fft_ai_probability,
    )

    # Confidence means confidence in the chosen side.
    # For an inconclusive result, the score's closeness
    # to 50% is intentionally retained.
    if (
        classification["classification_code"]
        == "likely_real"
    ):
        confidence = final_real_probability

    elif (
        classification["classification_code"]
        == "potentially_ai_generated"
    ):
        confidence = final_ai_probability

    else:
        confidence = 1.0 - abs(
            final_ai_probability - 0.50
        )

    return {
        "final_ai_probability": (
            final_ai_probability
        ),
        "final_real_probability": (
            final_real_probability
        ),
        "final_score_percentage": (
            final_ai_probability * 100
        ),
        "label": classification["label"],
        "classification_code": (
            classification[
                "classification_code"
            ]
        ),
        "risk_level": (
            classification["risk_level"]
        ),
        "colour": classification["colour"],
        "confidence": confidence,
        "cnn_ai_probability": (
            cnn_ai_probability
        ),
        "fft_ai_probability": (
            fft_ai_probability
        ),
        "cnn_weight": normalized_cnn_weight,
        "fft_weight": normalized_fft_weight,
        "likely_real_threshold": (
            LIKELY_REAL_THRESHOLD
        ),
        "ai_generated_threshold": (
            AI_GENERATED_THRESHOLD
        ),
        **agreement_information,
    }


# ---------------------------------------------------------
# BUILD SUPPORTING OBSERVATIONS
# ---------------------------------------------------------

def build_supporting_observations(
    fusion_result,
    metadata_result=None,
):
    """
    Create understandable observations for the user report.

    Metadata is supporting evidence only and does not directly
    change the numerical final score.
    """

    observations = []

    cnn_probability = fusion_result[
        "cnn_ai_probability"
    ]

    fft_probability = fusion_result[
        "fft_ai_probability"
    ]

    if cnn_probability >= 0.65:
        observations.append(
            "The CNN detected strong visual indicators "
            "associated with AI-generated images."
        )

    elif cnn_probability <= 0.35:
        observations.append(
            "The CNN result was more consistent with "
            "a real image."
        )

    else:
        observations.append(
            "The CNN result was uncertain."
        )

    if fft_probability >= 0.65:
        observations.append(
            "The FFT classifier detected suspicious "
            "frequency-domain characteristics."
        )

    elif fft_probability <= 0.35:
        observations.append(
            "The FFT result was more consistent with "
            "the frequency characteristics of real images."
        )

    else:
        observations.append(
            "The FFT frequency result was uncertain."
        )

    observations.append(
        fusion_result["agreement_message"]
    )

    if metadata_result is not None:
        if metadata_result.get(
            "ai_software_detected",
            False,
        ):
            observations.append(
                "Image metadata contains the name of "
                "a known AI-generation tool."
            )

        elif metadata_result.get(
            "software"
        ):
            observations.append(
                "Software metadata was detected: "
                f"{metadata_result['software']}."
            )

        else:
            observations.append(
                "No software metadata was detected."
            )

        if (
            metadata_result.get(
                "camera_make"
            )
            or metadata_result.get(
                "camera_model"
            )
        ):
            observations.append(
                "Camera make or model information "
                "was detected."
            )
        else:
            observations.append(
                "No camera make or model information "
                "was detected."
            )

    observations.append(
        "This result is probabilistic and should not be "
        "treated as definitive proof of image origin."
    )

    return observations