import hashlib
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.explainability import generate_gradcam
from src.inference import analyze_image
from src.metadata_analysis import analyze_metadata
from src.report_generator import generate_forensic_report
from src.score_fusion import (
    build_supporting_observations,
    combine_scores,
)


st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
)


def probability(result, key, default=0.0):
    try:
        value = float(result.get(key, default))
        return max(0.0, min(1.0, value))
    except (TypeError, ValueError, AttributeError):
        return default


def prediction(result):
    value = result.get(
        "prediction",
        result.get("label"),
    )

    if value is None:
        value = (
            "AI-Generated"
            if probability(result, "ai_probability") >= 0.5
            else "Real"
        )

    return str(value)


def confidence(result):
    if result.get("confidence") is not None:
        return probability(result, "confidence")

    ai_score = probability(
        result,
        "ai_probability",
    )

    real_score = probability(
        result,
        "real_probability",
        1.0 - ai_score,
    )

    return max(ai_score, real_score)


def show_fft_feature(features, key, title):
    value = features.get(key)

    if value is not None:
        try:
            st.write(
                f"**{title}:** {float(value):.6f}"
            )
        except (TypeError, ValueError):
            st.write(f"**{title}:** {value}")


st.title("AI-Generated Image Detector")

st.write(
    "Upload an image to analyse it using CNN, FFT, metadata, "
    "score fusion, Grad-CAM explainability, and an automatic "
    "forensic PDF report."
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)


if uploaded_file is None:
    st.info(
        "Upload a JPG, JPEG, PNG, or WebP image to begin."
    )
    st.stop()


try:
    file_bytes = uploaded_file.getvalue()

    image = Image.open(
        BytesIO(file_bytes)
    ).convert("RGB")

except UnidentifiedImageError:
    st.error(
        "The uploaded file is not a valid supported image."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to open image: {error}")
    st.stop()


file_hash = hashlib.sha256(
    file_bytes
).hexdigest()


# Clear results when a different image is uploaded.
if (
    st.session_state.get("current_file_hash")
    != file_hash
):
    st.session_state["current_file_hash"] = file_hash
    st.session_state["analysis_result"] = None
    st.session_state["pdf_report"] = None


# ---------------------------------------------------------
# UPLOADED IMAGE INFORMATION
# ---------------------------------------------------------

image_column, information_column = st.columns(
    [1.4, 1]
)

with image_column:
    st.subheader("Uploaded Image")

    st.image(
        image,
        width="stretch",
    )

with information_column:
    st.subheader("Image Information")

    st.write(
        f"**Filename:** {uploaded_file.name}"
    )

    st.write(
        f"**File type:** "
        f"{uploaded_file.type or 'Unknown'}"
    )

    st.write(
        f"**File size:** "
        f"{len(file_bytes) / 1024:.2f} KB"
    )

    st.write(
        f"**Width:** {image.width} pixels"
    )

    st.write(
        f"**Height:** {image.height} pixels"
    )

    st.write(
        f"**Colour mode:** {image.mode}"
    )

    st.write("**SHA-256 hash:**")

    st.code(file_hash)


# ---------------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------------

if st.button(
    "Analyse Image",
    type="primary",
    width="stretch",
):
    try:
        with st.spinner(
            "Running CNN, FFT, metadata, score fusion, "
            "and Grad-CAM analysis..."
        ):
            model_results = analyze_image(image)

            cnn_result = model_results.get(
                "cnn",
                {},
            )

            fft_result = model_results.get(
                "fft",
                {},
            )

            metadata_result = analyze_metadata(
                file_bytes,
                filename=uploaded_file.name,
                mime_type=uploaded_file.type,
            )

            fusion_result = combine_scores(
                probability(
                    cnn_result,
                    "ai_probability",
                ),
                probability(
                    fft_result,
                    "ai_probability",
                ),
            )

            observations = (
                build_supporting_observations(
                    fusion_result,
                    metadata_result,
                )
            )

            gradcam_result = generate_gradcam(
                image
            )

            st.session_state["analysis_result"] = {
                "models": model_results,
                "metadata": metadata_result,
                "fusion": fusion_result,
                "observations": observations,
                "gradcam": gradcam_result,
            }

            st.session_state["pdf_report"] = None

    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.exception(error)


result = st.session_state.get(
    "analysis_result"
)

if result is None:
    st.stop()


models = result.get("models", {})

cnn_result = models.get("cnn", {})
fft_result = models.get("fft", {})

metadata_result = result.get(
    "metadata",
    {},
)

fusion_result = result.get(
    "fusion",
    {},
)

observations = result.get(
    "observations",
    [],
)

gradcam_result = result.get(
    "gradcam",
    {},
)


cnn_ai = probability(
    cnn_result,
    "ai_probability",
)

cnn_real = probability(
    cnn_result,
    "real_probability",
    1.0 - cnn_ai,
)

fft_ai = probability(
    fft_result,
    "ai_probability",
)

fft_real = probability(
    fft_result,
    "real_probability",
    1.0 - fft_ai,
)

final_ai = probability(
    fusion_result,
    "final_ai_probability",
)

final_real = 1.0 - final_ai

final_label = fusion_result.get(
    "label",
    fusion_result.get(
        "classification",
        "Inconclusive",
    ),
)

agreement = fusion_result.get(
    "agreement_status",
    "Unavailable",
)


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

st.divider()
st.header("Final Forensic Result")

if final_label == "Likely Real":
    st.success(
        f"Classification: {final_label}"
    )

elif final_label == "Potentially AI-Generated":
    st.error(
        f"Classification: {final_label}"
    )

else:
    st.warning(
        f"Classification: {final_label}"
    )


column_1, column_2, column_3, column_4 = (
    st.columns(4)
)

column_1.metric(
    "Final AI Probability",
    f"{final_ai * 100:.2f}%",
)

column_2.metric(
    "CNN AI Score",
    f"{cnn_ai * 100:.2f}%",
)

column_3.metric(
    "FFT AI Score",
    f"{fft_ai * 100:.2f}%",
)

column_4.metric(
    "Final Real Probability",
    f"{final_real * 100:.2f}%",
)


st.write(
    "**Combined AI-generation probability:**"
)

st.progress(
    final_ai,
    text=f"{final_ai * 100:.2f}%",
)


if "disagree" in str(agreement).lower():
    st.warning(
        "The CNN and FFT models disagree. Interpret this "
        "result carefully and review both model results."
    )
else:
    st.info(
        f"Model agreement: {agreement}"
    )


st.caption(
    "Fusion weights: CNN 70% and FFT 30%. Metadata is "
    "supporting evidence and does not directly change "
    "the probability score."
)


# ---------------------------------------------------------
# INDIVIDUAL MODELS
# ---------------------------------------------------------

st.divider()
st.header("Individual Model Analysis")

cnn_column, fft_column = st.columns(2)

with cnn_column:
    st.subheader("CNN Analysis")

    st.write(
        f"**Prediction:** {prediction(cnn_result)}"
    )

    st.write(
        f"**AI probability:** {cnn_ai * 100:.2f}%"
    )

    st.write(
        f"**Real probability:** {cnn_real * 100:.2f}%"
    )

    st.write(
        f"**Confidence:** "
        f"{confidence(cnn_result) * 100:.2f}%"
    )

    st.write("**Model:** ResNet18")

with fft_column:
    st.subheader("FFT Analysis")

    st.write(
        f"**Prediction:** {prediction(fft_result)}"
    )

    st.write(
        f"**AI probability:** {fft_ai * 100:.2f}%"
    )

    st.write(
        f"**Real probability:** {fft_real * 100:.2f}%"
    )

    st.write(
        f"**Confidence:** "
        f"{confidence(fft_result) * 100:.2f}%"
    )

    st.write("**Model:** Random Forest")


# ---------------------------------------------------------
# GRAD-CAM
# ---------------------------------------------------------

st.divider()
st.header("CNN Explainability — Grad-CAM")

if gradcam_result:
    original_column, overlay_column, strong_column = st.columns(3)

    with original_column:
        st.subheader("Original Image")

        st.image(
            image,
            width="stretch",
        )

    with overlay_column:
        st.subheader("Grad-CAM Overlay")

        overlay_image = gradcam_result.get(
            "overlay_image"
        )

        if overlay_image is not None:
            st.image(
                overlay_image,
                caption=(
                    "Red and yellow regions had the strongest "
                    "influence on the CNN prediction."
                ),
                width="stretch",
            )
        else:
            st.warning(
                "Grad-CAM overlay is unavailable."
            )

    with strong_column:
        st.subheader("Strongest Influence")

        strong_influence_image = gradcam_result.get(
            "strong_influence_image"
        )

        if strong_influence_image is not None:
            st.image(
                strong_influence_image,
                caption=(
                    "The top 20% of positive Grad-CAM activation is "
                    "emphasized in red and outlined."
                ),
                width="stretch",
            )
        else:
            st.warning(
                "Strong-influence visualization is unavailable."
            )

    explained_label = gradcam_result.get(
        "explained_label",
        gradcam_result.get(
            "prediction",
            "Unknown",
        ),
    )

    gradcam_ai = probability(
        gradcam_result,
        "ai_probability",
    )

    st.write(
        f"**CNN prediction explained:** "
        f"{explained_label}"
    )

    st.write(
        f"**AI probability:** "
        f"{gradcam_ai * 100:.2f}%"
    )

    st.info(
        "Grad-CAM highlights regions that influenced the CNN. "
        "Red and yellow indicate stronger influence, while blue "
        "indicates weaker influence. The Strongest Influence view "
        "emphasizes the top 20% of positive activation. These regions "
        "are CNN attention areas, not proven AI-generated or edited "
        "parts of the image."
    )

else:
    st.warning(
        "Grad-CAM explanation is unavailable."
    )


# ---------------------------------------------------------
# FFT ANALYSIS
# ---------------------------------------------------------

st.divider()
st.header("Frequency-Domain Analysis")

spectrum_image = fft_result.get(
    "spectrum_image"
)

fft_features = fft_result.get(
    "features",
    {},
)

if spectrum_image is not None:
    spectrum_column, features_column = st.columns(
        [1, 1.3]
    )

    with spectrum_column:
        st.subheader("FFT Spectrum")

        st.image(
            spectrum_image,
            caption="Log-scaled frequency spectrum",
            width="stretch",
        )

    with features_column:
        st.subheader("Important FFT Features")

        show_fft_feature(
            fft_features,
            "low_frequency_energy",
            "Low-frequency energy",
        )

        show_fft_feature(
            fft_features,
            "mid_frequency_energy",
            "Mid-frequency energy",
        )

        show_fft_feature(
            fft_features,
            "high_frequency_energy",
            "High-frequency energy",
        )

        show_fft_feature(
            fft_features,
            "high_low_energy_ratio",
            "High/low energy ratio",
        )

        show_fft_feature(
            fft_features,
            "spectral_entropy",
            "Spectral entropy",
        )

        show_fft_feature(
            fft_features,
            "spectrum_mean",
            "Spectrum mean",
        )

        show_fft_feature(
            fft_features,
            "spectrum_standard_deviation",
            "Spectrum standard deviation",
        )

else:
    st.warning(
        "FFT spectrum is unavailable."
    )


# ---------------------------------------------------------
# METADATA
# ---------------------------------------------------------

st.divider()
st.header("Metadata Analysis")

metadata_column_1, metadata_column_2 = (
    st.columns(2)
)

dimensions = metadata_result.get(
    "dimensions",
    f"{image.width} × {image.height}",
)

exif_count = metadata_result.get(
    "exif_field_count",
    metadata_result.get(
        "exif_fields",
        0,
    ),
)

if isinstance(exif_count, dict):
    exif_count = len(exif_count)


camera_value = metadata_result.get(
    "camera",
    metadata_result.get(
        "camera_model",
        "Not detected",
    ),
)

software_value = metadata_result.get(
    "software",
    "Not detected",
)

date_value = metadata_result.get(
    "date_time",
    metadata_result.get(
        "datetime",
        "Not detected",
    ),
)

gps_value = metadata_result.get(
    "gps",
    metadata_result.get(
        "gps_info",
        "Not detected",
    ),
)


with metadata_column_1:
    st.write(
        f"**Filename:** "
        f"{metadata_result.get('filename', uploaded_file.name)}"
    )

    st.write(
        f"**Format:** "
        f"{metadata_result.get('format', 'Unknown')}"
    )

    st.write(
        f"**Dimensions:** {dimensions}"
    )

    st.write(
        f"**EXIF fields:** {exif_count}"
    )


with metadata_column_2:
    st.write(
        f"**Camera:** "
        f"{camera_value or 'Not detected'}"
    )

    st.write(
        f"**Software:** "
        f"{software_value or 'Not detected'}"
    )

    st.write(
        f"**Creation date:** "
        f"{date_value or 'Not detected'}"
    )

    st.write(
        f"**GPS:** "
        f"{gps_value or 'Not detected'}"
    )


st.write("**SHA-256:**")

st.code(
    metadata_result.get(
        "sha256",
        file_hash,
    )
)


metadata_observations = metadata_result.get(
    "observations",
    [],
)

if metadata_observations:
    st.subheader("Metadata Observations")

    for item in metadata_observations:
        st.write(f"- {item}")


# ---------------------------------------------------------
# FORENSIC OBSERVATIONS
# ---------------------------------------------------------

st.divider()
st.header("Supporting Forensic Observations")

if observations:
    for item in observations:
        st.write(f"- {item}")
else:
    st.write(
        "No supporting observations were produced."
    )


# ---------------------------------------------------------
# PDF DOWNLOAD
# ---------------------------------------------------------

st.divider()
st.header("Download Forensic Report")

try:
    if st.session_state.get("pdf_report") is None:
        with st.spinner(
            "Preparing the forensic PDF report..."
        ):
            st.session_state["pdf_report"] = (
                generate_forensic_report(
                    image=image,
                    filename=uploaded_file.name,
                    file_size_bytes=len(file_bytes),
                    analysis_result=result,
                )
            )

    pdf_report = st.session_state.get(
        "pdf_report"
    )

    report_name = (
        uploaded_file.name.rsplit(".", 1)[0]
        + "_forensic_report.pdf"
    )

    st.download_button(
        label="Download PDF Forensic Report",
        data=pdf_report,
        file_name=report_name,
        mime="application/pdf",
        type="primary",
        width="stretch",
    )

    st.caption(
        "The PDF includes the final result, uploaded image, "
        "CNN and FFT scores, Grad-CAM overlay, strongest CNN-influence "
        "view, frequency features, metadata, and forensic observations."
    )

except Exception as report_error:
    st.error(
        f"PDF report generation failed: {report_error}"
    )


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.info(
    "This application provides a probabilistic forensic "
    "screening result. It cannot prove with complete certainty "
    "whether an image is real or AI-generated. The findings "
    "should be combined with human review and other evidence."
)
