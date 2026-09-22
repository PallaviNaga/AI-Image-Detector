from datetime import datetime
from io import BytesIO

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportLabImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _safe_number(value, default=0.0):
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentage(value):
    """
    Convert a probability to percentage text.

    Accepts:
        0.8941 -> 89.41%
        89.41  -> 89.41%
    """
    number = _safe_number(value)

    if 0.0 <= number <= 1.0:
        number *= 100.0

    return f"{number:.2f}%"


def _first_value(dictionary, keys, default=None):
    """Return the first existing value from a list of dictionary keys."""
    if not isinstance(dictionary, dict):
        return default

    for key in keys:
        if key in dictionary and dictionary[key] is not None:
            return dictionary[key]

    return default


def _dictionary(value):
    """Return value if it is a dictionary, otherwise an empty dictionary."""
    return value if isinstance(value, dict) else {}


def _prepare_image(image, maximum_size=(1000, 750)):
    """Convert a PIL image into bytes suitable for ReportLab."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    prepared = image.copy().convert("RGB")
    prepared.thumbnail(maximum_size)

    buffer = BytesIO()
    prepared.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    return buffer, prepared.width, prepared.height


def _add_page_number(canvas, document):
    """Add footer and page number to every PDF page."""
    canvas.saveState()

    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.line(
        document.leftMargin,
        35,
        A4[0] - document.rightMargin,
        35,
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))

    canvas.drawString(
        document.leftMargin,
        22,
        "AI-Generated Image Detector — Forensic Screening Report",
    )

    canvas.drawRightString(
        A4[0] - document.rightMargin,
        22,
        f"Page {document.page}",
    )

    canvas.restoreState()


def generate_forensic_report(
    image,
    filename,
    file_size_bytes,
    analysis_result,
):
    """
    Generate a forensic PDF report and return it as bytes.

    Parameters
    ----------
    image:
        PIL image uploaded by the user.

    filename:
        Original image filename.

    file_size_bytes:
        Uploaded file size in bytes.

    analysis_result:
        Dictionary returned by the detector.

    Returns
    -------
    bytes
        Complete PDF file contents.
    """
    result = _dictionary(analysis_result)

    # app.py stores CNN and FFT results inside analysis_result["models"].
    # Keep top-level fallbacks for compatibility with older saved results.
    models = _dictionary(result.get("models"))
    cnn = _dictionary(models.get("cnn") or result.get("cnn"))
    fft = _dictionary(models.get("fft") or result.get("fft"))
    metadata = _dictionary(result.get("metadata"))
    fusion = _dictionary(result.get("fusion"))
    gradcam = _dictionary(result.get("gradcam"))

    gradcam_overlay = _first_value(
        gradcam,
        ["overlay_image", "overlay", "gradcam_overlay"],
        None,
    )

    strong_influence_image = _first_value(
        gradcam,
        ["strong_influence_image", "focused_overlay", "highlighted_image"],
        None,
    )

    strong_influence_percentile = _first_value(
        gradcam,
        ["strong_influence_percentile"],
        80,
    )

    gradcam_label = _first_value(
        gradcam,
        ["explained_label", "prediction", "label"],
        cnn_label if "cnn_label" in locals() else "Not available",
    )

    gradcam_ai_probability = _first_value(
        gradcam,
        ["ai_probability", "ai_prob"],
        None,
    )

    # Support applications where the fusion fields are at the top level.
    final_ai_probability = _first_value(
        fusion,
        [
            "final_ai_probability",
            "ai_probability",
            "combined_ai_probability",
            "score",
        ],
        None,
    )

    if final_ai_probability is None:
        final_ai_probability = _first_value(
            result,
            [
                "final_ai_probability",
                "ai_probability",
                "combined_ai_probability",
                "final_score",
            ],
            0.0,
        )

    final_real_probability = _first_value(
        fusion,
        ["final_real_probability", "real_probability"],
        None,
    )

    if final_real_probability is None:
        final_real_probability = _first_value(
            result,
            ["final_real_probability", "real_probability"],
            None,
        )

    final_ai_number = _safe_number(final_ai_probability)

    if final_ai_number > 1.0:
        final_ai_decimal = final_ai_number / 100.0
    else:
        final_ai_decimal = final_ai_number

    if final_real_probability is None:
        final_real_probability = max(0.0, 1.0 - final_ai_decimal)

    classification = _first_value(
        fusion,
        ["label", "classification", "prediction"],
        None,
    )

    if classification is None:
        classification = _first_value(
            result,
            ["label", "classification", "prediction"],
            "Not available",
        )

    agreement = _first_value(
        fusion,
        ["agreement_status", "agreement"],
        None,
    )

    if agreement is None:
        agreement = _first_value(
            result,
            ["agreement_status", "agreement"],
            "Not available",
        )

    cnn_ai_probability = _first_value(
        cnn,
        ["ai_probability", "ai_prob", "probability"],
        0.0,
    )

    cnn_real_probability = _first_value(
        cnn,
        ["real_probability", "real_prob"],
        None,
    )

    cnn_ai_decimal = _safe_number(cnn_ai_probability)
    if cnn_ai_decimal > 1.0:
        cnn_ai_decimal /= 100.0

    if cnn_real_probability is None:
        cnn_real_probability = max(0.0, 1.0 - cnn_ai_decimal)

    cnn_label = _first_value(
        cnn,
        ["label", "prediction", "class_name"],
        "Not available",
    )

    cnn_confidence = _first_value(
        cnn,
        ["confidence"],
        max(cnn_ai_decimal, 1.0 - cnn_ai_decimal),
    )

    cnn_model = _first_value(
        cnn,
        ["model", "model_name"],
        "ResNet18",
    )

    # Use the CNN label as a fallback once it has been resolved.
    if gradcam_label == "Not available":
        gradcam_label = cnn_label

    if gradcam_ai_probability is None:
        gradcam_ai_probability = cnn_ai_probability

    fft_ai_probability = _first_value(
        fft,
        ["ai_probability", "ai_prob", "probability"],
        0.0,
    )

    fft_real_probability = _first_value(
        fft,
        ["real_probability", "real_prob"],
        None,
    )

    fft_ai_decimal = _safe_number(fft_ai_probability)
    if fft_ai_decimal > 1.0:
        fft_ai_decimal /= 100.0

    if fft_real_probability is None:
        fft_real_probability = max(0.0, 1.0 - fft_ai_decimal)

    fft_label = _first_value(
        fft,
        ["label", "prediction", "class_name"],
        "Not available",
    )

    fft_confidence = _first_value(
        fft,
        ["confidence"],
        max(fft_ai_decimal, 1.0 - fft_ai_decimal),
    )

    fft_model = _first_value(
        fft,
        ["model", "model_name"],
        "Random Forest",
    )

    features = _dictionary(fft.get("features"))
    if not features:
        features = fft

    image_format = _first_value(
        metadata,
        ["format", "image_format"],
        getattr(image, "format", None) or "Unknown",
    )

    width = _first_value(metadata, ["width"], getattr(image, "width", "Unknown"))
    height = _first_value(
        metadata,
        ["height"],
        getattr(image, "height", "Unknown"),
    )

    dimensions = _first_value(
        metadata,
        ["dimensions"],
        f"{width} × {height}",
    )

    exif_count = _first_value(
        metadata,
        ["exif_fields", "exif_count"],
        0,
    )

    camera = _first_value(
        metadata,
        ["camera", "camera_model", "make_model"],
        "Not detected",
    )

    software = _first_value(
        metadata,
        ["software"],
        "Not detected",
    )

    creation_date = _first_value(
        metadata,
        ["creation_date", "datetime", "date_time"],
        "Not detected",
    )

    gps = _first_value(
        metadata,
        ["gps", "gps_information"],
        "Not detected",
    )

    sha256_hash = _first_value(
        metadata,
        ["sha256", "sha_256", "hash"],
        "Not available",
    )

    observations = result.get("observations", [])
    if not isinstance(observations, list):
        observations = []

    metadata_observations = metadata.get("observations", [])
    if not isinstance(metadata_observations, list):
        metadata_observations = []

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=50,
        title="AI Image Forensic Report",
        author="AI-Generated Image Detector",
        subject="Probabilistic AI-generated image screening report",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceAfter=14,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#1F4E78"),
            spaceBefore=12,
            spaceAfter=9,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#4B5563"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="Disclaimer",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#374151"),
            backColor=colors.HexColor("#EAF2F8"),
            borderColor=colors.HexColor("#7FB3D5"),
            borderWidth=1,
            borderPadding=9,
            spaceBefore=10,
            spaceAfter=10,
        )
    )

    story = []

    story.append(
        Paragraph(
            "AI-Generated Image Detector",
            styles["ReportTitle"],
        )
    )

    story.append(
        Paragraph(
            "Forensic Screening Report",
            styles["Heading1"],
        )
    )

    story.append(
        Paragraph(
            f"Report generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}",
            styles["SmallText"],
        )
    )

    story.append(Spacer(1, 12))

    file_information = [
        ["Filename", str(filename)],
        ["File size", f"{file_size_bytes / 1024:.2f} KB"],
        ["Format", str(image_format)],
        ["Dimensions", str(dimensions)],
        ["SHA-256", str(sha256_hash)],
    ]

    file_table = Table(
        file_information,
        colWidths=[1.25 * inch, 5.55 * inch],
    )

    file_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(file_table)
    story.append(Spacer(1, 14))

    try:
        image_buffer, image_width, image_height = _prepare_image(image)

        available_width = 6.6 * inch
        available_height = 3.8 * inch

        scale = min(
            available_width / image_width,
            available_height / image_height,
        )

        report_image = ReportLabImage(
            image_buffer,
            width=image_width * scale,
            height=image_height * scale,
        )

        story.append(
            KeepTogether(
                [
                    Paragraph("Analysed Image", styles["SectionHeading"]),
                    report_image,
                ]
            )
        )
    except Exception:
        story.append(
            Paragraph(
                "The uploaded image could not be embedded in the PDF.",
                styles["BodyText"],
            )
        )

    story.append(Spacer(1, 12))
    story.append(Paragraph("Final Forensic Result", styles["SectionHeading"]))

    final_table = Table(
        [
            ["Classification", str(classification)],
            ["Final AI probability", _percentage(final_ai_probability)],
            ["Final real probability", _percentage(final_real_probability)],
            ["Model agreement", str(agreement)],
            ["Fusion weights", "CNN 70% and FFT 30%"],
        ],
        colWidths=[2.1 * inch, 4.7 * inch],
    )

    classification_text = str(classification).lower()

    if "ai-generated" in classification_text and "inconclusive" not in classification_text:
        result_colour = colors.HexColor("#FEE2E2")
    elif "real" in classification_text:
        result_colour = colors.HexColor("#DCFCE7")
    else:
        result_colour = colors.HexColor("#FEF3C7")

    final_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), result_colour),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#E5E7EB")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(final_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Individual Model Results", styles["SectionHeading"]))

    model_table = Table(
        [
            ["Measurement", "CNN Analysis", "FFT Analysis"],
            ["Prediction", str(cnn_label), str(fft_label)],
            [
                "AI probability",
                _percentage(cnn_ai_probability),
                _percentage(fft_ai_probability),
            ],
            [
                "Real probability",
                _percentage(cnn_real_probability),
                _percentage(fft_real_probability),
            ],
            [
                "Confidence",
                _percentage(cnn_confidence),
                _percentage(fft_confidence),
            ],
            ["Model", str(cnn_model), str(fft_model)],
        ],
        colWidths=[2.0 * inch, 2.4 * inch, 2.4 * inch],
    )

    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#E5E7EB")),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(model_table)

    # ---------------------------------------------------------
    # CNN VISUAL EXPLANATION — GRAD-CAM
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(
        Paragraph(
            "CNN Visual Explanation — Grad-CAM",
            styles["SectionHeading"],
        )
    )

    if gradcam_overlay is not None or strong_influence_image is not None:
        try:
            visual_cells = []

            for title, visual in (
                ("Complete Grad-CAM overlay", gradcam_overlay),
                ("Strongest CNN influence", strong_influence_image),
            ):
                if visual is None:
                    continue

                visual_buffer, visual_width, visual_height = _prepare_image(
                    visual,
                    maximum_size=(900, 700),
                )
                maximum_width = 3.25 * inch
                maximum_height = 3.7 * inch
                visual_scale = min(
                    maximum_width / visual_width,
                    maximum_height / visual_height,
                )
                report_visual = ReportLabImage(
                    visual_buffer,
                    width=visual_width * visual_scale,
                    height=visual_height * visual_scale,
                )
                visual_cells.append(
                    [
                        Paragraph(f"<b>{title}</b>", styles["BodyText"]),
                        report_visual,
                    ]
                )

            # Each item becomes a column containing its title and image.
            visual_table = Table(
                [
                    [cell[0] for cell in visual_cells],
                    [cell[1] for cell in visual_cells],
                ],
                colWidths=[6.6 * inch / len(visual_cells)] * len(visual_cells),
            )
            visual_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(visual_table)
            story.append(Spacer(1, 10))

            gradcam_rows = [
                ["CNN class explained", str(gradcam_label)],
            ]

            if gradcam_ai_probability is not None:
                gradcam_rows.append(
                    [
                        "CNN AI probability",
                        _percentage(gradcam_ai_probability),
                    ]
                )

            gradcam_table = Table(
                gradcam_rows,
                colWidths=[2.1 * inch, 4.7 * inch],
            )

            gradcam_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (0, -1),
                            colors.HexColor("#E5E7EB"),
                        ),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.HexColor("#CBD5E1"),
                        ),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )

            story.append(gradcam_table)
            story.append(Spacer(1, 10))
            story.append(
                Paragraph(
                    "<b>Heatmap interpretation:</b> Red and yellow regions "
                    "had the strongest influence on the CNN prediction, green "
                    "regions had moderate influence, and blue regions had "
                    "weaker influence.",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 7))
            story.append(
                Paragraph(
                    "<b>Focused highlight:</b> The red outlined view emphasizes "
                    f"approximately the top {100 - _safe_number(strong_influence_percentile):.0f}% "
                    "of positive Grad-CAM activation, making the CNN's strongest "
                    "influence area easier to inspect.",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 7))
            story.append(
                Paragraph(
                    "<b>Important:</b> The highlighted regions show CNN model "
                    "attention. They do not independently prove that a specific "
                    "part of the image was generated or edited by AI.",
                    styles["Disclaimer"],
                )
            )

        except Exception:
            story.append(
                Paragraph(
                    "The Grad-CAM overlay could not be embedded in the PDF.",
                    styles["BodyText"],
                )
            )
    else:
        story.append(
            Paragraph(
                "Grad-CAM overlay was unavailable for this analysis.",
                styles["BodyText"],
            )
        )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Frequency-Domain Analysis",
            styles["SectionHeading"],
        )
    )

    feature_rows = [
        [
            "Low-frequency energy",
            f"{_safe_number(_first_value(features, ['low_frequency_energy'], 0.0)):.6f}",
        ],
        [
            "Mid-frequency energy",
            f"{_safe_number(_first_value(features, ['mid_frequency_energy'], 0.0)):.6f}",
        ],
        [
            "High-frequency energy",
            f"{_safe_number(_first_value(features, ['high_frequency_energy'], 0.0)):.6f}",
        ],
        [
            "Spectral entropy",
            f"{_safe_number(_first_value(features, ['spectral_entropy'], 0.0)):.6f}",
        ],
        [
            "Spectrum mean",
            f"{_safe_number(_first_value(features, ['spectrum_mean'], 0.0)):.6f}",
        ],
        [
            "Spectrum standard deviation",
            f"{_safe_number(_first_value(features, ['spectrum_standard_deviation', 'spectrum_std'], 0.0)):.6f}",
        ],
    ]

    feature_table = Table(
        feature_rows,
        colWidths=[3.4 * inch, 3.4 * inch],
    )

    feature_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E5E7EB")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(feature_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Metadata Analysis", styles["SectionHeading"]))

    metadata_table = Table(
        [
            ["Format", str(image_format)],
            ["Dimensions", str(dimensions)],
            ["EXIF fields", str(exif_count)],
            ["Camera", str(camera)],
            ["Software", str(software)],
            ["Creation date", str(creation_date)],
            ["GPS", str(gps)],
            ["SHA-256", str(sha256_hash)],
        ],
        colWidths=[1.6 * inch, 5.2 * inch],
    )

    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E5E7EB")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(metadata_table)

    all_observations = metadata_observations + observations

    if all_observations:
        story.append(Spacer(1, 12))
        story.append(
            Paragraph(
                "Supporting Forensic Observations",
                styles["SectionHeading"],
            )
        )

        for observation in all_observations:
            story.append(
                Paragraph(
                    f"• {str(observation)}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 5))

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "<b>Important limitation:</b> This report provides a probabilistic "
            "forensic screening result. It cannot prove with complete certainty "
            "whether an image is real or AI-generated. Metadata can be removed, "
            "changed or fabricated. The result should be combined with human "
            "review, provenance information and other forensic evidence.",
            styles["Disclaimer"],
        )
    )

    document.build(
        story,
        onFirstPage=_add_page_number,
        onLaterPages=_add_page_number,
    )

    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()

    return pdf_bytes
