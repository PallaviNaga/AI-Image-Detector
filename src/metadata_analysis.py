from io import BytesIO
import hashlib

from PIL import ExifTags, Image, UnidentifiedImageError


# ---------------------------------------------------------
# SOFTWARE KEYWORDS
# ---------------------------------------------------------

AI_SOFTWARE_KEYWORDS = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dalle",
    "openai",
    "comfyui",
    "automatic1111",
    "invokeai",
    "firefly",
    "generative fill",
]

EDITING_SOFTWARE_KEYWORDS = [
    "photoshop",
    "lightroom",
    "gimp",
    "snapseed",
    "canva",
    "paint.net",
    "affinity photo",
]


# ---------------------------------------------------------
# CONVERT EXIF VALUES TO DISPLAYABLE TEXT
# ---------------------------------------------------------

def convert_exif_value(value):
    """
    Convert EXIF values into safe readable text.
    """

    if isinstance(value, bytes):
        try:
            return value.decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            return repr(value)

    if isinstance(value, (list, tuple)):
        return [
            convert_exif_value(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): convert_exif_value(item)
            for key, item in value.items()
        }

    try:
        return str(value)
    except Exception:
        return repr(value)


# ---------------------------------------------------------
# EXTRACT EXIF METADATA
# ---------------------------------------------------------

def extract_exif_data(image):
    """
    Extract available EXIF tags from a PIL image.
    """

    exif_information = {}

    try:
        raw_exif = image.getexif()
    except Exception:
        raw_exif = None

    if not raw_exif:
        return exif_information

    for tag_id, value in raw_exif.items():
        tag_name = ExifTags.TAGS.get(
            tag_id,
            str(tag_id),
        )

        # GPS information may be stored as an EXIF subdirectory.
        if tag_name == "GPSInfo":
            try:
                gps_directory = raw_exif.get_ifd(
                    tag_id
                )

                gps_information = {}

                for gps_tag_id, gps_value in (
                    gps_directory.items()
                ):
                    gps_tag_name = (
                        ExifTags.GPSTAGS.get(
                            gps_tag_id,
                            str(gps_tag_id),
                        )
                    )

                    gps_information[gps_tag_name] = (
                        convert_exif_value(
                            gps_value
                        )
                    )

                exif_information[
                    "GPSInfo"
                ] = gps_information

            except Exception:
                exif_information[
                    "GPSInfo"
                ] = convert_exif_value(value)

        else:
            exif_information[
                tag_name
            ] = convert_exif_value(value)

    return exif_information


# ---------------------------------------------------------
# FIND IMPORTANT METADATA FIELD
# ---------------------------------------------------------

def find_metadata_value(
    exif_information,
    possible_names,
):
    """
    Find the first available EXIF field from a list.
    """

    for field_name in possible_names:
        value = exif_information.get(
            field_name
        )

        if value not in {
            None,
            "",
            "None",
        }:
            return value

    return None


# ---------------------------------------------------------
# INTERPRET SOFTWARE FIELD
# ---------------------------------------------------------

def analyse_software_field(software_value):
    """
    Check whether the software field mentions an AI-generation
    or image-editing application.
    """

    if not software_value:
        return {
            "ai_software_detected": False,
            "editing_software_detected": False,
            "matched_ai_keyword": None,
            "matched_editing_keyword": None,
        }

    software_text = str(
        software_value
    ).lower()

    matched_ai_keyword = next(
        (
            keyword
            for keyword in AI_SOFTWARE_KEYWORDS
            if keyword in software_text
        ),
        None,
    )

    matched_editing_keyword = next(
        (
            keyword
            for keyword in EDITING_SOFTWARE_KEYWORDS
            if keyword in software_text
        ),
        None,
    )

    return {
        "ai_software_detected": (
            matched_ai_keyword is not None
        ),
        "editing_software_detected": (
            matched_editing_keyword is not None
        ),
        "matched_ai_keyword": matched_ai_keyword,
        "matched_editing_keyword": (
            matched_editing_keyword
        ),
    }


# ---------------------------------------------------------
# COMPLETE METADATA ANALYSIS
# ---------------------------------------------------------

def analyze_metadata(
    file_bytes,
    filename=None,
    mime_type=None,
):
    """
    Analyse file properties, SHA-256 hash and EXIF metadata.

    file_bytes must contain the original uploaded image bytes.
    """

    if not isinstance(
        file_bytes,
        (bytes, bytearray),
    ):
        raise TypeError(
            "file_bytes must be bytes or bytearray."
        )

    if len(file_bytes) == 0:
        raise ValueError(
            "The uploaded image file is empty."
        )

    sha256_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    try:
        with Image.open(
            BytesIO(file_bytes)
        ) as image:
            image.verify()

        # Reopen after verify because verify closes the image data.
        with Image.open(
            BytesIO(file_bytes)
        ) as image:
            image_format = image.format
            width, height = image.size
            colour_mode = image.mode

            exif_information = (
                extract_exif_data(image)
            )

            image_information = {
                str(key): convert_exif_value(value)
                for key, value in image.info.items()
                if key != "exif"
            }

    except UnidentifiedImageError as error:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from error

    camera_make = find_metadata_value(
        exif_information,
        ["Make"],
    )

    camera_model = find_metadata_value(
        exif_information,
        ["Model"],
    )

    software = find_metadata_value(
        exif_information,
        ["Software", "ProcessingSoftware"],
    )

    creation_date = find_metadata_value(
        exif_information,
        [
            "DateTimeOriginal",
            "DateTimeDigitized",
            "DateTime",
        ],
    )

    lens_model = find_metadata_value(
        exif_information,
        ["LensModel", "LensMake"],
    )

    artist = find_metadata_value(
        exif_information,
        ["Artist", "Copyright"],
    )

    gps_available = (
        "GPSInfo" in exif_information
        and bool(exif_information["GPSInfo"])
    )

    software_analysis = analyse_software_field(
        software
    )

    observations = []

    if len(exif_information) == 0:
        observations.append(
            "No EXIF metadata was detected."
        )
    else:
        observations.append(
            f"{len(exif_information)} EXIF fields were detected."
        )

    if camera_make or camera_model:
        camera_description = " ".join(
            str(value)
            for value in [
                camera_make,
                camera_model,
            ]
            if value
        )

        observations.append(
            "Camera information was detected: "
            f"{camera_description}."
        )
    else:
        observations.append(
            "No camera make or model was detected."
        )

    if software:
        observations.append(
            f"Software metadata was detected: {software}."
        )
    else:
        observations.append(
            "No software metadata was detected."
        )

    if software_analysis[
        "ai_software_detected"
    ]:
        observations.append(
            "The software field contains a known "
            "AI-generation tool keyword."
        )

    if software_analysis[
        "editing_software_detected"
    ]:
        observations.append(
            "The software field contains an "
            "image-editing application keyword."
        )

    if gps_available:
        observations.append(
            "GPS metadata is present."
        )
    else:
        observations.append(
            "No GPS metadata was detected."
        )

    observations.append(
        "Metadata can be removed, changed or fabricated. "
        "Missing metadata does not prove AI generation."
    )

    return {
        "filename": filename,
        "mime_type": mime_type,
        "file_size_bytes": len(file_bytes),
        "file_size_kb": len(file_bytes) / 1024,
        "sha256": sha256_hash,
        "format": image_format,
        "width": width,
        "height": height,
        "colour_mode": colour_mode,
        "exif_field_count": len(
            exif_information
        ),
        "camera_make": camera_make,
        "camera_model": camera_model,
        "lens_model": lens_model,
        "creation_date": creation_date,
        "software": software,
        "artist_or_copyright": artist,
        "gps_available": gps_available,
        "ai_software_detected": (
            software_analysis[
                "ai_software_detected"
            ]
        ),
        "editing_software_detected": (
            software_analysis[
                "editing_software_detected"
            ]
        ),
        "exif": exif_information,
        "image_info": image_information,
        "observations": observations,
    }