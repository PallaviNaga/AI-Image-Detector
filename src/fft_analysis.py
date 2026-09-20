import math

import numpy as np
from PIL import Image


# ---------------------------------------------------------
# FFT SETTINGS
# ---------------------------------------------------------

FFT_IMAGE_SIZE = 256
NUMBER_OF_RADIAL_RINGS = 10

FEATURE_NAMES = [
    "low_frequency_energy",
    "mid_frequency_energy",
    "high_frequency_energy",
    "high_to_low_ratio",
    "spectral_entropy",
    "spectrum_mean",
    "spectrum_standard_deviation",
    "horizontal_vertical_imbalance",
]

FEATURE_NAMES.extend(
    [
        f"radial_ring_{ring_number}_energy"
        for ring_number in range(1, NUMBER_OF_RADIAL_RINGS + 1)
    ]
)


# ---------------------------------------------------------
# PREPARE IMAGE FOR FFT
# ---------------------------------------------------------

def prepare_grayscale_image(image):
    """
    Convert a PIL image to a fixed-size grayscale NumPy array.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("The supplied image must be a PIL image.")

    grayscale_image = image.convert("L")

    grayscale_image = grayscale_image.resize(
        (FFT_IMAGE_SIZE, FFT_IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    grayscale_array = np.asarray(
        grayscale_image,
        dtype=np.float32,
    )

    # Scale pixel values from 0-255 to 0-1.
    grayscale_array = grayscale_array / 255.0

    return grayscale_array


# ---------------------------------------------------------
# APPLY WINDOW FUNCTION
# ---------------------------------------------------------

def apply_window_function(grayscale_array):
    """
    Apply a two-dimensional Hann window.

    Windowing reduces strong frequency artifacts produced by
    the outer edges of the image.
    """

    height, width = grayscale_array.shape

    vertical_window = np.hanning(height)
    horizontal_window = np.hanning(width)

    window_2d = np.outer(
        vertical_window,
        horizontal_window,
    ).astype(np.float32)

    return grayscale_array * window_2d


# ---------------------------------------------------------
# CALCULATE FFT SPECTRUM
# ---------------------------------------------------------

def calculate_fft_spectrum(grayscale_array):
    """
    Calculate the shifted two-dimensional FFT and magnitude spectrum.
    """

    windowed_image = apply_window_function(
        grayscale_array
    )

    fft_result = np.fft.fft2(windowed_image)

    shifted_fft = np.fft.fftshift(fft_result)

    magnitude = np.abs(shifted_fft)

    log_magnitude = np.log1p(magnitude)

    return magnitude, log_magnitude


# ---------------------------------------------------------
# CREATE FREQUENCY RADIUS MAP
# ---------------------------------------------------------

def create_radius_map(height, width):
    """
    Calculate the distance of each frequency value from the centre.
    """

    centre_y = height // 2
    centre_x = width // 2

    y_coordinates, x_coordinates = np.ogrid[
        :height,
        :width,
    ]

    radius_map = np.sqrt(
        (x_coordinates - centre_x) ** 2
        + (y_coordinates - centre_y) ** 2
    )

    maximum_radius = np.max(radius_map)

    normalized_radius = radius_map / maximum_radius

    return normalized_radius


# ---------------------------------------------------------
# CALCULATE ENERGY IN A FREQUENCY REGION
# ---------------------------------------------------------

def calculate_region_energy(power_spectrum, mask):
    """
    Calculate the fraction of total energy inside a selected region.
    """

    total_energy = float(np.sum(power_spectrum))

    if total_energy <= 0:
        return 0.0

    region_energy = float(
        np.sum(power_spectrum[mask])
    )

    return region_energy / total_energy


# ---------------------------------------------------------
# CALCULATE SPECTRAL ENTROPY
# ---------------------------------------------------------

def calculate_spectral_entropy(power_spectrum):
    """
    Calculate normalized spectral entropy between 0 and 1.
    """

    flattened_power = power_spectrum.flatten()

    total_power = float(np.sum(flattened_power))

    if total_power <= 0:
        return 0.0

    probability_distribution = (
        flattened_power / total_power
    )

    probability_distribution = probability_distribution[
        probability_distribution > 0
    ]

    entropy = -np.sum(
        probability_distribution
        * np.log2(probability_distribution)
    )

    maximum_entropy = math.log2(
        len(flattened_power)
    )

    if maximum_entropy == 0:
        return 0.0

    return float(entropy / maximum_entropy)


# ---------------------------------------------------------
# EXTRACT RADIAL FREQUENCY PROFILE
# ---------------------------------------------------------

def extract_radial_profile(
    power_spectrum,
    normalized_radius,
):
    """
    Divide the spectrum into concentric rings and calculate
    the energy contained in each ring.
    """

    radial_features = {}

    ring_edges = np.linspace(
        0.0,
        1.0,
        NUMBER_OF_RADIAL_RINGS + 1,
    )

    for ring_index in range(NUMBER_OF_RADIAL_RINGS):
        inner_radius = ring_edges[ring_index]
        outer_radius = ring_edges[ring_index + 1]

        if ring_index == NUMBER_OF_RADIAL_RINGS - 1:
            ring_mask = (
                (normalized_radius >= inner_radius)
                & (normalized_radius <= outer_radius)
            )
        else:
            ring_mask = (
                (normalized_radius >= inner_radius)
                & (normalized_radius < outer_radius)
            )

        ring_energy = calculate_region_energy(
            power_spectrum,
            ring_mask,
        )

        feature_name = (
            f"radial_ring_{ring_index + 1}_energy"
        )

        radial_features[feature_name] = ring_energy

    return radial_features


# ---------------------------------------------------------
# CALCULATE DIRECTIONAL IMBALANCE
# ---------------------------------------------------------

def calculate_directional_imbalance(power_spectrum):
    """
    Compare the energy around the horizontal and vertical frequency axes.
    """

    height, width = power_spectrum.shape

    centre_y = height // 2
    centre_x = width // 2

    band_width = max(
        2,
        FFT_IMAGE_SIZE // 64,
    )

    horizontal_band = power_spectrum[
        centre_y - band_width:
        centre_y + band_width + 1,
        :
    ]

    vertical_band = power_spectrum[
        :,
        centre_x - band_width:
        centre_x + band_width + 1,
    ]

    horizontal_energy = float(
        np.mean(horizontal_band)
    )

    vertical_energy = float(
        np.mean(vertical_band)
    )

    denominator = (
        horizontal_energy
        + vertical_energy
        + 1e-12
    )

    imbalance = abs(
        horizontal_energy - vertical_energy
    ) / denominator

    return float(imbalance)


# ---------------------------------------------------------
# CONVERT SPECTRUM INTO DISPLAYABLE IMAGE
# ---------------------------------------------------------

def create_spectrum_image(log_magnitude):
    """
    Normalize the logarithmic magnitude spectrum and convert
    it into a grayscale PIL image for Streamlit.
    """

    minimum_value = float(
        np.min(log_magnitude)
    )

    maximum_value = float(
        np.max(log_magnitude)
    )

    normalized_spectrum = (
        log_magnitude - minimum_value
    ) / (
        maximum_value
        - minimum_value
        + 1e-12
    )

    spectrum_uint8 = (
        normalized_spectrum * 255
    ).astype(np.uint8)

    spectrum_image = Image.fromarray(
        spectrum_uint8,
        mode="L",
    )

    return spectrum_image


# ---------------------------------------------------------
# COMPLETE FFT ANALYSIS
# ---------------------------------------------------------

def analyze_fft(image):
    """
    Perform complete FFT analysis on a PIL image.

    Returns:
        spectrum_image:
            A displayable FFT magnitude-spectrum image.

        features:
            Dictionary containing all extracted FFT features.

        feature_vector:
            NumPy array containing features in a fixed order.
            This will later be used by the FFT classifier.
    """

    grayscale_array = prepare_grayscale_image(
        image
    )

    magnitude, log_magnitude = calculate_fft_spectrum(
        grayscale_array
    )

    power_spectrum = magnitude ** 2

    height, width = power_spectrum.shape

    normalized_radius = create_radius_map(
        height,
        width,
    )

    # Frequency regions:
    # Centre = low frequency
    # Middle = medium frequency
    # Outside = high frequency

    low_frequency_mask = normalized_radius <= 0.15

    mid_frequency_mask = (
        (normalized_radius > 0.15)
        & (normalized_radius <= 0.45)
    )

    high_frequency_mask = (
        normalized_radius > 0.45
    )

    low_frequency_energy = calculate_region_energy(
        power_spectrum,
        low_frequency_mask,
    )

    mid_frequency_energy = calculate_region_energy(
        power_spectrum,
        mid_frequency_mask,
    )

    high_frequency_energy = calculate_region_energy(
        power_spectrum,
        high_frequency_mask,
    )

    high_to_low_ratio = (
        high_frequency_energy
        / (low_frequency_energy + 1e-12)
    )

    spectral_entropy = calculate_spectral_entropy(
        power_spectrum
    )

    spectrum_mean = float(
        np.mean(log_magnitude)
    )

    spectrum_standard_deviation = float(
        np.std(log_magnitude)
    )

    directional_imbalance = (
        calculate_directional_imbalance(
            power_spectrum
        )
    )

    features = {
        "low_frequency_energy": low_frequency_energy,
        "mid_frequency_energy": mid_frequency_energy,
        "high_frequency_energy": high_frequency_energy,
        "high_to_low_ratio": high_to_low_ratio,
        "spectral_entropy": spectral_entropy,
        "spectrum_mean": spectrum_mean,
        "spectrum_standard_deviation": (
            spectrum_standard_deviation
        ),
        "horizontal_vertical_imbalance": (
            directional_imbalance
        ),
    }

    radial_features = extract_radial_profile(
        power_spectrum,
        normalized_radius,
    )

    features.update(radial_features)

    feature_vector = np.array(
        [
            features[feature_name]
            for feature_name in FEATURE_NAMES
        ],
        dtype=np.float32,
    )

    spectrum_image = create_spectrum_image(
        log_magnitude
    )

    return {
        "spectrum_image": spectrum_image,
        "features": features,
        "feature_vector": feature_vector,
    }