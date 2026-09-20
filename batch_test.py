import csv
from pathlib import Path

from PIL import Image

from src.inference import analyze_image
from src.score_fusion import combine_scores


PROJECT_DIR = Path(__file__).resolve().parent
TEST_DIR = PROJECT_DIR / "test_samples"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "batch_test_results.csv"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_probability(value):
    """Convert probability into the range 0.0 to 1.0."""
    value = float(value)

    if value > 1:
        value = value / 100.0

    return max(0.0, min(1.0, value))


def simple_label(label):
    """Convert different result labels into Real, AI-Generated or Inconclusive."""
    text = str(label).strip().lower()

    if "inconclusive" in text or "uncertain" in text:
        return "Inconclusive"

    if "ai" in text:
        return "AI-Generated"

    if "real" in text:
        return "Real"

    return str(label)


def get_images(folder):
    """Return supported image files from a folder."""
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def test_image(image_path, actual_class):
    """Analyse one image and return its results."""
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
        result = analyze_image(image)

    cnn_result = result.get("cnn", {})
    fft_result = result.get("fft", {})

    cnn_ai_probability = normalize_probability(
        cnn_result.get("ai_probability", 0.0)
    )

    fft_ai_probability = normalize_probability(
        fft_result.get("ai_probability", 0.0)
    )

    fusion_result = combine_scores(
        cnn_ai_probability,
        fft_ai_probability,
    )

    final_ai_probability = normalize_probability(
        fusion_result.get(
            "final_ai_probability",
            fusion_result.get("ai_probability", 0.0),
        )
    )

    predicted_class = simple_label(
        fusion_result.get(
            "label",
            fusion_result.get("classification", "Unknown"),
        )
    )

    cnn_prediction = simple_label(
        cnn_result.get("label", cnn_result.get("prediction", "Unknown"))
    )

    fft_prediction = simple_label(
        fft_result.get("label", fft_result.get("prediction", "Unknown"))
    )

    agreement = fusion_result.get(
        "agreement_status",
        fusion_result.get("agreement", "Unknown"),
    )

    is_correct = predicted_class == actual_class

    return {
        "filename": image_path.name,
        "actual_class": actual_class,
        "predicted_class": predicted_class,
        "cnn_prediction": cnn_prediction,
        "cnn_ai_probability": round(cnn_ai_probability * 100, 2),
        "fft_prediction": fft_prediction,
        "fft_ai_probability": round(fft_ai_probability * 100, 2),
        "final_ai_probability": round(final_ai_probability * 100, 2),
        "agreement": agreement,
        "correct": "Yes" if is_correct else "No",
        "error": "",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_groups = [
        (TEST_DIR / "real", "Real"),
        (TEST_DIR / "ai_generated", "AI-Generated"),
    ]

    rows = []

    print("\nStarting batch image testing...\n")

    for folder, actual_class in test_groups:
        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue

        images = get_images(folder)

        print(f"{actual_class} images found: {len(images)}")

        for number, image_path in enumerate(images, start=1):
            print(
                f"Testing {actual_class} image "
                f"{number}/{len(images)}: {image_path.name}"
            )

            try:
                row = test_image(image_path, actual_class)
                rows.append(row)

                print(
                    f"  Prediction: {row['predicted_class']} | "
                    f"Final AI: {row['final_ai_probability']:.2f}% | "
                    f"Correct: {row['correct']}"
                )

            except Exception as error:
                print(f"  Failed: {error}")

                rows.append(
                    {
                        "filename": image_path.name,
                        "actual_class": actual_class,
                        "predicted_class": "Error",
                        "cnn_prediction": "",
                        "cnn_ai_probability": "",
                        "fft_prediction": "",
                        "fft_ai_probability": "",
                        "final_ai_probability": "",
                        "agreement": "",
                        "correct": "No",
                        "error": str(error),
                    }
                )

    fieldnames = [
        "filename",
        "actual_class",
        "predicted_class",
        "cnn_prediction",
        "cnn_ai_probability",
        "fft_prediction",
        "fft_ai_probability",
        "final_ai_probability",
        "agreement",
        "correct",
        "error",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [
        row for row in rows if row["predicted_class"] != "Error"
    ]

    correct_rows = [
        row for row in successful_rows if row["correct"] == "Yes"
    ]

    total = len(successful_rows)
    correct = len(correct_rows)
    accuracy = (correct / total * 100) if total else 0.0

    real_rows = [
        row for row in successful_rows if row["actual_class"] == "Real"
    ]

    ai_rows = [
        row
        for row in successful_rows
        if row["actual_class"] == "AI-Generated"
    ]

    real_correct = sum(row["correct"] == "Yes" for row in real_rows)
    ai_correct = sum(row["correct"] == "Yes" for row in ai_rows)

    print("\nBatch testing completed.")
    print(f"Successfully tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Real images correct: {real_correct}/{len(real_rows)}")
    print(f"AI images correct: {ai_correct}/{len(ai_rows)}")
    print(f"\nResults saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()