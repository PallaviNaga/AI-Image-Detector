import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from src.inference import load_cnn_model


GRADCAM_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class GradCAM:
    """Generate a Grad-CAM heatmap from a CNN model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input_value, output_value):
        self.activations = output_value.detach()

    def _save_gradients(self, module, gradient_input, gradient_output):
        self.gradients = gradient_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        output[:, target_class].sum().backward()

        if self.activations is None:
            raise RuntimeError("Grad-CAM could not capture CNN activations.")
        if self.gradients is None:
            raise RuntimeError("Grad-CAM could not capture CNN gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1)
        heatmap = torch.relu(heatmap).squeeze().cpu().numpy()

        minimum = float(heatmap.min())
        maximum = float(heatmap.max())
        if maximum > minimum:
            heatmap = (heatmap - minimum) / (maximum - minimum)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.float32)
        return heatmap

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def _resize_heatmap(original_image, heatmap):
    width, height = original_image.size
    return cv2.resize(
        np.asarray(heatmap, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )


def create_gradcam_overlay(original_image, heatmap, alpha=0.45):
    """Place the complete Grad-CAM colour map over the original image."""
    image_rgb = np.array(original_image.convert("RGB"))
    resized_heatmap = _resize_heatmap(original_image, heatmap)
    heatmap_uint8 = np.uint8(np.clip(resized_heatmap, 0, 1) * 255)
    coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, coloured, alpha, 0)
    return Image.fromarray(overlay)


def create_strong_influence_image(
    original_image,
    heatmap,
    percentile=80,
    highlight_alpha=0.68,
):
    """
    Emphasize only the strongest Grad-CAM activations.

    The selected pixels are CNN-influence regions, not proven AI-generated
    pixels. A red tint and outline make the high-influence areas easy to see.
    """
    image_rgb = np.array(original_image.convert("RGB"))
    resized = np.clip(_resize_heatmap(original_image, heatmap), 0, 1)
    positive_values = resized[resized > 0]

    if positive_values.size == 0:
        return Image.fromarray(image_rgb), None, 1.0

    threshold = float(np.percentile(positive_values, percentile))
    mask = np.uint8(resized >= threshold) * 255

    height, width = mask.shape
    kernel_size = max(3, int(round(min(width, height) * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Slightly dim the rest of the image, then add a vivid red/yellow tint
    # only where the CNN influence is strongest.
    result = np.uint8(np.clip(image_rgb.astype(np.float32) * 0.72, 0, 255))
    red_layer = np.zeros_like(image_rgb)
    red_layer[:, :] = (255, 35, 20)
    highlighted = cv2.addWeighted(
        image_rgb,
        1.0 - highlight_alpha,
        red_layer,
        highlight_alpha,
        0,
    )
    result[mask > 0] = highlighted[mask > 0]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    minimum_area = width * height * 0.002
    important_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= minimum_area
    ]

    bounding_box = None
    if important_contours:
        cv2.drawContours(result, important_contours, -1, (255, 255, 255), 3)
        points = np.vstack(important_contours)
        x, y, box_width, box_height = cv2.boundingRect(points)
        bounding_box = (int(x), int(y), int(box_width), int(box_height))
        cv2.rectangle(
            result,
            (x, y),
            (x + box_width, y + box_height),
            (255, 0, 0),
            max(3, int(round(min(width, height) * 0.006))),
        )

        label = "Strongest CNN influence"
        font_scale = max(0.55, min(width, height) / 900.0)
        thickness = max(1, int(round(font_scale * 2)))
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_x = max(0, min(x, width - text_width - 12))
        label_y = y - 10
        if label_y - text_height - baseline < 0:
            label_y = min(height - baseline - 4, y + text_height + baseline + 12)
        cv2.rectangle(
            result,
            (label_x, label_y - text_height - baseline - 6),
            (label_x + text_width + 12, label_y + baseline + 4),
            (255, 0, 0),
            -1,
        )
        cv2.putText(
            result,
            label,
            (label_x + 6, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return Image.fromarray(result), bounding_box, threshold


def create_heatmap_image(original_image, heatmap):
    """Convert the numerical heatmap into a coloured image."""
    resized = np.clip(_resize_heatmap(original_image, heatmap), 0, 1)
    heatmap_uint8 = np.uint8(255 * resized)
    coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return Image.fromarray(coloured)


def generate_gradcam(image, target_class=None):
    """Run the CNN and generate Grad-CAM explanations (0=Real, 1=AI)."""
    if not isinstance(image, Image.Image):
        raise TypeError("The input must be a PIL image.")

    image = image.convert("RGB")
    model, device = load_cnn_model()
    model.eval()
    target_layer = model.layer4[-1]
    input_tensor = GRADCAM_TRANSFORM(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    with torch.enable_grad():
        with torch.no_grad():
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())
        if target_class is None:
            target_class = predicted_class
        if target_class not in (0, 1):
            raise ValueError("target_class must be 0 or 1.")

        gradcam = GradCAM(model=model, target_layer=target_layer)
        try:
            heatmap = gradcam.generate(input_tensor, target_class)
        finally:
            gradcam.remove_hooks()

    overlay_image = create_gradcam_overlay(image, heatmap)
    heatmap_image = create_heatmap_image(image, heatmap)
    strong_image, bounding_box, threshold = create_strong_influence_image(
        image,
        heatmap,
        percentile=80,
    )
    class_names = {0: "Real", 1: "AI-Generated"}

    return {
        "predicted_class": predicted_class,
        "prediction": class_names[predicted_class],
        "explained_class": target_class,
        "explained_label": class_names[target_class],
        "real_probability": float(probabilities[0].item()),
        "ai_probability": float(probabilities[1].item()),
        "heatmap": heatmap,
        "heatmap_image": heatmap_image,
        "overlay_image": overlay_image,
        "strong_influence_image": strong_image,
        "strong_influence_bbox": bounding_box,
        "strong_influence_threshold": threshold,
        "strong_influence_percentile": 80,
    }
