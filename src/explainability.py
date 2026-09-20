import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from src.inference import load_cnn_model


# ImageNet normalization used by the ResNet18 CNN.
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
    """
    Generates a Grad-CAM heatmap from a CNN model.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._save_activations
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._save_gradients
            )
        )

    def _save_activations(
        self,
        module,
        input_value,
        output_value,
    ):
        self.activations = output_value.detach()

    def _save_gradients(
        self,
        module,
        gradient_input,
        gradient_output,
    ):
        self.gradients = gradient_output[0].detach()

    def generate(
        self,
        input_tensor,
        target_class,
    ):
        """
        Generates the normalized Grad-CAM heatmap.
        """

        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        class_score = output[:, target_class].sum()

        class_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM could not capture CNN activations."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM could not capture CNN gradients."
            )

        # Average the gradients across height and width.
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted combination of activation maps.
        heatmap = (
            weights * self.activations
        ).sum(dim=1)

        # Retain only positive influence.
        heatmap = torch.relu(heatmap)

        heatmap = heatmap.squeeze().cpu().numpy()

        minimum = float(heatmap.min())
        maximum = float(heatmap.max())

        if maximum > minimum:
            heatmap = (
                heatmap - minimum
            ) / (
                maximum - minimum
            )
        else:
            heatmap = np.zeros_like(
                heatmap,
                dtype=np.float32,
            )

        return heatmap

    def remove_hooks(self):
        """
        Removes the PyTorch hooks after Grad-CAM generation.
        """
        self.forward_handle.remove()
        self.backward_handle.remove()


def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=0.45,
):
    """
    Places the Grad-CAM heatmap over the original image.
    """

    image_rgb = np.array(
        original_image.convert("RGB")
    )

    original_height, original_width = (
        image_rgb.shape[:2]
    )

    resized_heatmap = cv2.resize(
        heatmap,
        (original_width, original_height),
    )

    heatmap_uint8 = np.uint8(
        255 * resized_heatmap
    )

    coloured_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    # OpenCV produces BGR, so convert it to RGB.
    coloured_heatmap = cv2.cvtColor(
        coloured_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        image_rgb,
        1.0 - alpha,
        coloured_heatmap,
        alpha,
        0,
    )

    return Image.fromarray(overlay)


def create_heatmap_image(
    original_image,
    heatmap,
):
    """
    Converts the numerical heatmap into a coloured image.
    """

    width, height = original_image.size

    resized_heatmap = cv2.resize(
        heatmap,
        (width, height),
    )

    heatmap_uint8 = np.uint8(
        255 * resized_heatmap
    )

    coloured_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    coloured_heatmap = cv2.cvtColor(
        coloured_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(
        coloured_heatmap
    )


def generate_gradcam(
    image,
    target_class=None,
):
    """
    Runs the CNN and generates its Grad-CAM explanation.

    Class mapping:
        0 = Real
        1 = AI-Generated
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "The input must be a PIL image."
        )

    image = image.convert("RGB")

    model, device = load_cnn_model()

    model.eval()

    # The last convolutional block of ResNet18.
    target_layer = model.layer4[-1]

    input_tensor = GRADCAM_TRANSFORM(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # This ensures gradients remain available even when
    # the pretrained backbone parameters are frozen.
    input_tensor.requires_grad_(True)

    with torch.enable_grad():
        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        if target_class is None:
            target_class = predicted_class

        if target_class not in (0, 1):
            raise ValueError(
                "target_class must be 0 or 1."
            )

        gradcam = GradCAM(
            model=model,
            target_layer=target_layer,
        )

        try:
            heatmap = gradcam.generate(
                input_tensor=input_tensor,
                target_class=target_class,
            )
        finally:
            gradcam.remove_hooks()

    overlay_image = create_gradcam_overlay(
        original_image=image,
        heatmap=heatmap,
    )

    heatmap_image = create_heatmap_image(
        original_image=image,
        heatmap=heatmap,
    )

    class_names = {
        0: "Real",
        1: "AI-Generated",
    }

    return {
        "predicted_class": predicted_class,
        "prediction": class_names[predicted_class],
        "explained_class": target_class,
        "explained_label": class_names[target_class],
        "real_probability": float(
            probabilities[0].item()
        ),
        "ai_probability": float(
            probabilities[1].item()
        ),
        "heatmap": heatmap,
        "heatmap_image": heatmap_image,
        "overlay_image": overlay_image,
    }