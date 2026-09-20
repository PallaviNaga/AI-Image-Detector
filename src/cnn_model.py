import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


NUMBER_OF_CLASSES = 2

CLASS_NAMES = {
    0: "real",
    1: "ai_generated"
}


def create_resnet18_model(
    pretrained=True,
    fine_tune=False
):
    """
    Create a ResNet18 model for binary image classification.

    pretrained=True:
        Start with ImageNet-trained weights.

    fine_tune=False:
        Freeze the ResNet feature-extraction layers and train only
        the final classification layer.
    """

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    if not fine_tune:
        for parameter in model.parameters():
            parameter.requires_grad = False

    number_of_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(
            in_features=number_of_features,
            out_features=NUMBER_OF_CLASSES
        )
    )

    # The newly created classifier must always be trainable.
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


def count_trainable_parameters(model):
    """Return the number of model parameters that will be trained."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )