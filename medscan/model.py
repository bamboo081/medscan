"""Pretrained, single- and multi-label 2D image backbones.

Checkpoint architecture strings are stable; existing ResNet18/EfficientNet-B0
checkpoints remain loadable. Always compare models on validation data.
"""
from torch import nn
from torchvision.models import (
    resnet18, ResNet18_Weights, resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)

MODELS = ("resnet18", "resnet50", "efficientnet_b0", "efficientnet_v2_s", "convnext_tiny")


def build_model(name, classes, pretrained=True):
    if name == "resnet18":
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, len(classes))
    elif name == "resnet50":
        net = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, len(classes))
    elif name == "efficientnet_b0":
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(classes))
    elif name == "efficientnet_v2_s":
        net = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(classes))
    elif name == "convnext_tiny":
        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(classes))
    else:
        raise ValueError(f"Unknown architecture {name}; supported: {MODELS}")
    return net
