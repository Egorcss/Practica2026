# Определения 5 архитектур: ResNet-50, EfficientNet-B0, MobileNetV3-Large,
# DenseNet-121, ViT-Base.
import torch.nn as nn
import timm
import torchvision.models as tvm


def build_model(model_name: str, num_classes: int = 43, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenetv3":
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "densenet121":
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "vit_base":
        model = timm.create_model("vit_base_patch16_224", pretrained=pretrained,
                                  num_classes=num_classes)
        
    elif model_name == "vgg16":
        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.vgg16(weights=weights)
        # У VGG классификатор — Sequential, последний слой nn.Linear(4096, 1000)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)    
    else:
        raise ValueError(f"Неизвестная архитектура: {model_name}")

    return model


SUPPORTED_MODELS = ["resnet50", "efficientnet_b0", "mobilenetv3", "densenet121", "vgg16"]