
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from preprocessing import enhance_kidney_image


# CPU is the safer starting point for an 8 GB MacBook.
# We can test MPS acceleration later.
DEVICE = torch.device("cpu")

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_models():
    print("Loading pretrained DenseNet121...")
    densenet = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )
    densenet.classifier = torch.nn.Identity()
    densenet.eval().to(DEVICE)

    print("Loading pretrained ResNet101...")
    resnet = models.resnet101(
        weights=models.ResNet101_Weights.DEFAULT
    )
    resnet.fc = torch.nn.Identity()
    resnet.eval().to(DEVICE)

    for model in (densenet, resnet):
        for parameter in model.parameters():
            parameter.requires_grad = False

    return densenet, resnet


@torch.inference_mode()
def extract_features(image, densenet, resnet):
    enhanced = enhance_kidney_image(image)
    tensor = TRANSFORM(enhanced).unsqueeze(0).to(DEVICE)

    dense_features = densenet(tensor).flatten().cpu().numpy()
    res_features = resnet(tensor).flatten().cpu().numpy()

    fused_features = np.concatenate(
        [dense_features, res_features]
    )

    return fused_features


if __name__ == "__main__":
    files = sorted(
        p for p in Path("dataset/Normal").rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not files:
        raise RuntimeError("No Normal CT images found.")

    densenet, resnet = load_models()

    with Image.open(files[0]) as img:
        image = img.convert("RGB")

    features = extract_features(image, densenet, resnet)

    print("DenseNet121 features: 1024")
    print("ResNet101 features: 2048")
    print("Fused feature shape:", features.shape)
    print("Feature extraction successful!")
