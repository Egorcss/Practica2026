# Загрузка и предобработка датасета GTSRB(German Traffic Sign Recognition Benchmark) 
# с сайта kaggle.
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    # Аугментации для train: повороты, яркость, контраст, сдвиги, масштаб.
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    # Без аугментаций — только resize и нормализация.
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class GTSRBDataset(Dataset):
    
    # Читает официальный GTSRB.
    # CSV-разметка (Train.csv/Test.csv) содержит колонки 'Path' и 'ClassId'.
    # Пути в CSV относительны корня GTSRB (например, 'Train/00000_00000.ppm').
    

    def __init__(self, root: str, csv_file: str, transform=None):
        self.root = Path(root)
        self.transform = transform

        csv_path = self.root / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Не найден CSV: {csv_path}\n"
                f"Проверьте раскладку: {self.root}/Train.csv и Test.csv"
            )

        df = pd.read_csv(csv_path)
        # Колонка 'Path' — путь относительно корня GTSRB
        self.paths = df["Path"].astype(str).tolist()
        self.labels = df["ClassId"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.root / self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataloaders(config: dict):
    
    # Собирает train/val/test DataLoader'-ы для официального GTSRB.
    # Ожидаемая структура:
    #     data/raw/GTSRB/Train/*.ppm
    #     data/raw/GTSRB/Test/*.ppm
    #     data/raw/GTSRB/Train.csv
    #     data/raw/GTSRB/Test.csv
    
    root = config["data"]["root"]
    input_size = config["input_size"]
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 4)
    val_split = config["data"].get("val_split", 0.2)
    seed = config.get("seed", 42)

    # Полный train-датасет (без transforms — разделим, потом навесим)
    full_train = GTSRBDataset(root, "Train.csv", transform=None)

    # Число классов
    num_classes = max(full_train.labels) + 1
    class_names = [f"class_{i}" for i in range(num_classes)]

    # Разбиение train -> train + val
    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train, [n_train, n_val], generator=generator
    )

    # Обёртки с transforms
    train_dataset = _TransformSubset(train_subset, get_train_transforms(input_size))
    val_dataset = _TransformSubset(val_subset, get_val_transforms(input_size))

    # Test — читаем отдельным датасетом
    test_dataset = GTSRBDataset(root, "Test.csv", transform=get_val_transforms(input_size))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


class _TransformSubset(Dataset):
    # Обёртка над Subset, чтобы применить transforms.

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label