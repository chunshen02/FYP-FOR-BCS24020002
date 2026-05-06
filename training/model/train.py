"""
train.py

Usage:
    python train.py --data_dir <crops_root> --label_csv <labels.csv>

Arguments:
    --data_dir      Root directory containing face crop images
    --label_csv     CSV file with columns: filename, label (0=real, 1=fake)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import roc_auc_score
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


# -- Dataset -------------------------------------------------------------------


class FaceDataset(Dataset):
    def __init__(self, data_dir: str, samples: list, transform=None):
        self.data_dir = data_dir
        self.samples = samples  # list of (filename, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# -- Transforms ----------------------------------------------------------------

train_transform = transforms.Compose(
    [
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.INPUT_MEAN, std=config.INPUT_STD),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.INPUT_MEAN, std=config.INPUT_STD),
    ]
)


# -- Utilities -----------------------------------------------------------------


def load_csv(label_csv: str):
    samples = []
    with open(label_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row["filename"], int(row["label"])))
    return samples


def split_samples(samples: list):
    """Video-level split: 70% train, 15% val, 15% test."""
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    n = len(samples)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = train_end + int(n * config.VAL_SPLIT)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]
    return train, val, test


def compute_class_weights(samples: list) -> torch.Tensor:
    labels = [s[1] for s in samples]
    n_total = len(labels)
    n_fake = sum(labels)
    n_real = n_total - n_fake
    weight_real = n_total / (2 * n_real) if n_real > 0 else 1.0
    weight_fake = n_total / (2 * n_fake) if n_fake > 0 else 1.0
    return torch.tensor([weight_real, weight_fake], dtype=torch.float32)


def build_model() -> nn.Module:
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=True,
        num_classes=0,  # Remove default head
    )
    in_features = model.num_features
    model.classifier = nn.Sequential(nn.Dropout(p=config.DROPOUT), nn.Linear(in_features, 2))
    return model


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# -- Training Loop -------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        labels_one_hot = F.one_hot(labels, num_classes=2).float()
        loss = criterion(outputs, labels_one_hot)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels_one_hot = F.one_hot(labels, num_classes=2).float()
            loss = criterion(outputs, labels_one_hot)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, auc


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--label_csv", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    # Load and split data
    samples = load_csv(args.label_csv)
    train_samples, val_samples, test_samples = split_samples(samples)
    print(
        f"[INFO] Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}"
    )

    # Save test split for evaluation (sealed)
    test_csv_path = os.path.join(config.CHECKPOINTS_DIR, "test_split.csv")
    with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(test_samples)
    print(f"[INFO] Test split sealed and saved to: {test_csv_path}")

    # Datasets and loaders
    train_dataset = FaceDataset(args.data_dir, train_samples, train_transform)
    val_dataset = FaceDataset(args.data_dir, val_samples, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = build_model().to(device)

    # Class weights
    class_weights = compute_class_weights(train_samples).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=class_weights.view(1, 2))

    # -- Stage 1: Train head only (epochs 1 to STAGE1_EPOCHS) -----------------
    print(f"\n[STAGE 1] Training head only for {config.STAGE1_EPOCHS} epochs...")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.STAGE1_LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(1, config.STAGE1_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        print(
            f"  Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

    # -- Stage 2: Full fine-tune (epochs STAGE1_EPOCHS+1 to NUM_EPOCHS) -------
    print(
        f"\n[STAGE 2] Full fine-tuning for {config.NUM_EPOCHS - config.STAGE1_EPOCHS} epochs..."
    )
    unfreeze_all(model)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for n, p in model.named_parameters() if "classifier" not in n],
                "lr": config.STAGE2_BACKBONE_LR,
            },
            {"params": model.classifier.parameters(), "lr": config.STAGE2_HEAD_LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    best_auc = 0.0
    patience_counter = 0
    best_ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_model.pt")

    for epoch in range(config.STAGE1_EPOCHS + 1, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        print(
            f"  Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  [SAVED] New best model (AUC: {best_auc:.4f}) -> {best_ckpt_path}")
        else:
            patience_counter += 1
            print(
                f"  [INFO] No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  [EARLY STOP] Stopping at epoch {epoch}.")
                break

    print(f"\n[DONE] Training complete. Best Val AUC: {best_auc:.4f}")
    print(f"[DONE] Best checkpoint saved at: {best_ckpt_path}")


if __name__ == "__main__":
    main()
