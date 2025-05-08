import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import torchvision.transforms.functional as TF

# === Constants ===
AA_CODES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_CODES)}
NUM_FEATURES = len(AA_CODES)

# === Dataset Classes ===
class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image) if self.transform else TF.to_tensor(TF.resize(image, (224, 224)))
        return image, self.labels[idx]

class SequenceOnlyDataset(Dataset):
    def __init__(self, seq_paths, labels, max_seq_length):
        self.seq_paths = seq_paths
        self.labels = labels
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.seq_paths[idx]
        try:
            onehot = pd.read_csv(path).values.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error reading file: {path} — {e}")

        if onehot.shape[0] == 0 or onehot.shape[1] != NUM_FEATURES:
            raise ValueError(f"Invalid or empty one-hot file: {path}")

        onehot = onehot[:self.max_seq_length]
        if len(onehot) < self.max_seq_length:
            pad_len = self.max_seq_length - len(onehot)
            onehot = np.vstack([onehot, np.zeros((pad_len, NUM_FEATURES), dtype=np.float32)])

        mean = np.mean(onehot)
        std = np.std(onehot)
        if not (np.isnan(mean) or np.isnan(std) or std < 1e-6):
            onehot = (onehot - mean) / (std + 1e-6)

        return torch.tensor(onehot, dtype=torch.float32), self.labels[idx]

# === Training Function ===
def train_model(model, train_loader, val_loader=None, num_epochs=10, lr=1e-5, device='cpu', model_name="Model"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    print(f"\n--- Starting training for {model_name} ---")
    for epoch in range(num_epochs):
        model.train()
        correct, total, total_loss = 0, 0, 0.0
        for batch in train_loader:
            if len(batch) == 3:
                img, seq, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = model(img, seq)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                outputs = model(x)

            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        train_accs.append(correct / total)
        train_losses.append(total_loss / len(train_loader))

        if val_loader:
            model.eval()
            correct, total, total_loss = 0, 0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        img, seq, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                        outputs = model(img, seq)
                    else:
                        x, y = batch[0].to(device), batch[1].to(device)
                        outputs = model(x)

                    loss = criterion(outputs, y)
                    total_loss += loss.item()
                    correct += (outputs.argmax(1) == y).sum().item()
                    total += y.size(0)

            val_accs.append(correct / total)
            val_losses.append(total_loss / len(val_loader))
            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
        else:
            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | Train Acc: {train_accs[-1]:.4f} | Train Loss: {train_losses[-1]:.4f}")

    return model, train_accs, val_accs, train_losses, val_losses

# === Evaluation ===
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)

# === Path Utility ===
def get_max_seq_length(csv_dirs):
    max_len = 0
    for path in csv_dirs:
        try:
            df = pd.read_csv(path)
            if df.shape[1] == NUM_FEATURES:
                max_len = max(max_len, len(df))
        except Exception:
            continue
    return max_len


def load_image_sequence_paths(base_dir, classes):
    image_paths, onehot_paths, labels = [], [], []
    for class_idx, class_name in enumerate(classes):
        img_dir = os.path.join(base_dir, class_name, "voronoi_images")
        csv_dir = os.path.join(base_dir, class_name, "onehot_sequences")
        for img_path in glob.glob(os.path.join(img_dir, "*.png")):
            base = os.path.basename(img_path).replace(".pdb.png", "").replace(".png", "")
            base = base.replace("AF_REMOTE_", "") if base.startswith("AF_REMOTE_AF-") else base
            for suffix in ["_onehot.csv", ".csv"]:
                onehot_path = os.path.join(csv_dir, base + suffix)
                if os.path.exists(onehot_path):
                    image_paths.append(img_path)
                    onehot_paths.append(onehot_path)
                    labels.append(class_idx)
                    break
    return image_paths, onehot_paths, labels