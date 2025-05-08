# hybrid_cnn_transformer.py

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np


# CNN for image processing
class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)  # flatten

# Transformer for one-hot sequence
class SequenceTransformer(nn.Module):
    def __init__(self, input_dim=20, emb_dim=64, nhead=4, hidden_dim=128, nlayers=2):
        super(SequenceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, emb_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch, emb_dim]
        x = self.transformer(x)  # [seq_len, batch, emb_dim]
        x = x.permute(1, 2, 0)  # [batch, emb_dim, seq_len]
        x = self.pool(x).squeeze(2)  # [batch, emb_dim]
        return x

# Combined model
class CNNTransformerNet(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNTransformerNet, self).__init__()
        self.cnn = CNNBackbone()
        self.transformer = SequenceTransformer()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, seq):
        img_feat = self.cnn(img)
        seq_feat = self.transformer(seq)
        combined = torch.cat((img_feat, seq_feat), dim=1)
        return self.classifier(combined)

# === Dataset Class ===
class HybridProteinDataset(Dataset):
    def __init__(self, image_paths, onehot_paths, labels, transform=None, max_seq_length=512):
        self.image_paths = image_paths
        self.onehot_paths = onehot_paths
        self.labels = labels
        self.transform = transform
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            import torchvision.transforms as transforms
            image = transforms.ToTensor()(image)

        onehot = pd.read_csv(self.onehot_paths[idx]).values.astype("float32")

        # Pad or crop to self.max_seq_length
        onehot = onehot[:self.max_seq_length]
        if len(onehot) < self.max_seq_length:
            pad_len = self.max_seq_length - len(onehot)
            onehot = np.vstack([onehot, np.zeros((pad_len, onehot.shape[1]), dtype=np.float32)])

        onehot_tensor = torch.tensor(onehot)

        return image, onehot_tensor, self.labels[idx]

