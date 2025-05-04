#!/usr/bin/env python3
"""
Minimal contrastive‑learning demo on Chowell 2023 MSK‑IMPACT data.
Runs 20 epochs of triplet training, printing the average loss per epoch.
"""

import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
# 1. Load the training sheet
# ----------------------------------------------------------------------
DATA_PATH = "data/chowell-2023-msk-impact.xlsx"      # adjust if needed

df = pd.read_excel(
    DATA_PATH,
    sheet_name="Training",
    usecols=[
        "Age",
        "Cancer_type_grouped_2",
        "Response (1:Responder; 0:Non-responder)",
    ],
)

# ----------------------------------------------------------------------
# 2. Split features / labels
# ----------------------------------------------------------------------
y = df["Response (1:Responder; 0:Non-responder)"].values
X_raw = df.drop(columns=["Response (1:Responder; 0:Non-responder)"])

# ----------------------------------------------------------------------
# 3. Encode + scale
# ----------------------------------------------------------------------
num_cols = ["Age"]
cat_cols = ["Cancer_type_grouped_2"]

ct = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ]
)

X_np = ct.fit_transform(X_raw)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# ----------------------------------------------------------------------
# 4. Dataset that yields (anchor, positive, negative)
# ----------------------------------------------------------------------
class TripletDS(Dataset):
    def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
        self.X = feats
        self.y = labels
        # map: label -> tensor of indices with that label
        self.label_to_idx = {
            int(lab): torch.where(labels == lab)[0] for lab in labels.unique()
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        anchor_x = self.X[idx]
        anchor_y = int(self.y[idx].item())

        pos_pool = self.label_to_idx[anchor_y]
        pos_idx = int(pos_pool[random.randrange(len(pos_pool))])

        neg_label = 1 - anchor_y  # only 0/1 here
        neg_pool = self.label_to_idx[neg_label]
        neg_idx = int(neg_pool[random.randrange(len(neg_pool))])

        return anchor_x, self.X[pos_idx], self.X[neg_idx]


dl = DataLoader(TripletDS(X, y), batch_size=128, shuffle=True, drop_last=True)

# ----------------------------------------------------------------------
# 5. Tiny encoder + triplet loss
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(in_dim, emb_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(z), dim=1)


model = Encoder(X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.TripletMarginLoss(margin=1.0)

# ----------------------------------------------------------------------
# 6. Training loop
# ----------------------------------------------------------------------
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for anc, pos, neg in dl:
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

        optimizer.zero_grad()
        loss = criterion(model(anc), model(pos), model(neg))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dl)
    print(f"Epoch {epoch:02d}/{EPOCHS} — triplet loss: {avg_loss:.4f}")

print("\nTraining complete.")
