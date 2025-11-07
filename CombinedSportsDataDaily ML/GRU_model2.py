# gru_model.py
"""
Simple GRU training script for the preprocessed data.
Loads X_train.npy / X_val.npy and y arrays from preprocessDaily.py output.
Provides options for class weighting, focal loss, and optional slicing
to a smaller SEQUENCE_LENGTH if you want to test different lengths without
rewriting the preprocessing stage (but cannot increase length beyond preprocessed).
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle

DATA_DIR = "preprocessed_data2"
SEQUENCE_LENGTH = 7   # set this <= TIME_STEPS in meta; if < TIME_STEPS, model will use the last SEQUENCE_LENGTH timesteps
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FOCAL = False  # set True to use focal loss
FOCAL_GAMMA = 2.0

# --- load preprocessed data
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

# Optional: load meta to check TIME_STEPS
try:
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
        print("Meta:", meta)
        TIME_STEPS = meta['TIME_STEPS']
except Exception:
    TIME_STEPS = X_train.shape[1]

print("Preprocessed shapes:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)
if X_train.shape[1] < SEQUENCE_LENGTH:
    raise ValueError(f"Requested SEQUENCE_LENGTH {SEQUENCE_LENGTH} > available TIME_STEPS {X_train.shape[1]}")
if X_train.shape[1] > SEQUENCE_LENGTH:
    # slice to last SEQUENCE_LENGTH timesteps (keep recent history)
    X_train = X_train[:, -SEQUENCE_LENGTH:, :]
    X_val = X_val[:, -SEQUENCE_LENGTH:, :]
    print(f"Sliced sequences to last {SEQUENCE_LENGTH} timesteps; new shape:", X_train.shape)

N_FEATURES = X_train.shape[2]

# PyTorch Dataset
class SeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# Model
class GRUModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0.0, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        # x: (batch, seq, features)
        out, h = self.gru(x)  # out: (batch, seq, hidden)
        last = out[:, -1, :]  # last time-step output
        logits = self.fc(last).squeeze(-1)
        return logits

model = GRUModel(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)

# Loss: use BCEWithLogits with pos_weight, or focal loss
pos_rate = y_train.mean()
pos_weight = (1 - pos_rate) / (pos_rate + 1e-9)
# clip extreme values
pos_weight_clipped = max(1.0, min(100.0, pos_weight))
print("Train pos rate:", pos_rate, "raw pos_weight:", pos_weight, "clipped:", pos_weight_clipped)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_clipped).to(DEVICE))

# Optional focal loss implementation
if USE_FOCAL:
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=None):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
        def forward(self, logits, targets):
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            p = torch.sigmoid(logits)
            p_t = p*targets + (1-p)*(1-targets)
            loss = bce * ((1 - p_t) ** self.gamma)
            if self.alpha is not None:
                alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
                loss = loss * alpha_t
            return loss.mean()
    criterion = FocalLoss(gamma=FOCAL_GAMMA)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
)
import json
import csv

# --- training setup ---
best_val_auc = 0.0
patience = 6
patience_ctr = 0

# Track metrics per epoch
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_auc": [],
    "val_pr": [],
    "val_acc": [],
    "val_f1": [],
    "val_precision": [],
    "val_recall": []
}

def evaluate_metrics(model, loader):
    model.eval()
    ys, probs, losses = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(probs)
    y_bin = (y_pred >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = np.nan
    try:
        pr = average_precision_score(y_true, y_pred)
    except:
        pr = np.nan
    acc = accuracy_score(y_true, y_bin)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    prec = precision_score(y_true, y_bin, zero_division=0)
    rec = recall_score(y_true, y_bin, zero_division=0)
    return {
        "loss": np.mean(losses),
        "auc": auc,
        "pr": pr,
        "acc": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    }

# --- main training loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    val_metrics = evaluate_metrics(model, val_loader)
    mean_train_loss = np.mean(train_losses)

    history["epoch"].append(epoch)
    history["train_loss"].append(mean_train_loss)
    for k in ["loss", "auc", "pr", "acc", "f1", "precision", "recall"]:
        history[f"val_{k}"].append(val_metrics[k])

    print(
        f"Epoch {epoch:02d} | "
        f"TrainLoss {mean_train_loss:.4f} | ValLoss {val_metrics['loss']:.4f} | "
        f"AUC {val_metrics['auc']:.4f} | F1 {val_metrics['f1']:.4f} | Acc {val_metrics['acc']:.4f}"
    )

    if val_metrics["auc"] > best_val_auc:
        best_val_auc = val_metrics["auc"]
        patience_ctr = 0
        # Save model weights as NumPy archive
        weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        np.savez(os.path.join(DATA_DIR, "gru_best_weights.npz"), **weights)
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print("Early stopping triggered.")
            break

# --- After training ---
print("\nTraining complete. Best val AUC:", best_val_auc)
weights_final = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
np.savez(os.path.join(DATA_DIR, "gru_final_weights.npz"), **weights_final)

# --- Save metrics to CSV and JSON ---
metrics_csv = os.path.join(DATA_DIR, "training_metrics.csv")
with open(metrics_csv, "w", newline="") as f:
    writer = csv.writer(f)
    header = list(history.keys())
    writer.writerow(header)
    rows = zip(*[history[k] for k in header])
    writer.writerows(rows)
print("Metrics saved to", metrics_csv)

summary_json = os.path.join(DATA_DIR, "final_results.json")
best_epoch_idx = int(np.nanargmax(history["val_auc"]))
summary = {
    "best_epoch": history["epoch"][best_epoch_idx],
    "best_val_auc": history["val_auc"][best_epoch_idx],
    "final_val_auc": history["val_auc"][-1],
    "final_val_f1": history["val_f1"][-1],
    "final_val_precision": history["val_precision"][-1],
    "final_val_recall": history["val_recall"][-1],
    "final_val_accuracy": history["val_acc"][-1]
}
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=4)
print("Summary JSON saved to", summary_json)

# --- Plot metrics ---
plt.figure(figsize=(10, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DATA_DIR, "loss_curve.png"), dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history["epoch"], history["val_auc"], label="AUC")
plt.plot(history["epoch"], history["val_f1"], label="F1")
plt.plot(history["epoch"], history["val_acc"], label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Metrics")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DATA_DIR, "metrics_curve.png"), dpi=200)
plt.close()

print("Plots saved as loss_curve.png and metrics_curve.png in", DATA_DIR)
