#!/usr/bin/env python3
"""Run a PyTorch MLP classifier for HRS diabetes prediction."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RESULT, ROOT, classification_metrics, get_xy, load_analytic, make_preprocessor, save_json


CACHE = ROOT / "archive" / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str((CACHE / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((CACHE / "xdg").resolve()))

import matplotlib.pyplot as plt

SEED = 263
BATCH_SIZE = 256
MAX_EPOCHS = 400
PATIENCE = 40


class MLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            total += float(loss.item()) * len(yb)
            n += len(yb)
    return total / n


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def main() -> None:
    set_seed(SEED)
    df = load_analytic()
    X, y = get_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train)

    preprocessor = make_preprocessor(X)
    X_fit_t = preprocessor.fit_transform(X_fit).astype(np.float32)
    X_val_t = preprocessor.transform(X_val).astype(np.float32)
    X_test_t = preprocessor.transform(X_test).astype(np.float32)
    y_fit_arr = y_fit.to_numpy(dtype=np.float32)
    y_val_arr = y_val.to_numpy(dtype=np.float32)

    pos_weight = torch.tensor([(len(y_fit_arr) - y_fit_arr.sum()) / y_fit_arr.sum()], dtype=torch.float32)
    train_loader = make_loader(X_fit_t, y_fit_arr, shuffle=True)
    val_loader = make_loader(X_val_t, y_val_arr, shuffle=False)

    model = MLP(n_features=X_fit_t.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_loss = evaluate_loss(model, train_loader, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    proba = predict_proba(model, X_test_t)
    out_dir = RESULT / "nn"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": "mlp", **classification_metrics(y_test, proba)}]).to_csv(out_dir / "perf.csv", index=False)
    pd.DataFrame({"y_true": y_test.to_numpy(), "mlp": proba}, index=y_test.index).to_csv(out_dir / "pred.csv", index=True)
    pd.DataFrame(history).to_csv(out_dir / "loss.csv", index=False)

    hist = pd.DataFrame(history)
    plt.figure(figsize=(6, 4))
    plt.plot(hist["epoch"], hist["train_loss"], label="train")
    plt.plot(hist["epoch"], hist["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE loss")
    plt.title("PyTorch MLP training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=220)
    plt.close()

    save_json(
        out_dir / "meta.json",
        {
            "model": "PyTorch MLP classifier",
            "n_train_full": len(X_train),
            "n_train_fit": len(X_fit),
            "n_validation": len(X_val),
            "n_test": len(X_test),
            "n_features_after_preprocessing": int(X_fit_t.shape[1]),
            "hidden_layers": [128, 64, 32],
            "dropout": [0.25, 0.15],
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "pos_weight": float(pos_weight.item()),
            "epochs_run": int(len(history)),
            "best_validation_loss": float(best_val),
        },
    )
    print(f"Saved PyTorch neural network results to {out_dir}")


if __name__ == "__main__":
    main()
