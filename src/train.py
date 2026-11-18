"""Training entry point: comparison experiments + local test evaluation + result plot.

Usage:
    python src/train.py
The dataset must be downloaded first (see the "Data preparation" section of the README).
"""
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Allow the script to be run directly as well as imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DEVICE, EPOCHS, FIG_DIR, HIDDEN_DIM, LR,
                    SEED, WEIGHT_DIR)
from data import build_datasets, load_data, make_loaders, validate
from models import CnnLstm, LstmBaseline, MaskedMAELoss


def evaluate_all_metrics(model, loader, device=DEVICE):
    """Compute inhalation MAE, full-sequence MAE and full-sequence RMSE."""
    model.eval()
    inhale_abs = inhale_cnt = all_abs = all_sq = all_cnt = 0.0
    with torch.no_grad():
        for x, y, u_out in loader:
            x, y, u_out = x.to(device), y.to(device), u_out.to(device)
            pred = model(x)
            abs_err, sq_err = torch.abs(pred - y), torch.square(pred - y)
            mask = u_out == 0
            inhale_abs += torch.sum(abs_err[mask]).item()
            inhale_cnt += torch.sum(mask).item()
            all_abs += torch.sum(abs_err).item()
            all_sq += torch.sum(sq_err).item()
            all_cnt += torch.numel(y)
    return {
        "val_inhale_mae": inhale_abs / inhale_cnt if inhale_cnt else float("inf"),
        "full_mae": all_abs / all_cnt,
        "full_rmse": math.sqrt(all_sq / all_cnt),
    }


def train_one(model, train_loader, val_loader, device=DEVICE, epochs=EPOCHS, lr=LR,
              verbose=True):
    """Train a model and return (best weights, best validation inhalation MAE)."""
    criterion = MaskedMAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mae, best_w = float("inf"), None
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss, n_batch = 0.0, 0
        for x, y, u in train_loader:
            x, y, u = x.to(device), y.to(device), u.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y, u)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        met = evaluate_all_metrics(model, val_loader, device)
        if verbose:
            print(f"Epoch {epoch+1:2d}/{epochs} | train_loss:{total_loss/n_batch:.4f} | "
                  f"val_inhale_MAE:{met['val_inhale_mae']:.4f} | time:{time.time()-t0:.2f}s")
        if met["val_inhale_mae"] < best_mae:
            best_mae = met["val_inhale_mae"]
            best_w = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_w is None:
        best_w = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_w, best_mae


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device = {DEVICE}")

    df = load_data()
    validate(df)
    datasets, feat_dim = build_datasets(df)
    loaders = make_loaders(datasets)

    results = []

    # ---- 1. Train the primary models and save weights ----
    print("\n===== Training LSTM baseline =====")
    lstm = LstmBaseline(feat_dim, HIDDEN_DIM).to(DEVICE)
    w, best = train_one(lstm, loaders["train"], loaders["val"])
    lstm.load_state_dict(w)
    torch.save(w, WEIGHT_DIR / "lstm_best.pt")
    print(f"LSTM best validation inhalation MAE = {best:.4f}")

    print("\n===== Training primary CNN-LSTM =====")
    cnn = CnnLstm(feat_dim, HIDDEN_DIM, kernel_size=3).to(DEVICE)
    w, best = train_one(cnn, loaders["train"], loaders["val"])
    cnn.load_state_dict(w)
    torch.save(w, WEIGHT_DIR / "cnn_lstm_best.pt")
    print(f"CNN-LSTM best validation inhalation MAE = {best:.4f}")

    # ---- 2. Hyperparameter comparison ----
    print("\n===== Comparison: LSTM hidden dim 32 / 64 =====")
    for hidden in (32, 64):
        model = LstmBaseline(feat_dim, hidden).to(DEVICE)
        best_w, _ = train_one(model, loaders["train"], loaders["val"], verbose=False)
        model.load_state_dict(best_w)
        met = evaluate_all_metrics(model, loaders["val"])
        results.append({"model_name": f"LSTM_h{hidden}", **met,
                        "params": count_params(model), "epoch_run": EPOCHS})

    print("===== Comparison: CNN-LSTM kernel size 3 / 5 =====")
    for kernel in (3, 5):
        model = CnnLstm(feat_dim, HIDDEN_DIM, kernel_size=kernel).to(DEVICE)
        best_w, _ = train_one(model, loaders["train"], loaders["val"], verbose=False)
        model.load_state_dict(best_w)
        met = evaluate_all_metrics(model, loaders["val"])
        results.append({"model_name": f"CNN-LSTM_k{kernel}", **met,
                        "params": count_params(model), "epoch_run": EPOCHS})

    print("\n===== Validation metrics summary =====")
    df_res = pd.DataFrame(results)
    try:
        print(df_res.to_markdown(index=False, floatfmt=".4f"))
    except ImportError:
        print(df_res.to_string(index=False))

    # ---- 3. Local test set evaluation ----
    print("\n===== Local test set metrics =====")
    lstm.load_state_dict(torch.load(WEIGHT_DIR / "lstm_best.pt", map_location=DEVICE, weights_only=True))
    cnn.load_state_dict(torch.load(WEIGHT_DIR / "cnn_lstm_best.pt", map_location=DEVICE, weights_only=True))
    for name, model in (("LSTM_h64", lstm), ("CNN-LSTM_k3", cnn)):
        met = evaluate_all_metrics(model, loaders["test"])
        print(f"{name} | inhalation MAE:{met['val_inhale_mae']:.4f} | "
              f"full MAE:{met['full_mae']:.4f} | full RMSE:{met['full_rmse']:.4f}")

    # ---- 4. Plot 4 test samples ----
    preds, ys, us = [], [], []
    for x, y, u in loaders["test"]:
        x = x.to(DEVICE)
        p = cnn(x).detach().cpu().numpy()
        preds.extend(p)
        ys.extend(y.numpy())
        us.extend(u.numpy())

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for i in range(4):
        yt = np.asarray(ys[i]).reshape(-1)
        pc = np.asarray(preds[i]).reshape(-1)
        uf = np.asarray(us[i]).reshape(-1)
        xx = np.arange(len(yt))
        ax0 = axes[i, 0]
        ax0.plot(xx, yt, "k-", label="Ground truth")
        ax0.plot(xx, pc, "orange", label="CNN-LSTM pred")
        ax0.fill_between(xx, 0, np.max(yt) + 5, where=(uf == 0), color="#7ccd7c",
                         alpha=0.3, label="Inhalation")
        ax0.set_ylabel("Pressure")
        ax0.set_xlabel("Time step")
        ax0.legend(fontsize=8)
        ax0.grid(alpha=0.3)

        ax1 = axes[i, 1]
        ax1.plot(xx, np.abs(yt - pc), "orange", label="Absolute error")
        ax1.set_ylabel("Absolute error")
        ax1.set_xlabel("Time step")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "four_samples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nResult figure saved: {FIG_DIR / 'four_samples.png'}")


if __name__ == "__main__":
    main()
