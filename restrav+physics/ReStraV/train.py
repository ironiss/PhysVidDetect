"""
train.py
--------
Train a binary classifier (real vs AI-generated video).

Usage
-----
    # Early fusion, MLP (default)
    python train.py --features both

    # Early fusion, Logistic Regression with per-block normalisation
    python train.py --features both --model logreg

    # Early fusion, XGBoost
    python train.py --features both --model xgb

    # Late fusion: two separate models, combined probs (alpha tuned on val)
    python train.py --features both --fusion late

    # Single-modality
    python train.py --features dinov2
    python train.py --features physics
"""

import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve,
)
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--features", choices=["dinov2", "physics", "both"],
                    default="dinov2")
parser.add_argument("--model",   choices=["mlp", "logreg", "xgb"],
                    default="mlp",
                    help="Classifier: mlp (default), logreg, xgb")
parser.add_argument("--fusion",  choices=["early", "late"],
                    default="early",
                    help="'early'=concatenate features (default), "
                         "'late'=combine model outputs (only for --features both)")
parser.add_argument("--epochs",  type=int,   default=10)
parser.add_argument("--lr",      type=float, default=1e-3)
parser.add_argument("--batch",   type=int,   default=32)
args = parser.parse_args()

prefix     = f"{args.features}_{args.model}_{args.fusion}"
PLOTS_DIR  = Path(f"plots_{prefix}")
PLOTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Load features
# ─────────────────────────────────────────────────────────────
H5_DINOV2  = Path("DATA/training_features.h5")
H5_PHYSICS = Path("DATA/physics_features.h5")


def load_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        X     = f["features"][:].astype(np.float32)
        y     = f["label"][:].astype(np.int64)
        paths = f["path"][:].astype(str)
    return X, y, paths


def fill_nan(X: np.ndarray) -> np.ndarray:
    col_medians = np.nanmedian(X, axis=0)
    nan_mask    = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


def merge_by_path(X1, y1, p1, X2, y2, p2):
    """Inner-join on filename; returns aligned (X1, X2, y, paths, filenames)."""
    name1 = {Path(p).name: i for i, p in enumerate(p1)}
    name2 = {Path(p).name: i for i, p in enumerate(p2)}
    common = sorted(set(name1) & set(name2))
    if not common:
        raise ValueError("No common video paths between the two feature files!")
    i1 = np.array([name1[n] for n in common])
    i2 = np.array([name2[n] for n in common])
    return X1[i1], X2[i2], y1[i1], np.array([p1[j] for j in i1]), common


# ── Load ────────────────────────────────────────────────────
X_d = X_p = None   # keep separate blocks for late fusion / per-block norm

if args.features == "dinov2":
    X, y, paths = load_h5(H5_DINOV2)
    dim_d = X.shape[1]

elif args.features == "physics":
    X, y, paths = load_h5(H5_PHYSICS)
    X = fill_nan(X)

else:  # both
    X_d, y_d, p_d = load_h5(H5_DINOV2)
    X_p, y_p, p_p = load_h5(H5_PHYSICS)
    X_p = fill_nan(X_p)
    X_d, X_p, y, paths, common_names = merge_by_path(X_d, y_d, p_d, X_p, y_p, p_p)
    dim_d, dim_p = X_d.shape[1], X_p.shape[1]

    # ── Step A: alignment diagnostics ───────────────────────
    print(f"\n=== Step A: Alignment check ===")
    print(f"DINOv2  features : {X_d.shape}")
    print(f"Physics features : {X_p.shape}")
    print(f"Common videos    : {len(common_names)}")
    print("First 3 filenames:")
    for nm in common_names[:3]:
        print(f"  {nm}")

    X = np.concatenate([X_d, X_p], axis=1)   # used for early fusion

print(f"\nFeature mode : {args.features}  model={args.model}  fusion={args.fusion}")
print(f"Loaded {len(X)} samples  dim={X.shape[1]}")

# ─────────────────────────────────────────────────────────────
# Balance classes
# ─────────────────────────────────────────────────────────────
idx_real = np.where(y == 1)[0]
idx_fake = np.where(y == 0)[0]
n_bal    = min(len(idx_real), len(idx_fake))
rng      = np.random.default_rng(42)
bal_idx  = np.concatenate([
    rng.choice(idx_real, n_bal, replace=False),
    rng.choice(idx_fake, n_bal, replace=False),
])
rng.shuffle(bal_idx)
print(f"Balanced: {len(bal_idx)} (real={n_bal}, fake={n_bal})")

y_bal     = y[bal_idx]
paths_bal = paths[bal_idx]

# ─────────────────────────────────────────────────────────────
# Normalise — per-block for 'both', joint for single modality
# ─────────────────────────────────────────────────────────────
def zscore(X_raw):
    mu  = X_raw.mean(0, keepdims=True)
    sig = X_raw.std(0,  keepdims=True) + 1e-8
    return (X_raw - mu) / sig, mu, sig


if args.features == "both":
    # Per-block: normalise dino and physics separately, then concatenate.
    # This prevents one block's scale from dominating the other.
    Xd_bal = X_d[bal_idx]
    Xp_bal = X_p[bal_idx]
    Xd_n, mu_d, sig_d = zscore(Xd_bal)
    Xp_n, mu_p, sig_p = zscore(Xp_bal)
    X_bal   = np.concatenate([Xd_n, Xp_n], axis=1)
    mean    = np.concatenate([mu_d, mu_p], axis=1)
    std     = np.concatenate([sig_d, sig_p], axis=1)
else:
    X_bal       = X[bal_idx]
    X_bal, mean, std = zscore(X_bal)

# ─────────────────────────────────────────────────────────────
# 70 / 15 / 15 split  (indices used for both blocks in late fusion)
# ─────────────────────────────────────────────────────────────
n      = len(X_bal)
idx_all = np.arange(n)
tr_idx, tmp_idx = train_test_split(idx_all, test_size=0.30, stratify=y_bal, random_state=42)
va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, stratify=y_bal[tmp_idx], random_state=42)

X_train, y_train = X_bal[tr_idx], y_bal[tr_idx]
X_val,   y_val   = X_bal[va_idx], y_bal[va_idx]
X_test,  y_test  = X_bal[te_idx], y_bal[te_idx]
p_test           = paths_bal[te_idx]

print(f"Train={len(tr_idx)}  Val={len(va_idx)}  Test={len(te_idx)}")

# ─────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        h1, h2 = max(64, in_dim * 4), max(32, in_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),    nn.ReLU(),
            nn.Linear(h2, 1),
        )
    def forward(self, x): return self.net(x)


def train_mlp(X_tr, y_tr, X_va, y_va):
    in_dim = X_tr.shape[1]
    mdl    = MLP(in_dim).to(device)
    crit   = nn.BCEWithLogitsLoss()
    opt    = torch.optim.Adam(mdl.parameters(), lr=args.lr)
    ds_tr  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_va  = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    ld_tr  = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    ld_va  = DataLoader(ds_va, batch_size=args.batch)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for ep in range(args.epochs):
        mdl.train(); run = 0.0
        for xb, yb in tqdm(ld_tr, desc=f"Epoch {ep+1}/{args.epochs}", leave=False):
            xb, yb = xb.to(device), yb.float().unsqueeze(1).to(device)
            opt.zero_grad(); loss = crit(mdl(xb), yb); loss.backward(); opt.step()
            run += loss.item() * xb.size(0)
        tl = run / len(ds_tr)

        mdl.eval(); vl = 0.0; vp_l, vy_l = [], []
        with torch.no_grad():
            for xb, yb in ld_va:
                out = mdl(xb.to(device))
                vl += crit(out, yb.float().unsqueeze(1).to(device)).item() * xb.size(0)
                vp_l.append(torch.sigmoid(out).cpu().numpy().ravel())
                vy_l.append(yb.numpy())
        vl /= len(ds_va)
        vp, vy = np.concatenate(vp_l), np.concatenate(vy_l)
        va  = accuracy_score(vy, (vp >= 0.5).astype(int))
        vf  = f1_score(vy,      (vp >= 0.5).astype(int))
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["val_acc"].append(va);    history["val_f1"].append(vf)
        print(f"Epoch {ep+1}  train={tl:.4f}  val={vl:.4f}  acc={va:.3f}  f1={vf:.3f}")

    return mdl, history


def probs_mlp(mdl, X):
    mdl.eval()
    ds  = TensorDataset(torch.from_numpy(X))
    ld  = DataLoader(ds, batch_size=256)
    out = []
    with torch.no_grad():
        for (xb,) in ld:
            out.append(torch.sigmoid(mdl(xb.to(device))).cpu().numpy().ravel())
    return np.concatenate(out)


def train_sklearn(X_tr, y_tr):
    if args.model == "logreg":
        mdl = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    else:  # xgb
        from xgboost import XGBClassifier
        mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", random_state=42)
    mdl.fit(X_tr, y_tr)
    return mdl


def probs_sklearn(mdl, X):
    return mdl.predict_proba(X)[:, 1]


def train_any(X_tr, y_tr, X_va, y_va):
    if args.model == "mlp":
        return train_mlp(X_tr, y_tr, X_va, y_va), None   # (model, history)
    else:
        return train_sklearn(X_tr, y_tr), None


def predict_probs(mdl, X):
    if args.model == "mlp":
        return probs_mlp(mdl, X)
    return probs_sklearn(mdl, X)


def find_best_tau(probs, labels):
    best_f1, best_tau = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        f = f1_score(labels, (probs >= t).astype(int))
        if f > best_f1:
            best_f1, best_tau = f, t
    return best_tau, best_f1


# ─────────────────────────────────────────────────────────────
# STEP B — early fusion (single model on concatenated features)
# STEP C — late fusion  (two models, combined probs, tuned α)
# ─────────────────────────────────────────────────────────────
history = None

if args.fusion == "late" and args.features == "both":
    # ── Step C: Late fusion ───────────────────────────────────
    # Separate the two normalised blocks back out
    Xd_tr = X_train[:, :dim_d];  Xp_tr = X_train[:, dim_d:]
    Xd_va = X_val[:,   :dim_d];  Xp_va = X_val[:,   dim_d:]
    Xd_te = X_test[:,  :dim_d];  Xp_te = X_test[:,  dim_d:]

    print("\n--- Late fusion: training DINOv2 model ---")
    mdl_d, hist_d = train_any(Xd_tr, y_train, Xd_va, y_val)
    print("--- Late fusion: training physics model ---")
    mdl_p, hist_p = train_any(Xp_tr, y_train, Xp_va, y_val)

    pd_val = predict_probs(mdl_d, Xd_va)
    pp_val = predict_probs(mdl_p, Xp_va)

    # Tune α on val set
    best_f1_alpha, best_alpha = 0.0, 0.5
    for alpha in np.linspace(0.0, 1.0, 101):
        p_comb = alpha * pd_val + (1 - alpha) * pp_val
        f = f1_score(y_val, (p_comb >= 0.5).astype(int))
        if f > best_f1_alpha:
            best_f1_alpha, best_alpha = f, alpha
    print(f"Best α={best_alpha:.2f}  val F1={best_f1_alpha:.3f}  "
          f"(α=DINOv2 weight, 1-α=physics weight)")

    pd_te  = predict_probs(mdl_d, Xd_te)
    pp_te  = predict_probs(mdl_p, Xp_te)
    test_logits = best_alpha * pd_te + (1 - best_alpha) * pp_te
    np.save(f"best_alpha_{prefix}.npy", best_alpha)

    # Save both sub-models
    if args.model == "mlp":
        torch.save(mdl_d.state_dict(), f"model_{prefix}_dino.pt")
        torch.save(mdl_p.state_dict(), f"model_{prefix}_phys.pt")
    else:
        pickle.dump(mdl_d, open(f"model_{prefix}_dino.pkl", "wb"))
        pickle.dump(mdl_p, open(f"model_{prefix}_phys.pkl", "wb"))

    # For threshold search use combined val probs
    best_tau, _ = find_best_tau(best_alpha * pd_val + (1-best_alpha) * pp_val, y_val)

else:
    # ── Step B: Early fusion (single model) ──────────────────
    mdl, history = train_any(X_train, y_train, X_val, y_val)

    val_probs   = predict_probs(mdl, X_val)
    best_tau, _ = find_best_tau(val_probs, y_val)

    test_logits = predict_probs(mdl, X_test)

    if args.model == "mlp":
        torch.save(mdl.state_dict(), f"model_{prefix}.pt")
    else:
        pickle.dump(mdl, open(f"model_{prefix}.pkl", "wb"))

print(f"Best τ*={best_tau:.3f}")

# ─────────────────────────────────────────────────────────────
# Evaluate on test set
# ─────────────────────────────────────────────────────────────
test_preds = (test_logits >= best_tau).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, test_preds, average="binary")
auc = roc_auc_score(y_test, test_logits)
acc = accuracy_score(y_test, test_preds)
cm  = confusion_matrix(y_test, test_preds)

print(f"\n=== Test [{prefix}] ===")
print(f"Acc={acc:.3f}  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
print("Confusion matrix (rows=true [0=fake,1=real], cols=pred):")
print(cm)

pd.DataFrame({
    "path": p_test, "true": y_test, "pred": test_preds, "prob": test_logits,
}).to_csv(f"test_predictions_{prefix}.csv", index=False)

np.save(f"mean_{prefix}.npy",     mean)
np.save(f"std_{prefix}.npy",      std)
np.save(f"best_tau_{prefix}.npy", best_tau)

# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────
# Training curves (only for MLP early fusion — sklearn has no epoch history)
if history is not None:
    ep = range(1, args.epochs + 1)
    fig, ax = plt.subplots()
    ax.plot(ep, history["train_loss"], label="Train loss")
    ax.plot(ep, history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Loss [{prefix}]"); ax.legend(); ax.grid(True)
    fig.savefig(PLOTS_DIR / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(ep, history["val_acc"], label="Val acc")
    ax.plot(ep, history["val_f1"],  label="Val F1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title(f"Metrics [{prefix}]"); ax.legend(); ax.grid(True)
    fig.savefig(PLOTS_DIR / "metrics_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

fpr, tpr, _ = roc_curve(y_test, test_logits)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title(f"ROC [{prefix}]"); ax.legend(); ax.grid(True)
fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Fake", "Pred Real"])
ax.set_yticklabels(["True Fake", "True Real"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title(f"Confusion Matrix [{prefix}]")
fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Plots → {PLOTS_DIR.resolve()}/")
