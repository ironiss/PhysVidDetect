import warnings
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)




def load_h5(path):
    with h5py.File(path, "r") as f:
        X = f["features"][:].astype(np.float32)
        y = f["label"][:].astype(np.int64)
        if "path" in f:
            paths = f["path"][:].astype(str)
        else:
            paths = None

        if "feat_names" in f:
            names = f["feat_names"][:].astype(str)
        else:
            names = None
    return X, y, paths, names

def fill_nan(X):
    X = X.copy()
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return X


def extract_generator(video_paths):
    generators = []
    for p in video_paths:
        parts = Path(p).parts
        gen = "unknown"
        for i, part in enumerate(parts):
            if part.lower() in ("real", "fake") and i+1<len(parts):
                gen = parts[i+1]
                break
        generators.append(gen)
    return np.array(generators)



def extract_video_id(video_paths):
    return np.array([Path(p).stem for p in video_paths])



def merge_h5_by_filename(X1, y1, paths1, names1, X2, y2, paths2, names2):
    name1 = {Path(p).name: i for i, p in enumerate(paths1)}
    name2 = {Path(p).name: i for i, p in enumerate(paths2)}
    common = sorted(set(name1) & set(name2))
    if not common:
        raise ValueError("no common video paths between the two feature files")

    i1 = np.array([name1[n] for n in common])
    i2 = np.array([name2[n] for n in common])
    X = np.concatenate([X1[i1], X2[i2]], axis=1)
    y = y1[i1]
    paths = paths1[i1]
    if names1 is not None and names2 is not None:
        names = np.concatenate([names1, names2])
    else:
        names = None
    print(f"merged {len(common)} common videos: {X1.shape[1]}+{X2.shape[1]}={X.shape[1]} features")
    return X, y, paths, names



def align_by_filename(X1, y1, paths1, names1, X2, y2, paths2, names2):
    name1 = {Path(p).name: i for i, p in enumerate(paths1)}
    name2 = {Path(p).name: i for i, p in enumerate(paths2)}
    common = sorted(set(name1) & set(name2))
    if not common:
        raise ValueError("no common videos between the two feature files")

    i1 = np.array([name1[n] for n in common])
    i2 = np.array([name2[n] for n in common])
    print(f"aligned {len(common)} common videos")
    return (X1[i1], X2[i2], y1[i1], paths1[i1], names1, names2)



def safe_auc(col, y):
    if np.std(col)<1e-12:
        return 0.5
    a = roc_auc_score(y, col)
    return max(a, 1-a)



def basic_cleanup(X, y, names, nan_thresh=80, corr_thresh=0.9):
    nan_pct = np.isnan(X).mean(axis=0)*100
    keep = nan_pct<=nan_thresh
    dropped_nan = (~keep).sum()

    X = fill_nan(X)
    X, names = X[:, keep], names[keep]

    aucs = np.array([safe_auc(X[:, i], y) for i in range(X.shape[1])])
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0)
    to_drop = set()
    for i in range(len(names)):
        if i in to_drop:
            continue
        for j in range(i+1, len(names)):
            if j in to_drop:
                continue
            if abs(corr[i, j])>corr_thresh:
                if aucs[i]>=aucs[j]:
                    victim = j
                else:
                    victim = i
                to_drop.add(victim)

    keep2 = np.array([i not in to_drop for i in range(len(names))])
    dropped_corr = (~keep2).sum()
    X, names = X[:, keep2], names[keep2]

    print(f"cleanup: {dropped_nan} NaN, {dropped_corr} correlated -> {X.shape[1]} features")
    return X, names



def cleanup_features(X, y, names, nan_thresh=80, corr_thresh=0.9, auc_thresh=0.0):
    nan_pct = np.isnan(X).mean(axis=0)*100
    keep = nan_pct<=nan_thresh
    dropped_nan = names[~keep].tolist()

    X = fill_nan(X)
    X, names = X[:, keep], names[keep]

    aucs = np.array([safe_auc(X[:, i], y) for i in range(X.shape[1])])

    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0)
    to_drop = set()
    for i in range(len(names)):
        if i in to_drop:
            continue
        for j in range(i+1, len(names)):
            if j in to_drop:
                continue
            if abs(corr[i, j])>corr_thresh:
                if aucs[i]>=aucs[j]:
                    victim = j
                else:
                    victim = i
                to_drop.add(victim)

    keep2 = np.array([i not in to_drop for i in range(len(names))])
    dropped_corr = names[~keep2].tolist()
    X, names, aucs = X[:, keep2], names[keep2], aucs[keep2]

    keep3 = aucs>auc_thresh
    dropped_auc = names[~keep3].tolist()
    X, names, aucs = X[:, keep3], names[keep3], aucs[keep3]

    print(f"feature cleanup: {len(dropped_nan)} NaN, {len(dropped_corr)} correlated, "
          f"{len(dropped_auc)} low-AUC -> {X.shape[1]} features remaining")
    return X, names, aucs



def stratified_split(y, generators, video_ids, test_size=0.20, seed=42):
    """stratified train/test split by class+generator"""
    strat_labels = np.array([
        "real" if yi==1 else f"fake_{g}" for yi, g in zip(y, generators)
    ])
    n = len(y)
    idx = np.arange(n)

    unique_vids = np.unique(video_ids)
    if len(unique_vids)<n:
        vid_to_strat = {}
        for vi, sl in zip(video_ids, strat_labels):
            vid_to_strat[vi] = sl
        unique_strat = np.array([vid_to_strat[v] for v in unique_vids])
        vid_idx = np.arange(len(unique_vids))
        tr_v, te_v = train_test_split(vid_idx, test_size=test_size, stratify=unique_strat, random_state=seed)
        tr_vids = set(unique_vids[tr_v])
        te_vids = set(unique_vids[te_v])
        tr_idx = np.array([i for i in idx if video_ids[i] in tr_vids])
        te_idx = np.array([i for i in idx if video_ids[i] in te_vids])
    else:
        tr_idx, te_idx = train_test_split(idx, test_size=test_size, stratify=strat_labels, random_state=seed)

    return tr_idx, te_idx



def holdout_split(y, generators, holdout_gen, seed=42):
    """LOGO split -- train on other generators, test on holdout+real"""
    real_idx = np.where(y==1)[0]
    holdout_idx = np.where(generators==holdout_gen)[0]

    rng = np.random.default_rng(seed)
    n_test_real = min(len(holdout_idx), len(real_idx)//3)
    test_real_idx = rng.choice(real_idx, n_test_real, replace=False)
    test_real_set = set(test_real_idx)

    train_real_idx = np.array([i for i in real_idx if i not in test_real_set])
    train_fake_idx = np.where((y==0) & (generators != holdout_gen))[0]

    tr_idx = np.concatenate([train_real_idx, train_fake_idx])
    te_idx = np.concatenate([test_real_idx, holdout_idx])
    return tr_idx, te_idx



def train_and_eval(X_train, y_train, X_test, y_test, model):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model.fit(X_tr, y_train)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs>=0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    if len(np.unique(y_test))>1:
        auc = roc_auc_score(y_test, probs)
    else:
        auc = 0.5

    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "probs": probs, "preds": preds, "model": model, "scaler": scaler}



def eval_probs(y_true, probs, label=""):
    preds = (probs>=0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    bacc = balanced_accuracy_score(y_true, preds)
    if len(np.unique(y_true))>1:
        f1 = f1_score(y_true, preds)
    else:
        f1 = 0.0
    if len(np.unique(y_true))>1:
        auc = roc_auc_score(y_true, probs)
    else:
        auc = 0.5
    if label:
        print(f"  {label:25s} acc={acc:.3f} bacc={bacc:.3f} f1={f1:.3f} auc={auc:.3f}")
    return {"acc": acc, "bacc": bacc, "f1": f1, "auc": auc, "probs": probs, "preds": preds}



def train_branch(X_train, y_train, X_test, seed=42):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    mdl = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    mdl.fit(X_tr, y_train)
    probs_tr = mdl.predict_proba(X_tr)[:, 1]
    probs_te = mdl.predict_proba(X_te)[:, 1]
    return mdl, scaler, probs_tr, probs_te



def plot_roc(y_true, probs, title, path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_roc_multi(roc_data, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (yt, pr) in roc_data.items():
        fpr, tpr, _ = roc_curve(yt, pr)
        auc = roc_auc_score(yt, pr)
        ax.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_confusion(y_true, preds, title, path):
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Fake", "Pred Real"])
    ax.set_yticklabels(["True Fake", "True Real"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j]>cm.max()/2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_per_gen_bars(results_dict, metric, title, path):
    gens = list(results_dict.keys())
    vals = [results_dict[g][metric] for g in gens]
    fig, ax = plt.subplots(figsize=(8, max(3, len(gens)*0.4)))
    order = np.argsort(vals)[::-1]
    bars = ax.barh(range(len(gens)), [vals[i] for i in order], color="steelblue")
    ax.set_yticks(range(len(gens)))
    ax.set_yticklabels([gens[i] for i in order], fontsize=9)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    for bar, v in zip(bars, [vals[i] for i in order]):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
    
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_comparison_bars(rows, metric, title, path):
    labels = [r["method"] for r in rows]
    vals = [r[metric] for r in rows]
    order = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels)*0.4)))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    bars = ax.barh(range(len(labels)), [vals[i] for i in order], color=[colors[i % len(colors)] for i in range(len(labels))])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([labels[i] for i in order], fontsize=9)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    for bar, v in zip(bars, [vals[i] for i in order]):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_holdout_grouped(holdout_rows, fake_gens, path):
    x = np.arange(len(fake_gens))
    n = len(holdout_rows)
    width = 0.8/n
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(max(8, len(fake_gens)*1.5), 5))
    for i, row in enumerate(holdout_rows):
        vals = [row.get(g, 0.5) for g in fake_gens]
        offset = (i - n/2 + 0.5)*width
        ax.bar(x+offset, vals, width, label=row["method"], color=colors[i], edgecolor="k", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(fake_gens, rotation=30, ha="right")
    ax.set_ylabel("Holdout AUC")
    ax.set_title("Leave-one-generator-out AUC by method")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
