import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import safe_auc, basic_cleanup

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent.parent
KNOWN_FAKE = {"DynamicCrafter", "SVD", "ZeroScope", "Pika", "Latte", "OpenSora", "VideoCrafter", "SEINE"}
KNOWN_REAL = {"GenVideo-Real", "GenVideo-Real-clean-3k", "Kinetics-400", "Kinetics-400-additional-4k", "UCF-101-7k"}


def extract_gen(p):
    parts = Path(p).parts
    for i, part in enumerate(parts):
        if part.lower() in ("real", "fake") and i+1<len(parts):
            return parts[i+1]
    for part in parts:
        if part in KNOWN_FAKE or part in KNOWN_REAL:
            return part
    return "unknown"


def load_and_prepare():
    lat = pd.read_csv(BASE / "feature_data" / "latent_noise_festures.csv")
    phy = pd.read_csv(BASE / "feature_data" / "object_based_features.csv")
    cam = pd.read_csv(BASE / "feature_data" / "camera_motion_features.csv")

    lat["_key"] = lat["path"].apply(lambda p: Path(p).name)
    phy["_key"] = phy["path"].apply(lambda p: Path(p).name)
    cam["_key"] = cam["video"].apply(lambda p: Path(p).name)

    common = set(lat["_key"]) & set(phy["_key"]) & set(cam["_key"])
    for df in [lat, phy, cam]:
        df.drop_duplicates("_key", inplace=True)
    lat = lat[lat["_key"].isin(common)].sort_values("_key").reset_index(drop=True)
    phy = phy[phy["_key"].isin(common)].sort_values("_key").reset_index(drop=True)
    cam = cam[cam["_key"].isin(common)].sort_values("_key").reset_index(drop=True)

    generators = np.array([extract_gen(p) for p in lat["path"]])
    for src in KNOWN_REAL:
        generators[generators==src] = "Real"

    target = KNOWN_FAKE | {"Real"}
    mask = np.array([g in target for g in generators])
    lat = lat[mask].reset_index(drop=True)
    phy = phy[mask].reset_index(drop=True)
    cam = cam[mask].reset_index(drop=True)
    generators = generators[mask]

    y = lat["label"].values.astype(int)
    meta = {"path", "label", "_key", "video"}

    lat_cols = [c for c in lat.columns if c not in meta]
    phy_cols = [c for c in phy.columns if c not in meta]
    cam_cols = [c for c in cam.columns if c not in meta]

    X_raw = np.hstack([
        lat[lat_cols].values.astype(np.float32),
        phy[phy_cols].values.astype(np.float32),
        cam[cam_cols].values.astype(np.float32),
    ])
    all_names = np.array(lat_cols + phy_cols + cam_cols)
    X_clean, clean_names = basic_cleanup(X_raw, y, all_names)
    return X_clean, y, generators, clean_names


def safe_filter(X, y, generators, all_names, threshold=0.65):
    fake_gens = sorted(set(generators[y==0]))
    X_fake = X[y==0]
    gen_fake = generators[y==0]

    pair_list = []
    for i, g1 in enumerate(fake_gens):
        for g2 in fake_gens[i+1:]:
            pmask = (gen_fake==g1) | (gen_fake==g2)
            y_pair = (gen_fake[pmask]==g1).astype(int)
            pair_list.append(np.array([safe_auc(X_fake[pmask, j], y_pair) for j in range(X.shape[1])]))

    gen_mean = np.mean(pair_list, axis=0)
    safe_idx = np.where(gen_mean<=threshold)[0]
    return X[:, safe_idx]


def find_threshold_youden(y_true, probs):
    """maximize sensitivity + specificity"""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return thresholds[np.argmax(tpr - fpr)]


def find_threshold_max_f1(y_true, probs):
    """maximize F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    return thresholds[np.argmax(f1s[:-1])]


def find_threshold_max_balanced_acc(y_true, probs):
    """maximize balanced accuracy"""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    tnr = 1 - fpr
    balanced = (tpr + tnr) / 2
    return thresholds[np.argmax(balanced)]


def find_threshold_min_error(y_true, probs):
    """minimize total misclassification"""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    pos_rate = y_true.mean()
    neg_rate = 1 - pos_rate
    error = fpr * neg_rate + (1 - tpr) * pos_rate
    return thresholds[np.argmin(error)]


METHODS = {
    "t=0.5 (default)": lambda y, p: 0.5,
    "Youden J": find_threshold_youden,
    "Max F1": find_threshold_max_f1,
    "Max Balanced Acc": find_threshold_max_balanced_acc,
    "Min Error": find_threshold_min_error,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--safe-threshold", type=float, default=0.65)
    args = parser.parse_args()

    X_all, y, generators, all_names = load_and_prepare()
    X_safe = safe_filter(X_all, y, generators, all_names, threshold=args.safe_threshold)

    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]
    spw = (y==0).sum() / (y==1).sum()

    print(f"safe features: {X_safe.shape[1]}D, samples: {len(y)}")

    all_rows = []

    for holdout_gen in fake_gens:
        hold_idx = np.where(generators==holdout_gen)[0]
        rng = np.random.default_rng(42)
        n_te = min(len(hold_idx), len(real_idx)//3)
        te_real = rng.choice(real_idx, n_te, replace=False)
        tr_real = np.array([i for i in real_idx if i not in set(te_real)])
        tr_fake = np.where((y==0) & (generators!=holdout_gen))[0]
        train_pool = np.concatenate([tr_real, tr_fake])
        te = np.concatenate([te_real, hold_idx])

        strat = np.array(["r" if y[i]==1 else f"f_{generators[i]}" for i in train_pool])
        pool_idx = np.arange(len(train_pool))
        tr_local, val_local = train_test_split(pool_idx, test_size=0.2, stratify=strat, random_state=42)
        tr = train_pool[tr_local]
        val = train_pool[val_local]

        sc = StandardScaler()
        clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=spw, eval_metric="logloss", random_state=42)
        clf.fit(sc.fit_transform(X_safe[tr]), y[tr])

        val_probs = clf.predict_proba(sc.transform(X_safe[val]))[:, 1]
        test_probs = clf.predict_proba(sc.transform(X_safe[te]))[:, 1]
        auc = roc_auc_score(y[te], test_probs)

        print(f"\n{holdout_gen} (AUC={auc:.3f}):")
        for method_name, method_fn in METHODS.items():
            if method_name == "t=0.5 (default)":
                t = 0.5
            else:
                t = method_fn(y[val], val_probs)

            acc = accuracy_score(y[te], (test_probs>=t).astype(int))
            f1 = f1_score(y[te], (test_probs>=t).astype(int))
            print(f"  {method_name:25s} t={t:.3f}  acc={acc:.3f}  f1={f1:.3f}")

            all_rows.append({
                "holdout_gen": holdout_gen, "method": method_name,
                "threshold": t, "auc": auc, "acc": acc, "f1": f1,
            })

    df = pd.DataFrame(all_rows)

    print(f"\nSUMMARY (mean across generators):")
    print(f"{'Method':25s} | {'Mean t':>8s} | {'Mean Acc':>8s} | {'Mean F1':>8s}")
    print("-" * 60)
    for method_name in METHODS:
        mdf = df[df["method"]==method_name]
        print(f"  {method_name:23s} | {mdf['threshold'].mean():8.3f} | {mdf['acc'].mean():8.3f} | {mdf['f1'].mean():8.3f}")

    if args.save_csv:
        out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "threshold"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "threshold_methods_comparison.csv", index=False)
        print(f"\nsaved to {out_dir / 'threshold_methods_comparison.csv'}")


if __name__ == "__main__":
    main()
