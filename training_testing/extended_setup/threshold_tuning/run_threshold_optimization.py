import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
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


def youden_threshold(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return thresholds[np.argmax(j)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--safe-threshold", type=float, default=0.65)
    args = parser.parse_args()

    X_all, y, generators, all_names = load_and_prepare()
    X_safe = safe_filter(X_all, y, generators, all_names, threshold=args.safe_threshold)

    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]

    n_neg = (y==0).sum()
    n_pos = (y==1).sum()
    spw = n_neg/n_pos

    print(f"safe features: {X_safe.shape[1]}D, samples: {len(y)}")
    print(f"\nPER-FOLD THRESHOLD CALIBRATION:")
    print(f"{'Generator':15s} | {'Youden t':>9s} | {'AUC':>6s} | {'Acc t=0.5':>9s} | {'Acc Youden':>10s} | {'Delta':>7s}")
    print("-" * 75)

    rows = []
    for holdout_gen in fake_gens:
        hold_idx = np.where(generators==holdout_gen)[0]
        rng = np.random.default_rng(42)
        n_te = min(len(hold_idx), len(real_idx)//3)
        te_real = rng.choice(real_idx, n_te, replace=False)
        tr_real = np.array([i for i in real_idx if i not in set(te_real)])
        tr_fake = np.where((y==0) & (generators!=holdout_gen))[0]
        train_pool = np.concatenate([tr_real, tr_fake])
        te = np.concatenate([te_real, hold_idx])

        # split train into train/val for threshold selection
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

        # find youden threshold on val
        val_probs = clf.predict_proba(sc.transform(X_safe[val]))[:, 1]
        optimal_t = youden_threshold(y[val], val_probs)

        # evaluate on test with both thresholds
        test_probs = clf.predict_proba(sc.transform(X_safe[te]))[:, 1]
        auc = roc_auc_score(y[te], test_probs)
        acc_05 = accuracy_score(y[te], (test_probs>=0.5).astype(int))
        acc_youden = accuracy_score(y[te], (test_probs>=optimal_t).astype(int))
        delta = acc_youden - acc_05

        print(f"  {holdout_gen:13s} | {optimal_t:9.3f} | {auc:6.3f} | {acc_05:9.3f} | {acc_youden:10.3f} | {delta:+7.3f}")
        rows.append({
            "holdout_gen": holdout_gen,
            "youden_t": optimal_t, "auc": auc,
            "acc_t05": acc_05, "acc_youden": acc_youden,
            "delta_acc": delta,
        })

    df = pd.DataFrame(rows)
    thresholds = df["youden_t"].values
    print(f"\n  Mean    | {thresholds.mean():.3f}+-{thresholds.std():.3f} | {df['auc'].mean():.3f} | {df['acc_t05'].mean():.3f}     | {df['acc_youden'].mean():.3f}      | {df['delta_acc'].mean():+.3f}")

    if args.save_csv:
        out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "threshold"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "threshold_optimization.csv", index=False)
        print(f"\nsaved to {out_dir / 'threshold_optimization.csv'}")


if __name__ == "__main__":
    main()
