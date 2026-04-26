import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import safe_auc, basic_cleanup

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
KNOWN_FAKE = {"DynamicCrafter", "SVD", "ZeroScope", "Pika", "Latte", "OpenSora", "VideoCrafter", "SEINE"}
KNOWN_REAL = {"GenVideo-Real", "GenVideo-Real-clean-3k", "Kinetics-400", "Kinetics-400-additional-4k", "UCF-101-7k"}
GEN_SHORT = {"DynamicCrafter": "DC", "SVD": "SVD", "ZeroScope": "ZS", "Pika": "Pika", "Latte": "Latte", "OpenSora": "OSora", "VideoCrafter": "VCraft", "SEINE": "SEINE"}

BEST = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss"}




def extract_gen(p):
    parts = Path(p).parts
    for i, part in enumerate(parts):
        if part.lower() in ("real", "fake") and i+1<len(parts):
            return parts[i+1]
    for part in parts:
        if part in KNOWN_FAKE or part in KNOWN_REAL:
            return part
    return "unknown"



def load_and_safe_filter():
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
    lat, phy, cam = lat[mask].reset_index(drop=True), phy[mask].reset_index(drop=True), cam[mask].reset_index(drop=True)
    generators = generators[mask]

    y = lat["label"].values.astype(int)
    meta = {"path", "label", "_key", "video"}
    X_raw = np.hstack([lat[[c for c in lat.columns if c not in meta]].values.astype(np.float32), phy[[c for c in phy.columns if c not in meta]].values.astype(np.float32), cam[[c for c in cam.columns if c not in meta]].values.astype(np.float32)])
    all_names = np.array([c for c in lat.columns if c not in meta] + [c for c in phy.columns if c not in meta] + [c for c in cam.columns if c not in meta])
    X_clean, _ = basic_cleanup(X_raw, y, all_names)

    fake_gens = sorted(set(generators[y==0]))
    X_fake = X_clean[y==0]
    gen_fake = generators[y==0]
    pair_list = []

    for i, g1 in enumerate(fake_gens):
        for g2 in fake_gens[i+1:]:
            pmask = (gen_fake==g1) | (gen_fake==g2)
            y_pair = (gen_fake[pmask]==g1).astype(int)
            pair_list.append(np.array([safe_auc(X_fake[pmask, j], y_pair) for j in range(X_clean.shape[1])]))
    
    gen_mean = np.mean(pair_list, axis=0)
    return X_clean[:, gen_mean<=0.65], y, generators



def build_fold(y, generators, holdout_gen):
    real_idx = np.where(y==1)[0]
    hold_idx = np.where(generators==holdout_gen)[0]
    rng = np.random.default_rng(42)
    n_te = min(len(hold_idx), len(real_idx)//3)
    te_real = rng.choice(real_idx, n_te, replace=False)
    tr_real = np.array([i for i in real_idx if i not in set(te_real)])
    tr_fake = np.where((y==0) & (generators!=holdout_gen))[0]
    return np.concatenate([tr_real, tr_fake]), np.concatenate([te_real, hold_idx])



if __name__ == "__main__":
    X, y, generators = load_and_safe_filter()
    fake_gens = sorted(set(generators[y==0]))
    print(f"Safe: {X.shape[1]}D, samples: {len(y)}")

    print("TRAIN-TEST GAP:")
    gap_results = []
    for g in fake_gens:
        tr, te = build_fold(y, generators, g)
        sc = StandardScaler()
        n_pos, n_neg = (y[tr]==1).sum(), (y[tr]==0).sum()
        clf = XGBClassifier(**BEST, scale_pos_weight=n_neg/n_pos, random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        train_auc = roc_auc_score(y[tr], clf.predict_proba(sc.fit_transform(X[tr]))[:, 1])
        test_auc = roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1])
        gap = train_auc - test_auc
        gap_results.append({"gen": GEN_SHORT[g], "train": train_auc, "test": test_auc, "gap": gap})
        print(f"{GEN_SHORT[g]:>8s}: train={train_auc:.4f} test={test_auc:.4f} gap={gap:+.4f}")

    gaps = [r["gap"] for r in gap_results]
    max_gap = max(gaps)
    print(f"mean gap: {np.mean(gaps):+.4f}, max gap: {max_gap:+.4f}")
    if max_gap<0.05:
        print("-> no overfitting")
    elif max_gap<0.10:
        print("-> minimal overfitting")
    else:
        print("-> possible overfitting")


    print("SEED STABILITY:")
    seeds = [42, 7, 123, 2024, 9999]
    seed_results = []
    for seed in seeds:
        aucs = {}
        for g in fake_gens:
            tr, te = build_fold(y, generators, g)
            sc = StandardScaler()
            n_pos, n_neg = (y[tr]==1).sum(), (y[tr]==0).sum()
            clf = XGBClassifier(**BEST, scale_pos_weight=n_neg/n_pos, random_state=seed)
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            aucs[g] = roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1])
        mean_auc = np.mean(list(aucs.values()))
        min_auc = min(aucs.values())
        seed_results.append({"seed": seed, "mean_auc": mean_auc, "min_auc": min_auc})
        print(f"seed={seed:>5d}: mean={mean_auc:.4f} min={min_auc:.4f}")

    means = [r["mean_auc"] for r in seed_results]
    std_val = np.std(means)
    print(f"std: {std_val:.4f}")
    if std_val<0.003:
        print("-> very stable")
    elif std_val<0.005:
        print("-> stable")
    else:
        print("-> somewhat variable")


    print("LEARNING CURVES:")
    lc_results = []
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        aucs = {}
        for g in fake_gens:
            tr, te = build_fold(y, generators, g)
            rng = np.random.default_rng(42)
            tr_pos = tr[y[tr]==1]
            tr_neg = tr[y[tr]==0]
            sel = np.concatenate([rng.choice(tr_pos, int(len(tr_pos)*frac), replace=False), rng.choice(tr_neg, int(len(tr_neg)*frac), replace=False)])
            sc = StandardScaler()
            n_pos, n_neg = (y[sel]==1).sum(), (y[sel]==0).sum()
            clf = XGBClassifier(**BEST, scale_pos_weight=n_neg/n_pos, random_state=42)
            clf.fit(sc.fit_transform(X[sel]), y[sel])
            aucs[g] = roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1])
        mean_auc = np.mean(list(aucs.values()))
        min_auc = min(aucs.values())
        lc_results.append({"fraction": frac, "n_train": len(sel), "mean_auc": mean_auc, "min_auc": min_auc})
        print(f"frac={frac:.1f} (n={len(sel)}): mean={mean_auc:.4f} min={min_auc:.4f}")

    delta = lc_results[-1]["mean_auc"] - lc_results[-2]["mean_auc"]
    print(f"delta (100% vs 80%): {delta:+.4f}")
    if abs(delta)<0.003:
        print("-> converged (more data won't help)")
    elif abs(delta)<0.005:
        print("-> nearly converged")
    else:
        print("-> still improving (more data could help)")

    out_dir = Path(__file__).resolve().parent.parent / "results" / "overfit"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gap_results).to_csv(out_dir / "overfit_gap.csv", index=False)
    pd.DataFrame(seed_results).to_csv(out_dir / "overfit_seed.csv", index=False)
    pd.DataFrame(lc_results).to_csv(out_dir / "overfit_lc.csv", index=False)
    print(f"saved to {out_dir}")
