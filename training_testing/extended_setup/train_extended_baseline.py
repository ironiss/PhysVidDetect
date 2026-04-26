import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score #, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import fill_nan, safe_auc, basic_cleanup

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
KNOWN_FAKE = {"DynamicCrafter", "SVD", "ZeroScope", "Pika", "Latte", "OpenSora", "VideoCrafter", "SEINE"}
KNOWN_REAL = {"GenVideo-Real", "GenVideo-Real-clean-3k", "Kinetics-400", "Kinetics-400-additional-4k", "UCF-101-7k"}


ALL_CONFIGS = {
    "physics": ["physics"],
    "latent": ["latent"],
    "noise": ["noise"],
    "camera": ["camera"],
    "phys+lat": ["physics", "latent"],
    "phys+lat+noise": ["physics", "latent", "noise"],
    "all": ["physics", "latent", "noise", "camera"],
}

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

    z0_cols = [c for c in lat.columns if c not in meta and not c.startswith("eps_")]
    eps_cols = [c for c in lat.columns if c.startswith("eps_")]
    phy_cols = [c for c in phy.columns if c not in meta]
    cam_cols = [c for c in cam.columns if c not in meta]

    X_z0 = lat[z0_cols].values.astype(np.float32)
    X_eps = lat[eps_cols].values.astype(np.float32)
    X_phy = phy[phy_cols].values.astype(np.float32)
    X_cam = cam[cam_cols].values.astype(np.float32)

    X_z0_c, _ = basic_cleanup(X_z0, y, np.array(z0_cols))
    X_eps_c, _ = basic_cleanup(X_eps, y, np.array(eps_cols))
    X_phy_c, _ = basic_cleanup(X_phy, y, np.array(phy_cols))
    X_cam_c, _ = basic_cleanup(X_cam, y, np.array(cam_cols))

    branches = {"latent": X_z0_c, "noise": X_eps_c, "physics": X_phy_c, "camera": X_cam_c}
    return branches, y, generators


def seen_eval(X, y, test_size=0.20):
    tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=42)
    sc = StandardScaler()
    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    clf.fit(sc.fit_transform(X[tr]), y[tr])
    probs = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y[te], probs)


def logo_eval(X, y, generators, fake_gens, real_idx):
    aucs = {}
    for holdout_gen in fake_gens:
        hold_idx = np.where(generators==holdout_gen)[0]
        rng = np.random.default_rng(42)
        n_te = min(len(hold_idx), len(real_idx)//3)
        te_real = rng.choice(real_idx, n_te, replace=False)
        tr_real = np.array([i for i in real_idx if i not in set(te_real)])
        tr_fake = np.where((y==0) & (generators!=holdout_gen))[0]
        tr = np.concatenate([tr_real, tr_fake])
        te = np.concatenate([te_real, hold_idx])

        sc = StandardScaler()
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        probs = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs[holdout_gen] = roc_auc_score(y[te], probs)
    
    return aucs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="which configs to run (e.g. physics latent phys+lat all). Default: all configs")
    parser.add_argument("--custom", nargs="*", default=None,
                        help="custom branch combo (e.g. --custom latent camera)")
    args = parser.parse_args()

    branches, y, generators = load_and_prepare()
    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]

    print(f"samples: {len(y)} (fake={sum(y==0)}, real={sum(y==1)})")
    print(f"branches: {', '.join(f'{n}={X.shape[1]}D' for n, X in branches.items())}")

    if args.custom:
        configs = {"+".join(args.custom): args.custom}
    elif args.configs:
        configs = {k: ALL_CONFIGS[k] for k in args.configs if k in ALL_CONFIGS}
    else:
        configs = ALL_CONFIGS

    results = []

    print("--SEEN--")
    for label, branch_list in configs.items():
        X = np.concatenate([branches[b] for b in branch_list], axis=1)
        auc = seen_eval(X, y)
        print(f"{label:30s} {X.shape[1]:3d}D  AUC={auc:.3f}")
        results.append({"config": label, "n": X.shape[1], "seen_auc": auc})

    print("--LOGO--")
    for label, branch_list in configs.items():
        X = np.concatenate([branches[b] for b in branch_list], axis=1)
        aucs = logo_eval(X, y, generators, fake_gens, real_idx)
        mean_auc = np.mean(list(aucs.values()))
        min_auc = min(aucs.values())
        parts = "  ".join(f"{g}={a:.3f}" for g, a in aucs.items())
        
        print(f"{label:30s} {X.shape[1]:3d}D  mean={mean_auc:.3f}  min={min_auc:.3f}  {parts}")

        for r in results:
            if r["config"]==label:
                r["logo_mean"] = mean_auc
                r["logo_min"] = min_auc
                r.update(aucs)

    if args.save_csv:
        out_dir = Path(__file__).resolve().parent.parent / "results" / "extended_baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "extended_baseline.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"\nsaved to {out}")

    print(f"<TA-DAM DONE>")



if __name__ == "__main__":
    main()
