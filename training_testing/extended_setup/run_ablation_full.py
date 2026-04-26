import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import fill_nan, safe_auc, basic_cleanup

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
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



def load_data():
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

    y = lat["label"].values.astype(int)
    generators = np.array([extract_gen(p) for p in lat["path"]])
    for src in KNOWN_REAL:
        generators[generators==src] = "Real"

    target = KNOWN_FAKE | {"Real"}
    valid = np.array([g in target for g in generators])
    y, generators = y[valid], generators[valid]

    meta = {"path", "label", "_key", "video"}
    
    z0_cols = [c for c in lat.columns if c not in meta and not c.startswith("eps_")]
    eps_cols = [c for c in lat.columns if c.startswith("eps_")]
    phy_cols = [c for c in phy.columns if c not in meta]
    cam_cols = [c for c in cam.columns if c not in meta]

    X_z0 = lat[z0_cols].values[valid].astype(np.float32)
    X_eps = lat[eps_cols].values[valid].astype(np.float32)
    X_phy = phy[phy_cols].values[valid].astype(np.float32)
    X_cam = cam[cam_cols].values[valid].astype(np.float32)

    X_z0, _ = basic_cleanup(X_z0, y, np.array(z0_cols))
    X_eps, _ = basic_cleanup(X_eps, y, np.array(eps_cols))
    X_phy, _ = basic_cleanup(X_phy, y, np.array(phy_cols))
    X_cam, _ = basic_cleanup(X_cam, y, np.array(cam_cols))

    branches = {"latent": X_z0, "noise": X_eps, "physics": X_phy, "camera": X_cam}
    return branches, y, generators



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
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        
        probs = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs[holdout_gen] = roc_auc_score(y[te], probs)
    return aucs


def combine(branches, *names):
    return np.hstack([branches[n] for n in names])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    branches, y, generators = load_data()

    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]
    branch_names = list(branches.keys())

    print(f"samples: {len(y)} (fake={sum(y==0)}, real={sum(y==1)})")
    for name, X in branches.items():
        print(f"{name}: {X.shape[1]}D")

    all_results = []

    print("STANDALONE:")
    for name in branch_names:
        aucs = logo_eval(branches[name], y, generators, fake_gens, real_idx)
        mean_auc = np.mean(list(aucs.values()))
        print(f"{name:10s} mean={mean_auc:.3f}  min={min(aucs.values()):.3f}")
        all_results.append({"config": name, "mean": mean_auc, "min": min(aucs.values()), **aucs})

    print("PAIRWISE:")
    for a, b in combinations(branch_names, 2):
        X = combine(branches, a, b)
        aucs = logo_eval(X, y, generators, fake_gens, real_idx)
        mean_auc = np.mean(list(aucs.values()))
        print(f"{a}+{b:10s} mean={mean_auc:.3f}  min={min(aucs.values()):.3f}")
        all_results.append({"config": f"{a}+{b}", "mean": mean_auc, "min": min(aucs.values()), **aucs})

    print("TRIPLE:")
    for combo in combinations(branch_names, 3):
        X = combine(branches, *combo)
        label = "+".join(combo)
        aucs = logo_eval(X, y, generators, fake_gens, real_idx)
        mean_auc = np.mean(list(aucs.values()))
        print(f"{label:30s} mean={mean_auc:.3f}  min={min(aucs.values()):.3f}")
        all_results.append({"config": label, "mean": mean_auc, "min": min(aucs.values()), **aucs})



    print("FULL:")
    X_all = combine(branches, *branch_names)
    aucs = logo_eval(X_all, y, generators, fake_gens, real_idx)
    full_mean = np.mean(list(aucs.values()))
    full_min = min(aucs.values())
    print(f"all: mean={full_mean:.3f}  min={full_min:.3f}")
    all_results.append({"config": "all", "mean": full_mean, "min": full_min, **aucs})

    df = pd.DataFrame(all_results)


    print("MARGINAL (removed from full):")
    for removed in branch_names:
        remaining = [b for b in branch_names if b != removed]
        X = combine(branches, *remaining)
        aucs = logo_eval(X, y, generators, fake_gens, real_idx)
        mean_auc = np.mean(list(aucs.values()))
        delta = mean_auc - full_mean
        print(f"--{removed:10s} mean={mean_auc:.3f}  delta={delta:+.3f}  min={min(aucs.values()):.3f}")

    print("PAIRWISE SYNERGY:")
    standalone = {r["config"]: r["mean"] for r in all_results if "+" not in r["config"] and r["config"] != "all"}
    synergies = []
    for r in all_results:
        parts = r["config"].split("+")
        
        if len(parts) != 2:
            continue
        
        a_alone = standalone[parts[0]]
        b_alone = standalone[parts[1]]
        synergy = r["mean"] - max(a_alone, b_alone)
        synergies.append((r["config"], a_alone, b_alone, r["mean"], synergy))

    synergies.sort(key=lambda x: -x[4])
    for label, a, b, combo, syn in synergies:
        print(f"{label:20s}  A={a:.3f}  B={b:.3f}  A+B={combo:.3f}  synergy={syn:+.3f}")

    if args.save_csv:
        out_dir = Path(__file__).resolve().parent.parent / "results" / "ablation"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out = out_dir / "results_ablation.csv"
        df.to_csv(out, index=False)
        print(f"saved to {out}")

    print(f"<TA-DAM DONE>")


if __name__ == "__main__":
    main()
