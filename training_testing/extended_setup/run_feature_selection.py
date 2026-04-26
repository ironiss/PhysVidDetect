import argparse
import sys
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


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

    X_all = np.concatenate([X_z0_c, X_phy_c, X_eps_c, X_cam_c], axis=1)
    all_names = np.concatenate([np.array([f"z0_{c}" for c in range(X_z0_c.shape[1])]), np.array([f"phy_{c}" for c in range(X_phy_c.shape[1])]), np.array([f"eps_{c}" for c in range(X_eps_c.shape[1])]), np.array([f"cam_{c}" for c in range(X_cam_c.shape[1])])])

    return X_all, y, generators, all_names



def compute_rankings(X, y, generators, fake_gens):
    is_real = (y==1)
    X_fake = X[y==0]
    gen_fake = generators[y==0]
    n_feats = X.shape[1]

    global_auc = np.array([safe_auc(X[:, fi], y) for fi in range(n_feats)])


    pair_list = []
    for i, g1 in enumerate(fake_gens):
        for g2 in fake_gens[i+1:]:
            pmask = (gen_fake==g1) | (gen_fake==g2)
            y_pair = (gen_fake[pmask]==g1).astype(int)
            pair_list.append(np.array([safe_auc(X_fake[pmask, j], y_pair) for j in range(n_feats)]))
    
    gen_mean_pw = np.mean(pair_list, axis=0)
    gen_max_pw = np.max(pair_list, axis=0)


    gen_aucs = []
    for g in fake_gens:
        combined = is_real | (generators==g)
        gen_aucs.append(np.array([safe_auc(X[combined, fi], y[combined]) for fi in range(n_feats)]))
    
    gen_aucs = np.array(gen_aucs)
    stability = gen_aucs.mean(axis=0) - gen_aucs.std(axis=0)

    return global_auc, gen_mean_pw, gen_max_pw, stability


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


def eval_and_print(X, y, generators, fake_gens, real_idx, label, n):
    aucs = logo_eval(X, y, generators, fake_gens, real_idx)
    mean_v = np.mean(list(aucs.values()))
    min_v = min(aucs.values())
    parts = "  ".join(f"{g}={a:.3f}" for g, a in aucs.items())
    print(f"{label:35s} n={n:3d}  mean={mean_v:.3f}  min={min_v:.3f}  {parts}")
    row = {"method": label, "n": n, "mean": mean_v, "min": min_v}
    row.update(aucs)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--best-only", action="store_true", help="only run best 5 configs for main table")
    args = parser.parse_args()

    X_all, y, generators, _ = load_and_prepare()
    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]

    print(f"features: {X_all.shape[1]}D, samples: {len(y)}")
    global_auc, gen_mean_pw, gen_max_pw, stability = compute_rankings(X_all, y, generators, fake_gens)
    results = []

    print("Baseline:")
    results.append(eval_and_print(X_all, y, generators, fake_gens, real_idx, "All cleaned", X_all.shape[1]))

    if args.best_only:
        safe_idx = np.where(gen_mean_pw<=0.65)[0]
        results.append(eval_and_print(X_all[:, safe_idx], y, generators, fake_gens, real_idx, "Safe (gen_mean<=0.65)", len(safe_idx)))

        top = np.argsort(-stability)[:20]
        results.append(eval_and_print(X_all[:, top], y, generators, fake_gens, real_idx, "Stable top-20", 20))

        soft = global_auc - 0.7*gen_mean_pw
        top = np.argsort(-soft)[:50]
        results.append(eval_and_print(X_all[:, top], y, generators, fake_gens, real_idx, "Soft (lambda=0.7) top-50", 50))

        top = np.argsort(-global_auc)[:20]
        results.append(eval_and_print(X_all[:, top], y, generators, fake_gens, real_idx, "Global top-20", 20))

    else:
        print("Global top-K:")
        for k in [20, 30, 40, 50, 60, 80]:
            top = np.argsort(-global_auc)[:k]
            results.append(eval_and_print(X_all[:, top], y, generators, fake_gens, real_idx, f"Global top-{k}", len(top)))

        print("Stable top-K:")
        for k in [20, 30, 40, 50, 60, 80]:
            top = np.argsort(-stability)[:k]
            results.append(eval_and_print(X_all[:, top], y, generators, fake_gens, real_idx, f"Stable top-{k}", len(top)))

        print("Safe (pairwise gen_mean):")
        for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
            safe_idx = np.where(gen_mean_pw<=thresh)[0]
            if len(safe_idx)>0:
                results.append(eval_and_print(X_all[:, safe_idx], y, generators, fake_gens, real_idx, f"Safe (<={thresh})", len(safe_idx)))

        print("Safe_max (pairwise gen_max):")
        for thresh in [0.80]:
            safe_idx = np.where(gen_max_pw<=thresh)[0]
            if len(safe_idx)>0:
                results.append(eval_and_print(X_all[:, safe_idx], y, generators, fake_gens, real_idx, f"Safe_max (<={thresh})", len(safe_idx)))

        print("Soft scoring:")
        for lam in [0.7, 1.0]:
            soft = global_auc - lam*gen_mean_pw
            top = np.argsort(-soft)[:50]
            results.append(eval_and_print(X_all[:, top], y, generators, fake_gens, real_idx, f"Soft (lambda={lam}) top-50", 50))

    if args.save_csv:
        out_dir = Path(__file__).resolve().parent.parent / "results" / "feature_selection"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = "best" if args.best_only else "full"
        out = out_dir / f"feature_selection_{suffix}.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"saved to {out}")

    print(f"<TA-DAM DONE>")




if __name__ == "__main__":
    main()
