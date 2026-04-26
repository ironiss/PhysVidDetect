import argparse
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import safe_auc, basic_cleanup


warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent

KNOWN_FAKE = {"DynamicCrafter", "SVD", "ZeroScope", "Pika", "Latte", "OpenSora", "VideoCrafter", "SEINE"}
KNOWN_REAL = {"GenVideo-Real", "GenVideo-Real-clean-3k", "Kinetics-400", "Kinetics-400-additional-4k", "UCF-101-7k"}
GEN_SHORT = {"DynamicCrafter": "DC", "SVD": "SVD", "ZeroScope": "ZS", "Pika": "Pika", "Latte": "Latte", "OpenSora": "OSora", "VideoCrafter": "VCraft", "SEINE": "SEINE"}

XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)

DEFAULT_SPLITS = [["DynamicCrafter", "SVD"],
["OpenSora", "Latte"],
["VideoCrafter", "Pika"],
["ZeroScope", "SEINE"]]



def extract_gen(p):
    parts = Path(p).parts

    for i, part in enumerate(parts):
        if part.lower() in ("real", "fake") and i+1 < len(parts):
            return parts[i+1]

    for part in parts:
        if part in KNOWN_FAKE or part in KNOWN_REAL:
            return part

    return "unknown"



def load_and_prepare(no_latent=False):
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
        generators[generators == src] = "Real"

    mask = np.array([g in (KNOWN_FAKE | {"Real"}) for g in generators])
    lat, phy, cam = lat[mask].reset_index(drop=True), phy[mask].reset_index(drop=True), cam[mask].reset_index(drop=True)
    generators = generators[mask]

    y = lat["label"].values.astype(int)
    meta = {"path", "label", "_key", "video"}
    phy_cols = [c for c in phy.columns if c not in meta]
    cam_cols = [c for c in cam.columns if c not in meta]
    eps_cols = [c for c in lat.columns if c not in meta and c.startswith("eps_")]
    X_eps, _ = basic_cleanup(lat[eps_cols].values.astype(np.float32), y, np.array(eps_cols))
    X_phy, _ = basic_cleanup(phy[phy_cols].values.astype(np.float32), y, np.array(phy_cols))
    X_cam, _ = basic_cleanup(cam[cam_cols].values.astype(np.float32), y, np.array(cam_cols))

    if no_latent:
        return np.hstack([X_eps, X_phy, X_cam]), y, generators

    z0_cols = [c for c in lat.columns if c not in meta and not c.startswith("eps_")]
    X_z0, _ = basic_cleanup(lat[z0_cols].values.astype(np.float32), y, np.array(z0_cols))

    return np.hstack([X_z0, X_eps, X_phy, X_cam]), y, generators




def eval_unseen(X, y, generators, train_gens):
    train_set = set(train_gens)
    test_gens = [g for g in sorted(KNOWN_FAKE) if g not in train_set]
    real_idx = np.where(y==1)[0]

    train_fake = np.where(np.array([g in train_set for g in generators]) & (y==0))[0]
    test_fake = np.where(np.array([g in set(test_gens) for g in generators]) & (y==0))[0]
    real_train, real_test = train_test_split(real_idx, test_size=0.3, random_state=42)

    tr = np.concatenate([train_fake, real_train])
    te = np.concatenate([test_fake, real_test])

    n_pos, n_neg = (y[tr]==1).sum(), (y[tr]==0).sum()
    clf = XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg/n_pos)

    sc = StandardScaler()
    clf.fit(sc.fit_transform(X[tr]), y[tr])
    probs = clf.predict_proba(sc.transform(X[te]))[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {"protocol": f"{len(train_gens)}->N (unseen)", "train": "+".join(GEN_SHORT[g] for g in train_gens), "auc": roc_auc_score(y[te], probs), "acc": accuracy_score(y[te], preds)}



def eval_seen(X, y, generators, train_gens):
    gen_set = set(train_gens)
    gen_mask = np.array([g in gen_set or y[i]==1 for i, g in enumerate(generators)])
    X_sub, y_sub = X[gen_mask], y[gen_mask]
    gen_sub = generators[gen_mask]

    strat = np.array([f"{'r' if yi==1 else 'f'}_{gen_sub[i]}" for i, yi in enumerate(y_sub)])
    idx_all = np.arange(len(y_sub))
    tr_idx, tmp_idx = train_test_split(idx_all, test_size=0.30, stratify=strat, random_state=42)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=strat[tmp_idx], random_state=42)
    trainval = np.concatenate([tr_idx, val_idx])

    n_pos, n_neg = (y_sub[trainval]==1).sum(), (y_sub[trainval]==0).sum()
    clf = XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg/n_pos)

    sc = StandardScaler()
    clf.fit(sc.fit_transform(X_sub[trainval]), y_sub[trainval])
    probs = clf.predict_proba(sc.transform(X_sub[te_idx]))[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {"protocol": f"N->N (seen, {len(train_gens)} gen)", "train": "+".join(GEN_SHORT[g] for g in train_gens), "auc": roc_auc_score(y_sub[te_idx], probs), "acc": accuracy_score(y_sub[te_idx], preds)}




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-latent", action="store_true")
    ap.add_argument("--all-pairs", action="store_true", help="all C(8,2)=28 pairs")
    ap.add_argument("--save-csv", action="store_true")
    args = ap.parse_args()

    X, y, generators = load_and_prepare(no_latent=args.no_latent)
    print(f"features: {X.shape[1]}D, samples: {len(y)}")

    if args.all_pairs:
        two_gen_splits = [list(pair) for pair in combinations(sorted(KNOWN_FAKE), 2)]
    else:
        two_gen_splits = DEFAULT_SPLITS



    results = []
    for train_gens in two_gen_splits:
        r = eval_unseen(X, y, generators, train_gens)
        print(f"[{r['train']}]: AUC={r['auc']:.3f}, Acc={r['acc']:.3f}")

        results.append(r)

    if len(two_gen_splits) > 1:
        unseen_aucs = [r["auc"] for r in results]
        unseen_accs = [r["acc"] for r in results]
        print(f"mean AUC: {np.mean(unseen_aucs):.3f}, mean Acc: {np.mean(unseen_accs):.3f}")



    for train_gens in two_gen_splits:
        r = eval_seen(X, y, generators, train_gens)
        print(f"[{r['train']}]: AUC={r['auc']:.3f}, Acc={r['acc']:.3f}")
        results.append(r)



    if args.save_csv:
        out = Path(__file__).resolve().parent.parent / "results" / "comparison_protocols"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out / "two_to_many.csv", index=False)
        print(f"saved to {out / 'two_to_many.csv'}")

    print(f"<TA-DAM DONE>")



if __name__ == "__main__":
    main()
