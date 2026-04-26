import argparse
import sys
import warnings
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import fill_nan, safe_auc, basic_cleanup



warnings.filterwarnings("ignore")

N_TEST_PER_GEN = 200
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

    lat_cols = [c for c in lat.columns if c not in meta]
    phy_cols = [c for c in phy.columns if c not in meta]
    cam_cols = [c for c in cam.columns if c not in meta]

    X_raw = np.hstack([lat[lat_cols].values.astype(np.float32), phy[phy_cols].values.astype(np.float32), cam[cam_cols].values.astype(np.float32)])
    all_names = np.array(lat_cols + phy_cols + cam_cols)
    X_clean, clean_names = basic_cleanup(X_raw, y, all_names)
    
    return X_clean, y, generators, clean_names, all_names


def safe_filter(X, y, generators, clean_names, threshold=0.65):
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
    safe_mask = gen_mean<=threshold
    
    return X[:, safe_mask], clean_names[safe_mask], safe_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-test-per-gen", type=int, default=N_TEST_PER_GEN)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent.parent / "saved_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_clean, y, generators, clean_names, _ = load_and_prepare()
    X_safe, safe_names, safe_mask = safe_filter(X_clean, y, generators, clean_names)

    fake_gens = sorted(set(generators[y==0]))
    print(f"features: {X_safe.shape[1]}D, samples: {len(y)}")

    rng = np.random.default_rng(42)
    test_idx = []

    for g in fake_gens:
        idx_g = np.where(generators==g)[0]
        n = min(args.n_test_per_gen, len(idx_g))
        test_idx.extend(rng.choice(idx_g, n, replace=False))
        print(f"{g}: {n} test from {len(idx_g)}")

    n_test_real = len(fake_gens) * args.n_test_per_gen
    real_idx = np.where(y==1)[0]
    test_real = rng.choice(real_idx, min(n_test_real, len(real_idx)), replace=False)
    test_idx.extend(test_real)
    print(f"Real: {len(test_real)} test")

    test_idx = np.array(test_idx)
    train_idx = np.array([i for i in range(len(y)) if i not in set(test_idx)])
    print(f"train: {len(train_idx)}, test: {len(test_idx)}")


    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_safe[train_idx])
    X_te = scaler.transform(X_safe[test_idx])

    n_pos = (y[train_idx]==1).sum()
    n_neg = (y[train_idx]==0).sum()
    params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "scale_pos_weight": n_neg/n_pos, "eval_metric": "logloss", "random_state": 42}

    clf = XGBClassifier(**params)
    clf.fit(X_tr, y[train_idx])


    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs>=0.5).astype(int)

    auc = roc_auc_score(y[test_idx], probs)
    acc = accuracy_score(y[test_idx], preds)
    f1 = f1_score(y[test_idx], preds)
    cm = confusion_matrix(y[test_idx], preds, labels=[0, 1])

    print(f"results: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    print(f"per-generator:")
    per_gen = {}
    for g in fake_gens:
        g_mask = generators[test_idx]==g
        real_te = y[test_idx]==1
        combined = g_mask | real_te
        if combined.sum()>0 and len(np.unique(y[test_idx][combined]))>1:
            g_auc = roc_auc_score(y[test_idx][combined], probs[combined])
            g_acc = accuracy_score(y[test_idx][combined], preds[combined])
            
            print(f"{g}: AUC={g_auc:.4f}, Acc={g_acc:.4f}")
            per_gen[g] = {"auc": float(g_auc), "acc": float(g_acc)}


    clf.save_model(str(out_dir / "final_model.json"))

    with open(out_dir / "final_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save(out_dir / "final_train_indices.npy", train_idx)
    np.save(out_dir / "final_test_indices.npy", test_idx)
    np.save(out_dir / "safe_feature_mask.npy", safe_mask)

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(safe_names.tolist(), f)

    metadata = {"hyperparameters": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in params.items()},"n_features": int(X_safe.shape[1]), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "overall": {"auc": float(auc), "acc": float(acc), "f1": float(f1)},
        "per_generator": per_gen, "confusion_matrix": cm.tolist()}


    with open(out_dir / "final_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"<TA-DAM DONE>, saved to {out_dir}")



if __name__ == "__main__":
    main()
