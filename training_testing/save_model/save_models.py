import argparse
import sys
import warnings
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else BASE / "saved_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logo").mkdir(exist_ok=True)

    X_clean, y, generators, clean_names, all_names = load_and_prepare()
    X_safe, safe_names, safe_mask = safe_filter(X_clean, y, generators, clean_names)

    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]

    params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss", "random_state": 42}

    print(f"features: {X_safe.shape[1]}D, samples: {len(y)}")



    print("PRODUCTION MODEL (all data):")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_safe)
    n_pos, n_neg = (y==1).sum(), (y==0).sum()
    clf = XGBClassifier(**params, scale_pos_weight=n_neg/n_pos)
    clf.fit(X_scaled, y)

    train_auc = roc_auc_score(y, clf.predict_proba(X_scaled)[:, 1])
    print(f"train AUC (in-sample): {train_auc:.4f}")

    clf.save_model(str(out_dir / "production_model.json"))
    with open(out_dir / "production_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save(out_dir / "safe_feature_mask.npy", safe_mask)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump({"all_features": all_names.tolist(), "safe_features": safe_names.tolist()}, f, indent=2)


    print("LOGO MODELS:")
    logo_aucs = {}
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
        n_pos_f = (y[tr]==1).sum()
        n_neg_f = (y[tr]==0).sum()
        clf_f = XGBClassifier(**params, scale_pos_weight=n_neg_f/n_pos_f)
        clf_f.fit(sc.fit_transform(X_safe[tr]), y[tr])

        probs = clf_f.predict_proba(sc.transform(X_safe[te]))[:, 1]
        auc = roc_auc_score(y[te], probs)
        logo_aucs[holdout_gen] = auc

        clf_f.save_model(str(out_dir / "logo" / f"model_holdout_{holdout_gen}.json"))
        
        with open(out_dir / "logo" / f"scaler_holdout_{holdout_gen}.pkl", "wb") as f:
            pickle.dump(sc, f)
        
        print(f"{holdout_gen}: AUC={auc:.4f}")

    print(f"mean AUC: {np.mean(list(logo_aucs.values())):.4f}")
    print(f"min AUC: {min(logo_aucs.values()):.4f}")



    metadata = {"hyperparameters": params, "n_features": int(X_safe.shape[1]), "n_samples": int(len(y)), "logo_performance": {g: float(a) for g, a in logo_aucs.items()}, "mean_logo_auc": float(np.mean(list(logo_aucs.values())))}
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"<TA-DAM DONE>, saved to {out_dir}")



if __name__ == "__main__":
    main()
