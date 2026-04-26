import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import fill_nan, safe_auc, basic_cleanup

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

    X_raw = np.hstack([lat[lat_cols].values.astype(np.float32), phy[phy_cols].values.astype(np.float32), cam[cam_cols].values.astype(np.float32)])
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
    return X[:, safe_idx], all_names[safe_idx]


def get_models():
    return {
        "XGBoost (scale_pos_weight)": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.0, eval_metric="logloss", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    }


def logo_eval_model(X, y, generators, model_name, clf_template):
    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]
    per_gen = {}

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
        clf = clone(clf_template)

        if model_name == "XGBoost (scale_pos_weight)":
            clf.set_params(scale_pos_weight=(y[tr]==0).sum()/(y[tr]==1).sum())

        clf.fit(sc.fit_transform(X[tr]), y[tr])
        probs = clf.predict_proba(sc.transform(X[te]))[:, 1]
        preds = (probs>=0.5).astype(int)

        per_gen[holdout_gen] = {"auc": roc_auc_score(y[te], probs), "acc": accuracy_score(y[te], preds),"f1": f1_score(y[te], preds)}

    aucs = [v["auc"] for v in per_gen.values()]
    accs = [v["acc"] for v in per_gen.values()]
    f1s = [v["f1"] for v in per_gen.values()]

    return {"mean_auc": np.mean(aucs), "min_auc": min(aucs), "mean_acc": np.mean(accs), "mean_f1": np.mean(f1s), "per_gen": per_gen}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--safe-threshold", type=float, default=0.65)
    args = parser.parse_args()

    X_all, y, generators, all_names = load_and_prepare()
    X_safe, _ = safe_filter(X_all, y, generators, all_names, threshold=args.safe_threshold)

    print(f"all: {X_all.shape[1]}D, safe: {X_safe.shape[1]}D, samples: {len(y)}")

    models = get_models()
    results = []

    print(f"MODEL COMPARISON (Safe {X_safe.shape[1]}D, LOGO):")
    for model_name, clf in models.items():
        print(f"{model_name}:")
        res = logo_eval_model(X_safe, y, generators, model_name, clf)
        parts = "  ".join(f"{g}={v['auc']:.3f}" for g, v in res["per_gen"].items())
        print(f"AUC={res['mean_auc']:.3f}  Acc={res['mean_acc']:.3f}  F1={res['mean_f1']:.3f}  Min={res['min_auc']:.3f}")
        print(f"{parts}")
        results.append({"model": model_name, "n": X_safe.shape[1], "auc_mean": res["mean_auc"], "auc_min": res["min_auc"], "acc": res["mean_acc"], "f1": res["mean_f1"]})

    print(f"SUMMARY:")
    print(f"{'Model':35s} {'AUC':>7s} {'Acc':>7s} {'F1':>7s} {'Min AUC':>7s}")
    
    for r in sorted(results, key=lambda x: -x["auc_mean"]):
        print(f"{r['model']:33s} {r['auc_mean']:7.3f} {r['acc']:7.3f} {r['f1']:7.3f} {r['auc_min']:7.3f}")

    if args.save_csv:
        out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "model_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_dir / "model_comparison.csv", index=False)
        print(f"saved to {out_dir / 'model_comparison.csv'}")

    print(f"<TA-DAM DONE>")




if __name__ == "__main__":
    main()
