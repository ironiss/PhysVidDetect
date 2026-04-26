import argparse
import sys
import warnings
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
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


def build_logo_folds(y, generators):
    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]
    folds = []

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

        folds.append({"gen": holdout_gen, "train_pool": train_pool, "tr": train_pool[tr_local], "val": train_pool[val_local], "test": te})
    
    return folds


def eval_on_val(X, y, folds, params):
    val_aucs = {}
    for fs in folds:
        sc = StandardScaler()
        n_pos = (y[fs["tr"]]==1).sum()
        n_neg = (y[fs["tr"]]==0).sum()
        clf = XGBClassifier(**params, scale_pos_weight=n_neg/n_pos, eval_metric="logloss", random_state=42)
        clf.fit(sc.fit_transform(X[fs["tr"]]), y[fs["tr"]])
        probs = clf.predict_proba(sc.transform(X[fs["val"]]))[:, 1]
        val_aucs[fs["gen"]] = roc_auc_score(y[fs["val"]], probs)
    
    return np.mean(list(val_aucs.values())), min(val_aucs.values())


def eval_on_test(X, y, folds, params):
    test_aucs = {}
    test_accs = {}
    test_f1s = {}

    for fs in folds:
        sc = StandardScaler()
        n_pos = (y[fs["train_pool"]]==1).sum()
        n_neg = (y[fs["train_pool"]]==0).sum()
        clf = XGBClassifier(**params, scale_pos_weight=n_neg/n_pos, eval_metric="logloss", random_state=42)
        clf.fit(sc.fit_transform(X[fs["train_pool"]]), y[fs["train_pool"]])
        probs = clf.predict_proba(sc.transform(X[fs["test"]]))[:, 1]
        preds = (probs>=0.5).astype(int)
        test_aucs[fs["gen"]] = roc_auc_score(y[fs["test"]], probs)
        test_accs[fs["gen"]] = accuracy_score(y[fs["test"]], preds)
        test_f1s[fs["gen"]] = f1_score(y[fs["test"]], preds)
    
    return test_aucs, test_accs, test_f1s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--safe-threshold", type=float, default=0.65)
    args = parser.parse_args()

    X_all, y, generators, all_names = load_and_prepare()
    X_safe, _ = safe_filter(X_all, y, generators, all_names, threshold=args.safe_threshold)

    print(f"safe features: {X_safe.shape[1]}D, samples: {len(y)}")

    folds = build_logo_folds(y, generators)

    print("GRID SEARCH (val AUC):")
    grid = list(product([100, 200, 300, 500], [0.01, 0.05, 0.1], [3, 4, 5, 6]))
    grid_results = []

    for i, (n_est, lr, depth) in enumerate(grid, 1):
        params = {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr, "subsample": 0.8, "colsample_bytree": 0.8}
        mean_val, min_val = eval_on_val(X_safe, y, folds, params)
        grid_results.append({**params, "val_mean": mean_val, "val_min": min_val})
        
        print(f"[{i:2d}/{len(grid)}] n={n_est:3d} d={depth} lr={lr:.2f}  val_mean={mean_val:.4f} val_min={min_val:.4f}")




    df_grid = pd.DataFrame(grid_results).sort_values("val_mean", ascending=False)

    print(f"top 5 by val_mean:")
    print(df_grid.head(5)[["n_estimators", "max_depth", "learning_rate", "val_mean", "val_min"]].to_string())

    best = df_grid.iloc[0]
    best_params = {"n_estimators": int(best["n_estimators"]), "max_depth": int(best["max_depth"]), "learning_rate": best["learning_rate"], "subsample": 0.8, "colsample_bytree": 0.8}


    print(f"FINAL (retrain on full train, test on holdout):")
    print(f"best params: {best_params}")

    test_aucs, test_accs, test_f1s = eval_on_test(X_safe, y, folds, best_params)
    for g in sorted(test_aucs):
        print(f"{g}: AUC={test_aucs[g]:.3f}, Acc={test_accs[g]:.3f}, F1={test_f1s[g]:.3f}")

    print(f"mean AUC: {np.mean(list(test_aucs.values())):.3f}")
    print(f"min AUC: {min(test_aucs.values()):.3f}")
    print(f"mean Acc: {np.mean(list(test_accs.values())):.3f}")
    print(f"mean F1: {np.mean(list(test_f1s.values())):.3f}")



    print(f"COMPARISON:")
    configs = [("Baseline (300/4/0.05)", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}), (f"Val-tuned ({best_params['n_estimators']}/{best_params['max_depth']}/{best_params['learning_rate']})", best_params)]
    
    for label, params in configs:
        aucs, accs, f1s = eval_on_test(X_safe, y, folds, params)
        print(f"{label:40s}  mean={np.mean(list(aucs.values())):.3f}  min={min(aucs.values()):.3f}  acc={np.mean(list(accs.values())):.3f}  f1={np.mean(list(f1s.values())):.3f}")

    if args.save_csv:
        out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "xgb_tuning"
        out_dir.mkdir(parents=True, exist_ok=True)
        df_grid.to_csv(out_dir / "xgb_tuning_grid.csv", index=False)
        print(f"saved to {out_dir / 'xgb_tuning_grid.csv'}")

    print(f"<TA-DAM DONE>")




if __name__ == "__main__":
    main()
