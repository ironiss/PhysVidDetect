import argparse
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from utils import load_h5, fill_nan, extract_generator, merge_h5_by_filename, safe_auc, basic_cleanup

warnings.filterwarnings("ignore")



def compute_ranking(X, y, generators, names):
    """per-feature AUC ranking: global + per-generator + stability"""
    is_real = (y == 1)
    fake_gens = sorted(set(generators[y==0]))
    rows = []

    for fi in range(X.shape[1]):
        row = {"feature": names[fi]}
        row["global_auc"] = safe_auc(X[:, fi], y)

        for g in fake_gens:
            g_mask = generators==g
            combined = is_real | g_mask
            row[g] = safe_auc(X[combined, fi], y[combined])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("feature")
    df["mean_auc"] = df[fake_gens].mean(axis=1)
    df["std_auc"] = df[fake_gens].std(axis=1)
    df["stability"] = df["mean_auc"] - df["std_auc"]

    return df, fake_gens


def logo_eval(X, y, generators, label="", model_type="xgb"):
    """LOGO evaluation with feature subset"""

    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]
    aucs = {}

    for holdout_gen in fake_gens:
        holdout_idx = np.where(generators==holdout_gen)[0]
        rng = np.random.default_rng(42)
        n_test_real = min(len(holdout_idx), len(real_idx)//3)
        test_real_idx = rng.choice(real_idx, n_test_real, replace=False)
        test_real_set = set(test_real_idx)

        train_real = np.array([i for i in real_idx if i not in test_real_set])
        train_fake = np.where((y==0) & (generators != holdout_gen))[0]
        tr = np.concatenate([train_real, train_fake])
        te = np.concatenate([test_real_idx, holdout_idx])

        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])

        if model_type=="xgb":
            clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
        else:
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        clf.fit(Xtr, y[tr])
        probs = clf.predict_proba(Xte)[:, 1]
        aucs[holdout_gen] = roc_auc_score(y[te], probs)

    mean_auc = np.mean(list(aucs.values()))
    min_auc = min(aucs.values())
    parts = "  ".join(f"{g}={a:.3f}" for g, a in aucs.items())
    print(f"  {label:30s} n={X.shape[1]:3d}  mean={mean_auc:.3f}  min={min_auc:.3f}  {parts}")
    return aucs, mean_auc


def select_features(X, names, ranking, method, k):
    if method=="global":
        top = ranking.sort_values("global_auc", ascending=False).head(k).index
    elif method=="stable":
        top = ranking.sort_values("stability", ascending=False).head(k).index
    else:
        raise ValueError(f"unknown method: {method}")

    idx = [i for i, n in enumerate(names) if n in top]
    return X[:, idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-feats", required=True, help="path to h5 file with latent features")
    parser.add_argument("--object-based-feats", help="optional path to h5 file with object-based features (will be merged with latent features)")
    parser.add_argument("--model", default="xgb", choices=["xgb", "logreg"])
    args = parser.parse_args()

    X, y, paths, feat_names = load_h5(args.latent_feats)

    if args.object_based_feats:
        X2, y2, p2, n2 = load_h5(args.object_based_feats)
        X, y, paths, feat_names = merge_h5_by_filename(X, y, paths, feat_names, X2, y2, p2, n2)

    X = fill_nan(X)
    generators = extract_generator(paths)

    print(f"total: {len(y)} samples, {X.shape[1]} features")
    print(f"real: {(y==1).sum()}, fake: {(y==0).sum()}")

    X_clean, clean_names = basic_cleanup(X, y, feat_names)
    print(f"after cleanup: {X_clean.shape[1]} features")

    ranking, fake_gens = compute_ranking(X_clean, y, generators, clean_names)

    print(f"\nLOGO evaluation ({args.model}):")
    logo_eval(X_clean, y, generators, "All cleaned", model_type=args.model)

    for k in [20, 30, 40]:
        if k>len(clean_names):
            continue

        X_sel = select_features(X_clean, clean_names, ranking, "global", k)
        logo_eval(X_sel, y, generators, f"Global top-{k}", model_type=args.model)

    for k in [20, 30, 40]:
        if k>len(clean_names):
            continue

        X_sel = select_features(X_clean, clean_names, ranking, "stable", k)
        logo_eval(X_sel, y, generators, f"Stable top-{k}", model_type=args.model)

    print(f"\ntop 10 most stable features:")
    for i, (feat, row) in enumerate(ranking.sort_values("stability", ascending=False).head(10).iterrows()):
        gens_str = "  ".join(f"{g}={row[g]:.3f}" for g in fake_gens)
        print(f"  {i+1}. {feat:35s} stability={row['stability']:.3f}  {gens_str}")




if __name__ == "__main__":
    main()
