import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import load_h5, fill_nan, extract_generator, extract_video_id, merge_h5_by_filename, cleanup_features, stratified_split, train_and_eval, holdout_split



def get_model(name):
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    else:
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)



def cross_validation(X, y, model_name, n_folds=5, seed=42):
    """cross-validation check"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    metrics = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        res = train_and_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], get_model(model_name))
        metrics.append({k: res[k] for k in ["acc", "precision", "recall", "f1", "auc"]})
        print(f"fold {fold+1}: acc={res['acc']:.3f}, f1={res['f1']:.3f}, auc={res['auc']:.3f}")

    df = pd.DataFrame(metrics)

    print("cross-val results:")
    for col in ["acc", "f1", "auc"]:
        print(f"{col}: {df[col].mean():.3f}+-{df[col].std():.3f}")
    return df




def run_baseline(X, y, generators, video_ids, model_name, test_size=0.20, cv_folds=5, save_csv=None):
    """seen generators -- + some split"""
    print(f"BASELINE -- real vs fake")

    tr_idx, te_idx = stratified_split(y, generators, video_ids, test_size=test_size)
    print(f"train={len(tr_idx)} test={len(te_idx)}")

    for name, idx in [("train", tr_idx), ("test", te_idx)]:
        real_n = (y[idx]==1).sum()
        fake_n = (y[idx]==0).sum()
        print(f"{name}: real={real_n} fake={fake_n}")

    selected_cv_df = None

    if model_name == "auto" and cv_folds>0:
        print(f"model selection ({cv_folds}-fold CV on train)")
        best_model_name = None
        best_cv_auc = -np.inf
        cv_results = {}

        for candidate in ["logreg", "random_forest", "xgb"]:
            print(f"{candidate}:")
            cv_df = cross_validation(X[tr_idx], y[tr_idx], candidate, n_folds=cv_folds)
            cv_results[candidate] = cv_df
            mean_auc = cv_df["auc"].mean()

            if mean_auc>best_cv_auc:
                best_cv_auc = mean_auc
                best_model_name = candidate
                selected_cv_df = cv_df

        print(f"best model: {best_model_name} (CV AUC={best_cv_auc:.3f})")
        model_name = best_model_name

    elif cv_folds>0:
        print(f"{cv_folds}-fold CV on train ({model_name})")
        selected_cv_df = cross_validation(X[tr_idx], y[tr_idx], model_name, n_folds=cv_folds)

    res = train_and_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], get_model(model_name))

    print(f"test ({model_name}): acc={res['acc']:.3f}, f1={res['f1']:.3f}, auc={res['auc']:.3f}")

    if selected_cv_df is not None:
        cv_auc = selected_cv_df["auc"].mean()
        gap = cv_auc - res["auc"]
        status = "possible overfitting" if abs(gap)>0.05 else "looks stable"
        print(f"cv-test gap: {gap:+.3f} -- {status}")

    print(f"per-generator breakdown (same model, split by generator in test):")
    real_te_mask = y[te_idx]==1
    real_te_probs = res["probs"][real_te_mask]
    real_te_y = y[te_idx][real_te_mask]

    rows = []
    for g in sorted(set(generators[te_idx][y[te_idx]==0])):
        g_mask = generators[te_idx]==g
        g_y = np.concatenate([real_te_y, y[te_idx][g_mask]])
        g_probs = np.concatenate([real_te_probs, res["probs"][g_mask]])
        g_preds = (g_probs>=0.5).astype(int)

        g_acc = accuracy_score(g_y, g_preds)
        g_auc = roc_auc_score(g_y, g_probs)

        print(f"{g} (n={g_mask.sum()}): acc={g_acc:.3f}, auc={g_auc:.3f}")
        rows.append({"generator": g, "acc": g_acc, "auc": g_auc, "n": int(g_mask.sum())})

    if save_csv:
        pd.DataFrame(rows).to_csv(save_csv, index=False)
        print(f"saved to {save_csv}")

    return res


def run_per_generator(X, y, generators, video_ids, model_name, save_csv=None):
    """per-generator -- real vs each generator separately"""
    print(f"PER-GENERATOR -- real vs each ({model_name})")

    is_real = (y == 1)
    is_fake = (y == 0)
    fake_gens = sorted(set(generators[is_fake]))
    all_results = {}

    rows = []
    for g in fake_gens:
        print(f"real vs {g}")

        is_this_gen = (generators == g)
        combined = is_real | is_this_gen
        X_sub, y_sub = X[combined], y[combined]
        vid_sub, gen_sub = video_ids[combined], generators[combined]

        tr_idx, te_idx = stratified_split(y_sub, gen_sub, vid_sub)

        res = train_and_eval(X_sub[tr_idx], y_sub[tr_idx], X_sub[te_idx], y_sub[te_idx], get_model(model_name))
        print(f"acc={res['acc']:.3f}, f1={res['f1']:.3f}, auc={res['auc']:.3f}")

        all_results[g] = res
        rows.append({"generator": g, "acc": res["acc"], "f1": res["f1"], "auc": res["auc"]})

    if save_csv:
        pd.DataFrame(rows).to_csv(save_csv, index=False)
        print(f"saved to {save_csv}")

    return all_results


def run_holdout_generator(X, y, generators, video_ids, model_name, save_csv=None):
    """LOGO protocol -- train on N-1 generators, test on holdout"""
    print(f"HOLDOUT GENERATOR -- generalization test ({model_name})")

    fake_gens = sorted(set(generators[y==0]))
    all_results = {}

    rows = []
    for holdout_gen in fake_gens:
        print(f"holdout gen: {holdout_gen}")
        tr_idx, te_idx = holdout_split(y, generators, holdout_gen)

        train_gens = sorted(set(generators[tr_idx][y[tr_idx]==0]))
        print(f"train: {len(tr_idx)} from {train_gens}")
        print(f"test: {len(te_idx)}")

        res = train_and_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], get_model(model_name))
        print(f"acc={res['acc']:.3f}, f1={res['f1']:.3f}, auc={res['auc']:.3f}")

        all_results[holdout_gen] = res
        rows.append({"holdout_gen": holdout_gen, "acc": res["acc"], "f1": res["f1"], "auc": res["auc"]})

    aucs = [r["auc"] for r in all_results.values()]
    print(f"mean AUC: {np.mean(aucs):.3f}, min AUC: {min(aucs):.3f}")

    if save_csv:
        pd.DataFrame(rows).to_csv(save_csv, index=False)
        print(f"saved to {save_csv}")

    return all_results




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-feats", default="DATA/latent_features.h5", help="latent features H5")
    parser.add_argument("--object-based-feats", default=None, help="object-based physics features H5")
    parser.add_argument("--model", default="auto", choices=["logreg", "random_forest", "xgb", "auto"])
    parser.add_argument("--mode", default="all", choices=["baseline", "per-gen", "holdout-gen", "all"])
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv", type=int, default=5, help="CV folds for baseline (0 to skip)")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--nan-thresh", type=float, default=80)
    parser.add_argument("--corr-thresh", type=float, default=0.9)
    parser.add_argument("--save-csv", default=None, help="directory to save CSV results")
    args = parser.parse_args()

    try:
        X, y, video_paths, feat_names = load_h5(args.latent_feats)
    except Exception as e:
        print(f"error {args.latent_feats}: {e}")
        return

    if args.object_based_feats:
        X2, y2, paths2, names2 = load_h5(args.object_based_feats)
        X, y, video_paths, feat_names = merge_h5_by_filename(X, y, video_paths, feat_names, X2, y2, paths2, names2)

    print(f"real: {(y==1).sum()}, fake: {(y==0).sum()}")

    generators = extract_generator(video_paths)
    video_ids = extract_video_id(video_paths)

    unique_gens = sorted(set(generators))
    print(f"generators: {unique_gens}")
    for g in unique_gens:
        cnt = (generators==g).sum()
        label = "real" if y[generators==g][0]==1 else "fake"
        print(f"{g} ({label}): {cnt}")

    if not args.no_cleanup and feat_names is not None:
        X, feat_names, _ = cleanup_features(X, y, feat_names, nan_thresh=args.nan_thresh, corr_thresh=args.corr_thresh)
    else:
        X = fill_nan(X)

    csv_dir = None
    if args.save_csv:
        csv_dir = Path(args.save_csv)
        csv_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("baseline", "all"):
        csv_path = csv_dir / f"baseline_{args.model}.csv" if csv_dir else None
        run_baseline(X, y, generators, video_ids, args.model, test_size=args.test_size, cv_folds=args.cv, save_csv=csv_path)

    if args.mode in ("per-gen", "all"):
        csv_path = csv_dir / f"per_gen_{args.model}.csv" if csv_dir else None
        run_per_generator(X, y, generators, video_ids, args.model, save_csv=csv_path)

    if args.mode in ("holdout-gen", "all"):
        csv_path = csv_dir / f"holdout_gen_{args.model}.csv" if csv_dir else None
        run_holdout_generator(X, y, generators, video_ids, args.model, save_csv=csv_path)

    print(f"<TA-DAM DONE>")



if __name__ == "__main__":
    main()
