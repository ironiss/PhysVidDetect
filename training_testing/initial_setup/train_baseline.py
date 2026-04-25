import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from utils import load_h5, fill_nan, extract_generator, extract_video_id, merge_h5_by_filename, cleanup_features, stratified_split, train_and_eval, holdout_split #, plot_roc, plot_confusion, plot_per_gen_bars



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




def run_baseline(X, y, generators, video_ids, model_name, out_dir, cv_folds=5):
    """seen generators -- 70/15/15 split"""
    print(f"BASELINE -- real vs fake")

    tr_idx, va_idx, te_idx = stratified_split(y, generators, video_ids)
    print(f"train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        real_n = (y[idx]==1).sum()
        fake_n = (y[idx]==0).sum()
        print(f"{name}: real={real_n} fake={fake_n}")

    trainval_idx = np.concatenate([tr_idx, va_idx])
    selected_cv_df = None

    # model selection via CV
    if model_name == "auto" and cv_folds>0:
        print(f"model selection ({cv_folds}-fold CV on train+val)")
        best_model_name = None
        best_cv_auc = -np.inf
        cv_results = {}

        for candidate in ["logreg", "random_forest", "xgb"]:
            print(f"{candidate}:")
            cv_df = cross_validation(X[trainval_idx], y[trainval_idx], candidate, n_folds=cv_folds)
            cv_results[candidate] = cv_df
            mean_auc = cv_df["auc"].mean()

            if mean_auc>best_cv_auc:
                best_cv_auc = mean_auc
                best_model_name = candidate
                selected_cv_df = cv_df

        print(f"best model: {best_model_name} (CV AUC={best_cv_auc:.3f})")
        model_name = best_model_name

        for name, df in cv_results.items():
            df.to_csv(out_dir / f"cv_{name}.csv", index=False)

    elif cv_folds>0:
        print(f"{cv_folds}-fold CV on train+val ({model_name})")
        selected_cv_df = cross_validation(X[trainval_idx], y[trainval_idx], model_name, n_folds=cv_folds)
        selected_cv_df.to_csv(out_dir / "cv_results.csv", index=False)

    res = train_and_eval(X[trainval_idx], y[trainval_idx], X[te_idx], y[te_idx], get_model(model_name))

    print(f"\ntest ({model_name}): acc={res['acc']:.3f}, f1={res['f1']:.3f}, auc={res['auc']:.3f}")

    if selected_cv_df is not None:
        cv_auc = selected_cv_df["auc"].mean()
        gap = cv_auc - res["auc"]
        status = "possible overfitting" if abs(gap)>0.05 else "looks stable"
        print(f"cv-test gap: {gap:+.3f} -- {status}")

    print(f"\nper-generator breakdown:")
    gen_results = {}
    for g in sorted(set(generators[te_idx])):
        g_mask = generators[te_idx]==g
        g_y = y[te_idx][g_mask]
        g_preds = res["preds"][g_mask]
        g_acc = accuracy_score(g_y, g_preds)
        g_f1 = f1_score(g_y, g_preds, zero_division=0)

        label_str = "real" if g_y[0]==1 else "fake"
        print(f"  {g} ({label_str}, n={len(g_y)}): acc={g_acc:.3f}, f1={g_f1:.3f}")
        gen_results[g] = {"acc": g_acc, "f1": g_f1, "n": len(g_y)}

    # plot_roc(y[te_idx], res["probs"], f"ROC baseline {model_name}", out_dir / "roc.png")
    # plot_confusion(y[te_idx], res["preds"], f"Confusion baseline {model_name}", out_dir / "confusion.png")

    preds_df = pd.DataFrame()
    if video_ids is not None:
        preds_df["path"] = video_ids[te_idx]
    else:
        preds_df["path"] = np.arange(len(te_idx))
    
    preds_df["true"] = y[te_idx]
    preds_df["pred"] = res["preds"]
    preds_df["prob"] = res["probs"]
    preds_df["generator"] = generators[te_idx]
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)

    with open(out_dir / f"model_{model_name}.pkl", "wb") as f:
        pickle.dump(res["model"], f)
    with open(out_dir / f"scaler_{model_name}.pkl", "wb") as f:
        pickle.dump(res["scaler"], f)

    return res


def run_per_generator(X, y, generators, video_ids, model_name, out_dir):
    """per-generator -- real vs each generator separately"""
    print(f"PER-GENERATOR -- real vs each ({model_name})")

    is_real = (y == 1)
    is_fake = (y == 0)
    fake_gens = sorted(set(generators[is_fake]))
    all_results = {}

    for g in fake_gens:
        print(f"real vs {g}")

        is_this_gen = (generators == g)
        combined = is_real | is_this_gen
        X_sub, y_sub = X[combined], y[combined]
        vid_sub, gen_sub = video_ids[combined], generators[combined]

        tr_idx, va_idx, te_idx = stratified_split(y_sub, gen_sub, vid_sub)
        trainval_idx = np.concatenate([tr_idx, va_idx])

        res = train_and_eval(X_sub[trainval_idx], y_sub[trainval_idx], X_sub[te_idx], y_sub[te_idx], get_model(model_name))
        print(f"acc={res['acc']:.3f}, f1={res['f1']:.3f}, auc={res['auc']:.3f}")

        all_results[g] = res

        # plot_roc(y_sub[te_idx], res["probs"], f"ROC real vs {g}", out_dir / f"roc_vs_{g}.png")

    # plot_per_gen_bars(all_results, "auc", f"AUC per generator ({model_name})", out_dir / "per_gen_auc.png")
    # plot_per_gen_bars(all_results, "f1", f"F1 per generator ({model_name})", out_dir / "per_gen_f1.png")

    rows = []
    for g, res in all_results.items():
        rows.append({"generator": g, "acc": res["acc"], "f1": res["f1"], "auc": res["auc"], "precision": res["precision"], "recall": res["recall"]})
    
    pd.DataFrame(rows).to_csv(out_dir / "per_gen_summary.csv", index=False)
    return all_results


def run_holdout_generator(X, y, generators, video_ids, model_name, out_dir):
    """LOGO protocol -- train on N-1 generators, test on holdout"""
    print(f"HOLDOUT GENERATOR -- generalization test ({model_name})")

    fake_gens = sorted(set(generators[y==0]))
    all_results = {}

    for holdout_gen in fake_gens:
        print(f"holdout gen: {holdout_gen}")
        tr_idx, te_idx = holdout_split(y, generators, holdout_gen)

        train_gens = sorted(set(generators[tr_idx][y[tr_idx]==0]))
        print(f"train: {len(tr_idx)} from {train_gens}")
        print(f"test: {len(te_idx)}")

        res = train_and_eval(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], get_model(model_name))
        print(f"acc={res['acc']:.3f}, f1={res['f1']:.3f}, auc={res['auc']:.3f}")

        all_results[holdout_gen] = res

        # plot_roc(y[te_idx], res["probs"], f"ROC holdout {holdout_gen}", out_dir / f"roc_holdout_{holdout_gen}.png")

    # plot_per_gen_bars(all_results, "auc", f"LOGO AUC ({model_name})", out_dir / "holdout_gen_auc.png")
    # plot_per_gen_bars(all_results, "f1", f"LOGO F1 ({model_name})", out_dir / "holdout_gen_f1.png")

    rows = []
    for g, res in all_results.items():
        rows.append({"holdout_gen": g, "acc": res["acc"], "f1": res["f1"], "auc": res["auc"], "precision": res["precision"], "recall": res["recall"]})
    
    pd.DataFrame(rows).to_csv(out_dir / "holdout_gen_summary.csv", index=False)
    return all_results




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-feats", default="DATA/latent_features.h5", help="latent features H5")
    parser.add_argument("--object-based-feats", default=None, help="object-based physics features H5")
    parser.add_argument("--model", default="auto", choices=["logreg", "random_forest", "xgb", "auto"])
    parser.add_argument("--mode", default="all", choices=["baseline", "per-gen", "holdout-gen", "all"])
    parser.add_argument("--cv", type=int, default=5, help="CV folds for baseline (0 to skip)")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--nan-thresh", type=float, default=80)
    parser.add_argument("--corr-thresh", type=float, default=0.9)
    parser.add_argument("--out-dir", default="results_baseline")
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
    out = Path(args.out_dir)
    

    if args.mode in ("baseline", "all"):
        d = out / f"baseline_{args.model}"
        d.mkdir(parents=True, exist_ok=True)
        run_baseline(X, y, generators, video_ids, args.model, d, cv_folds=args.cv)

    if args.mode in ("per-gen", "all"):
        d = out / f"per_gen_{args.model}"
        d.mkdir(parents=True, exist_ok=True)
        run_per_generator(X, y, generators, video_ids, args.model, d)

    if args.mode in ("holdout-gen", "all"):
        d = out / f"holdout_gen_{args.model}"
        d.mkdir(parents=True, exist_ok=True)
        run_holdout_generator(X, y, generators, video_ids, args.model, d)

    print(f"<DONE> saved to {out.resolve()}")



if __name__ == "__main__":
    main()
