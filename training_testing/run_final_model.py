import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import fill_nan, safe_auc, basic_cleanup



warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
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
    lat = lat[lat["_key"].isin(common)].drop_duplicates("_key").sort_values("_key").reset_index(drop=True)
    phy = phy[phy["_key"].isin(common)].drop_duplicates("_key").sort_values("_key").reset_index(drop=True)
    cam = cam[cam["_key"].isin(common)].drop_duplicates("_key").sort_values("_key").reset_index(drop=True)

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

    X_z0_c, n_z0 = basic_cleanup(X_z0, y, np.array(z0_cols))
    X_eps_c, n_eps = basic_cleanup(X_eps, y, np.array(eps_cols))
    X_phy_c, n_phy = basic_cleanup(X_phy, y, np.array(phy_cols))
    X_cam_c, n_cam = basic_cleanup(X_cam, y, np.array(cam_cols))

    X_all = np.concatenate([X_z0_c, X_phy_c, X_eps_c, X_cam_c], axis=1)
    all_names = np.concatenate([n_z0, n_phy, n_eps, n_cam])

    return X_all, y, generators, all_names


def safe_filter(X, y, generators, all_names, threshold=0.65):
    """pairwise gen_mean filtering"""
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


def get_tuned_model(y):
    n_neg, n_pos = (y==0).sum(), (y==1).sum()

    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg/n_pos,
        eval_metric="logloss", random_state=42,
    )


def run_logo(X, y, generators):
    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]
    rows = []

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
        clf = get_tuned_model(y[tr])
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        probs = clf.predict_proba(sc.transform(X[te]))[:, 1]
        preds = (probs>=0.5).astype(int)

        auc = roc_auc_score(y[te], probs)
        acc = accuracy_score(y[te], preds)
        f1 = f1_score(y[te], preds)
        print(f"{holdout_gen}: AUC={auc:.3f}, Acc={acc:.3f}, F1={f1:.3f}")
        rows.append({"holdout_gen": holdout_gen, "auc": auc, "acc": acc, "f1": f1})

    df = pd.DataFrame(rows)
    print(f"Mean AUC: {df['auc'].mean():.3f}, Min AUC: {df['auc'].min():.3f}")
    print(f"Mean Acc: {df['acc'].mean():.3f}, Mean F1: {df['f1'].mean():.3f}")
    return df


def run_seen(X, y, generators, test_size=0.10):
    fake_gens = sorted(set(generators[y==0]))

    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=42)
    sc = StandardScaler()
    clf = get_tuned_model(y[tr_idx])
    clf.fit(sc.fit_transform(X[tr_idx]), y[tr_idx])
    probs = clf.predict_proba(sc.transform(X[te_idx]))[:, 1]
    preds = (probs>=0.5).astype(int)

    print(f"AUC: {roc_auc_score(y[te_idx], probs):.3f}, Acc: {accuracy_score(y[te_idx], preds):.3f}, F1: {f1_score(y[te_idx], preds):.3f}")

    real_te_probs = probs[y[te_idx]==1]
    real_te_y = y[te_idx][y[te_idx]==1]
    rows = []

    for g in fake_gens:
        g_mask = generators[te_idx]==g
        if g_mask.sum()==0:
            continue
        g_y = np.concatenate([real_te_y, y[te_idx][g_mask]])
        g_probs = np.concatenate([real_te_probs, probs[g_mask]])
        g_preds = (g_probs>=0.5).astype(int)
        g_auc = roc_auc_score(g_y, g_probs)
        g_acc = accuracy_score(g_y, g_preds)
        print(f"{g}: AUC={g_auc:.3f}, Acc={g_acc:.3f}")
        rows.append({"generator": g, "auc": g_auc, "acc": g_acc})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--safe-threshold", type=float, default=0.65)
    parser.add_argument("--seen-test-size", type=float, default=0.10)
    args = parser.parse_args()

    X_all, y, generators, all_names = load_and_prepare()

    X_safe, _ = safe_filter(X_all, y, generators, all_names, threshold=args.safe_threshold)
    print(f"Safe features: {X_safe.shape[1]}D, samples: {len(y)}")

    print("--LOGO--")
    logo_df = run_logo(X_safe, y, generators)

    
    print("--SEEN ({}/{}  split)--".format(int((1-args.seen_test_size)*100), int(args.seen_test_size*100)))
    seen_df = run_seen(X_safe, y, generators, test_size=args.seen_test_size)

    if args.save_csv:
        out_dir = Path(__file__).resolve().parent / "results" / "final_model"
        out_dir.mkdir(parents=True, exist_ok=True)
        logo_df.to_csv(out_dir / "logo_results.csv", index=False)
        seen_df.to_csv(out_dir / "seen_per_generator.csv", index=False)
        print(f"\nsaved to {out_dir}")




if __name__ == "__main__":
    main()
