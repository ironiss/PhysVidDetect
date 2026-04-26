"""
Threshold optimization on Safe(74D) features with XGBoost.
Compares: default(0.5), Youden's J, max-F1, max-balanced-accuracy, grid search.
LOGO protocol, per-generator breakdown.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                              f1_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE = Path(__file__).parent

KNOWN_FAKE = {"DynamicCrafter","SVD","ZeroScope","Pika","Latte","OpenSora","VideoCrafter","SEINE"}
KNOWN_REAL = {"GenVideo-Real","GenVideo-Real-clean-3k","Kinetics-400","Kinetics-400-additional-4k","UCF-101-7k"}
GEN_SHORT = {"DynamicCrafter":"DC","SVD":"SVD","ZeroScope":"ZS","Pika":"Pika",
             "Latte":"Latte","OpenSora":"OSora","VideoCrafter":"VCraft","SEINE":"SEINE"}

def extract_gen(paths):
    gens = []
    for p in paths:
        parts = Path(p).parts
        gen = "unknown"
        for i, part in enumerate(parts):
            if part.lower() in ("real","fake") and i+1<len(parts):
                gen = parts[i+1]; break
        if gen == "unknown":
            for part in parts:
                if part in KNOWN_FAKE or part in KNOWN_REAL:
                    gen = part; break
        gens.append(gen)
    return np.array(gens)

def fill_nan(X):
    X = X.copy()
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    nm = np.isnan(X)
    if nm.any(): X[nm] = np.take(med, np.where(nm)[1])
    return X

def safe_auc(col, y):
    if np.std(col) < 1e-12: return 0.5
    a = roc_auc_score(y, col); return max(a, 1-a)

def cleanup(X, names, y):
    nan_pct = np.isnan(X).mean(axis=0)*100
    keep = nan_pct <= 80
    X = fill_nan(X); X, names = X[:,keep], names[keep]
    if X.shape[1] <= 1: return X, names
    aucs = np.array([safe_auc(X[:,i], y) for i in range(X.shape[1])])
    corr = np.corrcoef(X.T); np.fill_diagonal(corr, 0)
    to_drop = set()
    for i in range(len(names)):
        if i in to_drop: continue
        for j in range(i+1, len(names)):
            if j in to_drop: continue
            if abs(corr[i,j]) > 0.9:
                to_drop.add(j if aucs[i]>=aucs[j] else i)
    keep2 = np.array([i not in to_drop for i in range(len(names))])
    return X[:,keep2], names[keep2]

# ── Load & prepare ──
print("Loading data...")
df_lat = pd.read_csv(BASE / "latent_features/GENERAL_ALL_LATENT.csv")
df_phy = pd.read_csv(BASE / "physics_features/GENERAL_ALL_PHYSICS.csv")
df_cam = pd.read_csv(BASE / "features_v2.csv")
df_lat["_key"] = df_lat["path"].apply(lambda p: Path(p).name)
df_phy["_key"] = df_phy["path"].apply(lambda p: Path(p).name)
df_cam["_key"] = df_cam["video"].apply(lambda p: Path(p).name)
common = set(df_lat["_key"]) & set(df_phy["_key"]) & set(df_cam["_key"])
for df in [df_lat, df_phy, df_cam]: df.drop_duplicates("_key", inplace=True)
df_lat = df_lat[df_lat["_key"].isin(common)].sort_values("_key").reset_index(drop=True)
df_phy = df_phy[df_phy["_key"].isin(common)].sort_values("_key").reset_index(drop=True)
df_cam = df_cam[df_cam["_key"].isin(common)].sort_values("_key").reset_index(drop=True)
paths = df_lat["path"].values; y = df_lat["label"].values.astype(int)
generators = extract_gen(paths)
valid = np.array([g in KNOWN_FAKE or g in KNOWN_REAL for g in generators])
y, paths, generators = y[valid], paths[valid], generators[valid]
meta = {"path","label","_key","video"}
lat_cols = [c for c in df_lat.columns if c not in meta]
phy_cols = [c for c in df_phy.columns if c not in meta]
cam_cols = [c for c in df_cam.columns if c not in meta]
X_all_raw = np.hstack([
    df_lat[lat_cols].values[valid].astype(np.float32),
    df_phy[phy_cols].values[valid].astype(np.float32),
    df_cam[cam_cols].values[valid].astype(np.float32),
])
all_names_raw = np.array(lat_cols + phy_cols + cam_cols)
X_all, all_names = cleanup(X_all_raw, all_names_raw, y)

# Safe filtering
fake_gens = sorted(set(generators[y==0]))
real_mask = y == 1; fake_mask = y == 0; real_idx = np.where(real_mask)[0]

safe_mask = np.ones(X_all.shape[1], dtype=bool)
for fi in range(X_all.shape[1]):
    pair_aucs = []
    for gi, g1 in enumerate(fake_gens):
        for g2 in fake_gens[gi+1:]:
            m1 = generators == g1; m2 = generators == g2
            comb = m1 | m2
            if comb.sum() > 10:
                pair_aucs.append(safe_auc(X_all[comb, fi], (generators[comb]==g1).astype(int)))
    if np.mean(pair_aucs) > 0.65:
        safe_mask[fi] = False

X = X_all[:, safe_mask]
print(f"Safe features: {X.shape[1]}D, Total: {len(y)}")

# ── Threshold finding functions ──
def find_youden(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return thresholds[np.argmax(j)]

def find_max_f1(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * precision * recall / (precision + recall + 1e-10)
    return thresholds[np.argmax(f1s[:-1])]

def find_max_balacc(y_true, probs, n_steps=200):
    best_t, best_ba = 0.5, 0
    for t in np.linspace(0.1, 0.9, n_steps):
        preds = (probs >= t).astype(int)
        ba = balanced_accuracy_score(y_true, preds)
        if ba > best_ba:
            best_ba = ba; best_t = t
    return best_t

def find_max_acc(y_true, probs, n_steps=200):
    best_t, best_a = 0.5, 0
    for t in np.linspace(0.1, 0.9, n_steps):
        preds = (probs >= t).astype(int)
        a = accuracy_score(y_true, preds)
        if a > best_a:
            best_a = a; best_t = t
    return best_t

# ── LOGO with threshold optimization ──
def logo_with_thresholds(X, y, generators):
    # Collect all predictions first
    fold_data = []  # list of (y_te, probs, gen_label)

    for holdout_gen in fake_gens:
        holdout_idx = np.where(generators == holdout_gen)[0]
        rng = np.random.default_rng(42)
        n_test_real = min(len(holdout_idx), len(real_idx)//3)
        test_real_idx = rng.choice(real_idx, n_test_real, replace=False)
        test_real_set = set(test_real_idx)
        train_real = np.array([i for i in real_idx if i not in test_real_set])
        train_fake = np.where(fake_mask & (generators != holdout_gen))[0]
        tr = np.concatenate([train_real, train_fake])
        te = np.concatenate([test_real_idx, holdout_idx])

        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])

        n_pos = (y[tr]==1).sum(); n_neg = (y[tr]==0).sum()
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=n_neg/n_pos,
                             eval_metric="logloss", random_state=42)
        clf.fit(Xtr, y[tr])
        probs = clf.predict_proba(Xte)[:,1]

        fold_data.append({
            "gen": holdout_gen,
            "y_te": y[te],
            "probs": probs,
        })

    # All predictions pooled
    all_y = np.concatenate([f["y_te"] for f in fold_data])
    all_p = np.concatenate([f["probs"] for f in fold_data])

    # Compute thresholds
    thresholds = {
        "default (0.5)": 0.5,
        "Youden's J": find_youden(all_y, all_p),
        "Max F1": find_max_f1(all_y, all_p),
        "Max Bal.Acc": find_max_balacc(all_y, all_p),
        "Max Accuracy": find_max_acc(all_y, all_p),
    }

    # Also grid of fixed thresholds
    for t in [0.3, 0.35, 0.4, 0.45, 0.55, 0.6]:
        thresholds[f"fixed ({t})"] = t

    # Evaluate each threshold
    results = []
    for name, t in thresholds.items():
        # Pooled metrics
        preds = (all_p >= t).astype(int)
        pooled_auc = roc_auc_score(all_y, all_p)
        pooled_acc = accuracy_score(all_y, preds)
        pooled_ba = balanced_accuracy_score(all_y, preds)
        pooled_f1 = f1_score(all_y, preds)

        # Per-gen metrics
        per_gen = {}
        for fold in fold_data:
            g = fold["gen"]
            yt = fold["y_te"]; pp = fold["probs"]
            pd_gen = (pp >= t).astype(int)
            per_gen[GEN_SHORT[g]] = {
                "auc": roc_auc_score(yt, pp),
                "acc": accuracy_score(yt, pd_gen),
                "balacc": balanced_accuracy_score(yt, pd_gen),
                "f1": f1_score(yt, pd_gen) if len(np.unique(pd_gen)) > 1 else 0,
            }

        gen_aucs = [v["auc"] for v in per_gen.values()]
        gen_accs = [v["acc"] for v in per_gen.values()]
        gen_baccs = [v["balacc"] for v in per_gen.values()]

        results.append({
            "threshold_name": name,
            "threshold": t,
            "auc_mean": np.mean(gen_aucs),
            "auc_min": min(gen_aucs),
            "acc_pooled": pooled_acc,
            "balacc_pooled": pooled_ba,
            "f1_pooled": pooled_f1,
            "acc_mean": np.mean(gen_accs),
            "acc_min": min(gen_accs),
            "balacc_mean": np.mean(gen_baccs),
            "balacc_min": min(gen_baccs),
            **{f"auc_{k}": v["auc"] for k, v in per_gen.items()},
            **{f"acc_{k}": v["acc"] for k, v in per_gen.items()},
            **{f"balacc_{k}": v["balacc"] for k, v in per_gen.items()},
        })

    return results, thresholds

# ── Run ──
print(f"\n{'='*120}")
print("THRESHOLD OPTIMIZATION — XGBoost(spw) on Safe(74D)")
print(f"{'='*120}")

results, thresholds = logo_with_thresholds(X, y, generators)

print(f"\n{'Threshold':20s} {'t':>6s} {'AUC':>7s} {'Acc':>7s} {'BalAcc':>7s} {'F1':>7s} {'AccMin':>7s} {'BAMin':>7s}")
print("-"*80)
for r in results:
    if "fixed" not in r["threshold_name"]:
        print(f"  {r['threshold_name']:18s} {r['threshold']:6.3f} {r['auc_mean']:7.3f} "
              f"{r['acc_pooled']:7.3f} {r['balacc_pooled']:7.3f} {r['f1_pooled']:7.3f} "
              f"{r['acc_min']:7.3f} {r['balacc_min']:7.3f}")

print(f"\n{'='*120}")
print("FIXED THRESHOLDS")
print(f"{'='*120}")
print(f"{'Threshold':20s} {'t':>6s} {'Acc':>7s} {'BalAcc':>7s} {'F1':>7s}")
print("-"*50)
for r in sorted(results, key=lambda x: x["threshold"]):
    print(f"  {r['threshold_name']:18s} {r['threshold']:6.3f} "
          f"{r['acc_pooled']:7.3f} {r['balacc_pooled']:7.3f} {r['f1_pooled']:7.3f}")

# Per-gen breakdown for best strategies
print(f"\n{'='*120}")
print("PER-GENERATOR BREAKDOWN (key strategies)")
print(f"{'='*120}")
gen_order = ["DC","Latte","OSora","Pika","SEINE","SVD","VCraft","ZS"]

for r in results:
    if "fixed" in r["threshold_name"]:
        continue
    name = r["threshold_name"]
    print(f"\n  {name} (t={r['threshold']:.3f}):")
    print(f"    {'Gen':>8s} {'AUC':>7s} {'Acc':>7s} {'BalAcc':>7s}")
    for g in gen_order:
        print(f"    {g:>8s} {r[f'auc_{g}']:7.3f} {r[f'acc_{g}']:7.3f} {r[f'balacc_{g}']:7.3f}")
    print(f"    {'Mean':>8s} {r['auc_mean']:7.3f} {r['acc_mean']:7.3f} {r['balacc_mean']:7.3f}")
    print(f"    {'Min':>8s} {r['auc_min']:7.3f} {r['acc_min']:7.3f} {r['balacc_min']:7.3f}")

df_results = pd.DataFrame(results)
df_results.to_csv(BASE / "results_threshold_optimization.csv", index=False)
print(f"\nSaved to results_threshold_optimization.csv")
