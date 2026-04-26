"""
Proper threshold optimization: per LOGO fold, split train into train+val,
find Youden threshold on val, apply on holdout test.
Tuned XGBoost (500/6/0.1/spw) + Safe(74D).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                              f1_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

def find_youden(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return thresholds[np.argmax(tpr - fpr)]

# ── Load & prepare ──
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
X_all_raw = np.hstack([
    df_lat[[c for c in df_lat.columns if c not in meta]].values[valid].astype(np.float32),
    df_phy[[c for c in df_phy.columns if c not in meta]].values[valid].astype(np.float32),
    df_cam[[c for c in df_cam.columns if c not in meta]].values[valid].astype(np.float32),
])
all_names_raw = np.array([c for c in df_lat.columns if c not in meta] +
                          [c for c in df_phy.columns if c not in meta] +
                          [c for c in df_cam.columns if c not in meta])
X_all, all_names = cleanup(X_all_raw, all_names_raw, y)

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
print(f"Safe: {X.shape[1]}D, Total: {len(y)}")

# ── LOGO with proper train/val/test split ──
gen_order = ["DC","Latte","OSora","Pika","SEINE","SVD","VCraft","ZS"]

results_default = []
results_youden = []
fold_info = []

for holdout_gen in fake_gens:
    short = GEN_SHORT[holdout_gen]
    holdout_idx = np.where(generators == holdout_gen)[0]

    # Test real: subsample
    rng = np.random.default_rng(42)
    n_test_real = min(len(holdout_idx), len(real_idx)//3)
    test_real_idx = rng.choice(real_idx, n_test_real, replace=False)
    test_real_set = set(test_real_idx)

    # Train pool: real (minus test real) + fake (minus holdout gen)
    train_real_all = np.array([i for i in real_idx if i not in test_real_set])
    train_fake_all = np.where(fake_mask & (generators != holdout_gen))[0]
    train_pool = np.concatenate([train_real_all, train_fake_all])

    # Split train pool into train (80%) + val (20%)
    strat = np.array(["r" if y[i] == 1 else f"f_{generators[i]}" for i in train_pool])
    pool_idx = np.arange(len(train_pool))
    tr_local, val_local = train_test_split(pool_idx, test_size=0.2, stratify=strat, random_state=42)
    tr_global = train_pool[tr_local]
    val_global = train_pool[val_local]
    te_global = np.concatenate([test_real_idx, holdout_idx])

    # Train model on train only
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr_global])
    Xval = sc.transform(X[val_global])
    Xte = sc.transform(X[te_global])

    n_pos = (y[tr_global]==1).sum(); n_neg = (y[tr_global]==0).sum()
    clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=1.0,
                         scale_pos_weight=n_neg/n_pos,
                         eval_metric="logloss", random_state=42)
    clf.fit(Xtr, y[tr_global])

    # Find threshold on validation
    val_probs = clf.predict_proba(Xval)[:,1]
    youden_t = find_youden(y[val_global], val_probs)

    # Evaluate on holdout test
    test_probs = clf.predict_proba(Xte)[:,1]
    test_auc = roc_auc_score(y[te_global], test_probs)

    preds_default = (test_probs >= 0.5).astype(int)
    preds_youden = (test_probs >= youden_t).astype(int)

    acc_def = accuracy_score(y[te_global], preds_default)
    ba_def = balanced_accuracy_score(y[te_global], preds_default)
    acc_you = accuracy_score(y[te_global], preds_youden)
    ba_you = balanced_accuracy_score(y[te_global], preds_youden)

    fold_info.append({
        "gen": short, "n_train": len(tr_global), "n_val": len(val_global),
        "n_test": len(te_global), "youden_t": youden_t, "auc": test_auc,
        "acc_default": acc_def, "balacc_default": ba_def,
        "acc_youden": acc_you, "balacc_youden": ba_you,
    })

    for yt, pp, pd_d, pd_y in zip(y[te_global], test_probs, preds_default, preds_youden):
        results_default.append((yt, pd_d, short, pp))
        results_youden.append((yt, pd_y, short, pp))

    print(f"  {short:>8s}: train={len(tr_global)} val={len(val_global)} test={len(te_global)}  "
          f"t_youden={youden_t:.3f}  AUC={test_auc:.3f}  "
          f"acc: {acc_def:.3f}->{acc_you:.3f}  balacc: {ba_def:.3f}->{ba_you:.3f}")

# ── Aggregate results ──
def aggregate(results_list, name):
    yt = np.array([r[0] for r in results_list])
    pd_ = np.array([r[1] for r in results_list])
    gs = np.array([r[2] for r in results_list])
    pp = np.array([r[3] for r in results_list])

    auc = roc_auc_score(yt, pp)
    acc = accuracy_score(yt, pd_)
    ba = balanced_accuracy_score(yt, pd_)
    f1 = f1_score(yt, pd_)

    per_gen = {}
    for g in gen_order:
        mask = gs == g
        per_gen[g] = accuracy_score(yt[mask], pd_[mask])

    acc_min = min(per_gen.values())
    return {"name": name, "auc": auc, "acc": acc, "balacc": ba, "f1": f1,
            "acc_min": acc_min, "per_gen": per_gen}

r_def = aggregate(results_default, "default (t=0.5)")
r_you = aggregate(results_youden, "Youden (t on val)")

print(f"\n{'='*120}")
print("SUMMARY (proper train/val/test split)")
print(f"{'='*120}")
print(f"{'Strategy':30s} {'AUC':>7s} {'Acc':>7s} {'BalAcc':>7s} {'F1':>7s} {'AccMin':>7s}  "
      + "  ".join(f"{g:>6s}" for g in gen_order))
print("-"*120)
for r in [r_def, r_you]:
    gen_str = "  ".join(f"{r['per_gen'][g]:6.3f}" for g in gen_order)
    print(f"  {r['name']:28s} {r['auc']:7.4f} {r['acc']:7.4f} {r['balacc']:7.4f} "
          f"{r['f1']:7.4f} {r['acc_min']:7.4f}  {gen_str}")

# Delta
print(f"\n  {'Delta':28s} {'':7s} {r_you['acc']-r_def['acc']:+7.4f} "
      f"{r_you['balacc']-r_def['balacc']:+7.4f} {r_you['f1']-r_def['f1']:+7.4f} "
      f"{r_you['acc_min']-r_def['acc_min']:+7.4f}")

print(f"\n  Per-fold Youden thresholds: "
      + ", ".join(f"{fi['gen']}={fi['youden_t']:.3f}" for fi in fold_info))
print(f"  Mean threshold: {np.mean([fi['youden_t'] for fi in fold_info]):.3f} "
      f"+/- {np.std([fi['youden_t'] for fi in fold_info]):.3f}")
