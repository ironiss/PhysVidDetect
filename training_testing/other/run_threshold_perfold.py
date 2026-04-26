"""
Threshold optimization per fold: find threshold on train, apply on holdout.
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

# Load
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

# ── LOGO with per-fold threshold ──
gen_order = ["DC","Latte","OSora","Pika","SEINE","SVD","VCraft","ZS"]

# Collect results for 3 approaches:
# 1. default (0.5)
# 2. oracle Youden (threshold found on test = cheating)
# 3. proper Youden (threshold found on train validation split, applied on test)

results_default = []   # (y_true, pred, gen)
results_oracle = []
results_proper = []

fold_thresholds = []

for holdout_gen in fake_gens:
    short = GEN_SHORT[holdout_gen]
    holdout_idx = np.where(generators == holdout_gen)[0]
    rng = np.random.default_rng(42)
    n_test_real = min(len(holdout_idx), len(real_idx)//3)
    test_real_idx = rng.choice(real_idx, n_test_real, replace=False)
    test_real_set = set(test_real_idx)
    train_real = np.array([i for i in real_idx if i not in test_real_set])
    train_fake = np.where(fake_mask & (generators != holdout_gen))[0]
    tr_full = np.concatenate([train_real, train_fake])
    te = np.concatenate([test_real_idx, holdout_idx])

    # Split train into train_inner + val for threshold calibration (80/20)
    strat_tr = np.array(["r" if yi == 1 else "f" for yi in y[tr_full]])
    tr_inner, val_idx = train_test_split(
        np.arange(len(tr_full)), test_size=0.2, stratify=strat_tr, random_state=42)
    tr_inner_global = tr_full[tr_inner]
    val_global = tr_full[val_idx]

    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr_inner_global])
    Xval = sc.transform(X[val_global])
    Xte = sc.transform(X[te])

    n_pos = (y[tr_inner_global]==1).sum(); n_neg = (y[tr_inner_global]==0).sum()
    clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=1.0,
                         scale_pos_weight=n_neg/n_pos,
                         eval_metric="logloss", random_state=42)
    clf.fit(Xtr, y[tr_inner_global])

    # Val predictions -> find threshold
    val_probs = clf.predict_proba(Xval)[:,1]
    youden_val = find_youden(y[val_global], val_probs)

    # Test predictions
    test_probs = clf.predict_proba(Xte)[:,1]
    oracle_t = find_youden(y[te], test_probs)

    fold_thresholds.append({
        "gen": short,
        "val_threshold": youden_val,
        "oracle_threshold": oracle_t,
        "n_train": len(tr_inner_global),
        "n_val": len(val_global),
        "n_test": len(te),
    })

    # Predictions with each strategy
    for yt, pp in zip(y[te], test_probs):
        results_default.append((yt, int(pp >= 0.5), short, pp))
        results_oracle.append((yt, int(pp >= oracle_t), short, pp))
        results_proper.append((yt, int(pp >= youden_val), short, pp))

# ── Print fold thresholds ──
print(f"\n{'='*100}")
print("PER-FOLD THRESHOLDS")
print(f"{'='*100}")
print(f"{'Gen':>8s} {'Val threshold':>14s} {'Oracle threshold':>16s}")
print("-"*42)
for ft in fold_thresholds:
    print(f"{ft['gen']:>8s} {ft['val_threshold']:14.3f} {ft['oracle_threshold']:16.3f}")
val_ts = [ft["val_threshold"] for ft in fold_thresholds]
print(f"{'Mean':>8s} {np.mean(val_ts):14.3f}")
print(f"{'Std':>8s} {np.std(val_ts):14.3f}")

# ── Evaluate all strategies ──
def evaluate(results_list, name):
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
        per_gen[g] = {
            "auc": roc_auc_score(yt[mask], pp[mask]),
            "acc": accuracy_score(yt[mask], pd_[mask]),
            "balacc": balanced_accuracy_score(yt[mask], pd_[mask]),
        }

    acc_min = min(v["acc"] for v in per_gen.values())
    ba_min = min(v["balacc"] for v in per_gen.values())
    gen_acc = "  ".join(f"{per_gen[g]['acc']:.3f}" for g in gen_order)

    print(f"\n  {name}:")
    print(f"    AUC={auc:.4f}  Acc={acc:.4f}  BalAcc={ba:.4f}  F1={f1:.4f}")
    print(f"    Acc min={acc_min:.4f}  BalAcc min={ba_min:.4f}")
    print(f"    Per-gen acc: {gen_acc}")
    return {"name": name, "auc": auc, "acc": acc, "balacc": ba, "f1": f1,
            "acc_min": acc_min, "ba_min": ba_min}

print(f"\n{'='*100}")
print("RESULTS COMPARISON")
print(f"{'='*100}")
print(f"  Generators: {' '.join(f'{g:>7s}' for g in gen_order)}")

r1 = evaluate(results_default, "default (t=0.5)")
r2 = evaluate(results_oracle, "Youden oracle (t on test)")
r3 = evaluate(results_proper, "Youden proper (t on val)")

# ── Summary table ──
print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")
print(f"{'Strategy':30s} {'AUC':>7s} {'Acc':>7s} {'BalAcc':>7s} {'F1':>7s} {'AccMin':>7s}")
print("-"*75)
for r in [r1, r2, r3]:
    print(f"  {r['name']:28s} {r['auc']:7.4f} {r['acc']:7.4f} {r['balacc']:7.4f} {r['f1']:7.4f} {r['acc_min']:7.4f}")
