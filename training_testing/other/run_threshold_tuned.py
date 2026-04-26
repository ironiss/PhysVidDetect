"""Threshold optimization on tuned XGBoost (500/6/0.1) + Safe(74D)."""
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

# Collect LOGO predictions
all_y, all_p, fold_gens = [], [], []
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
    clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=1.0,
                         scale_pos_weight=n_neg/n_pos,
                         eval_metric="logloss", random_state=42)
    clf.fit(Xtr, y[tr])
    probs = clf.predict_proba(Xte)[:,1]
    all_y.extend(y[te]); all_p.extend(probs)
    fold_gens.extend([holdout_gen]*len(te))

all_y = np.array(all_y); all_p = np.array(all_p); fold_gens = np.array(fold_gens)
print(f"Predictions collected: {len(all_y)}")

# Find thresholds
fpr, tpr, th_roc = roc_curve(all_y, all_p)
youden_t = th_roc[np.argmax(tpr - fpr)]

prec, rec, th_pr = precision_recall_curve(all_y, all_p)
f1s = 2*prec*rec/(prec+rec+1e-10)
maxf1_t = th_pr[np.argmax(f1s[:-1])]

best_ba_t, best_ba = 0.5, 0
best_acc_t, best_acc = 0.5, 0
for t in np.linspace(0.1, 0.95, 500):
    pd_t = (all_p >= t).astype(int)
    ba = balanced_accuracy_score(all_y, pd_t)
    ac = accuracy_score(all_y, pd_t)
    if ba > best_ba: best_ba = ba; best_ba_t = t
    if ac > best_acc: best_acc = ac; best_acc_t = t

strategies = {
    "default (0.5)": 0.5,
    "Youden J": youden_t,
    "Max F1": maxf1_t,
    "Max Bal.Acc": best_ba_t,
    "Max Accuracy": best_acc_t,
}

gen_order = ["DC","Latte","OSora","Pika","SEINE","SVD","VCraft","ZS"]

print(f"\n{'='*160}")
print("THRESHOLD OPTIMIZATION — Tuned XGBoost (500/6/0.1/spw) on Safe(74D)")
print(f"{'='*160}")
header = f"{'Strategy':20s} {'t':>6s} {'AUC':>7s} {'Acc':>7s} {'BalAcc':>7s} {'F1':>7s} {'AccMin':>7s} {'BAMin':>7s}  "
header += "  ".join(f"{g:>6s}" for g in gen_order)
print(header)
print("-"*160)

for name, t in strategies.items():
    preds = (all_p >= t).astype(int)
    auc = roc_auc_score(all_y, all_p)
    acc = accuracy_score(all_y, preds)
    ba = balanced_accuracy_score(all_y, preds)
    f1 = f1_score(all_y, preds)

    per_gen_acc = {}
    per_gen_ba = {}
    for g in fake_gens:
        mask = fold_gens == g
        yt = all_y[mask]; pp = all_p[mask]; pd_g = (pp >= t).astype(int)
        gs = GEN_SHORT[g]
        per_gen_acc[gs] = accuracy_score(yt, pd_g)
        per_gen_ba[gs] = balanced_accuracy_score(yt, pd_g)

    acc_min = min(per_gen_acc.values())
    ba_min = min(per_gen_ba.values())
    gen_str = "  ".join(f"{per_gen_acc[g]:6.3f}" for g in gen_order)

    print(f"{name:20s} {t:6.3f} {auc:7.4f} {acc:7.4f} {ba:7.4f} {f1:7.4f} {acc_min:7.4f} {ba_min:7.4f}  {gen_str}")
