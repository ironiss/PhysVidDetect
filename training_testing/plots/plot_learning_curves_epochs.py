import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import safe_auc, basic_cleanup

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11, "figure.dpi": 300, "text.usetex": False,
    "axes.axisbelow": True, "axes.labelsize": 12, "axes.titlesize": 13,
})

BASE = Path(__file__).resolve().parent.parent.parent
KNOWN_FAKE = {"DynamicCrafter", "SVD", "ZeroScope", "Pika", "Latte", "OpenSora", "VideoCrafter", "SEINE"}
KNOWN_REAL = {"GenVideo-Real", "GenVideo-Real-clean-3k", "Kinetics-400", "Kinetics-400-additional-4k", "UCF-101-7k"}
GEN_SHORT = {"DynamicCrafter": "DC", "SVD": "SVD", "ZeroScope": "ZS", "Pika": "Pika", "Latte": "Latte", "OpenSora": "OSora", "VideoCrafter": "VCraft", "SEINE": "SEINE"}


def extract_gen(p):
    parts = Path(p).parts
    for i, part in enumerate(parts):
        if part.lower() in ("real", "fake") and i+1<len(parts):
            return parts[i+1]
        
    for part in parts:
        if part in KNOWN_FAKE or part in KNOWN_REAL:
            return part
        
    return "unknown"


def load_and_safe_filter():
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
    lat, phy, cam = lat[mask].reset_index(drop=True), phy[mask].reset_index(drop=True), cam[mask].reset_index(drop=True)
    generators = generators[mask]

    y = lat["label"].values.astype(int)
    meta = {"path", "label", "_key", "video"}
    X_raw = np.hstack([lat[[c for c in lat.columns if c not in meta]].values.astype(np.float32), phy[[c for c in phy.columns if c not in meta]].values.astype(np.float32), cam[[c for c in cam.columns if c not in meta]].values.astype(np.float32)])
    all_names = np.array([c for c in lat.columns if c not in meta] + [c for c in phy.columns if c not in meta] + [c for c in cam.columns if c not in meta])
    X_clean, _ = basic_cleanup(X_raw, y, all_names)

    fake_gens = sorted(set(generators[y==0]))
    X_fake = X_clean[y==0]
    gen_fake = generators[y==0]
    pair_list = []
    for i, g1 in enumerate(fake_gens):
        for g2 in fake_gens[i+1:]:
            pmask = (gen_fake==g1) | (gen_fake==g2)
            y_pair = (gen_fake[pmask]==g1).astype(int)
            pair_list.append(np.array([safe_auc(X_fake[pmask, j], y_pair) for j in range(X_clean.shape[1])]))
    gen_mean = np.mean(pair_list, axis=0)
    X = X_clean[:, gen_mean<=0.65]

    return X, y, generators


if __name__ == "__main__":
    X, y, generators = load_and_safe_filter()
    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]
    n_rounds = 500

    print(f"Safe: {X.shape[1]}D, samples: {len(y)}")
    print("Training LOGO models with per-round tracking...")

    all_train_loss, all_test_loss = [], []
    all_train_auc, all_test_auc = [], []

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
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        n_pos, n_neg = (y[tr]==1).sum(), (y[tr]==0).sum()

        clf = XGBClassifier(n_estimators=n_rounds, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=n_neg/n_pos, eval_metric=["logloss", "auc"], random_state=42)
        clf.fit(Xtr, y[tr], eval_set=[(Xtr, y[tr]), (Xte, y[te])], verbose=False)

        results = clf.evals_result()
        all_train_loss.append(results["validation_0"]["logloss"])
        all_test_loss.append(results["validation_1"]["logloss"])
        all_train_auc.append(results["validation_0"]["auc"])
        all_test_auc.append(results["validation_1"]["auc"])
        print(f"  {GEN_SHORT[holdout_gen]}: done")

    train_loss = np.mean(all_train_loss, axis=0)
    test_loss = np.mean(all_test_loss, axis=0)
    train_auc = np.mean(all_train_auc, axis=0)
    test_auc = np.mean(all_test_auc, axis=0)
    rounds = np.arange(1, n_rounds+1)

    C_TRAIN, C_TEST = "#2166ac", "#d6604d"

    fig, (ax_auc, ax_loss) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax_auc.plot(rounds, train_auc, color=C_TRAIN, linewidth=1.8, label="Train AUC")
    ax_auc.plot(rounds, test_auc, color=C_TEST, linewidth=1.8, label="Test AUC (LOGO)")
    ax_auc.axvline(500, color="#888888", linestyle=":", alpha=0.6, label="Used: 500 trees")
    ax_auc.set_xlabel("Boosting round")
    ax_auc.set_ylabel("AUC")
    ax_auc.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax_auc.spines["top"].set_visible(False)
    ax_auc.spines["right"].set_visible(False)
    ax_auc.legend(loc="lower right", fontsize=10)
    ax_auc.set_title("AUC per Boosting Round", pad=10, fontweight="bold")

    ax_loss.plot(rounds, train_loss, color=C_TRAIN, linewidth=1.8, label="Train LogLoss")
    ax_loss.plot(rounds, test_loss, color=C_TEST, linewidth=1.8, label="Test LogLoss (LOGO)")
    ax_loss.axvline(500, color="#888888", linestyle=":", alpha=0.6, label="Used: 500 trees")
    ax_loss.set_xlabel("Boosting round")
    ax_loss.set_ylabel("Log loss")
    ax_loss.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)
    ax_loss.legend(loc="upper right", fontsize=10)
    ax_loss.set_title("Log Loss per Boosting Round", pad=10, fontweight="bold")

    fig.suptitle("XGBoost Learning Curves (avg. across 8 LOGO folds)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("learning_curve_per_round.pdf", bbox_inches="tight")
    fig.savefig("learning_curve_per_round.png", bbox_inches="tight", dpi=300)
    plt.close(fig)



    pd.DataFrame({"round": rounds, "train_auc": train_auc, "test_auc": test_auc, "train_logloss": train_loss, "test_logloss": test_loss}).to_csv(Path(__file__).resolve().parent.parent / "results" / "learning_curve_per_round.csv", index=False)

    print(f"Final (round 500): Train AUC={train_auc[-1]:.4f}, Test AUC={test_auc[-1]:.4f}, Gap={train_auc[-1]-test_auc[-1]:+.4f}")
    print(f"Best test AUC: {test_auc.max():.4f} at round {np.argmax(test_auc)+1}")
    print("saved: learning_curve_per_round")
