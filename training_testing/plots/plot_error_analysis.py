import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
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
    lat, phy, cam = lat[mask].reset_index(drop=True), phy[mask].reset_index(drop=True), cam[mask].reset_index(drop=True)
    generators = generators[mask]

    y = lat["label"].values.astype(int)
    meta = {"path", "label", "_key", "video"}
    X_raw = np.hstack([lat[[c for c in lat.columns if c not in meta]].values.astype(np.float32), phy[[c for c in phy.columns if c not in meta]].values.astype(np.float32), cam[[c for c in cam.columns if c not in meta]].values.astype(np.float32)])
    all_names = np.array([c for c in lat.columns if c not in meta] + [c for c in phy.columns if c not in meta] + [c for c in cam.columns if c not in meta])
    X_clean, clean_names = basic_cleanup(X_raw, y, all_names)
    
    return X_clean, y, generators, clean_names



def safe_filter(X, y, generators, threshold=0.65):
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
    return X[:, gen_mean<=threshold]


def collect_logo_predictions(X, y, generators):
    fake_gens = sorted(set(generators[y==0]))
    real_idx = np.where(y==1)[0]
    fold_data = {}

    for holdout_gen in fake_gens:
        short = GEN_SHORT[holdout_gen]
        hold_idx = np.where(generators==holdout_gen)[0]
        rng = np.random.default_rng(42)
        n_te = min(len(hold_idx), len(real_idx)//3)
        te_real = rng.choice(real_idx, n_te, replace=False)
        tr_real = np.array([i for i in real_idx if i not in set(te_real)])
        tr_fake = np.where((y==0) & (generators!=holdout_gen))[0]
        tr = np.concatenate([tr_real, tr_fake])
        te = np.concatenate([te_real, hold_idx])

        sc = StandardScaler()
        n_pos, n_neg = (y[tr]==1).sum(), (y[tr]==0).sum()
        clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=n_neg/n_pos, eval_metric="logloss", random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        fold_data[short] = {"y": y[te], "probs": clf.predict_proba(sc.transform(X[te]))[:, 1]}
        print(f"{short}: done")

    return fold_data


if __name__ == "__main__":
    X_clean, y, generators, _ = load_and_prepare()
    X = safe_filter(X_clean, y, generators)
    print(f"Safe: {X.shape[1]}D, samples: {len(y)}")

    print("Computing LOGO predictions...")
    fold_data = collect_logo_predictions(X, y, generators)

    gen_order = sorted(fold_data.keys(), key=lambda g: -roc_auc_score(fold_data[g]["y"], fold_data[g]["probs"]))

    C_FN = "#d6604d"
    C_FP = "#2166ac"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fn_rates, fp_rates, aucs = [], [], []
    for g in gen_order:
        yt, pp = fold_data[g]["y"], fold_data[g]["probs"]
        preds = (pp>=0.5).astype(int)
        fn_rates.append(((preds==1) & (yt==0)).sum() / (yt==0).sum() * 100)
        fp_rates.append(((preds==0) & (yt==1)).sum() / (yt==1).sum() * 100)
        aucs.append(roc_auc_score(yt, pp))

    x_pos = np.arange(len(gen_order))
    width = 0.4
    ax.bar(x_pos-width/2, fn_rates, width, color=C_FN, edgecolor="white", linewidth=0.6, label="FN: fake misclassified as real")
    ax.bar(x_pos+width/2, fp_rates, width, color=C_FP, edgecolor="white", linewidth=0.6, label="FP: real misclassified as fake")
    for i, (fn, fp) in enumerate(zip(fn_rates, fp_rates)):
        ax.text(i-width/2, fn+0.5, f"{fn:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(i+width/2, fp+0.5, f"{fp:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{g}\nAUC={a:.3f}" for g, a in zip(gen_order, aucs)], fontsize=10)
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(0, max(fn_rates)*1.18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Error Rates per Unseen Generator (LOGO, $t=0.5$)", pad=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("error_rates_per_gen.pdf", bbox_inches="tight")
    fig.savefig("error_rates_per_gen.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved: error_rates_per_gen")

    all_y = np.concatenate([fold_data[g]["y"] for g in gen_order])
    all_p = np.concatenate([fold_data[g]["probs"] for g in gen_order])

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 50)
    ax.hist(all_p[all_y==0], bins=bins, alpha=0.65, color=C_FN, edgecolor="white", linewidth=0.4, label=f"True Fake (n={int((all_y==0).sum())})")
    ax.hist(all_p[all_y==1], bins=bins, alpha=0.65, color=C_FP, edgecolor="white", linewidth=0.4, label=f"True Real (n={int((all_y==1).sum())})")
    ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.4, alpha=0.8, label="Decision threshold ($t=0.5$)")
    ax.set_xlabel("Predicted $P$(real)")
    ax.set_ylabel("Number of samples")
    ax.legend(loc="upper center", fontsize=10)
    ax.set_title("Distribution of Predicted Probabilities (LOGO, pooled)", pad=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("probability_distribution.pdf", bbox_inches="tight")
    fig.savefig("probability_distribution.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved: probability_distribution")

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for i, g in enumerate(gen_order):
        fpr, tpr, _ = roc_curve(fold_data[g]["y"], fold_data[g]["probs"])
        auc = roc_auc_score(fold_data[g]["y"], fold_data[g]["probs"])
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{g} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("ROC Curves per Unseen Generator (LOGO)", pad=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("roc_curves_per_gen.pdf", bbox_inches="tight")
    fig.savefig("roc_curves_per_gen.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved: roc_curves_per_gen")

