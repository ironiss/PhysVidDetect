import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11, "figure.dpi": 300, "text.usetex": False,
    "axes.axisbelow": True, "axes.labelsize": 12, "axes.titlesize": 13,
})

BASE = Path(__file__).resolve().parent.parent.parent
C_TRAIN = "#2166ac"
C_TEST = "#d6604d"


gap_df = pd.read_csv(BASE / "results_overfit_final_gap.csv")
gap_df = gap_df.sort_values("test", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 5.5))
x_pos = np.arange(len(gap_df))
width = 0.4
ax.bar(x_pos-width/2, gap_df["train"], width, color=C_TRAIN, edgecolor="white", linewidth=0.6, label="Train AUC")
ax.bar(x_pos+width/2, gap_df["test"], width, color=C_TEST, edgecolor="white", linewidth=0.6, label="Test AUC (holdout)")

for i, row in gap_df.iterrows():
    ax.text(i-width/2, row["train"]-0.01, f"{row['train']:.3f}", ha="center", va="top", fontsize=9, fontweight="bold", color="white")
    ax.text(i+width/2, row["test"]+0.002, f"{row['test']:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels([f"{r['gen']}\n(gap={r['gap']:+.3f})" for _, r in gap_df.iterrows()], fontsize=10)
ax.set_ylabel("AUC")
ax.set_ylim(0.93, 1.01)
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", fontsize=10)
ax.set_title("Train vs Test AUC per Unseen Generator (LOGO)", pad=12, fontweight="bold")
fig.tight_layout()
fig.savefig("overfit_train_test_gap.pdf", bbox_inches="tight")
fig.savefig("overfit_train_test_gap.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved: overfit_train_test_gap")


seed_df = pd.read_csv(BASE / "results_overfit_final_seed.csv")

fig, ax = plt.subplots(figsize=(8, 4.5))
x_pos = np.arange(len(seed_df))
ax.scatter(x_pos, seed_df["mean_auc"], s=90, color=C_TRAIN, edgecolor="white", linewidth=1.2, zorder=3, label="Mean AUC")
ax.scatter(x_pos, seed_df["min_auc"], s=90, color=C_TEST, edgecolor="white", linewidth=1.2, zorder=3, label="Min AUC")

mean_m = seed_df["mean_auc"].mean()
mean_n = seed_df["min_auc"].mean()
std_m = seed_df["mean_auc"].std()
std_n = seed_df["min_auc"].std()
ax.axhline(mean_m, color=C_TRAIN, linestyle="--", alpha=0.5, linewidth=1.2)
ax.axhline(mean_n, color=C_TEST, linestyle="--", alpha=0.5, linewidth=1.2)
ax.fill_between([-0.5, len(seed_df)-0.5], mean_m-std_m, mean_m+std_m, color=C_TRAIN, alpha=0.12)
ax.fill_between([-0.5, len(seed_df)-0.5], mean_n-std_n, mean_n+std_n, color=C_TEST, alpha=0.12)

for i, row in seed_df.iterrows():
    ax.text(i, row["mean_auc"]+0.002, f"{row['mean_auc']:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(i, row["min_auc"]-0.002, f"{row['min_auc']:.4f}", ha="center", va="top", fontsize=9, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels([f"seed\n{s}" for s in seed_df["seed"]], fontsize=10)
ax.set_ylabel("AUC")
ax.set_xlim(-0.5, len(seed_df)-0.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="center right", fontsize=10)
ax.set_title(f"Seed Stability: Mean={mean_m:.4f}+-{std_m:.4f}, Min={mean_n:.4f}+-{std_n:.4f}", pad=12, fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig("overfit_seed_stability.pdf", bbox_inches="tight")
fig.savefig("overfit_seed_stability.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved: overfit_seed_stability")


lc_df = pd.read_csv(BASE / "results_overfit_final_lc.csv")

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(lc_df["n_train"], lc_df["mean_auc"], marker="o", markersize=9, linewidth=2, color=C_TRAIN, markeredgecolor="white", markeredgewidth=1.2, label="Mean AUC")
ax.plot(lc_df["n_train"], lc_df["min_auc"], marker="s", markersize=9, linewidth=2, color=C_TEST, markeredgecolor="white", markeredgewidth=1.2, label="Min AUC")

for _, row in lc_df.iterrows():
    ax.text(row["n_train"], row["mean_auc"]+0.0015, f"{row['mean_auc']:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(row["n_train"], row["min_auc"]-0.0015, f"{row['min_auc']:.3f}", ha="center", va="top", fontsize=9, fontweight="bold")

ax.set_xlabel("Number of training samples")
ax.set_ylabel("AUC")
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", fontsize=10)
ax.set_title("Learning Curve: AUC vs Training Set Size", pad=12, fontweight="bold")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
fig.tight_layout()
fig.savefig("overfit_learning_curve.pdf", bbox_inches="tight")
fig.savefig("overfit_learning_curve.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved: overfit_learning_curve")
