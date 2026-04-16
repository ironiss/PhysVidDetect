from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.preprocessing import StandardScaler


def single_video_dashboard(vggt_output, features, output_path):
    from features import compute_timeseries

    ts = compute_timeseries(vggt_output)
    pos = ts["positions"]
    N = pos.shape[0]
    frames_idx = np.arange(N)

    fig = plt.figure(figsize=(20, 24), constrained_layout=True)
    fig.suptitle("Video Camera Physics — Single Video Dashboard", fontsize=16, y=1.01)
    gs = GridSpec(4, 2, figure=fig)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], "o-", markersize=3, linewidth=1)
    ax3d.scatter(*pos[0], color="green", s=60, zorder=5, label="start")
    ax3d.scatter(*pos[-1], color="red", s=60, zorder=5, label="end")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("Camera trajectory (world)")
    ax3d.legend(fontsize=8)

    ax_pos = fig.add_subplot(gs[0, 1])
    for dim, label, c in zip(range(3), "XYZ", ("tab:blue", "tab:orange", "tab:green")):
        ax_pos.plot(frames_idx, pos[:, dim], label=label, color=c)
    ax_pos.set_xlabel("Frame")
    ax_pos.set_ylabel("Position")
    ax_pos.set_title("Position components")
    ax_pos.legend()

    ax_av = fig.add_subplot(gs[1, 0])
    ax_av.plot(np.arange(len(ts["angular_velocity"])), ts["angular_velocity"], color="tab:purple")
    ax_av.set_xlabel("Frame pair")
    ax_av.set_ylabel("Angle (rad)")
    ax_av.set_title("Angular velocity")

    ax_jit = fig.add_subplot(gs[1, 1])
    ax_jit.plot(np.arange(len(ts["acceleration_mag"])), ts["acceleration_mag"], color="tab:red")
    ax_jit.set_xlabel("Frame triplet")
    ax_jit.set_ylabel("||acceleration||")
    ax_jit.set_title("Position jitter")

    ax_dc = fig.add_subplot(gs[2, 0])
    if len(ts["depth_change"]):
        ax_dc.plot(np.arange(len(ts["depth_change"])), ts["depth_change"], color="tab:cyan")
    ax_dc.set_xlabel("Frame pair")
    ax_dc.set_ylabel("Std of |Δdepth|")
    ax_dc.set_title("Depth temporal change")

    ax_fl = fig.add_subplot(gs[2, 1])
    ax_fl.plot(frames_idx, ts["focal_length"], color="tab:brown")
    ax_fl.set_xlabel("Frame")
    ax_fl.set_ylabel("f_x (px)")
    ax_fl.set_title("Focal length")

    ax_rp = fig.add_subplot(gs[3, 0])
    if len(ts["reprojection_error"]):
        ax_rp.plot(np.arange(len(ts["reprojection_error"])), ts["reprojection_error"], color="tab:olive")
    ax_rp.set_xlabel("Frame pair")
    ax_rp.set_ylabel("Rel. depth error")
    ax_rp.set_title("Reprojection error")

    ax_tbl = fig.add_subplot(gs[3, 1])
    ax_tbl.axis("off")
    rows = sorted(features.items())
    table = ax_tbl.table(
        cellText=[[name, f"{val:.6f}"] for name, val in rows],
        colLabels=["Feature", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    ax_tbl.set_title("Feature summary", pad=12)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def batch_comparison(all_features, labels, names, output_path=None, csv_path=None):
    df = pd.DataFrame(all_features)
    df.insert(0, "label", labels)
    df.insert(1, "video", names)

    feat_cols = [c for c in df.columns if c not in ("label", "video")]

    if csv_path:
        df.to_csv(csv_path, index=False)

    print_comparison_table(df, feat_cols)

    if output_path is None:
        return

    n_feats = len(feat_cols)
    ncols = 3
    nrows = (n_feats + ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), constrained_layout=True)
    fig.suptitle("Feature distributions: Real vs Generated", fontsize=14)
    axes_flat = axes.flatten()

    real_mask = df["label"] == "real"
    gen_mask = df["label"] == "generated"

    for idx, feat in enumerate(feat_cols):
        ax = axes_flat[idx]
        data_r = df.loc[real_mask, feat].dropna().values
        data_g = df.loc[gen_mask, feat].dropna().values
        bp = ax.boxplot(
            [data_r, data_g],
            labels=["real", "generated"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_title(feat, fontsize=9)

    for idx in range(n_feats, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if len(df)>=5:
        scatter_embedding(df, feat_cols, output_path)





def scatter_embedding(df, feat_cols, base_path):
    X = df[feat_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)

    try:
        import umap
        emb = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        method = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        perp = min(30, max(2, len(X)-1))
        emb = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X)
        method = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, color in [("real", "#4C72B0"), ("generated", "#DD8452")]:
        mask = df["label"].values == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1], label=lbl, color=color, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_title(f"{method} projection of features")
    ax.legend()
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")

    stem = Path(base_path).stem
    suffix = Path(base_path).suffix
    emb_path = str(Path(base_path).with_name(f"{stem}_embedding{suffix}"))
    fig.savefig(emb_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




def print_comparison_table(df, feat_cols):
    real = df[df["label"] == "real"]
    gen = df[df["label"] == "generated"]

    header = f"{'Feature':>30s} | {'Real (mean±std)':>20s} | {'Gen (mean±std)':>20s} | {'p-value':>10s} | {'Cohen d':>8s}"
    sep = "-"*len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for feat in feat_cols:
        r_vals = real[feat].dropna().values
        g_vals = gen[feat].dropna().values
        r_mean, r_std = r_vals.mean(), r_vals.std()
        g_mean, g_std = g_vals.mean(), g_vals.std()

        if len(r_vals)>=2 and len(g_vals)>=2:
            _, p = stats.mannwhitneyu(r_vals, g_vals, alternative="two-sided")
            pooled = np.sqrt((r_std**2 + g_std**2)/2) if (r_std + g_std)>0 else 1e-10
            d = abs(r_mean - g_mean)/pooled
        else:
            p, d = float("nan"), float("nan")

        print(
            f"{feat:>30s} | {r_mean:9.4f} ± {r_std:<8.4f} | {g_mean:9.4f} ± {g_std:<8.4f} | "
            f"{p:10.4g} | {d:8.2f}"
        )
    print(sep)


