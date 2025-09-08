# %%
import h5py, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ---- config ----
output_dir = "../output"
bin_idx, track_idx = 448, 5110  # set as needed
genemodels = [
    "refSeq_v20240129",
    "MANE/1.3",
    "GENCODE_46_basic_PRI",
    "GENCODE/46/comprehensive/ALL",
]

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, gm in zip(axes, genemodels):
    gm_safe = gm.replace("/", "_")
    gene_tss_file = f"{output_dir}/{gm_safe}_singleTSS.txt"
    enformer_predictions_file = f"{output_dir}/{gm_safe}_singleTSS.h5"

    # load RPKM
    df = pd.read_csv(gene_tss_file, sep="\t")
    y = df["rpkm"].to_numpy(dtype=np.float32)

    # load predictions at (bin, track)
    with h5py.File(enformer_predictions_file, "r") as f:
        D = f["D"]  # (N, B, T)
        x = D[:, bin_idx, track_idx].astype(np.float32)

    # clean + mask (finite only, like your original)
    x = np.where(np.isinf(x), np.nan, x)
    m = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m], y[m]

    # min_x = np.min(xv[xv > 0]) if np.any(xv > 0) else 1.0
    # min_y = np.min(yv[yv > 0]) if np.any(yv > 0) else 1.0
    # xv = np.where(xv == 0, min_x, xv)
    # yv = np.where(yv == 0, min_y, yv)

    # Spearman on masked points
    rho = spearmanr(xv, yv).correlation if len(xv) > 1 else np.nan
    print(f"{gm_safe}: n={len(xv)}, Spearman={rho:.4f}")

    # scatter
    ax.scatter(xv, yv, s=10, alpha=0.5)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 100)

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(f"Prediction (bin={bin_idx}, track={track_idx})")
    ax.set_ylabel("RPKM")
    ax.set_title(f"{gm} · Spearman={rho:.3f} · n={len(xv)}")  # exactly your style

    # identity line y=x spanning data range
    if len(xv):
        lo = min(xv.min(), yv.min())
        hi = max(xv.max(), yv.max())
        ax.plot([lo, hi], [lo, hi], "r--")

plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_2x2_all_models_bin{bin_idx}_track{track_idx}.png", dpi=150)
print(f"Saved: {output_dir}/scatter_2x2_all_models_bin{bin_idx}_track{track_idx}.png")

# %%
