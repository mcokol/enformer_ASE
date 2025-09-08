# %%
import h5py, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatterSciNotation

# ---- config ----
output_dir = "../output"
bin_idx, track_idx = 448, 5110  # set as needed
genemodel = "refSeq_v20240129"
# genemodel = "MANE/1.3"
genemodel = "GENCODE_46_basic_PRI"
genemodel = "GENCODE/46/comprehensive/ALL"

gm_safe = genemodel.replace("/", "_")

gene_tss_file = f"{output_dir}/{gm_safe}_singleTSS.txt"
enformer_predictions_file = f"{output_dir}/{gm_safe}_singleTSS.h5"


# load RPKM
df = pd.read_csv(gene_tss_file, sep="\t")
y = df["rpkm"].to_numpy(dtype=np.float32)

zerorpkm = (df["rpkm"] == 0).sum()
zerorpkmpercent = zerorpkm / len(df) * 100


# load predictions at (bin, track)
with h5py.File(enformer_predictions_file, "r") as f:
    D = f["D"]  # (N, B, T)
    x = D[:, bin_idx, track_idx].astype(np.float32)

# clean + mask (finite only)
x = np.where(np.isinf(x), np.nan, x)
m = np.isfinite(x) & np.isfinite(y)
xv, yv = x[m], y[m]

# Spearman on masked points
rho = spearmanr(xv, yv).correlation if len(xv) > 1 else np.nan
print(f"{gm_safe}: n={len(xv)}, Spearman={rho:.4f}")






###################################################################
# %%
# ---- figure ----
fig, ax = plt.subplots(figsize=(6, 5))

# scatter
ax.scatter(xv, yv, s=10, alpha=0.5)
# linear fit
coef = np.polyfit(xv, yv, 1)   # degree=1 fit
poly1d_fn = np.poly1d(coef)
ax.plot(xv, poly1d_fn(xv), color="green", linewidth=2)
ax.legend()

ax.set_xlim(0, 700)
ax.set_ylim(0, 200)
ax.set_xlabel(f"Prediction (bin={bin_idx}, track={track_idx})")
ax.set_ylabel("RPKM")
stats_text = (
    f"Spearman = {rho:.3f}\n"
    f"n = {len(xv)}\n"
    f"0 RPKM = {zerorpkm} ({zerorpkmpercent:.1f}%)\n"
    f"x=y: red\n"
    f"Linear: green\n"
    f"Lowess: magenta"
    
)
ax.text(
    0.05, 0.95, stats_text,
    transform=ax.transAxes,
    fontsize=10,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)



# identity line y=x spanning data range
if len(xv):
    lo = min(xv.min(), yv.min())
    hi = max(xv.max(), yv.max())
    ax.plot([lo, hi], [lo, hi], "r--")


from statsmodels.nonparametric.smoothers_lowess import lowess

# --- LOESS fit ---
loess_result = lowess(yv, xv, frac=0.2)  # frac = smoothing parameter (0.1 = very wiggly, 0.5 = smoother)
ax.plot(loess_result[:, 0], loess_result[:, 1], color="magenta", linewidth=2, label="LOESS fit")



########## MARGINALS ########################################
# add marginal histogram on top
divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", size=1.2, pad=0.1, sharex=ax)


# --- split the data ---
x_zero = xv[yv == 0]      # predictions for zero-RPKM genes
x_nonzero = xv[yv > 0]    # predictions for non-zero-RPKM genes

# --- histogram data ---
counts_zero, bin_edges = np.histogram(x_zero, bins=50)
counts_nonzero, _ = np.histogram(x_nonzero, bins=bin_edges)  # use same bins
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# --- plot both as step-lines ---
ax_histx.plot(bin_centers, counts_zero, drawstyle="steps-mid", color="red", label="0 RPKM")
ax_histx.plot(bin_centers, counts_nonzero, drawstyle="steps-mid", color="blue", label=">0 RPKM")

ax_histx.set_ylabel("Count")
ax_histx.set_yscale("log")
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histx.legend(loc="upper right", fontsize=8, frameon=False)
ax_histx.set_title(gm_safe)

# add marginal histogram on the right
ax_histy = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)

# histogram data for Y values
counts_y, bin_edges_y = np.histogram(yv, bins=200)
bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

# plot as step-line
ax_histy.plot(counts_y, bin_centers_y, drawstyle="steps-mid", color="black")
ax_histy.set_xlabel("Count")
ax_histy.tick_params(axis="y", labelleft=False)

ax_histy.set_xscale("log")
ax_histy.set_xticks([1e0, 1e1, 1e2, 1e3, 1e4])       # only these ticks
ax_histy.xaxis.set_major_formatter(LogFormatterSciNotation())

######################################################################

plt.tight_layout()
outfile = f"{output_dir}/scatter_{gm_safe}_bin{bin_idx}_track{track_idx}.png"
plt.savefig(outfile, dpi=150)
print(f"Saved: {outfile}")

# %%
