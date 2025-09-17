# %%
import h5py, pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatterSciNotation
from statsmodels.nonparametric.smoothers_lowess import lowess

# ---- config ----
output_dir = "../output"
bin_idx, track_idx = 448, 5110  # set as needed
genemodel = "MANE/1.3"          # or "refSeq_v20240129", etc.
# genemodel = "refSeq_v20240129"
genemodel = "GENCODE/46/comprehensive/ALL"
genemodel = "GENCODE/46/basic/PRI"


gm_safe = genemodel.replace("/", "_")
gene_tss_file = f"{output_dir}/{gm_safe}_singleTSS.txt"
enformer_predictions_file = f"{output_dir}/{gm_safe}_singleTSS.h5"

# ---- load RPKM and coding info ----
df = pd.read_csv(gene_tss_file, sep="\t")
y = df["rpkm"].to_numpy(dtype=np.float32)

zerorpkm = (df["rpkm"] == 0).sum()
zerorpkmpercent = zerorpkm / len(df) * 100

iscoding = (df["is_coding"] == 1).sum()
iscodingpercent = iscoding / len(df) * 100

# ---- load predictions ----
with h5py.File(enformer_predictions_file, "r") as f:
    D = f["D"]  # (N, B, T)
    x = D[:, bin_idx, track_idx].astype(np.float32)

# clean + mask (finite only)
x = np.where(np.isinf(x), np.nan, x)
m = np.isfinite(x) & np.isfinite(y)
xv, yv = x[m], y[m]

# Spearman correlation
rho = spearmanr(xv, yv).correlation if len(xv) > 1 else np.nan
print(f"{gm_safe}: n={len(df)}, Spearman={rho:.4f}")  # n = full dataframe size



###################################################################
# ---- figure ----
fig, ax = plt.subplots(figsize=(9, 5))

# scatter
ax.scatter(xv, yv, s=10, alpha=0.5)

# linear fit
coef = np.polyfit(xv, yv, 1)   # degree=1 fit
poly1d_fn = np.poly1d(coef)
ax.plot(xv, poly1d_fn(xv), color="cyan", linewidth=2, label="Linear fit")

# --- LOESS fit ---
loess_result = lowess(yv, xv, frac=0.2)
ax.plot(loess_result[:, 0], loess_result[:, 1], color="magenta", linewidth=2, label="LOESS fit")

ax.set_xlim(0, 700)
ax.set_ylim(0, 200)
ax.set_xlabel(f"Prediction (bin={bin_idx}, track={track_idx})")
ax.set_ylabel("RPKM")



# categories for stats text
mask_coding     = df["is_coding"] == 1
mask_noncoding  = df["is_coding"] == 0

mask_na = df["rpkm"].isna()
mask_0  = df["rpkm"] == 0
mask_p  = df["rpkm"] > 0

# counts
coding_na  = np.sum(mask_coding & mask_na)
coding_0   = np.sum(mask_coding & mask_0)
coding_p   = np.sum(mask_coding & mask_p)

noncoding_na = np.sum(mask_noncoding & mask_na)
noncoding_0  = np.sum(mask_noncoding & mask_0)
noncoding_p  = np.sum(mask_noncoding & mask_p)

# Compute Spearman for coding only
mask_coding_valid = m & (df["is_coding"] == 1)
if np.sum(mask_coding_valid) > 1:
    rho_coding = spearmanr(x[mask_coding_valid], y[mask_coding_valid]).correlation
else:
    rho_coding = np.nan


stats_text = (
    f"Total (N) = {len(df)}\n"
    f"With RPKM (n) = {np.sum(m)}\n"
    f"\n"
    f"{'':10s}{'NA':>8}{'0':>8}{'>0':>8}\n"
    f"{'coding':10s}{coding_na:8d}{coding_0:8d}{coding_p:8d}\n"
    f"{'noncoding':10s}{noncoding_na:8d}{noncoding_0:8d}{noncoding_p:8d}\n\n"
    f"Spearman (n, {np.sum(m)}) = {rho:.3f}\n"
    f"Spearman (coding, {np.sum(mask_coding_valid)}) = {rho_coding:.3f}\n\n"
    f"Linear: green\n" 
    f"Lowess: magenta"
)



ax.text(
    0.05, 0.95, stats_text,
    transform=ax.transAxes,
    fontsize=9,
    va="top", ha="left",
    family="monospace",   # <<–– add this
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.3)
)

########## MARGINALS ########################################
divider = make_axes_locatable(ax)

# --- top marginal: TAP distributions split into 4 categories ---
ax_histx = divider.append_axes("top", size=1.2, pad=0.1, sharex=ax)

# categories
x_0_coding     = xv[(yv == 0) & (df.loc[m, "is_coding"] == 1)]
x_0_noncoding  = xv[(yv == 0) & (df.loc[m, "is_coding"] == 0)]
x_p_coding     = xv[(yv > 0) & (df.loc[m, "is_coding"] == 1)]
x_p_noncoding  = xv[(yv > 0) & (df.loc[m, "is_coding"] == 0)]
x_na_coding    = x[(df["rpkm"].isna()) & (df["is_coding"] == 1)]
x_na_noncoding = x[(df["rpkm"].isna()) & (df["is_coding"] == 0)]


# histograms with shared bin edges
bin_edges = np.histogram_bin_edges(xv, bins=50)  # use all predictions
counts_0c, _ = np.histogram(x_0_coding, bins=bin_edges)
counts_0n, _ = np.histogram(x_0_noncoding, bins=bin_edges)
counts_pc, _ = np.histogram(x_p_coding, bins=bin_edges)
counts_pn, _ = np.histogram(x_p_noncoding, bins=bin_edges)
counts_na_c, _ = np.histogram(x_na_coding, bins=bin_edges)
counts_na_n, _ = np.histogram(x_na_noncoding, bins=bin_edges)

# normalize within group
counts_0c = counts_0c / counts_0c.sum() if counts_0c.sum() > 0 else counts_0c
counts_0n = counts_0n / counts_0n.sum() if counts_0n.sum() > 0 else counts_0n
counts_pc = counts_pc / counts_pc.sum() if counts_pc.sum() > 0 else counts_pc
counts_pn = counts_pn / counts_pn.sum() if counts_pn.sum() > 0 else counts_pn
counts_na_c = counts_na_c / counts_na_c.sum() if counts_na_c.sum() > 0 else counts_na_c
counts_na_n = counts_na_n / counts_na_n.sum() if counts_na_n.sum() > 0 else counts_na_n

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# plot step-lines
ax_histx.plot(bin_centers, counts_0c, drawstyle="steps-mid", color="red",    label="0 RPKM, coding")
ax_histx.plot(bin_centers, counts_0n, drawstyle="steps-mid", color="orange", label="0 RPKM, noncoding")
ax_histx.plot(bin_centers, counts_pc, drawstyle="steps-mid", color="blue",   label=">0 RPKM, coding")
ax_histx.plot(bin_centers, counts_pn, drawstyle="steps-mid", color="green",  label=">0 RPKM, noncoding")
ax_histx.plot(bin_centers, counts_na_c, drawstyle="steps-mid", color="red",  linestyle=":", label="NA RPKM, coding")
ax_histx.plot(bin_centers, counts_na_n, drawstyle="steps-mid", color="green", linestyle=":",  label="NA RPKM, noncoding")

ax_histx.set_ylabel("Fraction")
ax_histx.set_yscale("log")
ax_histx.tick_params(axis="x", labelbottom=False)
# ax_histx.legend(loc="upper right", fontsize=7, frameon=False)
ax_histx.legend(
    loc="upper right",
    fontsize=7,
    frameon=False,
    ncol=3,         # number of columns
    columnspacing=1 # optional: tighten spacing
)


ax_histx.set_title(gm_safe)
ax_histx.set_ylim(0.0001, 2)
ax_histx.yaxis.grid(True, linestyle="--", alpha=0.7)


# --- right marginal: distribution of RPKM values split into 4 categories ---
ax_histy = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax)

# masks
mask_0c = (yv == 0) & (df.loc[m, "is_coding"] == 1)
mask_0n = (yv == 0) & (df.loc[m, "is_coding"] == 0)
mask_pc = (yv > 0) & (df.loc[m, "is_coding"] == 1)
mask_pn = (yv > 0) & (df.loc[m, "is_coding"] == 0)

# bin edges from all y values
bin_edges_y = np.histogram_bin_edges(yv, bins=200)

# histograms
counts_0c, _ = np.histogram(yv[mask_0c], bins=bin_edges_y)
counts_0n, _ = np.histogram(yv[mask_0n], bins=bin_edges_y)
counts_pc, _ = np.histogram(yv[mask_pc], bins=bin_edges_y)
counts_pn, _ = np.histogram(yv[mask_pn], bins=bin_edges_y)

# normalize within group
counts_0c = counts_0c / counts_0c.sum() if counts_0c.sum() > 0 else counts_0c
counts_0n = counts_0n / counts_0n.sum() if counts_0n.sum() > 0 else counts_0n
counts_pc = counts_pc / counts_pc.sum() if counts_pc.sum() > 0 else counts_pc
counts_pn = counts_pn / counts_pn.sum() if counts_pn.sum() > 0 else counts_pn

bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2

# plot all four groups
ax_histy.plot(counts_0c, bin_centers_y, drawstyle="steps-mid", color="red")
ax_histy.plot(counts_0n, bin_centers_y, drawstyle="steps-mid", color="orange")
ax_histy.plot(counts_pc, bin_centers_y, drawstyle="steps-mid", color="blue")
ax_histy.plot(counts_pn, bin_centers_y, drawstyle="steps-mid", color="green")

ax_histy.set_xlabel("Fraction")
ax_histy.tick_params(axis="y", labelleft=False)
ax_histy.set_xlim(0.0001, 2)
ax_histy.set_xscale("log")

######################################################################
plt.tight_layout()
outfile = f"{output_dir}/scatter_{gm_safe}_bin{bin_idx}_track{track_idx}.png"
plt.savefig(outfile, dpi=150)
print(f"Saved: {outfile}")

# %%
