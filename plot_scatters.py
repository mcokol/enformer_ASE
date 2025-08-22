# %%
import h5py, pandas as pd, numpy as np, matplotlib.pyplot as plt

# ---- config (adjust these 3 only if needed) ----
genemodel = "GENCODE/46/comprehensive/ALL".replace("/", "_")
# genemodel = "MANE/1.3".replace("/", "_")

output_dir = "../output"
bin_idx, track_idx = 448, 5110   # <— set these

gene_tss_file = f"{output_dir}/{genemodel}_singleTSS.txt"
enformer_predictions_file = f"{output_dir}/{genemodel}_singleTSS.h5"

# load RPKM
df = pd.read_csv(gene_tss_file, sep="\t")
y = df["rpkm"].to_numpy(dtype=np.float32)

# read predictions at (bin, track) for ALL genes
with h5py.File(enformer_predictions_file, "r") as f:
    D = f["D"]              # (N, B, T)
    x = D[:, bin_idx, track_idx].astype(np.float32)

# clean + align
x = np.where(np.isinf(x), np.nan, x)
m = np.isfinite(x) & np.isfinite(y)
xv, yv = x[m], y[m]

from scipy.stats import spearmanr
rho, _ = spearmanr(xv, yv)
print("Spearman correlation:", rho)

# scatter
plt.figure(figsize=(6,5))
plt.scatter(xv, yv, s=10, alpha=0.5)
plt.xlabel(f"Enformer prediction (bin={bin_idx}, track={track_idx})")
plt.ylabel("Observed RPKM")
plt.xscale("log"); plt.yscale("log")
plt.title(f"{genemodel} · Spearman={rho:.3f} · n={len(xv)}")
lo = min(xv.min(), yv.min())
hi = max(xv.max(), yv.max())
plt.plot([lo, hi], [lo, hi], 'r--')

plt.tight_layout()
plt.savefig(f"{output_dir}/{genemodel}_scatter_bin{bin_idx}_track{track_idx}.png", dpi=150)




# %%
