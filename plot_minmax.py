# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# ---------------- I/O ----------------
inputpath = "../output/denovo_minmax_output.txt"
variants = pd.read_csv(inputpath, sep="\t")
print(variants.head())

# Open the file and read dataset D
file_path = "../output/enformer_predictions_track5110.h5"
with h5py.File(file_path, "r") as h5file:
    data = h5file["D"][:]  # Load entire dataset into a NumPy array
print(f"Loaded dataset 'D' with shape: {data.shape}")

# Separate ref (index 0) and alt (index 1)
# (You re-assign within loop for each variant; keeping these for clarity)
ref_all = data[:, 0, :]
alt_all = data[:, 1, :]

# --------- select extremes (min/max) ----------
n = 1
lowest_dif_min = variants.nsmallest(n, "dif_min").copy()
lowest_dif_min["extreme_type"] = "min"
highest_dif_max = variants.nlargest(n, "dif_max").copy()
highest_dif_max["extreme_type"] = "max"

# Concatenate both
extremes = pd.concat([lowest_dif_min, highest_dif_max])

# If a row appears in both, mark as "both"
extremes = extremes.groupby(extremes.index).agg(
    lambda col: "both" if (col.nunique() > 1 and col.name == "extreme_type") else col.iloc[0]
)

# Add variantno = original index
extremes = extremes.copy()
extremes["variantno"] = extremes.index

# Reset index for a clean 0..N
extremes = extremes.reset_index(drop=True)

print(f"Lowest dif_min rows: {lowest_dif_min.shape}")
print(f"Highest dif_max rows: {highest_dif_max.shape}")
print(f"Combined extremes shape: {extremes.shape}")
print(extremes.head)
print("Number of rows with extreme_type == 'both':", (extremes["extreme_type"] == "both").sum())

# ---------------- Load gene models ONCE (DAE) ----------------
from dae.genomic_resources.gene_models import build_gene_models_from_resource_id

GM = build_gene_models_from_resource_id("hg38/gene_models/MANE/1.3").load()

records = []
for tm in GM.transcript_models.values():
    tx_start, tx_end = tm.tx
    records.append({
        "gene": tm.gene,
        "transcript": getattr(tm, "transcript", None),
        "chrom": tm.chrom,
        "start": int(tx_start),
        "end": int(tx_end),
        "strand": tm.strand,
        "tss": int(tx_start) if tm.strand == "+" else int(tx_end),
    })
gdf = pd.DataFrame(records)

# ---------------- Helpers ----------------
BIN_SIZE = 128
N_BINS = 896
WINDOW_BP = BIN_SIZE * N_BINS
HALF = WINDOW_BP // 2

def coord_to_bin_raw(coord: int, win_start: int) -> float:
    """Return raw (unclipped) bin index for a genomic coordinate relative to window start."""
    return (coord - win_start) / BIN_SIZE

# Ensure output dir
output_dir = "../output/extremeplots"
os.makedirs(output_dir, exist_ok=True)

# ---------------- Main plotting loop ----------------
for _, row in extremes.iterrows():
    # Extract metadata
    carrier_id = row["carrier person ids"]
    chrom = str(row["CHROM"])
    pos = int(row["POS"])
    ref_base = row["REF"]
    alt_base = row["ALT"]
    vno = int(row["variantno"])
    extreme_type = row["extreme_type"]

    variant_data = data[vno, :]
    ref = variant_data[0, :]
    alt = variant_data[1, :]
    dif = alt - ref

    middlebin = 448  # middle of 0..895

    # ---- window (114,688 bp centered at POS) ----
    win_start = max(1, pos - HALF)
    win_end = win_start + WINDOW_BP - 1

    # ---- overlap with window on same chromosome ----
    on_chr = gdf[gdf["chrom"] == chrom].copy()
    overlap_mask = (on_chr["start"] <= win_end) & (on_chr["end"] >= win_start)
    overlap = on_chr.loc[overlap_mask].copy()

    # Compute overlap_bp for sorting/plot
    if not overlap.empty:
        overlap[["start","end"]] = overlap[["start","end"]].astype(int)
        overlap["overlap_bp"] = (
            np.minimum(overlap["end"], win_end) - np.maximum(overlap["start"], win_start) + 1
        ).clip(lower=0)
        overlap = overlap.sort_values(["overlap_bp", "gene"], ascending=[False, True]).reset_index(drop=True)

    # ---------------- Figure & subplots ----------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), gridspec_kw={"hspace": 0.35})

    # Add min/max position vertical lines to first three plots (thin dashed; keep your colors)
    for ax in axes[:3]:
        ax.axvline(row["dif_min_pos"], color="orange", linewidth=1.5, linestyle="--")
        ax.axvline(row["dif_max_pos"], color="green", linewidth=1.5, linestyle="--")
        ax.axvline(middlebin, color="blue", linewidth=1.5, linestyle="--")

    # First two share y-axis
    axes[0].sharey(axes[1])

    # Figure title
    fig.suptitle(
        f"variantno: {vno} | Carrier: {carrier_id} | {chrom}:{pos} {ref_base}>{alt_base}",
        fontsize=14, fontweight="bold", y=0.98
    )

    # ---- Plot REF / ALT / DIF (subplots 1–3) ----
    axes[0].plot(ref, color="black", linewidth=3)
    axes[0].set_ylabel("REF")
    axes[0].grid(True)

    axes[1].plot(alt, color="black", linewidth=3)
    axes[1].set_ylabel("ALT")
    axes[1].grid(True)

    axes[2].plot(dif, color="black", linewidth=3)
    axes[2].set_ylabel("DIF")
    axes[2].set_xlabel("Bins")
    axes[2].grid(True)

    # Add 4 rows of text in top-left corner of DIF plot
    text_str = (
        f"dif_min: {row['dif_min']}\n"
        f"dif_min_pos: {row['dif_min_pos']}\n"
        f"dif_max: {row['dif_max']}\n"
        f"dif_max_pos: {row['dif_max_pos']}"
    )
    axes[2].text(
        0.02, 0.98, text_str,
        transform=axes[2].transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # ---- Fourth subplot: gene track with base-pair x-axis ----
    ax = axes[3]
    ax.set_xlim(0, N_BINS)
    # y-limits = number of transcripts (or at least 1 to keep space)
    n_tx = max(1, len(overlap))
    ax.set_ylim(0, n_tx)

    # Plot transcripts if any
    if n_tx > 0:
        for i, tx in overlap.iterrows():
            gene_name = str(tx["gene"])
            tx_start, tx_end = int(tx["start"]), int(tx["end"])
            tx_strand = str(tx["strand"])

            b_start = coord_to_bin_raw(tx_start, win_start)
            b_end = coord_to_bin_raw(tx_end, win_start)

            # Visible portion clipped to plotting window
            x0_vis = np.clip(min(b_start, b_end), 0, N_BINS)
            x1_vis = np.clip(max(b_start, b_end), 0, N_BINS)
            y = i + 0.5

            # Black transcript line
            ax.hlines(y, x0_vis, x1_vis, lw=2, color="black")
            # Gene name left of visible start
            ax.text(x0_vis - 5, y, gene_name, ha="right", va="center", fontsize=11, clip_on=False, color="black")

            # Black arrow at strand end (can be outside window)
            if tx_strand == "+":
                ax.annotate("", xy=(b_end, y), xytext=(b_end - 5, y),
                            arrowprops=dict(arrowstyle="->", lw=1.3, color="black"), clip_on=False)
            else:
                ax.annotate("", xy=(b_start, y), xytext=(b_start + 5, y),
                            arrowprops=dict(arrowstyle="->", lw=1.3, color="black"), clip_on=False)

    # Vertical half-lines for dif_min_pos (red) and dif_max_pos (green)
    mid_y = n_tx / 2.0
    # Default dashed; make the "extreme" side(s) solid & thicker
    ls_min, lw_min = "--", 1.5
    ls_max, lw_max = "--", 1.5
    if extreme_type == "min":
        ls_min, lw_min = "-", 3
    elif extreme_type == "max":
        ls_max, lw_max = "-", 3
    elif extreme_type == "both":
        ls_min, lw_min = "-", 3
        ls_max, lw_max = "-", 3

    if not np.isnan(row["dif_min_pos"]):
        ax.vlines(row["dif_min_pos"], 0, mid_y, color="red", linestyle=ls_min, lw=lw_min)
    if not np.isnan(row["dif_max_pos"]):
        ax.vlines(row["dif_max_pos"], mid_y, n_tx, color="green", linestyle=ls_max, lw=lw_max)

    # Large black dot at the middle bin and vertical center
    mid_bin = N_BINS // 2
    ax.plot(mid_bin, mid_y, "o", color="black", markersize=12, zorder=5)

    # X-axis in base pairs (keep same tick positions as bins)
    tick_bins = np.arange(0, N_BINS + 1, 128)
    tick_bases = win_start + tick_bins * BIN_SIZE
    ax.set_xticks(tick_bins)
    ax.set_xticklabels([f"{int(b):,}" for b in tick_bases])
    ax.set_xlabel("Genomic position (bp)")

    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"{chrom}:{win_start:,}-{win_end:,} (center {pos:,})", fontsize=11)

    # -------- Save per-variant plot --------
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, f"variant_{vno}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Plot saved to: {save_path}")
# %%
