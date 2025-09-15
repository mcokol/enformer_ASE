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
ref_all = data[:, 0, :]
alt_all = data[:, 1, :]

# --------- select extremes (min/max) ----------
n = 100
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

# ---------------- Load gene models (DAE) ----------------
from dae.genomic_resources.gene_models import build_gene_models_from_resource_id

def gm_to_df(GM):
    """Convert loaded gene model to a DataFrame like your previous gdf."""
    rec = []
    for tm in GM.transcript_models.values():
        tx_start, tx_end = tm.tx
        rec.append({
            "gene": tm.gene,
            "transcript": getattr(tm, "transcript", None),
            "chrom": tm.chrom,
            "start": int(tx_start),
            "end": int(tx_end),
            "strand": tm.strand,
            "tss": int(tx_start) if tm.strand == "+" else int(tx_end),
        })
    return pd.DataFrame(rec)

# Primary gene model: MANE 1.3 (same as before)
GM1_LABEL = "MANE 1.3"
GM1_ID = "hg38/gene_models/MANE/1.3"
GM1 = build_gene_models_from_resource_id(GM1_ID).load()
gdf1 = gm_to_df(GM1)

# Second gene model: RefSeq v20240129
GM2_LABEL = "refSeq_v20240129"
GM2_ID = "hg38/gene_models/refSeq_v20240129"
GM2 = build_gene_models_from_resource_id(GM2_ID).load()
gdf2 = gm_to_df(GM2)


# ---------------- Helpers ----------------
BIN_SIZE = 128
N_BINS = 896
WINDOW_BP = BIN_SIZE * N_BINS
HALF = WINDOW_BP // 2

def coord_to_bin_raw(coord: int, win_start: int) -> float:
    """Return raw (unclipped) bin index for a genomic coordinate relative to window start."""
    return (coord - win_start) / BIN_SIZE

def compute_overlap(gdf, chrom, win_start, win_end):
    """Return transcripts overlapping [win_start, win_end] on chrom, sorted by overlap."""
    on_chr = gdf[gdf["chrom"] == chrom].copy()
    if on_chr.empty:
        return on_chr
    overlap_mask = (on_chr["start"] <= win_end) & (on_chr["end"] >= win_start)
    overlap = on_chr.loc[overlap_mask].copy()
    if overlap.empty:
        return overlap
    overlap[["start", "end"]] = overlap[["start", "end"]].astype(int)
    overlap["overlap_bp"] = (
        np.minimum(overlap["end"], win_end) - np.maximum(overlap["start"], win_start) + 1
    ).clip(lower=0)
    return overlap.sort_values(["overlap_bp", "gene"], ascending=[False, True]).reset_index(drop=True)

def plot_transcript_track(ax, overlap, win_start, model_label, dif_min_pos, dif_max_pos, extreme_type):
    """Draw one transcript track; label y-axis with the gene model name and add half-height markers."""
    ax.set_xlim(0, N_BINS)
    n_tx = max(1, len(overlap))
    ax.set_ylim(0, n_tx)

    if len(overlap) > 0:
        for i, tx in overlap.iterrows():
            gene_name = str(tx["gene"])
            tx_start, tx_end = int(tx["start"]), int(tx["end"])
            tx_strand = str(tx["strand"])

            b_start = coord_to_bin_raw(tx_start, win_start)
            b_end   = coord_to_bin_raw(tx_end,   win_start)

            x0_vis = np.clip(min(b_start, b_end), 0, N_BINS)
            x1_vis = np.clip(max(b_start, b_end), 0, N_BINS)
            y = i + 0.5

            ax.hlines(y, x0_vis, x1_vis, lw=2, color="black")
            ax.text(x0_vis - 5, y, gene_name, ha="right", va="center",
                    fontsize=11, clip_on=False, color="black")

            # Arrowhead at visible edge
            if tx_strand == "+":
                arrow_tip  = x1_vis
                arrow_base = max(arrow_tip - 5, 0)
            else:
                arrow_tip  = x0_vis
                arrow_base = min(arrow_tip + 5, N_BINS)
            ax.annotate("", xy=(arrow_tip, y), xytext=(arrow_base, y),
                        arrowprops=dict(arrowstyle="->", lw=4, color="black"),
                        clip_on=False)

    # Half-height verticals with dashed/solid based on extreme_type
    mid_y = n_tx / 2.0
    ls_min, ls_max = "--", "--"
    if extreme_type == "min":
        ls_min, ls_max = "-", "--"
    elif extreme_type == "max":
        ls_min, ls_max = "--", "-"
    elif extreme_type == "both":
        ls_min, ls_max = "-", "-"

    lw_min = 3 if ls_min == "-" else 1.5
    lw_max = 3 if ls_max == "-" else 1.5

    if not np.isnan(dif_min_pos):
        ax.vlines(dif_min_pos, 0, mid_y, color="red", linestyle=ls_min, linewidth=lw_min)
    if not np.isnan(dif_max_pos):
        ax.vlines(dif_max_pos, mid_y, n_tx, color="green", linestyle=ls_max, linewidth=lw_max)

    # X-axis ticks in base pairs
    tick_bins = np.arange(0, N_BINS + 1, 128)
    tick_bases = win_start + tick_bins * BIN_SIZE
    ax.set_xticks(tick_bins)
    ax.set_xticklabels([f"{int(b):,}" for b in tick_bases])
    ax.set_xlabel("Genomic position (bp)")

    # Label the y-axis with the gene model name (your request)
    ax.set_ylabel(model_label)

    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)

# Ensure output dir
output_dir = "../output/extremeplots"
os.makedirs(output_dir, exist_ok=True)

# %%
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

    # ---- overlaps for both gene models ----
    overlap1 = compute_overlap(gdf1, chrom, win_start, win_end)
    overlap2 = compute_overlap(gdf2, chrom, win_start, win_end)

    # ---------------- Figure & subplots ----------------
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), gridspec_kw={"hspace": 0.4})
    for ax in axes[:3]:
        ax.set_xlim(0, 896)

    # Global title
    fig.suptitle(
        f"variantno: {vno} | {chrom}:{pos} {ref_base}>{alt_base} | extreme_type: {extreme_type}",
        fontsize=14, fontweight="bold", y=0.99
    )

    # --------- Subplot 1: REF ----------
    axes[0].plot(ref, color="black", linewidth=2)
    if not np.isnan(row["dif_min_pos"]):
        axes[0].axvline(row["dif_min_pos"], color="red", linestyle="--", linewidth=1.5)
    if not np.isnan(row["dif_max_pos"]):
        axes[0].axvline(row["dif_max_pos"], color="green", linestyle="--", linewidth=1.5)
    axes[0].set_ylabel("REF")
    axes[0].grid(True)
    yl0, yl1 = axes[0].get_ylim()
    axes[0].plot(448, (yl0 + yl1) / 2, "o", markersize=12, markerfacecolor="none", markeredgecolor="black", zorder=5)

    # --------- Subplot 2: ALT ----------
    axes[1].plot(alt, color="black", linewidth=2)
    if not np.isnan(row["dif_min_pos"]):
        axes[1].axvline(row["dif_min_pos"], color="red", linestyle="--", linewidth=1.5)
    if not np.isnan(row["dif_max_pos"]):
        axes[1].axvline(row["dif_max_pos"], color="green", linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("ALT")
    axes[1].grid(True)
    axes[0].sharey(axes[1])
    yl0, yl1 = axes[1].get_ylim()
    axes[1].plot(448, (yl0 + yl1) / 2, "o", markersize=12, markerfacecolor="none", markeredgecolor="black", zorder=5)

    # --------- Subplot 3: DIF ----------
    axes[2].plot(dif, color="black", linewidth=2)
    if not np.isnan(row["dif_min_pos"]):
        axes[2].axvline(row["dif_min_pos"], color="red", linestyle="--", linewidth=1.5)
    if not np.isnan(row["dif_max_pos"]):
        axes[2].axvline(row["dif_max_pos"], color="green", linestyle="--", linewidth=1.5)
    axes[2].set_ylabel("DIF = ALT - REF")
    axes[2].set_xlabel("Bins")
    axes[2].grid(True)
    yl0, yl1 = axes[2].get_ylim()
    axes[2].plot(448, (yl0 + yl1) / 2, "o", markersize=12, markerfacecolor="none", markeredgecolor="black", zorder=5)

    # ---- Your preferred 4-line text block (kept exactly) ----
    text_str = (
        f"dif_max: {row['dif_max']}\n"
        f"dif_max_pos: {row['dif_max_pos']}\n\n"
        f"dif_min: {row['dif_min']}\n"
        f"dif_min_pos: {row['dif_min_pos']}"
    )
    axes[2].text(
        0.02, 0.95, text_str, transform=axes[2].transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # --------- Subplot 4: Transcript track (MANE 1.3) ----------
    plot_transcript_track(
        ax=axes[3],
        overlap=overlap1,
        win_start=win_start,
        model_label=GM1_LABEL,
        dif_min_pos=row["dif_min_pos"],
        dif_max_pos=row["dif_max_pos"],
        extreme_type=extreme_type,
    )

    # --------- Subplot 5: Transcript track (RefSeq v20240129) ----------
    plot_transcript_track(
        ax=axes[4],
        overlap=overlap2,
        win_start=win_start,
        model_label=GM2_LABEL,
        dif_min_pos=row["dif_min_pos"],
        dif_max_pos=row["dif_max_pos"],
        extreme_type=extreme_type,
    )

    # -------- Save/show --------
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    save_path = os.path.join(output_dir, f"variant_{vno}_{extreme_type}_with_refseq.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Plot saved to: {save_path}")

# %%
