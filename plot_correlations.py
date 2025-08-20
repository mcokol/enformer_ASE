# %%
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np

genemodel = "refSeq_v20240129"
# genemodel = 'refSeq_v20240129_spearman'
genemodel = "MANE/1.3"

### for variants13 and 14
# genemodel = 'variants13'


genemodel = genemodel + '_spearman'
genemodel = genemodel.replace("/", "_")
strands = ["plus", "minus"]
# strands = ["plus"]

# read enformertracks
enformertracks = pd.read_excel('../input/enformer_tracks.xlsx')
enformertracks = enformertracks[['index', 'description']]
cage_rows = enformertracks[enformertracks["description"].str.contains("cage:", case=False, na=False)].reset_index(drop=True)
cage_tracks = cage_rows['index'].tolist()
LCL_row = enformertracks[enformertracks["description"].str.contains("CAGE:B lymphoblastoid cell line", case=False, na=False)].reset_index(drop=True)
LCL_track = LCL_row['index'].tolist()[0]


plotname = "_Extremes"
plotname = "_BALANCED"
plotname = '_ALL'




# --- Plot only LCL CAGE correlation for all bins
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for i, strand in enumerate(strands):
    input_file_name = f"../output/{genemodel}_singleTSS-Corr_{strand}{plotname}.h5"
    
    with h5py.File(input_file_name, "r") as epf:
        correlations = epf["C"][:]
    
    axes[i].plot(correlations[:, LCL_track])
    axes[i].set_ylabel("Correlation")
    axes[i].set_title(f"LCL CAGE correlation in {strand} strand")
    axes[i].grid(True)

axes[1].set_xlabel("Bin index")
plt.tight_layout()
output_plot_file = f"../output/{genemodel}_LCL_CAGE_correlation_plot{plotname}.png"
plt.savefig(output_plot_file, dpi=300)
plt.show()



# --- Plot LCL CAGE correlation and all other CAGE experiments in less bins
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
bin_start, bin_end = 400, 500

for i, strand in enumerate(strands):
    input_file_name = f"../output/{genemodel}_singleTSS-Corr_{strand}{plotname}.h5"
    
    with h5py.File(input_file_name, "r") as epf:
        correlations = epf["C"][:]  # shape: (bins, tracks)
    
    ax = axes[i]
    # Plot all CAGE tracks in light gray
    for track_index in cage_tracks:
        ax.plot(correlations[bin_start:bin_end, track_index], color="lightgray", linewidth=0.8, alpha=0.7)

    # Plot LCL track in dark blue
    ax.plot(correlations[bin_start:bin_end, LCL_track], color="navy", linewidth=2, label="LCL")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{strand} strand")
    ax.set_xticks(np.arange(0, bin_end - bin_start, 10))
    ax.set_xticklabels(np.arange(bin_start, bin_end, 10))
    ax.axvline(x=448 - bin_start, color="red", linestyle="--", linewidth=1)
    ax.grid(True)

axes[1].set_xlabel("Bin index")
plt.tight_layout()
output_plot_file = f"../output/{genemodel}_CAGE_correlation_plot{plotname}.png"
plt.savefig(output_plot_file, dpi=300)
plt.show()


# --- Plot all tracks
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
bin_start, bin_end = 400, 500

for i, strand in enumerate(strands):
    input_file_name = f"../output/{genemodel}_singleTSS-Corr_{strand}{plotname}.h5"
    
    with h5py.File(input_file_name, "r") as epf:
        correlations = epf["C"][:]  # shape: (bins, tracks)
    
    ax = axes[i]
    ax.plot(correlations[bin_start:bin_end, :], color="yellow", linewidth=0.8, alpha=0.7)
    # Plot all CAGE tracks in light gray
    for track_index in cage_tracks:
        ax.plot(correlations[bin_start:bin_end, track_index], color="lightgray", linewidth=0.8, alpha=0.7)

    # Plot LCL track in dark blue
    ax.plot(correlations[bin_start:bin_end, LCL_track], color="navy", linewidth=2, label="LCL")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{strand} strand")
    ax.set_xticks(np.arange(0, bin_end - bin_start, 10))
    ax.set_xticklabels(np.arange(bin_start, bin_end, 10))
    ax.axvline(x=448 - bin_start, color="red", linestyle="--", linewidth=1)
    ax.grid(True)

axes[1].set_xlabel("Bin index")
plt.tight_layout()
output_plot_file = f"../output/{genemodel}_ALL_correlation_plot{plotname}.png"
plt.savefig(output_plot_file, dpi=300)
plt.show()
# %%
