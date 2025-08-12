
# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



# Load variant info (tab-separated)
variants = pd.read_csv("../input/ASEvariantinfo.txt", sep="\t")
print(f"Loaded variants: shape={variants.shape}")
# print(variants.head())
# print(variants.columns)
columns_to_keep = ['number', 'index', 'min', 'max', 'family id', 'study', 
                   'study phenotype', 'location', 'variant', 
                   'CHROM', 'POS', 'REF', 'ALT', 'carrier person ids']
columns_to_keep = ['carrier person ids', 
                   'CHROM', 'POS', 'REF', 'ALT']

variants = variants[columns_to_keep]

# Path to your file
file_path = "../output/enformer_predictions_track5110.h5"

# Open the file and read dataset D
with h5py.File(file_path, "r") as h5file:
    data = h5file["D"][:]  # Load entire dataset into a NumPy array

print(f"Loaded dataset 'D' with shape: {data.shape}")

# Separate ref (index 0) and alt (index 1)
ref = data[:, 0, :]
alt = data[:, 1, :]

# Difference (alt - ref)
dif = alt - ref  # shape: (variants, bins)

# Calculate stats
min_vals = np.min(dif, axis=1)
min_pos = np.argmin(dif, axis=1)
max_vals = np.max(dif, axis=1)
max_pos = np.argmax(dif, axis=1)

# Stack into result (variants, 4)
result = np.column_stack((min_vals, min_pos, max_vals, max_pos))

print(f"Result shape: {result.shape}")
result_cols = ["dif_min", "dif_min_pos", "dif_max", "dif_max_pos"]

# Add columns to DataFrame
variants[result_cols] = result

print(variants.head())
print(variants.shape)  # should be (268440, original_cols + 4)


# Lowest and highest n
n = 1
lowest_dif_min = variants.nsmallest(n, "dif_min")
highest_dif_max = variants.nlargest(n, "dif_max")
extremes = pd.concat([lowest_dif_min, highest_dif_max]).drop_duplicates()
extremes = extremes.copy()
extremes["variantno"] = extremes.index
extremes = extremes.reset_index(drop=True)

print(f"Lowest dif_min rows: {lowest_dif_min.shape}")
print(f"Highest dif_max rows: {highest_dif_max.shape}")
print(f"Combined extremes shape: {extremes.shape}")

for _, row in extremes.iterrows():

    # Extract metadata
    carrier_id = row["carrier person ids"]
    chrom = row["CHROM"]
    pos = row["POS"]
    ref_base = row["REF"]
    alt_base = row["ALT"]
    vno = row["variantno"]

    variant_data = data[vno, :]

    # variant_data: shape (2, 896)
    ref = variant_data[0, :]
    alt = variant_data[1, :]
    dif = alt - ref

    middlebin = 448

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"hspace": 0.3})
    
    # Add min/max position vertical lines to all three plots
    for ax in axes:
        ax.axvline(row["dif_min_pos"], color="orange", linewidth=1.5, linestyle="--")
        ax.axvline(row["dif_max_pos"], color="green", linewidth=1.5, linestyle="--")
        ax.axvline(middlebin, color="blue", linewidth=1.5, linestyle="--")

    # First two share y-axis, third separate
    
    axes[0].sharey(axes[1])


    # Figure title with neat formatting
    fig.suptitle(
        f"variantno: {vno} | Carrier: {carrier_id} | {chrom}:{pos} {ref_base}>{alt_base}",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Plot REF
    axes[0].plot(ref, color="black", linewidth=3)
    axes[0].set_ylabel("REF")
    axes[0].grid(True)





    # Plot ALT
    axes[1].plot(alt, color="black", linewidth=3)
    axes[1].set_ylabel("ALT")
    axes[1].grid(True)

    # Plot DIF
    axes[2].plot(dif, color="black", linewidth=3)
    axes[2].set_ylabel("DIF")
    axes[2].set_xlabel("Bins")
    axes[2].grid(True)

    # Add 4 rows of text in top-left corner of REF plot
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

    plt.show()

    output_dir = "../output/extremeplots"

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot
    save_path = os.path.join(output_dir, f"variant_{vno}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)  # close to free memory

    print(f"Plot saved to: {save_path}")
    # %%
