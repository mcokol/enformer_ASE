import sys
import h5py
import pandas as pd
import numpy as np

# Redo the parameter parsing through the argparse module
genemodel = "refSeq_v20240129"
output_dir = "../output"
gene_tss_file = f"{output_dir}/{genemodel}_singleTSS.txt"
enformer_predictions_file = f"{output_dir}/{genemodel}_singleTSS.h5"
output_file_prefix = f"{output_dir}/{genemodel}_singleTSS"

if len(sys.argv) > 1:
    gene_tss_file = sys.argv[1]
if len(sys.argv) > 2:
    enformer_predictions_file = sys.argv[2]

with h5py.File(enformer_predictions_file, "r") as epf:
    # enformer_predictions = epf["D"][:]
    enformer_predictions = epf["D"][:, :, :]

gene_tss_df = pd.read_csv(gene_tss_file, sep="\t")

N, B, T = enformer_predictions.shape
assert (
    len(gene_tss_df) == N
), "Number of rows in gene TSS file does not match number of predictions in HDF5 file."


# E = enformer_predictions[:, 200:300, 200:300]
E = enformer_predictions[:, :, :]

N, B, T = E.shape

for strand in ["-", '+']:
    print(f"Processing strand: {strand}")
    strand_indices = gene_tss_df[gene_tss_df["strand"] == strand].index.tolist()
    R = gene_tss_df.rpkm[strand_indices].values
    R_centered = R - np.nanmean(R)
    R_std = np.nanstd(R_centered, ddof=1)
    N_strand = np.sum(~np.isnan(R))
    
    C_chunks = []
    chunk_size = 10
    for b_start in range(0, B, chunk_size):
        b_end = min(b_start + chunk_size, B)
        print(f"  Processing bins {b_start} to {b_end - 1}...")

        # Load just the chunk from disk
        E_chunk = E[strand_indices, b_start:b_end, :].astype(np.float32)

        # Cleanup
        E_chunk = np.where(np.isinf(E_chunk), np.nan, E_chunk)
        E_chunk = np.clip(E_chunk, -1e6, 1e6)

        # Centering
        E_centered = E_chunk - np.nanmean(E_chunk, axis=0, keepdims=True)
        E_std = np.nanstd(E_centered, axis=0, ddof=1)

        # Correlation computation
        numerator = np.tensordot(
            np.nan_to_num(R_centered),
            np.nan_to_num(E_centered),
            axes=([0], [0])
        ) / (N_strand - 1)

        eps = 1e-12
        denominator = R_std * E_std + eps
        C_chunk = numerator / denominator  # shape: (b_end - b_start, T)

        C_chunks.append(C_chunk)

    # Concatenate chunks along bin axis
    C = np.concatenate(C_chunks, axis=0)  # shape: (B, T)

    strand_name = {"+" : "plus", "-" : "minus"}[strand]
    with h5py.File(f"{output_file_prefix}-Corr_{strand_name}.h5", "w") as f:
        f.create_dataset("C", data=C)
    
    # # Slow computation for verification
    # C_slow = np.zeros((B, T))
    # for b in range(B):
    #     print(b)
    #     for t in range(T):
    #         C_slow[b, t] = np.corrcoef(R, E[strand_indices, b, t])[0, 1]
    
    # # Compute difference
    # diff = C - C_slow
    # # Flatten the matrix to 1D for histogram
    # diff_flat = diff.flatten()
    
    # # Plot histogram
    # plt.figure(figsize=(6, 4))
    # plt.hist(diff_flat, bins=50, edgecolor='black')
    # plt.xlabel("Difference (C - C_slow)")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Differences Between Fast and Slow Correlations")
    # plt.grid(True)
    # plt.show()
    
    
    # # Get indices of where the difference is very large
    # bad_idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    # print(f"Largest diff at {bad_idx}: C = {C[bad_idx]}, C_slow = {C_slow[bad_idx]}")