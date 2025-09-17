
import sys
import h5py
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import time



# Redo the parameter parsing through the argparse module
genemodel = "refSeq_v20240129"
genemodel = "MANE/1.3"
# genemodel = "GENCODE_46_basic_PRI"
# genemodel = "GENCODE/46/comprehensive/ALL"

genemodel = genemodel.replace("/", "_")

output_dir = "../output"
gene_tss_file = f"{output_dir}/{genemodel}_singleTSS.txt"
enformer_predictions_file = f"{output_dir}/{genemodel}_singleTSS.h5"
output_file_prefix = f"{output_dir}/{genemodel}_spearman_singleTSS"

# ### for variants13 experiments
# gene_tss_file = '../output/variants13_singleTSS.txt'
# enformer_predictions_file = '../output/variants13_singleTSS.h5'
# output_file_prefix = '../output/variants13_spearman_singleTSS2'

print(gene_tss_file)
print(enformer_predictions_file)
print(output_file_prefix)


if len(sys.argv) > 1:
    gene_tss_file = sys.argv[1]
if len(sys.argv) > 2:
    enformer_predictions_file = sys.argv[2]

gene_tss_df = pd.read_csv(gene_tss_file, sep="\t")

# allows to select some of the genes for correlation analysis - not used
def select_genes(gene_tss_df, strand, method="ALL", n=100, random_state=42):
    strand_df = gene_tss_df[gene_tss_df["strand"] == strand].copy()

    if method.upper() == "ALL":
        return strand_df.index.tolist()

    elif method.upper() == "EXTREMES":
        # Lowest n
        lowest = strand_df.nsmallest(n, "rpkm")
        # Highest n
        highest = strand_df.nlargest(n, "rpkm")
        # Median n
        median_rpkm = strand_df["rpkm"].median()
        strand_df["dist"] = (strand_df["rpkm"] - median_rpkm).abs()
        median = strand_df.nsmallest(n, "dist")
        # Combine and remove duplicates (in case of ties)
        selected = pd.concat([lowest, highest, median]).drop_duplicates()
        return selected.index.tolist()
    
    elif method.upper() == "BALANCED":
        strand_df = strand_df.sort_values("rpkm")
        quantiles = np.linspace(0, 1, 11)  # 0%, 10%, ..., 100%
        bins = strand_df["rpkm"].quantile(quantiles).values
        selected_indices = []
        for i in range(10):
            lower, upper = bins[i], bins[i + 1]
            bin_df = strand_df[(strand_df["rpkm"] >= lower) & (strand_df["rpkm"] <= upper)]
            if len(bin_df) >= n:
                selected = bin_df.sample(n=n, random_state=random_state)
            else:
                selected = bin_df  # if not enough, take all
            selected_indices.extend(selected.index.tolist())

        print(f"BALANCED selection for strand {strand}: {len(selected_indices)} genes selected.")
        return selected_indices
    else:
        raise ValueError(f"Unknown selection method: {method}")




with h5py.File(enformer_predictions_file, "r") as epf:
    D = epf["D"]
    N, B, T = D.shape
    assert (len(gene_tss_df) == N), "Number of rows in gene TSS file does not match number of predictions in HDF5 file."

    for strand in ['+', '-']:
        print(f"Processing strand: {strand}")
        mymethod = 'ALL' # 'BALANCED' #'EXTREMES'
        strand_indices = select_genes(gene_tss_df, strand, method=mymethod, n=100)
        R = gene_tss_df.loc[strand_indices, "rpkm"].values
            
        R_ranked = rankdata(R, method='average')
        R_centered = R_ranked - np.nanmean(R_ranked)
        R_std = np.nanstd(R_centered, ddof=1)
        N_strand = np.sum(~np.isnan(R))
        
        C_chunks = []
        chunk_size = 10
        for b_start in range(0, B, chunk_size):
            b_end = min(b_start + chunk_size, B)
            print(f"  Processing bins {b_start} to {b_end - 1}...")
            start_time = time.time()
            # Load just the chunk from disk
            E_chunk = D[strand_indices, b_start:b_end, :].astype(np.float32)
            # Cleanup
            E_chunk = np.where(np.isinf(E_chunk), np.nan, E_chunk)
            E_chunk = np.clip(E_chunk, -1e6, 1e6)

            E_ranked = rankdata(E_chunk, axis=0, method='average')

            # E_ranked = np.apply_along_axis(rankdata, 0, E_chunk)
            # Centering
            E_centered = E_ranked - np.nanmean(E_ranked, axis=0, keepdims=True)
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
            execution_time = time.time() - start_time
            print(f"Execution Time: {execution_time:.2f} seconds")

        # Concatenate chunks along bin axis
        C = np.concatenate(C_chunks, axis=0)  # shape: (B, T)


        strand_name = {"+" : "plus", "-" : "minus"}[strand]
        with h5py.File(f"{output_file_prefix}-Corr_{strand_name}_{mymethod}.h5", "w") as f:
            f.create_dataset("C", data=C)


# # with h5py.File(enformer_predictions_file, "r") as epf:
# with h5py.File(enformer_predictions_file, "r") as epf:
#     D = epf["D"]
#     N, B, T = D.shape
#     assert (len(gene_tss_df) == N), "Number of rows in gene TSS file does not match number of predictions in HDF5 file."

#     for strand in ['+', '-']:
#         print(f"\n=== Starting strand: {strand} ===")
#         t0_strand = time.perf_counter()

#         mymethod = 'ALL' # 'BALANCED' #'EXTREMES'
#         strand_indices = select_genes(gene_tss_df, strand, method=mymethod, n=100)
#         R = gene_tss_df.loc[strand_indices, "rpkm"].values
            
#         R_ranked = rankdata(R, method='average')
#         R_centered = R_ranked - np.nanmean(R_ranked)
#         R_std = np.nanstd(R_centered, ddof=1)
#         N_strand = np.sum(~np.isnan(R))

#         # Preallocate output
#         C = np.empty((B, T), dtype=np.float32)
#         chunk_size = 20

#         for b_start in range(0, B, chunk_size):
#             t0_chunk = time.perf_counter()
#             b_end = min(b_start + chunk_size, B)

#             # Load just the chunk from disk
#             E_chunk = D[strand_indices, b_start:b_end, :].astype(np.float32)
#             E_chunk = np.where(np.isinf(E_chunk), np.nan, E_chunk)
#             E_chunk = np.clip(E_chunk, -1e6, 1e6)

#             E_ranked = rankdata(E_chunk, axis=0, method='average')

#             # Centering
#             E_centered = E_ranked - np.nanmean(E_ranked, axis=0, keepdims=True)
#             E_std = np.nanstd(E_centered, axis=0, ddof=1)

#             # Correlation computation
#             numerator = np.tensordot(
#                 np.nan_to_num(R_centered),
#                 np.nan_to_num(E_centered),
#                 axes=([0], [0])
#             ) / (N_strand - 1)
#             eps = 1e-12
#             denominator = R_std * E_std + eps
#             C[b_start:b_end, :] = numerator / denominator  # shape: (chunk, T)

#             t_chunk = time.perf_counter() - t0_chunk
#             print(f"  Finished bins {b_start}-{b_end-1} / {B} "
#                   f"(chunk time {t_chunk:.2f}s)")

#         strand_name = {"+" : "plus", "-" : "minus"}[strand]
#         with h5py.File(f"{output_file_prefix}-Corr_{strand_name}_{mymethod}.h5", "w") as f:
#             f.create_dataset("C", data=C)

#         elapsed_strand = time.perf_counter() - t0_strand
#         print(f"=== Finished strand {strand} in {elapsed_strand:.1f}s ===")

