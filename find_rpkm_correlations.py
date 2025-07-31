import sys
import h5py
import pandas as pd
import numpy as np

# Redo the parameter parsing through the argparse module
gene_tss_file = "./test_data/GENCODE_46_basic_PRI_singleTSS.txt"
enformer_predictions_file = "./test_data/GENCODE_46_basic_PRI_singleTSS.h5"
output_file_prefix = "GENCODE_46_basic_PRI_singleTSS"

if len(sys.argv) > 1:
    gene_tss_file = sys.argv[1]
if len(sys.argv) > 2:
    enformer_predictions_file = sys.argv[2]

with h5py.File(enformer_predictions_file, "r") as epf:
    enformer_predictions = epf["D"][:]

gene_tss_df = pd.read_csv(gene_tss_file, sep="\t")

N, B, T = enformer_predictions.shape
assert (
    len(gene_tss_df) == N
), "Number of rows in gene TSS file does not match number of predictions in HDF5 file."

for strand in ["-"]:
    strand_indices = gene_tss_df[gene_tss_df["strand"] == strand].index.tolist()
    R = gene_tss_df.rpkm[strand_indices].values
    E = enformer_predictions[strand_indices, :, :]

    # Slow computation for verification
    C_slow = np.zeros((B, T))
    for b in range(B):
        print(b)
        for t in range(T):
            C_slow[b, t] = np.corrcoef(R, E[:, b, t])[0, 1]

    # Vectorized fast computation, but slightly different from the above
    E_centered = E - E.mean(axis=0, keepdims=True)
    R_centered = R - R.mean()

    E_std = E_centered.std(axis=0, ddof=1)
    R_std = R_centered.std(ddof=1)

    N_strand = len(R)

    eps = 0.0  # 1e-12
    numerator = np.tensordot(R_centered, E_centered, axes=([0], [0])) / (N_strand - 1)
    denominator = R_std * E_std + eps

    C = numerator / denominator

    with h5py.File(f"{output_file_prefix}-C{strand}.h5", "w") as f:
        f.create_dataset("C", data=C)
