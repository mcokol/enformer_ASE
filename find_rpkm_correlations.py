import numpy as np
import pandas as pd
import h5py


# read enformertracks
enformertracks = pd.read_excel('../input/enformer_tracks.xlsx')
enformertracks = enformertracks[['index', 'description']]
cage_rows = enformertracks[enformertracks["description"].str.contains("cage:", case=False, na=False)].reset_index(drop=True)
cage_tracks = cage_rows['index'].tolist()

genemodel = "refSeq_v20240129"
genemodel = "MANE/1.3"
genemodel = "GENCODE/46/comprehensive/ALL"
genemodel = "GENCODE/46/basic/PRI"
genemodel = genemodel.replace("/", "_")

# read the gene, RPKM, strand, chrom, pos
input_file_name = "../output/" + genemodel + "_singleTSS.txt"
df = pd.read_csv(input_file_name, sep='\t')


def compute_bin_correlations(h5file_path, track_indices, selectedbins, rpkm_series):
    with h5py.File(h5file_path, "r") as h5file:
        # track_indices can be a single int or a list
        variant_data = h5file["D"][:, :, track_indices]
    
    # If a single track is selected, variant_data shape is (N, 896)
    if variant_data.ndim == 2:
        variant_data = variant_data[:, :, np.newaxis]  # shape: (N, 896, 1)
    
    num_variants, _, num_tracks_selected = variant_data.shape
    
    # Build dataframe columns: flatten bins � tracks into columns
    flattened_columns = {}
    
    for t_idx, track in enumerate(np.atleast_1d(track_indices)):
        print(t_idx)
        for b in selectedbins:
            col_name = f"track_{track}_bin_{b}"
            flattened_columns[col_name] = variant_data[:, b, t_idx]
    
    df_bins = pd.DataFrame(flattened_columns)
    
    # Compute correlations
    correlations = {}
    for col in df_bins.columns:
        corr = rpkm_series.corr(df_bins[col])
        correlations[col] = corr
    
    correlations_df = pd.DataFrame(list(correlations.items()), columns=["bin_track", "correlation"])
    return correlations_df

rpkm = pd.to_numeric(df["rpkm"], errors="coerce")


input_file_name = "../output/" + genemodel + "_singleTSS.h5"
correlations_df = compute_bin_correlations(
    input_file_name,
    track_indices=cage_tracks,
    selectedbins=[400, 410, 420, 430, 435, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 455, 460, 470, 480, 490, 500],
    rpkm_series=rpkm
)

print(correlations_df)

# First, split the bin_track column into track and bin parts
correlations_df[["track", "bin"]] = correlations_df["bin_track"].str.extract(r"track_(\d+)_bin_(\d+)")

# Convert to int for clean indexing
correlations_df["track"] = correlations_df["track"].astype(int)
correlations_df["bin"] = correlations_df["bin"].astype(int)

# Pivot the table
pivot_df = correlations_df.pivot(index="track", columns="bin", values="correlation")

# Optional: sort columns in order of bin
pivot_df = pivot_df.sort_index(axis=1)
print(pivot_df)


