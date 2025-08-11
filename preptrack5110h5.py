import numpy as np
import pandas as pd
import h5py


predictions_dir = "/grid/iossifov/data/real_output/"
n_files = 269
track_index = 5110

all_chunks = []

for i in range(1, n_files + 1):
    print(i)
    file_path = f"{predictions_dir}/bigvariantlist_{i}.txt"
    with h5py.File(file_path, "r") as h5file:
        data = h5file["D"][:, :, :, track_index]  # shape: (N_i, 2, 896)
        all_chunks.append(data)

# Concatenate along first dimension (variants)
merged_data = np.concatenate(all_chunks, axis=0)  # final shape: (~290000, 2, 896)

# Save to a new HDF5 file
output_path = f"../output/enformer_predictions_track{track_index}.h5"
with h5py.File(output_path, "w") as h5out:
    h5out.create_dataset("D", data=merged_data, compression="gzip")

print(f"Merged file written to: {output_path}")
print(f"Shape: {merged_data.shape}")

