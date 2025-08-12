# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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
# %%

import matplotlib.pyplot as plt

plt.scatter(result[:, 0], result[:, 2])
plt.xlabel("min_vals")
plt.ylabel("max_vals")
plt.title("Min vs Max values")
plt.grid(True)
plt.show()
