
# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



# # Load variant info (tab-separated)
# variants = pd.read_csv("../input/ASEvariantinfo.txt", sep="\t")
# print(f"Loaded variants: shape={variants.shape}")
# # print(variants.head())
# # print(variants.columns)
# columns_to_keep = ['number', 'index', 'min', 'max', 'family id', 'study', 
#                    'study phenotype', 'location', 'variant', 
#                    'CHROM', 'POS', 'REF', 'ALT', 'carrier person ids']
# columns_to_keep = ['carrier person ids', 
#                    'CHROM', 'POS', 'REF', 'ALT', 'worst effect', 'genes']

# variants = variants[columns_to_keep]

# Path to your file
file_path = "../output/enformer_predictions_track5110.h5"

# Open the file and read dataset D
with h5py.File(file_path, "r") as h5file:
    data = h5file["D"][:]  # Load entire dataset into a NumPy array

print(f"Loaded dataset 'D' with shape: {data.shape}")