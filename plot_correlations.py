import numpy as np
import pandas as pd
import h5py


# read enformertracks
enformertracks = pd.read_excel('../input/enformer_tracks.xlsx')
enformertracks = enformertracks[['index', 'description']]
cage_rows = enformertracks[enformertracks["description"].str.contains("cage:", case=False, na=False)].reset_index(drop=True)
cage_tracks = cage_rows['index'].tolist()

genemodel = "refSeq_v20240129"
# genemodel = "MANE/1.3"
# genemodel = "GENCODE/46/comprehensive/ALL"
# genemodel = "GENCODE/46/basic/PRI"
genemodel = genemodel.replace("/", "_")


# input_file_name = "../output/" + genemodel + "_singleTSS.txt"
# df = pd.read_csv(input_file_name, sep='\t')


### read correlations, 5313 tracks as rows, 896 tracks 
input_file_name = "../output/" + genemodel + "_plus_cage_correlations.txt"
df2 = pd.read_csv(input_file_name, sep='\t')


# Get the row and column of the minimum value
min_row, min_col = df2.stack().idxmin()

# Get the row and column of the maximum value
max_row, max_col = df2.stack().idxmax()

min_value = df2.min().min()  # Minimum value across all columns
max_value = df2.max().max()  # Maximum value across all columns

# Print the results
print(f"Minimum value: {min_value} at row {min_row}, column {min_col}")
print(f"Maximum value: {max_value} at row {max_row}, column {max_col}")



# df2 = df2.iloc[:, ::10]

import matplotlib.pyplot as plt

# Extract row 5110
row_5110 = df2.iloc[493]  # Indexing is 0-based, so row 5110 is at index 5109

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(row_5110.index, row_5110.values)
plt.xlabel('Bins')
plt.ylabel('Values')
plt.title('Plot of Row 5110')
plt.grid(True)
plt.show()