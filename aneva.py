#%%

import pandas as pd

# Read the tab-separated file into a DataFrame
variants = pd.read_csv("../output/extremes100.csv", sep="\t")

# Quick check
print(variants.shape)

# %%
###### get the aneva scores and match with variants file
aneva = pd.read_csv('../input/aneva_250328.txt', sep='\t') 
for index, row in variants.iterrows():
    gene_name = row['genes']
    person = row['carrier person ids']
    
    # Find the relevant value from aneva
    if person in aneva.columns:
        aneva_value = aneva.loc[aneva['geneName'] == gene_name, person]
        if not aneva_value.empty:
            value = aneva_value.values[0]
            # use the value as needed
        else:
            # gene_name not found ? skip or set default
            value = None
    else:
        # person column doesn't exist ? skip or set default
        value = None    
    # If the value exists, add it to the new column; otherwise, add NaN
    variants.at[index, 'aneva'] = aneva_value.values[0] if not aneva_value.empty else None

# Check the result
# variants = variants.dropna(subset=['aneva'])
# print(variants)
print("variants with aneva matches")
print(variants.shape)  
variants.to_csv("../output/extremes100aneva.csv", sep="\t", index=False, header=True)

# %%