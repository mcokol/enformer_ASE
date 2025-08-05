import numpy as np
import pandas as pd

import h5py
variants = pd.read_csv('../input/ASEvariantinfo.txt', sep='\t') # Previously b.txt

columns_to_keep = ['number', 'index', 'min', 'max', 'family id', 'study', 
                   'study phenotype', 'location', 'variant', 
                   'CHROM', 'POS', 'REF', 'ALT', 'carrier person ids']

threshold = 10
variants = variants[(variants['max'] > threshold) | (variants['min'] < -1*threshold)][columns_to_keep]
variants['minmax'] = variants[['min', 'max']].apply(lambda x: x['min'] if abs(x['min']) > abs(x['max']) else x['max'], axis=1)
variants = variants.drop(columns=['min', 'max']).sort_values(by='minmax')

####################################################################################
###### get the scores and regions from h5 files
predictions_dir = "/grid/iossifov/data/real_output"
minmax_indices = []
regions = []
for index, row in variants.iterrows():
    filenumber = row['number']
    variantnumber = row['index']
    file_path = f"{predictions_dir}/bigvariantlist_{filenumber}.txt"
    
    with h5py.File(file_path, "r") as h5file:
        variant_data = h5file["D"][variantnumber, :, :, 5110]
    ref_p = variant_data[0,:]
    alt_p = variant_data[1,:]
    
    # Compute the difference
    dif = alt_p - ref_p
    
    # Find the index where dif is closest to the minmax value for this row
    minmax_value = row['minmax']
    
    # Finding the index where dif is closest to the minmax value
    closest_index = (np.abs(dif - minmax_value)).argmin()
    
    # Append the closest_index to the list
    minmax_indices.append(closest_index)
    
    effectpos = row['POS'] + (closest_index - 447) * 128

    regionhalfsize = 1000
    region = str(row['CHROM']) + ":" + str(effectpos - regionhalfsize) + "-" + str(effectpos + regionhalfsize)
    # print(region)
    regions.append(region)


variants['minmaxindex'] = minmax_indices
variants['region'] = regions

print("variants with larger than threshold effect")
print(variants.shape)    

# # data = variants['minmaxindex'].dropna()
# # import matplotlib.pyplot as plt
# # plt.figure(figsize=(8, 6))
# # plt.hist(data, bins=100, color='skyblue', edgecolor='black')
    


#### LEARN THE TSS POSITIONS AND GENE NAME PER TRANSCRIPT
from dae.genomic_resources.gene_models import build_gene_models_from_resource_id
GM = build_gene_models_from_resource_id("hg38/gene_models/MANE/1.3").load() 
# print(dir(GM))
data = []
for tm in GM.transcript_models.values(): #islice(GM.transcript_models.values(), 5):
    data.append({
        "gene": tm.gene,
        "chrom":tm.chrom,
        "left": tm.tx[0],
        "right": tm.tx[1],
        "strand": tm.strand
    })

df = pd.DataFrame(data)
df["tss"] = df.apply(lambda row: row["left"] if row["strand"] == "+" else row["right"], axis=1)

df = df.drop(columns = ['left', 'right', 'strand'])
df = df[df["chrom"].str.len() <= 5]



def annotate_variants_with_genes(variants_df, genes_df):
    """
    For each variant region, find all overlapping genes by TSS, and expand rows
    so each gene gets its own row.

    Parameters:
        variants_df (pd.DataFrame): Must contain a 'region' column in 'chr:start-end' format.
        genes_df (pd.DataFrame): Must contain 'gene', 'chrom', and 'tss' columns.

    Returns:
        pd.DataFrame: A new dataframe where each row contains one gene and one variant.
    """
    genes_df['tss'] = genes_df['tss'].astype(int)

    expanded_rows = []

    for idx, row in variants_df.iterrows():
        chrom, coords = row['region'].split(':')
        start, end = map(int, coords.split('-'))

        matches = genes_df[
            (genes_df['chrom'] == chrom) &
            (genes_df['tss'] >= start) &
            (genes_df['tss'] <= end)
        ]

        if matches.empty:
            new_row = row.copy()
            new_row['gene'] = ''
            expanded_rows.append(new_row)
        else:
            for _, gene_row in matches.iterrows():
                new_row = row.copy()
                new_row['gene'] = gene_row['gene']
                expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)



variants = annotate_variants_with_genes(variants, df)
variants = variants[['CHROM', 'POS', 'carrier person ids', 'minmax', 'minmaxindex', 'region', 'gene']]


print("variants with assigned genes in region")
print(variants.shape)    



####################################################################################
###### get the aneva scores and match with variants file
aneva = pd.read_csv('../input/aneva_250328.txt', sep='\t') 
for index, row in variants.iterrows():
    gene_name = row['gene']
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
variants = variants.dropna(subset=['aneva'])
# print(variants)
print("variants with aneva matches")
print(variants.shape)    


####################################################################################
###### plot minmax vs aneva score

import matplotlib.pyplot as plt
def plot_minmax_vs_aneva(variants_df, jitter_strength=0.01):
    plt.figure(figsize=(8, 6))
    
    # Add jitter
    x_jittered = variants_df['minmax'] + np.random.normal(0, 1000*jitter_strength, size=len(variants_df))
    y_jittered = variants_df['aneva'] + np.random.normal(0, jitter_strength, size=len(variants_df))
    y_jittered = np.clip(y_jittered, 0, 1)

    plt.scatter(x_jittered, y_jittered, alpha=0.7)
    
    
    plt.xlabel('minmax')
    plt.ylabel('aneva')
    plt.title('Scatter plot of minmax vs aneva')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_minmax_vs_aneva(variants)

print('variants with significant aneva matches')
result = variants.loc[variants.aneva < 0.1]

print(result)