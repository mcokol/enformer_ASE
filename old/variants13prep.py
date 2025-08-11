
import pandas as pd

# Load the file
df = pd.read_csv("../input/variants13.txt", sep="\t")

# Split 'position' into 'chrom' and 'pos'
df[['chrom', 'pos']] = df['position'].str.split(":", expand=True)

# Reorder columns
df = df[['chrom', 'pos', 'desc']]

# Save to a new file (optional)
df.to_csv("variants13_processed.txt", sep="\t", index=False)

# Show the result
print(df)


genemodel = "refSeq_v20240129"
genemodelpath = "hg38/gene_models/" + genemodel

from dae.genomic_resources.gene_models import build_gene_models_from_resource_id
GM = build_gene_models_from_resource_id(genemodelpath).load() 

### get all transcripts in gene model ###########################################################
data = []
for tm in GM.transcript_models.values():
    data.append({
        "gene": tm.gene,
        "chrom":tm.chrom,
        "left": tm.tx[0],
        "right": tm.tx[1],
        "strand": tm.strand
    })

GMdf = pd.DataFrame(data)

GMdf["pos"] = GMdf.apply(lambda row: row["left"] if row["strand"] == "+" else row["right"], axis=1)
GMdf = GMdf.drop(columns = ['left', 'right'])

df['pos'] = df['pos'].astype(int)
GMdf['pos'] = GMdf['pos'].astype(int)
merged = pd.merge(df, GMdf, on=["chrom", "pos"], how="inner")
merged = merged.drop_duplicates(subset=["chrom", "pos"])




### load the experimental RNA rpkms ############################################################
rpkmdf = pd.read_csv("../input/median_rpkms.txt", sep="\t")

### Perform inner merge
### result has five columns gene, rpkm, strand, chrom, pos
result = pd.merge(merged, rpkmdf, on="gene", how="inner")
result = result.dropna()

# Drop unwanted columns
result = result.drop(columns=["chrom_y", "desc"])

# Rename columns
result = result.rename(columns={"chrom_x": "chrom", "median_rpkm": "rpkm"})

# Reorder columns
result = result[["gene", "rpkm", "strand", "chrom", "pos"]]

outputpath = "../output/variants13_singleTSS.txt"
result.to_csv(outputpath, sep="\t", index=False)