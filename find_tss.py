import pandas as pd

genemodel = "MANE/1.3"
# genemodel = "refSeq_v20240129"
genemodel = "GENCODE/46/comprehensive/ALL"
genemodel = "GENCODE/46/basic/PRI"
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

df = pd.DataFrame(data)
### column pos has TSS
df["pos"] = df.apply(lambda row: row["left"] if row["strand"] == "+" else row["right"], axis=1)
df = df.drop(columns = ['left', 'right'])
### remove noncanonical chromosomes
df = df[df["chrom"].str.len() <= 5]


### if a gene is repeated, then it has multiple TSS, drop them #################################
gene_counts = df['gene'].value_counts()
unique_genes = gene_counts[gene_counts == 1].index
df = df[df['gene'].isin(unique_genes)]

### load the experimental RNA rpkms ############################################################
rpkmdf = pd.read_csv("../input/median_rpkms.txt", sep="\t")

### Perform inner merge
result = pd.merge(df, rpkmdf, on="gene", how="inner")
result = result.dropna()

### if chromosomes do not match, drop those rows
# Keep only rows where chrom_x equals chrom_y
result = result[result["chrom_x"] == result["chrom_y"]]
result = result.drop(columns=["chrom_y"]).rename(columns={"chrom_x": "chrom"})
print(result.shape)

### replace / so it isnt interpreted as directory
genemodel = genemodel.replace("/", "_")
outputpath = "../output/" + genemodel
result.to_csv(outputpath, sep="\t", index=False)