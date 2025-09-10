#%%
import pandas as pd

# input: gene model, experimental rpkm (60k genes)
# output: result (rows are genes with single TSS, 6 columns are gene, rpkm, strand, chrom, pos, is_coding)

genemodel = "MANE/1.3"
genemodel = "refSeq_v20240129"
genemodel = "GENCODE/46/basic/PRI"
genemodel = "GENCODE/46/comprehensive/ALL"

genemodelpath = "hg38/gene_models/" + genemodel
print(genemodelpath)

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
        "strand": tm.strand,
        "is_coding": 1 if tm.is_coding() else 0
    })

df = pd.DataFrame(data)
### column pos has TSS
df["pos"] = df.apply(lambda row: row["left"] if row["strand"] == "+" else row["right"], axis=1)
df = df.drop(columns = ['left', 'right'])
### remove noncanonical chromosomes
df = df[df["chrom"].str.len() <= 5]
print("all transcripts")
print(df.shape)

### if a gene is repeated, then it has multiple TSS, drop them #################################
gene_counts = df['gene'].value_counts()
unique_genes = gene_counts[gene_counts == 1].index
df = df[df['gene'].isin(unique_genes)]
print("unique genes")
print(df.shape)

### load the experimental RNA rpkms ############################################################
rpkmdf = pd.read_csv("../input/median_rpkms.txt", sep="\t")

### Perform left merge
### result has five columns gene, rpkm, strand, chrom, pos
result = pd.merge(df, rpkmdf, on="gene", how="left")
print("unique genes with rpkm")
print(result.shape)

# # result size for different gene models
# # genemodel = "MANE/1.3                         19163 all - 18724 with rpkm
# # genemodel = "refSeq_v20240129"                22239 - 10515
# # genemodel = "GENCODE/46/comprehensive/ALL"    37572 - 17711
# # genemodel = "GENCODE/46/basic/PRI"            43234 - 21692

# ### if chromosomes do not match, drop those rows (this didnt happen for these 4 gene models)
result = result[(result["chrom_x"] == result["chrom_y"]) | (result["chrom_y"].isna())]

result = result.drop(columns=["chrom_y"]).rename(columns={"chrom_x": "chrom"})

# renameand order columns to match onenote plan
result = result.rename(columns={'median_rpkm': 'rpkm'})
result = result[['gene', 'rpkm', 'strand', 'chrom', 'pos', 'is_coding']]
print("unique genes with rpkm")
print(result.shape)
### replace / so it isnt interpreted as directory
genemodel = genemodel.replace("/", "_")
outputpath = "../output/" + genemodel + "_singleTSS.txt"
result.to_csv(outputpath, sep="\t", index=False)
# %%
