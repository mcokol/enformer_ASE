
import pandas as pd

# input: gene model, experimental rpkm (60k genes)
# output: result (rows are genes with single TSS, 5 columns are gene, rpkm, strand, chrom, pos)

genemodel = "MANE/1.3"
genemodel = "refSeq_v20240129"
genemodel = "GENCODE/46/basic/PRI"
# genemodel = "GENCODE/46/comprehensive/ALL"

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

print(df.shape)

### load the experimental RNA rpkms ############################################################
rpkmdf = pd.read_csv("../input/median_rpkms.txt", sep="\t")

### Perform inner merge
### result has five columns gene, rpkm, strand, chrom, pos
result = pd.merge(df, rpkmdf, on="gene", how="inner")
result = result.dropna()
print(result.shape)

# result size for different gene models
# genemodel = "MANE/1.3                         18724
# genemodel = "refSeq_v20240129"                10515
# genemodel = "GENCODE/46/comprehensive/ALL"    17711
# genemodel = "GENCODE/46/basic/PRI"            21692

### if chromosomes do not match, drop those rows (this didnt happen for these 4 gene models)
# Keep only rows where chrom_x equals chrom_y
result = result[result["chrom_x"] == result["chrom_y"]]
result = result.drop(columns=["chrom_y"]).rename(columns={"chrom_x": "chrom"})

# renameand order columns to match onenote plan
result = result.rename(columns={'median_rpkm': 'rpkm'})
result = result[['gene', 'rpkm', 'strand', 'chrom', 'pos']]

### replace / so it isnt interpreted as directory
genemodel = genemodel.replace("/", "_")
outputpath = "../output/" + genemodel + "_singleTSS.txt"
result.to_csv(outputpath, sep="\t", index=False)