import sys

refgenomeid = "hg38/genomes/GRCh38-hg38"
section_size = 5
model_path = "https://tfhub.dev/deepmind/enformer/1"

genemodel = "refSeq_v20240129"
genemodel = "MANE/1.3"
genemodel = "GENCODE/46/basic/PRI"
genemodel = 'GENCODE_46_comprehensive_ALL'

genemodel = genemodel.replace("/", "_")
input_file_name = "../output/" + genemodel + "_singleTSS.txt"
output_file_name = "../output/" + genemodel + "_singleTSS.h5"

### this is for custom lists (variants13 and 14 from before)
input_file_name = "../output/variants13_singleTSS.txt"
output_file_name = "../output/variants13_singleTSS.h5"

print(input_file_name)
print(output_file_name)

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import time
import os
from dae.genomic_resources.reference_genome import build_reference_genome_from_resource_id
import h5py

refgenome = build_reference_genome_from_resource_id(refgenomeid).open()
INPUT_SEQ_LEN = 393_216
model = hub.load(model_path).model
num_bins = 896
num_tracks = 5313

# -------------------- helpers --------------------
#########################################################################################################
def get_sequence(refgenome, chrom, position, window_size=INPUT_SEQ_LEN):
    pos = int(position)
    start = pos - (window_size // 2)
    end = pos + (window_size // 2)
    chr_length = refgenome.get_chrom_length(chrom)
    pad_left = max(0, 1 - start)
    pad_right = max(0, end - chr_length)
    start = max(1, start)
    end = min(chr_length, end)
    seq = refgenome.get_sequence(chrom, start, end)
    seq = ('N' * pad_left) + seq + ('N' * pad_right)
    return seq[:-1]  # keep identical to your original


#########################################################################################################
### returns a seq_length x 4 binary matrix
def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: any = 0,
                   dtype=np.float32) -> np.ndarray:
  """One-hot encode sequence."""
  def to_uint8(string):
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  return hash_table[to_uint8(sequence)]

#########################################################################################################
def modelpredict(sequences):
    # One-hot encode each sequence and stack into a batch
    seq_one_hot_batch = np.stack([one_hot_encode(seq) for seq in sequences])
    # Batch prediction
    prediction = model.predict_on_batch(seq_one_hot_batch)
    return prediction['human'].numpy()

#########################################################################################################
# -------------------- read input once to know N --------------------
import csv
with open(input_file_name, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    rows = [row for row in reader][1:]  # skip header
num_variants = len(rows)
num_sections = num_variants // section_size

with h5py.File(output_file_name, "w") as h5:
    dset = h5.create_dataset(
        "D",
        shape=(num_variants, num_bins, num_tracks),
        dtype="f2"   # float16 on disk, same as before
    )

    start_idx = 0
    sequences = []
    section_counter = 0

    for row in rows:
        chrom, position = row[-2:]
        sequences.append(get_sequence(refgenome, chrom, position))

        if len(sequences) == section_size:
            start_time = time.time()
            batch = modelpredict(sequences)                     # float32
            dset[start_idx:start_idx+len(sequences)] = np.asarray(batch, np.float16)
            start_idx += len(sequences)
            sequences.clear()
            section_counter += 1
            # print(section)
            execution_time = time.time() - start_time
            print(f"Section: {section_counter+1} of {num_sections}, Execution Time: {execution_time:.2f} seconds")

    if sequences:
        batch = modelpredict(sequences)
        dset[start_idx:start_idx+len(sequences)] = np.asarray(batch, np.float16)