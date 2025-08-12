
import sys

refgenomeid = "hg38/genomes/GRCh38-hg38"
section_size = 5
model_path = "https://tfhub.dev/deepmind/enformer/1"

genemodel = "refSeq_v20240129"
# genemodel = "MANE/1.3"
# genemodel = "GENCODE/46/comprehensive/ALL"
# genemodel = "GENCODE/46/basic/PRI"

genemodel = genemodel.replace("/", "_")
input_file_name = "../output/" + genemodel + "_singleTSS.txt"
output_file_name = "../output/" + genemodel + "_singleTSS.h5"


# ### this is for custom lists (variants13 and 14 from before)
# input_file_name = "../output/variants13_singleTSS.txt"
# output_file_name = "../output/variants13_singleTSS.h5"

print(input_file_name)
print(output_file_name)

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import time
import os
from dae.genomic_resources.reference_genome import build_reference_genome_from_resource_id


refgenome = build_reference_genome_from_resource_id(refgenomeid).open()
INPUT_SEQ_LEN = 393_216
model = hub.load(model_path).model
num_bins = 896
num_tracks = 5313


#########################################################################################################
### for each row in input file, returns two strings for ref and alt sequences
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
    return seq[:-1]


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
    # Extract 'human' track
    data = prediction['human']
    return data.numpy()



#########################################################################################################
import csv

# Read the file
with open(input_file_name, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')  # Use tab as the separator
    data = [row for row in reader]  # Convert to a list of rows

data = data[1:]
num_variants = len(data)

## for plus strand
# data = [row for row in data if row[2] == '+']


result_buffer = np.zeros((num_variants, num_bins, num_tracks), dtype=np.float16)

#########################################################################################################
sequences = []
positions = []
descriptions = []
types = []

# Initialize section counter
section_counter = 0
for row in data:
    chrom, position = row[-2:]

    seq = get_sequence(refgenome, chrom, position)

    sequences.append(seq)
    positions.append(f"{chrom}:{position}")

    if len(sequences) >= section_size:
        print(f"Reading section {section_counter + 1} with {section_size} sequences.")

        start_time = time.time()
        section_results = modelpredict(sequences)

        execution_time = time.time() - start_time
        print(f"Section: {section_counter+1}, Execution Time: {execution_time:.2f} seconds")
        sys.stdout.flush()

        result_buffer[section_counter * section_size: (section_counter + 1) * section_size] = section_results

        sequences.clear()
        section_counter += 1


# After the main loop, handle the remaining sequences (if any)
if len(sequences) > 0:
    print(f"Reading final section with {len(sequences)} sequences.")
    section_results = modelpredict(sequences)
    
    result_buffer[section_counter * section_size: section_counter * section_size + len(sequences)] = section_results

    # Reset for the next section (optional, not strictly necessary here)
    sequences.clear()

##########################################################################################################


import h5py

with h5py.File(output_file_name, "w") as f:
    # f.create_dataset("D", data=result_buffer, compression = 'gzip')
    f.create_dataset("D", data=result_buffer)


##########################################################################################################


