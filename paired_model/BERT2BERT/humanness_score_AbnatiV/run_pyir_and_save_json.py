import json
from crowelab_pyir import PyIR

# List of input FASTA files
files = ['true_sequences_dna.fasta', 'generated_sequences_dna.fasta']

# Process each file with PyIR
for FILE in files:
    pyirfile = PyIR(query=FILE, args=['--outfmt', 'tsv'])
    output = pyirfile.run()

