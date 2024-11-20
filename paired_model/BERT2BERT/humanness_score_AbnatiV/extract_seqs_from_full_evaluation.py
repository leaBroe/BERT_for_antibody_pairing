import re

# used env: abnativ

# input file has to be of the format:
# Sequence pair 58837:
# True Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A S Q G I S N I L A W Y Q Q K P G K A P K L L I Y G A S N L E S G V P S R F S G S G S G T D F T L T I S S L Q P E D F A T Y Y C Q S G Y Y S R G A T F G Q G T K V E I K
# Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A S Q G I S S Y L A W Y Q Q K P G K A P K L L I Y A A S T L Q S G V P S R F S G S G S G T D F T L T I S S L Q P E D F A T Y Y C Q Q Y N S Y P L T F G G G T K V E I K
# BLOSUM Score: 403.0
# Similarity Percentage: 79.43925233644859%
# Perplexity: 2.13580584526062
# model is on device cuda:0

# this script generates 2 fasta files: true_sequences.fasta and generated_sequences.fasta, 
# which can be further processed with PyIR to get the kappa and lambda subtypes required for abnativ

def parse_sequences(filename):
    sequences = []
    with open(filename, 'r') as f:
        content = f.read()
    # Split the content into sections for each sequence pair
    pairs = re.split(r'Sequence pair \d+:\n', content)[1:]  # Skip the first empty element
    for i, pair in enumerate(pairs):
        true_seq_match = re.search(r'True Sequence: (.+)', pair)
        gen_seq_match = re.search(r'Generated Sequence: (.+)', pair)
        if true_seq_match and gen_seq_match:
            true_seq = true_seq_match.group(1).replace(' ', '')
            gen_seq = gen_seq_match.group(1).replace(' ', '')
            sequences.append((true_seq, gen_seq))
    return sequences

# Function to convert amino acid sequence to DNA
# Z: Glutamic acid or Glutamine and X for any amino acid (new in new datasets)
def amino_acid_to_dna(aa_sequence):
    codon_table = {
        'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
        'K': 'AAA', 'L': 'TTA', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT', 'S': 'TCT',
        'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT', '*': 'TAA', 'Z':'NNN', 'X':'NNN', 'B':'NNN'
    }
    dna_sequence = ''.join(codon_table[aa] for aa in aa_sequence)
    return dna_sequence

def write_fasta(sequences, true_filename='true_sequences.fasta', gen_filename='generated_sequences.fasta'):
    with open(true_filename, 'w') as true_file, open(gen_filename, 'w') as gen_file:
        for idx, (true_seq, gen_seq) in enumerate(sequences, 1):
            true_file.write(f'>TrueSeq_{idx}\n{true_seq}\n')
            gen_file.write(f'>GenSeq_{idx}\n{gen_seq}\n')

input_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/logs/full_eval_PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1.txt'
sequences = parse_sequences(input_file)
dna_sequences = [(amino_acid_to_dna(true_seq), amino_acid_to_dna(gen_seq)) for true_seq, gen_seq in sequences]
write_fasta(dna_sequences)

