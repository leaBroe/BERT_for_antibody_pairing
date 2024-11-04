import pandas as pd

# Mapping of one-letter to three-letter amino acid codes
aa_mapping = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU',
    'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
    'Y': 'TYR', 'V': 'VAL'
}

def load_attention_scores(csv_file):
    """Load attention scores from CSV and return a list of tuples with (one-letter code, score)."""
    df = pd.read_csv(csv_file)
    return list(zip(df['token'], df['attention']))


def update_pdb_bfactor(pdb_file, output_file, attention_scores):
    """Update the B-factor column in a PDB file with attention scores, applying the same score to all lines of a given residue."""
    score_index = 0  # Track current attention score index
    current_attention_score = None  # Track the attention score for each residue
    with open(pdb_file, 'r') as pdb, open(output_file, 'w') as out_pdb:
        for line in pdb:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Extract the amino acid three-letter code from the PDB line and residue sequence number
                pdb_residue = line[17:20].strip()
                pdb_residue_num = int(line[22:26].strip())
                
                # Check if there's an attention score left to apply and if we need to update to a new residue score
                if score_index < len(attention_scores):
                    # Get the current attention score and its corresponding one-letter code
                    aa_one_letter, attention_score = attention_scores[score_index]
                    
                    # Convert the one-letter code to a three-letter code
                    csv_residue = aa_mapping[aa_one_letter]
                    
                    # Update to a new attention score if we're at a new residue
                    if pdb_residue == csv_residue:
                        if score_index == 0 or pdb_residue_num > score_index:  # Move to next residue
                            current_attention_score = attention_score
                            score_index += 1
                    
                # Format the current attention score as the new B-factor (7.4f format)
                new_b_str = f"{current_attention_score:7.4f}"
                line = line[:60] + new_b_str + line[66:]
            
            # Write the (modified or unmodified) line to the output PDB
            out_pdb.write(line)

# Paths to input files
csv_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/attention_analysis/data/attention_scores_to_cls_exp_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1.csv"
pdb_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/attention_analysis/pdb_files/attention_analysis_gen_seq_2d1fe_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
output_pdb = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/attention_analysis/output_modified.pdb"

# Load the attention scores from CSV
attention_scores = load_attention_scores(csv_file)

# Update the PDB B-factors based on attention scores
update_pdb_bfactor(pdb_file, output_pdb, attention_scores)
