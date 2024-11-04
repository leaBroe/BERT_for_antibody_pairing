import prody
from attention_utils import attention_score_to_cls_token_and_to_all

# Load the PDB file
pdb_id = 'your_pdb_id'
chain = 'A'
pdb = prody.parsePDB(pdb_id, chain=chain)

# Get the sequence from the PDB file
pdb_seq = pdb.ca.getSequence()

# Get attention scores
seq_complete = 'your_protein_sequence'
model_bert = 'your_bert_model'  # Load your model
device = 'cuda'  # or 'cpu'
df_all_vs_all, att_to_cls, df_att_to_cls_exp = attention_score_to_cls_token_and_to_all(seq_complete, model_bert, device)
att_to_cls = att_to_cls.drop(labels = ['[CLS]', '[SEP]'])

# Map attention scores to PDB residues
att_for_masked_aa = att_to_cls.loc[pdb_seq] * 100000

# Check length consistency
if len(pdb.ca.getBetas()) != len(att_for_masked_aa):
    raise ValueError('Length of B factors and attention scores do not match')

# Set B factors
pdb.setBetas(0)
pdb.ca.setBetas(att_for_masked_aa)

# Save the modified PDB file
prody.writePDB(f'{pdb_id}_att_as_betas.pdb', pdb.select('ca'))