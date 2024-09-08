import torch
import numpy as np
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, RobertaForMaskedLM
import pandas as pd
from adapters import init
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# def get_mlm_last_layer_embeddings(model, tokenizer, sequences, device):
#     embeddings = []
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for seq in sequences:
#             # Tokenize the input sequence
#             inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            
#             # Execute the model
#             outputs = model(**inputs)
            
#             # Extract the last hidden states
#             last_hidden_states = outputs.last_hidden_state  # Accessing last hidden states directly
#             #last_hidden_states = outputs.hidden_states[-1]
            
#             # Mean pooling across the sequence length dimension to get a single vector representation
#             mean_pooled_output = last_hidden_states.mean(dim=1).cpu().numpy()
#             embeddings.append(mean_pooled_output)
    
#     return np.vstack(embeddings)  # Stack into a numpy array


def get_mlm_last_layer_embeddings(model, tokenizer, sequences, device):
    embeddings = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            
            # Ensure to request hidden states
            outputs = model(**inputs, output_hidden_states=True)
            
            # Access the last hidden states from the hidden_states tuple
            last_hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden states
            
            # Mean pooling across the sequence length dimension
            mean_pooled_output = last_hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled_output)
    
    return np.vstack(embeddings)  # Stack into a numpy array



small_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
small_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"

model_name = "light_model"

# Load a pre-trained BERT model and tokenizer
#tokenizer = BertTokenizer.from_pretrained(small_heavy_encoder)
#model = BertModel.from_pretrained(small_heavy_encoder)

tokenizer = BertTokenizer.from_pretrained(small_light_decoder)
model = RobertaForMaskedLM.from_pretrained(small_light_decoder)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FULL test path with disease and species
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/spaces_full_test_data_extraction_species_diseases_no_dupl.txt"

# FULL test path only with BType memory and naive input format: Species,Disease,BType,Isotype_light,sequence_alignment_aa_light,sequence_alignment_light,sequence_alignment_aa_heavy,sequence_alignment_heavy,sequence_alignment_heavy_sep_light
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/filtered_mem_naive_full_test_data_extraction_species_diseases_no_dupl.csv"

# FULL test path input format: "Species,Disease,BType,Isotype_light,locus_heavy,locus_light,v_call_heavy,d_call_heavy,j_call_heavy,sequence_alignment_aa_light,sequence_alignment_light,sequence_alignment_aa_heavy,sequence_alignment_heavy,sequence_alignment_heavy_sep_light"
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt"

# FULL test file path with v d and j genes 
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/fewer_genes_relevant_cols_spaces_heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt"

# FULL test path with lambda and kappa subtypes (only subtypes without specific gene) file format heavy[SEP]light
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/updated_test_file_subtypes.csv"

# small test path
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/filtered_mem_naive_full_test_data_extraction_species_diseases_no_dupl_small.csv"

# FULL unpaired light test seqs
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set_no_dupl.txt"

# FULL unpaired light test seqs 80'000 rows
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set_no_dupl_80000.txt"

#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/fewer_gene_names_extracted_seqs_light_model_test_set_no_dupl_80000.txt"

# load test file as csv
test_df_labels = pd.read_csv(test_file_path)

# def load_data(file_path):
#     data = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             data.append(line.strip())
#     sequences = []
#     for entry in data:
#         split_entry = entry.split(' [SEP] ')
#         if len(split_entry) == 2:
#             sequences.append(split_entry)
#         else:
#             print(f"Skipping invalid entry: {entry}")
#     df = pd.DataFrame(sequences, columns=['heavy', 'light'])
#     return df

filtered_df = pd.read_csv(test_file_path)

# if input file is of the format heavy[SEP]light use this function for the extrraction of heavy and light sequences
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())
    sequences = []
    for entry in data:
        split_entry = entry.split(' [SEP] ')
        if len(split_entry) == 2:
            sequences.append(split_entry)
        else:
            print(f"Skipping invalid entry: {entry}")
    df = pd.DataFrame(sequences, columns=['heavy', 'light'])
    return df

#target = "locus"
#target = "v_call"
#target = "j_call_fewer"
#target="BType"
target="Age"
#target = 'v_call_fewer_heavy_star'
#target = 'v_call_heavy'
#target = 'd_call_fewer_heavy'
#target = 'j_call_fewer_heavy'
#target = 'v_call_fewer_heavy'
#target = 'locus'
#target = 'subtype'



# test_df = load_data(test_file_path)
# heavy_sequences = test_df["heavy"].tolist()
# light_sequences = test_df["light"].tolist()
#labels = test_df_labels['v_family'].tolist()
#labels = test_df_labels['subtype'].tolist()
labels = test_df_labels[f'{target}'].tolist()

#plot_title_target = "B-Cell Types"
plot_title_target = "Age"
#plot_title_target = "J Gene Families"
#plot_title_target = "Kappa / Lambda Loci"
#plot_title_target = "J Gene Families"
#plot_title_target = "D Gene Families"
#plot_save_prefix = "heavy_v_gene_families"
#plot_save_prefix = "v_call_fewer_heavy_star_better_layout2"
#plot_save_prefix = "locus_unpaired_light_seqs_80000"
#plot_save_prefix = "jgenes_fewer_unpaired_light_seqs_80000"
#plot_save_prefix = "BTypes_unpaired_light_seqs_80000"
plot_save_prefix = "Age_unpaired_light_seqs_80000"
#plot_save_prefix = "fewer_heavy_j_gene_families"
#plot_save_prefix = "heavy_d_gene_families"

# if input file is of the format: Species,Disease,BType,Isotype_light,sequence_alignment_aa_light,sequence_alignment_light,sequence_alignment_aa_heavy,sequence_alignment_heavy,sequence_alignment_heavy_sep_light use this for the extraction of heavy and light sequences
# Extract sequences into lists
#light_sequences = filtered_df['sequence_alignment_aa_light'].tolist()
light_sequences = filtered_df['sequence_alignment_aa'].tolist()

#heavy_sequences = filtered_df['sequence_alignment_aa_heavy'].tolist()


# #test_df = load_data(test_file_path)
# light_sequences = test_df["light"].tolist()
# heavy_sequences = test_df["heavy"].tolist()
# labels = test_df_labels['Disease'].tolist()
# labels = test_df_labels['BType'].tolist()


# Get embeddings
embeddings = get_mlm_last_layer_embeddings(model, tokenizer, light_sequences, device)
#embeddings = get_mlm_last_layer_embeddings(model, tokenizer, light_sequences, device)
print(embeddings)  # This will print the array of embeddings


# Apply UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_reducer.fit_transform(embeddings)

# Ensure there are enough distinct colors
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, 20))

# If you need more than 20 colors, combine multiple colormaps or use ListedColormap
additional_cmap = plt.get_cmap('tab20b')
additional_colors = additional_cmap(np.linspace(0, 1, 20))

# Combine colors
all_colors = np.vstack((colors, additional_colors))

# Plot UMAP result with labels
fig, ax = plt.subplots(figsize=(10, 8))  # Create figure and axis
for idx, label in enumerate(set(labels)):
    indices = [i for i, l in enumerate(labels) if l == label]
    color = all_colors[idx % len(all_colors)]  # Cycle through colors if more subtypes than colors
    # plot u-map as scatter plot
    ax.scatter(umap_result[indices, 0], umap_result[indices, 1], label=label, alpha=0.5, color=color, s=10)
    # plot u-map as hexbin plot
    #ax.hexbin(umap_result[indices, 0], umap_result[indices, 1], label=label, color=color)
    
ax.set_title(f'Differentiation of {plot_title_target} via UMAP of Last Layer Embeddings')
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.7])

# Place the legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
# Save the plot before showing
plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/{model_name}/UMAP/{plot_save_prefix}_mlm_heavy2light_FULL_data_robertaformaskedlm.png')
# Display the plot
plt.show()

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# Define colors and markers
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, 20))
additional_cmap = plt.get_cmap('tab20b')
additional_colors = additional_cmap(np.linspace(0, 1, 20))
all_colors = np.vstack((colors, additional_colors))

markers = ['o', 'v', 's', 'P', '*', 'X', 'D']  # List of markers

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot PCA result with labels
unique_labels = set(labels)  
for idx, label in enumerate(unique_labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    color = all_colors[idx % len(all_colors)]
    marker = markers[idx % len(markers)]
    ax.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label, alpha=0.7, color=color, marker=marker, s=10)
    #ax.hexbin(pca_result[indices, 0], pca_result[indices, 1], label=label, alpha=0.3, color=color, gridsize=50)


# Set title and labels
ax.set_title(f'Differentiation of {plot_title_target} via PCA of Last Layer Embeddings')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')

# Shrink current axis's height by 20% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.7])
# Place the legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
plt.show()
plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/{model_name}/PCA/{plot_save_prefix}_mlm_heavy2light_FULL_data.png')



# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

# Define colors and markers
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, 20))
additional_cmap = plt.get_cmap('tab20b')
additional_colors = additional_cmap(np.linspace(0, 1, 20))
all_colors = np.vstack((colors, additional_colors))

markers = ['o', 'v', 's', 'P', '*', 'X', 'D']  # List of markers

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot t-SNE result with labels
unique_labels = set(labels)  # Sort to ensure consistent color/marker assignment
for idx, label in enumerate(unique_labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    color = all_colors[idx % len(all_colors)]
    marker = markers[idx % len(markers)]
    ax.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=label, alpha=0.7, color=color, marker=marker, s=10)

# Set title and labels
ax.set_title(f'Differentiation of {plot_title_target} via t-SNE of Last Layer Embeddings')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.3,
                 box.width, box.height * 0.7])

# Place the legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
plt.show()
plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/{model_name}/t-SNE/{plot_save_prefix}_mlm_heavy2light_FULL_data_better_layout.png')

