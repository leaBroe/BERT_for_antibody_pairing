from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig
import torch
import pandas as pd
from adapters import init
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Initialize the model, tokenizer, and generation configuration
def initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    init(model)
    print(f"model is on device: {model.device}")
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    return model, tokenizer, generation_config

# # Define paths
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
# tokenizer_path = f"{model_path}/checkpoint-336040"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
# adapter_name = "heavy2light_adapter"

# heavy2light 60 epochs diverse beam search beam = 5
# best model so far
run_name="full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
tokenizer_path=f"{model_path}/checkpoint-504060"
adapter_path=f"{model_path}/final_adapter"
generation_config_path=model_path
adapter_name="heavy2light_adapter"

# Initialize model and tokenizer
model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)


# FULL test path with disease and species
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/spaces_full_test_data_extraction_species_diseases_no_dupl.txt"

# FULL test path only with BType memory and naive
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/filtered_mem_naive_full_test_data_extraction_species_diseases_no_dupl.csv"

# small test path
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/filtered_mem_naive_full_test_data_extraction_species_diseases_no_dupl_small.csv"

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

# Extract sequences into lists
light_sequences = filtered_df['sequence_alignment_aa_light'].tolist()
heavy_sequences = filtered_df['sequence_alignment_aa_heavy'].tolist()


#test_df = load_data(test_file_path)
#light_sequences = test_df["light"].tolist()
#heavy_sequences = test_df["heavy"].tolist()
#labels = test_df_labels['Disease'].tolist()
labels = test_df_labels['BType'].tolist()



# # Function to extract embeddings from the last layer
# def get_last_layer_embeddings(model, tokenizer, sequences, device):
#     embeddings = []
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for seq in sequences:
#             inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
#             outputs = model.encoder(**inputs)
#             last_hidden_states = outputs.last_hidden_state
#             embeddings.append(last_hidden_states.mean(dim=1).cpu().numpy())  # Mean pooling
#     return np.vstack(embeddings)  # Stack into a numpy array


def get_decoder_last_layer_embeddings(model, tokenizer, sequences, device):
    embeddings = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            # Tokenize the input sequence for the encoder
            inputs = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            
            outputs = model(**inputs, decoder_input_ids=inputs['input_ids'])

            # For models returning CausalLMOutputWithCrossAttentions, the hidden states are usually found under 'last_hidden_states' attribute.
            if hasattr(outputs, 'hidden_states'):
                last_hidden_states = outputs.hidden_states[-1]  # Access the last layer's hidden states
            else:
                last_hidden_states = outputs[0]  # outputs[0]: the last_hidden_state

            # Mean pooling across the sequence length dimension
            mean_pooled_output = last_hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled_output)
    
    return np.vstack(embeddings)  # Stack into a numpy array


# Get embeddings from the last layer
#embeddings = get_last_layer_embeddings(model, tokenizer, light_sequences, device)
embeddings = get_decoder_last_layer_embeddings(model, tokenizer, heavy_sequences, device)

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
    
ax.set_title('UMAP Visualization of B-Type Types in Last Layer Embeddings')
ax.set_xlabel('UMAP Component 1')
ax.set_ylabel('UMAP Component 2')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

# Place the legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
# Save the plot before showing
plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/heavy2light/{run_name}/UMAP/btype_heavy_input_tsne_heavy2light_FULL_data.png')
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
unique_labels = sorted(set(labels))  # Sort to ensure consistent color/marker assignment
for idx, label in enumerate(unique_labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    color = all_colors[idx % len(all_colors)]
    marker = markers[idx % len(markers)]
    ax.scatter(pca_result[indices, 0], pca_result[indices, 1], label=label, alpha=0.7, color=color, marker=marker, s=10)
    #ax.hexbin(pca_result[indices, 0], pca_result[indices, 1], label=label, alpha=0.3, color=color, gridsize=50)


# Set title and labels
ax.set_title('Differentiation of B-Cell Types via PCA of Last Layer Embeddings')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')

# Shrink current axis's height by 20% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])
# Place the legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.show()
plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/heavy2light/{run_name}/PCA/Btypes_heavy_input_tsne_heavy2light_FULL_data.png')

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
ax.set_title('Differentiation of B-Cell Types via t-SNE of Last Layer Embeddings')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

# Place the legend below the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.show()
plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/heavy2light/{run_name}/t-SNE/diseases_heavy_input_tsne_heavy2light_FULL_data.png')

