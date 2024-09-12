from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import torch
import numpy as np
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, RobertaForMaskedLM
import pandas as pd
from adapters import init
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# Assuming you have already computed `embeddings` and `labels`
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

model_name = "heavy_model"

# Load a pre-trained BERT model and tokenizer
#tokenizer = BertTokenizer.from_pretrained(small_heavy_encoder)
#model = BertModel.from_pretrained(small_heavy_encoder)

tokenizer = BertTokenizer.from_pretrained(small_heavy_encoder)
model = RobertaForMaskedLM.from_pretrained(small_heavy_encoder)

# FULL unpaired heavy test seqs
# col names: v_call_fewer,j_call_fewer,d_call_fewer,v_call_fewer_star,v_call,d_call,j_call,sequence_alignment_aa
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/fewer_genes_extracted_seqs_heavy_model_test_set_80000.txt"

# Small unpaired HEAVY test seqs
# col names: v_call_fewer,j_call_fewer,d_call_fewer,v_call_fewer_star,v_call,d_call,j_call,sequence_alignment_aa
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/fewer_genes_extracted_seqs_heavy_model_test_set_100.txt"

# FULL unpaired LIGHT test seqs 80'000 rows
# col names: v_call_fewer,j_call_fewer,v_call_fewer_star,v_call,d_call,j_call,sequence_alignment_aa
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/fewer_gene_names_extracted_seqs_light_model_test_set_no_dupl_80000.txt"

# col names: locus,v_call,d_call,j_call,sequence_alignment_aa,fwr1_aa,cdr1_aa,fwr2_aa,cdr2_aa,fwr3_aa,fwr4_aa,cdr3_aa,BType,Disease,Age,Species,Vaccine
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set_no_dupl_80000.txt"


filtered_df = pd.read_csv(test_file_path)


target = 'j_call_fewer'
plot_title_target = "J Gene Families"
plot_save_prefix = "j_call_fewer"

labels = filtered_df[f'{target}'].tolist()

heavy_sequences = filtered_df['sequence_alignment_aa'].tolist()
#light_sequences = filtered_df['sequence_alignment_aa'].tolist()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get embeddings
embeddings = get_mlm_last_layer_embeddings(model, tokenizer, heavy_sequences, device)
#embeddings = get_mlm_last_layer_embeddings(model, tokenizer, light_sequences, device)
print(embeddings)  # This will print the array of embeddings


# Ensure the labels are encoded as numerical values (if they aren't already)
# use LabelEncoder to encode target labels with value between 0 and n_classes-1. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Apply LDA (Number of components should be less than number of classes)
lda = LDA(n_components=2) # n_components = 2 for 2d visualization, j_call_fewer has 6 classes, therefore max n_components = 5
#lda = LDA(n_components=5) # n_components = 2 for 2d visualization, j_call_fewer has 6 classes, therefore max n_components = 5, if you choose n_components > 2 you cannot visualize the result in a plot


lda_result = lda.fit_transform(embeddings, encoded_labels)

# # Define colors and markers (same as PCA/t-SNE/UMAP plots)
# cmap = plt.get_cmap('tab20')
# colors = cmap(np.linspace(0, 1, 20))
# additional_cmap = plt.get_cmap('tab20b')
# additional_colors = additional_cmap(np.linspace(0, 1, 20))
# all_colors = np.vstack((colors, additional_colors))

# markers = ['o', 'v', 's', 'P', '*', 'X', 'D']  # List of markers

# # Plot LDA result
# fig, ax = plt.subplots(figsize=(10, 8))
# unique_labels = set(labels)
# for idx, label in enumerate(unique_labels):
#     indices = [i for i, l in enumerate(labels) if l == label]
#     color = all_colors[idx % len(all_colors)]
#     marker = markers[idx % len(markers)]
#     ax.scatter(lda_result[indices, 0], lda_result[indices, 1], label=label, alpha=0.7, color=color, marker=marker, s=10)

# # Set title and labels
# ax.set_title(f'Differentiation of {plot_title_target} via LDA of Last Layer Embeddings')
# ax.set_xlabel('LDA Component 1')
# ax.set_ylabel('LDA Component 2')

# # Adjust legend position
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.3, box.width, box.height * 0.7])
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

# # Display and save the plot
# plt.show()
# plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/{model_name}/LDA/full_{plot_save_prefix}_mlm_heavy2light_LDA.png')

# explained_variance_ratio = lda.explained_variance_ratio_

# plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
# plt.xlabel('LDA Component')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance by LDA Components')
# plt.show()
# plt.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/{model_name}/LDA/explained_variance_{plot_save_prefix}_mlm_heavy2light_LDA.png')

# Save the LDA components to a DataFrame
lda_result_df = pd.DataFrame(lda_result, columns=['Component 1', 'Component 2'])

# Add the original labels to the DataFrame
lda_result_df['j_call_fewer'] = labels

# Create a larger JointGrid to plot the LDA components with marginal distributions
g = sns.JointGrid(data=lda_result_df, x="Component 1", y="Component 2", height=10)  # Increase height to make plot larger

# Plot the scatterplot of LDA components, explicitly passing the DataFrame and hue for coloring
g = g.plot(sns.scatterplot, sns.histplot, data=lda_result_df, hue='j_call_fewer', palette='muted')

# Manually add a single legend, renaming it to 'J Gene Family'
handles, labels = g.ax_joint.get_legend_handles_labels()
g.ax_joint.legend(handles=handles, labels=labels, title='J Gene Family', loc='upper left')

# Adjust the layout to make room for the title and the legend
plt.subplots_adjust(top=0.9, right=0.85)

# Customize the plot title
#g.fig.suptitle(f'Differentiation of {plot_title_target} via LDA of Last Layer Embeddings with Marginals', y=1.02)

# Save the plot
g.savefig(f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/analysis_plots/{model_name}/LDA/full4_marginals_{plot_save_prefix}_mlm_heavy2light_LDA.png')

# Show the plot
plt.show()

