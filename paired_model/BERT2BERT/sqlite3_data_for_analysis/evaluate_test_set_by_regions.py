# !pip install biopython
from Bio.Align import substitution_matrices
import re
import pandas as pd

# Load the BLOSUM62 matrix
blosum62 = substitution_matrices.load("BLOSUM62")

# Function to calculate BLOSUM score
def calculate_blosum_score(true_seq, generated_seq, matrix):
    score = 0
    matches = 0

    min_length = min(len(true_seq), len(generated_seq))

    for i in range(min_length):
        pair = (true_seq[i], generated_seq[i])
        if pair in matrix:
            score += matrix[pair]
        elif (pair[1], pair[0]) in matrix:
            score += matrix[(pair[1], pair[0])]
        if true_seq[i] == generated_seq[i]:
            matches += 1

    if min_length == 0:
        similarity_percentage = 0
    else:
        similarity_percentage = (matches / min_length) * 100

    return score, min_length, matches, similarity_percentage

# Function to extract regions
def extract_region(sequence, start, end):
    return sequence[start-1:end]  # -1 because positions are 1-based

# Read the CSV file with true sequences and their regions
regions_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extracted_data_from_PyIR/full_extraction_from_pyir_small_data.csv' 
regions_df = pd.read_csv(regions_file)

# Read the file containing generated and true sequences
# heavy2light run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1
file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/HEAVY2LIGHT_114312.o"
# output path: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/blosum_score_114416.o

with open(file_path, 'r') as file:
    file_content = file.read()

# Extract sequences from the log file
pattern = r"decoded light sequence:  ([A-Z ]+)\ntrue light sequence:  ([A-Z ]+)"
matches = re.findall(pattern, file_content)
df = pd.DataFrame(matches, columns=['Generated Sequence', 'True Sequence'])

# Initialize lists to store scores and similarities
scores = {region: [] for region in ['fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4']}
similarities = {region: [] for region in ['fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4']}

# Iterate through each sequence pair
for index, row in df.iterrows():
    true_sequence = row['True Sequence'].replace(" ", "")
    generated_sequence = row['Generated Sequence'].replace(" ", "")
    
    sequence_id = regions_df.loc[index, 'sequence_id']
    true_seq_row = regions_df[regions_df['sequence_id'] == sequence_id]
    
    for region in ['fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4']:
        start = true_seq_row[f'{region}_start'].values[0]
        end = true_seq_row[f'{region}_end'].values[0]
        
        true_region_seq = extract_region(true_sequence, start, end)
        generated_region_seq = extract_region(generated_sequence, start, end)
        
        score, min_length, matches, similarity_percentage = calculate_blosum_score(true_region_seq, generated_region_seq, blosum62)
        scores[region].append(score)
        similarities[region].append(similarity_percentage)

        # Print results for each region
        print(f"\nRegion {region} for sequence pair {index+1}:")
        print(f"True {region} Sequence: {true_region_seq}")
        print(f"Generated {region} Sequence: {generated_region_seq}")
        print(f"BLOSUM Score: {score}")
        print(f"Minimum Length: {min_length}")
        print(f"Matches: {matches}")
        print(f"Similarity Percentage: {similarity_percentage}%")


# Calculate average scores and similarities for each region
average_scores = {region: sum(scores[region]) / len(scores[region]) for region in scores}
average_similarities = {region: sum(similarities[region]) / len(similarities[region]) for region in similarities}


print("\nAverage BLOSUM Scores and Similarity Percentages for each region:")
for region in scores:
    print(f"{region.upper()}:")
    print(f"Average BLOSUM Score for region {region}: {average_scores[region]}")
    print(f"Average Similarity Percentage for region {region}: {average_similarities[region]}%")


