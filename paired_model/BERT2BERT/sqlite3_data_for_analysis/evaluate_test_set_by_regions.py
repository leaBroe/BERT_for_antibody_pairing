# !pip install biopython
# used env: OAS_paired_env
from Bio.Align import substitution_matrices
import re
import pandas as pd

blosum62 = substitution_matrices.load("BLOSUM62")

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

    # raise error if min_length is 0
    if min_length == 0:
        #raise ValueError("Minimum length is 0")
        min_length = 1
    
    similarity_percentage = (matches / min_length) * 100

    return score, min_length, matches, similarity_percentage


def extract_region(sequence, start, end):
    return sequence[start-1:end]  # -1 because positions are 1-based

#regions_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extracted_data_from_PyIR/full_extraction_from_pyir_small_data.csv'
regions_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extracted_data_from_PyIR/full_extraction_pyir_full_data.csv'
regions_df = pd.read_csv(regions_file)

# Read the file containing generated and true sequences
# heavy2light run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1
#file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/HEAVY2LIGHT_114312.o"
file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o'
# output path: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/blosum_score_114416.o

with open(file_path, 'r') as file:
    file_content = file.read()

pattern = r"decoded light sequence:  ([A-Z ]+)\ntrue light sequence:  ([A-Z ]+)"
matches = re.findall(pattern, file_content)
df = pd.DataFrame(matches, columns=['Generated Sequence', 'True Sequence'])

column_list = ['fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4']

#column_list = ['fwr1']

scores = {region: [] for region in column_list}
similarities = {region: [] for region in column_list}

for index, row in df.iterrows():
    true_sequence = row['True Sequence'].replace(" ", "")
    generated_sequence = row['Generated Sequence'].replace(" ", "")
    
    sequence_id = regions_df.loc[index, 'sequence_id']
    print(f"\nSequence ID: {sequence_id}")
    true_seq_row = regions_df[regions_df['sequence_id'] == sequence_id]
    print(true_seq_row)
    
    for region in column_list:
        start = true_seq_row[f'{region}_start'].values[0]
        start = 1 if start == 1 else int(start//3)
        print(f"start: {start}")
        end = true_seq_row[f'{region}_end'].values[0]
        end = int(end//3)
        print(f"end: {end}")
        
        true_region_seq = extract_region(true_sequence, start, end)
        print(f"true_region_seq: {true_region_seq}")
        print(f"true_region_seq length: {len(true_region_seq)}")
        generated_region_seq = extract_region(generated_sequence, start, end)
        print(f"generated_region_seq: {generated_region_seq}")
        print(f"generated_region_seq length: {len(generated_region_seq)}")
        
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



