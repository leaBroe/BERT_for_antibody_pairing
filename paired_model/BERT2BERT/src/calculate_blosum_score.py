# !pip install biopython
from Bio.Align import substitution_matrices
import re
import pandas as pd

# Read the file content
with open('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/HEAVY2LIGHT_114312.o', 'r') as file:
    file_content = file.read()

# Use regex to extract the sequences
pattern = r"decoded light sequence:  ([A-Z ]+)\ntrue light sequence:  ([A-Z ]+)"
matches = re.findall(pattern, file_content)

# Convert the matches into a dataframe
df = pd.DataFrame(matches, columns=['Generated Sequence', 'True Sequence'])

# Print the dataframe
#print(df)

def calculate_blosum_score(true_seq, generated_seq, matrix):
    score = 0
    matches = 0

    generated_light_seq = generated_seq.replace(" ", "")
    true_light_seq = true_seq.replace(" ", "")

    min_length = min(len(true_light_seq), len(generated_light_seq))

    for i in range(min_length):
        pair = (true_light_seq[i], generated_light_seq[i])
        if pair in matrix:
            score += matrix[pair]
        elif (pair[1], pair[0]) in matrix:
            score += matrix[(pair[1], pair[0])]
        if true_light_seq[i] == generated_light_seq[i]:
            matches += 1
    
    similarity_percentage = (matches / min_length) * 100
    
    return score, min_length, matches, similarity_percentage

# Load the BLOSUM62 matrix
blosum62 = substitution_matrices.load("BLOSUM62")

# Calculate the BLOSUM score for all sequences
scores = []
similarities = []

for index, row in df.iterrows():
    score, min_length, matches, similarity_percentage = calculate_blosum_score(row['True Sequence'], row['Generated Sequence'], blosum62)
    scores.append(score)
    similarities.append(similarity_percentage)

    # Print results for each sequence pair
    print(f"\nSequence pair {index+1}:")
    print(f"True Sequence: {row['True Sequence']}")
    print(f"Generated Sequence: {row['Generated Sequence']}")
    print(f"BLOSUM Score: {score}")
    print(f"Minimum Length: {min_length}")
    print(f"Matches: {matches}")
    print(f"Similarity Percentage: {similarity_percentage}%")

# Add the scores and similarities to the dataframe
df['BLOSUM Score'] = scores
df['Similarity Percentage'] = similarities

# Calculate the average BLOSUM score and average similarity percentage
average_blosum_score = sum(scores) / len(scores)
average_similarity_percentage = sum(similarities) / len(similarities)

# Print the averages
print(f"\nAverage BLOSUM Score: {average_blosum_score}")
print(f"Average Similarity Percentage: {average_similarity_percentage}%")

# Save the dataframe to a CSV file (if needed)
#df.to_csv('sequences_with_scores.csv', index=False)

