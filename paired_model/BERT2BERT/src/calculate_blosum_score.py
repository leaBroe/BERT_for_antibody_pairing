# !pip install biopython
# used env: adap_2

from Bio.Align import substitution_matrices

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


# Define the true and generated sequences
true_light_sequence = "D I Q M T Q S P S S L S A S V G D I V T I T C R A N Q T I D N Y L N W Y Q Q K R G E A P K L L I Y T A S S L Q T G V P S R F S G S G S G T D F T L T I S S L Q P E D F A T Y Y C Q Q S Y S S P R S F G Q G T K L E M K"
generated_light_sequence = "D I Q M T Q S P S S L S A S V G D R V T I T C R A F Q G I S N Y L A W Y Q Q K P G K A P K L L I Y D A T T L E S G V P T R F S E R G F G T E F T V S I N N L Q P E D F A V Y Y C Q H Y N S Y P L T F V V W T K V E I K"

# Load the BLOSUM62 matrix
blosum62 = substitution_matrices.load("BLOSUM62")

# Calculate the BLOSUM score
score, min_length, matches, similarity_percentage = calculate_blosum_score(true_light_sequence, generated_light_sequence, blosum62)

# Print the results
print(f"BLOSUM Score: {score}")
print(f"Minimum Length: {min_length}")
print(f"Matches: {matches}")
print(f"Similarity Percentage: {similarity_percentage}%")

