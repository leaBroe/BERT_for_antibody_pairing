using Statistics

# Define the file path
#file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_decoding_strategies/nucleus_sampling/logs/full_eval_nucleus_0.9_temp_0.1_full_PLAbDab_healthy_human_184933.o"
#file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/logs/heavy2light_eval_correct_2_184930.o"
#file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_decoding_strategies/nucleus_sampling/logs/full_eval_nucleus_0.9_temp_0.1_full_PLAbDab_healthy_human_184933.o"
#file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/logs/full_eval_DoLa_layers_high_rep_penal_1.2_184932.o"
file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_decoding_strategies/DoLa/logs/full_eval_full_PLAbDab_healthy_human_[1,340]_DoLa_max_length_120_rep_penalty_1.2_num_epochs_30_184978.o"

# Initialize accumulators
blosum_scores = Float64[]
similarity_percentages = Float64[]
perplexities = Float64[]

# Open and read the file line by line
open(file_path, "r") do file
    for line in eachline(file)
        if occursin("BLOSUM Score:", line)
            # Extract and parse BLOSUM Score
            push!(blosum_scores, parse(Float64, strip(split(line, ":")[2])))
        elseif occursin("Similarity Percentage:", line)
            # Extract, remove '%' and parse Similarity Percentage
            raw_value = strip(split(line, ":")[2])
            clean_value = replace(raw_value, "%" => "")
            push!(similarity_percentages, parse(Float64, clean_value))
        elseif occursin("Perplexity:", line)
            # Extract and parse Perplexity
            push!(perplexities, parse(Float64, strip(split(line, ":")[2])))
        end
    end
end

# Calculate averages
average_blosum = mean(blosum_scores)
average_similarity = mean(similarity_percentages)
average_perplexity = mean(perplexities)

# Display results
println("Average BLOSUM Score: $average_blosum")
println("Average Similarity Percentage: $average_similarity%")
println("Average Perplexity: $average_perplexity")


