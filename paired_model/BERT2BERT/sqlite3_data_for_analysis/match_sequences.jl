#!/home/leab/.juliaup/bin/julia

# Extract IDs from query file
function get_ids(file)
    ## Populate set with lines from the file
    queries = Set(eachline(file))
    return queries
end

# Modified compare_ids function to accept a file handle for output -> IDs here: heavy[SEP]light sequences!
function compare_ids(queries, db, output_file)
    # Loop through db_lines
    for ln in eachline(db)
        # Extract ID with regex
        parts = split(ln, ',')  # Splitting based on comma
        ID = strip(parts[end])  # Assuming the ID is the last part after the comma!
        # Search in queries
        if ID in queries
            println(output_file, ln)  # Writing to file instead of standard output
        end
    end
end

# Open output file for writing
open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases.txt", "w") do output_file
    # Run the actual functions with output redirection
    open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/paired_full_seqs_sep_test_no_ids.txt") do query_file
        queries = get_ids(query_file)
        open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/paired_oas_db_full_extraction.csv") do db_file
            compare_ids(queries, db_file, output_file)
        end
    end
end

# # Open output file for writing
# open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data.csv", "w") do output_file
#     # Run the actual functions with output redirection
#     open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/paired_full_seqs_sep_test_no_ids.txt") do query_file
#         queries = get_ids(query_file)
#         open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_data_for_analysis.csv") do db_file
#             compare_ids(queries, db_file, output_file)
#         end
#     end
# end
