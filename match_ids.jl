# Extract IDs from query file 
# function get_ids(file)
#     ## Initiate empty vector
#     queries = Set(eachline(file))
#     ## Loop through file and populate vector with text
#     for ln in eachline(file)
#         push!(queries, ln)
#     end
#     return queries
# end

# Extract IDs from query file
function get_ids(file)
    ## Populate set with lines from the file
    queries = Set(eachline(file))
    return queries
end

# open("centroid_IDs_strings_2.txt") do query_file
#     queries = get_ids(query_file)
#     print(queries)

# Modified compare_ids function to accept a file handle for output
function compare_ids(queries, db, output_file)
    # Loop through db_lines
    for ln in eachline(db)
        # Extract ID with regex
        parts = split(ln, ',')  # Splitting based on comma
        ID = strip(parts[1])    # Assuming the ID is the first part before the comma
        # Search in queries
        if ID in queries
            println(output_file, ln)  # Writing to file instead of standard output
        end
    end
end

# Open output file for writing
open("output_julia_6.txt", "w") do output_file
    # Run the actual functions with output redirection
    open("/Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt") do query_file
        queries = get_ids(query_file)
        open("/Users/leabroennimann/Downloads/master_thesis_data/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt") do db_file
            compare_ids(queries, db_file, output_file)
        end
    end
end
