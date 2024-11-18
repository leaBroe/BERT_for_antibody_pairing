import json
from crowelab_pyir import PyIR

# List of input FASTA files
files = ['true_sequences.fasta', 'generated_sequences.fasta']

# Initialize lists to store sequences grouped by locus
igk_sequences = []
igl_sequences = []

# Process each file with PyIR
for FILE in files:
    pyirfile = PyIR(query=FILE, args=['--outfmt', 'json'])
    output = pyirfile.run()
    
    # Parse the JSON output
    try:
        results = json.loads(output)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output from PyIR for file {FILE}: {e}")
        continue  # Skip to the next file if parsing fails

    # Check if results is a dictionary with 'results' key
    if isinstance(results, dict) and 'results' in results:
        records = results['results']
    else:
        # If results is not in expected format, print an error and continue
        print(f"Unexpected format in PyIR output for file {FILE}")
        continue

    # Iterate over each record in the results
    for record in records:
        locus = record.get('locus')
        sequence_id = record.get('sequence_id')
        # Use 'sequence_alignment' or 'sequence_vdj' depending on your needs
        sequence = record.get('sequence_vdj')

        # Check if the necessary fields are present
        if locus and sequence_id and sequence:
            if locus == 'IGK':
                igk_sequences.append((sequence_id, sequence))
            elif locus == 'IGL':
                igl_sequences.append((sequence_id, sequence))
            else:
                # If the locus is neither IGK nor IGL, you can choose to handle it or skip
                pass
        else:
            # Handle cases where fields might be missing
            print(f"Missing data for sequence {sequence_id} in file {FILE}")

# Write IGK sequences to a FASTA file
with open('igk_sequences.fasta', 'w') as igk_file:
    for seq_id, seq in igk_sequences:
        igk_file.write(f'>{seq_id}\n{seq}\n')

# Write IGL sequences to a FASTA file
with open('igl_sequences.fasta', 'w') as igl_file:
    for seq_id, seq in igl_sequences:
        igl_file.write(f'>{seq_id}\n{seq}\n')

print("Sequences have been grouped by locus and written to 'igk_sequences.fasta' and 'igl_sequences.fasta'.")
