## Initialize PyIR and set example file for processing
from crowelab_pyir import PyIR
FILE = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_sequence_alignment_light.fasta'

# pyirfiltered = PyIR(query=FILE, args=['--outfmt', 'dict', '--enable_filter'])
# result = pyirfiltered.run()

# #Prints size of Python returned dictionary
# print(len(result))

# pyirfile = PyIR(query=FILE)
# result = pyirfile.run()

pyirfile = PyIR(query=FILE, args=['--outfmt', 'tsv'])
result = pyirfile.run()

#Prints the output file
print(result)
