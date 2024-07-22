## Initialize PyIR and set example file for processing
from crowelab_pyir import PyIR
FILE = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/small_test_data/small_paired_data_sequence_alignment_light.fasta'

pyirfiltered = PyIR(query=FILE, args=['--outfmt', 'dict', '--enable_filter'])
result = pyirfiltered.run()

#Prints size of Python returned dictionary
print(len(result))


