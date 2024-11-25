# Global Alignment

## 20/11/2024

So far used R with Biostrings and PairwiseAlignment default options:

```bash
"pairwiseAlignment"(pattern, subject, patternQuality=PhredQuality(22L), subjectQuality=PhredQuality(22L), type="global", substitutionMatrix=NULL, fuzzyMatrix=NULL, gapOpening=10, gapExtension=4, scoreOnly=FALSE)
```

[pairwiseAlignment function - RDocumentation](https://www.rdocumentation.org/packages/Biostrings/versions/2.40.2/topics/pairwiseAlignment)

Now, in the python script for global alignment (so far only used for dataset 4 (human healthy and covid) 

```bash
paired_model/BERT2BERT/src/full_test_set_evaluation_blosum_perplexity_similarity.py
```

used:

```bash
alignments = pairwise2.align.globalds(seq1, seq2, blosum_matrix, -10, -4)
```

where 

```
•	-10: Gap opening penalty
•	-4: Gap extension penalty

```

So the penalties should be the same (python requires the scores to be negative, while R wants them to be positive)