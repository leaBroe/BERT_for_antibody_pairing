# Tokenizer with new data

Heavy2Light Model 

“Z” “X” and “B” in new dataset (true sequences) → Z is not in tokenizer

Amino acid table:

[Codes Used in Sequence Description](https://www.ddbj.nig.ac.jp/ddbj/code-e.html)

[Nomenclature for the description of sequence variants: codons and amino acids](https://www.hgvs.org/mutnomen/codon.html)


```bash
# Function to convert amino acid sequence to DNA
# Z: Glutamic acid or Glutamine and X for any amino acid (new in new datasets)
def amino_acid_to_dna(aa_sequence):
    codon_table = {
        'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
        'K': 'AAA', 'L': 'TTA', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT', 'S': 'TCT',
        'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT', '*': 'TAA', 'Z':'NNN', 'X':'NNN'
    }
    dna_sequence = ''.join(codon_table[aa] for aa in aa_sequence)
    return dna_sequence
```

```bash
{
  "version": "1.0",
  "truncation": {
    "direction": "Right",
    "max_length": 200,
    "strategy": "LongestFirst",
    "stride": 0
  },
  "padding": {
    "strategy": {
      "Fixed": 200
    },
    "direction": "Right",
    "pad_to_multiple_of": null,
    "pad_id": 0,
    "pad_type_id": 0,
    "pad_token": "[PAD]"
  },
  "added_tokens": [
    {
      "id": 0,
      "content": "[PAD]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 1,
      "content": "[UNK]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 2,
      "content": "[CLS]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 3,
      "content": "[SEP]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 4,
      "content": "[MASK]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 5,
      "content": "A",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 6,
      "content": "C",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 7,
      "content": "D",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 8,
      "content": "E",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 9,
      "content": "F",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 10,
      "content": "G",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 11,
      "content": "H",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 12,
      "content": "I",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 13,
      "content": "K",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 14,
      "content": "L",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 15,
      "content": "M",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 16,
      "content": "N",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 17,
      "content": "P",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 18,
      "content": "Q",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 19,
      "content": "R",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 20,
      "content": "S",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 21,
      "content": "T",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 22,
      "content": "V",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 23,
      "content": "W",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 24,
      "content": "Y",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    }
  ],
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": null,
    "lowercase": false
  },
  "pre_tokenizer": {
    "type": "BertPreTokenizer"
  },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {
        "SpecialToken": {
          "id": "[CLS]",
          "type_id": 0
        }
      },
      {
        "Sequence": {
          "id": "A",
          "type_id": 0
        }
      },
      {
        "SpecialToken": {
          "id": "[SEP]",
          "type_id": 0
        }
      }
    ],
    "pair": [
      {
        "SpecialToken": {
          "id": "[CLS]",
          "type_id": 0
        }
      },
      {
        "Sequence": {
          "id": "A",
          "type_id": 0
        }
      },
      {
        "SpecialToken": {
          "id": "[SEP]",
          "type_id": 0
        }
      },
      {
        "Sequence": {
          "id": "B",
          "type_id": 1
        }
      },
      {
        "SpecialToken": {
          "id": "[SEP]",
          "type_id": 1
        }
      }
    ],
    "special_tokens": {
      "[CLS]": {
        "id": "[CLS]",
        "ids": [
          2
        ],
        "tokens": [
          "[CLS]"
        ]
      },
      "[SEP]": {
        "id": "[SEP]",
        "ids": [
          3
        ],
        "tokens": [
          "[SEP]"
        ]
      }
    }
  },
  "decoder": {
    "type": "WordPiece",
    "prefix": "##",
    "cleanup": true
  },
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0,
      "[UNK]": 1,
      "[CLS]": 2,
      "[SEP]": 3,
      "[MASK]": 4
    }
  }
}
```