# Attention Analysis

The pipeline should be:
1. have a pdb with the generated sequences
2. extract attention score (i did to the cls token) for each aa
3. edit the pdb putting the scores in the bscore field of the pdb (the reason is that with pymol you can color the protein based on the b field)
4. load the pdb in pymol and visualize the scores

# 4/11/2024

[Pymol Color By Data](https://betainverse.github.io/blog/2014/10/13/pymol-color-by-data/)

```bash
spectrum b
```

[Color - PyMOLWiki](https://pymolwiki.org/index.php/Color#B-Factors)

[Pymol Color By Data](https://betainverse.github.io/blog/2014/10/13/pymol-color-by-data/)

[Color - PyMOLWiki](https://pymolwiki.org/index.php/Color#Reassigning_B-Factors_and_Coloring)

# 06/11/2024

Extract Attention:

https://github.com/ibmm-unibe-ch/TemBERTure/blob/main/dev/analysis/attention_utils.py

Visualize on Pymol:

https://github.com/ibmm-unibe-ch/TemBERTure/blob/main/dev/analysis/pymol_commands.pymol

create he pdb to visualize the attention scores as betas before loading them to pymol

https://github.com/ibmm-unibe-ch/TemBERTure/blob/main/dev/analysis/sasa_utils.py

# 12/11/2024

## BertViz

[GitHub - jessevig/bertviz: BertViz: Visualize Attention in NLP Models (BERT, GPT2, BART, etc.)](https://github.com/jessevig/bertviz?tab=readme-ov-file)

[Google Colab](https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ?usp=sharing)

[Deconstructing BERT, Part 2: Visualizing the Inner Workings of Attention](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1)

