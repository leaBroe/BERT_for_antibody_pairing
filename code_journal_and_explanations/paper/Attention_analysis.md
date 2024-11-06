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