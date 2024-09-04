# 10 sequences with intermediate similarity

# Alphafold Structure prediction

model with run name:

```bash
full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1
```

```bash
Sequence pair 64190:
True Sequence: Q A G L T Q P P S V S K G L R Q T A T L T C T G N S N N V G N Q G A A W L Q Q H Q G H P P K L L S Y R N N N R P S G I S E R L S A S R S G N T A S L T I T G L Q P E D E A D Y Y C S A W D S S L S A W V F G G G T K L T V L
Generated Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S G S S S N I G S N T V N W Y Q Q L P G T A P K L L I Y S N N Q R P S G V P D R F S G S K S G T S A S L A I S G L Q S E D E A D Y Y C A A W D D S L N G W V F G G G T K L T V L
BLOSUM Score: 360.0
Similarity Percentage: 60.909090909090914%
Perplexity: 1.6024678945541382
model is on device cuda:0
```

```bash
Sequence pair 64606:
True Sequence: E I V L T Q S P G T L S F S P G E T A T L S C R A S Q N V N R Y L A W Y Q H K P G Q A P R L L I Y G A S D R A I D I P D R F T A S G S G T D F T L T I S R L E P E D S A V Y H C Q H Y A N S P P Y S F G Q G T K L E I K
Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A S Q S I S T Y L N W Y Q Q K P G K A P K L L I Y A A S S L Q S G V P S R F S G S G S G T D F T L T I S S L Q P E D F A T Y Y C Q Q S Y S T P P I T F G Q G T R L E I K
BLOSUM Score: 356.0
Similarity Percentage: 60.18518518518518%
Perplexity: 2.027750015258789
model is on device cuda:0
```

```bash
Sequence pair 64860:
True Sequence: E I V L T Q S P G T L S V S P G E S A I L S C R A G Q R L T S S Y L A W Y Q Q K P G Q A P R L L I Y A A S R R A T G I P D R F S G S G S E T D F T L T I S G L E P E D V A V Y Y C Q H Y G G S P L F S F G P G T K V D I K
Generated Sequence: E I V L T Q S P G T L S L S P G E R A T L S C R A S Q S V S S N Y L A W Y Q Q K P G Q A P R L L I Y G A S S R A T G I P D R F S G S G S G T D F T L T I S R L E P E D F A V Y Y C Q Q Y G S S P P Y T F G Q G T K L E I K
BLOSUM Score: 460.0
Similarity Percentage: 80.73394495412845%
Perplexity: 1.9261778593063354
model is on device cuda:0
```

```bash
Sequence pair 64901:
True Sequence: S Y E L T Q P S S V S V S P G Q T A R I T C S G D V L A K K Y A R W F Q Q K P G Q A P V L V I Y K D S E R P S G I P E R F S G S S S G T T V T L T I S G A Q V E D E A D Y Y C Y S A A D N V I F G G G T K L T V L
Generated Sequence: Q S A L T Q P A S V S G S P G Q S I T I S C T G T S S D V G G Y N Y V S W Y Q Q H P G K A P K L M I Y D V S N R P S G V S N R F S G S K S G N T A S L T I S G L Q A E D E A D Y Y C S S Y T S S S T L V F G G G T K L T V L
BLOSUM Score: -22.0
Similarity Percentage: 16.19047619047619%
Perplexity: 1.360642433166504
model is on device cuda:0
```

```bash
Sequence pair 64923:
True Sequence: Q A V L T Q P S S L S A S P G A S A S L T C T L R S D I N V G P Y R I Y W Y Q Q K P G S P P Q Y L L R Y N S D S D K R Q G L G V P S R F S G S K D A S A N A G I L L I S G L Q S E D E A D Y Y C L I W H S S A W V F G G G T K L T V L
Generated Sequence: Q S A L T Q P A S V S G S P G Q S I T I S C T G T S S D V G G Y N Y V S W Y Q Q H P G K A P K L M I Y E V S N R P S G V S N R F S G S K S G N T A S L T I S G L Q A E D E A D Y Y C S S Y T S S S T R V F G G G T K L T V L
BLOSUM Score: 59.0
Similarity Percentage: 23.636363636363637%
Perplexity: 1.9959008693695068
model is on device cuda:0

```

```bash
Sequence pair 461:
True Sequence: N F M L T Q P H S V S E S P G K T V T I S C T R S S G S I A S N S V Q W Y Q Q R P G S S P T T V I Y E D N Q R P S G V P D R F S G S I D S S S N S A S L T I S G L K T E D E A D Y Y C Q S Y D S S N V M F G G G T K L T V L
Generated Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S G S S S N I G S N Y V Y W Y Q Q L P G T A P K L L I Y R N N Q R P S G V P D R F S G S K S G T S A S L A I S G L R S E D E A D Y Y C A A W D D S L S G W V F G G G T K L T V L
BLOSUM Score: 260.0
Similarity Percentage: 50.90909090909091%
Perplexity: 1.4107027053833008
model is on device cuda:0
```

```bash
Sequence pair 1221:
True Sequence: N F M L T Q P H S V S E S P G K T V T I S C T G S S G S I A S N Y V Q W Y Q Q R P G S A P T T V I Y E D N Q R P S G V P D R F S G S I D S S S N S A S L I I S G L K T E D E A D Y Y C Q S Y D S G N Q V F G G G T K L T V L
Generated Sequence: Q S V L T Q P P S V S A A P G Q K V T I S C S G S S S N I G N N Y V S W Y Q Q L P G T A P K L L I Y D N N K R P S G I P D R F S G S K S G T S A T L G I T G L Q T G D E A D Y Y C G T W D S S L S A G V F G G G T K L T V L
BLOSUM Score: 269.0
Similarity Percentage: 50.90909090909091%
Perplexity: 1.4474575519561768
model is on device cuda:0

```

```bash
Sequence pair 1224:
True Sequence: S Y E L T Q P P S V S V S P G Q T A R I T C S G D A L P K K Y A Y W Y Q Q R S G Q A P V L V I Y E D N K R P S G I P E R F S G S S S G T L A T L T I S G A Q V E D E A D Y Y C Y S T D I S G G A F G G G T K L S V L
Generated Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S G S S S N I G S N Y V Y W Y Q Q L P G T A P K L L I Y R N N Q R P S G V P D R F S G S K S G T S A S L A I S G L R S E D E A D Y Y C A A W D D S L S G R V F G G G T K L T V L
BLOSUM Score: 21.0
Similarity Percentage: 19.81132075471698%
Perplexity: 1.6391100883483887
model is on device cuda:0
```

```bash
Sequence pair 1920:
True Sequence: D I Q M T Q S P S S L S A S V G D R V T I I C R A S Q D I G T F L A W F Q H K P G R A P K S L I Y E V S T L E S G V P S K F S G S G S G T Y F T L T I S S L Q P E D F A T Y Y C Q Q Y K D Y P F T F G P G T K V D I E
Generated Sequence: E I V L T Q S P A T L S L S P G E R A T L S C R A S Q S V S S Y L A W Y Q Q K P G Q A P R L L I Y D A S N R A T G I P A R F S G S G S G T D F T L T I S S L E P E D F A V Y Y C Q Q R S N W P P L T F G G G T K V E I K
BLOSUM Score: 310.0
Similarity Percentage: 55.140186915887845%
Perplexity: 1.9980391263961792
model is on device cuda:0

```

```bash
Sequence pair 2274:
True Sequence: Q T V V T Q E P S F S V S P G G T V T L T C G L S S G S V S T S F Y P S W Y Q Q T P G Q A P H T L I Y N T N T R S S G V P D R F S G S I L G S K A A L T I T G A Q A D D E S D Y Y C V L Y M S G G T W V F G G G T K L T V L
Generated Sequence: Q S A L T Q P A S V S G S P G Q S I T I S C T G T S S D V G G Y N Y V S W Y Q Q H P G K A P K L M I Y D V S N R P S G V S N R F S G S K S G N T A S L T I S G L Q A E D E A D Y Y C S S Y T S S S T L V F G G G T K L T V L
BLOSUM Score: 308.0
Similarity Percentage: 55.45454545454545%
Perplexity: 1.5825886726379395
model is on device cuda:0

```