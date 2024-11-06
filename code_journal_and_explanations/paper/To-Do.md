# To-Do

- [ ]  sequence similarity between the predicted sequence and the original sequence germline
- [ ]  [IMGT DATABASE](https://www.imgt.org/IMGTindex/directory.php) for each sequence it should have the similarity to the germline,
- [ ]  doing the analysis using the IMGT database for the one that we are not predicting correctly the germile, DO IT AMINO ACID LEVELS!!! from ingt convert to aa
- [ ]  useful for discussion in the thesis/paper: https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2018.02249/full
- [ ]  attention analysis maybe on the structure (extract the attention score from the last layer and project them on the structure)
- [ ]  Do also the non human species analysis (as negative result) --> Do a non-human one as an out of distribution analysis --> did it learn something specific about humans?
- [ ]  check if one specific heavy chain gene has a prefered light chain gene (confusion matrix heavy genes vs light genes) --> from the zurich group they saw that there is always a pair that is preferred for i.e. HIV

germline swapping → harder to say any kind of coherence

- [ ]  we can also try to give the starting point of the framework during the translation
- [ ]  Do the recovery on the light chains with the light_lm, do the recovery on the heavy chains with the heavy_lm

if we only take humans (also diseases) would that help the model?

extrapolate other species?

three peaks in the similarity distribution

different decoding strategies

train with slightly different data

Classification model on heavy2light model generated sequences

Group zurich: experimental check

focus on the design part first

write email to ubelix cluster

change data, hyperparameters and decoding

gene type germline type correctly retrieved?

Run predictions for zurich group

correlation sequence recovery 

see if we have enough data only with the human 

only humans, healthy

maybe also add progressively some diseases

categorize some disease but start with healthy humans

see drop in performance if we compare human healthy to everything

no vaccine

everything human, human healthy, step by step finegrain

1. **Collaboration in Zurich**: It seems that the 60% recovery rate might be insufficient to confidently test the generated sequences in the lab.
2. **Different Options**: either test out new methods we didn’t have time to explore during the thesis or continue with our current focus on antibody design.
    - We could explore two distinct directions: improving antibody design or gaining insights into pairing.
3. **Human Data Focus**: Should we consider using only human samples? If so, we could initially start with healthy human samples and assess if there's sufficient data. If needed, we could broaden the dataset to include various diseases. Ideally, human samples would show the highest recovery rate. --> unfortunately last time we did use all the human data available with all diseases and vaccines
4. **Attention Analysis**: Provide the attention analysis code to Lea.
5. **Humanizing Non-Human Antibody Sequences**: Is it feasible to humanize antibody sequences from non-human species? Note: Data from literature might be poorly organized, with no structured databases available.
6. **Decoding Strategies**: Explore different decoding techniques to potentially improve model performance.
7. The EDA has shown promising separation between naïve and mature antibody sequences.

---

### Next Steps

- Focus on creating a paired human-only dataset and appropriate data splits. --> split at 30%
- Continue focusing on translation, testing various decoding techniques.
- Aim to develop the most robust model possible before deciding the final direction.
- Check the correlation between predicted pLDT (confidence in folding) and sequence quality during folding to assess accuracy. (and maybe find other metrics to asses quality of the predicted sequencd )
- Begin with data from healthy individuals and consider adding disease-related samples later.
- Investigate Zenodo and other sources for additional antibody repertoire data or any paired antibody databases that could expand our dataset. cluster at 70% and then