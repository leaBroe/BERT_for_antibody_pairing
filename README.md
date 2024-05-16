This Repo is intended for Backups and Notes  
The "official" repo for my master thesis can be found at  
[https://github.com/ibmm-unibe-ch/OAS_paired_sequences_cls/tree/main](https://github.com/ibmm-unibe-ch/OAS_paired_sequences_cls/tree/main)

# HEAVY-LIGHT CHAIN PAIRS CLASSIFICATION (OAS_paired_sequences_cls)

### Background: 
Antibodies are proteins produced by the immune system that play a crucial role in recognizing and neutralizing harmful substances called antigens, such as bacteria or viruses. Each antibody consists of two main components: heavy chains and light chains. The pairing of heavy and light chains forms the basic structure of an antibody. The heavy chain is larger and consists of a variable region (VH) and a constant region (CH). Similarly, the light chain has a variable region (VL) and a constant region (CL). The VH and VL regions are responsible for binding to the specific antigen, while the CH and CL regions provide stability and determine the antibody's class and function.
The pairing sequence refers to the unique combination of heavy and light chains in an antibody. There are different types of heavy chains (such as IgM, IgG, IgA, IgD, and IgE) and two types of light chains (kappa and lambda). During B cell development, the genetic rearrangement process leads to the production of diverse heavy and light chain pairings, resulting in a wide variety of antibodies with unique binding specificities. Next Generation Sequencing (NGS) has allowed the sequencing of the entire repertoire from patients. 

### Problem: 
Identifying the pairing sequence in antibodies allows us to comprehend their antigen specificity, aids in antibody engineering, contributes to immunological research, and facilitates the development of diagnostics and therapeutics.
### Hypothesis:  
We believe that NN NLP models such as BERT are able to identify heavy and light chains (Devlin et al. 2018) pairing as a  classification task .
### Idea: 
Train a BERT model (or a simple transformer = to classify paired light and heavy chain (heavy, light, paired (0=no,1=yes))
