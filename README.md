# Heavy-Light Chain Pair Identification and Prediction in Antibodies

### Background
Antibodies are vital proteins produced by the immune system to identify and neutralize harmful substances such as bacteria or viruses. Each antibody comprises two key components: heavy chains and light chains. The pairing of these chains forms the basic structure of an antibody, with the heavy chain containing a variable region (VH) and a constant region (CH), and the light chain containing a variable region (VL) and a constant region (CL). The VH and VL regions are responsible for antigen binding, while the CH and CL regions determine the antibody's class and function.

The sequence pairing of heavy and light chains is crucial as it dictates the unique binding specificity of an antibody. Various types of heavy chains (e.g., IgM, IgG, IgA, IgD, IgE) and two types of light chains (kappa and lambda) exist. During B cell development, genetic rearrangement leads to diverse heavy and light chain pairings, producing a vast array of antibodies with unique antigen-binding capabilities. Next Generation Sequencing (NGS) has enabled comprehensive sequencing of antibody repertoires from patients.

### Problem
Identifying the pairing sequences of heavy and light chains in antibodies is critical for understanding their antigen specificity, advancing antibody engineering, contributing to immunological research, and supporting the development of diagnostics and therapeutics.

### Hypothesis
We hypothesize that neural network-based natural language processing (NLP) models, such as BERT, can effectively classify heavy and light chain pairings as a classification task.

### Idea
We propose training a BERT model, or a simpler transformer model, to classify paired heavy and light chains. The model will be trained to determine whether a given heavy and light chain sequence pair is correctly paired (1) or not (0).