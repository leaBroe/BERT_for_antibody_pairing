# Heavy-Light Chain Pair Identification and Prediction in Antibodies

Master Thesis 2024  
Institute of Biochemistry and Molecular Medicine  
University of Bern  

### Background
Antibodies are vital proteins produced by the immune system to identify and neutralize harmful substances such as bacteria or viruses. Each antibody comprises two key components: heavy chains and light chains. The pairing of these chains forms the basic structure of an antibody, with the heavy chain containing a variable region (VH) and a constant region (CH), and the light chain containing a variable region (VL) and a constant region (CL). The VH and VL regions are responsible for antigen binding, while the CH and CL regions determine the antibody's class and function.

The sequence pairing of heavy and light chains is crucial as it dictates the unique binding specificity of an antibody. Various types of heavy chains (e.g., IgM, IgG, IgA, IgD, IgE) and two types of light chains (kappa and lambda) exist. During B cell development, genetic rearrangement leads to diverse heavy and light chain pairings, producing a vast array of antibodies with unique antigen-binding capabilities. Next Generation Sequencing (NGS) has enabled comprehensive sequencing of antibody repertoires from patients.

### Problem
Identifying the pairing sequences of heavy and light chains in antibodies is critical for understanding their antigen specificity, advancing antibody engineering, contributing to immunological research, and supporting the development of diagnostics and therapeutics.

### Hypothesis
We hypothesize that neural network-based natural language processing (NLP) models, such as BERT, can effectively classify heavy and light chain pairings as a classification task.

### Idea
We propose training a BERT model, or a simpler transformer model, to classify paired heavy and light chains. The model will be trained to determine whether a given heavy and light chain sequence pair is correctly paired (1) or not (0).

### Model Explanation

**BERT Model for Heavy and Light Chain Classification:**  
We employ the BERT model to classify the pairing of heavy and light chains. By leveraging its deep contextual understanding, BERT can distinguish between correctly and incorrectly paired sequences, providing a robust tool for antibody pairing classification.

**BERT2BERT Model for Sequence Generation:**  
In addition to classification, we utilize a BERT2BERT model for sequence generation, where the heavy model (encoder) generates a context vector from the heavy chain sequence, and the light model (decoder) generates the corresponding light chain sequence. 

**Encoder-Decoder Model:**  
In this context, the encoder-decoder model architecture can be utilized to improve antibody sequence pairing predictions. The encoder is the heavy model, which processes and encodes the heavy chain sequence into a context vector. The decoder is the light model, which uses this context vector to generate or predict the corresponding light chain sequence. This approach leverages the strengths of both models, providing a comprehensive method for pairing classification.

**Masked Language Modeling (MLM):**  
MLM is a pretraining technique where certain tokens in the input sequence are masked, and the model is trained to predict these masked tokens. This allows the model to learn deep contextual representations of sequences, which is essential for accurately predicting sequence pairings in antibodies.

**Next Sentence Prediction (NSP):**  
NSP is another pretraining task used in models like BERT. It trains the model to predict whether a given sentence (or sequence) logically follows another. In the context of antibody pairing, NSP could be adapted to predict whether a light chain sequence logically follows a heavy chain sequence, further refining the model's ability to classify pairings.

### Conclusion
By leveraging advanced NLP models like BERT and integrating techniques such as MLM and NSP, we aim to accurately classify heavy-light chain pairings, contributing valuable insights to immunology and therapeutic development.
