# TabulaSapiens

This repository holds code to analyze data from the Tabula Sapiens v2.0 dataset. The analysis is focused on transcription factors.

## Contents

Identification of transcription factor programs:
- based on correlation:
    - cosine
    - spearman
- with dimensionality reduction:
    - pca + spearman
- cNMF: https://github.com/dylkot/cNMF

Prediction of transcription factor binding sites:
- retrieval of transcription factor binding motifs from JASPAR: https://jaspar.elixir.no/
- retrieval of corresponding amino acid sequences
- fine-tuned custom esm2 model