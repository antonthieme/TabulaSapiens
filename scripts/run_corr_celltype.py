import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import anndata
import string
import gc
from anndata import read_h5ad
from anndata import read_csv
from pandas import DataFrame
import h5py
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import networkx as nx
import plotly.graph_objects as go
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import pickle

adata_tf = read_h5ad(os.path.join('..', '..', 'data', 'adata_tf.h5ad'))

# Step 1: Extract the gene expression matrix
# Assuming adata.X is sparse; convert to dense
X = adata_tf.X.toarray()  # Converts to dense (use .A for CSR matrices)

# Assign gene names as row/column labels if available
gene_names = adata_tf.var.index.tolist()  # List of gene names

# Identify genes with zero variance
gene_variance = np.var(X, axis=0)
non_constant_genes = gene_variance > 0

# Filter matrix
X_filtered = X[:, non_constant_genes]
gene_names_filtered = np.array(gene_names)[non_constant_genes]

adata_tf.obs.cell_ontology_class.unique()

# Step 1: Group cells by cell type
cell_types = adata_tf.obs['cell_ontology_class'].unique()

# Step 2: Initialize a dictionary to store results
correlation_matrices = {}

# Step 3: Loop through each cell type
for cell_type in cell_types:
    # Subset AnnData object for the current cell type
    cell_type_data = adata_tf[adata_tf.obs['cell_ontology_class'] == cell_type]
    
    # Step 4: Compute Spearman's rho
    spearman_corr_matrix, _ = spearmanr(X_filtered, axis=0)
    
    # Store the correlation matrix in the dictionary
    correlation_matrices[cell_type] = spearman_corr_matrix

    # Print progress
    print(f"Computed correlation for cell type: {cell_type}")

# Step 5: Access results
print(f"Available cell types: {list(correlation_matrices.keys())}")
print(f"Correlation matrix for first cell type:\n{correlation_matrices[cell_types[0]]}")

save_path = os.path.join('..', '..', 'results', 'corr', 'celltype', 'correlation_matrices.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(correlation_matrices, f)