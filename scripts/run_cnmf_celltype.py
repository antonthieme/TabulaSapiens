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
from sklearn.decomposition import PCA
import networkx as nx
import plotly.graph_objects as go
import scanpy as sc
from cnmf import cNMF
import pickle


numiter=200 # Number of NMF replicates. Set this to a larger value ~200 for real data. We set this to a relatively low value here for illustration at a faster speed
numhvgenes=1639 ## Number of over-dispersed genes to use for running the actual factorizations

## Results will be saved to [output_directory]/[run_name] which in this example is example_PBMC/cNMF/pbmc_cNMF
output_directory = '../../results/cnmf'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
run_name = 'tf_cNMF_dtc'

seed = 42 ## Specify a seed pseudorandom number generation for reproducibility

celltypes = ['fibroblast', 'neuron', 'epithelial cell']

k_to_try = np.arange(2, 30)

adata_tf = read_h5ad(os.path.join('..', '..', 'data', 'adata_tf.h5ad'))
sc.pp.filter_cells(adata_tf, min_genes=3)
sc.pp.filter_genes(adata_tf, min_cells=3)

"""
adata_tf.obs['group'] = (
    adata_tf.obs['donor'] + '_' + adata_tf.obs['tissue'] + '_' + adata_tf.obs['broad_cell_class']
)
group_counts = adata_tf.obs['group'].value_counts()
valid_groups = group_counts[group_counts >= 50].index
adata_tf = adata_tf[adata_tf.obs['group'].isin(valid_groups)].copy()
del adata_tf.obs['group']
"""

for ct in celltypes:
    print(ct)
    adata_tf_subs = adata_tf[adata_tf.obs['broad_cell_class'] == ct].copy()

    output_directory = os.path.join('../../results/cnmf/tf_cNMF_celltype', ct)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    run_name = 'run_1'

    cnmf_obj = cNMF(output_dir=output_directory, name=run_name)
    cnmf_obj.prepare(counts_fn=os.path.join('..', '..', 'data', 'adata_tf.h5ad'), components=k_to_try, n_iter=numiter, seed=seed, num_highvar_genes=numhvgenes)
    cnmf_obj.factorize(worker_i=0, total_workers=1)
    cnmf_obj.combine()

    cnmf_obj.k_selection_plot(close_fig=False)

    with open(os.path.join(output_directory, run_name, 'cnmf_obj_after_combine.pkl'), 'wb') as f:
        pickle.dump(cnmf_obj, f)