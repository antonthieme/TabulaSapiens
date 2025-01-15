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


numiter=100 # Number of NMF replicates. Set this to a larger value ~200 for real data. We set this to a relatively low value here for illustration at a faster speed
numhvgenes=2000 ## Number of over-dispersed genes to use for running the actual factorizations

## Results will be saved to [output_directory]/[run_name] which in this example is example_PBMC/cNMF/pbmc_cNMF
output_directory = '../../results/cnmf/script_run'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
run_name = 'tf_cNMF_4'

## Specify the Ks to use as a space separated list in this case "5 6 7 8 9 10"
K = ' '.join([str(i) for i in range(5,50)])

## To speed this up, you can run it for only K=7-8 with the option below
#K = ' '.join([str(i) for i in range(7,9)])


seed = 14 ## Specify a seed pseudorandom number generation for reproducibility

## Path to the filtered counts dataset we output previously
countfn = '../../data/adata_tf_cnmf.h5ad'


## Initialize the cnmf object that will be used to run analyses
cnmf_obj = cNMF(output_dir=output_directory, name=run_name)

## Prepare the data, I.e. subset to 2000 high-variance genes, and variance normalize
cnmf_obj.prepare(counts_fn=countfn, components=np.arange(5,50), n_iter=numiter, seed=14, num_highvar_genes=numhvgenes)

# Pickle the cNMF object
with open(os.path.join(output_directory, 'cnmf_obj_after_prepare.pkl'), 'wb') as f:
    pickle.dump(cnmf_obj, f)


## Specify that the jobs are being distributed over a single worker (total_workers=1) and then launch that worker
cnmf_obj.factorize(worker_i=0, total_workers=1)

with open(os.path.join(output_directory, 'cnmf_obj_after_factorize.pkl'), 'wb') as f:
    pickle.dump(cnmf_obj, f)

cnmf_obj.combine()

with open(os.path.join(output_directory, 'cnmf_obj_after_combine.pkl'), 'wb') as f:
    pickle.dump(cnmf_obj, f)

cnmf_obj.k_selection_plot(close_fig=False)