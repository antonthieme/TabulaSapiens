import os
import argparse
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmwrite
from scipy.sparse import issparse
from cnmf import cNMF
import pickle
from sklearn.linear_model import LinearRegression
import scipy.sparse as sp


def setup_logging(logfile="pipeline.log"):
    """Set up logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile)]
    )
    return logging.getLogger()


def calculate_dispersions_norm(adata, layer=None):
    """
    Calculate normalized dispersions and add them to adata.var.
    
    Parameters:
    - adata: AnnData object
    - layer: If specified, use the data from this layer (e.g., 'raw_counts').
    
    Returns:
    - Updated adata with `mean`, `variance`, `dispersion`, and `dispersions_norm` in `adata.var`.
    """
    if layer:
        data = adata.layers[layer]
    else:
        data = adata.X

    # Ensure the matrix is dense if it's sparse
    if sp.issparse(data):
        logger.info("Converting sparse matrix to dense for dispersion calculation...")
        data = data.toarray()
        print('data')
        print(data)

    # Compute mean and variance for each gene
    gene_means = np.mean(data, axis=0)  # Mean expression per gene
    gene_variances = np.var(data, axis=0)  # Variance of expression per gene
    print('gene_means')
    print(gene_means)
    print('gene_variances')
    print(gene_variances)

    # Compute dispersions
    gene_means = np.asarray(gene_means).flatten()
    gene_variances = np.asarray(gene_variances).flatten()
    dispersions = np.divide(
        gene_variances, gene_means, out=np.zeros_like(gene_variances), where=gene_means > 0
    )  # Set dispersion to 0 where mean = 0

    # Fit a trend line to mean-variance relationship
    log_means = np.log10(gene_means + 1e-10)  # Add small value to avoid log(0)
    log_dispersions = np.log10(dispersions + 1e-10)
    valid = np.isfinite(log_means) & np.isfinite(log_dispersions)  # Ignore NaN/infinite values
    print('log_means')
    print(log_means)
    print('log_dispersions')
    print(log_dispersions)
    print('valid')
    print(valid)

    # Use linear regression to approximate the mean-dispersion relationship
    model = LinearRegression()
    model.fit(log_means[valid].reshape(-1, 1), log_dispersions[valid])

    # Predicted (expected) log-dispersions
    expected_log_dispersions = model.predict(log_means.reshape(-1, 1))
    expected_dispersions = 10**expected_log_dispersions

    # Normalize dispersions
    dispersions_norm = (dispersions - expected_dispersions) / np.std(dispersions - expected_dispersions)

    # Add results to adata.var
    adata.var['mean'] = gene_means
    adata.var['variance'] = gene_variances
    adata.var['dispersion'] = dispersions
    adata.var['dispersions_norm'] = dispersions_norm

    return adata


def preprocess_data(adata, hvg_fraction=0.25):
    """
    Preprocess the AnnData object:
    - Filter cells and genes
    - Compute HVGs using raw counts
    - Subset the dataset to HVGs
    """
    logger.info(f"Starting preprocessing for dataset with shape {adata.shape}...")

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=3)
    sc.pp.filter_genes(adata, min_cells=3)

    # Ensure raw counts are used for HVG calculation
    if 'raw_counts' in adata.layers:
        logger.info("Using raw counts for HVG calculation...")
        print('adata')
        print(adata)
        print('raw_counts')
        print(adata.layers['raw_counts'])
        adata = calculate_dispersions_norm(adata, layer='raw_counts')
        adata.X = adata.layers['raw_counts']
    else:
        raise ValueError("Raw counts not found in `adata.layers['raw_counts']`.")

    # Select top HVGs based on dispersions_norm
    num_hvgs = int(len(adata.var) * hvg_fraction)
    top_hvgs = adata.var.sort_values('dispersions_norm', ascending=False).head(num_hvgs).index
    logger.info(f"Selecting {num_hvgs} highly variable genes from {adata.shape[1]} total genes.")
    adata = adata[:, top_hvgs]
    logger.info(f"Dataset now has {adata.shape[0]} cells and {adata.shape[1]} genes.")

    return adata


def save_raw_counts(adata, output_dir, run_name):
    """
    Save raw counts in a format compatible with cNMF.
    For sparse matrices, save in a 10x Genomics-compatible format (matrix.mtx, genes.tsv, barcodes.tsv).
    For dense matrices, save as a .tsv file.

    Parameters:
    - adata: AnnData object containing raw counts in `adata.layers['raw_counts']`
    - output_dir: Directory to save the files
    - run_name: Name prefix for the output files

    Returns:
    - counts_fn: Path to the saved file/directory
    - is_sparse: Whether the matrix is sparse (True if sparse, False if dense)
    """
    if 'raw_counts' not in adata.layers:
        raise ValueError("Raw counts not found in `adata.layers['raw_counts']`. Ensure raw counts are available.")

    raw_counts = adata.layers['raw_counts']
    is_sparse = sp.issparse(raw_counts)

    # Subset the raw counts to include only HVGs (filtered genes in adata.var_names)
    hvg_indices = adata.var_names.get_indexer(adata.var.index)  # Map gene names to integer indices
    raw_counts = raw_counts[:, hvg_indices]

    if is_sparse:
        # Save as a 10x-compatible directory
        counts_dir = os.path.join(output_dir, run_name)
        os.makedirs(counts_dir, exist_ok=True)

        # Save matrix.mtx (transpose to genes as rows, cells as columns)
        matrix_path = os.path.join(counts_dir, "matrix.mtx")
        mmwrite(matrix_path, raw_counts.T)  # Transpose the matrix before saving

        # Save genes.tsv (must match rows of matrix.mtx)
        genes_path = os.path.join(counts_dir, "genes.tsv")
        genes_df = pd.DataFrame({
            "gene_id": adata.var_names,
            "gene_name": adata.var["gene_symbol"].fillna(pd.Series(adata.var_names, index=adata.var_names)) if "gene_symbol" in adata.var else adata.var_names
        })
        genes_df.to_csv(genes_path, sep="\t", index=False, header=False)

        # Save barcodes.tsv (must match columns of matrix.mtx)
        barcodes_path = os.path.join(counts_dir, "barcodes.tsv")
        pd.Series(adata.obs_names).to_csv(barcodes_path, sep="\t", index=False, header=False)

        logger.info(f"Raw counts saved as a 10x-compatible directory: {counts_dir}")
        logger.info(f"Matrix shape: {raw_counts.shape}")
        return counts_dir, is_sparse

    else:
        # Save as a single dense .tsv file
        counts_fn = os.path.join(output_dir, f"{run_name}_raw_counts.tsv")
        pd.DataFrame(
            raw_counts if not is_sparse else raw_counts.toarray(),
            index=adata.var_names,
            columns=adata.obs_names
        ).to_csv(counts_fn, sep="\t")
        logger.info(f"Raw counts saved as a dense TSV file: {counts_fn}")
        logger.info(f"Matrix shape: {raw_counts.shape}")
        return counts_fn, is_sparse



def run_cnmf(adata, output_dir, run_name, k_to_try, numiter, seed):
    """
    Run the cNMF pipeline on the given AnnData object.
    """
    logger.info(f"Running cNMF for {run_name}...")
    
    # Save raw counts
    counts_fn, is_sparse = save_raw_counts(adata, output_dir, run_name)

    # If the matrix is sparse, point to the `.mtx` file inside the directory
    if is_sparse:
        counts_fn = os.path.join(counts_fn, "matrix.mtx")

    # Initialize and run cNMF
    cnmf_obj = cNMF(output_dir=output_dir, name=run_name)
    cnmf_obj.prepare(
        counts_fn=counts_fn,  # Pass the correct file path
        components=k_to_try,
        n_iter=numiter,
        seed=seed,
        num_highvar_genes=adata.shape[1],  # Use all HVGs from preprocessing
        densify=not is_sparse  # Densify if the input is not sparse
    )
    cnmf_obj.factorize(worker_i=0, total_workers=1)
    cnmf_obj.combine()

    # Save results
    cnmf_obj.k_selection_plot(close_fig=True)
    with open(os.path.join(output_dir, run_name, f"cnmf_obj.pkl"), 'wb') as f:
        pickle.dump(cnmf_obj, f)

    logger.info(f"cNMF completed and results saved for {run_name}.")




def main(task_id, job_id):
    """
    Main pipeline to preprocess data, run cNMF, and save results for a given task ID.
    """
    # Paths and parameters
    base_data_dir = os.path.join('..', '..', 'data', 'donor_tissue_celltype_split')
    base_results_dir = os.path.join('..', '..', 'results', 'cnmf', 'tf_cNMF_dtc', job_id)
    adata_path = os.path.join('..', '..', 'data', 'adata_tf.h5ad')
    os.makedirs(base_results_dir, exist_ok=True)

    k_to_try = np.arange(2, 10)
    numiter = 200
    hvg_fraction = 0.25
    seed = 42

    # Load full dataset
    logger.info("Loading full dataset...")
    adata = sc.read_h5ad(adata_path)
    sc.pp.filter_cells(adata, min_genes=3)
    sc.pp.filter_genes(adata, min_cells=3)

    # Get cell types for the task
    celltypes = list(adata.obs['broad_cell_class'].unique())
    celltype = celltypes[task_id]
    logger.info(f"Processing cell type: {celltype}")

    # Process individual donor-tissue-celltype datasets
    celltype_folder = os.path.join(base_data_dir, celltype)
    if not os.path.exists(celltype_folder):
        logger.warning(f"Folder for cell type {celltype} does not exist. Skipping...")
        return

    for filename in os.listdir(celltype_folder):
        filepath = os.path.join(celltype_folder, filename)
        logger.info(f"Processing file: {filepath}")

        adata_sub = sc.read_h5ad(filepath)

        # Skip datasets with fewer than 50 cells
        if adata_sub.shape[0] <= 50:
            logger.warning(f"Dataset {filename} has fewer than 50 cells. Skipping...")
            continue

        adata_sub = preprocess_data(adata_sub, hvg_fraction=hvg_fraction)

        # Prepare output directory
        output_dir = os.path.join(base_results_dir, celltype)
        os.makedirs(output_dir, exist_ok=True)

        # Run cNMF
        run_name = filename.replace('adata_tf_', '').split('.h5ad')[0]
        run_name = run_name.replace('_', '-', 1).replace('_', '-', run_name.count('_')) # fixing the dtc foldername to have - between d,t and c
        run_cnmf(adata_sub, output_dir, run_name, k_to_try, numiter, seed)


if __name__ == "__main__":
    # Parse task ID
    parser = argparse.ArgumentParser(description='TASK ID')
    parser.add_argument('--task', type=int, required=True, help='Task ID')
    parser.add_argument('--jobid', type=str, required=True, help='Job ID')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging()

    # Run the pipeline for the given task
    main(args.task, args.jobid)
