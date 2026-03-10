"""Preprocessing utilities for gene alignment and standardization."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix


def standardize_csr(X_csr, mean, std):
    """Standardize a CSR matrix column-wise using precomputed mean and std."""
    X = X_csr.tocoo(copy=True)
    col_std = std[X.col].copy()
    col_std[col_std == 0] = 1.0
    X.data = (X.data - mean[X.col]) / col_std
    return X.tocsr()


def align_genes(adata, train_genes, layer_key=None):
    """Align adata to the training gene order, zero-filling missing genes.

    Parameters
    ----------
    adata : AnnData
        Input data (log-normalized).
    train_genes : np.ndarray
        Gene names in training order.
    layer_key : str or None
        Layer to use. If None, uses ``adata.X``.

    Returns
    -------
    X_aligned : csr_matrix
        Aligned sparse matrix (n_cells x len(train_genes)).
    n_matched : int
        Number of matched genes.
    """
    # Get expression matrix
    if layer_key is not None and layer_key in adata.layers:
        X_src = adata.layers[layer_key]
    else:
        X_src = adata.X
    if not sp.issparse(X_src):
        X_src = csr_matrix(np.asarray(X_src, dtype=np.float32))
    else:
        X_src = X_src.tocsr()

    # Try var_names first, fall back to feature_name
    bm_vars = np.array(adata.var_names)
    if len(np.intersect1d(train_genes, bm_vars)) < int(0.5 * len(train_genes)):
        if "feature_name" in adata.var.columns:
            bm_vars = np.array(adata.var["feature_name"].astype(str))

    # Build index mapping
    bm_index = {g: i for i, g in enumerate(bm_vars)}
    present_mask = np.array([g in bm_index for g in train_genes])
    idx_tr_present = np.where(present_mask)[0]
    idx_bm_present = np.array(
        [bm_index[g] for g in train_genes[present_mask]]
    )

    n_matched = int(present_mask.sum())

    # Rearrange columns
    n_cells = adata.n_obs
    G = len(train_genes)
    X_src_csc = X_src.tocsc()
    X_eval_csc = csc_matrix((n_cells, G), dtype=np.float32)
    if idx_tr_present.size > 0:
        X_eval_csc[:, idx_tr_present] = X_src_csc[:, idx_bm_present]

    return X_eval_csc.tocsr(), n_matched
