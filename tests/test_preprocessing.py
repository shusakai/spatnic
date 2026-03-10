"""Tests for preprocessing utilities."""

import numpy as np
import pytest
import scipy.sparse as sp

from spatnic.preprocessing import align_genes, standardize_csr


class TestStandardizeCSR:

    def test_basic_standardization(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        X = sp.csr_matrix(data)
        mean = np.array([2.5, 3.5, 4.5], dtype=np.float32)
        std = np.array([1.5, 1.5, 1.5], dtype=np.float32)
        result = standardize_csr(X, mean, std)
        expected = (data - mean) / std
        np.testing.assert_allclose(result.toarray(), expected, atol=1e-5)

    def test_zero_std_handling(self):
        X = sp.csr_matrix(np.array([[1.0, 2.0]], dtype=np.float32))
        mean = np.array([0.0, 0.0], dtype=np.float32)
        std = np.array([0.0, 1.0], dtype=np.float32)
        # Should not raise; zero std columns replaced with 1.0
        result = standardize_csr(X, mean, std)
        assert result.shape == (1, 2)

    def test_sparse_output(self):
        X = sp.csr_matrix(np.eye(3, dtype=np.float32))
        mean = np.zeros(3, dtype=np.float32)
        std = np.ones(3, dtype=np.float32)
        result = standardize_csr(X, mean, std)
        assert sp.issparse(result)


class TestAlignGenes:

    def _make_adata(self, gene_names, n_cells=10):
        import anndata as ad
        import pandas as pd
        X = sp.random(n_cells, len(gene_names), density=0.3, format="csr",
                      dtype=np.float32)
        var = pd.DataFrame(index=gene_names)
        obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
        return ad.AnnData(X=X, var=var, obs=obs)

    def test_perfect_match(self):
        genes = ["A", "B", "C"]
        adata = self._make_adata(genes)
        X_aligned, n = align_genes(adata, np.array(genes))
        assert n == 3
        assert X_aligned.shape == (10, 3)

    def test_partial_match(self):
        adata = self._make_adata(["A", "B", "C", "D"])
        train_genes = np.array(["B", "C", "E"])
        X_aligned, n = align_genes(adata, train_genes)
        assert n == 2  # B, C matched; E missing
        assert X_aligned.shape == (10, 3)

    def test_no_match(self):
        adata = self._make_adata(["A", "B", "C"])
        train_genes = np.array(["X", "Y", "Z"])
        X_aligned, n = align_genes(adata, train_genes)
        assert n == 0
        assert X_aligned.shape == (10, 3)
        # All zeros
        assert X_aligned.nnz == 0

    def test_feature_name_fallback(self):
        import anndata as ad
        X = sp.random(5, 3, density=0.5, format="csr", dtype=np.float32)
        adata = ad.AnnData(
            X=X,
            var={"feature_name": ["GENE_A", "GENE_B", "GENE_C"]},
        )
        adata.var_names = ["id1", "id2", "id3"]
        train_genes = np.array(["GENE_A", "GENE_B", "GENE_C"])
        X_aligned, n = align_genes(adata, train_genes)
        assert n == 3

    def test_output_is_csr(self):
        adata = self._make_adata(["A", "B"])
        X_aligned, _ = align_genes(adata, np.array(["A", "B"]))
        assert sp.isspmatrix_csr(X_aligned)
