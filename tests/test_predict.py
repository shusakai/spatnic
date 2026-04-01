"""Tests for the predict API."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from spatnic.model import StudentTVAECls
from spatnic.predict import _load_checkpoint, predict


def _create_checkpoint(tmp_path, n_genes=50, n_classes=2, label_map=None):
    """Create a minimal checkpoint for testing."""
    model = StudentTVAECls(in_dim=n_genes, n_classes=n_classes)
    if label_map is None:
        label_map = {0: "Normal", 1: "Tumor"}
    ckpt = {
        "state_dict": model.state_dict(),
        "gene_names": [f"gene_{i}" for i in range(n_genes)],
        "mu_g": np.zeros(n_genes, dtype=np.float32),
        "std_g": np.ones(n_genes, dtype=np.float32),
        "label_map": label_map,
        "model_config": {
            "n_classes": n_classes,
            "latent_dim": 32,
            "hidden": 512,
            "dropout": 0.2,
            "heteroscedastic": True,
            "stu_nu": 5.0,
            "learnable_nu": False,
        },
    }
    path = str(tmp_path / "test_ckpt.pt")
    torch.save(ckpt, path)
    return path


def _create_adata(n_cells=20, n_genes=50):
    """Create a minimal AnnData for testing."""
    import anndata as ad
    import pandas as pd
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    X = sp.random(n_cells, n_genes, density=0.3, format="csr", dtype=np.float32)
    var = pd.DataFrame(index=gene_names)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    return ad.AnnData(X=X, var=var, obs=obs)


class TestLoadCheckpoint:

    def test_load_by_path(self, tmp_path):
        path = _create_checkpoint(tmp_path)
        ckpt = _load_checkpoint(path)
        assert "state_dict" in ckpt
        assert len(ckpt["gene_names"]) == 50

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            _load_checkpoint("/nonexistent/path/model.pt")

    def test_unknown_model_name(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_checkpoint("nonexistent_model_name")


class TestPredict:

    def test_basic_predict(self, tmp_path):
        path = _create_checkpoint(tmp_path)
        adata = _create_adata()
        result = predict(adata, model=path, device="cpu", num_workers=0)
        assert "spatnic_pred" in adata.obs.columns
        assert "spatnic_score" in adata.obs.columns
        assert "spatnic_score_Normal" in adata.obs.columns
        assert "spatnic_score_Tumor" in adata.obs.columns
        assert len(adata.obs["spatnic_pred"]) == 20
        assert set(adata.obs["spatnic_pred"].unique()).issubset({"Normal", "Tumor"})

    def test_custom_keys(self, tmp_path):
        path = _create_checkpoint(tmp_path)
        adata = _create_adata()
        predict(adata, model=path, key_added="my_pred",
                score_key_added="my_score", device="cpu", num_workers=0)
        assert "my_pred" in adata.obs.columns
        assert "my_score" in adata.obs.columns

    def test_return_latent(self, tmp_path):
        path = _create_checkpoint(tmp_path)
        adata = _create_adata()
        result, latent = predict(adata, model=path, return_latent=True,
                                 device="cpu", num_workers=0)
        assert latent.shape == (20, 32)

    def test_threshold(self, tmp_path):
        path = _create_checkpoint(tmp_path)
        adata1 = _create_adata()
        adata2 = _create_adata()
        predict(adata1, model=path, threshold=0.0, device="cpu", num_workers=0)
        predict(adata2, model=path, threshold=1.0, device="cpu", num_workers=0)
        # threshold=0.0 -> all Tumor; threshold=1.0 -> all Normal
        assert (adata1.obs["spatnic_pred"] == "Tumor").all()
        assert (adata2.obs["spatnic_pred"] == "Normal").all()

    def test_partial_gene_match(self, tmp_path):
        path = _create_checkpoint(tmp_path, n_genes=50)
        # adata with only 30 of the 50 genes
        import anndata as ad
        gene_names = [f"gene_{i}" for i in range(30)]
        X = sp.random(10, 30, density=0.3, format="csr", dtype=np.float32)
        adata = ad.AnnData(X=X, var={"gene": gene_names})
        # Should still work (60% match)
        predict(adata, model=path, device="cpu", num_workers=0)
        assert "spatnic_pred" in adata.obs.columns

    def test_scores_are_probabilities(self, tmp_path):
        path = _create_checkpoint(tmp_path)
        adata = _create_adata()
        predict(adata, model=path, device="cpu", num_workers=0)
        scores = adata.obs["spatnic_score"].values
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_per_class_probabilities_sum_to_one(self, tmp_path):
        path = _create_checkpoint(tmp_path)
        adata = _create_adata()
        predict(adata, model=path, device="cpu", num_workers=0)
        normal_prob = adata.obs["spatnic_score_Normal"].values
        tumor_prob = adata.obs["spatnic_score_Tumor"].values
        np.testing.assert_allclose(normal_prob + tumor_prob, 1.0, atol=1e-6)

    def test_custom_keys_per_class(self, tmp_path):
        path = _create_checkpoint(tmp_path)
        adata = _create_adata()
        predict(adata, model=path, score_key_added="my_score",
                device="cpu", num_workers=0)
        assert "my_score_Normal" in adata.obs.columns
        assert "my_score_Tumor" in adata.obs.columns
