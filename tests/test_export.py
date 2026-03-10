"""Tests for checkpoint export and loading."""

import numpy as np
import pytest
import torch

from spatnic.export import export_checkpoint
from spatnic.model import StudentTVAECls
from spatnic.predict import _load_checkpoint


class TestExportCheckpoint:

    def _save_dummy_state_dict(self, tmp_path, in_dim=50, n_classes=2):
        model = StudentTVAECls(in_dim=in_dim, n_classes=n_classes)
        sd_path = str(tmp_path / "dummy_sd.pth")
        torch.save(model.state_dict(), sd_path)
        return sd_path

    def test_export_with_save_path(self, tmp_path):
        sd_path = self._save_dummy_state_dict(tmp_path)
        out_path = str(tmp_path / "test_model.pt")
        gene_names = [f"gene_{i}" for i in range(50)]
        mu_g = np.zeros(50, dtype=np.float32)
        std_g = np.ones(50, dtype=np.float32)

        export_checkpoint(
            state_dict_path=sd_path,
            gene_names=gene_names,
            mu_g=mu_g,
            std_g=std_g,
            save_path=out_path,
            label_map={0: "Normal", 1: "Tumor"},
        )

        ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt
        assert "gene_names" in ckpt
        assert "mu_g" in ckpt
        assert "std_g" in ckpt
        assert "label_map" in ckpt
        assert "model_config" in ckpt
        assert len(ckpt["gene_names"]) == 50
        assert ckpt["label_map"] == {0: "Normal", 1: "Tumor"}

    def test_export_with_model_name(self, tmp_path, monkeypatch):
        # Redirect user weights dir to tmp_path
        import spatnic.export as export_mod
        monkeypatch.setattr(export_mod, "_WEIGHTS_DIR_USER", tmp_path / "weights")

        sd_path = self._save_dummy_state_dict(tmp_path)
        gene_names = [f"gene_{i}" for i in range(50)]

        export_checkpoint(
            state_dict_path=sd_path,
            gene_names=gene_names,
            mu_g=np.zeros(50, dtype=np.float32),
            std_g=np.ones(50, dtype=np.float32),
            model_name="test_model",
            label_map={0: "A", 1: "B"},
        )

        expected_path = tmp_path / "weights" / "test_model.pt"
        assert expected_path.exists()

    def test_export_requires_path_or_name(self, tmp_path):
        sd_path = self._save_dummy_state_dict(tmp_path)
        with pytest.raises(ValueError, match="Either save_path or model_name"):
            export_checkpoint(
                state_dict_path=sd_path,
                gene_names=["a"],
                mu_g=np.zeros(1),
                std_g=np.ones(1),
            )

    def test_roundtrip_load(self, tmp_path):
        """Export a checkpoint and load it back."""
        sd_path = self._save_dummy_state_dict(tmp_path, in_dim=30)
        out_path = str(tmp_path / "roundtrip.pt")
        gene_names = [f"g{i}" for i in range(30)]

        export_checkpoint(
            state_dict_path=sd_path,
            gene_names=gene_names,
            mu_g=np.random.randn(30).astype(np.float32),
            std_g=np.abs(np.random.randn(30)).astype(np.float32) + 0.1,
            save_path=out_path,
            label_map={0: "Normal", 1: "Tumor"},
        )

        ckpt = _load_checkpoint(out_path)
        model = StudentTVAECls(in_dim=30, n_classes=2)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        x = torch.randn(2, 30)
        with torch.no_grad():
            _, _, _, _, logits = model(x)
        assert logits.shape == (2, 2)
