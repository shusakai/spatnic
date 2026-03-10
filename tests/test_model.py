"""Tests for the StudentTVAECls model."""

import numpy as np
import pytest
import torch

from spatnic.model import StudentTVAECls


class TestStudentTVAECls:
    """Tests for model construction and forward pass."""

    def test_default_construction(self):
        model = StudentTVAECls(in_dim=100, n_classes=2)
        assert model.in_dim == 100
        assert model.latent_dim == 32
        assert model.hetero is True
        assert model.learnable_nu is False

    def test_forward_shape(self):
        model = StudentTVAECls(in_dim=100, n_classes=2)
        model.eval()
        x = torch.randn(8, 100)
        mu_z, logvar_z, dec_out, logvar_x, logits = model(x)
        assert mu_z.shape == (8, 32)
        assert logvar_z.shape == (8, 32)
        assert dec_out.shape == (8, 100)
        assert logvar_x.shape == (8, 100)
        assert logits.shape == (8, 2)

    def test_forward_multiclass(self):
        model = StudentTVAECls(in_dim=50, n_classes=5, latent_dim=16, hidden=64)
        model.eval()
        x = torch.randn(4, 50)
        _, _, _, _, logits = model(x)
        assert logits.shape == (4, 5)

    def test_non_heteroscedastic(self):
        model = StudentTVAECls(in_dim=50, n_classes=2, heteroscedastic=False)
        model.eval()
        x = torch.randn(4, 50)
        mu_z, logvar_z, dec_out, logvar_x, logits = model(x)
        assert dec_out.shape == (4, 50)
        assert logvar_x is None

    def test_learnable_nu(self):
        model = StudentTVAECls(in_dim=50, n_classes=2, learnable_nu=True)
        nu = model.current_nu()
        assert nu.item() > 2.0
        assert model.learnable_nu is True

    def test_fixed_nu(self):
        model = StudentTVAECls(in_dim=50, n_classes=2, stu_nu=8.0)
        assert model.current_nu().item() == pytest.approx(8.0)

    def test_denoising_only_in_training(self):
        model = StudentTVAECls(in_dim=50, n_classes=2, denoise_p=0.5)
        x = torch.ones(4, 50)

        # eval mode: no denoising
        model.eval()
        with torch.no_grad():
            mu_z1, _, _, _, _ = model(x)
            mu_z2, _, _, _, _ = model(x)
        assert torch.allclose(mu_z1, mu_z2)

    def test_encode_decode_roundtrip(self):
        model = StudentTVAECls(in_dim=50, n_classes=2)
        model.eval()
        x = torch.randn(4, 50)
        with torch.no_grad():
            mu_z, logvar_z = model.encode(x)
            z = mu_z  # use mean (no noise)
            dec_out, logvar_x = model.decode(z)
        assert dec_out.shape == x.shape

    def test_state_dict_save_load(self, tmp_path):
        model = StudentTVAECls(in_dim=100, n_classes=2)
        path = tmp_path / "model.pth"
        torch.save(model.state_dict(), path)

        # Load into a fresh model with same architecture
        model2 = StudentTVAECls(in_dim=100, n_classes=2)
        model2.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))

        x = torch.randn(2, 100)
        model.eval()
        model2.eval()
        with torch.no_grad():
            # Use encode (deterministic) instead of forward (has reparameterization noise)
            mu1, _ = model.encode(x)
            mu2, _ = model2.encode(x)
        assert torch.allclose(mu1, mu2)
