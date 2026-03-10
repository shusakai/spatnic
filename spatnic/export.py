"""Utilities to export model checkpoints from notebook artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


_WEIGHTS_DIR_USER = Path.home() / ".spatnic" / "weights"


def export_checkpoint(
    state_dict_path: str,
    gene_names: Sequence[str],
    mu_g: np.ndarray,
    std_g: np.ndarray,
    save_path: Optional[str] = None,
    model_name: Optional[str] = None,
    label_map: Optional[dict] = None,
    n_classes: int = 2,
    latent_dim: int = 32,
    hidden: int = 512,
    dropout: float = 0.2,
    denoise_p: float = 0.2,
    heteroscedastic: bool = True,
    stu_nu: float = 5.0,
    learnable_nu: bool = False,
):
    """Create a SPATNIC checkpoint from individual notebook artifacts.

    Parameters
    ----------
    state_dict_path : str
        Path to the saved ``model.state_dict()`` ``.pth`` file.
    gene_names : list of str
        Ordered gene names used during training (e.g.
        ``adata_hvg.var_names``).
    mu_g : np.ndarray
        Per-gene mean from the training set (shape ``(n_genes,)``).
    std_g : np.ndarray
        Per-gene std from the training set (shape ``(n_genes,)``).
    save_path : str
        Output path for the ``.pt`` checkpoint.
    label_map : dict or None
        Integer-to-string label map (e.g. ``{0: "Normal", 1: "Tumor"}``).
    n_classes, latent_dim, hidden, dropout, denoise_p,
    heteroscedastic, stu_nu, learnable_nu : various
        Model hyperparameters matching the trained model.

    Examples
    --------
    In the training notebook, after training::

        from spatnic import export_checkpoint
        export_checkpoint(
            state_dict_path="student_t_vae_cls_weights_primary.pth",
            gene_names=list(adata_hvg.var_names),
            mu_g=mu_g,
            std_g=std_g,
            save_path="tumor_normal.pt",
            label_map={0: "Normal", 1: "Tumor"},
        )
    """
    # Determine save path
    if save_path is None and model_name is None:
        raise ValueError("Either save_path or model_name must be specified.")
    if save_path is None:
        _WEIGHTS_DIR_USER.mkdir(parents=True, exist_ok=True)
        save_path = str(_WEIGHTS_DIR_USER / f"{model_name}.pt")

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)

    if label_map is None:
        label_map = {i: f"Class_{i}" for i in range(n_classes)}

    ckpt = {
        "state_dict": sd,
        "gene_names": list(gene_names),
        "mu_g": np.asarray(mu_g, dtype=np.float32),
        "std_g": np.asarray(std_g, dtype=np.float32),
        "label_map": label_map,
        "model_config": {
            "n_classes": n_classes,
            "latent_dim": latent_dim,
            "hidden": hidden,
            "dropout": dropout,
            "denoise_p": denoise_p,
            "heteroscedastic": heteroscedastic,
            "stu_nu": stu_nu,
            "learnable_nu": learnable_nu,
        },
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"[SPATNIC] Checkpoint saved to {save_path} ({Path(save_path).stat().st_size / 1e6:.1f} MB)")
