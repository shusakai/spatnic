"""Prediction API for SPATNIC."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import StudentTVAECls
from .preprocessing import align_genes, standardize_csr

_WEIGHTS_DIR_PKG = Path(__file__).parent / "weights"
_WEIGHTS_DIR_USER = Path.home() / ".spatnic" / "weights"

# Built-in model registry: name -> checkpoint filename
MODELS = {
    "tumor_normal": "tumor_normal.pt",
    "tumor_other": "tumor_other.pt",
}

# Label maps per model
LABEL_MAPS = {
    "tumor_normal": {0: "Normal", 1: "Tumor"},
    "tumor_other": {0: "Other", 1: "Tumor"},
}


class _CSRRowDataset(Dataset):
    def __init__(self, X_csr):
        self.X = X_csr
        self.n = X_csr.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        row = self.X[i].toarray().ravel().astype(np.float32)
        return torch.from_numpy(row), torch.tensor(0, dtype=torch.long)


def _load_checkpoint(model_name_or_path: str) -> dict:
    """Load a checkpoint from a built-in model name or a file path."""
    path = Path(model_name_or_path)
    if not path.exists():
        if model_name_or_path in MODELS:
            fname = MODELS[model_name_or_path]
            # Search: ~/.spatnic/weights/ -> package weights/
            for d in (_WEIGHTS_DIR_USER, _WEIGHTS_DIR_PKG):
                candidate = d / fname
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"Model '{model_name_or_path}' not found in "
                    f"{_WEIGHTS_DIR_USER} or {_WEIGHTS_DIR_PKG}. "
                    f"Run `spatnic.export_checkpoint(...)` to create it."
                )
        else:
            raise FileNotFoundError(
                f"Model '{model_name_or_path}' not found. "
                f"Available built-in models: {list(MODELS.keys())}"
            )
    return torch.load(path, map_location="cpu", weights_only=False)


def predict(
    adata,
    model: str = "tumor_normal",
    threshold: float = 0.5,
    batch_size: int = 2048,
    num_workers: int = 2,
    key_added: str = "spatnic_pred",
    score_key_added: str = "spatnic_score",
    layer_key: Optional[str] = None,
    return_latent: bool = False,
    device: Optional[str] = None,
):
    """Predict cell annotations using a pre-trained SPATNIC model.

    Parameters
    ----------
    adata : AnnData
        Input data. Must be log-normalized (``sc.pp.normalize_total`` +
        ``sc.pp.log1p``).
    model : str
        Built-in model name (``"tumor_normal"`` or ``"tumor_other"``) or
        path to a custom checkpoint (``.pt`` file).
    threshold : float
        Classification threshold for the positive class.
    batch_size : int
        Batch size for inference.
    num_workers : int
        DataLoader workers.
    key_added : str
        Key in ``adata.obs`` to store predicted labels.
    score_key_added : str
        Key in ``adata.obs`` to store prediction scores.
    layer_key : str or None
        AnnData layer to use. If None, uses ``adata.X``.
    return_latent : bool
        If True, also return the latent representation (mu_z).
    device : str or None
        Device string (e.g. ``"cuda"``). Auto-detected if None.

    Returns
    -------
    adata : AnnData
        The input AnnData with ``adata.obs[key_added]`` and
        ``adata.obs[score_key_added]`` added.
    latent : np.ndarray, optional
        Latent representation, only if ``return_latent=True``.
    """
    import scanpy as sc  # noqa: F401 – ensure scanpy is available

    # Load checkpoint
    ckpt = _load_checkpoint(model)
    train_genes = np.array(ckpt["gene_names"])
    mu_g = ckpt["mu_g"].astype(np.float32)
    std_g = ckpt["std_g"].astype(np.float32)
    std_safe = std_g.copy()
    std_safe[std_safe == 0] = 1.0
    state_dict = ckpt["state_dict"]
    model_config = ckpt.get("model_config", {})

    # Determine label map
    label_map = ckpt.get("label_map", None)
    if label_map is None:
        # Try to infer from model name
        model_name = model if model in MODELS else "tumor_normal"
        label_map = LABEL_MAPS.get(model_name, {0: "Class_0", 1: "Class_1"})

    # Build model
    net = StudentTVAECls(
        in_dim=len(train_genes),
        n_classes=model_config.get("n_classes", 2),
        latent_dim=model_config.get("latent_dim", 32),
        hidden=model_config.get("hidden", 512),
        dropout=model_config.get("dropout", 0.2),
        denoise_p=0.0,
        heteroscedastic=model_config.get("heteroscedastic", True),
        stu_nu=model_config.get("stu_nu", 5.0),
        learnable_nu=model_config.get("learnable_nu", False),
    )
    net.load_state_dict(state_dict)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    net.eval()

    # Align genes
    X_aligned, n_matched = align_genes(adata, train_genes, layer_key=layer_key)
    total_genes = len(train_genes)
    match_ratio = n_matched / total_genes
    print(f"[SPATNIC] Matched genes: {n_matched}/{total_genes} ({match_ratio:.1%})")
    if match_ratio < 0.5:
        warnings.warn(
            f"Only {match_ratio:.1%} of training genes matched. "
            "Predictions may be unreliable. Ensure the data is log-normalized "
            "and gene names match.",
            stacklevel=2,
        )

    # Standardize
    X_std = standardize_csr(X_aligned, mu_g, std_safe)

    # DataLoader
    loader = DataLoader(
        _CSRRowDataset(X_std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    # Inference
    probs_list, mu_z_list = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            mu_z, _, _, _, logits = net(xb)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_list.append(p)
            if return_latent:
                mu_z_list.append(mu_z.cpu().numpy())

    scores = np.concatenate(probs_list)
    preds = (scores >= threshold).astype(int)
    pred_labels = np.array([label_map[p] for p in preds])

    adata.obs[key_added] = pred_labels
    adata.obs[score_key_added] = scores

    print(
        f"[SPATNIC] Predictions: "
        + ", ".join(
            f"{label_map[k]}={int((preds == k).sum())}"
            for k in sorted(label_map)
        )
    )

    if return_latent:
        latent = np.concatenate(mu_z_list, axis=0)
        return adata, latent
    return adata
