"""Fine-tuning API for SPATNIC."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import StudentTVAECls
from .predict import _load_checkpoint


def _kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def _student_t_nll(x, mu, logvar, nu: float):
    var = logvar.exp().clamp_min(1e-8)
    sq = (x - mu) ** 2
    return (
        0.5 * torch.log(var)
        + 0.5 * (nu + 1.0) * torch.log1p(sq / (nu * var))
    ).sum(dim=1)


class _DenseLikeDataset(Dataset):
    def __init__(self, X, y):
        import scipy.sparse as sp

        self.is_sparse = sp.issparse(X)
        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.is_sparse:
            row = self.X.getrow(i).toarray().ravel().astype(np.float32)
        else:
            row = self.X[i].astype(np.float32)
        return torch.from_numpy(row), torch.tensor(int(self.y[i]), dtype=torch.long)


def finetune(
    adata,
    label_key: str = "cell_type",
    label_map: Optional[dict] = None,
    model: str = "tumor_normal",
    save_path: str = "spatnic_finetuned.pt",
    layer_key: Optional[str] = None,
    epochs_head: int = 5,
    epochs_joint: int = 20,
    lr_head: float = 1e-3,
    lr_joint: float = 5e-4,
    weight_decay: float = 1e-2,
    beta_kl: float = 0.1,
    lambda_cls: float = 2.0,
    batch_size: int = 1024,
    num_workers: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
    device: Optional[str] = None,
):
    """Fine-tune a pre-trained SPATNIC model on new labeled data.

    Parameters
    ----------
    adata : AnnData
        Input data (log-normalized) with labels in ``adata.obs[label_key]``.
    label_key : str
        Column in ``adata.obs`` containing cell type labels.
    label_map : dict or None
        Mapping from label strings to integers (e.g.
        ``{"Epithelial": 0, "Malignant": 1}``). If None, inferred
        automatically (alphabetical order, first=0).
    model : str
        Base model name or checkpoint path.
    save_path : str
        Where to save the fine-tuned checkpoint.
    layer_key : str or None
        AnnData layer to use.
    epochs_head : int
        Epochs for head-only training (encoder frozen).
    epochs_joint : int
        Epochs for joint training.
    lr_head, lr_joint : float
        Learning rates for each stage.
    weight_decay : float
        AdamW weight decay.
    beta_kl, lambda_cls : float
        Loss weights.
    batch_size, num_workers : int
        DataLoader parameters.
    test_size : float
        Fraction of data for validation.
    random_state : int
        Random seed for train/test split.
    device : str or None
        Device string. Auto-detected if None.

    Returns
    -------
    dict
        Training history with validation accuracy and loss.
    """
    from sklearn.model_selection import train_test_split

    from .preprocessing import align_genes, standardize_csr

    # Load base checkpoint
    ckpt = _load_checkpoint(model)
    train_genes = np.array(ckpt["gene_names"])
    mu_g = ckpt["mu_g"].astype(np.float32)
    std_g = ckpt["std_g"].astype(np.float32)
    std_safe = std_g.copy()
    std_safe[std_safe == 0] = 1.0
    model_config = ckpt.get("model_config", {})

    # Align and standardize
    X_aligned, n_matched = align_genes(adata, train_genes, layer_key=layer_key)
    print(f"[SPATNIC] Matched genes: {n_matched}/{len(train_genes)}")
    X_std = standardize_csr(X_aligned, mu_g, std_safe)

    # Labels
    labels_raw = adata.obs[label_key].values
    if label_map is None:
        unique_labels = sorted(set(str(x) for x in labels_raw))
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        print(f"[SPATNIC] Auto label map: {label_map}")
    y = np.array([label_map[str(x)] for x in labels_raw])
    n_classes = len(set(y))

    # Train/test split
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state
    )
    train_ds = _DenseLikeDataset(X_std[idx_tr], y[idx_tr])
    val_ds = _DenseLikeDataset(X_std[idx_te], y[idx_te])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Build model
    net = StudentTVAECls(
        in_dim=len(train_genes),
        n_classes=n_classes,
        latent_dim=model_config.get("latent_dim", 32),
        hidden=model_config.get("hidden", 512),
        dropout=model_config.get("dropout", 0.2),
        denoise_p=model_config.get("denoise_p", 0.2),
        heteroscedastic=model_config.get("heteroscedastic", True),
        stu_nu=model_config.get("stu_nu", 5.0),
        learnable_nu=model_config.get("learnable_nu", False),
    )

    # Load pre-trained weights (skip classifier head if n_classes differs)
    state_dict = ckpt["state_dict"]
    if n_classes != model_config.get("n_classes", 2):
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("cls.")
        }
        net.load_state_dict(state_dict, strict=False)
    else:
        net.load_state_dict(state_dict)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    history = {"stage": [], "epoch": [], "val_acc": [], "val_loss": []}

    def _run_epoch(loader, optimizer=None, joint=True):
        training = optimizer is not None
        net.train(training)
        n, loss_sum, acc_sum = 0, 0.0, 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mu_z, logvar_z, dec_out, logvar_x, logits = net(xb)

            if net.hetero:
                nu = net.current_nu()
                rec = _student_t_nll(xb, dec_out, logvar_x, float(nu)).mean()
            else:
                rec = F.mse_loss(dec_out, xb)
            kl = _kl_divergence(mu_z, logvar_z).mean()
            cls = F.cross_entropy(logits, yb)

            if joint:
                loss = rec + beta_kl * kl + lambda_cls * cls
            else:
                loss = lambda_cls * cls

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()

            bs = yb.size(0)
            n += bs
            loss_sum += loss.item() * bs
            acc_sum += (logits.argmax(1) == yb).float().sum().item()

        return loss_sum / n, acc_sum / n

    # Stage 1: head-only
    print("[SPATNIC] Stage 1: head-only fine-tuning")
    for p in net.enc.parameters():
        p.requires_grad = False
    for p in net.mu_z.parameters():
        p.requires_grad = False
    for p in net.logvar_z.parameters():
        p.requires_grad = False
    for p in net.dec.parameters():
        p.requires_grad = False
    if net.hetero:
        for p in net.mu_x.parameters():
            p.requires_grad = False
        for p in net.logvar_x.parameters():
            p.requires_grad = False

    opt_head = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr_head, weight_decay=weight_decay,
    )
    best_acc, best_state = -1.0, None
    for ep in range(1, epochs_head + 1):
        _run_epoch(train_loader, opt_head, joint=False)
        val_loss, val_acc = _run_epoch(val_loader, joint=False)
        print(f"  [head {ep:02d}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        history["stage"].append("head")
        history["epoch"].append(ep)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

    # Unfreeze all
    for p in net.parameters():
        p.requires_grad = True

    # Stage 2: joint
    print("[SPATNIC] Stage 2: joint fine-tuning")
    opt_joint = torch.optim.AdamW(
        net.parameters(), lr=lr_joint, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_joint, T_max=epochs_joint
    )
    for ep in range(1, epochs_joint + 1):
        _run_epoch(train_loader, opt_joint, joint=True)
        val_loss, val_acc = _run_epoch(val_loader, joint=True)
        scheduler.step()
        print(f"  [joint {ep:02d}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        history["stage"].append("joint")
        history["epoch"].append(ep)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

    if best_state is not None:
        net.load_state_dict(best_state)

    # Inverse label map for saving
    inv_label_map = {v: k for k, v in label_map.items()}

    # Save checkpoint
    new_config = dict(model_config)
    new_config["n_classes"] = n_classes
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "gene_names": list(train_genes),
            "mu_g": mu_g,
            "std_g": std_g,
            "model_config": new_config,
            "label_map": inv_label_map,
        },
        save_path,
    )
    print(f"[SPATNIC] Best val_acc={best_acc:.4f}, saved to {save_path}")

    return history
