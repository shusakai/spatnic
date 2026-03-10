"""Student-t VAE + Classifier model definition."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentTVAECls(nn.Module):
    """Student-t VAE with heteroscedastic decoder and classification head.

    Parameters
    ----------
    in_dim : int
        Number of input genes.
    n_classes : int
        Number of output classes.
    latent_dim : int
        Latent space dimensionality.
    hidden : int
        Hidden layer width.
    dropout : float
        Dropout rate.
    denoise_p : float
        Feature masking probability during training.
    heteroscedastic : bool
        If True, decoder predicts per-gene mean and log-variance.
    stu_nu : float
        Degrees of freedom for the Student-t reconstruction loss.
    learnable_nu : bool
        If True, nu is a learnable parameter.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = 2,
        latent_dim: int = 32,
        hidden: int = 512,
        dropout: float = 0.2,
        denoise_p: float = 0.2,
        heteroscedastic: bool = True,
        stu_nu: float = 5.0,
        learnable_nu: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.denoise_p = denoise_p
        self.hetero = heteroscedastic

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mu_z = nn.Linear(hidden, latent_dim)
        self.logvar_z = nn.Linear(hidden, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        if self.hetero:
            self.mu_x = nn.Linear(hidden, in_dim)
            self.logvar_x = nn.Linear(hidden, in_dim)
        else:
            self.recon = nn.Linear(hidden, in_dim)

        # Classifier head
        self.cls = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        # Student-t degrees of freedom
        if learnable_nu:
            init = float(stu_nu)
            init_log = np.log(np.exp(init - 2.0) - 1.0)
            self.log_nu = nn.Parameter(
                torch.tensor(init_log, dtype=torch.float32)
            )
            self.learnable_nu = True
        else:
            self.register_buffer(
                "nu_buf", torch.tensor(float(stu_nu), dtype=torch.float32)
            )
            self.learnable_nu = False

    def current_nu(self) -> torch.Tensor:
        if self.learnable_nu:
            return F.softplus(self.log_nu) + 2.0
        return self.nu_buf

    def encode(self, x):
        h = self.enc(x)
        return self.mu_z(h), self.logvar_z(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.dec(z)
        if self.hetero:
            mu = self.mu_x(h)
            logvar = self.logvar_x(h).clamp(min=-5.0, max=3.0)
            return mu, logvar
        xhat = self.recon(h)
        return xhat, None

    def forward(self, x):
        if self.training and self.denoise_p > 0:
            mask = (torch.rand_like(x) > self.denoise_p).float()
            x = x * mask

        mu_z, logvar_z = self.encode(x)
        z = self.reparam(mu_z, logvar_z)
        dec_out, logvar_x = self.decode(z)
        logits = self.cls(z)
        return mu_z, logvar_z, dec_out, logvar_x, logits
