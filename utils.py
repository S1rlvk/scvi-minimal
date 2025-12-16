import torch


def reparameterize(mu, logvar):
    """
    Reparameterization trick.
    mu: (B, D)
    logvar: (B, D)
    returns: (B, D)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
