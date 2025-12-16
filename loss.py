import torch
from ZINB import zinb_log_prob


def kl_normal(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def elbo_loss(x, mu, pi, theta, mu_z, logvar_z, mu_l, logvar_l):
    # zinb_log_prob expects arguments in the order: x, mu, pi, theta
    recon = zinb_log_prob(x, mu, pi, theta).sum(dim=1)
    kl_z = kl_normal(mu_z, logvar_z)
    kl_l = kl_normal(mu_l, logvar_l)
    loss = -(recon - kl_z - kl_l).mean()
    return loss
