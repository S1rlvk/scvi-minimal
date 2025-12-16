import torch
import torch.nn as nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = 10, hidden_dim: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(n_genes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # z heads
        self.mu_z = nn.Linear(hidden_dim, latent_dim)
        self.logvar_z = nn.Linear(hidden_dim, latent_dim)

        # l heads
        self.mu_l = nn.Linear(hidden_dim, 1)
        self.logvar_l = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (batch, genes)
        returns:
            mu_z, logvar_z : (batch, latent_dim)
            mu_l, logvar_l : (batch, 1)
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        mu_z = self.mu_z(h)
        logvar_z = self.logvar_z(h)

        mu_l = self.mu_l(h)
        logvar_l = self.logvar_l(h)

        return mu_z, logvar_z, mu_l, logvar_l


class Decoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = 10, hidden_dim: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, n_genes)
        self.fc_pi = nn.Linear(hidden_dim, n_genes)

        # global dispersion parameter (per gene)
        self.theta = nn.Parameter(torch.randn(n_genes))

    def forward(self, z, l):
        """
        z: (B, latent_dim)
        l: (B, 1)
        returns:
            mu: (B, G)
            pi: (B, G)
            theta: (G,)
        """
        h = torch.cat([z, l], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        mu = F.softplus(self.fc_mu(h))

        pi = torch.sigmoid(self.fc_pi(h))
        theta = F.softplus(self.theta)

        return mu, pi, theta


if __name__ == "__main__":
    from data import load_data
    from utils import reparameterize

    x = load_data(top_genes=1000)
    encoder = Encoder(n_genes=x.shape[1])
    decoder = Decoder(n_genes=x.shape[1])

    batch = x[:16]
    mu_z, logvar_z, mu_l, logvar_l = encoder(batch)

    z = reparameterize(mu_z, logvar_z)
    l = reparameterize(mu_l, logvar_l)

    mu, pi, theta = decoder(z, l)

    print(mu.shape, pi.shape, theta.shape)
