import torch
from model import Encoder, Decoder
from data import load_data
from utils import reparameterize
from loss import elbo_loss

latent_dim = 10
epochs = 50
lr = 1e-3
batch_size = 32

x = load_data(top_genes=1000)
n_cells, n_genes = x.shape

encoder = Encoder(n_genes, latent_dim)
decoder = Decoder(n_genes, latent_dim)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=lr
)

for epoch in range(epochs):
    perm = torch.randperm(n_cells)
    total_loss = 0.0

    for i in range(0, n_cells, batch_size):
        idx = perm[i:i+batch_size]
        xb = x[idx]
        mu_z, logvar_z, mu_l, logvar_l = encoder(xb)
        z = reparameterize(mu_z, logvar_z)
        l = reparameterize(mu_l, logvar_l)

        mu, pi, theta = decoder(z, l)
        loss = elbo_loss(xb, mu, pi, theta, mu_z, logvar_z, mu_l, logvar_l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_cells}")
