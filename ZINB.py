import torch
import torch.nn as nn

def zinb_log_prob(x, mu, pi, theta, eps=1e-8):

    theta = theta.unsqueeze(0)

    log_nb = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    )

    zero_case = torch.log(
        pi + (1.0 - pi) * torch.exp(log_nb) + eps

    )

    nonzero_case = torch.log(1.0 - pi + eps) + log_nb

    return torch.where(x < 1e-8, zero_case, nonzero_case)


if __name__ == "__main__":
    B, G = 4, 5
    x = torch.tensor([[0, 1, 2, 0, 3]]).repeat(B, 1).float()
    mu = torch.rand(B, G) + 0.1
    theta = torch.rand(G) + 0.1
    pi = torch.sigmoid(torch.randn(B, G))

    lp = ZINB_log_prob(x, mu, pi, theta)
    print(lp.shape)
    print(torch.isfinite(lp).all())
