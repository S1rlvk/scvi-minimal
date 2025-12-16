import torch


def load_data(top_genes: int = 1000) -> torch.Tensor:
    """
    Synthetic count data to validate scVI training.
    """
    torch.manual_seed(0)

    n_cells = 256
    rate = 1.5  # average count

    x = torch.poisson(rate * torch.ones(n_cells, top_genes))
    return x.float()
