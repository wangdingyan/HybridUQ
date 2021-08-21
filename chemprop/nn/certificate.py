# https://github.com/facebookresearch/SingleModelUncertainty/blob/master/toy/toy_example.py

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


def build_certificates(x,
                       k=100,
                       batch_size=128,
                       epochs=500,
                       device=torch.device('cpu')):
    c = torch.nn.Linear(x.size(1), k)
    c.to(device)
    loader = DataLoader(TensorDataset(x),
                        shuffle=True,
                        batch_size=batch_size)
    opt = torch.optim.Adam(c.parameters())

    for _ in tqdm(range(epochs)):
        for xi in loader:
            xi = xi[0].to(device)
            opt.zero_grad()
            error = c(xi).pow(2).mean()
            penalty = (c.weight @ c.weight.t() - torch.eye(k).to(device)).pow(2).mean()
            (error + penalty).backward()
            opt.step()

    return c
