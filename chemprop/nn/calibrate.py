import torch
from torch.utils.data import DataLoader, TensorDataset


def calibrate_distance(distances_val,
                       residues_val,
                       distances_test,
                       epochs=500,
                       device=torch.device('cpu')):

    methods_num = distances_val.shape[1]

    def loss(residues, calibrated_uncertainties):
        return torch.mean(torch.log(calibrated_uncertainties) + (residues*residues) / calibrated_uncertainties)

    class CModel(torch.nn.Module):
        def __init__(self, m_num):
            super(CModel, self).__init__()
            self.coefficients = torch.nn.Parameter(torch.ones(m_num, 1))

        def forward(self, x):
            return torch.matmul(x, torch.exp(self.coefficients)).squeeze()

    model = CModel(methods_num)
    model = model.train()
    residues_val = residues_val.reshape(-1, 1)
    loader = DataLoader(torch.cat([torch.from_numpy(distances_val), torch.from_numpy(residues_val)], dim=-1),
                        shuffle=True,
                        batch_size=64)

    opt = torch.optim.Adam(model.parameters(), lr=1e-1)

    for _ in range(epochs):
        for xi in loader:
            xi = xi.to(device)
            distances = xi[:, :-1].type(torch.float32)
            calibrated_distances = model(distances)
            residues = xi[:, -1]
            opt.zero_grad()
            l = loss(residues, calibrated_distances)
            l.backward()
            opt.step()

    model = model.eval()

    return model(torch.from_numpy(distances_test).type(torch.float32)).detach().numpy()


