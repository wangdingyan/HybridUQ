import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class dnn_nllloss(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_hidden=64,
                 num_layers=0):
        super(dnn_nllloss, self).__init__()
        if num_layers == 0:
            self.layers = nn.Linear(n_inputs, 1)
        else:
            layers = [nn.Linear(n_inputs, n_hidden)]
            for i in range(num_layers):
                layers.extend([torch.nn.GELU(), torch.nn.Linear(n_hidden, n_hidden)])
            layers.extend([nn.Linear(n_hidden, 1)])
            self.layers = nn.Sequential(*layers)

    def forward(self, x_test):
        return torch.exp(self.layers(x_test))

    def fit(self,
            x_train,
            y_train):

        opt = torch.optim.Adam(self.parameters())

        for _ in tqdm(range(1000)):

            preds = self.forward(x_train)
            loss = torch.log(preds).mean() + (y_train.pow(2)/preds).mean()
            print(loss)
            loss.backward()
            opt.step()



