"""Trains a chemprop model on a dataset."""
import rdkit
from rdkit import Chem
from chemprop.train import chemprop_train


if __name__ == '__main__':
    chemprop_train()
