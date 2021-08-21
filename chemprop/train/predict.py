from typing import List

import torch
from torch import Tensor
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel, PB_MoleculeModel


def predict(model                 :  MoleculeModel,
            data_loader           :  MoleculeDataLoader,
            disable_progress_bar  :  bool = False,
            scaler                :  StandardScaler = None,
            keep_dropout_seed     :  int =  None,
            tau                   =  0.5) -> (List[List[float]], List[List[float]]):

    mve = model.mve if model.mve is not None else False
    quantile = True if type(model) == PB_MoleculeModel else False
    assert not ((mve is True) and (quantile is True))

    model.eval()
    if keep_dropout_seed is not None:
        torch.manual_seed(keep_dropout_seed)
        model.ffn_dropout.train()
        for enc in model.encoder.encoder:
            enc.dropout_layer.train()

    preds     =  []
    variance  =  [] if mve else None

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):

        mol_batch, features_batch, atom_descriptors_batch = batch.batch_graph(), batch.features(), batch.atom_descriptors()

        with torch.no_grad():
            if not quantile:
                batch_preds, addition_outputs = model(mol_batch, features_batch, atom_descriptors_batch)
            else:
                fill_tau = torch.zeros(len(mol_batch[0].a_scope), 1).fill_(tau)
                batch_preds, addition_outputs = model(mol_batch, features_batch, atom_descriptors_batch, fill_tau)
            if mve:
                batch_logvar = addition_outputs
                batch_var = torch.exp(batch_logvar)

        batch_preds = batch_preds.data.cpu().numpy()
        batch_var = batch_var.data.cpu().numpy() if mve else None

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
            batch_var   = scaler.inverse_transform_variance(batch_var) if mve else None

        preds.extend(batch_preds.tolist())
        if mve:
            variance.extend(batch_var.tolist())

    if keep_dropout_seed is not None:
        model.ffn_dropout.eval()
        for enc in model.encoder.encoder:
            enc.dropout_layer.eval()

    if mve:
        return preds, variance
    else:
        return preds, None


def extract_readout(model                 :  MoleculeModel,
                    data_loader           :  MoleculeDataLoader,
                    disable_progress_bar  :  bool = False) -> Tensor:

    model.eval()
    readout = []
    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        batch : MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch = batch.batch_graph(), batch.features(), batch.atom_descriptors()

        with torch.no_grad():
            batch_readout = model.featurize(mol_batch, features_batch, atom_descriptors_batch)
        batch_readout = batch_readout.data.cpu()
        readout.append(batch_readout)
    return torch.cat(readout)
