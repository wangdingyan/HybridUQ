from typing import List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):

    def __init__(self,
                 args: TrainArgs,
                 featurizer: bool = False):

        super(MoleculeModel, self).__init__()

        self.classification =  args.dataset_type == 'classification'
        self.multiclass     =  args.dataset_type == 'multiclass'
        self.featurizer     =  featurizer
        self.ffn_dropout    =  nn.Dropout(args.dropout)
        self.mve            =  args.mve
        self.device         =  args.device

        self.output_size = args.num_tasks

        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        self.encoder = MPN(args)

    def create_ffn(self, args: TrainArgs) -> None:
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                self.ffn_dropout,
            ]
            last_linear_dim = first_linear_dim
        else:
            ffn = [
                self.ffn_dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    self.ffn_dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                self.ffn_dropout,
            ])
            last_linear_dim = args.ffn_hidden_size

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

        if self.mve:
            self.output_layer = nn.Linear(last_linear_dim, self.output_size)
            self.logvar_layer = nn.Linear(last_linear_dim, self.output_size)
        else:
            self.output_layer = nn.Linear(last_linear_dim, self.output_size)

    def featurize(self,
                  batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[BatchMolGraph]]],
                  features_batch: List[np.ndarray] = None,
                  atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:

        return self.ffn(self.encoder(batch, features_batch, atom_descriptors_batch))

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None) -> (torch.FloatTensor, torch.FloatTensor):

        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch)

        _output = self.ffn(self.encoder(batch, features_batch, atom_descriptors_batch))
        if self.mve:
            output = self.output_layer(_output)
            logvar = self.logvar_layer(_output)

            # mve can be true only in the case of regression task
            return output, logvar
        else:
            output = self.output_layer(_output)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output, None
