import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


#     Trains a model for an epoch.
def train(model        :  nn.Module,
          data_loader  :  MoleculeDataLoader,
          loss_func    :  Callable,
          optimizer    :  Optimizer,
          scheduler    :  _LRScheduler,
          args         :  TrainArgs,
          n_iter       :  int = 0,
          logger       :  logging.Logger = None,
          writer       :  SummaryWriter = None) -> int:

    debug = logger.debug if logger is not None else print
    
    model.train()
    loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):

        mol_batch               =  batch.batch_graph()
        features_batch          =  batch.features()
        target_batch            =  batch.targets()
        atom_descriptors_batch  =  batch.atom_descriptors()
        mask                    =  torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets                 =  torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        model.zero_grad()
        if args.quantileregression != 'None':
            if args.quantileregression == 'Pinball':
                tau = torch.rand(targets.shape[0], 1)
            elif args.quantileregression == 'Interval':
                tau = torch.rand(targets.shape[0], 1) / 2
            elif args.quantileregression in ['Calibration', 'ScaledCalibration', 'CalibrationSharpness']:
                tau = torch.rand(1, 1).repeat([targets.shape[0], 1])
            else:
                raise NotImplementedError('Unrecognized Quantile Regression Method')
            preds, addition_outputs = model(mol_batch, features_batch, atom_descriptors_batch, tau)
        else:
            preds, addition_outputs = model(mol_batch, features_batch, atom_descriptors_batch)

        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        elif args.mve or args.quantileregression != 'None':
            loss = loss_func(targets, preds, addition_outputs) * class_weights * mask
        else:
            assert addition_outputs is None
            loss = loss_func(preds, targets) * class_weights * mask

        loss = loss.sum() / mask.sum()
        loss_sum += loss.item()
        iter_count += 1
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs       =  scheduler.get_lr()
            pnorm     =  compute_pnorm(model)
            gnorm     =  compute_gnorm(model)
            loss_avg  =  loss_sum / iter_count
            loss_sum  =  iter_count = 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
