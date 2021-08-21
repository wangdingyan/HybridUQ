import csv
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

import torch
from .predict import predict
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils.reproduce import load_args, load_checkpoint, load_scalers, makedirs, timeit


@timeit()
def make_predictions(args: PredictArgs, smiles: List[List[str]] = None) -> (List[List[Optional[float]]], List[List[Optional[float]]]):

    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    if ((train_args.features_path is not None or train_args.features_generator is not None)
            and args.features_path is None
            and args.features_generator is None):
        raise ValueError('Features were used during training so they must be specified again during prediction '
                         'using the same type of features as before (with either --features_generator or '
                         '--features_path and using --no_features_scaling if applicable).')

    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(smiles               =  smiles,
                                         skip_invalid_smiles  =  False,
                                         features_generator   =  args.features_generator)
    else:
        full_data = get_data(path                 =  args.test_path,
                             target_columns       =  [],
                             ignore_columns       =  [],
                             skip_invalid_smiles  =  False,
                             args                 =  args,
                             store_row            =  True)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    sum_preds = []
    sum_mve = []
    if args.mc_dropout is not None:
        sum_mc_preds = []
        sum_mc_preds_vars = []
        sum_mc_mve = []

    # Create data loader
    test_data_loader = MoleculeDataLoader(dataset     =  test_data,
                                          batch_size  =  args.batch_size,
                                          num_workers =  args.num_workers)

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):

        # Load model and scalers
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler = load_scalers(checkpoint_path)

        # Normalize features
        if args.features_scaling:
            test_data.reset_features_and_targets()
            test_data.normalize_features(features_scaler)

        # Make predictions
        model_preds, model_vars = predict(model         =  model,
                                          data_loader   =  test_data_loader,
                                          scaler        =  scaler)
        sum_preds.append(torch.tensor(model_preds))

        if model_vars is not None:
            sum_mve.append(torch.tensor(model_vars))

        if args.mc_dropout is not None:
            model_mc_preds = []
            for sample_num in range(args.mc_dropout):

                sample_mc_preds, sample_mc_vars = predict(
                    model=model,
                    data_loader=test_data_loader,
                    scaler=scaler,
                    keep_dropout=True
                )

                model_mc_preds.append(torch.tensor(sample_mc_preds))
                sum_mc_mve.append(torch.tensor(sample_mc_vars))
                sum_mc_preds.append(torch.tensor(sample_mc_preds))
            sum_mc_preds_vars.append(torch.var(torch.stack(model_mc_preds), 0))

    # Ensemble predictions
    avg_preds = torch.mean(torch.stack(sum_preds), 0).tolist()
    avg_preds_vars = torch.var(torch.stack(sum_preds), 0).tolist() if len(sum_preds) > 1 else torch.zeros(sum_preds[0].shape).tolist()
    if sum_mve:
        avg_mve = torch.mean(torch.stack(sum_mve), 0).tolist()
    if args.mc_dropout:
        avg_mc_preds = torch.mean(torch.stack(sum_mc_preds), 0).tolist()
        avg_mc_preds_vars = torch.mean(torch.stack(sum_mc_preds_vars), 0).tolist()
        avg_mc_mve = torch.mean(torch.stack(sum_mc_mve), 0).tolist()

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(avg_preds)
    makedirs(args.preds_path, isfile=True)

    # Get prediction column names
    if args.dataset_type == 'multiclass':
        task_names = [f'{name}_class_{i}' for name in task_names for i in range(args.multiclass_num_classes)]
    else:
        task_names = task_names

    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = avg_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
        preds_vars = avg_preds_vars[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)

        if len(sum_mve) != 0:
            mve = avg_mve[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
        else:
            mve = ['NotPredicted'] * len(task_names)

        if args.mc_dropout is not None:
            mc_preds = avg_mc_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
            mc_preds_vars = avg_mc_preds_vars[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
            mc_mve = avg_mc_mve[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
        else:
            mc_preds = ['NotPredicted'] * len(task_names)
            mc_preds_vars = ['NotPredicted'] * len(task_names)
            mc_mve = ['NotPredicted'] * len(task_names)

        for pred_name, pred, pred_var, m in zip(task_names, preds, preds_vars, mve):
            datapoint.row[pred_name] = pred
            datapoint.row[pred_name+'_var'] = pred_var
            datapoint.row[pred_name+'_mve'] = m

        for pred_name, mc_pred, mc_pred_var, mc_m in zip(task_names, mc_preds, mc_preds_vars, mc_mve):
            datapoint.row[pred_name+'_mc'] = mc_pred
            datapoint.row[pred_name+'_mc_var'] = mc_pred_var
            datapoint.row[pred_name+'_mc_mve'] = mc_m

    # Save
    with open(args.preds_path, 'w', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
        writer.writeheader()

        for datapoint in full_data:
            writer.writerow(datapoint.row)

    return avg_preds


def chemprop_predict() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    make_predictions(args=PredictArgs().parse_args())
