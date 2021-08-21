from collections import defaultdict
import csv
from logging import Logger
import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
from chemprop.utils.reproduce import summary_results_dirs

from .run_training import run_training
from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_data, get_task_names, MoleculeDataset, validate_dataset_type
from chemprop.utils.reproduce import create_logger, makedirs, timeit, make_results_dirs
from chemprop.features import set_extra_atom_fdim


@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args        : TrainArgs,
                   train_func  : Callable[[TrainArgs, MoleculeDataset, Logger], Dict[str, List[float]]]
                   ) -> Tuple[np.ndarray, np.ndarray]:

    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir  = args.save_dir
    args.task_names = get_task_names(path            =  args.data_path,
                                     smiles_columns  =  args.smiles_columns,
                                     target_columns  =  args.target_columns,
                                     ignore_columns  =  args.ignore_columns)

    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')
    debug('Args')
    debug(args)
    makedirs(args.save_dir)
    make_results_dirs(args.save_dir)
    args.save(os.path.join(args.save_dir, 'args.json'))

    debug('Loading data')
    data = get_data(path               =  args.data_path,
                    args               =  args,
                    logger             =  logger,
                    skip_none_targets  =  True)
    validate_dataset_type(data, dataset_type=args.dataset_type)

    args.features_size = data.features_size()
    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = data.atom_descriptors_size()
        args.ffn_hidden_size += args.atom_descriptors_size
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)

    debug(f'Number of tasks = {args.num_tasks}')

    all_scores = defaultdict(list)
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        data.reset_features_and_targets()
        model_scores = train_func(args, data, logger)
        for metric, scores in model_scores.items():
            all_scores[metric].append(scores)

    if args.save_dir:
        summary_results_dirs(save_dir)

    all_scores = dict(all_scores)
    for metric, scores in all_scores.items():
        all_scores[metric] = np.array(scores)

    info(f'{args.num_folds}-fold cross validation')

    for fold_num in range(args.num_folds):
        for metric, scores in all_scores.items():
            info(f'\tSeed {init_seed + fold_num} ==> test {metric} = {np.nanmean(scores[fold_num]):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(args.task_names, scores[fold_num]):
                    info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {metric} = {score:.6f}')

    for metric, scores in all_scores.items():
        avg_scores = np.nanmean(scores, axis=1)  # average score for each model across tasks
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(args.task_names):
                info(f'\tOverall test {task_name} {metric} = '
                     f'{np.nanmean(scores[:, task_num]):.6f} +/- {np.nanstd(scores[:, task_num]):.6f}')

    with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        header = ['Task']
        for metric in args.metrics:
            header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                      [f'Fold {i} {metric}' for i in range(args.num_folds)]
        writer.writerow(header)

        for task_num, task_name in enumerate(args.task_names):
            row = [task_name]
            for metric, scores in all_scores.items():
                task_scores = scores[:, task_num]
                mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
                row += [mean, std] + task_scores.tolist()
            writer.writerow(row)

    avg_scores = np.nanmean(all_scores[args.metric], axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    return mean_score, std_score


def chemprop_train() -> None:
    cross_validate(args=TrainArgs().parse_args(), train_func=run_training)
