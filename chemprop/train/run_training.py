from .evaluate import evaluate, evaluate_predictions
from .predict import predict, extract_readout
from .train import train
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.models import MoleculeModel, PB_MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils.reproduce import load_checkpoint, makedirs, save_checkpoint, save_smiles_splits
from chemprop.nn.loss import get_loss_func
from chemprop.nn.optimize import build_optimizer, build_lr_scheduler
from scipy.spatial.distance import *
from chemprop.conformal.conformal import *
from chemprop.plot.confidence import *
from chemprop.plot.conformal import *
from chemprop.utils.distance import prior_unc
from chemprop.utils.uclass import uncertainties
from logging import Logger
from typing import Dict, List
from tensorboardX import SummaryWriter
import torch
from torch.optim.lr_scheduler import ExponentialLR
torch.set_num_threads(1)
import matplotlib

matplotlib.use('agg')


def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None) -> Dict[str, List[float]]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    torch.manual_seed(args.pytorch_seed)

    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path           =  args.separate_test_path,
                             args           =  args,
                             features_path  =  args.separate_test_features_path,
                             logger         =  logger)
    if args.separate_val_path:
        val_data = get_data(path           =  args.separate_val_path,
                            args           =  args,
                            features_path  =  args.separate_val_features_path,
                            logger         =  logger)
    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data        =  data,
                                              split_type  =  args.split_type,
                                              sizes       =  (0.8, 0.0, 0.2),
                                              seed        =  args.seed,
                                              num_folds   =  args.num_folds,
                                              args        =  args,
                                              logger      =  logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data        =  data,
                                             split_type  =  args.split_type,
                                             sizes       =  (0.8, 0.2, 0.0),
                                             seed        =  args.seed,
                                             num_folds   =  args.num_folds,
                                             args        =  args,
                                             logger      =  logger)
    else:
        train_data, val_data, test_data = split_data(data        =  data,
                                                     split_type  =  args.split_type,
                                                     sizes       =  args.split_sizes,
                                                     seed        =  args.seed,
                                                     num_folds   =  args.num_folds,
                                                     args        =  args,
                                                     logger      =  logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        save_smiles_splits(data_path       =  args.data_path,
                           save_dir        =  args.save_dir,
                           train_data      =  train_data,
                           val_data        =  val_data,
                           test_data       =  test_data,
                           smiles_columns  =  args.smiles_columns)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        scaler = train_data.normalize_targets()
    else:
        scaler = None

    loss_func = get_loss_func(args)

    test_smiles = test_data.smiles()
    test_targets = test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks,
                               args.multiclass_num_classes)) if args.dataset_type == 'multiclass' else np.zeros(
        (len(test_smiles), args.num_tasks))

    list_test_preds = [] if args.ensemble_size > 1 else None
    test_uncertainties = uncertainties()
    test_conformal_dict = {}

    val_smiles = val_data.smiles()
    val_targets = val_data.targets()
    sum_val_preds = np.zeros((len(val_smiles), args.num_tasks,
                              args.multiclass_num_classes)) if args.dataset_type == 'multiclass' else np.zeros(
        (len(val_smiles), args.num_tasks))
    list_val_preds = [] if args.ensemble_size > 1 else None
    val_uncertainties = uncertainties()

    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    train_data_loader = MoleculeDataLoader(dataset        =  train_data,
                                           batch_size     =  args.batch_size,
                                           num_workers    =  num_workers,
                                           class_balance  =  args.class_balance,
                                           shuffle        =  True,
                                           seed           =  args.seed)

    val_data_loader = MoleculeDataLoader(dataset      =  val_data,
                                         batch_size   =  args.batch_size,
                                         num_workers  =  num_workers)

    test_data_loader = MoleculeDataLoader(dataset      =  test_data,
                                          batch_size   =  args.batch_size,
                                          num_workers  =  num_workers)

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            if args.quantileregression != 'None':
                model = PB_MoleculeModel(args)
            else:
                model = MoleculeModel(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME),
                        model,
                        scaler,
                        features_scaler,
                        args)

        optimizer = build_optimizer(model, args)
        scheduler = build_lr_scheduler(optimizer, args)

        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        import time
        starttime = time.time()

        for epoch in range(999999):
            debug(f'Epoch {epoch}')

            n_iter = train(model        =  model,
                           data_loader  =  train_data_loader,
                           loss_func    =  loss_func,
                           optimizer    =  optimizer,
                           scheduler    =  scheduler,
                           args         =  args,
                           n_iter       =  n_iter,
                           logger       =  logger,
                           writer       =  writer)

            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            val_scores = evaluate(model         =  model,
                                  data_loader   =  val_data_loader,
                                  num_tasks     =  args.num_tasks,
                                  metrics       =  args.metrics,
                                  dataset_type  =  args.dataset_type,
                                  scaler        =  scaler,
                                  logger        =  logger)

            for metric, scores in val_scores.items():
                avg_val_score = np.nanmean(scores)
                debug(f'Validation {metric} = {avg_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)

                if args.show_individual_scores:
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)

            avg_val_score = np.nanmean(val_scores[args.metric])
            if args.minimize_score and avg_val_score < best_score or not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME),
                                model,
                                scaler,
                                features_scaler,
                                args)
                debug(f"REFRESH BEST EPOCH RECORD")
            else:
                debug(f"{epoch - best_epoch} EPOCHS NOT REFRESH")
                debug(f"NOW / BEST METRIC: {avg_val_score} / {best_score}")
                if epoch - best_epoch >= args.earlystop:
                    debug(f"STOP TRAINING FOR EARLY STOPPING")
                    endtime = time.time()
                    dtime = endtime - starttime
                    debug(f"Time Used：{dtime}")
                    break

        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        debug("Make prediction on test set and evaluate prediction")
        test_preds, test_vars = predict(model=model,
                                        data_loader=test_data_loader,
                                        scaler=scaler)

        test_scores = evaluate_predictions(preds=test_preds,
                                           targets=test_targets,
                                           num_tasks=args.num_tasks,
                                           metrics=args.metrics,
                                           dataset_type=args.dataset_type,
                                           logger=logger)

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)
            if args.mve:
                test_uncertainties.add_unc("MVE", np.array(test_vars)[:, 0])

        if args.ensemble_size > 1:
            list_test_preds.append(np.array(test_preds))

        debug("Make prediction on valid set")
        val_preds, val_vars = predict(
            model=model,
            data_loader=val_data_loader,
            scaler=scaler
        )

        if len(val_preds) != 0:
            sum_val_preds += np.array(val_preds)
            if args.mve:
                val_uncertainties.add_unc("MVE", np.array(val_vars)[:, 0])

        if args.ensemble_size > 1:
            list_val_preds.append(np.array(val_preds))

        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{metric}', avg_test_score, 0)

            if args.show_individual_scores:
                for task_name, test_score in zip(args.task_names, scores):
                    info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        writer.close()

        debug("Prepare Readouts and Labels")
        train_data_readout = extract_readout(model=model,
                                             data_loader=train_data_loader).to(args.device)

        val_data_readout = extract_readout(model=model,
                                           data_loader=val_data_loader).to(args.device)

        test_data_readout = extract_readout(model=model,
                                            data_loader=test_data_loader).to(args.device)

        test_ldist = prior_unc(train_data_readout.detach().cpu().numpy(),
                         test_data_readout.detach().cpu().numpy(),
                         fp_fun=lambda x: x,
                         dis_fun=lambda x, y: cosine(x, y),
                         sc_fun=lambda x: np.min(x))

        test_uncertainties.add_unc("LDIST", test_ldist)

        val_ldist = prior_unc(train_data_readout.detach().cpu().numpy(),
                        val_data_readout.detach().cpu().numpy(),
                        fp_fun=lambda x: x,
                        dis_fun=lambda x, y: cosine(x, y),
                        sc_fun=lambda x: np.min(x))

        val_uncertainties.add_unc("LDIST", val_ldist)

        from mhfp.encoder import MHFPEncoder
        mhfp_encoder = MHFPEncoder()

        test_fdist = prior_unc(train_data.mols(),
                         test_data.mols(),
                         fp_fun=lambda x: mhfp_encoder.encode_mol(x),
                         dis_fun=lambda x, y: mhfp_encoder.distance(x, y),
                         sc_fun=lambda x: np.min(x))
        test_uncertainties.add_unc('FDIST', test_fdist)

        val_fdist = prior_unc(train_data.mols(),
                        val_data.mols(),
                        fp_fun=lambda x: mhfp_encoder.encode_mol(x),
                        dis_fun=lambda x, y: mhfp_encoder.distance(x, y),
                        sc_fun=lambda x: np.min(x))
        val_uncertainties.add_unc('FDIST', val_fdist)

        epoch_df = pd.DataFrame({"smiles": test_data.smiles(flatten=True),
                                 "test_labels": np.array(test_targets).squeeze(),
                                 "test_preds": np.array(test_preds).squeeze(),
                                 "test_vars": np.array(test_vars).squeeze(),
                                 "test_fdist": test_fdist,
                                 "test_ldsit": test_ldist})
        epoch_df.to_csv(save_dir+".csv", index=None)

    avg_test_preds = (sum_test_preds / args.ensemble_size)[:, 0]
    test_ensemble_var = np.var(np.stack(list_test_preds), axis=0)[:, 0]
    test_uncertainties.add_unc("ENS", test_ensemble_var)
    test_uncertainties.mean()

    avg_val_preds = (sum_val_preds / args.ensemble_size)[:, 0]
    val_ensemble_var = np.var(np.stack(list_val_preds), axis=0)[:, 0]
    val_uncertainties.add_unc("ENS", val_ensemble_var)
    val_uncertainties.mean()

    ensemble_scores = evaluate_predictions(preds=avg_test_preds.reshape(-1, 1),
                                           targets=test_targets,
                                           num_tasks=args.num_tasks,
                                           metrics=args.metrics,
                                           dataset_type=args.dataset_type,
                                           logger=logger)

    for metric, scores in ensemble_scores.items():
        avg_ensemble_test_score = np.nanmean(scores)
        info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    # For Paper Results
    if args.save_preds:

        assert test_data.num_tasks() == 1
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles(flatten=True)})

        test_preds_dataframe['label']   = test_labels = np.array(test_targets)[:, 0]
        test_preds_dataframe['preds'] = test_preds  = avg_test_preds

        for name in test_uncertainties.uncertainty_collection:
            test_preds_dataframe[name] = test_uncertainties.uncertainty_collection[name]

        val_labels = np.array(val_targets)[:, 0]
        val_preds  = avg_val_preds

        if args.conformal:
            intervals = conformal_pipeline(1 - args.conformalp,
                                           val_labels,
                                           val_preds,
                                           test_preds)

            test_preds_dataframe[f'BKG_l'] = intervals[:, 0]
            test_preds_dataframe[f'BKG_u'] = intervals[:, 1]
            test_conformal_dict.update({'BKG': {'U': intervals[:, 1], 'L': intervals[:, 0]}})

            intervals = conformal_pipeline(1 - args.conformalp,
                                           val_labels,
                                           val_preds,
                                           test_preds,
                                           np.sqrt(val_uncertainties.uncertainty_collection["ENS"]),
                                           np.sqrt(test_uncertainties.uncertainty_collection["ENS"]))

            test_preds_dataframe[f'ESD_l'] = intervals[:, 0]
            test_preds_dataframe[f'ESD_u'] = intervals[:, 1]
            test_conformal_dict.update({f'ESD': {'U': intervals[:, 1], 'L': intervals[:, 0]}})

            results_df = conformal_curves(test_labels, test_conformal_dict)
            results_df.to_csv(os.path.join(os.path.dirname(args.save_dir),
                                           'cf_metrics',
                                           os.path.basename(args.save_dir) + '.csv'))

        # Stacking Different Uncertainty Estimations
        if len(test_uncertainties.uncertainty_collection) >= 1:

            test_ensemble_uncertainties = {}

            # 0 Ensemble_Variance + Alleatoric Error
            if args.mve and args.ensemble_size > 1:
                bayesian_uncertainty = test_uncertainties.simply_add(['ENS', 'MVE'])
                test_ensemble_uncertainties['BYS'] = bayesian_uncertainty

            # 1 Ensemble
            weights = {}
            SPR = {}
            val_uc_names = list(val_uncertainties.get_names())

            for uc_name in val_uc_names:
                spr, p = spearmanr(val_uncertainties.uncertainty_collection[uc_name], np.abs(val_preds - val_labels))
                if spr < 0:
                    spr = 0
                SPR[uc_name] = spr

            weights.update({"Unweighted": {name: 1. for name in val_uc_names}})
            weights.update({"Weighted": {name: SPR[name] for name in val_uc_names}})
            weights.update({"SPR_noMVE": {name: SPR[name] for name in val_uc_names if name != "MVE"}})
            weights.update({"SPR_noENS": {name: SPR[name] for name in val_uc_names if name != "ENS"}})
            weights.update({"SPR_noLDIST": {name: SPR[name] for name in val_uc_names if name != "LDIST"}})
            weights.update({"SPR_noFDIST": {name: SPR[name] for name in val_uc_names if name != "FDIST"}})
            debug(weights)

            test_ensemble_uncertainties.update(test_uncertainties.ensemble(weights))

            def calibrated_weights(unc_dict,
                                   residues,
                                   keys):

                from scipy.optimize import minimize
                weights0 = np.concatenate([np.ones(len(keys)), [1]])
                bounds   = tuple((1e-7, None) for _ in range(len(keys)+1))
                unc_mat = np.stack([unc_dict[k] for k in keys])

                def nll(w):
                    calibrated_uncertainties = np.sum([w[m] * unc_mat[m] for m in range(len(keys))],axis=0) + w[-1]
                    return np.mean(np.log(calibrated_uncertainties) + (residues * residues) / calibrated_uncertainties)

                res = minimize(nll, weights0, bounds=bounds)
                return res.x

            weights = calibrated_weights(val_uncertainties.uncertainty_collection,
                                         np.abs(val_preds - val_labels),
                                         keys=["MVE", "ENS", "FDIST", "LDIST"])
            debug(weights)
            calibrated_uncertainty = np.sum([weights[m] * test_uncertainties.uncertainty_collection[n] for m,n in
                                             enumerate(["MVE", "ENS", "FDIST", "LDIST"])], axis=0) \
                                     + weights[-1]
            test_ensemble_uncertainties.update({"NLLCAL": calibrated_uncertainty})

            VAL_UC = {"MVE": val_uncertainties.uncertainty_collection["MVE"],
                      "ENS": val_uncertainties.uncertainty_collection["ENS"],
                      "FDIST": val_uncertainties.uncertainty_collection["FDIST"],
                      "LDIST": val_uncertainties.uncertainty_collection["LDIST"]}

            TEST_UC = {"MVE": test_uncertainties.uncertainty_collection["MVE"],
                       "ENS": test_uncertainties.uncertainty_collection["ENS"],
                       "FDIST": test_uncertainties.uncertainty_collection["FDIST"],
                       "LDIST": test_uncertainties.uncertainty_collection["LDIST"]}

            weights = calibrated_weights(VAL_UC,
                                         np.abs(val_preds - val_labels),
                                         keys=["MVE", "ENS", "FDIST", "LDIST"])
            debug(weights)
            calibrated_uncertainty = weights[0] * TEST_UC["MVE"] + \
                                     weights[1] * TEST_UC["ENS"] + \
                                     weights[2] * TEST_UC["FDIST"] + \
                                     weights[3] * TEST_UC["LDIST"] + \
                                     weights[4]
            test_ensemble_uncertainties.update({"ALL_NLLCAL": calibrated_uncertainty})

            VAL_UC = {"MVE": val_uncertainties.uncertainty_collection["MVE"],
                      "ENS": val_uncertainties.uncertainty_collection["ENS"]}
            TEST_UC = {"MVE": test_uncertainties.uncertainty_collection["MVE"],
                       "ENS": test_uncertainties.uncertainty_collection["ENS"]}

            weights = calibrated_weights(VAL_UC,
                                         np.abs(val_preds - val_labels),
                                         keys=["MVE", "ENS"])
            debug(weights)
            calibrated_uncertainty = weights[0] * TEST_UC["MVE"] + \
                                     weights[1] * TEST_UC["ENS"] + \
                                     weights[2]
            test_ensemble_uncertainties.update({"BAY_NLLCAL": calibrated_uncertainty})

            VAL_UC = {"LDIST": val_uncertainties.uncertainty_collection["LDIST"]}
            TEST_UC = {"LDIST": test_uncertainties.uncertainty_collection["LDIST"]}

            weights = calibrated_weights(VAL_UC, np.abs(val_preds - val_labels), keys=["LDIST"])
            debug(weights)
            calibrated_uncertainty = weights[0] * TEST_UC["LDIST"] + weights[1]
            test_ensemble_uncertainties.update({"LDIST_NLLCAL": calibrated_uncertainty})

            test_uncertainties.uncertainty_collection.update(test_ensemble_uncertainties)

            for name in test_uncertainties.uncertainty_collection:
                test_preds_dataframe[name] = test_uncertainties.uncertainty_collection[name]

            multiple_evaluation(test_labels,
                                test_preds,
                                test_uncertainties.uncertainty_collection,
                                ranking_evaluation,
                                os.path.join(os.path.dirname(args.save_dir),
                                             'Confidence_Curves_metrics',
                                             os.path.basename(args.save_dir) + '.csv'))

            multiple_evaluation(test_labels,
                                test_preds,
                                test_uncertainties.uncertainty_collection,
                                cbc_evaluation,
                                os.path.join(os.path.dirname(args.save_dir),
                                             'CBC_Curves_metrics',
                                             os.path.basename(args.save_dir) + '.csv'))

            multiple_evaluation(test_labels,
                                test_preds,
                                test_uncertainties.uncertainty_collection,
                                ebc_evaluation,
                                os.path.join(os.path.dirname(args.save_dir),
                                             'EBC_Curves_metrics',
                                             os.path.basename(args.save_dir) + '.csv'))

            multiple_evaluation(test_labels,
                                test_preds,
                                test_uncertainties.uncertainty_collection,
                                CI_evaluation,
                                os.path.join(os.path.dirname(args.save_dir),
                                             'CI_metrics',
                                             os.path.basename(args.save_dir) + '.csv'))

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores
