import json
from logging import Logger
import os
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .loss_functions import get_loss_func
from .cbp_trainer import ContinualBackpropTrainer
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count, param_count_all
from chemprop.utils import build_optimizer, build_lr_scheduler, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model, multitask_mean, load_mpn_model


def get_structured_save_dir(args: TrainArgs, base_dir: str = None) -> str:
    """
    Generate a structured save directory path to avoid conflicts in parallel runs.
    
    :param args: Training arguments
    :param base_dir: Base directory to use (if None, uses args.save_dir)
    :return: Structured save directory path
    """
    # Extract dataset name from data path
    if hasattr(args, 'data_path') and args.data_path:
        dataset_name = Path(args.data_path).stem
    else:
        dataset_name = "default"
    
    # Determine mode
    mode = "cbp" if hasattr(args, 'cbp') and args.cbp else "standard"
    
    # Get seed
    seed = args.seed if hasattr(args, 'seed') else 0
    
    # Build structured path (remove timestamp to keep path short)
    if base_dir is None:
        base_dir = args.save_dir if hasattr(args, 'save_dir') and args.save_dir else "results"
    
    experiment_id = f"{mode}_seed{seed}"
    
    # If base_dir already contains fold information, append directly
    if 'fold_' in base_dir:
        structured_dir = os.path.join(base_dir, experiment_id)
    else:
        structured_dir = os.path.join(base_dir, dataset_name, experiment_id)
    
    return structured_dir


def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model (with optional CBP), and returns test scores.

    This unified function handles both standard and CBP training based on the args.cbp flag.
    
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Check if CBP mode is enabled
    use_cbp = hasattr(args, 'cbp') and args.cbp
    if use_cbp:
        debug('CBP (Continual Backpropagation) mode enabled')
    
    # Use structured save directory to avoid conflicts
    if not hasattr(args, 'no_structured_dir') or not args.no_structured_dir:
        original_save_dir = args.save_dir
        args.save_dir = get_structured_save_dir(args)
        debug(f'Using structured save directory: {args.save_dir}')

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data - handle both simple and complex splitting scenarios
    debug(f'Splitting data with seed {args.seed}')
    
    # For CBP mode or when no separate paths are provided, use simplified splitting
    if use_cbp or (not args.separate_test_path and not args.separate_val_path):
        # Simple split for CBP mode or standard mode without separate paths
        train_data, val_data, test_data = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule if hasattr(args, 'split_key_molecule') else None,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger
        )
    else:
        # Complex splitting with separate validation/test paths (standard mode only)
        if args.separate_test_path:
            test_data = get_data(
                path=args.separate_test_path,
                args=args,
                features_path=args.separate_test_features_path,
                atom_descriptors_path=args.separate_test_atom_descriptors_path,
                bond_features_path=args.separate_test_bond_features_path,
                phase_features_path=args.separate_test_phase_features_path,
                smiles_columns=args.smiles_columns,
                loss_function=args.loss_function,
                logger=logger
            )
        
        if args.separate_val_path:
            val_data = get_data(
                path=args.separate_val_path,
                args=args,
                features_path=args.separate_val_features_path,
                atom_descriptors_path=args.separate_val_atom_descriptors_path,
                bond_features_path=args.separate_val_bond_features_path,
                phase_features_path=args.separate_val_phase_features_path,
                smiles_columns=args.smiles_columns,
                loss_function=args.loss_function,
                logger=logger
            )

        if args.separate_val_path and args.separate_test_path:
            train_data = data
        elif args.separate_val_path:
            train_data, _, test_data = split_data(
                data=data,
                split_type=args.split_type,
                sizes=args.split_sizes,
                key_molecule_index=args.split_key_molecule if hasattr(args, 'split_key_molecule') else None,
                seed=args.seed,
                num_folds=args.num_folds,
                args=args,
                logger=logger
            )
        elif args.separate_test_path:
            train_data, val_data, _ = split_data(
                data=data,
                split_type=args.split_type,
                sizes=args.split_sizes,
                key_molecule_index=args.split_key_molecule if hasattr(args, 'split_key_molecule') else None,
                seed=args.seed,
                num_folds=args.num_folds,
                args=args,
                logger=logger
            )

    # Handle classification dataset statistics
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
        train_class_sizes = get_class_sizes(train_data, proportion=False)
        args.train_class_sizes = train_class_sizes

    # Save SMILES splits if requested
    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    # Feature scaling
    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    # Atom descriptor scaling
    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(
            replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(
            atom_descriptor_scaler, scale_atom_descriptors=True)
        test_data.normalize_features(
            atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    # Bond feature scaling
    if args.bond_feature_scaling and args.bond_features_size > 0:
        bond_feature_scaler = train_data.normalize_features(
            replace_nan_token=0, scale_bond_features=True)
        val_data.normalize_features(
            bond_feature_scaler, scale_bond_features=True)
        test_data.normalize_features(
            bond_feature_scaler, scale_bond_features=True)
    else:
        bond_feature_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Check for empty splits
    if len(val_data) == 0:
        raise ValueError('The validation data split is empty. During normal chemprop training (non-sklearn functions), \
            a validation set is required to conduct early stopping according to the selected evaluation metric. This \
            may have occurred because validation data provided with `--separate_val_path` was empty or contained only invalid molecules.')

    empty_test_set = len(test_data) == 0
    if empty_test_set:
        debug('The test data split is empty. This may be either because splitting with no test set was selected, \
            such as with `cv-no-test`, or because test data provided with `--separate_test_path` was empty or contained only invalid molecules. \
            Performance on the test set will not be evaluated and metric scores will return `nan` for each task.')

    # Initialize scaler and scale training targets
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        scaler = train_data.normalize_targets()
        args.spectra_phase_mask = None
    elif args.dataset_type == 'spectra':
        debug('Normalizing spectra and excluding spectra regions based on phase')
        args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
        for dataset in [train_data, test_data, val_data]:
            data_targets = normalize_spectra(
                spectra=dataset.targets(),
                phase_features=dataset.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=None,
                threshold=args.spectra_target_floor,
            )
            dataset.set_targets(data_targets)
        scaler = None
    else:
        args.spectra_phase_mask = None
        scaler = None

    # Get loss function
    loss_func = get_loss_func(args)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros(
            (len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    # Set CBP log directory if in CBP mode
    if use_cbp:
        cbp_log_dir = os.path.join(args.save_dir, 'cbp_logs')
        os.environ['CBP_LOG_DIR'] = cbp_log_dir
        makedirs(cbp_log_dir)
        debug(f'CBP logs will be saved to: {cbp_log_dir}')
    
    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # Disable TensorBoard writer to avoid generating events.out.tfevents.* files
        writer = None

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            if use_cbp:
                debug(f'Building model {model_idx} with CBP support')
            else:
                debug(f'Building model {model_idx}')
            model = MoleculeModel(args)  # Model now handles CBP internally based on args.cbp

        # Optionally, overwrite weights
        if args.checkpoint_frzn is not None:
            debug(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
            model = load_frzn_model(
                model=model, path=args.checkpoint_frzn, current_args=args, logger=logger)

        if hasattr(args, 'mpn_path') and args.mpn_path is not None:
            debug(f'Loading MPN parameters from {args.mpn_path}.')
            model = load_mpn_model(
                model=model, path=args.mpn_path, current_args=args, logger=logger)

        debug(model)

        if args.checkpoint_frzn is not None or (hasattr(args, 'freeze_mpn') and args.freeze_mpn):
            debug(f'Number of unfrozen parameters = {param_count(model):,}')
            debug(f'Total number of parameters = {param_count_all(model):,}')
        else:
            debug(f'Number of parameters = {param_count_all(model):,}')

        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Create CBP trainer if in CBP mode
        cbp_trainer: Optional[ContinualBackpropTrainer] = None
        if use_cbp:
            debug('Creating CBP trainer')
            cbp_trainer = ContinualBackpropTrainer(
                model=model,
                args=args,
                step_size=args.init_lr,
                replacement_rate=args.replacement_rate,
                decay_rate=args.decay_rate,
                maturity_threshold=args.maturity_threshold,
                util_type=args.util_type,
                accumulate=False,
            )
            # Use CBP trainer's optimizer
            optimizer = cbp_trainer.optimizer
        else:
            # Standard optimizer
            optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                        features_scaler, atom_descriptor_scaler, bond_feature_scaler, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        loss_history = []  # Track loss for each epoch
        
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')
            
            # Train for one epoch - unified train function handles both modes
            n_iter, epoch_loss = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                cbp_trainer=cbp_trainer,  # Pass CBP trainer if available
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            
            # Record epoch loss
            loss_history.append(epoch_loss)
            debug(f'Epoch {epoch} average loss = {epoch_loss:.6f}')
            
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            
            # Log CBP statistics if in CBP mode
            if cbp_trainer and hasattr(cbp_trainer, 'log_epoch_cbp_stats'):
                cbp_trainer.log_epoch_cbp_stats(epoch)
            
            # Evaluate on validation set
            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Log validation scores
            for metric, scores in val_scores.items():
                # Average validation score
                mean_val_score = multitask_mean(scores, metric=metric)
                debug(f'Validation {metric} = {mean_val_score:.6f}')
                if writer is not None:
                    writer.add_scalar(f'validation_{metric}', mean_val_score, n_iter)

                if args.show_individual_scores:
                    # Individual validation scores
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                        if writer is not None:
                            writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            mean_val_score = multitask_mean(val_scores[args.metric], metric=args.metric)
            if args.minimize_score and mean_val_score < best_score or \
                    not args.minimize_score and mean_val_score > best_score:
                best_score, best_epoch = mean_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler,
                               atom_descriptor_scaler, bond_feature_scaler, args)

        # Save loss history with experiment metadata
        loss_history_path = os.path.join(save_dir, 'loss_history.json')
        with open(loss_history_path, 'w') as f:
            json.dump({
                'epochs': list(range(len(loss_history))),
                'losses': loss_history,
                'best_epoch': best_epoch,
                'best_score': best_score,
                'model_idx': model_idx,
                'mode': 'cbp' if use_cbp else 'standard',
                'seed': args.seed if hasattr(args, 'seed') else 0,
                'dataset': Path(args.data_path).stem if hasattr(args, 'data_path') and args.data_path else 'unknown',
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        info(f'Model {model_idx} loss history saved to {loss_history_path}')
        
        # Print loss summary
        info(f'Model {model_idx} Loss Summary:')
        info(f'  Initial loss: {loss_history[0]:.6f}' if loss_history else '  No loss recorded')
        info(f'  Final loss: {loss_history[-1]:.6f}' if loss_history else '  No loss recorded')
        if len(loss_history) > 1:
            info(f'  Min loss: {min(loss_history):.6f} (epoch {loss_history.index(min(loss_history))})')
            info(f'  Average loss: {np.mean(loss_history):.6f}')
        
        # Save CBP summary if in CBP mode
        if cbp_trainer and hasattr(cbp_trainer, 'save_cbp_summary'):
            cbp_trainer.save_cbp_summary()
        
        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        if empty_test_set:
            info(f'Model {model_idx} provided with no test set, no metric evaluation will be performed.')
        else:
            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler
            )
            
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                gt_targets=test_data.gt_targets() if hasattr(test_data, 'gt_targets') else None,
                lt_targets=test_data.lt_targets() if hasattr(test_data, 'lt_targets') else None,
                logger=logger
            )

            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds)

            # Average test score
            for metric, scores in test_scores.items():
                avg_test_score = np.nanmean(scores)
                info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
                if writer is not None:
                    writer.add_scalar(f'test_{metric}', avg_test_score, 0)

                if args.show_individual_scores and args.dataset_type != 'spectra':
                    # Individual test scores
                    for task_name, test_score in zip(args.task_names, scores):
                        info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                        writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        
        if writer is not None:
            writer.close()

    # Evaluate ensemble on test set
    if empty_test_set:
        ensemble_scores = {
            metric: [np.nan for task in args.task_names] for metric in args.metrics
        }
    else:
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            gt_targets=test_data.gt_targets() if hasattr(test_data, 'gt_targets') else None,
            lt_targets=test_data.lt_targets() if hasattr(test_data, 'lt_targets') else None,
            logger=logger
        )

    for metric, scores in ensemble_scores.items():
        # Average ensemble score
        mean_ensemble_test_score = multitask_mean(scores, metric=metric)
        info(f'Ensemble test {metric} = {mean_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    # Save scores
    with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)
    
    # Save overall training summary with comprehensive metadata
    training_summary = {
        'ensemble_test_scores': ensemble_scores,
        'training_mode': 'CBP' if use_cbp else 'Standard',
        'num_epochs': args.epochs,
        'ensemble_size': args.ensemble_size,
        'best_validation_metric': args.metric,
        'dataset': Path(args.data_path).stem if hasattr(args, 'data_path') and args.data_path else 'unknown',
        'seed': args.seed if hasattr(args, 'seed') else 0,
        'save_dir': args.save_dir,
        'timestamp_completed': datetime.now().isoformat(),
        'cbp_enabled': use_cbp,
        'cbp_params': {
            'replacement_rate': args.replacement_rate if use_cbp and hasattr(args, 'replacement_rate') else None,
            'decay_rate': args.decay_rate if use_cbp and hasattr(args, 'decay_rate') else None,
            'maturity_threshold': args.maturity_threshold if use_cbp and hasattr(args, 'maturity_threshold') else None,
            'util_type': args.util_type if use_cbp and hasattr(args, 'util_type') else None
        } if use_cbp else None
    }
    
    with open(os.path.join(args.save_dir, 'training_summary.json'), 'w') as f:
        json.dump(training_summary, f, indent=4, sort_keys=True)
    
    info(f'\nTraining Summary saved to {os.path.join(args.save_dir, "training_summary.json")}')
    info(f'All results saved to: {args.save_dir}')

    # Optionally save test predictions
    if args.save_preds and not empty_test_set:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})

        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]
            test_preds_dataframe[task_name + '_true'] = [target[i] for target in test_targets]

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    # Store the actual save_dir used in ensemble_scores for cross_validate to find files
    ensemble_scores['_save_dir'] = args.save_dir
    
    return ensemble_scores


# Create backward compatibility alias
def run_training_cbp(args: TrainArgs,
                     data: MoleculeDataset,
                     logger: Logger = None) -> Dict[str, List[float]]:
    """
    Backward compatibility wrapper for CBP training.
    Ensures CBP mode is enabled and calls the unified run_training function.
    
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric to a list of values for each task.
    """
    # Ensure CBP mode is enabled
    args.cbp = True
    return run_training(args, data, logger)