"""
Simplified CBP Trainer for ChemProp
====================================
This module provides a lightweight trainer for Continual Backpropagation (CBP).
The CBP logic is primarily handled by CBPLinear layers embedded directly in the model.

Key simplifications from the original implementation:
1. Removed GnTForChemprop class - CBPLinear layers handle neuron replacement
2. Unified handling of MPN and FFN layers through CBPLinear
3. Automatic neuron replacement through forward/backward hooks
4. Simplified logging and statistics collection
"""

from typing import Dict, Callable, Optional
import torch
import torch.nn as nn
from torch.optim import SGD

from chemprop.models.model import MoleculeModel
from chemprop.models.AdamGnT import AdamGnT
from chemprop.args import TrainArgs
from chemprop.train.loss_functions import (
    mcc_class_loss,
    mcc_multiclass_loss,
    dirichlet_class_loss,
    dirichlet_multiclass_loss,
    evidential_loss,
    bounded_mse_loss,
    sid_loss,
    wasserstein_loss,
)
from .cbp_logger import CBPLogger


class ContinualBackpropTrainer:
    """Lightweight trainer that coordinates CBP training through CBPLinear layers embedded in the model."""

    def __init__(
        self,
        model: MoleculeModel,
        args: TrainArgs,
        step_size: float = 0.001,
        replacement_rate: float = 0.001,
        decay_rate: float = 0.9,
        maturity_threshold: int = 100,
        util_type: str = 'contribution',
        accumulate: bool = True,
        enable_cbp_logging: bool = True,
        log_dir: str = None,
        enable_gradient_logging: bool = True,
        gradient_log_frequency: int = 1000000,
        enable_wandb: bool = False,
        wandb_project: str = "cbp-training",
        wandb_entity: Optional[str] = None,
    ):
        self.model = model
        self.args = args
        self.device = args.device

        # Store CBP parameters
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.accumulate = accumulate

        # Collect all CBPLinear layers - if model doesn't have the method, CBP might not be enabled
        if hasattr(model, 'get_all_cbp_layers'):
            self.cbp_layers = model.get_all_cbp_layers()

            # Configure CBP parameters
            self._configure_cbp_layers(
                replacement_rate=replacement_rate,
                decay_rate=decay_rate,
                maturity_threshold=maturity_threshold,
                util_type=util_type,
                accumulate=accumulate
            )
        else:
            self.cbp_layers = []
            print("⚠️ No CBPLinear layers found - model might not have CBP enabled")

        # Setup unified CBP logging (including gradient logging)
        self.enable_cbp_logging = enable_cbp_logging and len(self.cbp_layers) > 0
        self.enable_gradient_logging = enable_gradient_logging
        self.cbp_logger = None

        if self.enable_cbp_logging or enable_gradient_logging:
            # Create unified CBP logger
            cbp_log_dir = log_dir if log_dir else "cbp_logs"
            self.cbp_logger = CBPLogger(
                log_dir=cbp_log_dir,
                log_frequency=gradient_log_frequency,
                enable_wandb=enable_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                track_histogram=True
            )

            # Attach logger to all CBP layers
            for cbp_layer in self.cbp_layers:
                cbp_layer.cbp_logger = self.cbp_logger
                cbp_layer.grad_log_frequency = gradient_log_frequency

            self._setup_cbp_logging()

            print(f"📊 CBP logging enabled - logs will be saved to {cbp_log_dir}")
            if enable_gradient_logging:
                print(f"📈 Gradient tracking enabled")
            if enable_wandb:
                print(f"📈 WandB integration enabled - project: {wandb_project}")

        # Setup optimizer - Always use AdamGnT for CBP mode
        # AdamGnT uses per-element step counters which is better for neuron replacement
        if args.optimizer == 'adam':
            self.optimizer = AdamGnT(
                model.parameters(),
                lr=step_size,
                betas=(0.9, 0.999),
                weight_decay=args.weight_decay
            )
            print("🚀 Using AdamGnT optimizer for CBP training (per-element step counters)")
        else:
            self.optimizer = SGD(
                model.parameters(),
                lr=step_size,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
            print("📈 Using SGD optimizer for CBP training")

        # Pass optimizer reference to CBP layers for AdamGnT step counter reset
        for cbp_layer in self.cbp_layers:
            cbp_layer._optimizer_ref = self.optimizer

    def _configure_cbp_layers(self, replacement_rate, decay_rate, maturity_threshold, util_type, accumulate):
        """Configure all CBPLinear layers with unified parameters."""
        for i, cbp_layer in enumerate(self.cbp_layers):
            # Update CBP parameters
            cbp_layer.replacement_rate = replacement_rate
            cbp_layer.decay_rate = decay_rate
            cbp_layer.maturity_threshold = maturity_threshold
            cbp_layer.util_type = util_type
            cbp_layer.accumulate = accumulate

            # Note: optimizer reference will be set after optimizer is created

            # Set layer name if not already set
            if not hasattr(cbp_layer, 'layer_name') or cbp_layer.layer_name is None:
                if cbp_layer.layer_type == 'FFN':
                    cbp_layer.layer_name = f'FFN_Layer{i}'
                else:
                    cbp_layer.layer_name = f'MPN_Layer{i}'

    def _setup_cbp_logging(self):
        """Setup logging for all CBPLinear layers."""
        for cbp_layer in self.cbp_layers:
            cbp_layer._batch_counter = 0
            cbp_layer.current_epoch = 0

            # Register gradient logging hook if gradient logging is enabled
            if self.enable_gradient_logging:
                # Import the log_gradients function from cbp_linear
                from .cbp_linear import log_gradients
                # Register the hook - we need to do this here because the logger
                # wasn't available when the CBPLinear layer was initialized
                # Remove any existing gradient hooks first to avoid duplicates
                if hasattr(cbp_layer, '_backward_hooks'):
                    # Create a list of hook ids to remove
                    hooks_to_remove = []
                    for hook_id, hook in cbp_layer._backward_hooks.items():
                        if hook.__name__ == 'log_gradients':
                            hooks_to_remove.append(hook_id)
                    # Remove the old hooks
                    for hook_id in hooks_to_remove:
                        del cbp_layer._backward_hooks[hook_id]

                # Register the new hook
                cbp_layer.register_full_backward_hook(log_gradients)

    def train_step(self,
                   batch_data: Dict = None,
                   targets: torch.Tensor = None,
                   atom_descriptors_batch: torch.Tensor = None,
                   atom_features_batch: torch.Tensor = None,
                   bond_descriptors_batch: torch.Tensor = None,
                   bond_features_batch: torch.Tensor = None,
                   data_weights: torch.Tensor = None,
                   target_weights: torch.Tensor = None,
                   mask: torch.Tensor = None,
                   lt_targets: torch.Tensor = None,
                   gt_targets: torch.Tensor = None,
                   batch_idx: int = 0,
                   epoch: int = 0,
                   loss_func: Callable = None):
        """
        Unified train step for CBP training using the embedded CBPLinear layers

        Args:
            batch_data: Dictionary containing batch data prepared by get_batch_data()
            targets: Target values
            atom_descriptors_batch: Atom-level descriptors
            atom_features_batch: Atom features
            bond_descriptors_batch: Bond descriptors (not currently used in molecules)
            bond_features_batch: Bond features
            data_weights: Per-sample weights
            target_weights: Per-target weights
            mask: Binary mask for valid values
            lt_targets: Less than targets for bounded regression
            gt_targets: Greater than targets for bounded regression
            batch_idx: Current batch index
            epoch: Current epoch
            loss_func: Loss function to use

        Returns:
            loss: Training loss value
        """
        # Prepare batch data if not already prepared
        if batch_data is None:
            # This should be prepared by the training loop
            raise ValueError("batch_data must be provided")

        # Get arguments for loss calculation
        args = self.args

        # Complete loss function handling logic from original train.py
        if loss_func is None:
            if args.dataset_type == 'classification':
                if args.loss_function == 'binary_cross_entropy':
                    loss_func = nn.BCEWithLogitsLoss(reduction='none')
                elif args.loss_function == 'mcc':
                    loss_func = mcc_class_loss
                elif args.loss_function == 'dirichlet':
                    loss_func = dirichlet_class_loss
                else:
                    raise ValueError(f'Loss function {args.loss_function} not supported for {args.dataset_type} dataset type')
            elif args.dataset_type == 'regression':
                if args.loss_function == 'mse':
                    loss_func = nn.MSELoss(reduction='none')
                elif args.loss_function == 'bounded_mse':
                    loss_func = bounded_mse_loss
                elif args.loss_function == 'evidential':
                    loss_func = evidential_loss
                else:
                    raise ValueError(f'Loss function {args.loss_function} not supported for {args.dataset_type} dataset type')
            elif args.dataset_type == 'multiclass':
                if args.loss_function == 'cross_entropy':
                    loss_func = nn.CrossEntropyLoss(reduction='none')
                elif args.loss_function == 'mcc':
                    loss_func = mcc_multiclass_loss
                elif args.loss_function == 'dirichlet':
                    loss_func = dirichlet_multiclass_loss
                else:
                    raise ValueError(f'Loss function {args.loss_function} not supported for {args.dataset_type} dataset type')
            elif args.dataset_type == 'spectra':
                if args.loss_function == 'sid':
                    loss_func = sid_loss
                elif args.loss_function == 'wasserstein':
                    loss_func = wasserstein_loss
                else:
                    raise ValueError(f'Loss function {args.loss_function} not supported for {args.dataset_type} dataset type')
            else:
                # Unsupported dataset type - use MSE as default
                loss_func = nn.MSELoss(reduction='none')
        self.model.train()

        # Forward pass - no need to track features, CBPLinear handles everything internally
        preds = self.model(
            batch_data['mol_batch'],
            batch_data['features_batch'],
            batch_data['atom_descriptors_batch'],
            batch_data['atom_features_batch'],
            batch_data['bond_features_batch']
        )

        # Move tensors to correct device
        torch_device = preds.device
        mask = mask.to(torch_device)
        targets = targets.to(torch_device)
        target_weights = target_weights.to(torch_device)
        data_weights = data_weights.to(torch_device)
        if lt_targets is not None:
            lt_targets = lt_targets.to(torch_device)
        if gt_targets is not None:
            gt_targets = gt_targets.to(torch_device)

        # Complete loss function handling logic - identical to original train.py
        if args.loss_function == 'mcc' and args.dataset_type == 'classification':
            loss = loss_func(preds, targets, data_weights, mask) * target_weights.squeeze(0)
        elif args.loss_function == 'mcc': # multiclass dataset type
            targets = targets.long()
            target_losses = []
            for target_index in range(preds.size(1)):
                target_loss = loss_func(preds[:, target_index, :], targets[:, target_index], data_weights, mask[:, target_index]).unsqueeze(0)
                target_losses.append(target_loss)
            loss = torch.cat(target_losses).to(torch_device) * target_weights.squeeze(0)
        elif args.dataset_type == 'multiclass':
            targets = targets.long()
            if args.loss_function == 'dirichlet':
                loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
            else:
                target_losses = []
                for target_index in range(preds.size(1)):
                    target_loss = loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1)
                    target_losses.append(target_loss)
                loss = torch.cat(target_losses, dim=1).to(torch_device) * target_weights * data_weights * mask
        elif args.dataset_type == 'spectra':
            loss = loss_func(preds, targets, mask) * target_weights * data_weights * mask
        elif args.loss_function == 'bounded_mse':
            loss = loss_func(preds, targets, lt_targets, gt_targets) * target_weights * data_weights * mask
        elif args.loss_function == 'evidential':
            loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
        elif args.loss_function == 'dirichlet': # classification
            loss = loss_func(preds, targets, args.evidential_regularization) * target_weights * data_weights * mask
        else:
            loss = loss_func(preds, targets) * target_weights * data_weights * mask

        loss = loss.sum() / mask.sum()

        # Standard backpropagation step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping if enabled
        if args.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)

        self.optimizer.step()

        # CBPLinear layers handle neuron replacement automatically through their hooks

        # Log training metrics
        if self.cbp_logger:
            self.cbp_logger.log_training_metrics(
                loss=loss.item(),
                batch_idx=batch_idx,
                epoch=epoch
            )

        return loss.item()

    def log_epoch_cbp_stats(self, epoch: int):
        """Update epoch information and save epoch summary."""
        import numpy as np

        # Update epoch for all CBP layers
        for cbp_layer in self.cbp_layers:
            cbp_layer.current_epoch = epoch

        # Calculate and log CBP-specific statistics
        if self.cbp_logger and self.cbp_layers:
            # Collect statistics from all CBP layers
            total_ages = []
            mature_count = 0
            total_neurons = 0

            for cbp_layer in self.cbp_layers:
                # Get ages as numpy array
                ages = cbp_layer.ages.detach().cpu().numpy() if torch.is_tensor(cbp_layer.ages) else cbp_layer.ages
                total_ages.extend(ages)

                # Count mature neurons (age > maturity_threshold)
                mature_count += np.sum(ages > cbp_layer.maturity_threshold)
                total_neurons += len(ages)

            # Calculate average age
            avg_age = float(np.mean(total_ages)) if total_ages else 0.0

            # Get total replacements for this epoch
            total_replacements = self.cbp_logger.get_total_replacements()

            # Force log final gradient/utility/activation data at epoch end
            for cbp_layer in self.cbp_layers:
                # Get current batch counter
                batch_idx = getattr(cbp_layer, '_forward_log_counter', 0)
                if batch_idx == 0:
                    batch_idx = getattr(cbp_layer, '_grad_log_counter', 0)

                # Get gradient magnitude
                grad_magnitude = torch.zeros_like(cbp_layer.util)
                if hasattr(cbp_layer.out_layer.weight, 'grad') and cbp_layer.out_layer.weight.grad is not None:
                    grad_magnitude = cbp_layer.out_layer.weight.grad.abs().mean(dim=0)

                # Log final state of this epoch
                self.cbp_logger.log_gradients(
                    layer_name=cbp_layer.layer_name,
                    gradients=grad_magnitude,
                    utilities=cbp_layer.util,
                    ages=cbp_layer.ages,
                    batch_idx=batch_idx,
                    epoch=epoch
                )

                # Also log final activations (using utility as proxy if no activations available)
                self.cbp_logger.log_activations(
                    layer_name=cbp_layer.layer_name,
                    activations=cbp_layer.util,  # Use utility as a proxy for neuron importance
                    batch_idx=batch_idx,
                    epoch=epoch
                )

            # Log complete CBP statistics
            self.cbp_logger.log_cbp_stats(
                replacement_count=total_replacements,
                replacement_rate=self.replacement_rate,  # Use the configured replacement rate
                avg_neuron_age=avg_age,
                mature_neurons=int(mature_count),
                epoch=epoch
            )

            # Save epoch summary after logging stats
            self.cbp_logger.save_epoch_summary(epoch)

        elif self.cbp_logger:
            # Still save summary even if no CBP layers
            self.cbp_logger.save_epoch_summary(epoch)

    def save_cbp_summary(self):
        """Save CBP training summary and close logger."""
        if self.cbp_logger:
            # Note: close() will call save_full_history() internally
            self.cbp_logger.close()
            print("📊 CBP history and gradients saved successfully")