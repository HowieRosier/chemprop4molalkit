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

from typing import Dict, Callable
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import json
import os
from datetime import datetime

from chemprop.models.model import MoleculeModel
from chemprop.models.AdamGnT import AdamGnT
from chemprop.args import TrainArgs


class CBPLogger:
    """Logger for monitoring CBP training statistics."""
    
    def __init__(self, log_dir: str = None, log_filename: str = None):
        # Use environment variable if set, otherwise use default
        self.log_dir = log_dir or os.environ.get('CBP_LOG_DIR', "cbp_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"cbp_training_{timestamp}.log"
        
        self.log_file = os.path.join(self.log_dir, log_filename)
        self.stats_history = []
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"CBP Training Log - Started at {datetime.now()}\n")
            f.write("="*80 + "\n")
    
    def log_batch_stats(self, batch_id: int, stats: Dict, layer_type: str = "FFN"):
        """
        Log batch-level CBP statistics, including replaced neuron indices
        
        :param batch_id: Batch number
        :param stats: Dictionary containing CBP statistics
        :param layer_type: Type of layer ("FFN" or "MPN")
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            total_replaced = stats.get('total_neurons_replaced', 0)
            if total_replaced > 0:
                f.write(f"Batch {batch_id:4d} ({timestamp}) [{layer_type}]: {total_replaced} neurons replaced")
                
                # Add layer replacement information
                if 'layer_replacements' in stats:
                    f.write(f" {stats['layer_replacements']}")
                
                # 🔥 Add specific indices of replaced neurons
                if 'replaced_neuron_indices' in stats:
                    indices = stats['replaced_neuron_indices']
                    f.write(f"\n    └─ Replaced indices: {indices}")
                
                # Add layer names if provided
                if 'layer_names' in stats:
                    f.write(f"\n    └─ Layers: {stats['layer_names']}")
                
                f.write("\n")
        
        # Console output key information (only when replacement occurs)
        total_replaced = stats.get('total_neurons_replaced', 0)
        if total_replaced > 0:
            # 🔥 Support multiple hidden layers: show replacement indices for each layer
            replaced_indices = stats.get('replaced_neuron_indices', [])
            layer_names = stats.get('layer_names', [])
            layer_info = []
            
            for layer_idx, indices in enumerate(replaced_indices):
                if indices:  # If this layer has neurons replaced
                    layer_name = layer_names[layer_idx] if layer_idx < len(layer_names) else f"L{layer_idx}"
                    layer_info.append(f"{layer_name}:{indices}")
            
            if layer_info:
                print(f"🔥 Batch {batch_id} [{layer_type}]: {total_replaced} neurons replaced [{', '.join(layer_info)}]")
            else:
                print(f"🔥 Batch {batch_id} [{layer_type}]: {total_replaced} neurons replaced")
    
    def log_epoch_summary(self, epoch: int, stats: Dict):
        """Log epoch-level summary statistics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to history
        epoch_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            **stats
        }
        self.stats_history.append(epoch_data)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"EPOCH {epoch} SUMMARY - {timestamp}\n")
            f.write(f"{'='*60}\n")
            
            # Key metrics
            total_replaced = stats.get('total_neurons_replaced', 0)
            f.write(f"🔥 Total neurons replaced this epoch: {total_replaced}\n")
            
            if 'layer_replacements' in stats:
                f.write(f"Layer-wise replacements: {stats['layer_replacements']}\n")
            
            if 'replacement_rates' in stats:
                f.write(f"Average replacement rates: {stats['replacement_rates']}\n")
            
            if 'avg_utilities' in stats:
                f.write(f"Average utilities: {stats['avg_utilities']}\n")
            
            if 'age_stats' in stats:
                f.write(f"Age statistics: {stats['age_stats']}\n")
            
            if 'active_batches' in stats:
                f.write(f"Active batches (with replacements): {stats['active_batches']}\n")
            
            if 'total_batches' in stats:
                f.write(f"Total batches processed: {stats['total_batches']}\n")
            
            # 🔥 Add summary information of replaced neuron indices
            if 'all_replaced_indices' in stats or 'ffn_replaced_indices' in stats or 'mpn_replaced_indices' in stats:
                f.write(f"\n📍 Replaced Neuron Indices Summary:\n")
                
                # Display FFN layer indices
                if 'ffn_replaced_indices' in stats:
                    ffn_indices = stats['ffn_replaced_indices']
                    if ffn_indices:
                        f.write(f"  FFN Layers:\n")
                        for layer_name, indices in sorted(ffn_indices.items()):
                            if indices:
                                f.write(f"    {layer_name}: {indices}\n")
                elif 'all_replaced_indices' in stats:
                    # Backward compatibility
                    all_indices = stats['all_replaced_indices']
                    for layer_idx, layer_indices in enumerate(all_indices):
                        if layer_indices:
                            f.write(f"   Layer {layer_idx}: {layer_indices}\n")
                
                # Display MPN layer indices
                if 'mpn_replaced_indices' in stats:
                    mpn_indices = stats['mpn_replaced_indices']
                    if mpn_indices:
                        f.write(f"  MPN Layers:\n")
                        for layer_name, indices in sorted(mpn_indices.items()):
                            if indices:
                                f.write(f"    {layer_name}: {indices}\n")
            
            f.write(f"{'='*60}\n\n")
        
        # Console output key information
        print(f"💡 CBP Epoch {epoch}: {total_replaced} neurons replaced total")
    
    def save_summary(self):
        """Save training summary"""
        if not self.stats_history:
            return
        
        summary_file = self.log_file.replace('.log', '_summary.json')
        
        # Calculate summary statistics
        total_epochs = len(self.stats_history)
        total_replacements = sum(epoch.get('total_neurons_replaced', 0) for epoch in self.stats_history)
        avg_replacements = total_replacements / total_epochs if total_epochs > 0 else 0
        
        summary = {
            'training_summary': {
                'total_epochs': total_epochs,
                'total_neuron_replacements': total_replacements,
                'average_replacements_per_epoch': avg_replacements,
                'start_time': self.stats_history[0]['timestamp'] if self.stats_history else None,
                'end_time': self.stats_history[-1]['timestamp'] if self.stats_history else None
            },
            'detailed_history': self.stats_history
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 CBP Training Summary saved to: {summary_file}")


# GnTForChemprop class removed - functionality now handled by CBPLinear layers directly


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
        use_adamgnt: bool = False,
    ):
        self.model = model
        self.args = args
        self.device = args.device

        # Collect all CBPLinear layers - if model doesn't have the method, CBP might not be enabled
        if hasattr(model, 'get_all_cbp_layers'):
            self.cbp_layers = model.get_all_cbp_layers()
            print(f"🔥 Found {len(self.cbp_layers)} CBPLinear layers in the model")

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

        # Setup logging
        self.enable_cbp_logging = enable_cbp_logging and len(self.cbp_layers) > 0
        if self.enable_cbp_logging:
            self.cbp_logger = CBPLogger(log_dir=log_dir)
            self.ffn_stats_buffer = []
            self.mpn_stats_buffer = []
            self._setup_cbp_logging()
        else:
            self.cbp_logger = None
            self.ffn_stats_buffer = []
            self.mpn_stats_buffer = []

        # Setup optimizer
        if use_adamgnt:
            self.optimizer = AdamGnT(
                model.parameters(),
                lr=step_size,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'adam':
            self.optimizer = Adam(
                model.parameters(),
                lr=step_size,
                betas=(0.9, 0.999),
                weight_decay=args.weight_decay
            )
        else:
            self.optimizer = SGD(
                model.parameters(),
                lr=step_size,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
    
    def _configure_cbp_layers(self, replacement_rate, decay_rate, maturity_threshold, util_type, accumulate):
        """Configure all CBPLinear layers with unified parameters."""
        for i, cbp_layer in enumerate(self.cbp_layers):
            # Update CBP parameters
            cbp_layer.replacement_rate = replacement_rate
            cbp_layer.decay_rate = decay_rate
            cbp_layer.maturity_threshold = maturity_threshold
            cbp_layer.util_type = util_type
            cbp_layer.accumulate = accumulate

            # Set layer name if not already set
            if not hasattr(cbp_layer, 'layer_name') or cbp_layer.layer_name is None:
                if cbp_layer.layer_type == 'FFN':
                    cbp_layer.layer_name = f'FFN_Layer{i}'
                else:
                    cbp_layer.layer_name = f'MPN_Layer{i}'

    def _setup_cbp_logging(self):
        """Setup logging for all CBPLinear layers."""
        for cbp_layer in self.cbp_layers:
            cbp_layer.cbp_logger = self.cbp_logger
            cbp_layer._batch_counter = 0
            cbp_layer._trainer = self  # Reference to trainer for stats collection

    def train_step(self,
                   batch_data: Dict = None,
                   targets: torch.Tensor = None,
                   mask: torch.Tensor = None,
                   target_weights: torch.Tensor = None,
                   data_weights: torch.Tensor = None,
                   loss_func: Callable = None,
                   args: TrainArgs = None,
                   lt_targets: torch.Tensor = None,
                   gt_targets: torch.Tensor = None,
                   # Legacy simple interface
                   batch=None) -> float:
        """
        Unified training step that supports both simple and advanced interfaces.

        Simple interface (legacy):
            train_step(batch=(mol_batch, features_batch), targets=targets)

        Advanced interface:
            train_step(batch_data=..., targets=..., mask=..., ...)

        :param batch_data: Dictionary containing all batch data
        :param targets: Target values
        :param mask: Mask for valid targets
        :param target_weights: Weights for different targets
        :param data_weights: Weights for different samples
        :param loss_func: Loss function to use
        :param args: Training arguments (uses self.args if not provided)
        :param lt_targets: Lower bound targets (for bounded_mse)
        :param gt_targets: Upper bound targets (for bounded_mse)
        :param batch: Legacy tuple of (mol_batch, features_batch)
        :return: Loss value
        """
        # Handle legacy simple interface
        if batch is not None and batch_data is None:
            mol_batch, features_batch = batch
            batch_data = {
                'mol_batch': mol_batch,
                'features_batch': features_batch,
                'atom_descriptors_batch': None,
                'atom_features_batch': None,
                'bond_features_batch': None
            }

        # Use default args if not provided
        if args is None:
            args = self.args

        # Create default mask and weights if not provided
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        if target_weights is None:
            num_tasks = targets.shape[1] if len(targets.shape) > 1 else 1
            target_weights = torch.ones(1, num_tasks)
        if data_weights is None:
            batch_size = targets.shape[0]
            data_weights = torch.ones(batch_size, 1)

        # Create default loss function if not provided
        if loss_func is None:
            if args.dataset_type == 'classification':
                loss_func = nn.BCEWithLogitsLoss(reduction='none')
            elif args.dataset_type == 'multiclass':
                loss_func = nn.CrossEntropyLoss(reduction='none')
            else:
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

        # Move tensors to correct device - 🔥 Keep original device transfer logic
        torch_device = preds.device
        mask = mask.to(torch_device)
        targets = targets.to(torch_device)
        target_weights = target_weights.to(torch_device)
        data_weights = data_weights.to(torch_device)
        if lt_targets is not None:
            lt_targets = lt_targets.to(torch_device)
        if gt_targets is not None:
            gt_targets = gt_targets.to(torch_device)

        # 🔥 Complete loss function handling logic - identical to original train.py
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

        # 🔥 Standard backpropagation step
        self.optimizer.zero_grad()
        loss.backward()

        # 🔥 Keep gradient clipping feature
        if args.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)

        self.optimizer.step()

        # 🔥 CBPLinear layers handle neuron replacement automatically through their hooks
        # No need for manual GnT step - the backward hooks trigger replacement
        
        return loss.item()

    # Note: Batch-level logging is now handled directly by CBPLinear layers
    # through their _log_replacement_stats method

    def log_epoch_cbp_stats(self, epoch: int):
        """Aggregate CBP statistics for current epoch from all CBPLinear layers."""
        if not self.enable_cbp_logging:
            return

        epoch_stats = {
            'total_neurons_replaced': 0,
            'ffn_replaced_indices': {},
            'mpn_replaced_indices': {}
        }

        # Aggregate FFN statistics
        if self.ffn_stats_buffer:
            for stats in self.ffn_stats_buffer:
                layer_name = stats.get('layer_name', 'Unknown')
                indices = stats.get('replaced_indices', [])
                if indices:
                    if layer_name not in epoch_stats['ffn_replaced_indices']:
                        epoch_stats['ffn_replaced_indices'][layer_name] = []
                    epoch_stats['ffn_replaced_indices'][layer_name].extend(indices)
                    epoch_stats['total_neurons_replaced'] += stats.get('num_replaced', 0)

        # Aggregate MPN statistics
        if self.mpn_stats_buffer:
            for stats in self.mpn_stats_buffer:
                layer_name = stats.get('layer_name', 'Unknown')
                indices = stats.get('replaced_indices', [])
                if indices:
                    if layer_name not in epoch_stats['mpn_replaced_indices']:
                        epoch_stats['mpn_replaced_indices'][layer_name] = []
                    epoch_stats['mpn_replaced_indices'][layer_name].extend(indices)
                    epoch_stats['total_neurons_replaced'] += stats.get('num_replaced', 0)

        # Log epoch-level summary
        if self.cbp_logger:
            self.cbp_logger.log_epoch_summary(epoch, epoch_stats)

        # Clear buffers for next epoch
        self.ffn_stats_buffer.clear()
        self.mpn_stats_buffer.clear()
    
    # Simplified aggregation methods are no longer needed since we aggregate directly
    # in log_epoch_cbp_stats. The CBPLinear layers handle their own statistics.
    
    def save_cbp_summary(self):
        """Save cbp training summary"""
        if self.enable_cbp_logging and self.cbp_logger:
            self.cbp_logger.save_summary() 