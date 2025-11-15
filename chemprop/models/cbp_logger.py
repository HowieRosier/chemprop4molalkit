"""
CBP Logger for Continual Backpropagation Training
==================================================
This module provides unified logging functionality for CBP training,
including gradients, utilities, replacements, and training statistics.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class CBPLogger:
    """
    Unified logger for CBP training that tracks all training information.

    This logger consolidates:
    - Per-neuron gradient raw values
    - Utility values over time
    - Neuron ages and maturity
    - Replacement events and statistics
    - Training metrics (loss, accuracy, etc.)
    - CBP-specific statistics
    """

    def __init__(self,
                 log_dir: str = "cbp_logs",
                 log_frequency: int = 100,
                 enable_wandb: bool = False,
                 wandb_project: str = "cbp-training",
                 wandb_entity: Optional[str] = None,
                 track_histogram: bool = True):
        """
        Initialize the CBP logger.

        Args:
            log_dir: Directory to save log files
            log_frequency: How often to log (in batches)
            enable_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
            wandb_entity: W&B entity/username
            track_histogram: Whether to track histograms in W&B
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_frequency = log_frequency
        self.track_histogram = track_histogram

        # Initialize counters
        self.batch_counter = 0
        self.epoch_counter = 0
        self.total_replacements = 0

        # Storage for raw data (no statistics, user will analyze)
        self.gradient_history = {}  # Layer -> batch -> raw gradients
        self.utility_history = {}   # Layer -> batch -> raw utilities
        self.age_history = {}       # Layer -> batch -> raw ages
        self.replacement_history = {}  # Layer -> replacement events

        # CBP training statistics (from cbp_trainer)
        self.cbp_stats = {
            'replacement_count_per_epoch': [],
            'replacement_rate_per_epoch': [],
            'avg_neuron_age_per_epoch': [],
            'mature_neurons_per_epoch': []
        }

        # Training metrics
        self.training_metrics = {
            'loss_history': [],
            'accuracy_history': [],
            'auc_history': []
        }

        # Initialize W&B if enabled
        self.wandb = None
        if enable_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config={
                        "log_frequency": log_frequency,
                        "track_histogram": track_histogram
                    },
                    reinit=True
                )
            except ImportError:
                print("Warning: wandb not installed. Install with 'pip install wandb'")
                self.wandb = None

    def log_gradients(self,
                     layer_name: str,
                     gradients: torch.Tensor,
                     utilities: torch.Tensor,
                     ages: torch.Tensor,
                     batch_idx: int,
                     epoch: int):
        """
        Log raw gradient, utility, and age data for a specific layer.

        Args:
            layer_name: Name of the layer
            gradients: Tensor of gradient values (raw, no statistics)
            utilities: Tensor of utility values
            ages: Tensor of neuron ages
            batch_idx: Current batch index
            epoch: Current epoch
        """
        self.batch_counter = batch_idx
        self.epoch_counter = epoch

        # Convert to CPU and numpy for storage
        grad_np = gradients.detach().cpu().numpy() if torch.is_tensor(gradients) else gradients
        util_np = utilities.detach().cpu().numpy() if torch.is_tensor(utilities) else utilities
        age_np = ages.detach().cpu().numpy() if torch.is_tensor(ages) else ages

        # Initialize storage for this layer if needed
        if layer_name not in self.gradient_history:
            self.gradient_history[layer_name] = []
            self.utility_history[layer_name] = []
            self.age_history[layer_name] = []

        # Store raw values only (user will analyze)
        self.gradient_history[layer_name].append({
            'batch': batch_idx,
            'epoch': epoch,
            'values': grad_np.tolist()  # Store all raw values
        })

        self.utility_history[layer_name].append({
            'batch': batch_idx,
            'epoch': epoch,
            'values': util_np.tolist()  # Store all raw values
        })

        self.age_history[layer_name].append({
            'batch': batch_idx,
            'epoch': epoch,
            'values': age_np.tolist()  # Store all raw values
        })

        # Log to W&B if enabled (compute statistics only for visualization)
        if self.wandb and batch_idx % self.log_frequency == 0:
            self._log_to_wandb(layer_name, grad_np, util_np, age_np)

    def log_replacement_event(self,
                            layer_name: str,
                            replaced_indices: List[int],
                            replacement_rate: float,
                            batch_idx: int = None,
                            epoch: int = None):
        """
        Log a neuron replacement event.

        Args:
            layer_name: Name of the layer
            replaced_indices: Indices of replaced neurons
            replacement_rate: The replacement rate used
            batch_idx: Current batch index
            epoch: Current epoch
        """
        if layer_name not in self.replacement_history:
            self.replacement_history[layer_name] = []

        event = {
            'batch': batch_idx or self.batch_counter,
            'epoch': epoch or self.epoch_counter,
            'indices': replaced_indices,
            'count': len(replaced_indices),
            'rate': replacement_rate,
            'timestamp': datetime.now().isoformat()
        }

        self.replacement_history[layer_name].append(event)
        self.total_replacements += len(replaced_indices)

        # Log to W&B
        if self.wandb:
            self.wandb.log({
                f"{layer_name}/replacements": len(replaced_indices),
                f"{layer_name}/replacement_rate": replacement_rate,
                "total_replacements": self.total_replacements
            })

    def log_training_metrics(self,
                            loss: float = None,
                            accuracy: float = None,
                            auc: float = None,
                            batch_idx: int = None,
                            epoch: int = None):
        """
        Log training metrics like loss, accuracy, AUC.

        Args:
            loss: Training loss value
            accuracy: Training accuracy
            auc: Area under curve
            batch_idx: Current batch index
            epoch: Current epoch
        """
        metrics = {
            'batch': batch_idx or self.batch_counter,
            'epoch': epoch or self.epoch_counter,
            'timestamp': datetime.now().isoformat()
        }

        if loss is not None:
            metrics['loss'] = loss
            self.training_metrics['loss_history'].append(metrics.copy())

        if accuracy is not None:
            metrics['accuracy'] = accuracy
            self.training_metrics['accuracy_history'].append(metrics.copy())

        if auc is not None:
            metrics['auc'] = auc
            self.training_metrics['auc_history'].append(metrics.copy())

        # Log to W&B
        if self.wandb:
            wandb_metrics = {}
            if loss is not None:
                wandb_metrics['train/loss'] = loss
            if accuracy is not None:
                wandb_metrics['train/accuracy'] = accuracy
            if auc is not None:
                wandb_metrics['train/auc'] = auc

            if wandb_metrics:
                self.wandb.log(wandb_metrics)

    def log_cbp_stats(self,
                     replacement_count: int = None,
                     replacement_rate: float = None,
                     avg_neuron_age: float = None,
                     mature_neurons: int = None,
                     epoch: int = None):
        """
        Log CBP-specific statistics per epoch.

        Args:
            replacement_count: Total replacements this epoch
            replacement_rate: Average replacement rate
            avg_neuron_age: Average age of neurons
            mature_neurons: Count of mature neurons
            epoch: Current epoch
        """
        epoch_idx = epoch or self.epoch_counter

        if replacement_count is not None:
            self.cbp_stats['replacement_count_per_epoch'].append({
                'epoch': epoch_idx,
                'value': replacement_count
            })

        if replacement_rate is not None:
            self.cbp_stats['replacement_rate_per_epoch'].append({
                'epoch': epoch_idx,
                'value': replacement_rate
            })

        if avg_neuron_age is not None:
            self.cbp_stats['avg_neuron_age_per_epoch'].append({
                'epoch': epoch_idx,
                'value': avg_neuron_age
            })

        if mature_neurons is not None:
            self.cbp_stats['mature_neurons_per_epoch'].append({
                'epoch': epoch_idx,
                'value': mature_neurons
            })

        # Log to W&B
        if self.wandb:
            wandb_stats = {}
            if replacement_count is not None:
                wandb_stats['cbp/replacement_count'] = replacement_count
            if replacement_rate is not None:
                wandb_stats['cbp/replacement_rate'] = replacement_rate
            if avg_neuron_age is not None:
                wandb_stats['cbp/avg_neuron_age'] = avg_neuron_age
            if mature_neurons is not None:
                wandb_stats['cbp/mature_neurons'] = mature_neurons

            if wandb_stats:
                self.wandb.log(wandb_stats)

    def _log_to_wandb(self, layer_name: str, gradients: np.ndarray, utilities: np.ndarray, ages: np.ndarray):
        """Log metrics to Weights & Biases (compute stats only for visualization)."""
        if not self.wandb:
            return

        # Compute statistics only for W&B visualization
        self.wandb.log({
            f"{layer_name}/gradient_mean": np.mean(gradients),
            f"{layer_name}/gradient_std": np.std(gradients),
            f"{layer_name}/utility_mean": np.mean(utilities),
            f"{layer_name}/age_mean": np.mean(ages),
            f"{layer_name}/mature_neurons": np.sum(ages > 50)
        })

        # Histograms for visualization
        if self.track_histogram:
            self.wandb.log({
                f"{layer_name}/gradient_hist": self.wandb.Histogram(gradients),
                f"{layer_name}/utility_hist": self.wandb.Histogram(utilities),
                f"{layer_name}/age_hist": self.wandb.Histogram(ages)
            })

    def save_epoch_summary(self, epoch: int):
        """Save a lightweight summary of the epoch."""
        summary_path = self.log_dir / f"epoch_{epoch}_summary.json"

        summary = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'total_replacements': self.total_replacements,
            'layers_tracked': list(self.gradient_history.keys()),
            'cbp_stats': {
                'replacement_count': next((s['value'] for s in self.cbp_stats['replacement_count_per_epoch'] if s['epoch'] == epoch), None),
                'replacement_rate': next((s['value'] for s in self.cbp_stats['replacement_rate_per_epoch'] if s['epoch'] == epoch), None),
                'avg_neuron_age': next((s['value'] for s in self.cbp_stats['avg_neuron_age_per_epoch'] if s['epoch'] == epoch), None),
                'mature_neurons': next((s['value'] for s in self.cbp_stats['mature_neurons_per_epoch'] if s['epoch'] == epoch), None)
            }
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def save_full_history(self):
        """Save the complete CBP training history to disk."""
        history_path = self.log_dir / f"cbp_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Limit gradient history to save space (keep last 1000 batches per layer)
        limited_gradient_history = {}
        for layer, history in self.gradient_history.items():
            limited_gradient_history[layer] = history[-1000:] if len(history) > 1000 else history

        history = {
            'gradient_history': limited_gradient_history,
            'utility_history': {k: v[-1000:] for k, v in self.utility_history.items()},
            'age_history': {k: v[-1000:] for k, v in self.age_history.items()},
            'replacement_history': self.replacement_history,
            'cbp_stats': self.cbp_stats,
            'training_metrics': self.training_metrics,
            'total_replacements': self.total_replacements
        }

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"CBP history saved to {history_path}")
        return history_path

    def get_layer_replacement_count(self, layer_name: str) -> int:
        """Get total replacement count for a specific layer."""
        if layer_name not in self.replacement_history:
            return 0
        return sum(event['count'] for event in self.replacement_history[layer_name])

    def get_total_replacements(self) -> int:
        """Get total replacements across all layers."""
        return self.total_replacements

    def close(self):
        """Close the logger and save final statistics."""
        self.save_full_history()
        if self.wandb:
            self.wandb.finish()