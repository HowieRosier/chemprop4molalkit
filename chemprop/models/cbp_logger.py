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

        # Initialize log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.log_dir / f"cbp_training_{timestamp}.log"
        self.log_file = None
        self._open_log_file()

        # Initialize counters
        self.batch_counter = 0
        self.epoch_counter = 0
        self.total_replacements = 0

        # Epoch tracking for summary
        self.epoch_batch_count = 0
        self.epoch_active_batches = 0
        self.epoch_replacements_by_layer = {}  # Track replacements per layer per epoch

        # Storage for raw data
        self.gradient_history = {}  # Layer -> batch -> raw gradients
        self.utility_history = {}   # Layer -> batch -> raw utilities
        self.age_history = {}       # Layer -> batch -> raw ages
        self.replacement_history = {}  # Layer -> replacement events

        # Current epoch data for per-epoch storage
        self.current_epoch_data = {
            'gradients': {},  # Layer -> list of gradient snapshots
            'utilities': {},  # Layer -> list of utility snapshots
            'activations': {}  # Layer -> list of activation snapshots
        }

        # All epochs data for consolidated storage
        self.all_epochs_data = []

        # Final epoch data for separate log
        self.final_epoch_data = None

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

    def _open_log_file(self):
        """Open the log file for writing."""
        try:
            self.log_file = open(self.log_file_path, 'w', buffering=1)  # Line buffering
            self._write_log_header()
        except Exception as e:
            print(f"Warning: Could not open log file {self.log_file_path}: {e}")
            self.log_file = None

    def _write_log_header(self):
        """Write the header information to the log file."""
        if self.log_file:
            self.log_file.write(f"CBP Training Log - Started at {datetime.now().isoformat()}\n")
            self.log_file.write("=" * 60 + "\n")
            self.log_file.write("This log tracks neuron replacement events during CBP training\n")
            self.log_file.write("=" * 60 + "\n\n")

    def _write_to_log(self, message: str):
        """Write a message to the log file."""
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()
        elif self.log_file and self.log_file.closed:
            # Reopen the log file if it was closed
            try:
                self.log_file = open(self.log_file_path, 'a', buffering=1)  # Append mode
                self.log_file.write(message)
                self.log_file.flush()
            except Exception as e:
                print(f"Warning: Could not reopen log file {self.log_file_path}: {e}")

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

        # Track batch count for epoch
        if batch_idx > self.epoch_batch_count:
            self.epoch_batch_count = batch_idx

        # Convert to CPU and numpy for storage
        grad_np = gradients.detach().cpu().numpy() if torch.is_tensor(gradients) else gradients
        util_np = utilities.detach().cpu().numpy() if torch.is_tensor(utilities) else utilities
        age_np = ages.detach().cpu().numpy() if torch.is_tensor(ages) else ages

        # Initialize storage for this layer if needed
        if layer_name not in self.gradient_history:
            self.gradient_history[layer_name] = []
            self.utility_history[layer_name] = []
            self.age_history[layer_name] = []

        # Store raw values only
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

        # Also store in current epoch data
        if layer_name not in self.current_epoch_data['gradients']:
            self.current_epoch_data['gradients'][layer_name] = []
            self.current_epoch_data['utilities'][layer_name] = []

        self.current_epoch_data['gradients'][layer_name].append({
            'batch': batch_idx,
            'values': grad_np.tolist()
        })

        self.current_epoch_data['utilities'][layer_name].append({
            'batch': batch_idx,
            'values': util_np.tolist()
        })

        # Log to W&B if enabled
        if self.wandb and batch_idx % self.log_frequency == 0:
            self._log_to_wandb(layer_name, grad_np, util_np, age_np)

    def log_activations(self,
                       layer_name: str,
                       activations: torch.Tensor,
                       batch_idx: int,
                       epoch: int):
        """
        Log activation values for a specific layer.

        Args:
            layer_name: Name of the layer
            activations: Tensor of activation values
            batch_idx: Current batch index
            epoch: Current epoch
        """
        # Convert to CPU and numpy for storage
        act_np = activations.detach().cpu().numpy() if torch.is_tensor(activations) else activations

        # Store in current epoch data
        if layer_name not in self.current_epoch_data['activations']:
            self.current_epoch_data['activations'][layer_name] = []

        self.current_epoch_data['activations'][layer_name].append({
            'batch': batch_idx,
            'values': act_np.tolist()
        })

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

        batch_id = batch_idx or self.batch_counter
        epoch_id = epoch or self.epoch_counter

        event = {
            'batch': batch_id,
            'epoch': epoch_id,
            'indices': replaced_indices,
            'count': len(replaced_indices),
            'rate': replacement_rate,
            'timestamp': datetime.now().isoformat()
        }

        self.replacement_history[layer_name].append(event)
        self.total_replacements += len(replaced_indices)

        # Track epoch-level replacements
        if layer_name not in self.epoch_replacements_by_layer:
            self.epoch_replacements_by_layer[layer_name] = []
        self.epoch_replacements_by_layer[layer_name].extend(replaced_indices)

        # Print real-time console output for replacements
        if len(replaced_indices) > 0:
            # Determine layer type for display
            if "FFN" in layer_name or "ffn" in layer_name.lower():
                layer_type = "[FFN]"
            elif "MPN" in layer_name or "mpn" in layer_name.lower():
                layer_type = "[MPN]"
            else:
                layer_type = ""

            # Format the indices list (show first 10 if too many)
            if len(replaced_indices) > 10:
                indices_str = str(replaced_indices[:10])[:-1] + ", ...]"
            else:
                indices_str = str(replaced_indices)

            # Print the real-time replacement notification
            print(f"🔥 Batch {batch_id} {layer_type}: {len(replaced_indices)} neurons replaced [{layer_name}:{indices_str}]")

            # Also write to log file
            self._log_batch_replacement(batch_id, layer_name, replaced_indices, epoch_id)

        # Log to W&B
        if self.wandb:
            self.wandb.log({
                f"{layer_name}/replacements": len(replaced_indices),
                f"{layer_name}/replacement_rate": replacement_rate,
                "total_replacements": self.total_replacements
            })

    def _log_batch_replacement(self, batch_id: int, layer_name: str, indices: List[int], epoch: int):
        """Write batch-level replacement event to log file."""
        if not self.log_file:
            return

        # Increment active batch count for this epoch
        self.epoch_active_batches += 1

        # Determine layer type (FFN or MPN)
        layer_type = "[FFN]" if "FFN" in layer_name else "[MPN]"

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write batch header
        message = f"\nBatch {batch_id} ({timestamp})\n"
        message += f"  {layer_type} {len(indices)} neurons replaced\n"
        message += f"  └─ {layer_name}: {indices}\n"

        self._write_to_log(message)

    def _write_epoch_summary_to_log(self, epoch: int, summary: dict):
        """Write epoch summary to log file."""
        if not self.log_file:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create the epoch summary header
        message = "\n" + "=" * 60 + "\n"
        message += f"EPOCH {epoch} SUMMARY - {timestamp}\n"
        message += "=" * 60 + "\n"

        # Get epoch-specific replacement count
        epoch_replacement_count = 0
        if summary['cbp_stats']['replacement_count'] is not None:
            epoch_replacement_count = summary['cbp_stats']['replacement_count']

        message += f"🔥 Total neurons replaced this epoch: {epoch_replacement_count}\n"

        # Layer-wise replacements
        layer_replacements = {}
        for layer_name, events in self.replacement_history.items():
            epoch_events = [e for e in events if e['epoch'] == epoch]
            if epoch_events:
                total_for_layer = sum(e['count'] for e in epoch_events)
                layer_replacements[layer_name] = total_for_layer

        if layer_replacements:
            message += f"Layer-wise replacements: {layer_replacements}\n"

        # Average replacement rates
        if summary['cbp_stats']['replacement_rate'] is not None:
            message += f"Average replacement rate: {summary['cbp_stats']['replacement_rate']:.6f}\n"

        # Average utilities - calculate from current data
        if self.current_epoch_data['utilities']:
            avg_utils = {}
            for layer_name, util_data in self.current_epoch_data['utilities'].items():
                if util_data and util_data[-1]['values']:  # Use last batch data
                    avg_utils[layer_name] = np.mean(util_data[-1]['values'])
            if avg_utils:
                message += f"Average utilities: {{{', '.join([f'{k}: {v:.4f}' for k, v in avg_utils.items()])}}}\n"

        # Age statistics
        age_stats = {
            'mean_age': summary['cbp_stats']['avg_neuron_age'],
            'mature_neurons': summary['cbp_stats']['mature_neurons']
        }
        message += f"Age statistics: {age_stats}\n"

        # Active batches (with replacements) vs total batches
        message += f"Active batches (with replacements): {self.epoch_active_batches}\n"
        message += f"Total batches processed: {self.batch_counter}\n"

        # Replaced Neuron Indices Summary
        message += "\n📍 Replaced Neuron Indices Summary:\n"

        # Separate FFN and MPN layers
        ffn_layers = {}
        mpn_layers = {}

        for layer_name, indices in self.epoch_replacements_by_layer.items():
            if indices:  # Only include layers with replacements
                unique_indices = sorted(list(set(indices)))  # Remove duplicates and sort
                if "FFN" in layer_name:
                    ffn_layers[layer_name] = unique_indices
                else:
                    mpn_layers[layer_name] = unique_indices

        # Display FFN layers
        if ffn_layers:
            message += "  FFN Layers:\n"
            for layer_name, indices in sorted(ffn_layers.items()):
                # Truncate if too many indices
                if len(indices) > 20:
                    display_indices = str(indices[:20])[:-1] + ", ...]"
                else:
                    display_indices = str(indices)
                message += f"    {layer_name}: {display_indices}\n"

        # Display MPN layers
        if mpn_layers:
            message += "  MPN Layers:\n"
            for layer_name, indices in sorted(mpn_layers.items()):
                # Truncate if too many indices
                if len(indices) > 20:
                    display_indices = str(indices[:20])[:-1] + ", ...]"
                else:
                    display_indices = str(indices)
                message += f"    {layer_name}: {display_indices}\n"

        message += "=" * 60 + "\n"

        self._write_to_log(message)

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
        """Save complete neuron-level data for the epoch."""
        # Prepare the complete epoch data
        summary = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'total_replacements': self.total_replacements,
            'layers_tracked': list(self.current_epoch_data['gradients'].keys()),
            'cbp_stats': {
                'replacement_count': next((s['value'] for s in self.cbp_stats['replacement_count_per_epoch'] if s['epoch'] == epoch), None),
                'replacement_rate': next((s['value'] for s in self.cbp_stats['replacement_rate_per_epoch'] if s['epoch'] == epoch), None),
                'avg_neuron_age': next((s['value'] for s in self.cbp_stats['avg_neuron_age_per_epoch'] if s['epoch'] == epoch), None),
                'mature_neurons': next((s['value'] for s in self.cbp_stats['mature_neurons_per_epoch'] if s['epoch'] == epoch), None)
            },
            # Add the complete neuron-level data for this epoch
            'neuron_data': {
                'gradients': self.current_epoch_data['gradients'],
                'utilities': self.current_epoch_data['utilities'],
                'activations': self.current_epoch_data['activations']
            }
        }

        # Calculate epoch-specific replacement count
        epoch_replacement_count = 0
        for layer_name, events in self.replacement_history.items():
            epoch_events = [e for e in events if e['epoch'] == epoch]
            epoch_replacement_count += sum(e['count'] for e in epoch_events)

        # Print epoch summary to console
        print(f"💡 CBP Epoch {epoch}: {epoch_replacement_count} neurons replaced total")

        # Append to all epochs data
        self.all_epochs_data.append(summary)

        # Save consolidated epochs summary
        epochs_summary_path = self.log_dir / "epochs_summary.json"
        with open(epochs_summary_path, 'w') as f:
            json.dump({
                'total_epochs': epoch + 1,
                'last_updated': datetime.now().isoformat(),
                'epochs': self.all_epochs_data
            }, f, indent=2)

        # Write epoch summary to log file
        self._write_epoch_summary_to_log(epoch, summary)

        # Store final epoch data for potential final_epoch.log
        self.final_epoch_data = summary.copy()

        # Clear current epoch data for next epoch
        self.reset_epoch_data()

        # Reset epoch counters
        self.epoch_batch_count = 0
        self.epoch_active_batches = 0
        self.epoch_replacements_by_layer = {}

        return summary

    def reset_epoch_data(self):
        """Reset current epoch data for the next epoch."""
        self.current_epoch_data = {
            'gradients': {},
            'utilities': {},
            'activations': {}
        }

    def mark_iteration_start(self, iteration: int):
        """Mark the start of a new active learning iteration in the unified log.

        Args:
            iteration: The iteration number to mark

        This creates a clear visual separator in the log file to show where
        each active learning iteration begins.
        """
        # Write iteration separator to log file
        if self.log_file:
            separator = "\n" + "=" * 80 + "\n"
            message = separator
            message += f"ACTIVE LEARNING ITERATION {iteration} STARTING\n"
            message += f"Timestamp: {datetime.now().isoformat()}\n"
            message += separator + "\n"
            self._write_to_log(message)

        # Print to console for visibility
        print(f"\n{'=' * 60}")
        print(f"🔄 Starting Active Learning Iteration {iteration}")
        print(f"{'=' * 60}\n")

    def save_full_history(self):
        """Save the complete CBP training summary to disk."""
        # Save history at cbp_logs level
        base_log_dir = self.log_dir

        # Create summary file with timestamp (one per training session)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = base_log_dir / f"cbp_training_summary_{timestamp}.json"

        # Create training summary section
        training_summary = {
            'total_epochs': self.epoch_counter + 1,
            'total_replacements': self.total_replacements,
            'layers_tracked': list(set(
                list(self.gradient_history.keys()) +
                list(self.utility_history.keys()) +
                list(self.replacement_history.keys())
            )),
            'final_metrics': {
                'loss': self.training_metrics['loss_history'][-1]['loss'] if self.training_metrics['loss_history'] else None,
                'accuracy': self.training_metrics['accuracy_history'][-1]['accuracy'] if self.training_metrics['accuracy_history'] else None,
                'auc': self.training_metrics['auc_history'][-1]['auc'] if self.training_metrics['auc_history'] else None
            },
            'cbp_statistics': {
                'total_replacement_events': sum(len(events) for events in self.replacement_history.values()),
                'avg_replacement_rate': np.mean([s['value'] for s in self.cbp_stats['replacement_rate_per_epoch']]) if self.cbp_stats['replacement_rate_per_epoch'] else 0,
                'final_avg_neuron_age': self.cbp_stats['avg_neuron_age_per_epoch'][-1]['value'] if self.cbp_stats['avg_neuron_age_per_epoch'] else None,
                'final_mature_neurons': self.cbp_stats['mature_neurons_per_epoch'][-1]['value'] if self.cbp_stats['mature_neurons_per_epoch'] else None
            }
        }

        # Create detailed history section (per-epoch summaries without neuron-level data)
        detailed_history = []
        for epoch in range(self.epoch_counter + 1):
            epoch_summary = {
                'epoch': epoch,
                'metrics': {},
                'cbp_stats': {},
                'replacement_events': {}
            }

            # Add training metrics for this epoch
            for loss_entry in self.training_metrics['loss_history']:
                if loss_entry['epoch'] == epoch:
                    epoch_summary['metrics']['loss'] = loss_entry['loss']
                    break

            for acc_entry in self.training_metrics['accuracy_history']:
                if acc_entry['epoch'] == epoch:
                    epoch_summary['metrics']['accuracy'] = acc_entry['accuracy']
                    break

            for auc_entry in self.training_metrics['auc_history']:
                if auc_entry['epoch'] == epoch:
                    epoch_summary['metrics']['auc'] = auc_entry['auc']
                    break

            # Add CBP stats for this epoch
            for stat_entry in self.cbp_stats['replacement_count_per_epoch']:
                if stat_entry['epoch'] == epoch:
                    epoch_summary['cbp_stats']['replacement_count'] = stat_entry['value']
                    break

            for stat_entry in self.cbp_stats['replacement_rate_per_epoch']:
                if stat_entry['epoch'] == epoch:
                    epoch_summary['cbp_stats']['replacement_rate'] = stat_entry['value']
                    break

            for stat_entry in self.cbp_stats['avg_neuron_age_per_epoch']:
                if stat_entry['epoch'] == epoch:
                    epoch_summary['cbp_stats']['avg_neuron_age'] = stat_entry['value']
                    break

            for stat_entry in self.cbp_stats['mature_neurons_per_epoch']:
                if stat_entry['epoch'] == epoch:
                    epoch_summary['cbp_stats']['mature_neurons'] = stat_entry['value']
                    break

            # Add replacement events summary for this epoch
            for layer_name, events in self.replacement_history.items():
                epoch_events = [e for e in events if e['epoch'] == epoch]
                if epoch_events:
                    epoch_summary['replacement_events'][layer_name] = {
                        'total_replacements': sum(e['count'] for e in epoch_events),
                        'num_events': len(epoch_events)
                    }

            # Only add if we have data for this epoch
            if epoch_summary['metrics'] or epoch_summary['cbp_stats'] or epoch_summary['replacement_events']:
                detailed_history.append(epoch_summary)

        # Create complete summary (no longer iteration-specific)
        complete_summary = {
            'timestamp': datetime.now().isoformat(),
            'training_summary': training_summary,
            'detailed_history': detailed_history
        }

        # Save the data
        with open(history_path, 'w') as f:
            json.dump(complete_summary, f, indent=2)

        print(f"CBP training summary saved to {history_path}")
        return history_path

    def get_layer_replacement_count(self, layer_name: str) -> int:
        """Get total replacement count for a specific layer."""
        if layer_name not in self.replacement_history:
            return 0
        return sum(event['count'] for event in self.replacement_history[layer_name])

    def get_total_replacements(self) -> int:
        """Get total replacements across all layers."""
        return self.total_replacements

    def save_iteration_summary(self):
        """Save iteration summary including final epoch data - called at end of each iteration."""
        # Save the full history (creates cbp_training_summary_TIMESTAMP.json)
        self.save_full_history()

        # Also save the final epoch log
        self.save_final_epoch_log()

        print(f"📊 Iteration summary saved to {self.log_dir}")

    def save_final_epoch_log(self):
        """Save the final epoch's neuron-level data to a separate log file."""
        if not hasattr(self, 'final_epoch_data') or not self.final_epoch_data:
            print("No final epoch data to save")
            return

        final_log_path = self.log_dir / "final_epoch.log"

        with open(final_log_path, 'w') as f:
            f.write(f"Final Epoch Neuron-Level Data\n")
            f.write(f"Epoch: {self.final_epoch_data['epoch']}\n")
            f.write(f"Timestamp: {self.final_epoch_data['timestamp']}\n")
            f.write("=" * 80 + "\n\n")

            neuron_data = self.final_epoch_data.get('neuron_data', {})

            # Write gradients for each layer
            if neuron_data.get('gradients'):
                f.write("GRADIENTS:\n")
                f.write("-" * 40 + "\n")
                for layer_name, grad_snapshots in neuron_data['gradients'].items():
                    f.write(f"\nLayer: {layer_name}\n")
                    if grad_snapshots:
                        # Use the last batch's data
                        last_snapshot = grad_snapshots[-1]
                        f.write(f"  Batch: {last_snapshot['batch']}\n")
                        f.write(f"  Values: {last_snapshot['values']}\n")
                f.write("\n")

            # Write utilities for each layer
            if neuron_data.get('utilities'):
                f.write("UTILITIES:\n")
                f.write("-" * 40 + "\n")
                for layer_name, util_snapshots in neuron_data['utilities'].items():
                    f.write(f"\nLayer: {layer_name}\n")
                    if util_snapshots:
                        # Use the last batch's data
                        last_snapshot = util_snapshots[-1]
                        f.write(f"  Batch: {last_snapshot['batch']}\n")
                        f.write(f"  Values: {last_snapshot['values']}\n")
                f.write("\n")

            # Write activations for each layer
            if neuron_data.get('activations'):
                f.write("ACTIVATIONS:\n")
                f.write("-" * 40 + "\n")
                for layer_name, act_snapshots in neuron_data['activations'].items():
                    f.write(f"\nLayer: {layer_name}\n")
                    if act_snapshots:
                        # Use the last batch's data
                        last_snapshot = act_snapshots[-1]
                        f.write(f"  Batch: {last_snapshot['batch']}\n")
                        f.write(f"  Values: {last_snapshot['values']}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("End of Final Epoch Data\n")

        print(f"Final epoch log saved to {final_log_path}")


    def close(self):
        """Close the logger and save final statistics."""
        self.save_full_history()

        # Save final epoch log
        self.save_final_epoch_log()

        # No need for separate master log since we're using a single log file

        # Close log file
        if self.log_file:
            self._write_to_log(f"\n\nTraining completed at {datetime.now().isoformat()}\n")
            self._write_to_log("=" * 60 + "\n")
            self.log_file.close()
            print(f"CBP training log saved to {self.log_file_path}")

        if self.wandb:
            self.wandb.finish()