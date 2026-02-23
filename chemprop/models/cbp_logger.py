"""Unified logger for CBP training: gradients, utilities, replacements, and metrics."""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


class CBPLogger:
    def __init__(self,
                 log_dir: str = "cbp_logs",
                 log_frequency: int = 100,
                 enable_wandb: bool = False,
                 wandb_project: str = "cbp-training",
                 wandb_entity: Optional[str] = None,
                 track_histogram: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_frequency = log_frequency
        self.track_histogram = track_histogram

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.log_dir / f"cbp_training_{timestamp}.log"
        self.log_file = None
        self._open_log_file()

        self.batch_counter = 0
        self.epoch_counter = 0
        self.total_replacements = 0

        self.epoch_batch_count = 0
        self.epoch_active_batches = 0
        self.epoch_replacements_by_layer = {}

        self.gradient_history = {}
        self.utility_history = {}
        self.age_history = {}
        self.replacement_history = {}

        self.current_epoch_data = {
            'gradients': {},
            'utilities': {},
            'activations': {}
        }

        self.all_epochs_data = []
        self.final_epoch_data = None

        self.cbp_stats = {
            'replacement_count_per_epoch': [],
            'replacement_rate_per_epoch': [],
            'avg_neuron_age_per_epoch': [],
            'mature_neurons_per_epoch': []
        }

        self.training_metrics = {
            'loss_history': [],
            'accuracy_history': [],
            'auc_history': []
        }

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
        try:
            self.log_file = open(self.log_file_path, 'w', buffering=1)  # Line buffering
            self._write_log_header()
        except Exception as e:
            print(f"Warning: Could not open log file {self.log_file_path}: {e}")
            self.log_file = None

    def _write_log_header(self):
        if self.log_file:
            self.log_file.write(f"CBP Training Log - Started at {datetime.now().isoformat()}\n")
            self.log_file.write("=" * 60 + "\n")
            self.log_file.write("This log tracks neuron replacement events during CBP training\n")
            self.log_file.write("=" * 60 + "\n\n")

    def _write_to_log(self, message: str):
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()
        elif self.log_file and self.log_file.closed:
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
        self.batch_counter = batch_idx
        self.epoch_counter = epoch

        if batch_idx > self.epoch_batch_count:
            self.epoch_batch_count = batch_idx

        grad_np = gradients.detach().cpu().numpy() if torch.is_tensor(gradients) else gradients
        util_np = utilities.detach().cpu().numpy() if torch.is_tensor(utilities) else utilities
        age_np = ages.detach().cpu().numpy() if torch.is_tensor(ages) else ages

        if layer_name not in self.gradient_history:
            self.gradient_history[layer_name] = []
            self.utility_history[layer_name] = []
            self.age_history[layer_name] = []

        self.gradient_history[layer_name].append({
            'batch': batch_idx,
            'epoch': epoch,
            'values': grad_np.tolist()
        })

        self.utility_history[layer_name].append({
            'batch': batch_idx,
            'epoch': epoch,
            'values': util_np.tolist()
        })

        self.age_history[layer_name].append({
            'batch': batch_idx,
            'epoch': epoch,
            'values': age_np.tolist()
        })

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

        if self.wandb and batch_idx % self.log_frequency == 0:
            self._log_to_wandb(layer_name, grad_np, util_np, age_np)

    def log_activations(self, layer_name: str, activations: torch.Tensor, batch_idx: int, epoch: int):
        act_np = activations.detach().cpu().numpy() if torch.is_tensor(activations) else activations

        if layer_name not in self.current_epoch_data['activations']:
            self.current_epoch_data['activations'][layer_name] = []

        self.current_epoch_data['activations'][layer_name].append({
            'batch': batch_idx,
            'values': act_np.tolist()
        })

    def log_replacement_event(self, layer_name: str, replaced_indices: List[int],
                            replacement_rate: float, batch_idx: int = None, epoch: int = None):
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

        if layer_name not in self.epoch_replacements_by_layer:
            self.epoch_replacements_by_layer[layer_name] = []
        self.epoch_replacements_by_layer[layer_name].extend(replaced_indices)

        if len(replaced_indices) > 0:
            if "FFN" in layer_name or "ffn" in layer_name.lower():
                layer_type = "[FFN]"
            elif "MPN" in layer_name or "mpn" in layer_name.lower():
                layer_type = "[MPN]"
            else:
                layer_type = ""

            if len(replaced_indices) > 10:
                indices_str = str(replaced_indices[:10])[:-1] + ", ...]"
            else:
                indices_str = str(replaced_indices)

            print(f"🔥 Batch {batch_id} {layer_type}: {len(replaced_indices)} neurons replaced [{layer_name}:{indices_str}]")
            self._log_batch_replacement(batch_id, layer_name, replaced_indices, epoch_id)

        if self.wandb:
            self.wandb.log({
                f"{layer_name}/replacements": len(replaced_indices),
                f"{layer_name}/replacement_rate": replacement_rate,
                "total_replacements": self.total_replacements
            })

    def _log_batch_replacement(self, batch_id: int, layer_name: str, indices: List[int], epoch: int):
        if not self.log_file:
            return

        self.epoch_active_batches += 1
        layer_type = "[FFN]" if "FFN" in layer_name else "[MPN]"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = f"\nBatch {batch_id} ({timestamp})\n"
        message += f"  {layer_type} {len(indices)} neurons replaced\n"
        message += f"  └─ {layer_name}: {indices}\n"

        self._write_to_log(message)

    def _write_epoch_summary_to_log(self, epoch: int, summary: dict):
        """Write epoch summary (replacements, utilities, ages) to log file."""
        if not self.log_file:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message = "\n" + "=" * 60 + "\n"
        message += f"EPOCH {epoch} SUMMARY - {timestamp}\n"
        message += "=" * 60 + "\n"

        epoch_replacement_count = 0
        if summary['cbp_stats']['replacement_count'] is not None:
            epoch_replacement_count = summary['cbp_stats']['replacement_count']

        message += f"🔥 Total neurons replaced this epoch: {epoch_replacement_count}\n"

        layer_replacements = {}
        for layer_name, events in self.replacement_history.items():
            epoch_events = [e for e in events if e['epoch'] == epoch]
            if epoch_events:
                total_for_layer = sum(e['count'] for e in epoch_events)
                layer_replacements[layer_name] = total_for_layer

        if layer_replacements:
            message += f"Layer-wise replacements: {layer_replacements}\n"

        if summary['cbp_stats']['replacement_rate'] is not None:
            message += f"Average replacement rate: {summary['cbp_stats']['replacement_rate']:.6f}\n"

        if self.current_epoch_data['utilities']:
            avg_utils = {}
            for layer_name, util_data in self.current_epoch_data['utilities'].items():
                if util_data and util_data[-1]['values']:  # Use last batch data
                    avg_utils[layer_name] = np.mean(util_data[-1]['values'])
            if avg_utils:
                message += f"Average utilities: {{{', '.join([f'{k}: {v:.4f}' for k, v in avg_utils.items()])}}}\n"

        age_stats = {
            'mean_age': summary['cbp_stats']['avg_neuron_age'],
            'mature_neurons': summary['cbp_stats']['mature_neurons']
        }
        message += f"Age statistics: {age_stats}\n"

        message += f"Active batches (with replacements): {self.epoch_active_batches}\n"
        message += f"Total batches processed: {self.batch_counter}\n"

        message += "\n📍 Replaced Neuron Indices Summary:\n"

        ffn_layers = {}
        mpn_layers = {}

        for layer_name, indices in self.epoch_replacements_by_layer.items():
            if indices:
                unique_indices = sorted(list(set(indices)))
                if "FFN" in layer_name:
                    ffn_layers[layer_name] = unique_indices
                else:
                    mpn_layers[layer_name] = unique_indices

        if ffn_layers:
            message += "  FFN Layers:\n"
            for layer_name, indices in sorted(ffn_layers.items()):
                if len(indices) > 20:
                    display_indices = str(indices[:20])[:-1] + ", ...]"
                else:
                    display_indices = str(indices)
                message += f"    {layer_name}: {display_indices}\n"

        if mpn_layers:
            message += "  MPN Layers:\n"
            for layer_name, indices in sorted(mpn_layers.items()):
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
        """Record training metrics (loss/accuracy/auc) and optionally log to W&B."""
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
        """Record per-epoch CBP statistics."""
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
        if not self.wandb:
            return

        self.wandb.log({
            f"{layer_name}/gradient_mean": np.mean(gradients),
            f"{layer_name}/gradient_std": np.std(gradients),
            f"{layer_name}/utility_mean": np.mean(utilities),
            f"{layer_name}/age_mean": np.mean(ages),
            f"{layer_name}/mature_neurons": np.sum(ages > 50)
        })

        if self.track_histogram:
            self.wandb.log({
                f"{layer_name}/gradient_hist": self.wandb.Histogram(gradients),
                f"{layer_name}/utility_hist": self.wandb.Histogram(utilities),
                f"{layer_name}/age_hist": self.wandb.Histogram(ages)
            })

    def save_epoch_summary(self, epoch: int):
        """Save epoch neuron-level data to JSON and log file, then reset for next epoch."""
        def get_last_match(stats_list, epoch_val):
            for s in reversed(stats_list):
                if s['epoch'] == epoch_val:
                    return s['value']
            return None

        summary = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'total_replacements': self.total_replacements,
            'layers_tracked': list(self.current_epoch_data['gradients'].keys()),
            'cbp_stats': {
                'replacement_count': get_last_match(self.cbp_stats['replacement_count_per_epoch'], epoch),
                'replacement_rate': get_last_match(self.cbp_stats['replacement_rate_per_epoch'], epoch),
                'avg_neuron_age': get_last_match(self.cbp_stats['avg_neuron_age_per_epoch'], epoch),
                'mature_neurons': get_last_match(self.cbp_stats['mature_neurons_per_epoch'], epoch)
            },
            'neuron_data': {
                'gradients': self.current_epoch_data['gradients'],
                'utilities': self.current_epoch_data['utilities'],
                'activations': self.current_epoch_data['activations']
            }
        }

        epoch_replacement_count = 0
        for layer_name, events in self.replacement_history.items():
            epoch_events = [e for e in events if e['epoch'] == epoch]
            epoch_replacement_count += sum(e['count'] for e in epoch_events)

        print(f"💡 CBP Epoch {epoch}: {epoch_replacement_count} neurons replaced total")

        self.all_epochs_data.append(summary)

        epochs_summary_path = self.log_dir / "epochs_summary.json"
        with open(epochs_summary_path, 'w') as f:
            json.dump({
                'total_epochs': epoch + 1,
                'last_updated': datetime.now().isoformat(),
                'epochs': self.all_epochs_data
            }, f, indent=2)

        self._write_epoch_summary_to_log(epoch, summary)
        self.final_epoch_data = summary.copy()
        self.reset_epoch_data()

        self.epoch_batch_count = 0
        self.epoch_active_batches = 0
        self.epoch_replacements_by_layer = {}

        return summary

    def reset_epoch_data(self):
        self.current_epoch_data = {
            'gradients': {},
            'utilities': {},
            'activations': {}
        }

    def mark_iteration_start(self, iteration: int):
        """Reset per-iteration counters and write separator to log file."""
        self.epoch_batch_count = 0
        self.epoch_active_batches = 0
        self.epoch_replacements_by_layer = {}
        self.current_epoch_data = {
            'gradients': {},
            'utilities': {},
            'activations': {}
        }

        if self.log_file:
            separator = "\n" + "=" * 80 + "\n"
            message = separator
            message += f"ACTIVE LEARNING ITERATION {iteration} STARTING\n"
            message += f"Timestamp: {datetime.now().isoformat()}\n"
            message += separator + "\n"
            self._write_to_log(message)

        print(f"\n{'=' * 60}")
        print(f"🔄 Starting Active Learning Iteration {iteration}")
        print(f"{'=' * 60}\n")

    def save_full_history(self):
        """Save complete training summary (metrics + CBP stats + per-epoch history) to JSON."""
        base_log_dir = self.log_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = base_log_dir / f"cbp_training_summary_{timestamp}.json"

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

        detailed_history = []
        for epoch in range(self.epoch_counter + 1):
            epoch_summary = {
                'epoch': epoch,
                'metrics': {},
                'cbp_stats': {},
                'replacement_events': {}
            }

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

            for layer_name, events in self.replacement_history.items():
                epoch_events = [e for e in events if e['epoch'] == epoch]
                if epoch_events:
                    epoch_summary['replacement_events'][layer_name] = {
                        'total_replacements': sum(e['count'] for e in epoch_events),
                        'num_events': len(epoch_events)
                    }

            if epoch_summary['metrics'] or epoch_summary['cbp_stats'] or epoch_summary['replacement_events']:
                detailed_history.append(epoch_summary)

        complete_summary = {
            'timestamp': datetime.now().isoformat(),
            'training_summary': training_summary,
            'detailed_history': detailed_history
        }

        with open(history_path, 'w') as f:
            json.dump(complete_summary, f, indent=2)

        print(f"CBP training summary saved to {history_path}")
        return history_path

    def get_total_replacements(self) -> int:
        return self.total_replacements

    def save_final_epoch_log(self):
        """Write final epoch's neuron-level gradients/utilities/activations to separate log."""
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

            if neuron_data.get('gradients'):
                f.write("GRADIENTS:\n")
                f.write("-" * 40 + "\n")
                for layer_name, grad_snapshots in neuron_data['gradients'].items():
                    f.write(f"\nLayer: {layer_name}\n")
                    if grad_snapshots:
                        last_snapshot = grad_snapshots[-1]
                        f.write(f"  Batch: {last_snapshot['batch']}\n")
                        f.write(f"  Values: {last_snapshot['values']}\n")
                f.write("\n")

            if neuron_data.get('utilities'):
                f.write("UTILITIES:\n")
                f.write("-" * 40 + "\n")
                for layer_name, util_snapshots in neuron_data['utilities'].items():
                    f.write(f"\nLayer: {layer_name}\n")
                    if util_snapshots:
                        last_snapshot = util_snapshots[-1]
                        f.write(f"  Batch: {last_snapshot['batch']}\n")
                        f.write(f"  Values: {last_snapshot['values']}\n")
                f.write("\n")

            if neuron_data.get('activations'):
                f.write("ACTIVATIONS:\n")
                f.write("-" * 40 + "\n")
                for layer_name, act_snapshots in neuron_data['activations'].items():
                    f.write(f"\nLayer: {layer_name}\n")
                    if act_snapshots:
                        last_snapshot = act_snapshots[-1]
                        f.write(f"  Batch: {last_snapshot['batch']}\n")
                        f.write(f"  Values: {last_snapshot['values']}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("End of Final Epoch Data\n")

        print(f"Final epoch log saved to {final_log_path}")


    def close(self):
        """Save all history, close log file and W&B."""
        try:
            self.save_full_history()
        except Exception as e:
            print(f"Warning: Failed to save full history: {e}")

        try:
            self.save_final_epoch_log()
        except Exception as e:
            print(f"Warning: Failed to save final epoch log: {e}")

        if self.log_file:
            try:
                self._write_to_log(f"\n\nTraining completed at {datetime.now().isoformat()}\n")
                self._write_to_log("=" * 60 + "\n")
            except Exception:
                pass  # Ignore write errors during close
            finally:
                self.log_file.close()
                print(f"CBP training log saved to {self.log_file_path}")

        if self.wandb:
            try:
                self.wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to close wandb: {e}")