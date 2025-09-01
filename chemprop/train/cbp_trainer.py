from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
import numpy as np
from math import sqrt
import json
import os
from datetime import datetime
import logging

from chemprop.models import MoleculeModel  # MoleculeModel now handles CBP internally
from chemprop.args import TrainArgs


class AdamGnT(Optimizer):
    r"""Implements Adam algorithm for generate-and-test.

    It is a modification of `Adam: A Method for Stochastic Optimization`_.
    The key difference is that step counters are per-parameter element rather than per-parameter group.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamGnT, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamGnT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # 🔥 Key modification: per-parameter element step counters
                    state['step'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # 🔥 Fix AdamGnT implementation - use correct Adam formula
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Calculate bias-corrected step size
                step_size = group['lr'] * (bias_correction2.sqrt() / bias_correction1)
                
                # Standard Adam parameter update: p = p - step_size * exp_avg / denom
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss


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
    
    def log_batch_stats(self, batch_id: int, stats: Dict):
        """Log batch-level CBP statistics, including replaced neuron indices"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            total_replaced = stats.get('total_neurons_replaced', 0)
            if total_replaced > 0:
                f.write(f"Batch {batch_id:4d} ({timestamp}): {total_replaced} neurons replaced")
                
                # Add layer replacement information
                if 'layer_replacements' in stats:
                    f.write(f" {stats['layer_replacements']}")
                
                # 🔥 Add specific indices of replaced neurons
                if 'replaced_neuron_indices' in stats:
                    indices = stats['replaced_neuron_indices']
                    f.write(f"\n    └─ Replaced indices: {indices}")
                
                f.write("\n")
        
        # Console output key information (only when replacement occurs)
        total_replaced = stats.get('total_neurons_replaced', 0)
        if total_replaced > 0:
            # 🔥 Support multiple hidden layers: show replacement indices for each layer
            replaced_indices = stats.get('replaced_neuron_indices', [])
            layer_info = []
            
            for layer_idx, indices in enumerate(replaced_indices):
                if indices:  # If this layer has neurons replaced
                    layer_info.append(f"L{layer_idx}:{indices}")
            
            if layer_info:
                print(f"🔥 Batch {batch_id}: {total_replaced} neurons replaced [{', '.join(layer_info)}]")
            else:
                print(f"🔥 Batch {batch_id}: {total_replaced} neurons replaced")
    
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
            if 'all_replaced_indices' in stats:
                f.write(f"\n📍 Replaced Neuron Indices Summary:\n")
                all_indices = stats['all_replaced_indices']
                for layer_idx, layer_indices in enumerate(all_indices):
                    if layer_indices:
                        f.write(f"   Layer {layer_idx}: {layer_indices}\n")
                    else:
                        f.write(f"   Layer {layer_idx}: No replacements\n")
            
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


class GnTForChemprop:
    """Generate-and-Test algorithm adapted for ChemProp's FFN layers."""
    
    def __init__(
        self,
        ffn_layers: nn.Sequential,
        hidden_activation: str,
        opt,
        decay_rate: float = 0.99,
        replacement_rate: float = 1e-4,
        init: str = 'kaiming',
        device: str = "cpu",
        maturity_threshold: int = 20,
        util_type: str = 'contribution',
        accumulate: bool = False,
    ):
        self.device = device
        self.layers = ffn_layers
        self.num_hidden_layers = sum(1 for layer in self.layers if isinstance(layer, nn.Linear)) - 1
        self.accumulate = accumulate

        self.opt = opt
        self.opt_type = 'adam' if isinstance(opt, Adam) else 'sgd'

        # Hyperparameters
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        # Extract linear layers
        self.linear_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        
        # Initialize utility tracking
        self.util = []
        self.bias_corrected_util = []
        self.ages = []
        self.mean_feature_act = []
        self.accumulated_num_features_to_replace = []
        
        for i in range(self.num_hidden_layers):
            out_features = self.linear_layers[i].out_features
            self.util.append(torch.zeros(out_features).to(device))
            self.bias_corrected_util.append(torch.zeros(out_features).to(device))
            self.ages.append(torch.zeros(out_features).to(device))
            self.mean_feature_act.append(torch.zeros(out_features).to(device))
            self.accumulated_num_features_to_replace.append(0)

        # Compute initialization bounds
        self.bounds = self.compute_bounds(hidden_activation, init)

    def compute_bounds(self, hidden_activation: str, init: str) -> List[float]:
        """Compute initialization bounds for new features."""
        bounds = []
        
        # Create activation function name mapping to convert ChemProp names to PyTorch expected names
        activation_mapping = {
            'ReLU': 'relu',
            'LeakyReLU': 'leaky_relu', 
            'PReLU': 'relu',  # PyTorch doesn't have PReLU gain, use relu as approximation
            'tanh': 'tanh',
            'SELU': 'selu',
            'ELU': 'relu'  # PyTorch doesn't have ELU gain, use relu as approximation
        }
        
        # Check if activation function is supported
        if hidden_activation not in activation_mapping:
            raise ValueError(
                f"Unsupported activation function '{hidden_activation}' for CBP training. "
                f"Supported activation functions: {list(activation_mapping.keys())}"
            )
        
        # Convert activation function name
        pytorch_activation = activation_mapping[hidden_activation]
        
        for i, layer in enumerate(self.linear_layers[:-1]):  # Exclude output layer
            if init == 'xavier':
                try:
                    gain = nn.init.calculate_gain(nonlinearity=pytorch_activation)
                except ValueError as e:
                    raise ValueError(
                        f"PyTorch does not support gain calculation for activation '{pytorch_activation}' "
                        f"(mapped from ChemProp's '{hidden_activation}'). "
                        f"Original PyTorch error: {str(e)}"
                    )
                bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
            elif init == 'lecun':
                bound = sqrt(3 / layer.in_features)
            else:  # kaiming
                try:
                    gain = nn.init.calculate_gain(nonlinearity=pytorch_activation)
                except ValueError as e:
                    raise ValueError(
                        f"PyTorch does not support gain calculation for activation '{pytorch_activation}' "
                        f"(mapped from ChemProp's '{hidden_activation}'). "
                        f"Original PyTorch error: {str(e)}"
                    )
                bound = gain * sqrt(3 / layer.in_features)
            bounds.append(bound)
        
        # Output layer
        last_layer = self.linear_layers[-1]
        bounds.append(sqrt(3 / last_layer.in_features))
        
        return bounds

    def update_utility(self, layer_idx: int, features: torch.Tensor):
        """Update utility scores for features."""
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            
            # Bias correction (Adam-style)
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]
            
            # Update mean activation
            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_feature_act[layer_idx] += (1 - self.decay_rate) * features.mean(dim=0)
            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction
            
            current_layer = self.linear_layers[layer_idx]
            next_layer = self.linear_layers[layer_idx + 1]
            
            # Get weight magnitudes
            output_weight_mag = next_layer.weight.data.abs().mean(dim=0)
            input_weight_mag = current_layer.weight.data.abs().mean(dim=1)
            
            # Compute utility based on type
            if self.util_type == 'weight':
                new_util = output_weight_mag
            elif self.util_type == 'contribution':
                new_util = output_weight_mag * features.abs().mean(dim=0)
            elif self.util_type == 'adaptation':
                new_util = 1 / input_weight_mag
            elif self.util_type == 'zero_contribution':
                new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0)
            elif self.util_type == 'adaptable_contribution':
                new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0) / input_weight_mag
            elif self.util_type == 'feature_by_input':
                new_util = (features - bias_corrected_act).abs().mean(dim=0) / input_weight_mag
            elif self.util_type == 'random':
                new_util = torch.rand_like(self.util[layer_idx])
            else:
                new_util = torch.zeros_like(self.util[layer_idx])
            
            self.util[layer_idx] += (1 - self.decay_rate) * new_util
            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction
            
            # Handle random utility (regenerate each time)
            if self.util_type == 'random':
                self.bias_corrected_util[layer_idx] = torch.rand_like(self.util[layer_idx])

    def test_features(self, features: List[torch.Tensor]):
        """Test features and determine which to replace."""
        features_to_replace = []
        num_features_to_replace = []
        
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace
        
        for i in range(min(len(features), self.num_hidden_layers)):
            self.ages[i] += 1
            self.update_utility(i, features[i])
            
            # Find eligible features
            eligible_indices = torch.where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_indices.shape[0] == 0:
                features_to_replace.append(torch.empty(0, dtype=torch.long).to(self.device))
                num_features_to_replace.append(0)
                continue
            
            # Calculate number to replace
            num_new = self.replacement_rate * eligible_indices.shape[0]
            self.accumulated_num_features_to_replace[i] += num_new
            
            if self.accumulate:
                num_new = int(self.accumulated_num_features_to_replace[i])
                self.accumulated_num_features_to_replace[i] -= num_new
            else:
                if num_new < 1:
                    num_new = 1 if torch.rand(1) <= num_new else 0
                num_new = int(num_new)
            
            if num_new == 0:
                features_to_replace.append(torch.empty(0, dtype=torch.long).to(self.device))
                num_features_to_replace.append(0)
                continue
            
            # Select features with lowest utility
            indices = torch.topk(-self.bias_corrected_util[i][eligible_indices], num_new)[1]
            indices = eligible_indices[indices]
            
            # Reset utility for new features
            self.util[i][indices] = 0
            self.mean_feature_act[i][indices] = 0
            
            features_to_replace.append(indices)
            num_features_to_replace.append(num_new)
        
        return features_to_replace, num_features_to_replace

    def test_features_epoch_end(self, dummy_features: List[torch.Tensor]):
        """Decide which features to replace at epoch end, without updating age and utility"""
        features_to_replace = []
        num_features_to_replace = []
        
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace
        
        for i in range(min(len(dummy_features), self.num_hidden_layers)):
            # Note: Don't update age and utility, these have been updated in batches
            
            # Find eligible features
            eligible_indices = torch.where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_indices.shape[0] == 0:
                features_to_replace.append(torch.empty(0, dtype=torch.long).to(self.device))
                num_features_to_replace.append(0)
                continue
            
            # Calculate number to replace - 🔥 Use epoch-level replacement rate
            epoch_replacement_rate = self.replacement_rate  # Replacement rate per epoch
            num_new = epoch_replacement_rate * eligible_indices.shape[0]
            self.accumulated_num_features_to_replace[i] += num_new
            
            if self.accumulate:
                num_new = int(self.accumulated_num_features_to_replace[i])
                self.accumulated_num_features_to_replace[i] -= num_new
            else:
                if num_new < 1:
                    num_new = 1 if torch.rand(1) <= num_new else 0
                num_new = int(num_new)
            
            if num_new == 0:
                features_to_replace.append(torch.empty(0, dtype=torch.long).to(self.device))
                num_features_to_replace.append(0)
                continue
            
            # Select features with lowest utility
            indices = torch.topk(-self.bias_corrected_util[i][eligible_indices], num_new)[1]
            indices = eligible_indices[indices]
            
            # Reset utility for new features
            self.util[i][indices] = 0
            self.mean_feature_act[i][indices] = 0
            
            features_to_replace.append(indices)
            num_features_to_replace.append(num_new)
        
        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace: List[torch.Tensor], num_features_to_replace: List[int]):
        """Generate new features by reinitializing weights."""
        with torch.no_grad():
            for i in range(len(features_to_replace)):
                if num_features_to_replace[i] == 0:
                    continue
                
                indices = features_to_replace[i]
                current_layer = self.linear_layers[i]
                next_layer = self.linear_layers[i + 1]
                
                # Reset input weights
                current_layer.weight.data[indices, :] = torch.empty(
                    num_features_to_replace[i], current_layer.in_features
                ).uniform_(-self.bounds[i], self.bounds[i]).to(self.device)
                
                # Reset bias
                current_layer.bias.data[indices] = 0
                
                # Update next layer bias and reset output weights
                next_layer.bias.data += (
                    next_layer.weight.data[:, indices] * 
                    self.mean_feature_act[i][indices] / 
                    (1 - self.decay_rate ** self.ages[i][indices])
                ).sum(dim=1)
                
                next_layer.weight.data[:, indices] = 0
                self.ages[i][indices] = 0

    def update_optim_params(self, features_to_replace: List[torch.Tensor], num_features_to_replace: List[int]):
        """Update optimizer state for replaced features."""
        if self.opt_type != 'adam':
            return
        
        for i in range(len(features_to_replace)):
            if num_features_to_replace[i] == 0:
                continue
            
            indices = features_to_replace[i]
            current_layer = self.linear_layers[i]
            next_layer = self.linear_layers[i + 1]
            
            # Reset current layer (input) weights and bias states
            if current_layer.weight in self.opt.state:
                state = self.opt.state[current_layer.weight]
                if 'exp_avg' in state:
                    state['exp_avg'][indices, :] = 0.0
                    state['exp_avg_sq'][indices, :] = 0.0
                # Standard Adam uses scalar step counter, no reset needed
                # if 'step' in state:
                #     # Standard Adam's step is scalar, not tensor
                #     pass
            
            if current_layer.bias in self.opt.state:
                state = self.opt.state[current_layer.bias]
                if 'exp_avg' in state:
                    state['exp_avg'][indices] = 0.0
                    state['exp_avg_sq'][indices] = 0.0
                # Standard Adam uses scalar step counter, no reset needed
                # if 'step' in state:
                #     # Standard Adam's step is scalar, not tensor
                #     pass
            
            # Reset next layer (output) weight states
            if next_layer.weight in self.opt.state:
                state = self.opt.state[next_layer.weight]
                if 'exp_avg' in state:
                    state['exp_avg'][:, indices] = 0.0
                    state['exp_avg_sq'][:, indices] = 0.0
                # Standard Adam uses scalar step counter, no reset needed
                # if 'step' in state:
                #     # Standard Adam's step is scalar, not tensor
                #     pass

    def gen_and_test(self, features: List[torch.Tensor]):
        """Perform generate-and-test step and collect statistics."""
        features_to_replace, num_features_to_replace = self.test_features(features)
        self.gen_new_features(features_to_replace, num_features_to_replace)
        self.update_optim_params(features_to_replace, num_features_to_replace)
        
        # 🔥 Collect CBP statistics
        stats = self.collect_cbp_stats(features_to_replace, num_features_to_replace)
        return stats
    
    def collect_cbp_stats(self, features_to_replace: List[torch.Tensor], num_features_to_replace: List[int]):
        """Collect CBP statistics, including specific indices of replaced neurons"""
        # Collect replaced neuron index information
        replaced_indices = []
        for i, indices_tensor in enumerate(features_to_replace):
            if len(indices_tensor) > 0:
                # Convert to Python list for JSON serialization and logging
                indices_list = indices_tensor.cpu().tolist()
                replaced_indices.append(indices_list)
            else:
                replaced_indices.append([])
        
        stats = {
            'total_neurons_replaced': sum(num_features_to_replace),
            'layer_replacements': num_features_to_replace.copy(),
            'replaced_neuron_indices': replaced_indices,  # 🔥 New: specific indices of replaced neurons
            'replacement_rates': [],
            'avg_utilities': [],
            'age_stats': {}
        }
        
        # Calculate replacement rate and utility statistics for each layer
        for i in range(len(self.bias_corrected_util)):
            if i < len(num_features_to_replace):
                # Replacement rate: actual replacements / mature neurons
                eligible_count = torch.sum(self.ages[i] > self.maturity_threshold).item()
                replacement_rate = num_features_to_replace[i] / max(eligible_count, 1)
                stats['replacement_rates'].append(round(replacement_rate, 4))
                
                # Average utility
                avg_utility = torch.mean(self.bias_corrected_util[i]).item()
                stats['avg_utilities'].append(round(avg_utility, 6))
        
        # Age statistics
        if len(self.ages) > 0:
            all_ages = torch.cat([ages.flatten() for ages in self.ages])
            stats['age_stats'] = {
                'mean_age': round(torch.mean(all_ages.float()).item(), 2),
                'max_age': torch.max(all_ages).item(),
                'mature_neurons': torch.sum(all_ages > self.maturity_threshold).item(),
                'total_neurons': len(all_ages)
            }
        
        return stats


class ContinualBackpropTrainer:
    """Enhanced trainer for ChemProp models using Continual Backpropagation with full ChemProp feature support."""
    
    def __init__(
        self,
        model: MoleculeModel,
        args: TrainArgs,
        step_size: float = 0.001,
        replacement_rate: float = 0.001,
        decay_rate: float = 0.9,
        maturity_threshold: int = 100,
        util_type: str = 'contribution',
        accumulate: bool = False,
        enable_cbp_logging: bool = True,
        log_dir: str = None,
    ):
        self.model = model
        self.args = args
        self.device = args.device
        
        # 🔥 CBP logger
        self.enable_cbp_logging = enable_cbp_logging
        if self.enable_cbp_logging:
            self.cbp_logger = CBPLogger(log_dir=log_dir)
            self.cbp_stats_buffer = []  # Collect statistics for each batch
        else:
            self.cbp_logger = None
            self.cbp_stats_buffer = []
        
        # Setup optimizer - 🔥 Temporarily use standard Adam, not AdamGnT
        if args.optimizer == 'adam':
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
        
        # Setup GnT for FFN layers
        if hasattr(model.ffn, 'layers'):
            self.gnt = GnTForChemprop(
                ffn_layers=model.ffn.layers,
                hidden_activation=args.activation,
                opt=self.optimizer,
                replacement_rate=replacement_rate,
                decay_rate=decay_rate,
                maturity_threshold=maturity_threshold,
                util_type=util_type,
                device=self.device,
                accumulate=accumulate,
            )
        else:
            self.gnt = None

    def train_step_advanced(self, 
                           batch_data: Dict,
                           targets: torch.Tensor,
                           mask: torch.Tensor,
                           target_weights: torch.Tensor,
                           data_weights: torch.Tensor,
                           loss_func: Callable,
                           args: TrainArgs,
                           lt_targets: torch.Tensor = None,
                           gt_targets: torch.Tensor = None) -> float:
        """
        Advanced training step that supports all ChemProp loss functions and data types.
        
        :param batch_data: Dictionary containing all batch data
        :param targets: Target values
        :param mask: Mask for valid targets
        :param target_weights: Weights for different targets  
        :param data_weights: Weights for different samples
        :param loss_func: Loss function to use
        :param args: Training arguments
        :param lt_targets: Lower bound targets (for bounded_mse)
        :param gt_targets: Upper bound targets (for bounded_mse)
        :return: Loss value
        """
        self.model.train()
        
        # 🔥 Use CBP model's predict method to get output and intermediate activations
        if hasattr(self.model, 'predict'):                             # assert hasattr(self.model, 'predict')
            # Construct complete input data
            full_batch_data = (
                batch_data['mol_batch'],
                batch_data['features_batch'],
                batch_data['atom_descriptors_batch'],
                batch_data['atom_features_batch'], 
                batch_data['bond_features_batch']
            )
            preds, features = self.model.predict(full_batch_data)
        else:
            # Fallback mode
            preds = self.model(
                batch_data['mol_batch'],
                batch_data['features_batch'],
                batch_data['atom_descriptors_batch'],
                batch_data['atom_features_batch'],
                batch_data['bond_features_batch']
            )
            features = []

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

        # 🔥 CBP-specific step: execute Generate-and-Test per original logic (per batch)
        if self.gnt is not None and features:
            self.optimizer.zero_grad()
            cbp_stats = self.gnt.gen_and_test(features)
            
            # Collect CBP statistics and log immediately (batch level)
            if self.enable_cbp_logging and cbp_stats:
                self.log_batch_cbp_stats(cbp_stats)
        
        return loss.item()
    
    def train_step(self, batch, targets):
        """Maintain compatibility with simple interface"""
        mol_batch, features_batch = batch
        
        # Simple batch_data construction
        batch_data = {
            'mol_batch': mol_batch,
            'features_batch': features_batch,
            'atom_descriptors_batch': None,
            'atom_features_batch': None,
            'bond_features_batch': None
        }
        
        # Create simple mask and weights
        batch_size = targets.shape[0]
        num_tasks = targets.shape[1] if len(targets.shape) > 1 else 1
        mask = torch.ones_like(targets, dtype=torch.bool)
        target_weights = torch.ones(1, num_tasks)
        data_weights = torch.ones(batch_size, 1)
        
        # Simple loss function
        if self.args.dataset_type == 'classification':
            loss_func = nn.BCEWithLogitsLoss()
        elif self.args.dataset_type == 'multiclass':
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.MSELoss()
        
        return self.train_step_advanced(
            batch_data=batch_data,
            targets=targets,
            mask=mask,
            target_weights=target_weights,
            data_weights=data_weights,
            loss_func=loss_func,
            args=self.args
        )
    
    def log_batch_cbp_stats(self, cbp_stats: Dict):
        """Log CBP statistics for each batch"""
        if not self.enable_cbp_logging or not self.cbp_logger:
            return
        
        # Add to buffer and immediately log to file
        self.cbp_stats_buffer.append(cbp_stats)
        
        # Write log for each batch (optional: for detailed tracking)
        total_replaced = cbp_stats.get('total_neurons_replaced', 0)
        if total_replaced > 0:  # Only log batches with replacements
            batch_id = len(self.cbp_stats_buffer)
            self.cbp_logger.log_batch_stats(batch_id, cbp_stats)
    
    def log_epoch_cbp_stats(self, epoch: int):
        """Aggregate CBP statistics for current epoch (based on batch-level data)"""
        if not self.enable_cbp_logging or not self.cbp_stats_buffer:
            return
        
        # Aggregate statistics from all batches
        epoch_stats = self.aggregate_cbp_stats(self.cbp_stats_buffer)
        
        # Log epoch-level summary
        if self.cbp_logger:
            self.cbp_logger.log_epoch_summary(epoch, epoch_stats)
        
        # Clear buffer for next epoch
        self.cbp_stats_buffer.clear()
    
    def aggregate_cbp_stats(self, stats_buffer: List[Dict]):
        """Aggregate CBP statistics from multiple batches"""
        if not stats_buffer:
            return {}
        
        # Accumulate total replacements
        total_replaced = sum(stats.get('total_neurons_replaced', 0) for stats in stats_buffer)
        
        # Count batches with replacements
        active_batches = sum(1 for stats in stats_buffer if stats.get('total_neurons_replaced', 0) > 0)
        
        # Accumulate replacements for each layer
        if stats_buffer:
            num_layers = len(stats_buffer[0].get('layer_replacements', []))
            layer_replacements = [0] * num_layers
            
            # Average replacement rates and utilities (only calculate for batches with data)
            avg_replacement_rates = [[] for _ in range(num_layers)]
            avg_utilities = [[] for _ in range(num_layers)]
            
            for stats in stats_buffer:
                layer_reps = stats.get('layer_replacements', [])
                for i, rep_count in enumerate(layer_reps):
                    if i < len(layer_replacements):
                        layer_replacements[i] += rep_count
                
                replacement_rates = stats.get('replacement_rates', [])
                for i, rate in enumerate(replacement_rates):
                    if i < len(avg_replacement_rates) and rate > 0:
                        avg_replacement_rates[i].append(rate)
                
                utilities = stats.get('avg_utilities', [])
                for i, util in enumerate(utilities):
                    if i < len(avg_utilities):
                        avg_utilities[i].append(util)
            
            # Calculate averages (only non-zero values)
            final_replacement_rates = [
                round(np.mean(rates), 4) if rates else 0.0 
                for rates in avg_replacement_rates
            ]
            
            final_avg_utilities = [
                round(np.mean(utils), 6) if utils else 0.0 
                for utils in avg_utilities
            ]
            
            # Use age statistics from latest batch
            age_stats = stats_buffer[-1].get('age_stats', {}) if stats_buffer else {}
        else:
            layer_replacements = []
            final_replacement_rates = []
            final_avg_utilities = []
            age_stats = {}
        
        # 🔥 Aggregate all replaced neuron indices
        all_replaced_indices = [[] for _ in range(num_layers)] if stats_buffer else []
        for stats in stats_buffer:
            batch_indices = stats.get('replaced_neuron_indices', [])
            for layer_idx, indices in enumerate(batch_indices):
                if layer_idx < len(all_replaced_indices) and indices:
                    all_replaced_indices[layer_idx].extend(indices)
        
        return {
            'total_neurons_replaced': total_replaced,
            'layer_replacements': layer_replacements,
            'replacement_rates': final_replacement_rates,
            'avg_utilities': final_avg_utilities,
            'age_stats': age_stats,
            'active_batches': active_batches,
            'total_batches': len(stats_buffer),
            'all_replaced_indices': all_replaced_indices  # 🔥 新增：汇总的神经元index
        }
    
    def save_cbp_summary(self):
        """保存CBP训练总结"""
        if self.enable_cbp_logging and self.cbp_logger:
            self.cbp_logger.save_summary() 