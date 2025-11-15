import torch
from torch import nn
from math import sqrt
from typing import Optional, Dict


def call_reinit(m, i, o):
    m.reinit()


def log_features(m, i, o):
    if not m.training:
        return
    with torch.no_grad():
        m.util.data *= m.decay_rate
        output_weight_mag = m.out_layer.weight.data.abs().mean(dim=0)

        # Calculate utility based on util_type
        if m.util_type == 'contribution':
            new_util = output_weight_mag * i[0].abs().mean(dim=0)
        elif m.util_type == 'weight':
            new_util = output_weight_mag
        elif m.util_type == 'adaptation':
            input_weight_mag = m.in_layer.weight.data.abs().mean(dim=1)
            new_util = 1 / (input_weight_mag + 1e-8)  # Add small epsilon for stability
        elif m.util_type == 'random':
            new_util = torch.rand_like(m.util.data)
        else:
            # Default to contribution
            new_util = output_weight_mag * i[0].abs().mean(dim=0)

        m.util.data += (1 - m.decay_rate) * new_util


def get_layer_bound(layer, init, gain):
    assert isinstance(layer, nn.Linear)
    if init == 'default':
        bound = sqrt(1 / layer.in_features)
    elif init == 'xavier':
        bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
    elif init == 'lecun':
        bound = sqrt(3 / layer.in_features)
    elif init == 'kaiming':
        bound = gain * sqrt(3 / layer.in_features)
    else:
        raise ValueError(f'Invalid weight initialization: {init}')
    return bound


class CBPLinear(nn.Module):
    def __init__(
            self,
            out_layer: nn.Linear,
            in_layer: nn.Linear,
            add_in_dim: int = 0,
            ln_layer: nn.LayerNorm = None,
            bn_layer: nn.BatchNorm1d = None,
            replacement_rate=1e-4,
            maturity_threshold=100,
            init='kaiming',
            act_type='relu',
            util_type='contribution',
            decay_rate=0.99,
            cbp_logger=None,  # CBPLogger instance or None
            layer_type='MPN',  # 'MPN' or 'FFN' to distinguish layer types
            layer_name=None,  # Optional custom name for logging
            accumulate=True,  # Whether to accumulate replacement counts
    ):
        super().__init__()
        if type(in_layer) is not nn.Linear:
            raise Warning("Make sure in_layer is a weight layer")
        if type(out_layer) is not nn.Linear:
            raise Warning("Make sure out_layer is a weight layer")
        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.previous_features = None
        self.layer_type = layer_type
        self.layer_name = layer_name or f'{layer_type}_layer'
        self.accumulate = accumulate
        """
        Register hooks
        """
        if self.replacement_rate > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)

        self.in_layer = in_layer
        self.add_in_dim = add_in_dim
        self.out_layer = out_layer
        self.ln_layer = ln_layer
        self.bn_layer = bn_layer
        """
        Utility of all features/neurons
        """
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_features + add_in_dim), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(self.in_layer.out_features + add_in_dim), requires_grad=False)
        self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)
        """
        CBP Logger for tracking neuron replacements
        """
        self.cbp_logger = cbp_logger
        self._batch_counter = 0
        """
        Calculate uniform distribution's bound for random feature initialization
        """
        self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=nn.init.calculate_gain(nonlinearity=act_type))

    def forward(self, _input):
        return _input

    def get_features_to_reinit(self):
        """
        Returns: Features to replace
        """
        features_to_replace = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.ages += 1
        """
        Calculate number of features to replace
        """
        # eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        eligible_feature_indices = torch.where((self.ages > self.maturity_threshold) & 
                                               (torch.arange(len(self.ages), device=self.ages.device) >= self.add_in_dim))[0]
        if eligible_feature_indices.shape[0] == 0:  return features_to_replace

        num_new_features_to_replace = self.replacement_rate * eligible_feature_indices.shape[0]

        if self.accumulate:
            # Accumulate replacement counts over time
            self.accumulated_num_features_to_replace += num_new_features_to_replace
            if self.accumulated_num_features_to_replace < 1:
                return features_to_replace
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
            self.accumulated_num_features_to_replace -= num_new_features_to_replace
        else:
            # Non-accumulating mode: use probabilistic replacement
            if num_new_features_to_replace < 1:
                num_new_features_to_replace = 1 if torch.rand(1) <= num_new_features_to_replace else 0
            else:
                num_new_features_to_replace = int(num_new_features_to_replace)
            if num_new_features_to_replace == 0:
                return features_to_replace
        """
        Find features with smallest utility
        """
        new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        features_to_replace = new_features_to_replace
        return features_to_replace

    def reinit_features(self, features_to_replace):
        """
        Reset input and output weights for low utility features
        """
        with torch.no_grad():
            num_features_to_replace = features_to_replace.shape[0]

            if num_features_to_replace == 0: return
            self.in_layer.weight.data[features_to_replace - self.add_in_dim, :] *= 0.0
            self.in_layer.weight.data[features_to_replace - self.add_in_dim, :] += \
                torch.empty(num_features_to_replace, self.in_layer.in_features, device=self.util.device).uniform_(-self.bound, self.bound)
            if self.in_layer.bias is not None:
                self.in_layer.bias.data[features_to_replace - self.add_in_dim] *= 0

            self.out_layer.weight.data[:, features_to_replace] = 0
            self.ages[features_to_replace] = 0

            """
            Reset the corresponding batchnorm/layernorm layers
            """
            if self.bn_layer is not None:
                self.bn_layer.bias.data[features_to_replace] = 0.0
                self.bn_layer.weight.data[features_to_replace] = 1.0
                self.bn_layer.running_mean.data[features_to_replace] = 0.0
                self.bn_layer.running_var.data[features_to_replace] = 1.0
            if self.ln_layer is not None:
                self.ln_layer.bias.data[features_to_replace] = 0.0
                self.ln_layer.weight.data[features_to_replace] = 1.0

        # Log CBP statistics
        self._log_replacement_stats(features_to_replace, num_features_to_replace)

    def _log_replacement_stats(self, features_to_replace: torch.Tensor, num_features_to_replace: int):
        """
        Log CBP replacement statistics to logger and trainer buffer.
        
        Args:
            features_to_replace: Tensor containing indices of replaced features
            num_features_to_replace: Number of features being replaced
        """
        if num_features_to_replace == 0:
            return
            
        try:
            # Log to CBPLogger if available
            if hasattr(self, 'cbp_logger') and self.cbp_logger is not None:
                layer_name = getattr(self, 'layer_name', f'{self.layer_type}_layer')

                stats = {
                    'total_neurons_replaced': int(num_features_to_replace),
                    'layer_replacements': [int(num_features_to_replace)],
                    'replaced_neuron_indices': [features_to_replace.detach().cpu().tolist()],
                    'layer_names': [layer_name],
                }

                # Initialize batch counter if needed
                if not hasattr(self, '_batch_counter'):
                    self._batch_counter = 0
                self._batch_counter += 1

                # Log batch statistics
                self.cbp_logger.log_batch_stats(self._batch_counter, stats, layer_type=self.layer_type)
            
            # Store stats in trainer's buffer for epoch aggregation
            if hasattr(self, '_trainer'):
                layer_name = getattr(self, 'layer_name', f'{self.layer_type}_layer')
                stats = {
                    'layer_name': layer_name,
                    'replaced_indices': features_to_replace.detach().cpu().tolist(),
                    'num_replaced': int(num_features_to_replace),
                    'layer_type': self.layer_type
                }
                # Use appropriate buffer based on layer type
                if self.layer_type == 'FFN' and hasattr(self._trainer, 'ffn_stats_buffer'):
                    self._trainer.ffn_stats_buffer.append(stats)
                elif self.layer_type == 'MPN' and hasattr(self._trainer, 'mpn_stats_buffer'):
                    self._trainer.mpn_stats_buffer.append(stats)
                
        except Exception:
            # Silently handle any logging errors to avoid disrupting training
            pass
    
    def reinit(self):
        """
        Perform selective reinitialization
        """
        features_to_replace = self.get_features_to_reinit()
        self.reinit_features(features_to_replace)
