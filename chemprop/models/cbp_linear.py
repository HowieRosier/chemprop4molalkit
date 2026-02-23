import torch
from torch import nn
from math import sqrt


def log_gradients(m, grad_input, grad_output):
    """Backward hook: periodically log gradient magnitudes to CBPLogger."""
    if not m.training:
        return

    if not hasattr(m, 'cbp_logger'):
        return

    if m.cbp_logger is None:
        return

    if hasattr(m, '_grad_log_counter'):
        m._grad_log_counter += 1
    else:
        m._grad_log_counter = 1

    log_frequency = getattr(m, 'grad_log_frequency', 100)

    if m._grad_log_counter % log_frequency != 0:
        return

    with torch.no_grad():
        if grad_output[0] is not None:
            grad_magnitude = grad_output[0].abs().mean(dim=0)

            if hasattr(m, 'cbp_logger') and m.cbp_logger is not None:
                m.cbp_logger.log_gradients(
                    layer_name=m.layer_name,
                    gradients=grad_magnitude,
                    utilities=m.util,
                    ages=m.ages,
                    batch_idx=m._grad_log_counter,
                    epoch=getattr(m, 'current_epoch', 0)
                )


def call_reinit(m, i, o):
    m.reinit()


def log_features(m, i, o):
    """Forward hook: update neuron utility via EMA and periodically log activations."""
    if not m.training:
        return
    with torch.no_grad():
        m.util.data *= m.decay_rate
        output_weight_mag = m.out_layer.weight.data.abs().mean(dim=0)

        if m.util_type == 'contribution':
            new_util = output_weight_mag * i[0].abs().mean(dim=0)
        elif m.util_type == 'weight':
            new_util = output_weight_mag
        elif m.util_type == 'adaptation':
            input_weight_mag = m.in_layer.weight.data.abs().mean(dim=1)
            new_util = 1 / (input_weight_mag + 1e-8)
        elif m.util_type == 'random':
            new_util = torch.rand_like(m.util.data)
        else:
            new_util = output_weight_mag * i[0].abs().mean(dim=0)

        m.util.data += (1 - m.decay_rate) * new_util

        if hasattr(m, 'cbp_logger') and m.cbp_logger is not None:
            if not hasattr(m, '_forward_log_counter'):
                m._forward_log_counter = 0
            m._forward_log_counter += 1

            log_freq = getattr(m, 'grad_log_frequency', 1000000)
            if m._forward_log_counter % log_freq == 0:
                if o is not None:
                    activations = o.abs().mean(dim=0) if o.dim() > 1 else o.abs()
                    m.cbp_logger.log_activations(
                        layer_name=m.layer_name,
                        activations=activations,
                        batch_idx=m._forward_log_counter,
                        epoch=getattr(m, 'current_epoch', 0)
                    )


def get_layer_bound(layer, init, gain):
    """Compute uniform init bound based on weight initialization scheme."""
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
    """Wraps a linear layer pair (in/out) to enable CBP neuron replacement."""

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
            cbp_logger=None,
            layer_type='MPN',
            layer_name=None,
            accumulate=True,
            grad_log_frequency=1000000,
    ):
        super().__init__()
        if type(in_layer) is not nn.Linear:
            raise Warning("Make sure in_layer is a weight layer")
        if type(out_layer) is not nn.Linear:
            raise Warning("Make sure out_layer is a weight layer")
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.previous_features = None
        self.layer_type = layer_type
        self.layer_name = layer_name or f'{layer_type}_layer'
        self.accumulate = accumulate
        self.cbp_logger = cbp_logger
        self.grad_log_frequency = grad_log_frequency
        self.current_epoch = 0

        # Register hooks: reinit on backward, track utility on forward
        if self.replacement_rate > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)

        self.in_layer = in_layer
        self.add_in_dim = add_in_dim
        self.out_layer = out_layer
        self.ln_layer = ln_layer
        self.bn_layer = bn_layer

        # Neuron utility and age tracking (non-trainable)
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_features + add_in_dim), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(self.in_layer.out_features + add_in_dim), requires_grad=False)
        self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._batch_counter = 0

        self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=nn.init.calculate_gain(nonlinearity=act_type))

    def forward(self, _input):
        return _input

    def get_features_to_reinit(self):
        """Select mature neurons with lowest utility for replacement."""
        features_to_replace = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.ages += 1

        eligible_feature_indices = torch.where((self.ages > self.maturity_threshold) &
                                               (torch.arange(len(self.ages), device=self.ages.device) >= self.add_in_dim))[0]
        if eligible_feature_indices.shape[0] == 0:  return features_to_replace

        num_new_features_to_replace = self.replacement_rate * eligible_feature_indices.shape[0]

        if self.accumulate:
            self.accumulated_num_features_to_replace += num_new_features_to_replace
            if self.accumulated_num_features_to_replace < 1:
                return features_to_replace
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
            self.accumulated_num_features_to_replace -= num_new_features_to_replace
        else:
            if num_new_features_to_replace < 1:
                num_new_features_to_replace = 1 if torch.rand(1) <= num_new_features_to_replace else 0
            else:
                num_new_features_to_replace = int(num_new_features_to_replace)
            if num_new_features_to_replace == 0:
                return features_to_replace
        # Select features with smallest utility
        new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        features_to_replace = new_features_to_replace
        return features_to_replace

    def reinit_features(self, features_to_replace):
        """Reset in/out weights, optimizer state, and norm layers for replaced neurons."""
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

            # Reset AdamGnT state (step counters + momentum) for replaced neurons
            if hasattr(self, '_optimizer_ref'):
                optimizer = self._optimizer_ref
                from ..models.AdamGnT import AdamGnT
                if isinstance(optimizer, AdamGnT):
                    for param in [self.in_layer.weight, self.out_layer.weight]:
                        if param in optimizer.state:
                            state = optimizer.state[param]
                            if 'step' in state and isinstance(state['step'], torch.Tensor):
                                if param is self.in_layer.weight:
                                    state['step'][features_to_replace - self.add_in_dim, :] = 0
                                elif param is self.out_layer.weight:
                                    state['step'][:, features_to_replace] = 0
                                if 'exp_avg' in state:
                                    if param is self.in_layer.weight:
                                        state['exp_avg'][features_to_replace - self.add_in_dim, :] = 0
                                    elif param is self.out_layer.weight:
                                        state['exp_avg'][:, features_to_replace] = 0
                                if 'exp_avg_sq' in state:
                                    if param is self.in_layer.weight:
                                        state['exp_avg_sq'][features_to_replace - self.add_in_dim, :] = 0
                                    elif param is self.out_layer.weight:
                                        state['exp_avg_sq'][:, features_to_replace] = 0

            # Reset batchnorm/layernorm for replaced neurons
            if self.bn_layer is not None:
                self.bn_layer.bias.data[features_to_replace] = 0.0
                self.bn_layer.weight.data[features_to_replace] = 1.0
                self.bn_layer.running_mean.data[features_to_replace] = 0.0
                self.bn_layer.running_var.data[features_to_replace] = 1.0
            if self.ln_layer is not None:
                self.ln_layer.bias.data[features_to_replace] = 0.0
                self.ln_layer.weight.data[features_to_replace] = 1.0

        self._log_replacement_stats(features_to_replace, num_features_to_replace)

    def _log_replacement_stats(self, features_to_replace: torch.Tensor, num_features_to_replace: int):
        if num_features_to_replace == 0:
            return

        if self.cbp_logger is not None:
            self._batch_counter += 1
            self.cbp_logger.log_replacement_event(
                layer_name=self.layer_name,
                replaced_indices=features_to_replace.detach().cpu().tolist(),
                replacement_rate=self.replacement_rate,
                batch_idx=self._batch_counter,
                epoch=self.current_epoch
            )
    
    def reinit(self):
        features_to_replace = self.get_features_to_reinit()
        self.reinit_features(features_to_replace)
