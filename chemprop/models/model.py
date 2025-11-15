from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from .cbp_linear import CBPLinear
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class FFNNetworkCBP(nn.Module):
    """A feed-forward neural network with CBPLinear layers for continual backpropagation."""

    def __init__(self, layers: nn.Sequential, enable_cbp: bool = False, cbp_params: dict = None):
        super(FFNNetworkCBP, self).__init__()
        self.enable_cbp = enable_cbp
        self.cbp_params = cbp_params or {}

        # Extract linear layers and build new structure with CBPLinear
        if enable_cbp and cbp_params:
            self.layers, self.cbp_layers = self._build_cbp_structure(layers)
        else:
            # Keep original structure for backward compatibility
            self.layers = layers
            self.cbp_layers = []

    def _build_cbp_structure(self, original_layers):
        """Build FFN structure with CBPLinear layers between linear layers."""
        new_layers = nn.ModuleList()
        cbp_layers = nn.ModuleList()
        linear_layers = []
        linear_indices = []

        # First pass: identify all linear layers
        for i, layer in enumerate(original_layers):
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
                linear_indices.append(i)

        # Second pass: reconstruct with CBPLinear
        layer_counter = 0
        for i, layer in enumerate(original_layers):
            new_layers.append(layer)

            # After each Linear layer (except the last), add CBPLinear
            if isinstance(layer, nn.Linear) and layer_counter < len(linear_layers) - 1:
                # Find the next linear layer
                next_linear = linear_layers[layer_counter + 1]

                # Create CBPLinear
                cbp_layer = CBPLinear(
                    in_layer=layer,
                    out_layer=next_linear,
                    layer_type='FFN',
                    layer_name=f'FFN_Layer{layer_counter}',
                    **self.cbp_params
                )
                cbp_layers.append(cbp_layer)
                new_layers.append(cbp_layer)

                layer_counter += 1

        return new_layers, cbp_layers

    def predict(self, x: torch.Tensor):
        """
        Forward pass that returns both output and intermediate features.

        :param x: Input tensor
        :return: Output and list of intermediate activations
        """
        features = []
        h = x

        if self.enable_cbp and self.cbp_params:
            # Process through layers and collect activations after each hidden layer
            for layer in self.layers:
                if isinstance(layer, CBPLinear):
                    # CBPLinear is a pass-through, but we can collect features here
                    h = layer(h)
                elif isinstance(layer, nn.Linear):
                    h = layer(h)
                elif isinstance(layer, (nn.ReLU, nn.ELU, nn.SELU, nn.Tanh, nn.LeakyReLU, nn.PReLU)):
                    h = layer(h)
                    # Collect features after activation
                    features.append(h)
                else:
                    h = layer(h)
        else:
            # Original behavior for backward compatibility
            i = 0
            while i < len(self.layers):
                layer = self.layers[i]

                if isinstance(layer, nn.Linear):
                    h = layer(h)
                    # Check if next layer is activation
                    if i + 1 < len(self.layers):
                        next_layer = self.layers[i + 1]
                        if isinstance(next_layer, (nn.ReLU, nn.ELU, nn.SELU, nn.Tanh, nn.LeakyReLU, nn.PReLU)):
                            i += 1  # Skip to activation
                            h = next_layer(h)
                            features.append(h)
                else:
                    h = layer(h)

                i += 1

        return h, features

    def forward(self, x: torch.Tensor):
        """Standard forward pass."""
        if self.enable_cbp and self.cbp_params:
            # Process through new structure
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            # Original behavior
            return self.layers(x)

    def get_cbp_layers(self):
        """Return all CBPLinear layers in the network."""
        return list(self.cbp_layers)


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.loss_function = args.loss_function

        # Store CBP flag
        self.use_cbp = hasattr(args, 'cbp') and args.cbp

        if hasattr(args, 'train_class_sizes'):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None

        # when using cross entropy losses, no sigmoid or softmax during training. But they are needed for mcc loss.
        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in ['cross_entropy', 'binary_cross_entropy']

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes
        if self.loss_function == 'mve':
            self.output_size *= 2  # return means and variances
        if self.loss_function == 'dirichlet' and self.classification:
            self.output_size *= 2  # return dirichlet parameters for positive and negative class
        if self.loss_function == 'evidential':
            self.output_size *= 4  # return four evidential parameters: gamma, lambda, alpha, beta

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        if self.loss_function in ['mve', 'evidential', 'dirichlet']:
            self.softplus = nn.Softplus()

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

        if args.checkpoint_frzn is not None or args.freeze_mpn:
            if args.freeze_first_only:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            if args.reaction_solvent:
                first_linear_dim = args.hidden_size + args.hidden_size_solvent
            else:
                first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # If spectra model, also include spectra activation
        if args.dataset_type == 'spectra':
            if args.spectra_activation == 'softplus':
                spectra_activation = nn.Softplus()
            else:  # default exponential activation which must be made into a custom nn module
                class nn_exp(torch.nn.Module):
                    def __init__(self):
                        super(nn_exp, self).__init__()

                    def forward(self, x):
                        return torch.exp(x)

                spectra_activation = nn_exp()
            ffn.append(spectra_activation)

        # Create FFN model - use CBP wrapper if CBP is enabled
        sequential = nn.Sequential(*ffn)
        if self.use_cbp:
            # Prepare CBP parameters for FFN layers
            cbp_params = {
                'replacement_rate': getattr(args, 'cbp_replacement_rate', 1e-4),
                'maturity_threshold': getattr(args, 'cbp_maturity_threshold', 100),
                'decay_rate': getattr(args, 'cbp_decay_rate', 0.99),
                'util_type': getattr(args, 'cbp_util_type', 'contribution'),
                'accumulate': getattr(args, 'cbp_accumulate', True),
                'init': getattr(args, 'cbp_init', 'kaiming'),
                'act_type': args.activation.lower() if args.activation else 'relu'
            }
            self.ffn = FFNNetworkCBP(sequential, enable_cbp=True, cbp_params=cbp_params)
        else:
            self.ffn = sequential

        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers > 0:
                # Access parameters correctly for both CBP and standard models
                if self.use_cbp:
                    params_to_freeze = list(self.ffn.layers.parameters())[0:2 * args.frzn_ffn_layers]
                else:
                    params_to_freeze = list(self.ffn.parameters())[0:2 * args.frzn_ffn_layers]
                for param in params_to_freeze:
                    param.requires_grad = False

    def predict(self, batch_data):
        """
        Forward pass that returns both output and intermediate features for CBP.
        Only used when CBP is enabled.
        
        :param batch_data: Tuple of input batch components or single batch input
        :return: Output predictions and intermediate features
        """
        if not self.use_cbp:
            raise RuntimeError("predict() method should only be called when CBP is enabled")
            
        # Handle different input formats
        if isinstance(batch_data, tuple):
            if len(batch_data) == 2:
                mol_batch, features_batch = batch_data
                atom_descriptors_batch = atom_features_batch = bond_features_batch = None
            else:
                # Handle extended format with additional descriptors
                mol_batch = batch_data[0]
                features_batch = batch_data[1] if len(batch_data) > 1 else None
                atom_descriptors_batch = batch_data[2] if len(batch_data) > 2 else None
                atom_features_batch = batch_data[3] if len(batch_data) > 3 else None
                bond_features_batch = batch_data[4] if len(batch_data) > 4 else None
        else:
            mol_batch = batch_data
            features_batch = atom_descriptors_batch = atom_features_batch = bond_features_batch = None
        
        # Encode molecules
        encodings = self.encoder(mol_batch, features_batch, atom_descriptors_batch,
                                atom_features_batch, bond_features_batch)
        
        # Pass through FFN and get features
        if hasattr(self.ffn, 'predict'):
            output, features = self.ffn.predict(encodings)
        else:
            # Fallback for standard FFN (shouldn't happen in CBP mode)
            output = self.ffn(encodings)
            features = []
        
        # Apply output transformations
        if self.classification and not (self.training and self.no_training_normalization) and self.loss_function != 'dirichlet':
            output = self.sigmoid(output)
        
        if self.multiclass:
            output = output.reshape((output.shape[0], -1, self.num_classes))
            if not (self.training and self.no_training_normalization) and self.loss_function != 'dirichlet':
                output = self.multiclass_softmax(output)
        
        # Handle special loss functions
        if self.loss_function == 'mve':
            means, variances = torch.split(output, output.shape[1] // 2, dim=1)
            variances = self.softplus(variances)
            output = torch.cat([means, variances], axis=1)
        elif self.loss_function == 'evidential':
            means, lambdas, alphas, betas = torch.split(output, output.shape[1] // 4, dim=1)
            lambdas = self.softplus(lambdas)
            alphas = self.softplus(alphas) + 1
            betas = self.softplus(betas)
            output = torch.cat([means, lambdas, alphas, betas], dim=1)
        elif self.loss_function == 'dirichlet':
            output = nn.functional.softplus(output) + 1
        
        return output, features

    def fingerprint(self,
                    batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                    features_batch: List[np.ndarray] = None,
                    atom_descriptors_batch: List[np.ndarray] = None,
                    atom_features_batch: List[np.ndarray] = None,
                    bond_features_batch: List[np.ndarray] = None,
                    fingerprint_type: str = 'MPN') -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == 'MPN':
            return self.encoder(batch, features_batch, atom_descriptors_batch,
                                atom_features_batch, bond_features_batch)
        elif fingerprint_type == 'last_FFN':
            encodings = self.encoder(batch, features_batch, atom_descriptors_batch,
                                   atom_features_batch, bond_features_batch)
            if self.use_cbp:
                # For CBP model, access the Sequential inside FFNNetworkCBP
                return self.ffn.layers[:-1](encodings)
            else:
                # For standard model, ffn is already Sequential
                return self.ffn[:-1](encodings)
        else:
            raise ValueError(f'Unsupported fingerprint type {fingerprint_type}.')

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions
        """

        output = self.ffn(self.encoder(batch, features_batch, atom_descriptors_batch,
                                       atom_features_batch, bond_features_batch))

        if self.classification and not (self.training and self.no_training_normalization) and self.loss_function != 'dirichlet':
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.shape[0], -1, self.num_classes))  # batch size x num targets x num classes per target
            if not (self.training and self.no_training_normalization) and self.loss_function != 'dirichlet':
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training when using CrossEntropyLoss

        # Modify multi-input loss functions
        if self.loss_function == 'mve':
            means, variances = torch.split(output, output.shape[1] // 2, dim=1)
            variances = self.softplus(variances)
            output = torch.cat([means, variances], axis=1)
        if self.loss_function == 'evidential':
            means, lambdas, alphas, betas = torch.split(output, output.shape[1]//4, dim=1)
            lambdas = self.softplus(lambdas)  # + min_val
            alphas = self.softplus(alphas) + 1  # + min_val # add 1 for numerical contraints of Gamma function
            betas = self.softplus(betas)  # + min_val
            output = torch.cat([means, lambdas, alphas, betas], dim=1)
        if self.loss_function == 'dirichlet':
            output = nn.functional.softplus(output) + 1

        return output

    def get_all_cbp_layers(self):
        """Collect all CBPLinear layers from both MPN and FFN."""
        cbp_layers = []

        # Collect from FFN
        if hasattr(self.ffn, 'get_cbp_layers'):
            cbp_layers.extend(self.ffn.get_cbp_layers())

        # Collect from MPN encoders
        if hasattr(self.encoder, 'encoder'):
            # Handle both single encoder and ModuleList of encoders
            encoders = self.encoder.encoder if isinstance(self.encoder.encoder, nn.ModuleList) else [self.encoder.encoder]

            for encoder in encoders:
                # Check for CBPLinear layers in each encoder
                for name, module in encoder.named_modules():
                    if isinstance(module, CBPLinear):
                        cbp_layers.append(module)

        # Also check reaction solvent encoder if present
        if hasattr(self.encoder, 'encoder_solvent'):
            for name, module in self.encoder.encoder_solvent.named_modules():
                if isinstance(module, CBPLinear):
                    cbp_layers.append(module)

        return cbp_layers


# Create an alias for backward compatibility
MoleculeModelCBP = MoleculeModel