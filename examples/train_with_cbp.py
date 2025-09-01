#!/usr/bin/env python
"""
Example script for training a ChemProp model with Continual Backpropagation (CBP).

This script demonstrates how to use the CBP functionality integrated into ChemProp4MolALKit
for molecular property prediction tasks.
"""

import os
import sys

# Example command to train with CBP
def main():
    # Basic CBP training command
    basic_cbp_command = """
    chemprop_train \\
        --data_path data/tox21.csv \\
        --dataset_type classification \\
        --save_dir models_cbp/tox21_cbp \\
        --cbp \\
        --replacement_rate 0.0001 \\
        --decay_rate 0.99 \\
        --maturity_threshold 20 \\
        --util_type contribution \\
        --cbp_init kaiming \\
        --epochs 30 \\
        --batch_size 50
    """
    
    # Advanced CBP training with hyperparameter tuning
    advanced_cbp_command = """
    chemprop_train \\
        --data_path data/qm9.csv \\
        --dataset_type regression \\
        --save_dir models_cbp/qm9_cbp \\
        --cbp \\
        --replacement_rate 0.0005 \\
        --decay_rate 0.95 \\
        --maturity_threshold 50 \\
        --util_type contribution \\
        --cbp_init xavier \\
        --optimizer adam \\
        --init_lr 0.0001 \\
        --max_lr 0.001 \\
        --final_lr 0.0001 \\
        --epochs 50 \\
        --batch_size 32 \\
        --hidden_size 300 \\
        --ffn_hidden_size 300 \\
        --ffn_num_layers 3 \\
        --dropout 0.1
    """
    
    print("ChemProp4MolALKit with Continual Backpropagation (CBP) Example")
    print("=" * 60)
    print("\nCBP is a technique that maintains neural network plasticity during training")
    print("by selectively replacing low-utility neurons with newly initialized ones.")
    print("\nKey CBP Parameters:")
    print("  --cbp: Enable CBP training")
    print("  --replacement_rate: Fraction of neurons to replace (default: 0.0001)")
    print("  --decay_rate: Decay rate for utility tracking (default: 0.99)")
    print("  --maturity_threshold: Minimum age for neuron replacement (default: 20)")
    print("  --util_type: Utility calculation method (default: 'contribution')")
    print("  --cbp_init: Weight initialization for new neurons (default: 'kaiming')")
    
    print("\n\nBasic CBP Training Command:")
    print(basic_cbp_command)
    
    print("\n\nAdvanced CBP Training Command:")
    print(advanced_cbp_command)
    
    print("\n\nTo run CBP training, execute one of the above commands in your terminal.")
    print("Make sure you have your data file ready and adjust paths accordingly.")
    
    # Example of how to run CBP programmatically
    print("\n\nProgrammatic CBP Usage:")
    print("-" * 40)
    
    example_code = '''
from chemprop.args import TrainArgs
from chemprop.train.cross_validate import cross_validate
from chemprop.train.run_training_cbp import run_training_cbp

# Set up arguments
args = TrainArgs()
args.data_path = 'data/tox21.csv'
args.dataset_type = 'classification'
args.save_dir = 'models_cbp/tox21_cbp'
args.cbp = True
args.replacement_rate = 0.0001
args.decay_rate = 0.99
args.maturity_threshold = 20
args.util_type = 'contribution'
args.cbp_init = 'kaiming'
args.epochs = 30

# Run training with CBP
cross_validate(args=args, train_func=run_training_cbp)
    '''
    
    print(example_code)
    
    print("\n\nBenefits of CBP:")
    print("1. Maintains network plasticity throughout training")
    print("2. Prevents dead neurons from accumulating")
    print("3. Can improve generalization on molecular property tasks")
    print("4. Particularly useful for continual learning scenarios")
    print("5. Well-integrated with MolALKit for active learning workflows")


if __name__ == "__main__":
    main() 