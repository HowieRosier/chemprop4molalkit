from .model import MoleculeModel, MoleculeModelCBP  # MoleculeModelCBP is now an alias
from .mpn import MPN, MPNEncoder

__all__ = [
    'MoleculeModel',
    'MoleculeModelCBP',  # Keep for backward compatibility
    'MPN',
    'MPNEncoder'
]
