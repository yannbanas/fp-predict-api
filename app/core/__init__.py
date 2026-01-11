"""
Core module - Feature extraction et prediction.
Version corrigée: sans ESM-2
"""

from .pdb_parser import PDBParser
from .feature_extractor import extract_all_features
from .predictor import get_predictor, FPPredictor

# ESM-2 supprimé - non utilisé par le modèle actuel

__all__ = [
    'PDBParser',
    'extract_all_features',
    'get_predictor',
    'FPPredictor',
]