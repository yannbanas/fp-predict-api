"""
Configuration centralisée du service FP-Predict.
Version corrigée: 85 features (sans ESM-2)
"""

from pathlib import Path
from typing import List

# =============================================================================
# CHEMINS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# =============================================================================
# GÉOMÉTRIE
# =============================================================================

HBOND_DISTANCE = 3.5
CAVITY_RADIUS_SMALL = 6.0
CAVITY_RADIUS_LARGE = 8.0

# =============================================================================
# RÉSIDUS
# =============================================================================

POLAR_RESIDUES = set('STNQYDE')
AROMATIC_RESIDUES = set('FYW')
HBOND_DONORS = set('STNQKRHWY')
HBOND_ACCEPTORS = set('STNQDEHY')
HYDROPHOBIC_RESIDUES = set('AVILMFYW')
CHARGED_POS = set('KRH')
CHARGED_NEG = set('DE')

# =============================================================================
# CHROMOPHORE
# =============================================================================

# Patterns connus
CHROMOPHORE_PATTERNS = [
    'SYG', 'TYG', 'GYG', 'AYG', 'CYG', 'MYG', 'VYG', 'LYG', 'QYG',
    'IYG', 'FYG', 'NYG', 'KYG', 'RYG', 'HYG', 'DYG', 'EYG', 'WYG', 'PYG',
    'SWG', 'TWG', 'GWG', 'AWG', 'MWG', 'CWG',
    'SHG', 'THG', 'GHG', 'AHG', 'MHG',
    'SFG', 'TFG', 'GFG', 'AFG',
]

# SMILES des chromophores
CHROMOPHORE_SMILES = {
    'YG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)C',
    'SYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CO',
    'TYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)C(C)O',
    'GYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)[H]',
    'AYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)C',
    'CYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CS',
    'VYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)C(C)C',
    'MYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CCSC',
    'QYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CCC(N)=O',
    'KYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CCCCN',
    'DYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CC(O)=O',
    'LYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CC(C)C',
    'IYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)C(C)CC',
    'NYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CC(N)=O',
    'FYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CC3=CC=CC=C3',
    'WYG': 'O=C1NC(=C(/C=C2C=CC(O)=CC2)N1)CC3=CNC4=CC=CC=C34',
    # CFP (Tryptophane)
    'WG': 'O=C1NC(=C(/C=C2C3=CC=CC=C3NC2)N1)C',
    'SWG': 'O=C1NC(=C(/C=C2C3=CC=CC=C3NC2)N1)CO',
    'TWG': 'O=C1NC(=C(/C=C2C3=CC=CC=C3NC2)N1)C(C)O',
    # BFP (Histidine)
    'HG': 'O=C1NC(=C(/C=C2N=CN=C2)N1)C',
    'SHG': 'O=C1NC(=C(/C=C2N=CN=C2)N1)CO',
}

# Familles de chromophores
GFP_LIKE_TRIPLETS = {'SYG', 'TYG', 'GYG', 'AYG', 'CYG', 'VYG', 'LYG', 'IYG', 'NYG', 'PYG', 'FYG'}
RFP_LIKE_TRIPLETS = {'MYG', 'QYG', 'KYG', 'DYG', 'EYG', 'RYG', 'HYG', 'WYG'}
CFP_LIKE_TRIPLETS = {'SWG', 'TWG', 'GWG', 'AWG', 'MWG', 'CWG'}
BFP_LIKE_TRIPLETS = {'SHG', 'THG', 'GHG', 'AHG', 'MHG'}

# Mapping AA
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# =============================================================================
# HYDROPATHIE (Kyte-Doolittle)
# =============================================================================

HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# =============================================================================
# FEATURES - 85 au total (ordre EXACT du modèle)
# =============================================================================

FEATURE_NAMES: List[str] = [
    # === Chromophore (8) ===
    'chrom_found',
    'chrom_position',
    'chrom_phi_tyr',
    'chrom_psi_tyr',
    'chrom_chi1_tyr',
    'chrom_ring_planarity',
    'chrom_plddt_mean',
    'chrom_plddt_min',
    
    # === Environnement (10) ===
    'env_n_neighbors',
    'env_n_hydrophobic',
    'env_n_polar',
    'env_n_charged_pos',
    'env_n_charged_neg',
    'env_n_aromatic',
    'env_ratio_hydrophobic',
    'env_ratio_polar',
    'env_net_charge',
    'env_plddt_mean',
    
    # === Global structure (11) ===
    'glob_n_residues',
    'glob_n_atoms',
    'glob_radius_gyration',
    'glob_extent_x',
    'glob_extent_y',
    'glob_extent_z',
    'glob_plddt_mean',
    'glob_plddt_std',
    'glob_plddt_min',
    'glob_frac_plddt_70',
    'glob_frac_plddt_90',
    
    # === Sequence frequencies (25) ===
    'seq_freq_A', 'seq_freq_C', 'seq_freq_D', 'seq_freq_E', 'seq_freq_F',
    'seq_freq_G', 'seq_freq_H', 'seq_freq_I', 'seq_freq_K', 'seq_freq_L',
    'seq_freq_M', 'seq_freq_N', 'seq_freq_P', 'seq_freq_Q', 'seq_freq_R',
    'seq_freq_S', 'seq_freq_T', 'seq_freq_V', 'seq_freq_W', 'seq_freq_Y',
    'seq_frac_hydrophobic',
    'seq_frac_polar',
    'seq_frac_aromatic',
    'seq_length',
    'seq_gravy',
    
    # === RDKit chromophore (24) ===
    'n_heavy_atoms',
    'n_double_bonds',
    'n_aromatic_bonds',
    'n_conjugated_bonds',
    'n_pi_electrons',
    'conjugation_ratio',
    'n_aromatic_rings',
    'aromatic_fraction',
    'logp',
    'mr',
    'tpsa',
    'charge_max',
    'charge_min',
    'charge_range',
    'chi0',
    'chi1',
    'kappa1',
    'estate_max',
    'estate_sum',
    'labute_asa',
    'mol_weight',
    'n_hbd',
    'n_hba',
    'max_conjugated_chain',
    
    # === Classification chromophore (7) ===
    'family_gfp',
    'family_rfp',
    'family_cfp',
    'family_bfp',
    'central_tyr',
    'central_trp',
    'central_his',
]

# Vérification
assert len(FEATURE_NAMES) == 85, f"Expected 85 features, got {len(FEATURE_NAMES)}"