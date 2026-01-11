"""
Extracteur de features structuraux et chimiques.
Version corrigée: 85 features (54 PDB + 24 RDKit + 7 classification)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from .config import (
    POLAR_RESIDUES, AROMATIC_RESIDUES, HYDROPHOBIC_RESIDUES,
    CHARGED_POS, CHARGED_NEG, HYDROPATHY,
    CHROMOPHORE_SMILES, CHROMOPHORE_PATTERNS,
    GFP_LIKE_TRIPLETS, RFP_LIKE_TRIPLETS, CFP_LIKE_TRIPLETS, BFP_LIKE_TRIPLETS,
    FEATURE_NAMES
)
from .pdb_parser import PDBParser

logger = logging.getLogger(__name__)

# RDKit (optionnel mais recommandé)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem import rdMolDescriptors, Crippen, MolSurf
    from rdkit.Chem.EState import EState
    from rdkit.Chem import GraphDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit non disponible - features chimiques désactivées")


# =============================================================================
# CALCULS GÉOMÉTRIQUES
# =============================================================================

def calculate_dihedral(p1: np.ndarray, p2: np.ndarray,
                       p3: np.ndarray, p4: np.ndarray) -> float:
    """Calcule l'angle dihédral entre 4 points (en degrés)."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1 = n1 / (np.linalg.norm(n1) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2) + 1e-10)
    
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.degrees(np.arctan2(y, x))


def calculate_planarity(coords: List[np.ndarray]) -> float:
    """Calcule la planarité (RMSD par rapport au plan)."""
    if len(coords) < 4:
        return 0.0
    
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    
    try:
        _, s, vh = np.linalg.svd(centered)
        normal = vh[-1]
        distances = np.abs(np.dot(centered, normal))
        return np.sqrt(np.mean(distances**2))
    except:
        return np.nan


# =============================================================================
# FEATURES STRUCTURAUX (54 features)
# =============================================================================

def extract_structural_features(pdb: PDBParser, chrom_ids: Optional[Tuple[int, int, int]]) -> Dict:
    """
    Extrait les 54 features structuraux du PDB.
    """
    features = {}
    sequence = pdb.sequence
    
    # =========================================================================
    # CHROMOPHORE (8 features)
    # =========================================================================
    
    if chrom_ids:
        res_x, res_y, res_g = chrom_ids
        features['chrom_found'] = 1
        
        sorted_ids = sorted(pdb.residues.keys())
        features['chrom_position'] = sorted_ids.index(res_y) if res_y in sorted_ids else np.nan
        
        # Angles dihédraux
        c_prev = pdb.get_atom_coord(res_x, 'C')
        n_y = pdb.get_atom_coord(res_y, 'N')
        ca_y = pdb.get_atom_coord(res_y, 'CA')
        c_y = pdb.get_atom_coord(res_y, 'C')
        n_next = pdb.get_atom_coord(res_g, 'N')
        cb_y = pdb.get_atom_coord(res_y, 'CB')
        cg_y = pdb.get_atom_coord(res_y, 'CG')
        
        # Phi
        if all(x is not None for x in [c_prev, n_y, ca_y, c_y]):
            features['chrom_phi_tyr'] = calculate_dihedral(c_prev, n_y, ca_y, c_y)
        else:
            features['chrom_phi_tyr'] = np.nan
        
        # Psi
        if all(x is not None for x in [n_y, ca_y, c_y, n_next]):
            features['chrom_psi_tyr'] = calculate_dihedral(n_y, ca_y, c_y, n_next)
        else:
            features['chrom_psi_tyr'] = np.nan
        
        # Chi1
        if all(x is not None for x in [n_y, ca_y, cb_y, cg_y]):
            features['chrom_chi1_tyr'] = calculate_dihedral(n_y, ca_y, cb_y, cg_y)
        else:
            features['chrom_chi1_tyr'] = np.nan
        
        # Ring planarity
        ring_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
        ring_coords = [pdb.get_atom_coord(res_y, a) for a in ring_atoms]
        ring_coords = [c for c in ring_coords if c is not None]
        features['chrom_ring_planarity'] = calculate_planarity(ring_coords) if len(ring_coords) >= 4 else np.nan
        
        # pLDDT chromophore
        plddt_chrom = []
        for rid in chrom_ids:
            atoms = pdb.get_residue_atoms(rid)
            plddt_chrom.extend([a['b_factor'] for a in atoms.values()])
        
        features['chrom_plddt_mean'] = np.mean(plddt_chrom) if plddt_chrom else np.nan
        features['chrom_plddt_min'] = np.min(plddt_chrom) if plddt_chrom else np.nan
    else:
        features['chrom_found'] = 0
        for f in ['chrom_position', 'chrom_phi_tyr', 'chrom_psi_tyr', 'chrom_chi1_tyr',
                  'chrom_ring_planarity', 'chrom_plddt_mean', 'chrom_plddt_min']:
            features[f] = np.nan
    
    # =========================================================================
    # ENVIRONNEMENT (10 features)
    # =========================================================================
    
    if chrom_ids:
        ca_y = pdb.get_atom_coord(chrom_ids[1], 'CA')
        if ca_y is not None:
            sorted_ids = sorted(pdb.residues.keys())
            chrom_set = set(chrom_ids)
            
            neighbors = []
            n_hydro = n_polar = n_pos = n_neg = n_arom = 0
            plddt_env = []
            
            for res_id in sorted_ids:
                if res_id in chrom_set:
                    continue
                
                ca = pdb.get_atom_coord(res_id, 'CA')
                if ca is None:
                    continue
                
                dist = np.linalg.norm(ca - ca_y)
                if dist < 5.0:
                    neighbors.append(res_id)
                    aa = pdb.residues[res_id]['aa']
                    
                    if aa in HYDROPHOBIC_RESIDUES: n_hydro += 1
                    if aa in POLAR_RESIDUES: n_polar += 1
                    if aa in CHARGED_POS: n_pos += 1
                    if aa in CHARGED_NEG: n_neg += 1
                    if aa in AROMATIC_RESIDUES: n_arom += 1
                    
                    atoms = pdb.get_residue_atoms(res_id)
                    plddt_env.extend([a['b_factor'] for a in atoms.values()])
            
            total = max(len(neighbors), 1)
            features['env_n_neighbors'] = len(neighbors)
            features['env_n_hydrophobic'] = n_hydro
            features['env_n_polar'] = n_polar
            features['env_n_charged_pos'] = n_pos
            features['env_n_charged_neg'] = n_neg
            features['env_n_aromatic'] = n_arom
            features['env_ratio_hydrophobic'] = n_hydro / total
            features['env_ratio_polar'] = n_polar / total
            features['env_net_charge'] = n_pos - n_neg
            features['env_plddt_mean'] = np.mean(plddt_env) if plddt_env else np.nan
        else:
            for f in ['env_n_neighbors', 'env_n_hydrophobic', 'env_n_polar',
                      'env_n_charged_pos', 'env_n_charged_neg', 'env_n_aromatic',
                      'env_ratio_hydrophobic', 'env_ratio_polar',
                      'env_net_charge', 'env_plddt_mean']:
                features[f] = np.nan
    else:
        for f in ['env_n_neighbors', 'env_n_hydrophobic', 'env_n_polar',
                  'env_n_charged_pos', 'env_n_charged_neg', 'env_n_aromatic',
                  'env_ratio_hydrophobic', 'env_ratio_polar',
                  'env_net_charge', 'env_plddt_mean']:
            features[f] = np.nan
    
    # =========================================================================
    # GLOBAL (11 features)
    # =========================================================================
    
    features['glob_n_residues'] = len(pdb.residues)
    features['glob_n_atoms'] = len(pdb.atoms)
    
    # Rayon de giration
    coords = np.array([a['coord'] for a in pdb.atoms])
    if len(coords) > 0:
        centroid = coords.mean(axis=0)
        features['glob_radius_gyration'] = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
        
        # Extent
        features['glob_extent_x'] = coords[:, 0].max() - coords[:, 0].min()
        features['glob_extent_y'] = coords[:, 1].max() - coords[:, 1].min()
        features['glob_extent_z'] = coords[:, 2].max() - coords[:, 2].min()
    else:
        features['glob_radius_gyration'] = np.nan
        features['glob_extent_x'] = np.nan
        features['glob_extent_y'] = np.nan
        features['glob_extent_z'] = np.nan
    
    # pLDDT global
    all_plddt = [a['b_factor'] for a in pdb.atoms]
    if all_plddt:
        features['glob_plddt_mean'] = np.mean(all_plddt)
        features['glob_plddt_std'] = np.std(all_plddt)
        features['glob_plddt_min'] = np.min(all_plddt)
        features['glob_frac_plddt_70'] = np.mean(np.array(all_plddt) > 70)
        features['glob_frac_plddt_90'] = np.mean(np.array(all_plddt) > 90)
    else:
        for f in ['glob_plddt_mean', 'glob_plddt_std', 'glob_plddt_min',
                  'glob_frac_plddt_70', 'glob_frac_plddt_90']:
            features[f] = np.nan
    
    # =========================================================================
    # SEQUENCE (25 features)
    # =========================================================================
    
    seq_len = len(sequence) if sequence else 0
    features['seq_length'] = seq_len
    
    if seq_len > 0:
        # Fréquences AA
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            features[f'seq_freq_{aa}'] = sequence.count(aa) / seq_len
        
        # Fractions
        n_hydro = sum(sequence.count(aa) for aa in HYDROPHOBIC_RESIDUES)
        n_polar = sum(sequence.count(aa) for aa in POLAR_RESIDUES)
        n_arom = sum(sequence.count(aa) for aa in AROMATIC_RESIDUES)
        
        features['seq_frac_hydrophobic'] = n_hydro / seq_len
        features['seq_frac_polar'] = n_polar / seq_len
        features['seq_frac_aromatic'] = n_arom / seq_len
        
        # GRAVY
        gravy = sum(HYDROPATHY.get(aa, 0) for aa in sequence) / seq_len
        features['seq_gravy'] = gravy
    else:
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            features[f'seq_freq_{aa}'] = 0.0
        features['seq_frac_hydrophobic'] = 0.0
        features['seq_frac_polar'] = 0.0
        features['seq_frac_aromatic'] = 0.0
        features['seq_gravy'] = 0.0
    
    return features


# =============================================================================
# FEATURES RDKIT (24 features)
# =============================================================================

def get_chromophore_smiles(triplet: str) -> Optional[str]:
    """Retourne le SMILES du chromophore."""
    if triplet in CHROMOPHORE_SMILES:
        return CHROMOPHORE_SMILES[triplet]
    
    # Fallback sur le type central
    if len(triplet) > 1:
        central = triplet[1]
        if central == 'Y':
            return CHROMOPHORE_SMILES.get('YG')
        elif central == 'W':
            return CHROMOPHORE_SMILES.get('WG')
        elif central == 'H':
            return CHROMOPHORE_SMILES.get('HG')
    return None


def compute_rdkit_features(smiles: str) -> Dict[str, float]:
    """
    Calcule les 24 descripteurs RDKit du chromophore.
    """
    features = {
        'n_heavy_atoms': 0.0, 'n_double_bonds': 0.0, 'n_aromatic_bonds': 0.0,
        'n_conjugated_bonds': 0.0, 'n_pi_electrons': 0.0, 'conjugation_ratio': 0.0,
        'n_aromatic_rings': 0.0, 'aromatic_fraction': 0.0, 'logp': 0.0, 'mr': 0.0,
        'tpsa': 0.0, 'charge_max': 0.0, 'charge_min': 0.0, 'charge_range': 0.0,
        'chi0': 0.0, 'chi1': 0.0, 'kappa1': 0.0, 'estate_max': 0.0, 'estate_sum': 0.0,
        'labute_asa': 0.0, 'mol_weight': 0.0, 'n_hbd': 0.0, 'n_hba': 0.0,
        'max_conjugated_chain': 0.0
    }
    
    if not HAS_RDKIT or not smiles:
        return features
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return features
        
        mol = Chem.AddHs(mol)
        
        # Atomes
        features['n_heavy_atoms'] = mol.GetNumHeavyAtoms()
        
        # Liaisons
        n_double = n_aromatic = n_conjugated = 0
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE:
                n_double += 1
            elif bt == Chem.BondType.AROMATIC:
                n_aromatic += 1
            if bond.GetIsConjugated():
                n_conjugated += 1
        
        features['n_double_bonds'] = n_double
        features['n_aromatic_bonds'] = n_aromatic
        features['n_conjugated_bonds'] = n_conjugated
        features['n_pi_electrons'] = 2 * (n_double + n_aromatic)
        
        total_bonds = mol.GetNumBonds()
        features['conjugation_ratio'] = n_conjugated / total_bonds if total_bonds > 0 else 0
        
        # Aromaticité
        features['n_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_arom = len([a for a in mol.GetAtoms() if a.GetIsAromatic()])
        features['aromatic_fraction'] = n_arom / mol.GetNumHeavyAtoms() if mol.GetNumHeavyAtoms() > 0 else 0
        
        # Propriétés
        features['logp'] = Crippen.MolLogP(mol)
        features['mr'] = Crippen.MolMR(mol)
        features['tpsa'] = Descriptors.TPSA(mol)
        
        # Charges Gasteiger
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for a in mol.GetAtoms():
                try:
                    charge = a.GetDoubleProp('_GasteigerCharge')
                    if not np.isnan(charge) and not np.isinf(charge):
                        charges.append(charge)
                except:
                    pass
            
            if charges:
                features['charge_max'] = max(charges)
                features['charge_min'] = min(charges)
                features['charge_range'] = max(charges) - min(charges)
        except:
            pass
        
        # Topologie
        features['chi0'] = GraphDescriptors.Chi0(mol)
        features['chi1'] = GraphDescriptors.Chi1(mol)
        features['kappa1'] = GraphDescriptors.Kappa1(mol)
        
        # EState
        try:
            estate = EState.EStateIndices(mol)
            if len(estate) > 0:
                features['estate_max'] = max(estate)
                features['estate_sum'] = sum(estate)
        except:
            pass
        
        # Surface et masse
        features['labute_asa'] = MolSurf.LabuteASA(mol)
        features['mol_weight'] = Descriptors.MolWt(mol)
        features['n_hbd'] = rdMolDescriptors.CalcNumHBD(mol)
        features['n_hba'] = rdMolDescriptors.CalcNumHBA(mol)
        
        # Système π conjugué
        conj_atoms = set()
        for bond in mol.GetBonds():
            if bond.GetIsConjugated() or bond.GetIsAromatic():
                conj_atoms.add(bond.GetBeginAtomIdx())
                conj_atoms.add(bond.GetEndAtomIdx())
        features['max_conjugated_chain'] = len(conj_atoms)
        
    except Exception as e:
        logger.warning(f"Erreur RDKit: {e}")
    
    return features


# =============================================================================
# FEATURES CLASSIFICATION (7 features)
# =============================================================================

def classify_chromophore(triplet: str) -> Dict[str, int]:
    """
    Classification chimique du chromophore (7 features).
    """
    triplet = triplet.upper() if triplet else ''
    
    return {
        'family_gfp': 1 if triplet in GFP_LIKE_TRIPLETS else 0,
        'family_rfp': 1 if triplet in RFP_LIKE_TRIPLETS else 0,
        'family_cfp': 1 if triplet in CFP_LIKE_TRIPLETS else 0,
        'family_bfp': 1 if triplet in BFP_LIKE_TRIPLETS else 0,
        'central_tyr': 1 if len(triplet) > 1 and triplet[1] == 'Y' else 0,
        'central_trp': 1 if len(triplet) > 1 and triplet[1] == 'W' else 0,
        'central_his': 1 if len(triplet) > 1 and triplet[1] == 'H' else 0,
    }


# =============================================================================
# EXTRACTION COMPLÈTE (85 features)
# =============================================================================

def extract_all_features(pdb: PDBParser) -> Dict:
    """
    Extrait les 85 features dans l'ordre exact du modèle:
    - 54 structuraux (PDB)
    - 24 chimiques (RDKit)
    - 7 classification
    """
    # Trouver le chromophore
    chrom_ids = pdb.find_chromophore()
    triplet = pdb.chromophore_triplet
    
    logger.info(f"Chromophore trouvé: {triplet} (IDs: {chrom_ids})")
    
    # 1. Features structuraux (54)
    structural = extract_structural_features(pdb, chrom_ids)
    
    # 2. Features RDKit (24)
    smiles = get_chromophore_smiles(triplet) if triplet else None
    if smiles:
        logger.info(f"SMILES chromophore: {smiles}")
    rdkit_feat = compute_rdkit_features(smiles) if smiles else compute_rdkit_features(None)
    
    # 3. Classification (7)
    classification = classify_chromophore(triplet)
    
    # Combiner dans l'ordre EXACT de FEATURE_NAMES
    all_features = {}
    all_features.update(structural)
    all_features.update(rdkit_feat)
    all_features.update(classification)
    
    # Vérifier qu'on a toutes les features
    missing = [f for f in FEATURE_NAMES if f not in all_features]
    if missing:
        logger.warning(f"Features manquantes: {missing}")
        for f in missing:
            all_features[f] = np.nan
    
    # Retourner dans le bon ordre
    return {f: all_features.get(f, np.nan) for f in FEATURE_NAMES}