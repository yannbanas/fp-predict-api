"""
Parser PDB avancé pour extraction de features structuraux.
Basé sur 10_chromophore_advanced.py du projet master.
Version corrigée: ajout chromophore_triplet + fix détection chemin/contenu
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import StringIO

from .config import AA_MAP, CHROMOPHORE_PATTERNS


class PDBParser:
    """
    Parser PDB pour extraction de features structuraux.
    Supporte les fichiers et le contenu brut.
    """
    
    def __init__(self, source):
        """
        Args:
            source: Path vers fichier PDB ou contenu string
        """
        self.atoms: List[Dict] = []
        self.residues: Dict[int, Dict] = {}
        self.sequence: str = ""
        self.chromophore_triplet: Optional[str] = None  # Ex: "SYG", "TYG"
        self._chromophore_ids: Optional[Tuple[int, int, int]] = None
        
        # Détecter si c'est un chemin ou du contenu PDB
        # Un contenu PDB contient des newlines ou commence par ATOM/MODEL/HEADER
        is_content = (
            isinstance(source, str) and (
                '\n' in source or
                source.startswith('ATOM') or
                source.startswith('MODEL') or
                source.startswith('HEADER') or
                source.startswith('REMARK') or
                len(source) > 500  # Les chemins sont courts
            )
        )
        
        if isinstance(source, Path):
            # C'est explicitement un Path
            with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                self._parse(f)
        elif is_content:
            # C'est du contenu PDB brut
            self._parse(StringIO(source))
        elif isinstance(source, str) and Path(source).exists():
            # C'est un chemin string vers un fichier existant
            with open(source, 'r', encoding='utf-8', errors='ignore') as f:
                self._parse(f)
        elif isinstance(source, str):
            # Fallback: traiter comme contenu même si pas de newline
            self._parse(StringIO(source))
        else:
            raise ValueError("Source doit être un chemin ou contenu PDB")
    
    def _parse(self, file_handle):
        """Parse le contenu PDB."""
        for line in file_handle:
            if line.startswith('ATOM'):
                atom = self._parse_atom_line(line)
                if atom:
                    self.atoms.append(atom)
                    
                    res_id = atom['res_id']
                    if res_id not in self.residues:
                        self.residues[res_id] = {
                            'name': atom['res_name'],
                            'aa': AA_MAP.get(atom['res_name'], 'X'),
                            'atoms': {}
                        }
                    self.residues[res_id]['atoms'][atom['name']] = atom
        
        # Construire la séquence
        sorted_ids = sorted(self.residues.keys())
        self.sequence = ''.join(self.residues[i]['aa'] for i in sorted_ids)
    
    def _parse_atom_line(self, line: str) -> Optional[Dict]:
        """Parse une ligne ATOM."""
        try:
            coord = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54])
            ])
            return {
                'name': line[12:16].strip(),
                'res_name': line[17:20].strip(),
                'res_id': int(line[22:26]),
                'x': coord[0],
                'y': coord[1],
                'z': coord[2],
                'b_factor': float(line[60:66]) if len(line) > 65 else 0.0,
                'coord': coord
            }
        except:
            return None
    
    def get_atom_coord(self, res_id: int, atom_name: str) -> Optional[np.ndarray]:
        """Récupère les coordonnées d'un atome."""
        if res_id in self.residues and atom_name in self.residues[res_id]['atoms']:
            return self.residues[res_id]['atoms'][atom_name]['coord']
        return None
    
    def get_residue_atoms(self, res_id: int) -> Dict:
        """Récupère tous les atomes d'un résidu."""
        if res_id in self.residues:
            return self.residues[res_id]['atoms']
        return {}
    
    def find_chromophore(self) -> Optional[Tuple[int, int, int]]:
        """
        Trouve le chromophore (tripeptide X-Y-G).
        Retourne (res_id_X, res_id_Y, res_id_G).
        Met aussi à jour self.chromophore_triplet.
        """
        # Retourner le cache si déjà trouvé
        if self._chromophore_ids is not None:
            return self._chromophore_ids
        
        sorted_ids = sorted(self.residues.keys())
        
        # Chercher les patterns connus
        for i in range(len(sorted_ids) - 2):
            triplet = ''.join(
                self.residues[sorted_ids[j]]['aa'] 
                for j in range(i, i + 3)
            )
            
            if triplet in CHROMOPHORE_PATTERNS:
                self._chromophore_ids = (sorted_ids[i], sorted_ids[i+1], sorted_ids[i+2])
                self.chromophore_triplet = triplet
                return self._chromophore_ids
        
        # Fallback: chercher X-Y-G (Tyrosine centrale)
        for i in range(len(sorted_ids) - 2):
            aa0 = self.residues[sorted_ids[i]]['aa']
            aa1 = self.residues[sorted_ids[i+1]]['aa']
            aa2 = self.residues[sorted_ids[i+2]]['aa']
            
            # Y au centre, G à la fin
            if aa1 in 'YWHF' and aa2 == 'G':
                triplet = aa0 + aa1 + aa2
                self._chromophore_ids = (sorted_ids[i], sorted_ids[i+1], sorted_ids[i+2])
                self.chromophore_triplet = triplet
                return self._chromophore_ids
        
        # Fallback ultime: chercher juste Y-G
        for i in range(len(sorted_ids) - 1):
            aa1 = self.residues[sorted_ids[i]]['aa']
            aa2 = self.residues[sorted_ids[i+1]]['aa']
            
            if aa1 == 'Y' and aa2 == 'G':
                if i > 0:
                    aa0 = self.residues[sorted_ids[i-1]]['aa']
                    triplet = aa0 + aa1 + aa2
                    self._chromophore_ids = (sorted_ids[i-1], sorted_ids[i], sorted_ids[i+1])
                    self.chromophore_triplet = triplet
                    return self._chromophore_ids
        
        return None
    
    def get_chromophore_position(self) -> Optional[int]:
        """Retourne l'index (0-based) de la Tyrosine du chromophore."""
        chrom = self.find_chromophore()
        if chrom is None:
            return None
        
        sorted_ids = sorted(self.residues.keys())
        try:
            return sorted_ids.index(chrom[1])  # Position de la Tyrosine
        except ValueError:
            return None