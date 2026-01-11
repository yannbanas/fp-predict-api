"""
Gestionnaire de jobs asynchrones pour le traitement PDB.
Version corrigée: 85 features (sans ESM-2)
"""

import asyncio
import uuid
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from ..core.pdb_parser import PDBParser
from ..core.feature_extractor import extract_all_features
from ..core.predictor import get_predictor
# ESM-2 supprimé - non utilisé par le modèle

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobResult:
    """Résultat d'un job de prédiction."""
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    # Input
    filename: Optional[str] = None
    
    # Résultats
    ex_max: Optional[float] = None
    em_max: Optional[float] = None
    stokes_shift: Optional[float] = None
    
    # Métadonnées extraites
    sequence: Optional[str] = None
    sequence_length: Optional[int] = None
    chromophore_found: bool = False
    chromophore_triplet: Optional[str] = None
    chromophore_position: Optional[int] = None
    n_residues: Optional[int] = None
    plddt_mean: Optional[float] = None
    
    # Erreur si échec
    error: Optional[str] = None
    
    # Callbacks WebSocket
    _callbacks: list = field(default_factory=list, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "filename": self.filename,
            "results": {
                "ex_max_nm": self.ex_max,
                "em_max_nm": self.em_max,
                "stokes_shift_nm": self.stokes_shift,
            } if self.status == JobStatus.COMPLETED else None,
            "metadata": {
                "sequence": self.sequence[:50] + "..." if self.sequence and len(self.sequence) > 50 else self.sequence,
                "sequence_length": self.sequence_length,
                "chromophore_found": self.chromophore_found,
                "chromophore_triplet": self.chromophore_triplet,
                "chromophore_position": self.chromophore_position,
                "n_residues": self.n_residues,
                "plddt_mean": round(self.plddt_mean, 2) if self.plddt_mean else None,
            } if self.status == JobStatus.COMPLETED else None,
            "error": self.error,
        }


class JobManager:
    """
    Gestionnaire de jobs asynchrones.
    Stocke les jobs en mémoire (pour prod, utiliser Redis).
    """
    
    def __init__(self, max_jobs: int = 1000):
        self.jobs: Dict[str, JobResult] = {}
        self.max_jobs = max_jobs
        self._lock = asyncio.Lock()
        self._callbacks: Dict[str, list] = {}
    
    async def create_job(self, filename: Optional[str] = None) -> JobResult:
        """Crée un nouveau job."""
        async with self._lock:
            if len(self.jobs) >= self.max_jobs:
                await self._cleanup_old_jobs()
            
            job_id = str(uuid.uuid4())[:8]
            job = JobResult(
                job_id=job_id,
                status=JobStatus.PENDING,
                created_at=datetime.now(),
                filename=filename
            )
            self.jobs[job_id] = job
            self._callbacks[job_id] = []
            
            logger.info(f"Job créé: {job_id}")
            return job
    
    async def get_job(self, job_id: str) -> Optional[JobResult]:
        """Récupère un job par ID."""
        return self.jobs.get(job_id)
    
    async def update_job(self, job_id: str, **kwargs):
        """Met à jour un job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                await self._notify_callbacks(job_id)
    
    def register_callback(self, job_id: str, callback: Callable):
        """Enregistre un callback pour être notifié quand le job est terminé."""
        if job_id in self._callbacks:
            self._callbacks[job_id].append(callback)
    
    def unregister_callback(self, job_id: str, callback: Callable):
        """Retire un callback."""
        if job_id in self._callbacks and callback in self._callbacks[job_id]:
            self._callbacks[job_id].remove(callback)
    
    async def _notify_callbacks(self, job_id: str):
        """Notifie tous les callbacks enregistrés pour un job."""
        if job_id in self._callbacks:
            job = self.jobs[job_id]
            for callback in self._callbacks[job_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(job)
                    else:
                        callback(job)
                except Exception as e:
                    logger.error(f"Erreur callback: {e}")
    
    async def _cleanup_old_jobs(self):
        """Nettoie les jobs les plus anciens."""
        if len(self.jobs) < self.max_jobs // 2:
            return
        
        sorted_jobs = sorted(
            self.jobs.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )
        
        keep = sorted_jobs[:self.max_jobs // 2]
        self.jobs = dict(keep)
        self._callbacks = {k: v for k, v in self._callbacks.items() if k in self.jobs}
        
        logger.info(f"Nettoyage: {len(self.jobs)} jobs conservés")


# Instance globale
job_manager = JobManager()


async def process_pdb_job(job_id: str, pdb_content: str):
    """
    Traite un fichier PDB de manière asynchrone.
    
    Pipeline simplifié (85 features, sans ESM-2):
    1. Parser PDB
    2. Extraire features structuraux + RDKit + classification
    3. Prédiction ExtraTrees
    
    Args:
        job_id: ID du job
        pdb_content: Contenu brut du fichier PDB
    """
    try:
        # Marquer comme en cours
        await job_manager.update_job(job_id, status=JobStatus.PROCESSING)
        
        logger.info(f"[{job_id}] Début du traitement...")
        
        # 1. Parser le PDB
        logger.info(f"[{job_id}] Parsing PDB...")
        pdb = PDBParser(pdb_content)
        
        if not pdb.residues:
            raise ValueError("Fichier PDB invalide ou vide")
        
        # 2. Extraire les 85 features (structuraux + RDKit + classification)
        logger.info(f"[{job_id}] Extraction des 85 features...")
        features = extract_all_features(pdb)
        
        n_features = len(features)
        logger.info(f"[{job_id}] {n_features} features extraites")
        
        if n_features != 85:
            logger.warning(f"[{job_id}] Attendu 85 features, obtenu {n_features}")
        
        # 3. Prédiction
        logger.info(f"[{job_id}] Prédiction...")
        predictor = get_predictor()
        ex_max, em_max = predictor.predict(features)
        
        if ex_max is None or em_max is None:
            raise ValueError("Échec de la prédiction")
        
        # 4. Récupérer les métadonnées
        chrom_found = features.get('chrom_found', 0) == 1
        chrom_pos = int(features.get('chrom_position', 0)) if chrom_found else None
        
        # 5. Mettre à jour le job avec les résultats
        await job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(),
            ex_max=round(ex_max, 1),
            em_max=round(em_max, 1),
            stokes_shift=round(em_max - ex_max, 1),
            sequence=pdb.sequence,
            sequence_length=len(pdb.sequence) if pdb.sequence else 0,
            chromophore_found=chrom_found,
            chromophore_triplet=pdb.chromophore_triplet,
            chromophore_position=chrom_pos,
            n_residues=len(pdb.residues),
            plddt_mean=features.get('glob_plddt_mean')
        )
        
        logger.info(f"[{job_id}] ✅ Terminé: ex={ex_max:.1f}nm, em={em_max:.1f}nm")
        
    except Exception as e:
        logger.error(f"[{job_id}] ❌ Erreur: {e}")
        await job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now(),
            error=str(e)
        )