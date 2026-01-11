"""
Prédicteur - Charge les modèles et fait les prédictions.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import MODELS_DIR, FEATURE_NAMES

logger = logging.getLogger(__name__)

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    logger.error("joblib requis pour charger les modèles!")

try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn non disponible - preprocessing limité")


class FPPredictor:
    """
    Prédicteur pour ex_max et em_max.
    """
    
    _instance = None
    
    def __new__(cls, models_dir: Optional[Path] = None):
        """Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, models_dir: Optional[Path] = None):
        if self._initialized:
            return
        
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.model_ex_max = None
        self.model_em_max = None
        self.imputer = None
        self.scaler = None
        self.ready = False
        
        self._load_models()
        self._initialized = True
    
    def _load_models(self):
        """Charge les modèles joblib."""
        if not HAS_JOBLIB:
            logger.error("joblib non disponible!")
            return
        
        ex_path = self.models_dir / "specialized_ex_max.joblib"
        em_path = self.models_dir / "specialized_em_max.joblib"
        
        if not ex_path.exists():
            logger.error(f"Modèle ex_max non trouvé: {ex_path}")
            return
        
        if not em_path.exists():
            logger.error(f"Modèle em_max non trouvé: {em_path}")
            return
        
        try:
            logger.info(f"Chargement de {ex_path.name}...")
            self.model_ex_max = joblib.load(ex_path)
            
            logger.info(f"Chargement de {em_path.name}...")
            self.model_em_max = joblib.load(em_path)
            
            # Preprocessing
            if HAS_SKLEARN:
                self.imputer = SimpleImputer(strategy='median')
                self.scaler = StandardScaler()
            
            self.ready = True
            logger.info("✅ Modèles chargés!")
            
        except Exception as e:
            logger.error(f"Erreur chargement: {e}")
    
    def predict(self, features: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Prédit ex_max et em_max à partir des features.
        
        Args:
            features: Dict avec les 113 features
        
        Returns:
            Tuple (ex_max, em_max) en nm
        """
        if not self.ready:
            logger.error("Modèles non chargés!")
            return None, None
        
        # Construire le vecteur dans le bon ordre
        X = np.array([features.get(f, np.nan) for f in FEATURE_NAMES]).reshape(1, -1)
        
        # Preprocessing
        if HAS_SKLEARN and self.imputer and self.scaler:
            # Note: En production, l'imputer et scaler devraient être
            # fittés sur les données d'entraînement et sauvegardés
            # Ici on fait un workaround simple
            X = np.nan_to_num(X, nan=0.0)
        else:
            X = np.nan_to_num(X, nan=0.0)
        
        try:
            ex_max = float(self.model_ex_max.predict(X)[0])
            em_max = float(self.model_em_max.predict(X)[0])
            
            return ex_max, em_max
            
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            return None, None
    
    def get_info(self) -> Dict:
        """Retourne les infos sur les modèles."""
        return {
            "ready": self.ready,
            "models_dir": str(self.models_dir),
            "n_features_expected": len(FEATURE_NAMES),
            "models": {
                "ex_max": {
                    "loaded": self.model_ex_max is not None,
                    "type": type(self.model_ex_max).__name__ if self.model_ex_max else None,
                    "n_estimators": getattr(self.model_ex_max, 'n_estimators', None)
                },
                "em_max": {
                    "loaded": self.model_em_max is not None,
                    "type": type(self.model_em_max).__name__ if self.model_em_max else None,
                    "n_estimators": getattr(self.model_em_max, 'n_estimators', None)
                }
            }
        }


# Instance globale
_predictor: Optional[FPPredictor] = None


def get_predictor(models_dir: Optional[Path] = None) -> FPPredictor:
    """Retourne l'instance singleton du prédicteur."""
    global _predictor
    if _predictor is None:
        _predictor = FPPredictor(models_dir)
    return _predictor
