"""
FP-Predict API
==============
Microservice pour prédire les spectres d'excitation/émission
des protéines fluorescentes à partir de fichiers PDB.

Features:
- Upload PDB via REST ou WebSocket
- Traitement asynchrone en background
- Push des résultats via WebSocket
- Polling via REST

Version: 2.1.0 (sans ESM-2, 85 features)
Auteur: Panda
Projet: FPbase Master - PNPH
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .core import get_predictor
from .jobs import job_manager, process_pdb_job, JobStatus

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles au démarrage."""
    logger.info("🚀 Démarrage FP-Predict API...")
    
    # Charger le prédicteur (modèles joblib)
    predictor = get_predictor()
    if predictor.ready:
        logger.info("✅ Modèles chargés (85 features, sans ESM-2)")
    else:
        logger.warning("⚠️ Modèles non disponibles!")
    
    yield
    
    logger.info("👋 Arrêt FP-Predict API")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="FP-Predict API",
    description="""
    🧬 **API de prédiction spectrale pour protéines fluorescentes**
    
    Upload un fichier PDB et obtenez:
    - **Excitation Max** (ex_max): Longueur d'onde d'excitation optimale
    - **Émission Max** (em_max): Longueur d'onde d'émission optimale
    - **Stokes Shift**: Différence entre émission et excitation
    
    ## Utilisation
    
    1. **POST /jobs/submit** - Upload un PDB, reçoit un job_id
    2. **WebSocket /ws/{job_id}** - Reçoit le résultat en push
    3. **GET /jobs/{job_id}** - Polling du statut (alternative)
    
    ## Modèle
    
    - ExtraTrees Regressor (85 features)
    - Features: structural + RDKit + classification
    - MAE: ~17nm (émission), ~21nm (excitation)
    
    Développé dans le cadre du projet PNPH.
    """,
    version="2.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod, mettre ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SCHEMAS
# =============================================================================

class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str
    websocket_url: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    n_features: int
    version: str


# =============================================================================
# WEBSOCKET MANAGER
# =============================================================================

class ConnectionManager:
    """Gère les connexions WebSocket."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connecté pour job {job_id}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        logger.info(f"WebSocket déconnecté pour job {job_id}")
    
    async def send_result(self, job_id: str, data: dict):
        """Envoie le résultat à tous les clients connectés pour ce job."""
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logger.error(f"Erreur envoi WebSocket: {e}")


ws_manager = ConnectionManager()


# =============================================================================
# ENDPOINTS - INFO
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Point d'entrée - info de base."""
    return {
        "service": "FP-Predict API",
        "version": "2.1.0",
        "docs": "/docs",
        "endpoints": {
            "submit": "POST /jobs/submit",
            "status": "GET /jobs/{job_id}",
            "websocket": "WS /ws/{job_id}"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    """Health check."""
    predictor = get_predictor()
    
    return HealthResponse(
        status="healthy" if predictor.ready else "degraded",
        models_loaded=predictor.ready,
        n_features=85,
        version="2.1.0"
    )


@app.get("/models/info", tags=["Info"])
async def models_info():
    """Informations sur les modèles."""
    predictor = get_predictor()
    info = predictor.get_info()
    info["feature_groups"] = {
        "structural": 54,
        "rdkit": 24,
        "classification": 7,
        "total": 85
    }
    return info


# =============================================================================
# ENDPOINTS - JOBS
# =============================================================================

@app.post("/jobs/submit", response_model=JobSubmitResponse, tags=["Jobs"])
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Fichier PDB à analyser")
):
    """
    📤 Soumet un fichier PDB pour analyse.
    
    Le traitement est asynchrone. Utilisez le WebSocket ou le polling
    pour récupérer le résultat.
    """
    # Vérifier l'extension
    if not file.filename.lower().endswith('.pdb'):
        raise HTTPException(400, "Le fichier doit être un .pdb")
    
    # Lire le contenu
    content = await file.read()
    try:
        pdb_content = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(400, "Le fichier doit être encodé en UTF-8")
    
    # Créer le job
    job = await job_manager.create_job(filename=file.filename)
    
    # Enregistrer le callback pour WebSocket
    async def send_ws_result(completed_job):
        await ws_manager.send_result(completed_job.job_id, completed_job.to_dict())
    
    job_manager.register_callback(job.job_id, send_ws_result)
    
    # Lancer le traitement en background
    background_tasks.add_task(process_pdb_job, job.job_id, pdb_content)
    
    return JobSubmitResponse(
        job_id=job.job_id,
        status=job.status.value,
        message="Job créé, traitement en cours",
        websocket_url=f"/ws/{job.job_id}"
    )


@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    📊 Récupère le statut d'un job.
    
    Alternative au WebSocket pour les clients qui ne supportent pas WS.
    """
    job = await job_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(404, f"Job {job_id} non trouvé")
    
    return job.to_dict()


# =============================================================================
# WEBSOCKET
# =============================================================================

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    🔌 WebSocket pour recevoir les résultats en temps réel.
    
    1. Connectez-vous avec l'URL retournée par /jobs/submit
    2. Recevez le résultat automatiquement quand il est prêt
    3. Format du message: JSON avec le résultat complet
    """
    await ws_manager.connect(websocket, job_id)
    
    try:
        # Vérifier si le job existe et est déjà terminé
        job = await job_manager.get_job(job_id)
        
        if job is None:
            await websocket.send_json({"error": f"Job {job_id} non trouvé"})
            await websocket.close()
            return
        
        # Si déjà terminé, envoyer immédiatement
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            await websocket.send_json(job.to_dict())
        else:
            # Envoyer le statut actuel
            await websocket.send_json({
                "job_id": job_id,
                "status": job.status.value,
                "message": "En attente du résultat..."
            })
        
        # Garder la connexion ouverte
        while True:
            try:
                # Attendre un message du client (ping/pong ou fermeture)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=300  # 5 minutes timeout
                )
                
                # Si le client envoie "close", fermer proprement
                if data.lower() == "close":
                    break
                    
            except asyncio.TimeoutError:
                # Envoyer un ping pour garder la connexion vivante
                await websocket.send_json({"type": "ping"})
                
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket, job_id)


# =============================================================================
# PAGE DE TEST
# =============================================================================

@app.get("/test", response_class=HTMLResponse, tags=["Test"])
async def test_page():
    """Page de test interactive."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FP-Predict Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .dropzone { border: 3px dashed #ccc; padding: 50px; text-align: center; margin: 20px 0; border-radius: 10px; }
            .dropzone.dragover { border-color: #2196F3; background: #e3f2fd; }
            .result { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .success { background: #e8f5e9; border-left: 4px solid #4caf50; }
            .error { background: #ffebee; border-left: 4px solid #f44336; }
            .processing { background: #fff3e0; border-left: 4px solid #ff9800; }
            pre { background: #263238; color: #aed581; padding: 15px; border-radius: 5px; overflow-x: auto; }
            h1 { color: #1976d2; }
            .spectrum { display: flex; gap: 20px; margin: 20px 0; }
            .spectrum-box { flex: 1; padding: 20px; border-radius: 10px; text-align: center; color: white; }
            .ex { background: linear-gradient(135deg, #42a5f5, #1976d2); }
            .em { background: linear-gradient(135deg, #66bb6a, #43a047); }
            .big-number { font-size: 48px; font-weight: bold; }
            .info-badge { display: inline-block; background: #e3f2fd; color: #1976d2; padding: 5px 10px; border-radius: 5px; font-size: 12px; margin: 5px; }
        </style>
    </head>
    <body>
        <h1>🧬 FP-Predict API Test</h1>
        <p>
            <span class="info-badge">v2.1.0</span>
            <span class="info-badge">85 features</span>
            <span class="info-badge">ExtraTrees</span>
        </p>
        
        <div class="dropzone" id="dropzone">
            <p>📁 Glissez-déposez un fichier PDB ici</p>
            <p>ou <input type="file" id="fileInput" accept=".pdb"></p>
        </div>
        
        <div id="status"></div>
        <div id="result"></div>
        
        <script>
            const dropzone = document.getElementById('dropzone');
            const fileInput = document.getElementById('fileInput');
            const statusDiv = document.getElementById('status');
            const resultDiv = document.getElementById('result');
            
            // Drag & Drop
            dropzone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
            
            dropzone.addEventListener('dragleave', () => {
                dropzone.classList.remove('dragover');
            });
            
            dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file) uploadFile(file);
            });
            
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) uploadFile(file);
            });
            
            async function uploadFile(file) {
                statusDiv.innerHTML = '<div class="result processing">⏳ Upload en cours...</div>';
                resultDiv.innerHTML = '';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    // Soumettre le job
                    const response = await fetch('/jobs/submit', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Erreur upload');
                    }
                    
                    statusDiv.innerHTML = `<div class="result processing">⏳ Traitement en cours... (Job: ${data.job_id})</div>`;
                    
                    // Connecter au WebSocket
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/${data.job_id}`);
                    
                    ws.onmessage = (event) => {
                        const result = JSON.parse(event.data);
                        
                        if (result.status === 'completed') {
                            displayResult(result);
                            ws.close();
                        } else if (result.status === 'failed') {
                            statusDiv.innerHTML = `<div class="result error">❌ Erreur: ${result.error}</div>`;
                            ws.close();
                        } else if (result.type !== 'ping') {
                            statusDiv.innerHTML = `<div class="result processing">⏳ ${result.message || 'Traitement...'}</div>`;
                        }
                    };
                    
                    ws.onerror = () => {
                        // Fallback au polling
                        pollStatus(data.job_id);
                    };
                    
                } catch (error) {
                    statusDiv.innerHTML = `<div class="result error">❌ Erreur: ${error.message}</div>`;
                }
            }
            
            async function pollStatus(jobId) {
                const interval = setInterval(async () => {
                    const response = await fetch(`/jobs/${jobId}`);
                    const result = await response.json();
                    
                    if (result.status === 'completed') {
                        clearInterval(interval);
                        displayResult(result);
                    } else if (result.status === 'failed') {
                        clearInterval(interval);
                        statusDiv.innerHTML = `<div class="result error">❌ Erreur: ${result.error}</div>`;
                    }
                }, 1000);
            }
            
            function displayResult(result) {
                statusDiv.innerHTML = '<div class="result success">✅ Analyse terminée!</div>';
                
                const r = result.results;
                const m = result.metadata;
                
                resultDiv.innerHTML = `
                    <div class="spectrum">
                        <div class="spectrum-box ex">
                            <div>Excitation Max</div>
                            <div class="big-number">${r.ex_max_nm}</div>
                            <div>nm</div>
                        </div>
                        <div class="spectrum-box em">
                            <div>Émission Max</div>
                            <div class="big-number">${r.em_max_nm}</div>
                            <div>nm</div>
                        </div>
                    </div>
                    <div class="result">
                        <h3>📊 Métadonnées</h3>
                        <p><strong>Stokes Shift:</strong> ${r.stokes_shift_nm} nm</p>
                        <p><strong>Séquence:</strong> ${m.sequence_length} résidus</p>
                        <p><strong>Chromophore:</strong> ${m.chromophore_found ? (m.chromophore_triplet || 'Trouvé') + ' (position ' + m.chromophore_position + ')' : 'Non trouvé'}</p>
                        <p><strong>pLDDT moyen:</strong> ${m.plddt_mean || 'N/A'}</p>
                    </div>
                    <details>
                        <summary>📝 Réponse JSON complète</summary>
                        <pre>${JSON.stringify(result, null, 2)}</pre>
                    </details>
                `;
            }
        </script>
    </body>
    </html>
    """