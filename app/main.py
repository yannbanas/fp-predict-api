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
    allow_origins=["*"],
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
# PAGE D'ACCUEIL
# =============================================================================

LANDING_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FP-Predict | Fluorescent Protein Spectrum Prediction</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #10b981;
            --primary-dark: #059669;
            --secondary: #3b82f6;
            --accent: #8b5cf6;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --glow-green: rgba(16, 185, 129, 0.4);
            --glow-blue: rgba(59, 130, 246, 0.4);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated background */
        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        
        .bg-animation::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(ellipse at 30% 20%, var(--glow-green) 0%, transparent 50%),
                        radial-gradient(ellipse at 70% 80%, var(--glow-blue) 0%, transparent 50%);
            animation: bgPulse 15s ease-in-out infinite;
        }
        
        @keyframes bgPulse {
            0%, 100% { transform: translate(0, 0) rotate(0deg); opacity: 0.5; }
            50% { transform: translate(-5%, -5%) rotate(5deg); opacity: 0.8; }
        }
        
        /* Header */
        header {
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 700;
            font-size: 1.5rem;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }
        
        .nav-links {
            display: flex;
            gap: 1rem;
        }
        
        .nav-link {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            color: var(--text-muted);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .nav-link:hover {
            color: var(--text);
            background: var(--bg-card);
        }
        
        .nav-link.primary {
            background: var(--primary);
            color: white;
        }
        
        .nav-link.primary:hover {
            background: var(--primary-dark);
        }
        
        /* Hero */
        .hero {
            max-width: 1200px;
            margin: 0 auto;
            padding: 4rem 2rem;
            text-align: center;
        }
        
        .hero h1 {
            font-size: 3.5rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, var(--text) 0%, var(--primary) 50%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .hero p {
            font-size: 1.25rem;
            color: var(--text-muted);
            max-width: 600px;
            margin: 0 auto 2rem;
            line-height: 1.6;
        }
        
        .badges {
            display: flex;
            justify-content: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 3rem;
        }
        
        .badge {
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 50px;
            font-size: 0.875rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .badge.green { border-color: var(--primary); color: var(--primary); }
        .badge.blue { border-color: var(--secondary); color: var(--secondary); }
        .badge.purple { border-color: var(--accent); color: var(--accent); }
        
        /* Upload Zone */
        .upload-container {
            max-width: 700px;
            margin: 0 auto 4rem;
        }
        
        .dropzone {
            border: 2px dashed var(--border);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: var(--bg-card);
            position: relative;
            overflow: hidden;
        }
        
        .dropzone::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, var(--glow-green), var(--glow-blue));
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .dropzone:hover, .dropzone.dragover {
            border-color: var(--primary);
            transform: translateY(-2px);
        }
        
        .dropzone:hover::before, .dropzone.dragover::before {
            opacity: 0.1;
        }
        
        .dropzone-content {
            position: relative;
            z-index: 1;
        }
        
        .dropzone-icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
        }
        
        .dropzone h3 {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }
        
        .dropzone p {
            color: var(--text-muted);
            margin-bottom: 1rem;
        }
        
        .file-input {
            display: none;
        }
        
        .browse-btn {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .browse-btn:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }
        
        /* Results */
        .results-container {
            max-width: 700px;
            margin: 0 auto;
            display: none;
        }
        
        .results-container.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .status-card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--warning);
        }
        
        .status-card.success { border-left-color: var(--success); }
        .status-card.error { border-left-color: var(--error); }
        
        .status-card h4 {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .spectrum-display {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .spectrum-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .spectrum-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
        }
        
        .spectrum-card.excitation::before {
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        }
        
        .spectrum-card.emission::before {
            background: linear-gradient(90deg, #10b981, #84cc16);
        }
        
        .spectrum-label {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .spectrum-value {
            font-size: 3rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .spectrum-card.excitation .spectrum-value {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .spectrum-card.emission .spectrum-value {
            background: linear-gradient(135deg, #10b981, #84cc16);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .spectrum-unit {
            font-size: 1.25rem;
            color: var(--text-muted);
        }
        
        .metadata-card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
        }
        
        .metadata-card h4 {
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .metadata-item {
            padding: 1rem;
            background: var(--bg-dark);
            border-radius: 8px;
        }
        
        .metadata-item label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metadata-item value {
            display: block;
            font-size: 1.125rem;
            font-weight: 600;
            margin-top: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Features Section */
        .features {
            max-width: 1200px;
            margin: 0 auto;
            padding: 4rem 2rem;
        }
        
        .features h2 {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 3rem;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }
        
        .feature-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            transition: all 0.3s;
        }
        
        .feature-card:hover {
            border-color: var(--primary);
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        .feature-icon {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .feature-card h3 {
            margin-bottom: 0.5rem;
        }
        
        .feature-card p {
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        /* API Section */
        .api-section {
            max-width: 900px;
            margin: 0 auto;
            padding: 4rem 2rem;
        }
        
        .api-section h2 {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 2rem;
        }
        
        .code-block {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }
        
        .code-header {
            background: var(--bg-card);
            padding: 0.75rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }
        
        .code-header span {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            color: var(--text-muted);
        }
        
        .copy-btn {
            background: var(--bg-card-hover);
            border: none;
            color: var(--text-muted);
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.2s;
        }
        
        .copy-btn:hover {
            background: var(--primary);
            color: white;
        }
        
        pre {
            padding: 1.5rem;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            line-height: 1.6;
        }
        
        pre code {
            color: #e6edf3;
        }
        
        .code-comment { color: #8b949e; }
        .code-string { color: #a5d6ff; }
        .code-keyword { color: #ff7b72; }
        .code-function { color: #d2a8ff; }
        
        /* Footer */
        footer {
            border-top: 1px solid var(--border);
            padding: 2rem;
            text-align: center;
            color: var(--text-muted);
        }
        
        footer a {
            color: var(--primary);
            text-decoration: none;
        }
        
        /* Spinner */
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 { font-size: 2.5rem; }
            .spectrum-display { grid-template-columns: 1fr; }
            .metadata-grid { grid-template-columns: 1fr; }
            header { flex-direction: column; gap: 1rem; }
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    
    <header>
        <div class="logo">
            <div class="logo-icon">🧬</div>
            <span>FP-Predict</span>
        </div>
        <nav class="nav-links">
            <a href="/docs" class="nav-link">API Docs</a>
            <a href="/health" class="nav-link">Status</a>
            <a href="https://github.com/Pmusic-FP/fp-predict-api" class="nav-link" target="_blank">GitHub</a>
        </nav>
    </header>
    
    <main>
        <section class="hero">
            <h1>Fluorescent Protein<br>Spectrum Prediction</h1>
            <p>
                Prédisez les longueurs d'onde d'excitation et d'émission de vos protéines fluorescentes 
                à partir de leur structure 3D. Powered by ML & structural features.
            </p>
            <div class="badges">
                <span class="badge green">✓ 85 Features</span>
                <span class="badge blue">⚡ ExtraTrees ML</span>
                <span class="badge purple">🎯 MAE ~17nm</span>
            </div>
        </section>
        
        <section class="upload-container">
            <div class="dropzone" id="dropzone">
                <div class="dropzone-content">
                    <div class="dropzone-icon">📄</div>
                    <h3>Upload your PDB file</h3>
                    <p>Drag & drop or click to browse</p>
                    <input type="file" id="fileInput" class="file-input" accept=".pdb">
                    <button class="browse-btn" onclick="document.getElementById('fileInput').click()">
                        Browse Files
                    </button>
                </div>
            </div>
        </section>
        
        <section class="results-container" id="results">
            <div class="status-card" id="statusCard">
                <h4><span class="spinner"></span> Processing...</h4>
                <p id="statusText">Analyzing protein structure...</p>
            </div>
            
            <div class="spectrum-display" id="spectrumDisplay" style="display: none;">
                <div class="spectrum-card excitation">
                    <div class="spectrum-label">Excitation Max</div>
                    <div class="spectrum-value" id="exValue">---</div>
                    <div class="spectrum-unit">nm</div>
                </div>
                <div class="spectrum-card emission">
                    <div class="spectrum-label">Emission Max</div>
                    <div class="spectrum-value" id="emValue">---</div>
                    <div class="spectrum-unit">nm</div>
                </div>
            </div>
            
            <div class="metadata-card" id="metadataCard" style="display: none;">
                <h4>📊 Analysis Details</h4>
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <label>Stokes Shift</label>
                        <value id="stokesValue">--</value>
                    </div>
                    <div class="metadata-item">
                        <label>Sequence Length</label>
                        <value id="seqLength">--</value>
                    </div>
                    <div class="metadata-item">
                        <label>Chromophore</label>
                        <value id="chromophore">--</value>
                    </div>
                    <div class="metadata-item">
                        <label>pLDDT Score</label>
                        <value id="plddt">--</value>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="features">
            <h2>How It Works</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">📁</div>
                    <h3>Upload PDB</h3>
                    <p>Submit your AlphaFold2 or experimental PDB structure file through our API or web interface.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔬</div>
                    <h3>Feature Extraction</h3>
                    <p>85 structural features are extracted including chromophore geometry, RDKit descriptors, and pLDDT scores.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <h3>ML Prediction</h3>
                    <p>ExtraTrees regressor predicts excitation and emission wavelengths with ~17nm MAE accuracy.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3>Get Results</h3>
                    <p>Receive predictions via WebSocket in real-time or poll the REST API for results.</p>
                </div>
            </div>
        </section>
        
        <section class="api-section">
            <h2>Quick Start</h2>
            
            <div class="code-block">
                <div class="code-header">
                    <span>Python</span>
                    <button class="copy-btn" onclick="copyCode('python-code')">Copy</button>
                </div>
                <pre><code id="python-code"><span class="code-keyword">import</span> requests

<span class="code-comment"># Submit PDB file</span>
<span class="code-keyword">with</span> <span class="code-function">open</span>(<span class="code-string">"protein.pdb"</span>, <span class="code-string">"rb"</span>) <span class="code-keyword">as</span> f:
    response = requests.post(
        <span class="code-string">"https://your-api.railway.app/jobs/submit"</span>,
        files={<span class="code-string">"file"</span>: f}
    )

job_id = response.json()[<span class="code-string">"job_id"</span>]

<span class="code-comment"># Poll for results</span>
result = requests.get(<span class="code-string">f"https://your-api.railway.app/jobs/{job_id}"</span>)
<span class="code-function">print</span>(result.json())</code></pre>
            </div>
            
            <div class="code-block">
                <div class="code-header">
                    <span>cURL</span>
                    <button class="copy-btn" onclick="copyCode('curl-code')">Copy</button>
                </div>
                <pre><code id="curl-code"><span class="code-comment"># Submit job</span>
curl -X POST "https://your-api.railway.app/jobs/submit" \
  -F "file=@protein.pdb"

<span class="code-comment"># Get results</span>
curl "https://your-api.railway.app/jobs/{job_id}"</code></pre>
            </div>
        </section>
    </main>
    
    <footer>
        <p>
            Built with ❤️ for the <strong>PNPH Project</strong> | 
            <a href="/docs">API Documentation</a> | 
            v2.1.0
        </p>
    </footer>
    
    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const results = document.getElementById('results');
        const statusCard = document.getElementById('statusCard');
        const statusText = document.getElementById('statusText');
        const spectrumDisplay = document.getElementById('spectrumDisplay');
        const metadataCard = document.getElementById('metadataCard');
        
        // Drag & Drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            dropzone.addEventListener(event, e => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        
        ['dragenter', 'dragover'].forEach(event => {
            dropzone.addEventListener(event, () => dropzone.classList.add('dragover'));
        });
        
        ['dragleave', 'drop'].forEach(event => {
            dropzone.addEventListener(event, () => dropzone.classList.remove('dragover'));
        });
        
        dropzone.addEventListener('drop', e => {
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
        });
        
        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', e => {
            if (e.target.files[0]) handleFile(e.target.files[0]);
        });
        
        async function handleFile(file) {
            if (!file.name.toLowerCase().endsWith('.pdb')) {
                alert('Please upload a .pdb file');
                return;
            }
            
            // Show results section
            results.classList.add('show');
            spectrumDisplay.style.display = 'none';
            metadataCard.style.display = 'none';
            statusCard.className = 'status-card';
            statusCard.innerHTML = '<h4><span class="spinner"></span> Processing...</h4><p id="statusText">Uploading file...</p>';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                // Submit job
                const response = await fetch('/jobs/submit', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error((await response.json()).detail || 'Upload failed');
                }
                
                const data = await response.json();
                document.getElementById('statusText').textContent = `Job ${data.job_id} created. Analyzing...`;
                
                // Connect WebSocket
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/${data.job_id}`);
                
                ws.onmessage = event => {
                    const result = JSON.parse(event.data);
                    
                    if (result.status === 'completed') {
                        showResults(result);
                        ws.close();
                    } else if (result.status === 'failed') {
                        showError(result.error);
                        ws.close();
                    } else if (result.type !== 'ping') {
                        document.getElementById('statusText').textContent = result.message || 'Processing...';
                    }
                };
                
                ws.onerror = () => pollResults(data.job_id);
                
            } catch (error) {
                showError(error.message);
            }
        }
        
        async function pollResults(jobId) {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/jobs/${jobId}`);
                    const result = await response.json();
                    
                    if (result.status === 'completed') {
                        clearInterval(interval);
                        showResults(result);
                    } else if (result.status === 'failed') {
                        clearInterval(interval);
                        showError(result.error);
                    }
                } catch (e) {
                    clearInterval(interval);
                    showError('Connection lost');
                }
            }, 1000);
        }
        
        function showResults(result) {
            statusCard.className = 'status-card success';
            statusCard.innerHTML = '<h4>✅ Analysis Complete</h4><p>Prediction successful!</p>';
            
            spectrumDisplay.style.display = 'grid';
            metadataCard.style.display = 'block';
            
            const r = result.results;
            const m = result.metadata;
            
            document.getElementById('exValue').textContent = r.ex_max_nm;
            document.getElementById('emValue').textContent = r.em_max_nm;
            document.getElementById('stokesValue').textContent = `${r.stokes_shift_nm} nm`;
            document.getElementById('seqLength').textContent = `${m.sequence_length} aa`;
            document.getElementById('chromophore').textContent = m.chromophore_found 
                ? `${m.chromophore_triplet} @ ${m.chromophore_position}` 
                : 'Not found';
            document.getElementById('plddt').textContent = m.plddt_mean || 'N/A';
        }
        
        function showError(message) {
            statusCard.className = 'status-card error';
            statusCard.innerHTML = `<h4>❌ Error</h4><p>${message}</p>`;
        }
        
        function copyCode(elementId) {
            const code = document.getElementById(elementId).textContent;
            navigator.clipboard.writeText(code);
        }
    </script>
</body>
</html>
"""


# =============================================================================
# ENDPOINTS - INFO
# =============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Info"])
async def landing_page():
    """Page d'accueil avec interface de test."""
    return LANDING_PAGE


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


@app.get("/api", tags=["Info"])
async def api_info():
    """Informations API (JSON)."""
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