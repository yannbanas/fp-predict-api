# 🧬 FP-Predict API

> **Prédiction spectrale des protéines fluorescentes à partir de fichiers PDB**

[![Railway](https://railway.app/button.svg)](https://railway.app)

## 📊 Performances

| Métrique | Excitation | Émission |
|----------|------------|----------|
| **MAE** | ~21 nm | ~17 nm |
| **R²** | 0.74 | 0.74 |

## 🚀 Fonctionnalités

- **Upload PDB** → Prédiction ex_max / em_max
- **API REST** + **WebSocket** temps réel
- **85 features** : structural + RDKit + classification
- **Modèle** : ExtraTrees Regressor

## 📦 Stack

- FastAPI + Uvicorn
- scikit-learn (ExtraTrees)
- RDKit (descripteurs chimiques)
- Docker ready

## 🛠️ Installation locale

```bash
# Clone
git clone https://github.com/ton-user/fp-predict-api.git
cd fp-predict-api

# Virtual env
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload --port 8000
```

## 🐳 Docker

```bash
# Build
docker build -t fp-predict-api:2.1.0 .

# Run
docker run -p 8000:8000 fp-predict-api:2.1.0
```

## 🚂 Déploiement Railway

1. Fork/Push ce repo sur GitHub
2. Connecte Railway à ton GitHub
3. New Project → Deploy from GitHub repo
4. Sélectionne le repo
5. Deploy ! 🎉

## 📡 API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Info API |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Documentation Swagger |
| `POST` | `/jobs/submit` | Upload PDB |
| `GET` | `/jobs/{id}` | Statut job |
| `WS` | `/ws/{id}` | WebSocket résultats |

## 📝 Exemple d'utilisation

### cURL

```bash
# Upload PDB
curl -X POST "https://ton-api.railway.app/jobs/submit" \
  -F "file=@ma_proteine.pdb"

# Réponse
{
  "job_id": "abc12345",
  "status": "pending",
  "websocket_url": "/ws/abc12345"
}
```

### Python

```python
import requests

# Upload
with open("protein.pdb", "rb") as f:
    r = requests.post(
        "https://ton-api.railway.app/jobs/submit",
        files={"file": f}
    )
job_id = r.json()["job_id"]

# Poll result
result = requests.get(f"https://ton-api.railway.app/jobs/{job_id}").json()
print(f"Ex: {result['results']['ex_max_nm']} nm")
print(f"Em: {result['results']['em_max_nm']} nm")
```

## 🧪 Test

Accède à `/test` pour une interface de test drag & drop.

## 📁 Structure

```
fp-predict-api/
├── app/
│   ├── main.py              # FastAPI app
│   ├── core/
│   │   ├── config.py        # 85 features config
│   │   ├── feature_extractor.py
│   │   ├── pdb_parser.py
│   │   └── predictor.py
│   ├── jobs/
│   │   └── processor.py     # Async job processing
│   └── models/
│       ├── specialized_em_max.joblib
│       └── specialized_ex_max.joblib
├── Dockerfile
├── requirements.txt
├── railway.toml
└── README.md
```

## 🔬 Projet PNPH

Ce projet fait partie du **PNPH** (Photonic-Neural Hybrid Platform) - recherche sur les systèmes de communication neurale biophotonique.

## 📄 Licence

MIT

---

Made with 🐼 by Panda