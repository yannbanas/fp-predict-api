# ============================================================
# FP-Predict API - Dockerfile
# ============================================================
# Version 2.1.0
# ============================================================

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Panda"
LABEL description="FP-Predict API - Prédiction spectrale FP (85 features)"
LABEL version="2.1.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Dépendances système (nécessaires pour RDKit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Création utilisateur non-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY --chown=appuser:appuser app/ ./app/

# Changer vers utilisateur non-root
USER appuser

# Port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]