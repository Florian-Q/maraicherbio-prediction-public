.PHONY: help install run train docker-build docker-run clean

# ── Variables ──────────────────────────────────────────────────────────────────
PYTHON      := python
PIP         := pip
STREAMLIT   := streamlit
APP         := app.py
PORT        := 8501
IMAGE       := maraicherbio
TAG         := latest

# ── Aide ───────────────────────────────────────────────────────────────────────
help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Installation ───────────────────────────────────────────────────────────────
install:  ## Installe les dépendances Python
	$(PIP) install -r requirements.txt

# ── Dashboard ──────────────────────────────────────────────────────────────────
run:  ## Lance le dashboard Streamlit
	$(STREAMLIT) run $(APP) --server.port=$(PORT) --server.address=0.0.0.0

# ── Entraînement ───────────────────────────────────────────────────────────────
train:  ## Exécute le pipeline d'entraînement et exporte les prédictions
	$(PYTHON) main/global_process.py

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:  ## Construit l'image Docker
	docker build -t $(IMAGE):$(TAG) .

docker-run:  ## Lance le conteneur Docker (dashboard sur le port 8501)
	docker run -p $(PORT):8501 -v $(PWD)/data:/app/data $(IMAGE):$(TAG)

# ── Nettoyage ──────────────────────────────────────────────────────────────────
clean:  ## Supprime les fichiers temporaires et caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*:Zone.Identifier' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true

clean-all: clean  ## Supprime aussi les exports de données
	rm -f data/Predictions.csv data/model_win_metric.csv
