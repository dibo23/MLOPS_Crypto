# Nom du bucket GCS utilisé pour stocker le code, les données et les modèles
BUCKET = bucket-crypto-data

# Région Google Cloud utilisée pour l'exécution des jobs Vertex AI
REGION = us-central1

# Chemin où le package Python sera uploadé dans GCS
CODE_PATH = code/src.tar.gz

# Paire crypto utilisée pour le dataset et l'entraînement
PAIR = BTC_USDT

# RUN_ID transmis automatiquement par run.py (détecté après le fetch)
RUN_ID ?= none


# Construction de l’archive du code source pour Vertex AI
package:
	@echo "Construction du package src.tar.gz..."
	@rm -f src.tar.gz
	tar -czf src.tar.gz src setup.py


# Upload du package vers Google Cloud Storage
upload:
	@echo "Upload du package vers gs://$(BUCKET)/$(CODE_PATH)..."
	gsutil cp src.tar.gz gs://$(BUCKET)/$(CODE_PATH)


# Création et lancement d'un job d'entraînement Vertex AI
train:
	@if [ "$(RUN_ID)" = "none" ]; then \
	    echo "Erreur: aucun RUN_ID fourni."; \
	    echo "Utilisation correcte : make train RUN_ID=<id>"; \
	    exit 1; \
	fi

	@echo "Lancement du job Vertex AI avec RUN_ID=$(RUN_ID)..."
	gcloud ai custom-jobs create \
	    --region=$(REGION) \
	    --display-name="crypto_train_$(RUN_ID)" \
	    --python-package-uris=gs://$(BUCKET)/$(CODE_PATH) \
	    --worker-pool-spec=machine-type=e2-standard-4,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest,python-module=MLOPS_Crypto.train_lstm \
	    --args=--data_prefix=gs://$(BUCKET)/ohlcv-data \
	    --args=--pair=$(PAIR) \
	    --args=--run_id=$(RUN_ID)


# Exécution locale du script visualize_model.py pour générer les graphes et métriques
visual:
	@if [ "$(RUN_ID)" = "none" ]; then \
	    echo "Erreur: aucun RUN_ID fourni."; \
	    echo "Utilisation correcte : make visual RUN_ID=<id>"; \
	    exit 1; \
	fi

	@echo "Génération des visualisations locales pour RUN_ID=$(RUN_ID)..."
	python3 src/visualize_model.py \
	    --bucket $(BUCKET) \
	    --gcs_path ohlcv-data \
	    --lookback 30 \
	    --run_id $(RUN_ID)


# Pipeline complet, orchestré par run.py
all: package upload train visual
