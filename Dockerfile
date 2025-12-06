# IMAGE DE BASE : TensorFlow GPU + Python
FROM tensorflow/tensorflow:2.15.0-gpu

# Dossier de travail
WORKDIR /app

# Copier tout le code dans l'image
COPY . /app

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Afficher le contenu
RUN ls -R /app

# Entrypoint exécuté par Vertex AI
ENTRYPOINT ["python", "src/train_lstm.py"]
