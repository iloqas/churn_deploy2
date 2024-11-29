# Utiliser une image Python légère
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers nécessaires dans l'image Docker
COPY train.py train.py
COPY app.py app.py
COPY DATA DATA
COPY templates/ templates/

# Exposer le port utilisé par Flask
EXPOSE 5012

# Exécuter le script pour entraîner le modèle et générer model.pkl
RUN python train.py

ENV PYTHONPATH=/app

# Commande pour démarrer l'application Flask
CMD ["python", "app.py"]
