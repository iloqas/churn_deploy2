# Utiliser une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le contenu de l'application dans le conteneur
COPY . .

# Exposer le port 5012 pour Flask
EXPOSE 5012

# Définir la commande de démarrage du conteneur
CMD ["python", "app.py"]