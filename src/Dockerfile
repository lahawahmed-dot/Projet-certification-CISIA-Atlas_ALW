# ======================================================================================
# Construction en deux étapes (multi-stage).
#
# Pourquoi : l'entraînement a besoin du jeu de données, mais l'image d'inférence NE DOIT
# PAS l'embarquer. Une image poussée sur un registre contiendrait sinon les données
# personnelles de 2 500 usagers — incompatible avec la minimisation (art. 5.1.c RGPD)
# et avec la section « Sécurité et cycle de vie de la donnée » du README.
# L'étape 1 entraîne, l'étape 2 ne récupère que les artefacts (modèle + encodeurs).
# ======================================================================================

# --------------------------------------------------------------------------------------
# Étape 1 — entraînement (cette couche n'est jamais publiée)
# --------------------------------------------------------------------------------------
FROM python:3.12-slim AS entrainement

WORKDIR /app

# libgomp1 : bibliothèque OpenMP requise par XGBoost et LightGBM au moment de l'import
# (absente de l'image « slim » par défaut — sans elle : "libgomp.so.1: cannot open shared
# object file"). Problème trouvé en construisant réellement l'image, pas en relisant le code.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY entrainer_modele.py dataset_trajectoire_emploi.csv ./

ENV MLFLOW_TRACKING_URI=sqlite:////data/mlflow.db
RUN mkdir -p /data && python entrainer_modele.py

# --------------------------------------------------------------------------------------
# Étape 2 — image d'inférence publiée (sans le jeu de données)
# --------------------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# curl : nécessaire pour le healthcheck. libgomp1 : requis par XGBoost à l'import.
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY --from=entrainement /app/artefacts_modele/ ./artefacts_modele/

# Répertoire des données d'exploitation (journal des requêtes, retours conseillers). Il est
# créé ici pour exister dans l'image : Docker s'en sert pour initialiser le volume nommé au
# premier démarrage, propriétaire compris. Sans lui, le volume serait créé vide et détenu par
# root, et l'API — qui tourne sous un utilisateur non privilégié — ne pourrait pas y écrire.
RUN mkdir -p /app/donnees
# La base MLflow produite à la construction sert à amorcer le volume partagé du
# docker-compose (Docker recopie le contenu de l'image dans un volume nommé vide).
COPY --from=entrainement /data/ /data/

# L'API ne tourne pas en root : un service exposé sur le réseau n'a aucune raison
# d'avoir les droits d'administration du conteneur.
RUN useradd --create-home --uid 10001 service && chown -R service:service /app /data /app/donnees
USER service

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
