 API de prediction du risque de retour a l'emploi

Le systeme predit, a partir des informations recueillies pendant un entretien de cadrage, la classe de retour a l'emploi d'un usager :

- **Classe 0** : retour rapide, inferieur a 6 mois ;
- **Classe 1** : retour moyen, entre 6 et 12 mois ;
- **Classe 2** : risque de chomage de longue duree, superieur a 12 mois.

La prediction constitue une aide a la decision pour le conseiller. Elle ne remplace pas son jugement et ne declenche pas automatiquement une decision d'accompagnement.

## Configuration finale retenue

- **Modele** : XGBoost ;
- **Scenario** : Scenario 2, sans variable sensible fournie au modele ;
- **Seuil de decision de la classe 2** : 0,20 ;
- **Version du modele** : `xgboost_scenario2_v2` ;
- **Regle de selection** : minimiser le taux d'erreur critique, c'est-a-dire les vrais usagers de classe 2 predits en classe 0, sous contrainte d'un F1-macro au moins egal a 0,68 en validation croisee.

Le jeu de test est reserve a l'evaluation finale. Le choix du modele, du scenario et du seuil est effectue sur le jeu d'entrainement avec validation croisee.

## Resultats finaux enregistres dans le projet

| Metrique | Valeur | Intervalle de confiance a 95 % |
|---|---:|---:|
| Accuracy | 0,708 | Non calcule |
| F1-macro | 0,691 | [0,646 ; 0,731] |
| AUC macro un-contre-tous | 0,856 | Non calcule |
| Taux d'erreur critique, classe 2 vers classe 0 | 7,8 %, soit 7 usagers sur 90 | [2,5 % ; 13,8 %] |

La classe majoritaire seule produirait une accuracy de 44,5 %.

## Structure du projet

```text
.
| Élément | Rôle |
|---|---|
| `certification_cisia_atlas.ipynb` | Notebook complet (sections 1 à 15 et annexe) |
| `executer_et_verifier.py` | Exécute le notebook depuis un noyau neuf et vérifie tout le projet |
| `src/entrainer_modele.py` | Seule implémentation de l'entraînement (XGBoost, Scénario 2, seuil 0,20) |
| `src/retours_conseillers.py` | Relie les retours des conseillers aux prédictions journalisées |
| `src/verifier_seuil_performance.py` | Garde-fou de qualité et d'équité avant toute promotion |
| `src/api.py` | API FastAPI : `/health`, `/metrics`, `/predict`, `/retrain`, `/retrain/run` |
| `src/interface.html` | Interface du conseiller |
| `src/tests/test_api.py` | Tests automatisés |
| `src/Dockerfile`, `src/docker-compose.yml` | Image d'inférence sans données, MLflow |
| `.github/workflows/ci-cd.yml` | Tests, image Docker, publication, déploiement |
```

Le code applicatif est regroupe dans `src/`. Le Notebook, le README et le workflow GitHub Actions restent a la racine du depot.

Les elements suivants sont generes pendant l'execution et ne sont pas des fichiers sources :

```text
src/artefacts_modele/
src/artefacts_candidat/
src/mlflow.db
src/journal_requetes.jsonl
src/feedback_reentrainement.jsonl
src/__pycache__/
```

## Prerequis

### Logiciels

- Anaconda ou Miniconda ;
- Git ;
- JupyterLab ou Jupyter Notebook, installe dans l'environnement Conda du projet ;
- Graphviz pour rejouer le diagramme d'architecture du Notebook ;
- Docker Desktop uniquement pour l'execution avec Docker.

### Version Python

Le projet utilise **Python 3.12**.

## Installation avec Anaconda et Jupyter

Les commandes suivantes sont prevues pour **Anaconda Prompt sous Windows**.

### 1. Ouvrir Anaconda Prompt

Ouvrir **Anaconda Prompt**, puis se placer a la racine du projet :

```bat
cd C:\chemin\vers\certification-cisia-atlas
```

Remplacer le chemin d'exemple par le chemin reel du projet.

### 2. Creer l'environnement Conda

Cette commande ne doit etre executee que lors de la premiere installation :

```bat
conda create --name cisia-ia python=3.12 pip -y
```

### 3. Activer l'environnement

```bat
conda activate cisia-ia
```

Le debut de l'invite de commandes doit afficher :

```text
(cisia-ia)
```

### 4. Installer Jupyter et le noyau du projet

Le fichier `requirements-dev.txt` contient les dependances de test et d'analyse du projet, mais Jupyter doit etre installe explicitement dans l'environnement Conda :

```bat
conda install -n cisia-ia -c conda-forge jupyterlab notebook ipykernel graphviz -y
```

Enregistrer ensuite l'environnement comme noyau Jupyter :

```bat
python -m ipykernel install --user --name cisia-ia --display-name "Python 3.12 - CISIA IA"
```

Le noyau visible dans Jupyter portera le nom :

```text
Python 3.12 - CISIA IA
```

### 5. Installer les dependances du projet

Se placer dans le dossier applicatif :

```bat
cd src
```

Mettre `pip` a jour, puis installer les dependances :

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

### 6. Verifier l'environnement actif

```bat
python --version
where python
```

Le premier chemin retourne par `where python` doit correspondre a l'environnement `cisia-ia`.

Verifier les principales bibliotheques :

```bat
python -c "import fastapi, pydantic, sklearn, xgboost, mlflow, pandas, numpy, scipy; print('OK - dependances principales importees')"
```

## Lancement du Notebook avec Jupyter

### 1. Activer l'environnement

Dans Anaconda Prompt :

```bat
conda activate cisia-ia
```

### 2. Revenir a la racine du projet

Si l'invite est actuellement dans `src/` :

```bat
cd ..
```

La racine doit contenir le fichier :

```text
certification_cisia_atlas.ipynb
```

### 3. Lancer JupyterLab

```bat
jupyter lab
```

Pour utiliser l'interface classique a la place de JupyterLab :

```bat
jupyter notebook
```

### 4. Selectionner le bon noyau

Dans Jupyter, ouvrir `certification_cisia_atlas.ipynb`, puis selectionner :

```text
Python 3.12 - CISIA IA
```

Le Notebook doit utiliser ce noyau pour retrouver les memes dependances que les scripts Python.

### 5. Verifier le noyau depuis le Notebook

Executer une cellule contenant :

```python
import sys
print(sys.executable)
print(sys.version)
```

Le chemin affiche doit correspondre a l'environnement Conda `cisia-ia`.

### 6. Acceder aux fichiers de `src/` depuis le Notebook

Le Notebook etant conserve a la racine, les imports des fichiers applicatifs doivent utiliser le dossier `src/` :

```python
from pathlib import Path
import sys

RACINE_PROJET = Path.cwd()
DOSSIER_SRC = RACINE_PROJET / "src"

if str(DOSSIER_SRC) not in sys.path:
    sys.path.insert(0, str(DOSSIER_SRC))
```

Les chemins de fichiers doivent ensuite etre construits explicitement, par exemple :

```python
CHEMIN_DATASET = DOSSIER_SRC / "dataset_trajectoire_emploi.csv"
CHEMIN_ARTEFACTS = DOSSIER_SRC / "artefacts_modele"
```

Cette methode evite de dependre du dossier de travail laisse par une ancienne execution.

### 7. Reexecution complete avant livraison

Avant la livraison finale :

1. ouvrir le Notebook dans Jupyter ;
2. selectionner le noyau `Python 3.12 - CISIA IA` ;
3. redemarrer le noyau ;
4. executer toutes les cellules dans l'ordre ;
5. verifier qu'aucune cellule ne depend d'une variable creee manuellement lors d'une execution precedente ;
6. enregistrer le Notebook avec les sorties finales coherentes.

N'inventez pas de resultat si une cellule echoue. Corrigez l'erreur, redemarrez le noyau et recommencez l'execution complete.

## Demarrage rapide sans Docker avec Anaconda

Toutes les commandes applicatives doivent etre executees depuis `src/`.

### 1. Activer l'environnement et entrer dans `src/`

```bat
conda activate cisia-ia
cd C:\chemin\vers\certification-cisia-atlas\src
```

### 2. Compiler les fichiers Python

```bat
python -m py_compile api.py entrainer_modele.py verifier_seuil_performance.py tests\test_api.py
```

L'absence de message signifie que la compilation a reussi.

### 3. Verifier les imports applicatifs

```bat
python -c "import api; import entrainer_modele; import verifier_seuil_performance; print('OK - imports applicatifs verifies')"
```

### 4. Verifier le style et les erreurs probables

```bat
ruff check .
```

### 5. Entrainer le modele

```bat
python entrainer_modele.py
```

Le script doit generer `artefacts_modele/` avec les fichiers suivants :

```text
modele_xgboost.joblib
onehot_encoder.joblib
tfidf_vectorizer.joblib
parametres_imputation.joblib
metriques.json
```

Il enregistre egalement une run MLflow.

### 6. Verifier les artefacts

```bat
dir artefacts_modele
```

Le dossier ne doit pas etre vide.

### 7. Executer le garde-fou qualite et equite

```bat
python verifier_seuil_performance.py --f1-macro-min 0.68 --erreur-critique-max 0.15 --ecart-rappel-max 0.15
```

Le modele n'est considere comme acceptable que si le script termine avec un code de sortie nul.

### 8. Executer les tests

```bat
pytest tests -v
```

Une ligne `PASSED` indique un test reussi. Une ligne `FAILED` indique une erreur a analyser avant le lancement du service.

### 8.1 Executer les verfication 
python executer_et_verifier.py        # exécution complète et rapport_execution.md
### 9. Lancer l'API

```bat
uvicorn api:app --reload
```

L'API est disponible localement sur le port `8000`.

Routes de controle :

- `/health` : etat du service et chargement du modele ;
- `/docs` : documentation interactive ;
- `/metrics` : compteurs au format Prometheus.

Arreter l'API avec `Ctrl + C`.

### 10. Ouvrir l'interface graphique

Pendant que l'API fonctionne, ouvrir le fichier suivant dans un navigateur :

```text
src/interface.html
```

L'interface permet de saisir un usager, d'obtenir une prediction, d'afficher les avertissements et d'enregistrer un retour conseiller.

## Lancement de MLflow avec Anaconda

Ouvrir une deuxieme fenetre Anaconda Prompt :

```bat
conda activate cisia-ia
cd C:\chemin\vers\certification-cisia-atlas\src
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

L'interface MLflow est disponible localement sur le port `5000`.

Le fichier `mlflow.db` est cree ou reutilise dans `src/`.

## Redemarrage lors des utilisations suivantes

Il ne faut pas recreer l'environnement Conda a chaque utilisation.

### Pour travailler dans Jupyter

```bat
conda activate cisia-ia
cd C:\chemin\vers\certification-cisia-atlas
jupyter lab
```

### Pour lancer l'API

```bat
conda activate cisia-ia
cd C:\chemin\vers\certification-cisia-atlas\src
uvicorn api:app --reload
```

### Pour relancer les tests

```bat
conda activate cisia-ia
cd C:\chemin\vers\certification-cisia-atlas\src
pytest tests -v
```

### Pour quitter l'environnement

```bat
conda deactivate
```

## Exemple de requete API

Lorsque l'API fonctionne :

```bat
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":45,\"niveau_diplome\":\"Bac\",\"date_debut_poste\":\"2014-03-01\",\"code_rome_vise\":\"M1607\",\"code_insee_commune\":\"07240\",\"est_allocataire\":1,\"synthese_entretien\":\"Freins peripheriques majeurs.\"}"
```

L'API attend une date de debut de poste et calcule elle-meme l'anciennete par rapport au jour de la requete.

## Routes de l'API

| Methode | Route | Role |
|---|---|---|
| GET | `/health` | Controle le chargement du modele et expose la version et le seuil actifs |
| GET | `/metrics` | Expose les compteurs du service au format Prometheus |
| POST | `/predict` | Retourne la classe predite, les probabilites et les avertissements |
| POST | `/retrain` | Enregistre un retour conseiller sans reentrainement immediat |
| POST | `/retrain/run` | Lance un entrainement challenger si la cle d'administration et les ressources necessaires sont disponibles |

## Validation des donnees entrantes

L'API distingue deux situations :

- les donnees structurellement invalides sont refusees ;
- les donnees acceptables mais inhabituelles produisent un avertissement.

Exemples de controles :

- date de debut de poste future ;
- anciennete calculee superieure a la limite acceptee ;
- code ROME mal forme ;
- code INSEE mal forme ;
- age en dehors des bornes autorisees ;
- incoherence entre l'age et l'anciennete ;
- departement ou code ROME absent des donnees d'entrainement.

La variable `nationalite_hors_ue` n'est pas acceptee par le schema d'entree, conformement au Scenario 2 retenu.

## Interface graphique

Le fichier `src/interface.html` fournit :

- un formulaire de saisie ;
- une prediction en trois classes ;
- les probabilites associees ;
- les avertissements de qualite des donnees ;
- l'enregistrement d'un retour conseiller ;
- L'historique affiché dans la page est conservé dans le `localStorage` du
   navigateur et survit à un rechargement. Il reste local au poste utilisé, n'est
   pas partagé entre conseillers et est limité aux 100 dernières évaluations.

   Pour chaque évaluation, ce stockage local conserve uniquement l'identifiant de
  session, la classe prédite, la durée de calcul, l'heure affichée et l'état du
   feedback. Il peut être supprimé depuis l'interface avec **« Vider
   l'historique »**.

   La trace complète et commune aux différents postes reste
   `journal_requetes.jsonl`, écrit côté API et soumis à la politique de
    conservation du service.

## Securite et cycle de vie des donnees

### Mecanismes implementes

- authentification facultative de service avec `CLE_API` ;
- cle d'administration distincte `CLE_ADMIN` pour `/retrain/run` ;
- liste d'origines CORS configurable avec `ORIGINES_AUTORISEES` ;
- purge du journal selon `RETENTION_JOURNAL_JOURS` ;
- image Docker d'inference sans dataset ;
- execution du conteneur avec un utilisateur non privilegie ;
- journalisation des requetes, sorties, dates, identifiants de session, version du modele et duree de traitement.

### Configuration de demonstration

Sans `CLE_API`, l'API reste ouverte pour la demonstration locale. Sans restriction d'origines, le CORS reste permissif. Ces choix doivent etre durcis avant une mise en production.

### Prealables identifies pour la production

- stockage chiffre des journaux ;
- gestion des secrets hors du depot ;
- restriction CORS ;
- authentification renforcee entre services ;
- limitation de debit ;
- analyse d'impact relative a la protection des donnees ;
- monitoring centralise ;
- mecanisme d'explication individuelle industrialise ;
- chargement controle d'une version de modele depuis un registre.

## Reentrainement et promotion du modele

Le reentrainement suit une logique challenger/champion :

1. `/retrain` enregistre les retours conseillers ;
2. un humain controle les retours ;
3. un challenger est entraine dans un dossier distinct ;
4. ses performances et indicateurs d'equite sont verifies ;
5. la promotion reste une decision humaine ;
6. le service est redemarre avec les artefacts approuves.

Le garde-fou `verifier_seuil_performance.py` controle :

- le F1-macro minimal ;
- le taux maximal d'erreur critique ;
- l'ecart maximal de rappel entre les groupes audites.

## Monitoring

La route `/metrics` expose notamment :

- le nombre de predictions par classe ;
- le nombre de requetes reussies ;
- le nombre d'erreurs de validation ;
- le nombre d'erreurs internes ;
- le nombre de feedbacks ;
- la somme des latences d'inference.

Les compteurs en memoire correspondent au processus Python courant. Une architecture multi-processus necessiterait un stockage de metriques partage ou un mode multiprocess adapte.

## Execution avec Docker

Le `Dockerfile` et `docker-compose.yml` sont places dans `src/`.

Depuis Anaconda Prompt ou un terminal :

```bat
cd C:\chemin\vers\certification-cisia-atlas\src
docker compose up --build
```

Services exposes :

- API sur le port `8000` ;
- interface MLflow sur le port `5000`.

Pour arreter les services :

```bat
docker compose down
```

## Integration continue

Le workflow doit etre place exactement ici :

```text
.github/workflows/ci-cd.yml
```

Le pipeline realise notamment :

1. la verification des fichiers indispensables ;
2. l'installation des dependances ;
3. la compilation Python ;
4. le controle des imports ;
5. le linting ;
6. l'entrainement ;
7. la verification des artefacts ;
8. le garde-fou qualite et equite ;
9. les tests ;
10. la construction de l'image Docker sur la branche principale ;
11. le controle de la route `/health` ;
12. le controle de l'absence de fichiers de donnees dans le contenu applicatif publie.

## Verification finale avant livraison

### Verification depuis Anaconda Prompt

```bat
conda activate cisia-ia
cd C:\chemin\vers\certification-cisia-atlas\src
python -m py_compile api.py entrainer_modele.py verifier_seuil_performance.py tests\test_api.py
python -c "import api; import entrainer_modele; import verifier_seuil_performance"
ruff check .
python entrainer_modele.py
python verifier_seuil_performance.py --f1-macro-min 0.68 --erreur-critique-max 0.15 --ecart-rappel-max 0.15
pytest tests -v
```

### Verification du Notebook

1. lancer JupyterLab depuis la racine ;
2. selectionner `Python 3.12 - CISIA IA` ;
3. redemarrer le noyau ;
4. executer toutes les cellules ;
5. verifier les sorties ;
6. enregistrer le Notebook final.

### Verification Docker

```bat
cd C:\chemin\vers\certification-cisia-atlas\src
docker compose up --build
```

Verifier ensuite `/health`, puis arreter les services avec :

```bat
docker compose down
```

## Responsabilité juridique

Le modèle est une aide à la décision : le conseiller décide. Le journal horodaté, qui associe chaque
prédiction à la version du modèle qui l'a produite, est la trace qui permet de répondre à un usager
ou à un contrôle ; il ne doit jamais pouvoir être perdu à moitié, d'où l'écriture atomique de la
purge. L'analyse des risques (erreur d'orientation, discrimination, explicabilité) figure au § 10
du notebook.

## Documentation complete

Le Notebook `certification_cisia_atlas.ipynb` contient la demarche complete :

1. Cadrage du besoin et hypothèses
2. Environnement et reproductibilité
3. Exploration des données
4. Contrôle qualité et préparation
5. Construction des quatre scénarios
6. Modèles candidats en validation croisée
7. Comparaison des scénarios en validation croisée
8. Réduction des erreurs critiques et choix du modèle
9. Évaluation finale sur le jeu de test
10. Éthique et cadre réglementaire
11. Industrialisation : artefacts, API, suivi, CI/CD
12. Limites
13. Conclusion
14. Journal de bord et tableau de suivi
15. Contrôle final de cohérence
