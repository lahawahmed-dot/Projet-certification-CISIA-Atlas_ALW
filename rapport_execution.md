# Rapport d'exécution

Généré le 2026-10-01T14:44:04+00:00 (UTC) par `executer_et_verifier.py`, sur Windows 11, Python 3.13.9.

| Étape | Verdict | Durée (s) |
|---|---|---|
| 1. Environnement | réussi | 0.5 |
| 2. Exécution du notebook depuis un noyau neuf | réussi | 509.1 |
| 3. Contrôles de cohérence du notebook (§ 15) | ÉCHEC | 0.0 |
| 4. Tests automatisés de l'API | réussi | 26.6 |
| 5. Garde-fou de qualité et d'équité | réussi | 0.1 |
| 6. Artefacts du modèle | réussi | 2.2 |
| 7. Registre MLflow | réussi | 4.2 |

## 1. Environnement

- Python 3.13.9 (Windows 11)
- ÉCART fastapi : installé 0.138.2, épinglé 0.141.1
- OK  graphviz : installé 0.21, épinglé 0.21
- OK  httpx : installé 0.28.1, épinglé 0.28.1
- ÉCART joblib : installé 1.5.2, épinglé 1.5.3
- OK  lightgbm : installé 4.7.0, épinglé 4.7.0
- ÉCART matplotlib : installé 3.10.6, épinglé 3.11.2
- ÉCART mlflow : installé 3.14.0, épinglé 3.15.2
- ÉCART numpy : installé 2.3.5, épinglé 2.4.4
- OK  pandas : installé 2.3.3, épinglé 2.3.3
- ÉCART pydantic : installé 2.12.4, épinglé 2.13.4
- ÉCART pytest : installé 8.4.2, épinglé 9.1.1
- ÉCART ruff : installé 0.12.0, épinglé 0.16.7
- ÉCART scikit-learn : installé 1.7.2, épinglé 1.8.0
- ÉCART scipy : installé 1.16.3, épinglé 1.17.1
- ÉCART uvicorn : installé 0.49.0, épinglé 0.52.4
- ÉCART xgboost : installé 3.4.1, épinglé 3.2.0
- OK  nbclient 0.10.2 (exécution du notebook)
- OK  nbformat 5.10.4 (exécution du notebook)
- OK  ipykernel 6.31.0 (exécution du notebook)
- 12 écart(s) de version : les résultats peuvent différer de ceux cités dans le notebook (les contrôles de l'étape 3 le diront).

## 2. Exécution du notebook depuis un noyau neuf

- Début : 2026-10-01T14:35:04+00:00 (UTC) — noyau neuf lancé par nbclient
- Cellules de code : 71 | exécutées : 71 | en erreur : 0
- Compteurs d'exécution continus de 1 à 71 : oui
- Notebook exécuté enregistré : certification_cisia_atlas_v2_execute.ipynb

## 3. Contrôles de cohérence du notebook (§ 15)

- Exécution 20261001T143518Z du 2026-10-01T14:43:30+00:00 : 256 contrôles vérifiés sur 257
- À REVOIR : le fichier de données est le fichier d'origine fourni par l'organisme (empreinte SHA-256)

## 4. Tests automatisés de l'API

- .............................                                            [100%]
- 29 passed in 24.36s

## 5. Garde-fou de qualité et d'équité

- Seuils requis   : F1-macro ≥ 0.68  |  erreur critique ≤ 0.15  |  écart de rappel ≤ 0.15
- Modèle obtenu   : F1-macro = 0.6906  |  erreur critique = 0.0778  |  écart de rappel = 0.0357
- ✅ Modèle accepté — dépasse les seuils minimaux de qualité et d'équité.
-    Rappel : l'écart de rappel est mesuré sur 20 usagers de classe 2 hors UE dans le jeu de test.
-    C'est un garde-fou, pas une preuve d'équité : l'audit doit être refait en continu en production.

## 6. Artefacts du modèle

- OK modele_xgboost.joblib — 581.9 Ko — SHA-256 2d605639edf3920d…
- OK onehot_encoder.joblib — 2.0 Ko — SHA-256 7d8046020057af12…
- OK tfidf_vectorizer.joblib — 2.2 Ko — SHA-256 607f267f94905909…
- OK parametres_imputation.joblib — 0.2 Ko — SHA-256 c77409734ebb019d…
- OK metriques.json — 0.4 Ko — SHA-256 7b5fa086d91e4d5e…
- OK run_mlflow.json — 0.2 Ko — SHA-256 29b517ff390fe358…
- Rechargement réussi — version xgboost_scenario2_v2, seuil 0.2
- F1-macro (test) 0.6906 | erreur critique 0.0778

## 7. Registre MLflow

- Alias « champion » : version 17, run bb217f11286d4b00b9b13254c2f89c59
- Run des artefacts en service : bb217f11286d4b00b9b13254c2f89c59 — identique
