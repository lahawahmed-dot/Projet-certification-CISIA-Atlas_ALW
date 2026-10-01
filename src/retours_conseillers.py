"""
retours_conseillers.py
======================
Transforme les retours correctifs des conseillers (route POST /retrain) en lignes
d'apprentissage exploitables par `entrainer_modele.py`.

Un retour ne contient que l'identifiant de session et la classe réellement observée.
Les caractéristiques de l'usager se trouvent dans le journal des requêtes, écrit par
POST /predict. Ce module relie les deux fichiers par `session_id` :

    journal_requetes.jsonl        feedback_reentrainement.jsonl
    (entrée + sortie + date)  +   (session_id + classe observée)
                    \\                /
                     ligne d'apprentissage (mêmes colonnes que le CSV)

Règles appliquées :
    - seule la dernière prédiction d'une session est retenue ;
    - seul le dernier retour d'une session est retenu (un conseiller peut se corriger) ;
    - un retour dont la session est absente du journal est ignoré et compté ;
    - l'ancienneté est recalculée à la date de la prédiction, comme dans l'API ;
    - une synthèse vide est traitée comme une synthèse absente, comme à l'entraînement ;
    - la nationalité n'est jamais disponible : l'API ne la collecte pas (Scénario 2).

Ce module ne dépend que de la bibliothèque standard et de pandas : il est testable seul.
"""
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

COLONNES_RETOURS = [
    "session_id",
    "age",
    "niveau_diplome",
    "anciennete_poste_ans",
    "code_rome_vise",
    "code_insee_commune",
    "est_allocataire",
    "synthese_entretien",
    "classe_retour_emploi",
]


def _lire_jsonl(chemin):
    """Lit un fichier JSON Lines en ignorant les lignes illisibles (comptées)."""
    lignes, illisibles = [], 0
    chemin = Path(chemin)
    if not chemin.is_file():
        return lignes, illisibles
    with chemin.open(encoding="utf-8") as fichier:
        for brut in fichier:
            if not brut.strip():
                continue
            try:
                lignes.append(json.loads(brut))
            except json.JSONDecodeError:
                illisibles += 1
    return lignes, illisibles


def charger_retours_conseillers(chemin_journal, chemin_feedback):
    """Associe chaque retour conseiller à la prédiction journalisée de la même session.

    Retourne un couple (DataFrame, diagnostic) :
        - le DataFrame a les colonnes de COLONNES_RETOURS, une ligne par retour exploitable ;
        - le diagnostic compte les retours lus, intégrés et écartés, pour la traçabilité.
    """
    entrees_journal, journal_illisibles = _lire_jsonl(chemin_journal)
    retours_bruts, retours_illisibles = _lire_jsonl(chemin_feedback)

    derniere_prediction = {}
    for entree in entrees_journal:
        session = entree.get("session_id")
        if session:
            derniere_prediction[session] = entree

    dernier_retour, retours_invalides = {}, 0
    for retour in retours_bruts:
        session = retour.get("session_id")
        classe = retour.get("classe_reelle_observee")
        if session and classe in (0, 1, 2):
            dernier_retour[session] = retour
        else:
            retours_invalides += 1

    lignes, sans_prediction = [], 0
    for session, retour in dernier_retour.items():
        prediction = derniere_prediction.get(session)
        if prediction is None:
            sans_prediction += 1
            continue
        donnees = prediction.get("entree", {})
        jour_prediction = datetime.fromisoformat(prediction["timestamp"]).date()
        debut_poste = date.fromisoformat(donnees["date_debut_poste"])
        lignes.append({
            "session_id": session,
            "age": donnees.get("age"),
            "niveau_diplome": donnees.get("niveau_diplome"),
            "anciennete_poste_ans": (jour_prediction - debut_poste).days / 365.25,
            "code_rome_vise": donnees.get("code_rome_vise"),
            "code_insee_commune": donnees.get("code_insee_commune"),
            "est_allocataire": donnees.get("est_allocataire"),
            "synthese_entretien": donnees.get("synthese_entretien") or None,
            "classe_retour_emploi": int(retour["classe_reelle_observee"]),
        })

    tableau = pd.DataFrame(lignes, columns=COLONNES_RETOURS)
    for colonne in ("age", "est_allocataire", "anciennete_poste_ans"):
        tableau[colonne] = pd.to_numeric(tableau[colonne], errors="coerce")
    tableau["code_insee_commune"] = tableau["code_insee_commune"].astype(str)

    diagnostic = {
        "retours_lus": len(retours_bruts),
        "retours_invalides": retours_invalides,
        "retours_retenus_apres_deduplication": len(dernier_retour),
        "retours_sans_prediction_journalisee": sans_prediction,
        "retours_integres": len(tableau),
        "lignes_illisibles": journal_illisibles + retours_illisibles,
    }
    return tableau, diagnostic
