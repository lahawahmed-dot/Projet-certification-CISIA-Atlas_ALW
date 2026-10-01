"""
entrainer_modele.py
====================
Script autonome d'entraînement du modèle retenu dans le notebook :
XGBoost, Scénario 2 (sans variable sensible), seuil de décision à 0,20 sur P(classe = 2).

Le couple (modèle, seuil) a été choisi en validation croisée sur le jeu d'entraînement,
avec la règle « minimiser le taux d'erreur critique (vrais 2 prédits 0) sous contrainte
d'un F1-macro >= 0,68 » (notebook, § 8). Ce plancher est le même que celui du garde-fou
de la CI (verifier_seuil_performance.py) : la règle qui choisit et la règle qui autorise
la mise en production sont identiques.

Le notebook n'enregistre pas lui-même le modèle : il exécute ce script, puis vérifie que
le modèle produit donne exactement les mêmes probabilités que le sien. Il n'existe donc
qu'une seule implémentation du prétraitement pour l'entraînement.

Usage :
    python entrainer_modele.py                                   # champion
    python entrainer_modele.py --sortie artefacts_candidat --avec-feedback   # challenger

Produit dans <sortie>/ :
    modele_xgboost.joblib, onehot_encoder.joblib, tfidf_vectorizer.joblib,
    parametres_imputation.joblib, metriques.json, run_mlflow.json
et une run MLflow (MLFLOW_TRACKING_URI, par défaut sqlite:///mlflow.db).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import warnings

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from retours_conseillers import charger_retours_conseillers

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
CHEMIN_DONNEES = os.environ.get("CHEMIN_DONNEES", "dataset_trajectoire_emploi.csv")
DOSSIER_ARTEFACTS = os.environ.get("DOSSIER_ARTEFACTS", "artefacts_modele")
FICHIER_JOURNAL = os.environ.get("FICHIER_JOURNAL", "journal_requetes.jsonl")
FICHIER_FEEDBACK = os.environ.get("FICHIER_FEEDBACK", "feedback_reentrainement.jsonl")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
SEUIL_CLASSE2 = 0.20
VERSION_MODELE = "xgboost_scenario2_v2"
GRAINE = 42

# Le plancher de F1-macro utilisé par la règle de sélection ET par le garde-fou CI.
PLANCHER_F1_MACRO = 0.68

FEATURE_COLS = [
    "age", "niveau_diplome", "anciennete_poste_ans", "code_rome_vise", "departement",
    "est_allocataire", "nationalite_hors_ue", "synthese_entretien", "anciennete_incoherente",
]


def ajouter_variables_derivees(tableau: pd.DataFrame) -> pd.DataFrame:
    """Département (2 premiers caractères) et drapeau d'incohérence âge / ancienneté.

    Le drapeau est calculé AVANT toute imputation : un âge manquant donne une comparaison
    fausse, donc un drapeau à 0. L'API applique exactement la même règle.
    """
    tableau = tableau.copy()
    tableau["departement"] = tableau["code_insee_commune"].astype(str).str.zfill(5).str[:2]
    age_debut_activite = tableau["age"] - tableau["anciennete_poste_ans"]
    tableau["anciennete_incoherente"] = (age_debut_activite < 14).fillna(False).astype(int)
    return tableau


def preparer_donnees(chemin_csv: str, retours: pd.DataFrame | None = None):
    """Reproduit le pipeline de préparation du notebook (sections 4 et 5).

    `retours` (facultatif) contient des lignes issues des retours conseillers. Elles sont
    ajoutées au SEUL jeu d'entraînement, après la séparation : le jeu de test de référence
    reste identique, si bien qu'un challenger et le champion sont comparés sur les mêmes
    500 usagers.
    """
    with open(chemin_csv, encoding="utf-8") as fichier:
        entete = fichier.readline()
    separateur = ";" if entete.count(";") > entete.count(",") else ","

    df = pd.read_csv(chemin_csv, sep=separateur, dtype={"code_insee_commune": str})
    df = ajouter_variables_derivees(df)

    motif_insee_valide = re.compile(r"^\d{5}$|^2[AB]\d{3}$")
    anomalies = ~df["code_insee_commune"].astype(str).apply(lambda x: bool(motif_insee_valide.match(x)))
    if anomalies.sum() > 0:
        print(f"Attention : {anomalies.sum()} code(s) INSEE au format inattendu.")

    X = df[FEATURE_COLS].copy()
    y = df["classe_retour_emploi"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=GRAINE
    )

    nb_retours = 0
    if retours is not None and len(retours) > 0:
        lignes = ajouter_variables_derivees(retours)
        lignes["nationalite_hors_ue"] = np.nan  # jamais collectée par l'API (Scénario 2)
        lignes.index = [f"retour_{i}" for i in range(len(lignes))]
        X_train = pd.concat([X_train, lignes[FEATURE_COLS]])
        y_train = pd.concat([y_train, lignes["classe_retour_emploi"].astype(int).rename(y_train.name)])
        nb_retours = len(lignes)

    # --- Indicateurs de valeur manquante (avant imputation) ---
    for col in ["age", "niveau_diplome", "est_allocataire", "synthese_entretien"]:
        X_train[f"{col}_manquant"] = X_train[col].isna().astype(int)
        X_test[f"{col}_manquant"] = X_test[col].isna().astype(int)

    # --- Imputation (statistiques calculées sur le train uniquement) ---
    age_median = X_train["age"].median()
    X_train["age"] = X_train["age"].fillna(age_median)
    X_test["age"] = X_test["age"].fillna(age_median)

    diplome_mode = X_train["niveau_diplome"].mode()[0]
    X_train["niveau_diplome"] = X_train["niveau_diplome"].fillna(diplome_mode)
    X_test["niveau_diplome"] = X_test["niveau_diplome"].fillna(diplome_mode)

    alloc_mode = X_train["est_allocataire"].mode()[0]
    X_train["est_allocataire"] = X_train["est_allocataire"].fillna(alloc_mode).astype(int)
    X_test["est_allocataire"] = X_test["est_allocataire"].fillna(alloc_mode).astype(int)

    X_train["synthese_entretien"] = X_train["synthese_entretien"].fillna("")
    X_test["synthese_entretien"] = X_test["synthese_entretien"].fillna("")

    # --- Encodage ordinal du diplôme ---
    mapping_diplome = {"Sans diplôme": 0, "Bac": 1, "Bac+2": 2, "Bac+5": 3}
    X_train["niveau_diplome_encode"] = X_train["niveau_diplome"].map(mapping_diplome)
    X_test["niveau_diplome_encode"] = X_test["niveau_diplome"].map(mapping_diplome)

    # --- One-hot département + code ROME (fit sur train uniquement) ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_train[["departement", "code_rome_vise"]])
    train_ohe = pd.DataFrame(
        ohe.transform(X_train[["departement", "code_rome_vise"]]),
        columns=ohe.get_feature_names_out(["departement", "code_rome_vise"]),
        index=X_train.index,
    )
    test_ohe = pd.DataFrame(
        ohe.transform(X_test[["departement", "code_rome_vise"]]),
        columns=ohe.get_feature_names_out(["departement", "code_rome_vise"]),
        index=X_test.index,
    )

    # --- Nettoyage texte + TF-IDF (fit sur train uniquement) ---
    X_train["synthese_entretien"] = X_train["synthese_entretien"].str.lower().str.strip()
    X_test["synthese_entretien"] = X_test["synthese_entretien"].str.lower().str.strip()

    tfidf = TfidfVectorizer()
    tfidf.fit(X_train["synthese_entretien"])
    texte_train = tfidf.transform(X_train["synthese_entretien"])
    texte_test = tfidf.transform(X_test["synthese_entretien"])

    # --- Assemblage Scénario 2 (sans nationalite_hors_ue) ---
    cols_num_bin = [
        "age", "anciennete_poste_ans", "niveau_diplome_encode", "est_allocataire", "nationalite_hors_ue",
        "age_manquant", "niveau_diplome_manquant", "est_allocataire_manquant",
        "synthese_entretien_manquant", "anciennete_incoherente",
    ]
    tabulaire_train = pd.concat([X_train[cols_num_bin].reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)
    tabulaire_test = pd.concat([X_test[cols_num_bin].reset_index(drop=True), test_ohe.reset_index(drop=True)], axis=1)

    colonnes_sans_sensible = [c for c in tabulaire_train.columns if c != "nationalite_hors_ue"]
    tab_tr = csr_matrix(tabulaire_train[colonnes_sans_sensible].values)
    tab_te = csr_matrix(tabulaire_test[colonnes_sans_sensible].values)

    X_s2_train = hstack([tab_tr, texte_train]).tocsr()
    X_s2_test = hstack([tab_te, texte_test]).tocsr()

    artefacts = {
        "ohe": ohe,
        "tfidf": tfidf,
        "age_median": float(age_median),
        "diplome_mode": diplome_mode,
        "alloc_mode": int(alloc_mode),
        "mapping_diplome": mapping_diplome,
        "colonnes_tabulaires": colonnes_sans_sensible,
        "nb_retours": nb_retours,
    }
    # `nationalite_hors_ue` du jeu de test est renvoyée séparément : elle ne sert JAMAIS
    # au modèle, uniquement à l'audit d'équité a posteriori (notebook, § 10).
    nat_test = X_test["nationalite_hors_ue"].to_numpy()
    return X_s2_train, X_s2_test, y_train.to_numpy(), y_test.to_numpy(), artefacts, nat_test


def predire_avec_seuil(probas, seuil=SEUIL_CLASSE2):
    """Si P(classe 2) atteint le seuil -> 2. Sinon, on choisit entre 0 et 1."""
    return np.where(probas[:, 2] >= seuil, 2, np.argmax(probas[:, :2], axis=1))


def entrainer_et_evaluer(X_train, y_train, X_test, y_test, nat_test):
    modele = XGBClassifier(random_state=GRAINE, eval_metric="mlogloss")
    modele.fit(X_train, y_train)

    probas = modele.predict_proba(X_test)
    predictions = predire_avec_seuil(probas)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    metriques = {
        "accuracy_test": accuracy_score(y_test, predictions),
        "f1_macro_test": f1_score(y_test, predictions, average="macro"),
        "taux_erreur_critique_test": cm[2, 0] / cm[2].sum(),
    }

    # --- Audit d'équité : le modèle est aveugle à la nationalité, son comportement ne
    # l'est pas forcément. On mesure, on ne suppose pas (notebook, § 10).
    for valeur in (0, 1):
        masque = nat_test == valeur
        cm_g = confusion_matrix(y_test[masque], predictions[masque], labels=[0, 1, 2])
        suffixe = "hors_ue" if valeur == 1 else "ue"
        metriques[f"part_predite_classe2_{suffixe}"] = float(np.mean(predictions[masque] == 2))
        metriques[f"rappel_classe2_{suffixe}"] = float(cm_g[2, 2] / cm_g[2].sum()) if cm_g[2].sum() else float("nan")
        metriques[f"effectif_classe2_{suffixe}_test"] = int(cm_g[2].sum())

    return modele, metriques


def empreinte_sha256(chemin: str) -> str:
    with open(chemin, "rb") as fichier:
        return hashlib.sha256(fichier.read()).hexdigest()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        # Sous Windows, une sortie redirigée (subprocess) n'est pas en UTF-8 par défaut.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Entraîne le modèle final et enregistre ses artefacts.")
    parser.add_argument("--sortie", default=DOSSIER_ARTEFACTS,
                        help="Dossier de sortie (artefacts_candidat/ pour un challenger)")
    parser.add_argument("--avec-feedback", action="store_true",
                        help="Ajoute au jeu d'entraînement les retours conseillers journalisés")
    args = parser.parse_args()
    dossier = args.sortie
    est_challenger = args.avec_feedback or os.path.abspath(dossier) != os.path.abspath(DOSSIER_ARTEFACTS)
    role = "challenger" if est_challenger else "champion"

    retours, diagnostic = None, {"retours_integres": 0}
    if args.avec_feedback:
        retours, diagnostic = charger_retours_conseillers(FICHIER_JOURNAL, FICHIER_FEEDBACK)
        print(f"0/4 — Retours conseillers : {diagnostic}")

    print("1/4 — Préparation des données (Scénario 2)...")
    X_train, X_test, y_train, y_test, artefacts, nat_test = preparer_donnees(CHEMIN_DONNEES, retours)
    print(f"     X_train: {X_train.shape}  |  X_test: {X_test.shape}  |  retours intégrés : {artefacts['nb_retours']}")

    print(f"2/4 — Entraînement du modèle (XGBoost, seuil={SEUIL_CLASSE2})...")
    modele, metriques = entrainer_et_evaluer(X_train, y_train, X_test, y_test, nat_test)
    metriques["nb_retours_conseillers_integres"] = int(artefacts["nb_retours"])
    for k, v in metriques.items():
        print(f"     {k}: {v:.4f}")

    print("3/4 — Sauvegarde des artefacts sur disque...")
    os.makedirs(dossier, exist_ok=True)
    joblib.dump(modele, f"{dossier}/modele_xgboost.joblib")
    joblib.dump(artefacts["ohe"], f"{dossier}/onehot_encoder.joblib")
    joblib.dump(artefacts["tfidf"], f"{dossier}/tfidf_vectorizer.joblib")
    joblib.dump(
        {
            "age_median": artefacts["age_median"],
            "diplome_mode": artefacts["diplome_mode"],
            "alloc_mode": artefacts["alloc_mode"],
            "mapping_diplome": artefacts["mapping_diplome"],
            "seuil_classe2": SEUIL_CLASSE2,
            "version_modele": VERSION_MODELE,
        },
        f"{dossier}/parametres_imputation.joblib",
    )
    # Métriques dans un fichier séparé, lu par verifier_seuil_performance.py (garde-fou CI)
    # et par la route /retrain/run (comparaison challenger / champion).
    with open(f"{dossier}/metriques.json", "w", encoding="utf-8") as f:
        json.dump(metriques, f, indent=2)
    print(f"     Artefacts écrits dans {dossier}/")

    print("4/4 — Enregistrement de la run MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("prediction_retour_emploi")
    with mlflow.start_run(run_name=f"{VERSION_MODELE}_{role}") as run:
        mlflow.set_tags({"role": role, "source": "entrainer_modele.py"})
        mlflow.log_params({
            "modele": "XGBoost",
            "scenario_donnees": "2_sans_variable_sensible",
            "seuil_classe2": SEUIL_CLASSE2,
            "regle_de_selection": f"min erreur critique | F1-macro >= {PLANCHER_F1_MACRO} (out-of-fold)",
            "random_state": GRAINE,
            "avec_feedback": bool(args.avec_feedback),
            "sha256_donnees": empreinte_sha256(CHEMIN_DONNEES),
        })
        mlflow.log_metrics({k: float(v) for k, v in metriques.items()})
        infos_modele = mlflow.xgboost.log_model(modele, name="modele")
    with open(f"{dossier}/run_mlflow.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run.info.run_id, "model_uri": infos_modele.model_uri, "role": role,
                   "retours_conseillers": diagnostic}, f, indent=2)
    print(f"     Run MLflow enregistrée : {run.info.run_id}")

    print("\nEntraînement terminé avec succès.")


if __name__ == "__main__":
    main()
