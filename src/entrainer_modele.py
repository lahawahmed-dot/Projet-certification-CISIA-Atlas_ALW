"""
entrainer_modele.py
====================
Script autonome d'entraînement du modèle final retenu (voir notebook, étapes 2-3-5-6-7) :
XGBoost, Scénario 2 (sans variable sensible), seuil de décision à 0,20 sur P(classe=2).

Le couple (modèle, seuil) n'a PAS été choisi en regardant le jeu de test : il est issu
d'une sélection out-of-fold sur le jeu d'entraînement, avec une règle de décision
annoncée avant de regarder les résultats (notebook, étape 6.4) :

    « minimiser le taux d'erreur critique (vrais 2 prédits 0) sous contrainte
      F1-macro >= 0,68 en validation croisée »

Ce plancher de 0,68 est le même que celui appliqué par le garde-fou de la CI
(verifier_seuil_performance.py) : la règle de sélection et la règle de promotion
sont volontairement identiques.

Usage :
    python entrainer_modele.py                      # entraîne le champion
    python entrainer_modele.py --sortie artefacts_candidat   # entraîne un challenger

Produit :
    - <sortie>/modele_xgboost.joblib
    - <sortie>/onehot_encoder.joblib
    - <sortie>/tfidf_vectorizer.joblib
    - <sortie>/parametres_imputation.joblib
    - <sortie>/metriques.json
    - une run MLflow (sqlite:///mlflow.db par défaut)
"""
import argparse
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")

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

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
CHEMIN_DONNEES = os.environ.get("CHEMIN_DONNEES", "dataset_trajectoire_emploi.csv")
DOSSIER_ARTEFACTS = os.environ.get("DOSSIER_ARTEFACTS", "artefacts_modele")
SEUIL_CLASSE2 = 0.20
VERSION_MODELE = "xgboost_scenario2_v2"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Le plancher de F1-macro utilisé par la règle de sélection ET par le garde-fou CI.
PLANCHER_F1_MACRO = 0.68


def preparer_donnees(chemin_csv: str):
    """Reproduit exactement le pipeline de nettoyage/feature engineering du notebook (étapes 2-3).

    Note sur le fichier source : le CSV fourni par l'organisme utilise « ; » comme
    séparateur. Le fichier livré ici a été converti en « , » ; cette conversion a
    remplacé les points-virgules internes à 1 043 verbatims par des virgules. Sans
    effet sur la vectorisation (la ponctuation est retirée par le TF-IDF), mais
    signalé par souci de traçabilité. Depuis cette version, le séparateur est détecté
    automatiquement : le script s'exécute indifféremment sur le fichier d'origine ou
    sur la copie convertie.
    """
    # Le fichier fourni par l'organisme utilise « ; » comme séparateur ; la copie de travail
    # utilise « , ». Plutôt que de dépendre de l'une des deux versions, on détecte le
    # séparateur sur la ligne d'en-tête : le pipeline s'exécute alors indifféremment sur le
    # fichier d'origine ou sur la copie, ce qui est la condition d'une reproductibilité qui
    # ne repose pas sur un fichier retouché.
    with open(chemin_csv, encoding="utf-8") as fichier:
        entete = fichier.readline()
    separateur = ";" if entete.count(";") > entete.count(",") else ","

    df = pd.read_csv(chemin_csv, sep=separateur, dtype={"code_insee_commune": str})

    # --- Feature engineering (étape 2) ---
    df["departement"] = df["code_insee_commune"].astype(str).str.zfill(5).str[:2]

    # Cohérence âge / ancienneté. Calculé AVANT toute imputation : si l'âge est manquant,
    # la comparaison est fausse et le drapeau vaut 0. L'API applique exactement la même
    # règle (age is None -> drapeau à 0), pour qu'un usager sans âge ne reçoive jamais
    # en production un signal que le modèle n'a jamais vu à l'entraînement.
    age_debut_activite = df["age"] - df["anciennete_poste_ans"]
    df["anciennete_incoherente"] = (age_debut_activite < 14).fillna(False).astype(int)

    # Contrôle de format des codes INSEE (5 chiffres, ou Corse 2A/2B + 3 chiffres)
    motif_insee_valide = re.compile(r"^\d{5}$|^2[AB]\d{3}$")
    anomalies = ~df["code_insee_commune"].astype(str).apply(lambda x: bool(motif_insee_valide.match(x)))
    if anomalies.sum() > 0:
        print(f"⚠️  {anomalies.sum()} code(s) INSEE au format inattendu détecté(s).")

    feature_cols = [
        "age", "niveau_diplome", "anciennete_poste_ans", "code_rome_vise", "departement",
        "est_allocataire", "nationalite_hors_ue", "synthese_entretien", "anciennete_incoherente",
    ]
    X = df[feature_cols].copy()
    y = df["classe_retour_emploi"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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
    }
    # `nationalite_hors_ue` du jeu de test est renvoyée séparément : elle ne sert JAMAIS
    # au modèle, uniquement à l'audit d'équité a posteriori (notebook, étape 8.2).
    nat_test = X_test["nationalite_hors_ue"].to_numpy()
    return X_s2_train, X_s2_test, y_train.to_numpy(), y_test.to_numpy(), artefacts, nat_test


def predire_avec_seuil(probas, seuil=SEUIL_CLASSE2):
    """Si P(classe 2) dépasse le seuil -> on prédit 2. Sinon, on choisit entre 0 et 1."""
    return np.where(probas[:, 2] >= seuil, 2, np.argmax(probas[:, :2], axis=1))


def entrainer_et_evaluer(X_train, y_train, X_test, y_test, nat_test):
    modele = XGBClassifier(random_state=42, eval_metric="mlogloss")
    modele.fit(X_train, y_train)

    probas = modele.predict_proba(X_test)
    predictions = predire_avec_seuil(probas)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    metriques = {
        "accuracy_test": accuracy_score(y_test, predictions),
        "f1_macro_test": f1_score(y_test, predictions, average="macro"),
        "taux_erreur_critique_test": cm[2, 0] / cm[2].sum(),
    }

    # --- Audit d'équité : le modèle est aveugle à la nationalité, mais son comportement
    # ne l'est pas forcément. On mesure, on ne suppose pas (notebook, étape 8.2).
    for valeur in (0, 1):
        masque = nat_test == valeur
        cm_g = confusion_matrix(y_test[masque], predictions[masque], labels=[0, 1, 2])
        suffixe = "hors_ue" if valeur == 1 else "ue"
        metriques[f"part_predite_classe2_{suffixe}"] = float(np.mean(predictions[masque] == 2))
        metriques[f"rappel_classe2_{suffixe}"] = float(cm_g[2, 2] / cm_g[2].sum()) if cm_g[2].sum() else float("nan")

    return modele, metriques


def main():
    parser = argparse.ArgumentParser(description="Entraîne le modèle final et enregistre ses artefacts.")
    parser.add_argument("--sortie", default=DOSSIER_ARTEFACTS,
                        help="Dossier de sortie (utiliser artefacts_candidat/ pour un challenger)")
    args = parser.parse_args()
    dossier = args.sortie

    print("1/4 — Préparation des données (Scénario 2)...")
    X_train, X_test, y_train, y_test, artefacts, nat_test = preparer_donnees(CHEMIN_DONNEES)
    print(f"     X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    print(f"2/4 — Entraînement du modèle final (XGBoost, seuil={SEUIL_CLASSE2})...")
    modele, metriques = entrainer_et_evaluer(X_train, y_train, X_test, y_test, nat_test)
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
    print(f"     Artefacts écrits dans {dossier}/")

    # Métriques dans un fichier séparé, lu par verifier_seuil_performance.py
    # (garde-fou de qualité, exécuté juste après dans le pipeline CI/CD).
    with open(f"{dossier}/metriques.json", "w") as f:
        json.dump(metriques, f, indent=2)

    print("4/4 — Enregistrement de la run MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("prediction_retour_emploi")
    with mlflow.start_run(run_name=VERSION_MODELE):
        mlflow.log_param("modele", "XGBoost")
        mlflow.log_param("scenario_donnees", "2_sans_variable_sensible")
        mlflow.log_param("seuil_classe2", SEUIL_CLASSE2)
        mlflow.log_param("regle_de_selection", f"min erreur critique | F1-macro >= {PLANCHER_F1_MACRO} (out-of-fold)")
        mlflow.log_param("random_state", 42)
        for k, v in metriques.items():
            mlflow.log_metric(k, v)
        mlflow.xgboost.log_model(modele, name="modele")
    print("     Run MLflow enregistrée.")

    print("\n✅ Entraînement terminé avec succès.")


if __name__ == "__main__":
    main()
