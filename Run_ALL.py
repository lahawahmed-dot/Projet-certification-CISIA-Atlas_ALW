"""

Exécute le notebook depuis un noyau neuf, puis vérifie l'ensemble du projet, et écrit un
rapport factuel (rapport_execution.md).

Étapes :
    1. environnement : versions installées comparées aux versions épinglées ;
    2. exécution complète du notebook dans un noyau neuf, cellule par cellule, dans l'ordre
       -> certification_cisia_atlas_v2_execute.ipynb ;
    3. bilan des contrôles de cohérence écrits par le notebook (§ 15) ;
    4. tests automatisés de l'API (pytest) ;
    5. garde-fou de qualité et d'équité (verifier_seuil_performance.py) ;
    6. rechargement des artefacts et empreintes SHA-256 ;
    7. registre MLflow : l'alias « champion » désigne bien le modèle en service.

Usage, depuis le dossier qui contient ce fichier :
    python executer_et_verifier.py                 # tout
    python executer_et_verifier.py --sans-notebook # vérifications seules (notebook déjà exécuté)

Code de sortie : 0 si toutes les étapes réussissent, 1 sinon.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

RACINE = Path(__file__).resolve().parent
SRC = RACINE / "src"
NOTEBOOK = RACINE / "certification_cisia_atlas.ipynb"
NOTEBOOK_EXECUTE = RACINE / "certification_cisia_atlas_v2_execute.ipynb"
CONTROLES = SRC / "journaux_notebook" / "controles_coherence.json"
ARTEFACTS = SRC / "artefacts_modele"
RAPPORT = RACINE / "rapport_execution.md"
URI_MLFLOW = f"sqlite:///{(SRC / 'mlflow.db').as_posix()}"
NOM_MODELE_REGISTRE = "prediction_retour_emploi"

RESULTATS = []


def etape(nom):
    """Décorateur : exécute une étape, mesure sa durée et enregistre son verdict."""
    def decorateur(fonction):
        def executer(*args, **kwargs):
            print(f"\n=== {nom}")
            debut = time.perf_counter()
            try:
                reussi, details = fonction(*args, **kwargs)
            except Exception as erreur:  # noqa: BLE001 — une étape en échec ne doit pas masquer les suivantes
                reussi, details = False, [f"Exception : {type(erreur).__name__} : {erreur}"]
            duree = time.perf_counter() - debut
            for ligne in details:
                print("   " + ligne)
            print(f"   -> {'RÉUSSI' if reussi else 'ÉCHEC'} ({duree:.0f} s)")
            RESULTATS.append({"etape": nom, "reussi": reussi, "duree_s": round(duree, 1), "details": details})
            return reussi
        return executer
    return decorateur


def lire_versions_epinglees():
    epinglees = {}
    for fichier in ("requirements.txt", "requirements-dev.txt"):
        chemin = SRC / fichier
        if chemin.is_file():
            for ligne in chemin.read_text(encoding="utf-8").splitlines():
                correspondance = re.match(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s#]+)", ligne)
                if correspondance:
                    epinglees[correspondance.group(1).lower()] = correspondance.group(2)
    return epinglees


@etape("1. Environnement")
def verifier_environnement():
    details = [f"Python {platform.python_version()} ({platform.system()} {platform.release()})"]
    ecarts, absents = 0, 0
    for paquet, version_attendue in sorted(lire_versions_epinglees().items()):
        try:
            version_installee = metadata.version(paquet)
        except metadata.PackageNotFoundError:
            version_installee = "absent"
            absents += 1
        conforme = version_installee == version_attendue
        ecarts += not conforme
        details.append(f"{'OK ' if conforme else 'ÉCART'} {paquet} : installé {version_installee}, épinglé {version_attendue}")
    for paquet in ("nbclient", "nbformat", "ipykernel"):
        try:
            details.append(f"OK  {paquet} {metadata.version(paquet)} (exécution du notebook)")
        except metadata.PackageNotFoundError:
            details.append(f"ABSENT {paquet} : installer jupyterlab")
    if ecarts:
        details.append(f"{ecarts} écart(s) de version : les résultats peuvent différer de ceux cités dans le notebook "
                       "(les contrôles de l'étape 3 le diront).")
    return absents == 0, details


@etape("2. Exécution du notebook depuis un noyau neuf")
def executer_notebook(delai_par_cellule):
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    if sys.platform.startswith("win"):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(notebook, timeout=delai_par_cellule, kernel_name="python3",
                            resources={"metadata": {"path": str(RACINE)}})
    debut = datetime.now(timezone.utc)
    erreur = None
    try:
        client.execute()
    except CellExecutionError as exception:
        erreur = exception
    nbformat.write(notebook, NOTEBOOK_EXECUTE)

    cellules_code = [c for c in notebook.cells if c.cell_type == "code"]
    compteurs = [c.get("execution_count") for c in cellules_code]
    en_erreur = [i for i, c in enumerate(cellules_code, start=1)
                 if any(sortie.get("output_type") == "error" for sortie in c.get("outputs", []))]
    ordre_continu = compteurs == list(range(1, len(cellules_code) + 1))
    details = [
        f"Début : {debut.isoformat(timespec='seconds')} (UTC) — noyau neuf lancé par nbclient",
        f"Cellules de code : {len(cellules_code)} | exécutées : {sum(c is not None for c in compteurs)} | en erreur : {len(en_erreur)}",
        f"Compteurs d'exécution continus de 1 à {len(cellules_code)} : {'oui' if ordre_continu else 'non'}",
        f"Notebook exécuté enregistré : {NOTEBOOK_EXECUTE.name}",
    ]
    if erreur is not None:
        details.append("Première erreur : " + str(erreur).strip().splitlines()[-1][:300])
    return erreur is None and not en_erreur and ordre_continu, details


@etape("3. Contrôles de cohérence du notebook (§ 15)")
def lire_controles():
    if not CONTROLES.is_file():
        return False, [f"Fichier absent : {CONTROLES.relative_to(RACINE)} (le notebook ne s'est pas exécuté jusqu'au bout)"]
    bilan = json.loads(CONTROLES.read_text(encoding="utf-8"))
    details = [f"Exécution {bilan['id_execution']} du {bilan['date_utc']} : "
               f"{bilan['controles_verifies']} contrôles vérifiés sur {bilan['controles_total']}"]
    details += ["À REVOIR : " + c["constat"] for c in bilan["controles"] if not c["verifie"]]
    return bilan["controles_verifies"] == bilan["controles_total"], details


@etape("4. Tests automatisés de l'API")
def lancer_tests():
    environnement = {cle: valeur for cle, valeur in os.environ.items() if cle not in ("CLE_ADMIN", "CLE_API")}
    environnement["PYTHONIOENCODING"] = "utf-8"
    resultat = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=SRC,
                              env=environnement, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lignes = [ligne for ligne in resultat.stdout.strip().splitlines() if ligne.strip()]
    return resultat.returncode == 0, lignes[-15:] or [resultat.stderr[-500:]]


@etape("5. Garde-fou de qualité et d'équité")
def lancer_garde_fou():
    environnement = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    resultat = subprocess.run([sys.executable, "verifier_seuil_performance.py", "--f1-macro-min", "0.68",
                               "--erreur-critique-max", "0.15", "--ecart-rappel-max", "0.15"], cwd=SRC,
                              env=environnement, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return resultat.returncode == 0, [ligne for ligne in resultat.stdout.strip().splitlines() if ligne.strip()]


@etape("6. Artefacts du modèle")
def verifier_artefacts():
    import joblib

    details, reussi = [], True
    for nom in ("modele_xgboost.joblib", "onehot_encoder.joblib", "tfidf_vectorizer.joblib",
                "parametres_imputation.joblib", "metriques.json", "run_mlflow.json"):
        chemin = ARTEFACTS / nom
        if not chemin.is_file():
            reussi = False
            details.append(f"ABSENT {nom}")
            continue
        empreinte = hashlib.sha256(chemin.read_bytes()).hexdigest()
        details.append(f"OK {nom} — {chemin.stat().st_size / 1024:.1f} Ko — SHA-256 {empreinte[:16]}…")
    if reussi:
        joblib.load(ARTEFACTS / "modele_xgboost.joblib")
        parametres = joblib.load(ARTEFACTS / "parametres_imputation.joblib")
        metriques = json.loads((ARTEFACTS / "metriques.json").read_text(encoding="utf-8"))
        details.append(f"Rechargement réussi — version {parametres['version_modele']}, seuil {parametres['seuil_classe2']}")
        details.append(f"F1-macro (test) {metriques['f1_macro_test']:.4f} | erreur critique {metriques['taux_erreur_critique_test']:.4f}")
        reussi = parametres["seuil_classe2"] == 0.20
    return reussi, details


@etape("7. Registre MLflow")
def verifier_mlflow():
    from mlflow.tracking import MlflowClient

    suivi = json.loads((ARTEFACTS / "run_mlflow.json").read_text(encoding="utf-8"))
    client = MlflowClient(tracking_uri=URI_MLFLOW)
    version = client.get_model_version_by_alias(NOM_MODELE_REGISTRE, "champion")
    conforme = version.run_id == suivi["run_id"]
    return conforme, [f"Alias « champion » : version {version.version}, run {version.run_id}",
                      f"Run des artefacts en service : {suivi['run_id']} — {'identique' if conforme else 'DIFFÉRENTE'}"]


def ecrire_rapport():
    maintenant = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lignes = ["# Rapport d'exécution", "",
              f"Généré le {maintenant} (UTC) par `executer_et_verifier.py`, sur {platform.system()} {platform.release()}, "
              f"Python {platform.python_version()}.", "",
              "| Étape | Verdict | Durée (s) |", "|---|---|---|"]
    lignes += [f"| {r['etape']} | {'réussi' if r['reussi'] else 'ÉCHEC'} | {r['duree_s']} |" for r in RESULTATS]
    for r in RESULTATS:
        lignes += ["", f"## {r['etape']}", ""] + [f"- {ligne}" for ligne in r["details"]]
    RAPPORT.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"\nRapport écrit : {RAPPORT.name}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    analyseur.add_argument("--sans-notebook", action="store_true", help="ne pas réexécuter le notebook")
    analyseur.add_argument("--delai-par-cellule", type=int, default=3600, help="délai maximal par cellule, en secondes")
    arguments = analyseur.parse_args()

    verifier_environnement()
    if not arguments.sans_notebook:
        executer_notebook(arguments.delai_par_cellule)
    lire_controles()
    lancer_tests()
    lancer_garde_fou()
    verifier_artefacts()
    verifier_mlflow()
    ecrire_rapport()
    echecs = [r["etape"] for r in RESULTATS if not r["reussi"]]
    print("\nToutes les étapes ont réussi." if not echecs else "\nÉtapes en échec : " + " ; ".join(echecs))
    sys.exit(1 if echecs else 0)


if __name__ == "__main__":
    main()
