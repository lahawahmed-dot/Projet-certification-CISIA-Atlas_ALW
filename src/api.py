"""
api.py
======
API de prédiction du risque de retour à l'emploi.

Charge le modèle et les artefacts sauvegardés par `entrainer_modele.py` (dossier
`artefacts_modele/` par défaut), et expose :
    - GET  /health       : contrôle de vie
    - GET  /metrics      : compteurs au format Prometheus (monitoring)
    - POST /predict      : prédiction du niveau de risque pour un usager
    - POST /retrain      : enregistrement d'un retour conseiller (feedback)
    - POST /retrain/run  : ré-entraînement CHALLENGER intégrant les retours conseillers
                           (clé admin requise), sans jamais promouvoir automatiquement

Sécurité (voir README.md, section « Sécurité et cycle de vie de la donnée ») :
    - CLE_API      : si définie, toute requête doit porter l'en-tête `X-Cle-Api`.
    - CLE_ADMIN    : si définie, exigée en plus sur /retrain/run.
    - ORIGINES_AUTORISEES : liste CORS séparée par des virgules (défaut « * », démo).
    - RETENTION_JOURNAL_JOURS : purge automatique du journal au démarrage (défaut 365).

Lancement en local :
    uvicorn api:app --reload

Documentation interactive une fois lancée :
    http://127.0.0.1:8000/docs
"""
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from scipy.sparse import csr_matrix, hstack

DOSSIER_API = Path(__file__).resolve().parent
DOSSIER_ARTEFACTS = os.environ.get("DOSSIER_ARTEFACTS", "artefacts_modele")
DOSSIER_CANDIDAT = os.environ.get("DOSSIER_CANDIDAT", "artefacts_candidat")
FICHIER_JOURNAL = os.environ.get("FICHIER_JOURNAL", "journal_requetes.jsonl")
FICHIER_FEEDBACK = os.environ.get("FICHIER_FEEDBACK", "feedback_reentrainement.jsonl")
CHEMIN_DONNEES = os.environ.get("CHEMIN_DONNEES", "dataset_trajectoire_emploi.csv")

CLE_API = os.environ.get("CLE_API", "")
CLE_ADMIN = os.environ.get("CLE_ADMIN", "")
ORIGINES_AUTORISEES = [o.strip() for o in os.environ.get("ORIGINES_AUTORISEES", "*").split(",") if o.strip()]
RETENTION_JOURNAL_JOURS = int(os.environ.get("RETENTION_JOURNAL_JOURS", "365"))

ARTEFACTS: dict = {}
COMPTEURS: Counter = Counter()
LATENCE_TOTALE_MS = {"somme": 0.0}
AGE_MIN_ENTRAINEMENT = 18
AGE_MAX_ENTRAINEMENT = 63


# --------------------------------------------------------------------------------------
# Cycle de vie de la donnée : purge du journal (art. 5.1.e RGPD)
# --------------------------------------------------------------------------------------
def purger_journal(chemin: str, retention_jours: int) -> int:
    """Supprime du journal les lignes plus anciennes que la durée de conservation.
    Implémentation volontairement simple (fichier JSON Lines) : en production, le
    journal doit être une base chiffrée avec purge planifiée, et la durée exacte doit
    être arbitrée avec le DPO (durée d'accompagnement + délai de recours). Ce qui compte
    ici est que la durée de conservation soit un paramètre effectif du système, et non
    une intention écrite dans un rapport. """
    if not os.path.exists(chemin) or retention_jours <= 0:
        return 0
    limite = datetime.now(timezone.utc) - timedelta(days=retention_jours)
    gardees, supprimees = [], 0
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            try:
                horodatage = datetime.fromisoformat(json.loads(ligne)["timestamp"])
            except Exception:  # noqa: BLE001 — une ligne illisible est conservée, jamais perdue silencieusement
                gardees.append(ligne)
                continue
            if horodatage >= limite:
                gardees.append(ligne)
            else:
                supprimees += 1
    if supprimees:
        # Écriture atomique : on écrit d'abord un fichier temporaire complet, puis on le
        # substitue à l'original par un renommage — opération atomique au niveau du système
        # de fichiers. Une coupure pendant la purge laisse donc soit l'ancien journal intact,
        # soit le nouveau complet, jamais un journal tronqué. Ce fichier ayant une valeur
        # probatoire (voir README.md, « Responsabilité juridique »), il ne doit jamais pouvoir
        # être perdu à moitié.
        temporaire = f"{chemin}.tmp"
        with open(temporaire, "w", encoding="utf-8") as f:
            f.writelines(gardees)
        os.replace(temporaire, chemin)
    return supprimees


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et les artefacts une seule fois, au démarrage de l'API."""
    try:
        ARTEFACTS["modele"] = joblib.load(f"{DOSSIER_ARTEFACTS}/modele_xgboost.joblib")
        ARTEFACTS["ohe"] = joblib.load(f"{DOSSIER_ARTEFACTS}/onehot_encoder.joblib")
        ARTEFACTS["tfidf"] = joblib.load(f"{DOSSIER_ARTEFACTS}/tfidf_vectorizer.joblib")
        params = joblib.load(f"{DOSSIER_ARTEFACTS}/parametres_imputation.joblib")
        ARTEFACTS.update(params)
        print(f"[startup] Artefacts chargés depuis {DOSSIER_ARTEFACTS}/ — version : {params.get('version_modele')}")
    except FileNotFoundError as e:
        print(f"[startup] ⚠️  Artefacts introuvables ({e}). Lancez d'abord `python entrainer_modele.py`.")
        ARTEFACTS["modele"] = None

    supprimees = purger_journal(FICHIER_JOURNAL, RETENTION_JOURNAL_JOURS)
    print(f"[startup] Journal purgé : {supprimees} ligne(s) au-delà de {RETENTION_JOURNAL_JOURS} jours.")
    if not CLE_API:
        print("[startup] ⚠️  CLE_API non définie : l'API est ouverte. Acceptable en démonstration, "
              "JAMAIS en production (voir README.md, section Sécurité).")
    if ORIGINES_AUTORISEES == ["*"]:
        print("[startup] ⚠️  CORS ouvert à toutes les origines. En production, définir ORIGINES_AUTORISEES "
              "sur le seul domaine de l'application de guichet unique.")
    yield
    ARTEFACTS.clear()


app = FastAPI(
    title="API — Risque de retour à l'emploi",
    description="Prédiction du niveau de risque de chômage de longue durée à partir des données d'entretien.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS : nécessaire pour que l'interface web (interface.html) puisse appeler cette API.
# La liste est paramétrable par variable d'environnement : le code SAIT se restreindre,
# c'est la configuration de démonstration qui est permissive, pas l'application.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINES_AUTORISEES,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def compter_erreurs_de_validation(request: Request, exc: RequestValidationError):
    """FastAPI rejette une requête mal formée avant d'entrer dans la route : sans ce
    gestionnaire, les erreurs 422 n'apparaîtraient jamais dans /metrics, et le compteur
    « erreur_validation » resterait à zéro même en cas de problème côté appelant.
    On conserve exactement le corps de réponse par défaut, on ne fait que compter."""
    COMPTEURS["erreur_validation"] += 1
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


# --------------------------------------------------------------------------------------
# Authentification
# --------------------------------------------------------------------------------------
def verifier_cle(x_cle_api: str | None = Header(default=None)):
    """Jeton de service partagé entre l'application de guichet unique et l'API.

    Si `CLE_API` n'est pas définie (mode démonstration), l'API reste ouverte mais le
    démarrage l'annonce explicitement. En production, ce jeton doit être porté par un
    canal mTLS et non par un simple en-tête.
    """
    if CLE_API and x_cle_api != CLE_API:
        raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide (en-tête X-Cle-Api).")


def verifier_cle_admin(x_cle_admin: str | None = Header(default=None)):
    if not CLE_ADMIN:
        raise HTTPException(
            status_code=503,
            detail="Ré-entraînement désactivé : aucune CLE_ADMIN configurée sur ce déploiement.",
        )
    if x_cle_admin != CLE_ADMIN:
        raise HTTPException(status_code=401, detail="Clé d'administration absente ou invalide (en-tête X-Cle-Admin).")


# --------------------------------------------------------------------------------------
# Schémas de validation
# --------------------------------------------------------------------------------------
class UsagerEntree(BaseModel):
    """Données reçues par la route /predict.

    La variable nationalite_hors_ue est volontairement absente,
    conformément au Scénario 2 retenu.

    La date de début de poste permet à l'API de calculer elle-même
    l'ancienneté par rapport à la date de la requête.
    """

    age: float | None = Field(
        None,
        ge=16,
        le=100,
    )

    niveau_diplome: (
        Literal[
            "Sans diplôme",
            "Bac",
            "Bac+2",
            "Bac+5",
        ]
        | None
    ) = None

    date_debut_poste: date = Field(
        ...,
        description=(
            "Date de début du poste actuel au format AAAA-MM-JJ."
        ),
    )

    code_rome_vise: str = Field(
        ...,
        pattern=r"^[A-Z]\d{4}$",
        description=(
            "Code ROME composé d'une lettre majuscule "
            "suivie de quatre chiffres."
        ),
        examples=["M1607"],
    )

    code_insee_commune: str = Field(
        ...,
        min_length=5,
        max_length=5,
        pattern=r"^(?:\d{5}|2[AB]\d{3})$",
        description=(
            "Code INSEE de commune : cinq chiffres, "
            "ou code corse 2A/2B suivi de trois chiffres."
        ),
        examples=["07240", "2B033"],
    )

    est_allocataire: int | None = Field(
        None,
        ge=0,
        le=1,
    )

    synthese_entretien: str | None = ""

    session_id: str | None = None

    @field_validator("date_debut_poste")
    @classmethod
    def valider_date_debut_poste(
        cls,
        valeur: date,
    ) -> date:
        aujourdhui = datetime.now(timezone.utc).date()

        if valeur > aujourdhui:
            raise ValueError(
                "La date de début de poste ne peut pas être "
                "dans le futur."
            )

        anciennete_estimee = (
            aujourdhui - valeur
        ).days / 365.25

        if anciennete_estimee > 60:
            raise ValueError(
                f"Ancienneté calculée "
                f"({anciennete_estimee:.1f} ans) "
                "au-delà de la borne maximale acceptée "
                "(60 ans). Vérifiez la date saisie."
            )

        return valeur


class Feedback(BaseModel):
    session_id: str
    classe_reelle_observee: int = Field(..., ge=0, le=2)
    commentaire_conseiller: str | None = None


# --------------------------------------------------------------------------------------
# Prétraitement (reproduit exactement la logique de entrainer_modele.py)
# --------------------------------------------------------------------------------------
def pretraiter(u: UsagerEntree):
    anciennete_poste_ans = (datetime.now(timezone.utc).date() - u.date_debut_poste).days / 365.25

    age_fourni = u.age is not None
    age = u.age if age_fourni else ARTEFACTS["age_median"]
    age_manquant = int(not age_fourni)
    diplome = u.niveau_diplome if u.niveau_diplome is not None else ARTEFACTS["diplome_mode"]
    diplome_manquant = int(u.niveau_diplome is None)
    alloc = u.est_allocataire if u.est_allocataire is not None else ARTEFACTS["alloc_mode"]
    alloc_manquant = int(u.est_allocataire is None)
    texte = (u.synthese_entretien or "").lower().strip()
    texte_manquant = int(not texte)
    departement = u.code_insee_commune.zfill(5)[:2]
    diplome_encode = ARTEFACTS["mapping_diplome"][diplome]

    # Cohérence entraînement / service : à l'entraînement, un âge manquant donne
    # NaN < 14 == False, donc un drapeau à 0. On applique ici exactement la même règle,
    # AVANT de considérer l'âge imputé — sinon un usager sans âge pourrait recevoir un
    # signal que le modèle n'a jamais rencontré pendant son apprentissage.
    anciennete_incoherente = int(age_fourni and (u.age - anciennete_poste_ans) < 14)
    age_hors_plage = age_fourni and not (AGE_MIN_ENTRAINEMENT <= u.age <= AGE_MAX_ENTRAINEMENT)
    valeurs_num_bin = [
        age, anciennete_poste_ans, diplome_encode, alloc,
        age_manquant, diplome_manquant, alloc_manquant, texte_manquant, anciennete_incoherente,
    ]
    ohe = ARTEFACTS["ohe"]
    onehot = ohe.transform(
        pd.DataFrame([[departement, u.code_rome_vise]], columns=["departement", "code_rome_vise"])
    )
    # handle_unknown="ignore" : un département ou un code ROME jamais vu à l'entraînement
    # produit une ligne de zéros plutôt qu'une erreur. C'est le bon comportement (pas de
    # plantage), mais il est SILENCIEUX : la prédiction se fait alors sans aucune
    # information géographique ou métier, sans que le conseiller le sache. On le détecte.
    modalites_inconnues = []
    if departement not in ohe.categories_[0]:
        modalites_inconnues.append(f"département {departement}")
    if u.code_rome_vise not in ohe.categories_[1]:
        modalites_inconnues.append(f"code ROME {u.code_rome_vise}")

    vecteur_tabulaire = np.concatenate([valeurs_num_bin, onehot[0]]).reshape(1, -1)
    vecteur_texte = ARTEFACTS["tfidf"].transform([texte])
    X = hstack([csr_matrix(vecteur_tabulaire), vecteur_texte]).tocsr()
    return X, anciennete_poste_ans, bool(anciennete_incoherente), modalites_inconnues,age_hors_plage


def journaliser(session_id, entree, sortie, duree_ms):
    ligne = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "entree": entree,
        "sortie": sortie,
        "version_modele": ARTEFACTS.get("version_modele"),
        "duree_ms": round(duree_ms, 2),
    }
    with open(FICHIER_JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if ARTEFACTS.get("modele") is not None else "modele_non_charge",
        "modele_charge": ARTEFACTS.get("modele") is not None,
        "version_modele": ARTEFACTS.get("version_modele"),
        "seuil_classe2": ARTEFACTS.get("seuil_classe2"),
        "authentification_active": bool(CLE_API),
        "cors_restreint": ORIGINES_AUTORISEES != ["*"],
        "retention_journal_jours": RETENTION_JOURNAL_JOURS,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Compteurs au format d'exposition Prometheus.

    Le monitoring recommandé pour ce service (voir notebook, § 11.13) est
    volontairement proportionné : ces quelques compteurs suffisent à alimenter des
    alertes sur la disponibilité, le taux d'erreurs et la dérive de la distribution
    des classes prédites. Un Prometheus + Grafana peut les collecter tels quels sans
    modification du code, le jour où l'échelle le justifie.
    """
    lignes = [
        "# NOTE Ces compteurs sont propres au PROCESSUS qui répond. Avec un seul worker",
        "# Uvicorn, ils décrivent la totalité du service ; avec plusieurs workers, chaque",
        "# collecte ne voit qu'un worker. La montée en charge exigerait alors un magasin",
        "# partagé (mode multiprocess de prometheus_client, ou exporteur dédié). Limite",
        "# connue, exposée ici plutôt que découverte en production.",
        "# HELP predictions_total Nombre de prédictions servies, par classe prédite.",
        "# TYPE predictions_total counter",
    ]
    for classe in (0, 1, 2):
        lignes.append(f'predictions_total{{classe="{classe}"}} {COMPTEURS[f"pred_{classe}"]}')
    lignes += [
        "# HELP requetes_total Nombre de requêtes traitées, par issue.",
        "# TYPE requetes_total counter",
        f'requetes_total{{issue="succes"}} {COMPTEURS["succes"]}',
        f'requetes_total{{issue="erreur_validation"}} {COMPTEURS["erreur_validation"]}',
        f'requetes_total{{issue="erreur_interne"}} {COMPTEURS["erreur_interne"]}',
        "# HELP feedbacks_total Retours conseillers enregistrés.",
        "# TYPE feedbacks_total counter",
        f'feedbacks_total {COMPTEURS["feedback"]}',
        "# HELP latence_inference_ms_somme Somme cumulée des latences de traitement.",
        "# TYPE latence_inference_ms_somme counter",
        f'latence_inference_ms_somme {LATENCE_TOTALE_MS["somme"]:.2f}',
    ]
    return "\n".join(lignes) + "\n"


@app.post("/predict", dependencies=[Depends(verifier_cle)])
def predict(usager: UsagerEntree):
    if ARTEFACTS.get("modele") is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Lancez `python entrainer_modele.py` puis redémarrez l'API.")

    t0 = time.perf_counter()
    try:
        X, anciennete_calculee, anciennete_incoherente, modalites_inconnues, age_hors_plage = pretraiter(usager)
    except KeyError as e:
        COMPTEURS["erreur_validation"] += 1
        raise HTTPException(status_code=422, detail=f"Valeur inattendue pour un champ catégoriel : {e}")
    except Exception:  # noqa: BLE001 — barrière volontairement large : toute erreur de
        # prétraitement imprévue doit renvoyer une 400 propre au client, jamais un stack trace.
        COMPTEURS["erreur_validation"] += 1
        raise HTTPException(status_code=400, detail="Erreur de prétraitement des données envoyées.")

    try:
        probas = ARTEFACTS["modele"].predict_proba(X)[0]
    except Exception:  # noqa: BLE001 — même logique : ne jamais exposer une erreur interne
        COMPTEURS["erreur_interne"] += 1
        raise HTTPException(status_code=500, detail="Erreur interne du modèle lors de la prédiction.")

    classe_predite = 2 if probas[2] >= ARTEFACTS["seuil_classe2"] else int(np.argmax(probas[:2]))
    session_id = usager.session_id or str(uuid.uuid4())

    # On ne REJETTE jamais une incohérence âge/ancienneté (on ne sait pas laquelle des deux
    # valeurs est fautive) — mais on la remonte explicitement au conseiller, qui peut
    # vérifier la saisie avant de faire confiance à la prédiction.
    avertissements = []
    if anciennete_incoherente:
        avertissements.append(
            "Ancienneté incohérente avec l'âge déclaré (le début d'activité implicite se situe "
            "avant 14 ans). Vérifiez les valeurs saisies — la prédiction reste calculée, mais à "
            "interpréter avec prudence."
        )
    if modalites_inconnues:
        avertissements.append(
            "Modalité(s) absente(s) des données d'entraînement : "
            + ", ".join(modalites_inconnues)
            + ". La prédiction est calculée sans cet apport — à interpréter avec prudence."
        )
    if age_hors_plage:
        avertissements.append(
            f"Âge ({usager.age:.0f} ans) en dehors de la plage observée à l'entraînement "
            f"({AGE_MIN_ENTRAINEMENT}-{AGE_MAX_ENTRAINEMENT} ans) — la prédiction reste calculée, "
            "mais elle résulte d'une extrapolation à interpréter avec prudence."
    )
       
    sortie = {
        "classe_predite": classe_predite,
        "probabilites": {
            "0_rapide": round(float(probas[0]), 3),
            "1_moyen": round(float(probas[1]), 3),
            "2_longue_duree": round(float(probas[2]), 3),
        },
        "anciennete_calculee_ans": round(anciennete_calculee, 2),
        "avertissements": avertissements,
        "version_modele": ARTEFACTS.get("version_modele"),
    }
    duree_ms = (time.perf_counter() - t0) * 1000
    COMPTEURS["succes"] += 1
    COMPTEURS[f"pred_{classe_predite}"] += 1
    LATENCE_TOTALE_MS["somme"] += duree_ms
    # mode="json" : convertit proprement `date_debut_poste` (un objet date Python, non
    # sérialisable tel quel en JSON) en chaîne ISO 8601 avant d'écrire dans le journal.
    journaliser(session_id, usager.model_dump(mode="json"), sortie, duree_ms)
    return {**sortie, "session_id": session_id, "duree_ms": round(duree_ms, 2)}


@app.post("/retrain", dependencies=[Depends(verifier_cle)])
def retrain_feedback(feedback: Feedback):
    """Enregistre un retour conseiller. Ne déclenche AUCUN ré-entraînement immédiat :
    les retours sont accumulés puis examinés lors d'un ré-entraînement supervisé
    (voir /retrain/run et le notebook, § 11.9 et § 11.11)."""
    ligne = {"timestamp": datetime.now(timezone.utc).isoformat(), **feedback.model_dump()}
    with open(FICHIER_FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    COMPTEURS["feedback"] += 1
    return {"statut": "feedback enregistré pour ré-entraînement supervisé ultérieur", "session_id": feedback.session_id}


@app.post("/retrain/run", dependencies=[Depends(verifier_cle_admin)])
def retrain_run():
    """Ré-entraînement monitoré : entraîne un CHALLENGER qui intègre les retours conseillers.

    1. `entrainer_modele.py --avec-feedback` relie chaque retour (POST /retrain) à la
       prédiction journalisée de la même session, et ajoute ces lignes au SEUL jeu
       d'entraînement : le jeu de test de référence ne change pas ;
    2. le challenger est écrit dans un dossier distinct et tracé dans MLflow ;
    3. la route renvoie la comparaison avec le champion en production.

    La promotion reste une décision humaine (copie des artefacts + redémarrage).

    En production, ce travail appartient à un job d'entraînement dédié disposant d'un
    accès au référentiel de données — pas au conteneur d'inférence, qui n'embarque aucune
    donnée personnelle (d'où la réponse 503 dans l'image Docker).
    """
    if not os.path.exists(CHEMIN_DONNEES):
        raise HTTPException(
            status_code=503,
            detail=("Jeu de données absent de ce conteneur (comportement attendu : l'image "
                    "d'inférence n'embarque aucune donnée personnelle). Le ré-entraînement "
                    "doit être exécuté par le job dédié qui a accès au référentiel."),
        )
    dossier_candidat = Path(DOSSIER_CANDIDAT)
    if not dossier_candidat.is_absolute():
        dossier_candidat = DOSSIER_API / dossier_candidat
    # Chemins absolus : le script s'exécute depuis le dossier de l'API, qui n'est pas
    # forcément le dossier courant du processus (notebook, tests, uvicorn lancé ailleurs).
    environnement = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "CHEMIN_DONNEES": os.path.abspath(CHEMIN_DONNEES),
        "FICHIER_JOURNAL": os.path.abspath(FICHIER_JOURNAL),
        "FICHIER_FEEDBACK": os.path.abspath(FICHIER_FEEDBACK),
    }
    commande = [sys.executable, str(DOSSIER_API / "entrainer_modele.py"),
                "--sortie", str(dossier_candidat), "--avec-feedback"]
    try:
        subprocess.run(commande, check=True, capture_output=True, timeout=900,
                       cwd=DOSSIER_API, env=environnement)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Échec du ré-entraînement : {e.stderr.decode('utf-8', errors='replace')[-500:]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Ré-entraînement interrompu (délai dépassé).")

    with open(dossier_candidat / "metriques.json", encoding="utf-8") as f:
        candidat = json.load(f)
    suivi = {}
    if (dossier_candidat / "run_mlflow.json").exists():
        with open(dossier_candidat / "run_mlflow.json", encoding="utf-8") as f:
            suivi = json.load(f)
    champion = {}
    if os.path.exists(f"{DOSSIER_ARTEFACTS}/metriques.json"):
        with open(f"{DOSSIER_ARTEFACTS}/metriques.json", encoding="utf-8") as f:
            champion = json.load(f)

    meilleur = (
        bool(champion)
        and candidat["f1_macro_test"] >= champion["f1_macro_test"]
        and candidat["taux_erreur_critique_test"] <= champion["taux_erreur_critique_test"]
    )
    return {
        "statut": "challenger entraîné, non promu",
        "dossier_challenger": str(dossier_candidat),
        "retours_conseillers": suivi.get("retours_conseillers", {}),
        "run_id_mlflow": suivi.get("run_id"),
        "metriques_challenger": candidat,
        "metriques_champion": champion,
        "recommandation": "promotion envisageable" if meilleur else "conserver le champion",
        "rappel": "La promotion est une décision humaine : copier les artefacts puis redémarrer le service.",
    }
