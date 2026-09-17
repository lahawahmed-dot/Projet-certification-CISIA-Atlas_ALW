"""
tests/test_api.py
==================
Tests unitaires de l'API. Nécessite que le modèle ait été entraîné au préalable :
    python entrainer_modele.py
    pytest tests/ -v
"""
import importlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def journaux_isoles(tmp_path_factory):
    """Isole les écritures des tests dans un répertoire temporaire.

    Sans cela, chaque exécution de la suite ajoute des lignes de test au journal réel — or
    ce journal contient des données personnelles et sert de trace probatoire (README,
    section « Responsabilité juridique »). Des enregistrements de test n'ont rien à y faire.
    Les variables d'environnement sont posées AVANT l'import de `api`, car le module lit ses
    chemins au chargement : c'est la raison pour laquelle `api` est importé dans la fixture
    `client` et non en tête de fichier.
    """
    dossier = tmp_path_factory.mktemp("journaux")
    os.environ["FICHIER_JOURNAL"] = str(dossier / "journal_requetes.jsonl")
    os.environ["FICHIER_FEEDBACK"] = str(dossier / "feedback_reentrainement.jsonl")
    yield
    os.environ.pop("FICHIER_JOURNAL", None)
    os.environ.pop("FICHIER_FEEDBACK", None)


@pytest.fixture(scope="module")
def client(journaux_isoles):
    """Utilise TestClient comme context manager : c'est ce qui déclenche réellement
    les événements de lifespan (startup/shutdown), donc le chargement du modèle."""
    import api
    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def verifier_modele_charge(client):
    """S'assure que le modèle est bien chargé avant de lancer les tests — sinon les
    tests échoueraient tous avec un message peu clair. On préfère échouer vite et bien."""
    reponse = client.get("/health")
    if not reponse.json().get("modele_charge"):
        pytest.exit(
            "Le modèle n'est pas chargé. Lancez d'abord `python entrainer_modele.py` "
            "avant de lancer les tests.",
            returncode=1,
        )


def date_debut_il_y_a(annees: float) -> str:
    """Construit une date de début de poste 'il y a X années', par rapport à AUJOURD'HUI (UTC) —
    jamais une date écrite en dur, pour que les tests restent valides quelle que soit la
    date à laquelle on les exécute."""
    return (datetime.now(timezone.utc).date() - timedelta(days=round(annees * 365.25))).isoformat()


# --------------------------------------------------------------------------------------
# Santé, métriques, contrat de sortie
# --------------------------------------------------------------------------------------
def test_health_repond_200(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"
    assert reponse.json()["modele_charge"] is True
    assert reponse.json()["seuil_classe2"] == 0.20


def test_metrics_expose_le_format_prometheus(client):
    client.post("/predict", json={
        "date_debut_poste": date_debut_il_y_a(3), "code_rome_vise": "M1607", "code_insee_commune": "07240",
    })
    reponse = client.get("/metrics")
    assert reponse.status_code == 200
    assert "predictions_total{classe=\"0\"}" in reponse.text
    assert "requetes_total{issue=\"succes\"}" in reponse.text
    assert "latence_inference_ms_somme" in reponse.text


def test_predict_cas_complet(client):
    payload = {
        "age": 45,
        "niveau_diplome": "Bac",
        "date_debut_poste": date_debut_il_y_a(12),
        "code_rome_vise": "M1607",
        "code_insee_commune": "07240",
        "est_allocataire": 1,
        "synthese_entretien": "Freins périphériques majeurs.",
    }
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 200
    data = reponse.json()
    assert data["classe_predite"] in (0, 1, 2)
    assert set(data["probabilites"].keys()) == {"0_rapide", "1_moyen", "2_longue_duree"}
    assert abs(sum(data["probabilites"].values()) - 1.0) < 0.01
    assert "session_id" in data
    assert data["version_modele"] == "xgboost_scenario2_v2"


def test_predict_calcule_correctement_lanciennete(client):
    """Vérifie que l'ancienneté est bien calculée par rapport à AUJOURD'HUI, pas une date fixe."""
    reponse = client.post("/predict", json={
        "date_debut_poste": date_debut_il_y_a(5), "code_rome_vise": "M1607", "code_insee_commune": "07240",
    })
    assert reponse.status_code == 200
    assert 4.9 <= reponse.json()["anciennete_calculee_ans"] <= 5.1


# --------------------------------------------------------------------------------------
# Validation des données entrantes
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "payload_partiel",
        "description",
    ),
    [
        (
            {
                "date_debut_poste": (
                    datetime.now(timezone.utc).date()
                    + timedelta(days=30)
                ).isoformat(),
            },
            "date de début de poste future",
        ),
        (
            {
                "date_debut_poste": date_debut_il_y_a(65),
            },
            "ancienneté supérieure à 60 ans",
        ),
        (
            {
                "code_rome_vise": "XX999",
            },
            "code ROME mal formé",
        ),
        (
            {
                "code_insee_commune": "1234",
            },
            "code INSEE trop court",
        ),
        (
            {
                "code_insee_commune": "ABCDE",
            },
            "code INSEE de cinq caractères non conforme",
        ),
        (
            {
                "code_insee_commune": "2C033",
            },
            "préfixe corse invalide",
        ),
        (
            {
                "age": 150,
            },
            "âge supérieur à la borne acceptée",
        ),
    ],
)
def test_predict_rejette_les_entrees_invalides(
    client,
    payload_partiel,
    description,
):
    payload = {
        "date_debut_poste": date_debut_il_y_a(3),
        "code_rome_vise": "M1607",
        "code_insee_commune": "07240",
    }

    payload.update(payload_partiel)

    reponse = client.post(
        "/predict",
        json=payload,
    )

    assert reponse.status_code == 422, (
        f"Le cas invalide suivant n'a pas été rejeté : "
        f"{description}. "
        f"Code HTTP obtenu : {reponse.status_code}. "
        f"Réponse : {reponse.json()}"
    )

def test_predict_champs_optionnels_absents(client):
    """Les champs optionnels (age, niveau_diplome, est_allocataire) doivent être
    imputés automatiquement, sans faire planter la requête."""
    reponse = client.post("/predict", json={
        "date_debut_poste": date_debut_il_y_a(3), "code_rome_vise": "M1607", "code_insee_commune": "07240",
    })
    assert reponse.status_code == 200
    
def test_predict_refuse_code_insee_cinq_lettres(client):
    """Une chaîne de cinq lettres n'est pas un code INSEE valide."""

    charge = {
        "age": 45,
        "niveau_diplome": "Bac",
        "date_debut_poste": "2020-01-01",
        "code_rome_vise": "M1607",
        "code_insee_commune": "ABCDE",
        "est_allocataire": 1,
        "synthese_entretien": (
            "Compétences à réactualiser "
            "sur les outils numériques."
        ),
    }

    reponse = client.post(
        "/predict",
        json=charge,
    )

    assert reponse.status_code == 422, (
        "Un code INSEE composé de cinq lettres doit être rejeté.\n"
        f"Statut obtenu : {reponse.status_code}\n"
        f"Réponse : {reponse.text}"
    )

    corps = reponse.json()

    assert "detail" in corps

    detail_normalise = str(
        corps["detail"]
    ).lower()

    assert "code_insee_commune" in detail_normalise

def test_predict_accepte_code_insee_corse(client):
    """Cas limite identifié dans le notebook (étape 2) : un code commençant par 2A/2B
    (Corse) est un format valide, pas une anomalie."""
    reponse = client.post("/predict", json={
        "date_debut_poste": date_debut_il_y_a(3), "code_rome_vise": "M1607", "code_insee_commune": "2B033",
    })
    assert reponse.status_code == 200


# --------------------------------------------------------------------------------------
# Avertissements métier
# --------------------------------------------------------------------------------------
def test_predict_signale_incoherence_age_anciennete(client):
    """Âge 16 ans avec 3 ans d'ancienneté implique un début d'activité à 13 ans (< 14) —
    accepté (on ignore laquelle des deux valeurs est fausse) mais SIGNALÉ."""
    reponse = client.post("/predict", json={
        "age": 16, "date_debut_poste": date_debut_il_y_a(3),
        "code_rome_vise": "M1607", "code_insee_commune": "07240",
    })
    assert reponse.status_code == 200
    avertissements = reponse.json()["avertissements"]
    assert any("incohérente" in a.lower() for a in avertissements)

def test_predict_signale_age_hors_plage(client):
    """Un âge valide mais hors du domaine observé produit un avertissement."""

    charge = {
        "age": 16,
        "niveau_diplome": "Bac",
        "date_debut_poste": "2025-01-01",
        "code_rome_vise": "M1607",
        "code_insee_commune": "07240",
        "est_allocataire": 1,
        "synthese_entretien": (
            "Compétences à réactualiser "
            "sur les outils numériques."
        ),
    }

    reponse = client.post(
        "/predict",
        json=charge,
    )

    assert reponse.status_code == 200, (
        "Une donnée hors de la plage d'entraînement ne doit pas "
        "être rejetée si elle respecte le schéma de l'API.\n"
        f"Statut obtenu : {reponse.status_code}\n"
        f"Réponse : {reponse.text}"
    )

    corps = reponse.json()

    assert "avertissements" in corps, (
        "La réponse doit contenir le champ 'avertissements'.\n"
        f"Réponse obtenue : {corps}"
    )

    avertissements = corps["avertissements"]

    assert isinstance(avertissements, list), (
        "Le champ 'avertissements' doit être une liste.\n"
        f"Type obtenu : {type(avertissements)}"
    )

    assert avertissements, (
        "La liste des avertissements ne doit pas être vide "
        "pour un âge de 16 ans.\n"
        f"Réponse obtenue : {corps}"
    )

    avertissements_normalises = [
        str(avertissement).lower()
        for avertissement in avertissements
    ]

    avertissement_age_hors_plage = any(
        (
            "âge" in avertissement
            or "age" in avertissement
        )
        and "plage" in avertissement
        for avertissement in avertissements_normalises
    )

    assert avertissement_age_hors_plage, (
        "Aucun avertissement relatif à l'âge hors plage "
        "n'a été trouvé.\n"
        f"Avertissements reçus : {avertissements}"
    )

def test_predict_pas_dincoherence_quand_lage_est_absent(client):
    """Cohérence entraînement / service : à l'entraînement, un âge manquant donne un
    drapeau `anciennete_incoherente` à 0 (NaN < 14 vaut False). L'API doit faire pareil,
    et ne PAS déclencher l'incohérence à partir de l'âge médian imputé."""
    reponse = client.post("/predict", json={
        "date_debut_poste": date_debut_il_y_a(30),  # 30 ans d'ancienneté, âge non renseigné
        "code_rome_vise": "M1607", "code_insee_commune": "07240",
    })
    assert reponse.status_code == 200
    assert not any("incohérente" in a.lower() for a in reponse.json()["avertissements"])


def test_predict_pas_davertissement_si_coherent(client):
    reponse = client.post("/predict", json={
        "age": 40, "date_debut_poste": date_debut_il_y_a(5),
        "code_rome_vise": "M1607", "code_insee_commune": "07240",
    })
    assert reponse.status_code == 200
    assert reponse.json()["avertissements"] == []


def test_predict_signale_un_code_rome_inconnu(client):
    """handle_unknown='ignore' produit une ligne de zéros pour une modalité jamais vue :
    la prédiction reste possible, mais le conseiller doit en être averti."""
    reponse = client.post("/predict", json={
        "age": 40, "date_debut_poste": date_debut_il_y_a(5),
        "code_rome_vise": "Z9999", "code_insee_commune": "07240",
    })
    assert reponse.status_code == 200
    assert any("code ROME Z9999" in a for a in reponse.json()["avertissements"])


def test_predict_signale_un_departement_inconnu(client):
    """Cas réel découvert en ajoutant cette détection : le code INSEE 75115 (Paris 15e),
    utilisé jusque-là dans les exemples du projet, correspond au département 75 — qui
    n'apparaît dans AUCUNE des 2 500 lignes du jeu de données. L'API répondait
    normalement, sans jamais signaler que la géographie n'entrait pas dans le calcul."""
    reponse = client.post("/predict", json={
        "age": 40, "date_debut_poste": date_debut_il_y_a(5),
        "code_rome_vise": "M1607", "code_insee_commune": "75115",
    })
    assert reponse.status_code == 200
    assert any("département 75" in a for a in reponse.json()["avertissements"])


# --------------------------------------------------------------------------------------
# Garanties structurelles et conformité
# --------------------------------------------------------------------------------------
def test_predict_ne_contient_jamais_nationalite(client):
    """Garantie structurelle : le schéma d'entrée ne doit jamais accepter
    `nationalite_hors_ue`, conformément au Scénario 2 retenu."""
    champs = client.app.openapi()["components"]["schemas"]["UsagerEntree"]["properties"]
    assert "nationalite_hors_ue" not in champs


def test_purge_du_journal_supprime_les_lignes_perimees(tmp_path):
    """Durée de conservation (art. 5.1.e RGPD) : le journal ne doit pas s'accumuler
    indéfiniment. On vérifie que la purge est effective, pas seulement documentée."""
    import api
    chemin = tmp_path / "journal.jsonl"
    vieux = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    recent = datetime.now(timezone.utc).isoformat()
    chemin.write_text(
        json.dumps({"timestamp": vieux, "session_id": "a"}) + "\n"
        + json.dumps({"timestamp": recent, "session_id": "b"}) + "\n"
    )
    supprimees = api.purger_journal(str(chemin), 365)
    assert supprimees == 1
    assert "\"b\"" in chemin.read_text() and "\"a\"" not in chemin.read_text()


def test_retrain_enregistre_le_feedback(client):
    reponse = client.post("/retrain", json={
        "session_id": "test-pytest-001", "classe_reelle_observee": 2, "commentaire_conseiller": "Test automatisé",
    })
    assert reponse.status_code == 200
    assert reponse.json()["session_id"] == "test-pytest-001"


def test_retrain_rejette_classe_invalide(client):
    reponse = client.post("/retrain", json={"session_id": "test-002", "classe_reelle_observee": 5})
    assert reponse.status_code == 422


def test_retrain_run_refuse_sans_cle_admin(client):
    """Le ré-entraînement ne peut jamais être déclenché anonymement : sans CLE_ADMIN
    configurée, la route est désactivée (503) plutôt qu'ouverte."""
    assert client.post("/retrain/run").status_code == 503


def test_authentification_exigee_quand_la_cle_est_definie(monkeypatch):
    """Quand CLE_API est définie, /predict doit refuser une requête sans en-tête."""
    monkeypatch.setenv("CLE_API", "secret-de-test")
    import api
    importlib.reload(api)
    with TestClient(api.app) as c:
        payload = {
            "date_debut_poste": date_debut_il_y_a(3),
            "code_rome_vise": "M1607", "code_insee_commune": "07240",
        }
        assert c.post("/predict", json=payload).status_code == 401
        assert c.post("/predict", json=payload, headers={"X-Cle-Api": "secret-de-test"}).status_code == 200
    monkeypatch.delenv("CLE_API")
    importlib.reload(api)

def test_schema_insee_rejette_une_chaine_arbitraire():
    """Le schéma Pydantic doit refuser cinq caractères non conformes."""

    
    from pydantic import ValidationError
    
    import api

    try:
        api.UsagerEntree(
            date_debut_poste=date_debut_il_y_a(3),
            code_rome_vise="M1607",
            code_insee_commune="ABCDE",
        )
    except ValidationError:
        pass
    else:
        pytest.fail(
            "Le schéma UsagerEntree a accepté le code INSEE "
            "invalide ABCDE."
        )
