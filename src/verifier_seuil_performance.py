"""
verifier_seuil_performance.py
==============================
Garde-fou de qualité et d'équité (pattern « challenger / champion ») : refuse de laisser
passer un modèle dont les performances — ou l'équité entre groupes — tombent sous un
seuil minimal acceptable.

Sans ce script, `entrainer_modele.py` réussirait TOUJOURS (code de sortie 0), même si le
modèle obtenu est inutilisable : un mauvais hyperparamètre, un bug silencieux dans le
prétraitement, ou un ré-entraînement sur des retours de terrain biaisés passeraient
inaperçus et pourraient être déployés sans que personne ne s'en rende compte avant qu'un
usager en subisse les conséquences.

Le plancher de F1-macro par défaut (0,68) est volontairement le MÊME que celui utilisé
comme contrainte dans la règle de sélection du modèle (notebook, étape 6.4) : ce qui a
servi à choisir le modèle est ce qui sert à autoriser sa promotion.

Limite méthodologique assumée : les métriques vérifiées ici sont celles produites par
`entrainer_modele.py` sur le jeu de TEST. À chaque exécution du pipeline, ce jeu joue donc
de fait le rôle d'un jeu de validation. En production, ce contrôle doit porter sur un jeu
de validation dédié, distinct du jeu de test de référence conservé pour l'évaluation
finale — sans quoi le test perd progressivement son indépendance.

Usage :
    python verifier_seuil_performance.py --f1-macro-min 0.68 --erreur-critique-max 0.15

Code de sortie 0 si le modèle passe les seuils, 1 sinon (fait échouer le job CI/CD).
"""
import argparse
import json
import sys

FICHIER_METRIQUES = "artefacts_modele/metriques.json"


def main():
    parser = argparse.ArgumentParser(description="Vérifie qu'un modèle entraîné dépasse les seuils minimaux de qualité et d'équité.")
    parser.add_argument("--metriques", default=FICHIER_METRIQUES, help=f"Fichier de métriques (défaut : {FICHIER_METRIQUES})")
    parser.add_argument("--f1-macro-min", type=float, default=0.68, help="F1-macro minimal acceptable (défaut : 0.68)")
    parser.add_argument("--erreur-critique-max", type=float, default=0.15, help="Taux d'erreur critique maximal toléré (défaut : 0.15)")
    parser.add_argument("--ecart-rappel-max", type=float, default=0.15,
                        help="Écart maximal toléré sur le rappel de la classe 2 entre groupes de nationalité (défaut : 0.15)")
    args = parser.parse_args()

    try:
        with open(args.metriques) as f:
            metriques = json.load(f)
    except FileNotFoundError:
        print(f"❌ Fichier {args.metriques} introuvable — lancez d'abord `python entrainer_modele.py`.")
        sys.exit(1)

    f1_macro = metriques["f1_macro_test"]
    erreur_critique = metriques["taux_erreur_critique_test"]
    rappel_ue = metriques.get("rappel_classe2_ue")
    rappel_hors_ue = metriques.get("rappel_classe2_hors_ue")
    ecart_rappel = abs(rappel_ue - rappel_hors_ue) if None not in (rappel_ue, rappel_hors_ue) else None

    print(f"Seuils requis   : F1-macro ≥ {args.f1_macro_min}  |  erreur critique ≤ {args.erreur_critique_max}"
          f"  |  écart de rappel ≤ {args.ecart_rappel_max}")
    print(f"Modèle obtenu   : F1-macro = {f1_macro:.4f}  |  erreur critique = {erreur_critique:.4f}"
          + (f"  |  écart de rappel = {ecart_rappel:.4f}" if ecart_rappel is not None else "  |  écart de rappel = non mesuré"))

    echecs = []
    if f1_macro < args.f1_macro_min:
        echecs.append(f"F1-macro trop faible ({f1_macro:.4f} < {args.f1_macro_min})")
    if erreur_critique > args.erreur_critique_max:
        echecs.append(f"Erreur critique trop élevée ({erreur_critique:.4f} > {args.erreur_critique_max})")
    if ecart_rappel is not None and ecart_rappel > args.ecart_rappel_max:
        echecs.append(
            f"Écart de rappel entre groupes trop élevé ({ecart_rappel:.4f} > {args.ecart_rappel_max}) — "
            f"UE {rappel_ue:.3f} vs hors UE {rappel_hors_ue:.3f}"
        )

    if echecs:
        print("\n❌ MODÈLE REFUSÉ — ne sera pas promu en production :")
        for e in echecs:
            print(f"   - {e}")
        sys.exit(1)

    print("\n✅ Modèle accepté — dépasse les seuils minimaux de qualité et d'équité.")
    print("   Rappel : l'écart de rappel est mesuré sur 20 usagers de classe 2 hors UE dans le jeu de test.")
    print("   C'est un garde-fou, pas une preuve d'équité : l'audit doit être refait en continu en production.")
    sys.exit(0)


if __name__ == "__main__":
    main()
