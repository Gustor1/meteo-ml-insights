def build_quality_insight(quality_report: dict) -> str:
    """Produit un résumé factuel de la qualité des données."""
    periode = quality_report["periode"]
    total_missing = sum(quality_report["valeurs_manquantes"].values())
    total_incoherences = sum(quality_report["incoherences_physiques"].values())

    lines = [
        "## Qualité des données",
        f"Le fichier contient {quality_report['nb_lignes']} lignes et {quality_report['nb_colonnes']} colonnes.",
        f"La période analysée va du {periode['date_min'].date()} au {periode['date_max'].date()}.",
        f"La série contient {periode['dates_manquantes']} date(s) manquante(s), "
        f"{quality_report['doublons_date']} doublon(s) de date et "
        f"{periode['dates_invalides']} date(s) invalide(s).",
    ]
    if total_missing == 0:
        lines.append("Aucune valeur manquante n'a été détectée.")
    else:
        lines.append(f"{total_missing} valeur(s) manquante(s) ont été détectées dans l'ensemble des colonnes.")
    if total_incoherences == 0:
        lines.append("Aucune incohérence physique contrôlée n'a été détectée (températures, précipitations et humidité).")
    else:
        lines.append(
            f"{total_incoherences} incohérence(s) physique(s) ont été détectées. "
            "Les données source ne sont pas modifiées automatiquement."
        )
    return "\n\n".join(lines)


def build_model_comparison_insight(model_metrics: dict) -> str:
    """Compare les modèles à partir de leurs métriques de test."""
    if not model_metrics:
        return "## Comparaison des modèles\n\nAucun résultat de modèle disponible."

    best_model_name = min(model_metrics, key=lambda name: model_metrics[name]["mae"])
    best_metrics = model_metrics[best_model_name]
    lines = [
        "## Comparaison des modèles",
        f"Le modèle ayant la MAE la plus faible est : **{best_model_name}**.",
        f"Sur la période de test, sa MAE est de {best_metrics['mae']:.3f} °C, "
        f"son RMSE est de {best_metrics['rmse']:.3f} °C et son biais est de "
        f"{best_metrics['bias']:+.3f} °C.",
    ]

    baseline_name = "Baseline naïve"
    if baseline_name in model_metrics and best_model_name != baseline_name:
        baseline_mae = model_metrics[baseline_name]["mae"]
        mae_gain = baseline_mae - best_metrics["mae"]
        relative_gain = (mae_gain / baseline_mae) * 100
        lines.append(
            f"Par rapport à la baseline naïve, ce modèle réduit la MAE de "
            f"{mae_gain:.3f} °C ({relative_gain:.1f} %)."
        )

    if best_metrics["bias"] < -0.05:
        lines.append("Le biais négatif indique une légère tendance à sous-estimer la température moyenne du lendemain.")
    elif best_metrics["bias"] > 0.05:
        lines.append("Le biais positif indique une légère tendance à surestimer la température moyenne du lendemain.")
    else:
        lines.append("Le biais est faible : le modèle ne présente pas de surestimation ou de sous-estimation moyenne marquée.")

    lines.append("Cette comparaison ne démontre pas une relation causale entre les variables météo et la température prévue.")
    return "\n\n".join(lines)


def build_anomalies_insight(daily_extremes: dict, monthly_anomalies) -> str:
    """Présente les extrêmes et anomalies calculés avec des règles explicites."""
    hot_count = int(monthly_anomalies["anomalie_chaude"].sum())
    cold_count = int(monthly_anomalies["anomalie_froide"].sum())
    wet_count = int(monthly_anomalies["anomalie_humide"].sum())
    dry_count = int(monthly_anomalies["anomalie_seche"].sum())
    return "\n\n".join([
        "## Extrêmes et anomalies saisonnières",
        f"{daily_extremes['jours_chauds']} jour(s) chaud(s) ont été comptés avec le seuil tmax ≥ {daily_extremes['seuil_jour_chaud_c']:.1f} °C.",
        f"{daily_extremes['jours_de_gel']} jour(s) de gel ont été comptés avec le seuil tmin < {daily_extremes['seuil_gel_c']:.1f} °C.",
        f"{daily_extremes['jours_forte_pluie']} jour(s) de forte pluie ont été comptés avec le seuil précipitations ≥ {daily_extremes['seuil_forte_pluie_mm']:.1f} mm.",
        f"Avec un seuil de |z| ≥ 2,0, {hot_count} mois sont anormalement chauds, {cold_count} froids, {wet_count} humides et {dry_count} secs pour leur mois calendaire.",
        "Les z-scores comparent un mois uniquement aux mêmes mois des autres années. Ils décrivent un écart statistique dans cette série, sans attribuer de cause.",
    ])


def build_limitations_insight() -> str:
    """Retourne les limites méthodologiques applicables au rapport."""
    return "\n\n".join([
        "## Limites méthodologiques",
        "Les prévisions concernent la température moyenne du lendemain et dépendent des variables disponibles dans le fichier local.",
        "Les performances affichées décrivent une période de test unique ; elles devront être confirmées plus tard avec une validation temporelle glissante.",
        "Les tendances, anomalies et importances de variables sont descriptives ou prédictives : elles ne permettent pas d'attribuer une cause scientifique aux évolutions observées.",
    ])


def build_full_insights(
    quality_report: dict,
    trend_text: str,
    model_metrics: dict,
    daily_extremes: dict,
    monthly_anomalies,
) -> str:
    """Assemble les conclusions automatiques du rapport local."""
    return "\n\n".join([
        "# Synthèse automatique météo",
        build_quality_insight(quality_report),
        "## Tendance climatique\n\n" + trend_text,
        build_anomalies_insight(daily_extremes, monthly_anomalies),
        build_model_comparison_insight(model_metrics),
        build_limitations_insight(),
    ])
