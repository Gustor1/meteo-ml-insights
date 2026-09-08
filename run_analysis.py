import argparse
from pathlib import Path

import pandas as pd

from src.anomalies import (
    add_seasonal_zscores,
    build_annual_statistics,
    build_monthly_statistics,
    detect_monthly_anomalies,
    get_daily_extremes,
    get_top_periods,
)
from src.climate_indicators import (
    build_annual_climate_indicators,
    build_indicator_trends,
    build_seasonal_temperature_statistics,
    build_seasonal_temperature_trends,
    filter_complete_years,
)
from src.evaluation import (
    calculate_regression_metrics,
    chronological_train_test_split,
    compute_permutation_importance,
    expanding_window_splits,
    summarize_walk_forward_results,
)
from src.features import build_j1_temperature_dataset, compute_temperature_trend, describe_trend
from src.forecasting import fit_linear_regression, fit_random_forest, naive_persistence_forecast
from src.insights import build_full_insights
from src.loader import EXCLUDED_VARIABLES, detect_column_mapping, get_usable_mapping, load_weather_csv
from src.monthly_forecasting import run_monthly_holt_winters_analysis
from src.reporting import (
    plot_annual_climate_summary,
    plot_monthly_climate_summary,
    plot_test_predictions,
    save_markdown_report,
)
from src.validator import build_quality_report

FEATURE_COLUMNS = [
    "tmin", "tmax", "tmean", "precipitation_mm", "vent_ms", "humidite_min",
    "humidite_max", "insolation_min", "rayonnement_pm", "rayonnement_total",
    "jour_annee_sin", "jour_annee_cos", "tmean_lag_1", "tmean_lag_7", "tmean_lag_30",
]

INDICATOR_LABELS = {
    "temperature_moyenne_annuelle": ("Température moyenne annuelle", "°C/décennie"),
    "jours_tres_chauds_30": ("Jours très chauds (tmax ≥ 30 °C)", "jours/décennie"),
    "jours_canicule_35": ("Jours de canicule (tmax ≥ 35 °C)", "jours/décennie"),
    "nuits_tropicales_20": ("Nuits tropicales (tmin ≥ 20 °C)", "jours/décennie"),
    "jours_de_gel": ("Jours de gel (tmin < 0 °C)", "jours/décennie"),
    "jours_sans_degel": ("Jours sans dégel (tmax < 0 °C)", "jours/décennie"),
    "precipitation_totale_annuelle": ("Précipitations annuelles", "mm/décennie"),
    "jours_pluie_10mm": ("Jours avec au moins 10 mm", "jours/décennie"),
    "jours_forte_pluie_20mm": ("Jours avec au moins 20 mm", "jours/décennie"),
    "precipitation_maximale_journaliere": ("Précipitation maximale journalière", "mm/décennie"),
}


def markdown_table(dataframe: pd.DataFrame, columns: list[str], formatters: dict[str, str] | None = None) -> str:
    formatters = formatters or {}
    lines = [
        "| " + " | ".join(column.replace("_", " ") for column in columns) + " |",
        "|" + "|".join(["---:"] * len(columns)) + "|",
    ]
    for _, row in dataframe[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in {"date", "test_debut", "test_fin"}:
                values.append(pd.Timestamp(value).strftime("%Y-%m" if column == "date" else "%Y-%m-%d"))
            elif column in formatters:
                values.append(formatters[column].format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_mapping_section(mapping: dict) -> str:
    lines = ["## Mapping des colonnes", "", "| Variable interne | Colonne détectée |", "|---|---|"]
    for variable, column in mapping.items():
        lines.append(f"| {variable} | {column if column is not None else 'Non trouvée'} |")
    return "\n".join(lines)


def format_excluded_variables_section() -> str:
    if not EXCLUDED_VARIABLES:
        return "## Variables exclues du ML\n\nAucune variable n'est exclue du pipeline ML."
    lines = ["## Variables exclues du ML", ""]
    lines.extend(f"- `{name}` : {reason}" for name, reason in EXCLUDED_VARIABLES.items())
    return "\n".join(lines)


def format_metrics_section(model_metrics: dict) -> str:
    lines = ["## Résultats des modèles", "", "| Modèle | MAE (°C) | RMSE (°C) | Biais (°C) |", "|---|---:|---:|---:|"]
    for name, metrics in model_metrics.items():
        lines.append(f"| {name} | {metrics['mae']:.3f} | {metrics['rmse']:.3f} | {metrics['bias']:+.3f} |")
    return "\n".join(lines)


def format_importance_section(importance_df: pd.DataFrame) -> str:
    lines = [
        "## Importance des variables", "",
        "L'augmentation de MAE indique la dégradation de la prédiction lorsque la variable est mélangée aléatoirement.", "",
        "| Variable | Augmentation moyenne de MAE (°C) | Écart-type |", "|---|---:|---:|",
    ]
    for _, row in importance_df.iterrows():
        lines.append(f"| {row['variable']} | {row['augmentation_mae_moyenne']:+.3f} | {row['ecart_type']:.3f} |")
    lines.extend(["", "Ces importances mesurent une utilité prédictive dans ce modèle. Elles ne démontrent pas une relation de causalité."])
    return "\n".join(lines)


def format_climate_section(daily_extremes, hottest_months, coldest_months, wettest_months, hottest_years, monthly_anomalies) -> str:
    return "\n".join([
        "## Climat et événements remarquables", "", "### Jours extrêmes", "",
        f"- Jours chauds (tmax ≥ {daily_extremes['seuil_jour_chaud_c']:.1f} °C) : {daily_extremes['jours_chauds']}",
        f"- Jours de gel (tmin < {daily_extremes['seuil_gel_c']:.1f} °C) : {daily_extremes['jours_de_gel']}",
        f"- Jours de forte pluie (précipitations ≥ {daily_extremes['seuil_forte_pluie_mm']:.1f} mm) : {daily_extremes['jours_forte_pluie']}",
        "", "### Cinq mois les plus chauds", "",
        markdown_table(hottest_months, ["date", "temperature_moyenne_mensuelle", "zscore_temperature_saisonnier"], {"temperature_moyenne_mensuelle": "{:.2f}", "zscore_temperature_saisonnier": "{:+.2f}"}),
        "", "### Cinq mois les plus froids", "",
        markdown_table(coldest_months, ["date", "temperature_moyenne_mensuelle", "zscore_temperature_saisonnier"], {"temperature_moyenne_mensuelle": "{:.2f}", "zscore_temperature_saisonnier": "{:+.2f}"}),
        "", "### Cinq mois les plus humides", "",
        markdown_table(wettest_months, ["date", "precipitation_totale_mensuelle", "zscore_precipitation_saisonnier"], {"precipitation_totale_mensuelle": "{:.1f}", "zscore_precipitation_saisonnier": "{:+.2f}"}),
        "", "### Cinq années les plus chaudes", "",
        markdown_table(hottest_years, ["annee", "temperature_moyenne_annuelle", "precipitation_totale_annuelle"], {"temperature_moyenne_annuelle": "{:.2f}", "precipitation_totale_annuelle": "{:.1f}"}),
        "", "### Anomalies saisonnières", "",
        f"{len(monthly_anomalies)} mois présentent au moins une anomalie saisonnière avec un seuil de |z| ≥ 2,0.",
        "Un z-score compare chaque mois aux mêmes mois calendaires des autres années.",
    ])


def format_climate_trends_section(annual_coverage, indicator_trends, seasonal_trends) -> str:
    lines = [
        "## Évolution climatique annuelle et saisonnière", "",
        "Les tendances annuelles utilisent uniquement les années dont la couverture atteint au moins 99 %.", "",
        "### Couverture des années", "",
        f"Années complètes retenues : {(annual_coverage['taux_couverture'] >= 0.99).sum()} sur {len(annual_coverage)}.",
        "", "### Tendances des indicateurs annuels", "",
        "| Indicateur | Tendance par décennie | p-value | R² | Significative |", "|---|---:|---:|---:|---|",
    ]
    for _, row in indicator_trends.iterrows():
        label, unit = INDICATOR_LABELS.get(row["indicateur"], (row["indicateur"], "par décennie"))
        lines.append(f"| {label} | {row['pente_par_decennie']:+.2f} {unit} | {row['p_value']:.4g} | {row['r_squared']:.3f} | {'Oui' if row['significatif'] else 'Non'} |")
    lines.extend(["", "### Tendances de température par saison", "", "| Saison | Tendance (°C/décennie) | p-value | R² | Significative |", "|---|---:|---:|---:|---|"])
    for _, row in seasonal_trends.iterrows():
        lines.append(f"| {row['saison']} | {row['pente_par_decennie']:+.2f} | {row['p_value']:.4g} | {row['r_squared']:.3f} | {'Oui' if row['significatif'] else 'Non'} |")
    lines.extend(["", "Les tendances décrivent la série analysée et ne permettent pas d'attribuer une cause scientifique aux évolutions observées."])
    return "\n".join(lines)


def format_walk_forward_section(fold_results: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "## Validation temporelle glissante", "",
        "Cinq fenêtres de test successives d'environ 365 jours ont été utilisées. Chaque modèle est entraîné uniquement sur les observations antérieures à la fenêtre testée.", "",
        "### Résumé par modèle", "",
        markdown_table(summary, ["modele", "mae_moyenne", "mae_ecart_type", "rmse_moyen", "biais_moyen", "nombre_fenetres"], {"mae_moyenne": "{:.3f}", "mae_ecart_type": "{:.3f}", "rmse_moyen": "{:.3f}", "biais_moyen": "{:+.3f}"}),
        "", "### Résultats par fenêtre", "",
        markdown_table(fold_results, ["fenetre", "test_debut", "test_fin", "modele", "mae", "rmse", "bias"], {"mae": "{:.3f}", "rmse": "{:.3f}", "bias": "{:+.3f}"}),
    ]
    return "\n".join(lines)


def format_monthly_forecasting_section(monthly_forecasting: dict) -> str:
    metrics = monthly_forecasting["comparison_metrics"]
    forecast = monthly_forecasting["future_forecast"]
    lines = [
        "## Prévision mensuelle Holt-Winters", "",
        "La prévision mensuelle est distincte de la prévision quotidienne à J+1. "
        "Elle projette les températures moyennes mensuelles à partir de la tendance et de la saisonnalité observées.", "",
        f"Période de test mensuelle : {monthly_forecasting['test_series'].index.min().strftime('%Y-%m')} à {monthly_forecasting['test_series'].index.max().strftime('%Y-%m')}.",
        "", "### Comparaison des modèles mensuels", "",
        "| Modèle | MAE (°C) | RMSE (°C) | Biais (°C) |", "|---|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(f"| {name} | {values['mae']:.3f} | {values['rmse']:.3f} | {values['bias']:+.3f} |")
    lines.extend(["", f"### Projection des {monthly_forecasting['forecast_months']} prochains mois", "", "| Mois | Température moyenne projetée (°C) |", "|---|---:|"])
    for date, value in forecast.items():
        lines.append(f"| {pd.Timestamp(date).strftime('%Y-%m')} | {value:.2f} |")
    lines.extend(["", "Ces valeurs sont des projections statistiques mensuelles. Elles ne sont pas des prévisions météo quotidiennes et ne constituent pas une certitude."])
    return "\n".join(lines)


def run_walk_forward_validation(model_data: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold_number, train_data, test_data in expanding_window_splits(model_data, n_splits=5, test_window_size=365, min_train_size=3650):
        y_test = test_data["target_tmean_j1"]
        predictions = {"Baseline naïve": naive_persistence_forecast(test_data)}
        _, predictions["Régression linéaire"] = fit_linear_regression(train_data=train_data, test_data=test_data, feature_columns=feature_columns)
        _, predictions["Random Forest"] = fit_random_forest(train_data=train_data, test_data=test_data, feature_columns=feature_columns)
        for model_name, values in predictions.items():
            rows.append({
                "fenetre": fold_number,
                "train_debut": train_data["date"].min(), "train_fin": train_data["date"].max(),
                "test_debut": test_data["date"].min(), "test_fin": test_data["date"].max(),
                "modele": model_name,
                **calculate_regression_metrics(y_test, values),
            })
    results = pd.DataFrame(rows)
    return results, summarize_walk_forward_results(results)


def run_full_analysis(filepath: str, test_size: float = 0.20, create_plots: bool = True, run_walk_forward: bool = True) -> dict:
    source_path = Path(filepath)
    print("\n=== 1. CHARGEMENT DU FICHIER ===")
    df = load_weather_csv(source_path)
    print(f"Fichier : {source_path.name}\nLignes : {len(df)}\nColonnes : {len(df.columns)}")

    print("\n=== 2. DÉTECTION DES COLONNES ===")
    mapping = detect_column_mapping(df)
    usable_mapping = get_usable_mapping(mapping)
    missing_required = [name for name in ["date", "tmean", "tmin", "tmax"] if usable_mapping.get(name) is None]
    if missing_required:
        raise ValueError(f"Colonnes obligatoires non détectées : {missing_required}.")
    for variable, column in mapping.items():
        print(f"{variable:22s} -> {column if column is not None else 'NON TROUVÉ'}")

    print("\n=== 3. RAPPORT DE QUALITÉ ===")
    quality_report = build_quality_report(df, mapping)
    print(f"Période : {quality_report['periode']['date_min'].date()} -> {quality_report['periode']['date_max'].date()}")
    print(f"Dates manquantes : {quality_report['periode']['dates_manquantes']}\nDoublons de date : {quality_report['doublons_date']}")

    print("\n=== 4. TENDANCE CLIMATIQUE ===")
    trend = compute_temperature_trend(df)
    trend_text = describe_trend(trend)
    print(trend_text)

    print("\n=== 5. EXTRÊMES ET ANOMALIES SAISONNIÈRES ===")
    monthly_statistics = add_seasonal_zscores(build_monthly_statistics(df))
    complete_daily_data, annual_coverage = filter_complete_years(df)
    annual_statistics = build_annual_statistics(complete_daily_data)
    monthly_anomalies = detect_monthly_anomalies(monthly_statistics)
    daily_extremes = get_daily_extremes(df)
    hottest_months = get_top_periods(monthly_statistics, "temperature_moyenne_mensuelle")
    coldest_months = get_top_periods(monthly_statistics, "temperature_moyenne_mensuelle", ascending=True)
    wettest_months = get_top_periods(monthly_statistics, "precipitation_totale_mensuelle")
    hottest_years = get_top_periods(annual_statistics, "temperature_moyenne_annuelle")
    print(f"Jours chauds (tmax >= {daily_extremes['seuil_jour_chaud_c']:.1f} °C) : {daily_extremes['jours_chauds']}")
    print(f"Jours de gel (tmin < {daily_extremes['seuil_gel_c']:.1f} °C) : {daily_extremes['jours_de_gel']}")
    print(f"Jours de forte pluie (précipitations >= {daily_extremes['seuil_forte_pluie_mm']:.1f} mm) : {daily_extremes['jours_forte_pluie']}")
    print(f"Mois avec au moins une anomalie saisonnière (|z| >= 2,0) : {len(monthly_anomalies)}")

    print("\n=== 6. INDICATEURS CLIMATIQUES ANNUELS ET SAISONNIERS ===")
    annual_indicators = build_annual_climate_indicators(complete_daily_data)
    indicator_trends = build_indicator_trends(annual_indicators)
    seasonal_statistics = build_seasonal_temperature_statistics(df)
    seasonal_trends = build_seasonal_temperature_trends(seasonal_statistics)
    complete_count = int((annual_coverage["taux_couverture"] >= 0.99).sum())
    print(f"Années complètes retenues : {complete_count} sur {len(annual_coverage)}")
    print("Tendances saisonnières (°C par décennie) :")
    print(seasonal_trends[["saison", "pente_par_decennie", "p_value", "r_squared", "significatif"]].to_string(index=False))

    print("\n=== 7. PRÉVISION MENSUELLE HOLT-WINTERS ===")
    monthly_forecasting = run_monthly_holt_winters_analysis(
        df,
        test_months=36,
        forecast_months=12,
    )
    for model_name, metrics in monthly_forecasting["comparison_metrics"].items():
        print(
            f"{model_name:36s} | MAE : {metrics['mae']:.3f} °C | "
            f"RMSE : {metrics['rmse']:.3f} °C | Biais : {metrics['bias']:+.3f} °C"
        )
    print(
        "Projection mensuelle : "
        f"{monthly_forecasting['future_forecast'].index.min().strftime('%Y-%m')} -> "
        f"{monthly_forecasting['future_forecast'].index.max().strftime('%Y-%m')}"
    )

    print("\n=== 8. PRÉPARATION DES DONNÉES J+1 ===")
    model_data = build_j1_temperature_dataset(df)
    available_features = [column for column in FEATURE_COLUMNS if column in model_data.columns]
    missing_features = [column for column in FEATURE_COLUMNS if column not in model_data.columns]
    if missing_features:
        raise ValueError(f"Certaines features nécessaires sont absentes : {missing_features}")
    print(f"Lignes utilisables : {len(model_data)}\nNombre de features : {len(available_features)}")

    print("\n=== 9. DÉCOUPAGE CHRONOLOGIQUE ===")
    train_data, test_data = chronological_train_test_split(model_data, test_size=test_size)
    print(f"Train : {len(train_data)} lignes | {train_data['date'].min().date()} -> {train_data['date'].max().date()}")
    print(f"Test : {len(test_data)} lignes | {test_data['date'].min().date()} -> {test_data['date'].max().date()}")

    print("\n=== 10. ENTRAÎNEMENT ET COMPARAISON ===")
    y_test = test_data["target_tmean_j1"]
    naive_predictions = naive_persistence_forecast(test_data)
    forest_model, forest_predictions = fit_random_forest(train_data=train_data, test_data=test_data, feature_columns=available_features)
    _, linear_predictions = fit_linear_regression(train_data=train_data, test_data=test_data, feature_columns=available_features)
    model_metrics = {
        "Baseline naïve": calculate_regression_metrics(y_test, naive_predictions),
        "Régression linéaire": calculate_regression_metrics(y_test, linear_predictions),
        "Random Forest": calculate_regression_metrics(y_test, forest_predictions),
    }
    for name, metrics in model_metrics.items():
        print(f"{name:22s} | MAE : {metrics['mae']:.3f} °C | RMSE : {metrics['rmse']:.3f} °C | Biais : {metrics['bias']:+.3f} °C")

    print("\n=== 11. IMPORTANCE PAR PERMUTATION ===")
    importance_df = compute_permutation_importance(model=forest_model, test_data=test_data, feature_columns=available_features, n_repeats=20)
    print(importance_df.to_string(index=False))

    walk_forward_results, walk_forward_summary = pd.DataFrame(), pd.DataFrame()
    if run_walk_forward:
        print("\n=== 12. VALIDATION TEMPORELLE GLISSANTE ===")
        print("Cinq fenêtres d'environ un an : le calcul peut prendre plus de temps que l'analyse standard.")
        walk_forward_results, walk_forward_summary = run_walk_forward_validation(model_data, available_features)
        print(walk_forward_summary.to_string(index=False))

    if create_plots:
        print("\n=== 13. GRAPHIQUES ===")
        plot_annual_climate_summary(annual_statistics, "outputs/climat_annuel.png")
        plot_monthly_climate_summary(monthly_statistics, "outputs/climatologie_mensuelle.png")
        for prediction, name, output in [
            (naive_predictions, "Baseline naïve", "outputs/predictions_baseline_naive.png"),
            (linear_predictions, "Régression linéaire", "outputs/predictions_regression_lineaire.png"),
            (forest_predictions, "Random Forest", "outputs/predictions_random_forest.png"),
        ]:
            plot_test_predictions(test_data=test_data, predictions=prediction, model_name=name, output_path=output)

    print("\n=== 14. RAPPORT AUTOMATIQUE ===")
    insights = build_full_insights(quality_report, trend_text, model_metrics, daily_extremes, monthly_anomalies)
    sections = [
        "# Rapport d'analyse météo", "", f"**Fichier analysé :** `{source_path.name}`", "",
        "Ce rapport a été généré entièrement en local. Aucun fichier météo brut n'a été transmis à un service externe.", "",
        format_mapping_section(mapping), "", format_excluded_variables_section(), "", insights, "",
        format_climate_section(daily_extremes, hottest_months, coldest_months, wettest_months, hottest_years, monthly_anomalies), "",
        format_climate_trends_section(annual_coverage, indicator_trends, seasonal_trends), "",
        format_monthly_forecasting_section(monthly_forecasting), "",
        "## Graphiques climatiques générés", "", "- `outputs/climat_annuel.png`", "- `outputs/climatologie_mensuelle.png`", "",
        format_metrics_section(model_metrics), "", format_importance_section(importance_df),
    ]
    if run_walk_forward:
        sections.extend(["", format_walk_forward_section(walk_forward_results, walk_forward_summary)])
    save_markdown_report("\n".join(sections), "outputs/rapport_meteo.md")

    print("\n=== ANALYSE TERMINÉE ===")
    print("Consulte le dossier outputs/ pour les graphiques et le rapport Markdown.")
    return {
        "source_filename": source_path.name, "quality_report": quality_report, "trend": trend, "trend_text": trend_text,
        "mapping": mapping, "usable_mapping": usable_mapping, "daily_extremes": daily_extremes,
        "monthly_statistics": monthly_statistics, "annual_statistics": annual_statistics, "monthly_anomalies": monthly_anomalies,
        "hottest_months": hottest_months, "coldest_months": coldest_months, "wettest_months": wettest_months, "hottest_years": hottest_years,
        "annual_coverage": annual_coverage, "annual_indicators": annual_indicators, "indicator_trends": indicator_trends,
        "seasonal_statistics": seasonal_statistics, "seasonal_trends": seasonal_trends,
        "monthly_forecasting": monthly_forecasting,
        "model_metrics": model_metrics, "importance_df": importance_df,
        "walk_forward_results": walk_forward_results, "walk_forward_summary": walk_forward_summary,
        "train_start": train_data["date"].min(), "train_end": train_data["date"].max(),
        "test_start": test_data["date"].min(), "test_end": test_data["date"].max(),
        "train_size": len(train_data), "test_size": len(test_data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse locale d'un fichier météo CSV.")
    parser.add_argument("filepath", help="Chemin vers le fichier CSV météo local à analyser.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Part récente réservée au test.")
    parser.add_argument("--no-plots", action="store_true", help="Ne génère pas les graphiques PNG.")
    parser.add_argument("--skip-walk-forward", action="store_true", help="Ignore la validation temporelle glissante.")
    args = parser.parse_args()
    run_full_analysis(args.filepath, args.test_size, not args.no_plots, not args.skip_walk_forward)


if __name__ == "__main__":
    main()
