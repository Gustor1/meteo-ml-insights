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
from src.evaluation import (
    calculate_regression_metrics,
    chronological_train_test_split,
    compute_permutation_importance,
    expanding_window_splits,
    summarize_walk_forward_results,
)
from src.features import (
    build_j1_temperature_dataset,
    compute_temperature_trend,
    describe_trend,
)
from src.forecasting import (
    fit_linear_regression,
    fit_random_forest,
    naive_persistence_forecast,
)
from src.insights import build_full_insights
from src.loader import (
    EXCLUDED_VARIABLES,
    detect_column_mapping,
    get_usable_mapping,
    load_weather_csv,
)
from src.reporting import (
    plot_annual_climate_summary,
    plot_monthly_climate_summary,
    plot_test_predictions,
    save_markdown_report,
)
from src.validator import build_quality_report

FEATURE_COLUMNS = [
    "tmin", "tmax", "tmean", "precipitation_mm", "vent_ms",
    "humidite_min", "humidite_max", "insolation_min", "rayonnement_pm",
    "rayonnement_total", "jour_annee_sin", "jour_annee_cos",
    "tmean_lag_1", "tmean_lag_7", "tmean_lag_30",
]


def format_mapping_section(mapping: dict) -> str:
    lines = ["## Mapping des colonnes", "", "| Variable interne | Colonne détectée |", "|---|---|"]
    for variable, column in mapping.items():
        lines.append(f"| {variable} | {column if column is not None else 'Non trouvée'} |")
    return "\n".join(lines)


def format_excluded_variables_section() -> str:
    lines = ["## Variables exclues du ML", ""]
    if not EXCLUDED_VARIABLES:
        return "\n".join(lines + ["Aucune variable n'est exclue du pipeline ML."])
    return "\n".join(lines + [f"- `{name}` : {reason}" for name, reason in EXCLUDED_VARIABLES.items()])


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


def format_periods_table(dataframe: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    lines = ["| " + " | ".join(column.replace("_", " ") for column in columns) + " |", "|" + "|".join(["---:"] * len(columns)) + "|"]
    for _, row in dataframe[columns].iterrows():
        values = []
        for column in columns:
            if column == "date":
                values.append(row[column].strftime("%Y-%m"))
            elif column in formats:
                values.append(formats[column].format(row[column]))
            else:
                values.append(str(row[column]))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_climate_section(daily_extremes, hottest_months, coldest_months, wettest_months, hottest_years, monthly_anomalies) -> str:
    lines = [
        "## Climat et événements remarquables", "", "### Jours extrêmes", "",
        f"- Jours chauds (tmax ≥ {daily_extremes['seuil_jour_chaud_c']:.1f} °C) : {daily_extremes['jours_chauds']}",
        f"- Jours de gel (tmin < {daily_extremes['seuil_gel_c']:.1f} °C) : {daily_extremes['jours_de_gel']}",
        f"- Jours de forte pluie (précipitations ≥ {daily_extremes['seuil_forte_pluie_mm']:.1f} mm) : {daily_extremes['jours_forte_pluie']}",
        "", "### Cinq mois les plus chauds", "",
        format_periods_table(hottest_months, ["date", "temperature_moyenne_mensuelle", "zscore_temperature_saisonnier", "percentile_temperature_mois"], {"temperature_moyenne_mensuelle": "{:.2f}", "zscore_temperature_saisonnier": "{:+.2f}", "percentile_temperature_mois": "{:.3f}"}),
        "", "### Cinq mois les plus froids", "",
        format_periods_table(coldest_months, ["date", "temperature_moyenne_mensuelle", "zscore_temperature_saisonnier", "percentile_temperature_mois"], {"temperature_moyenne_mensuelle": "{:.2f}", "zscore_temperature_saisonnier": "{:+.2f}", "percentile_temperature_mois": "{:.3f}"}),
        "", "### Cinq mois les plus humides", "",
        format_periods_table(wettest_months, ["date", "precipitation_totale_mensuelle", "zscore_precipitation_saisonnier", "percentile_precipitation_mois"], {"precipitation_totale_mensuelle": "{:.1f}", "zscore_precipitation_saisonnier": "{:+.2f}", "percentile_precipitation_mois": "{:.3f}"}),
        "", "### Cinq années les plus chaudes", "",
        format_periods_table(hottest_years, ["annee", "temperature_moyenne_annuelle", "precipitation_totale_annuelle"], {"temperature_moyenne_annuelle": "{:.2f}", "precipitation_totale_annuelle": "{:.1f}"}),
        "", "### Anomalies saisonnières", "",
        f"{len(monthly_anomalies)} mois présentent au moins une anomalie saisonnière avec un seuil de |z| ≥ 2,0.",
        "Un z-score compare chaque mois aux mêmes mois calendaires des autres années.",
    ]
    return "\n".join(lines)


def run_walk_forward_validation(model_data: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare les modèles sur cinq périodes futures distinctes, sans fuite temporelle."""
    rows = []
    folds = expanding_window_splits(model_data, n_splits=5, test_window_size=365, min_train_size=3650)
    for fold_number, train_data, test_data in folds:
        y_test = test_data["target_tmean_j1"]
        predictions_by_model = {
            "Baseline naïve": naive_persistence_forecast(test_data),
        }
        _, predictions_by_model["Régression linéaire"] = fit_linear_regression(
            train_data=train_data, test_data=test_data, feature_columns=feature_columns
        )
        _, predictions_by_model["Random Forest"] = fit_random_forest(
            train_data=train_data, test_data=test_data, feature_columns=feature_columns
        )
        for model_name, predictions in predictions_by_model.items():
            metrics = calculate_regression_metrics(y_test, predictions)
            rows.append({
                "fenetre": fold_number,
                "train_debut": train_data["date"].min(),
                "train_fin": train_data["date"].max(),
                "test_debut": test_data["date"].min(),
                "test_fin": test_data["date"].max(),
                "modele": model_name,
                **metrics,
            })
    fold_results = pd.DataFrame(rows)
    return fold_results, summarize_walk_forward_results(fold_results)


def format_walk_forward_section(fold_results: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "## Validation temporelle glissante", "",
        "Cinq fenêtres de test successives d'environ 365 jours ont été utilisées. Chaque modèle est entraîné uniquement sur les observations antérieures à la fenêtre testée.", "",
        "### Résumé par modèle", "",
        "| Modèle | MAE moyenne (°C) | Écart-type MAE | RMSE moyen (°C) | Biais moyen (°C) | Fenêtres |", "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row['modele']} | {row['mae_moyenne']:.3f} | {row['mae_ecart_type']:.3f} | {row['rmse_moyen']:.3f} | {row['biais_moyen']:+.3f} | {int(row['nombre_fenetres'])} |")
    lines.extend(["", "### Résultats par fenêtre", "", "| Fenêtre | Test début | Test fin | Modèle | MAE (°C) | RMSE (°C) | Biais (°C) |", "|---:|---|---|---|---:|---:|---:|"])
    for _, row in fold_results.iterrows():
        lines.append(f"| {int(row['fenetre'])} | {row['test_debut'].date()} | {row['test_fin'].date()} | {row['modele']} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['bias']:+.3f} |")
    return "\n".join(lines)


def run_full_analysis(filepath: str, test_size: float = 0.20, create_plots: bool = True, run_walk_forward: bool = True) -> dict:
    source_path = Path(filepath)
    print("\n=== 1. CHARGEMENT DU FICHIER ===")
    df = load_weather_csv(source_path)
    print(f"Fichier : {source_path.name}\nLignes : {len(df)}\nColonnes : {len(df.columns)}")

    print("\n=== 2. DÉTECTION DES COLONNES ===")
    mapping = detect_column_mapping(df)
    usable_mapping = get_usable_mapping(mapping)
    missing_required = [item for item in ["date", "tmean", "tmin", "tmax"] if usable_mapping.get(item) is None]
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
    annual_statistics = build_annual_statistics(df)
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

    print("\n=== 6. PRÉPARATION DES DONNÉES J+1 ===")
    model_data = build_j1_temperature_dataset(df)
    available_features = [column for column in FEATURE_COLUMNS if column in model_data.columns]
    missing_features = [column for column in FEATURE_COLUMNS if column not in model_data.columns]
    if missing_features:
        raise ValueError(f"Certaines features nécessaires sont absentes : {missing_features}")
    print(f"Lignes utilisables : {len(model_data)}\nNombre de features : {len(available_features)}")

    print("\n=== 7. DÉCOUPAGE CHRONOLOGIQUE ===")
    train_data, test_data = chronological_train_test_split(model_data, test_size=test_size)
    print(f"Train : {len(train_data)} lignes | {train_data['date'].min().date()} -> {train_data['date'].max().date()}")
    print(f"Test : {len(test_data)} lignes | {test_data['date'].min().date()} -> {test_data['date'].max().date()}")

    print("\n=== 8. ENTRAÎNEMENT ET COMPARAISON ===")
    y_test = test_data["target_tmean_j1"]
    naive_predictions = naive_persistence_forecast(test_data)
    linear_model, linear_predictions = fit_linear_regression(train_data=train_data, test_data=test_data, feature_columns=available_features)
    forest_model, forest_predictions = fit_random_forest(train_data=train_data, test_data=test_data, feature_columns=available_features)
    model_metrics = {
        "Baseline naïve": calculate_regression_metrics(y_test, naive_predictions),
        "Régression linéaire": calculate_regression_metrics(y_test, linear_predictions),
        "Random Forest": calculate_regression_metrics(y_test, forest_predictions),
    }
    for name, metrics in model_metrics.items():
        print(f"{name:22s} | MAE : {metrics['mae']:.3f} °C | RMSE : {metrics['rmse']:.3f} °C | Biais : {metrics['bias']:+.3f} °C")

    print("\n=== 9. IMPORTANCE PAR PERMUTATION ===")
    importance_df = compute_permutation_importance(model=forest_model, test_data=test_data, feature_columns=available_features, n_repeats=20)
    print(importance_df.to_string(index=False))

    walk_forward_results = pd.DataFrame()
    walk_forward_summary = pd.DataFrame()
    if run_walk_forward:
        print("\n=== 10. VALIDATION TEMPORELLE GLISSANTE ===")
        print("Cinq fenêtres d'environ un an : le calcul peut prendre plus de temps que l'analyse standard.")
        walk_forward_results, walk_forward_summary = run_walk_forward_validation(model_data, available_features)
        print(walk_forward_summary.to_string(index=False))

    if create_plots:
        print("\n=== 11. GRAPHIQUES ===")
        plot_annual_climate_summary(annual_statistics, "outputs/climat_annuel.png")
        plot_monthly_climate_summary(monthly_statistics, "outputs/climatologie_mensuelle.png")
        for predictions, name, output in [
            (naive_predictions, "Baseline naïve", "outputs/predictions_baseline_naive.png"),
            (linear_predictions, "Régression linéaire", "outputs/predictions_regression_lineaire.png"),
            (forest_predictions, "Random Forest", "outputs/predictions_random_forest.png"),
        ]:
            plot_test_predictions(test_data=test_data, predictions=predictions, model_name=name, output_path=output)

    print("\n=== 12. RAPPORT AUTOMATIQUE ===")
    insights = build_full_insights(quality_report, trend_text, model_metrics, daily_extremes, monthly_anomalies)
    report_sections = [
        "# Rapport d'analyse météo", "", f"**Fichier analysé :** `{source_path.name}`", "",
        "Ce rapport a été généré entièrement en local. Aucun fichier météo brut n'a été transmis à un service externe.", "",
        format_mapping_section(mapping), "", format_excluded_variables_section(), "", insights, "",
        format_climate_section(daily_extremes, hottest_months, coldest_months, wettest_months, hottest_years, monthly_anomalies), "",
        "## Graphiques climatiques générés", "", "- `outputs/climat_annuel.png`", "- `outputs/climatologie_mensuelle.png`", "",
        format_metrics_section(model_metrics), "", format_importance_section(importance_df),
    ]
    if run_walk_forward:
        report_sections.extend(["", format_walk_forward_section(walk_forward_results, walk_forward_summary)])
    save_markdown_report("\n".join(report_sections), "outputs/rapport_meteo.md")

    print("\n=== ANALYSE TERMINÉE ===")
    print("Consulte le dossier outputs/ pour les graphiques et le rapport Markdown.")
    return {
        "source_filename": source_path.name, "quality_report": quality_report, "trend": trend, "trend_text": trend_text,
        "mapping": mapping, "usable_mapping": usable_mapping, "daily_extremes": daily_extremes,
        "monthly_statistics": monthly_statistics, "annual_statistics": annual_statistics, "monthly_anomalies": monthly_anomalies,
        "hottest_months": hottest_months, "coldest_months": coldest_months, "wettest_months": wettest_months,
        "hottest_years": hottest_years, "model_metrics": model_metrics, "importance_df": importance_df,
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
