from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SEASON_COLORS = {
    "Hiver": "#2E86C1",
    "Printemps": "#27AE60",
    "Été": "#E67E22",
    "Automne": "#8E5A2B",
}


def _save_figure(figure, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Graphique enregistré localement : {output}")


def plot_test_predictions(
    test_data: pd.DataFrame,
    predictions: pd.Series,
    target_col: str = "target_tmean_j1",
    date_col: str = "date",
    model_name: str = "Random Forest",
    output_path: str = "outputs/predictions_random_forest.png",
    rolling_window: int = 30,
) -> None:
    """Compare les températures réelles et prédites sur la période de test."""
    if date_col not in test_data.columns:
        raise ValueError(f"Colonne de date absente : {date_col}")
    if target_col not in test_data.columns:
        raise ValueError(f"Colonne cible absente : {target_col}")

    plot_data = pd.DataFrame({
        "date": pd.to_datetime(test_data[date_col]),
        "reel": test_data[target_col].to_numpy(),
        "prediction": predictions.to_numpy(),
    }).sort_values("date")
    plot_data["reel_moyenne_mobile"] = plot_data["reel"].rolling(rolling_window, min_periods=1).mean()
    plot_data["prediction_moyenne_mobile"] = plot_data["prediction"].rolling(rolling_window, min_periods=1).mean()

    figure, axis = plt.subplots(figsize=(15, 7))
    axis.plot(plot_data["date"], plot_data["reel"], color="steelblue", alpha=0.18, linewidth=0.7, label="Température réelle quotidienne")
    axis.plot(plot_data["date"], plot_data["prediction"], color="darkorange", alpha=0.18, linewidth=0.7, label="Prédiction quotidienne")
    axis.plot(plot_data["date"], plot_data["reel_moyenne_mobile"], color="blue", linewidth=2, label=f"Réel — moyenne mobile {rolling_window} jours")
    axis.plot(plot_data["date"], plot_data["prediction_moyenne_mobile"], color="red", linewidth=2, label=f"Prédiction — moyenne mobile {rolling_window} jours")
    axis.set_title(f"Prédiction de température moyenne à J+1 — {model_name}")
    axis.set_xlabel("Date")
    axis.set_ylabel("Température moyenne (°C)")
    axis.legend()
    axis.grid(alpha=0.25)
    _save_figure(figure, output_path)


def plot_annual_climate_summary(annual_statistics: pd.DataFrame, output_path: str = "outputs/climat_annuel.png") -> None:
    """Affiche température moyenne et précipitations pour chaque année."""
    required = ["annee", "temperature_moyenne_annuelle", "precipitation_totale_annuelle"]
    missing = [column for column in required if column not in annual_statistics.columns]
    if missing:
        raise ValueError(f"Colonnes annuelles absentes : {missing}")
    data = annual_statistics.sort_values("annee")
    figure, (temp_axis, precip_axis) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    temp_axis.plot(data["annee"], data["temperature_moyenne_annuelle"], color="firebrick", marker="o", markersize=3, linewidth=1.5)
    temp_axis.set_title("Température moyenne annuelle")
    temp_axis.set_ylabel("Température (°C)")
    temp_axis.grid(alpha=0.25)
    precip_axis.bar(data["annee"], data["precipitation_totale_annuelle"], color="steelblue", width=0.8)
    precip_axis.set_title("Précipitations annuelles")
    precip_axis.set_xlabel("Année")
    precip_axis.set_ylabel("Précipitations (mm)")
    precip_axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Synthèse climatique annuelle", fontsize=15, y=0.99)
    _save_figure(figure, output_path)


def plot_monthly_climate_summary(monthly_statistics: pd.DataFrame, output_path: str = "outputs/climatologie_mensuelle.png") -> None:
    """Affiche les moyennes mensuelles calculées sur toute la période."""
    required = ["mois", "temperature_moyenne_mensuelle", "precipitation_totale_mensuelle"]
    missing = [column for column in required if column not in monthly_statistics.columns]
    if missing:
        raise ValueError(f"Colonnes mensuelles absentes : {missing}")
    data = monthly_statistics.groupby("mois", as_index=False).agg(
        temperature_moyenne=("temperature_moyenne_mensuelle", "mean"),
        precipitation_moyenne=("precipitation_totale_mensuelle", "mean"),
    ).sort_values("mois")
    labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    figure, (temp_axis, precip_axis) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    temp_axis.plot(data["mois"], data["temperature_moyenne"], color="firebrick", marker="o", linewidth=2)
    temp_axis.fill_between(data["mois"], data["temperature_moyenne"], color="tomato", alpha=0.20)
    temp_axis.set_title("Température moyenne par mois calendaire")
    temp_axis.set_ylabel("Température (°C)")
    temp_axis.grid(alpha=0.25)
    precip_axis.bar(data["mois"], data["precipitation_moyenne"], color="steelblue", width=0.7)
    precip_axis.set_title("Précipitations mensuelles moyennes")
    precip_axis.set_xlabel("Mois")
    precip_axis.set_ylabel("Précipitations (mm)")
    precip_axis.set_xticks(range(1, 13), labels)
    precip_axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Climatologie mensuelle de la période analysée", fontsize=15, y=0.99)
    _save_figure(figure, output_path)


def plot_seasonal_temperature_trends(
    seasonal_statistics: pd.DataFrame,
    seasonal_trends: pd.DataFrame,
    output_path: str = "outputs/tendances_temperature_saisons.png",
) -> None:
    """Trace les températures saisonnières et leur tendance linéaire."""
    required_statistics = ["annee", "saison", "temperature_moyenne_saisonniere"]
    required_trends = ["saison", "intercept", "pente_par_unite", "pente_par_decennie"]
    missing = [column for column in required_statistics if column not in seasonal_statistics.columns]
    missing += [column for column in required_trends if column not in seasonal_trends.columns]
    if missing:
        raise ValueError(f"Colonnes absentes pour les tendances saisonnières : {missing}")

    figure, axis = plt.subplots(figsize=(15, 7))
    for season in ["Hiver", "Printemps", "Été", "Automne"]:
        values = seasonal_statistics.loc[seasonal_statistics["saison"] == season].sort_values("annee")
        trend = seasonal_trends.loc[seasonal_trends["saison"] == season]
        if values.empty or trend.empty:
            continue
        trend_row = trend.iloc[0]
        color = SEASON_COLORS[season]
        axis.plot(values["annee"], values["temperature_moyenne_saisonniere"], marker="o", markersize=3.5, linewidth=1.2, alpha=0.65, color=color, label=f"{season} — observations")
        line = trend_row["intercept"] + trend_row["pente_par_unite"] * values["annee"]
        axis.plot(values["annee"], line, color=color, linewidth=2.5, linestyle="--", label=f"{season} — {trend_row['pente_par_decennie']:+.2f} °C/décennie")
    axis.set_title("Évolution des températures moyennes par saison")
    axis.set_xlabel("Année de la saison")
    axis.set_ylabel("Température moyenne saisonnière (°C)")
    axis.legend(ncol=2, fontsize=9)
    axis.grid(alpha=0.25)
    _save_figure(figure, output_path)


def _plot_indicator_with_trend(axis, data: pd.DataFrame, trends: pd.DataFrame, indicator: str, title: str, ylabel: str, color: str, unit: str) -> None:
    axis.plot(data["annee"], data[indicator], marker="o", markersize=3.5, linewidth=1.2, alpha=0.70, color=color, label="Valeur annuelle")
    trend = trends.loc[trends["indicateur"] == indicator]
    if not trend.empty:
        row = trend.iloc[0]
        line = row["intercept"] + row["pente_par_unite"] * data["annee"]
        axis.plot(data["annee"], line, color="black", linewidth=2, linestyle="--", label=f"Tendance : {row['pente_par_decennie']:+.2f} {unit}")
        axis.text(0.02, 0.96, f"p-value : {row['p_value']:.4g}\nR² : {row['r_squared']:.3f}", transform=axis.transAxes, verticalalignment="top", fontsize=9, bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "gray", "alpha": 0.9})
    axis.set_title(title)
    axis.set_xlabel("Année")
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=9)
    axis.grid(alpha=0.25)


def plot_thermal_extreme_trends(
    annual_indicators: pd.DataFrame,
    indicator_trends: pd.DataFrame,
    output_path: str = "outputs/tendances_extremes_thermiques.png",
) -> None:
    """Trace les évolutions annuelles des jours très chauds et de gel."""
    required = ["annee", "jours_tres_chauds_30", "jours_de_gel"]
    missing = [column for column in required if column not in annual_indicators.columns]
    if missing:
        raise ValueError(f"Colonnes annuelles absentes : {missing}")
    data = annual_indicators.sort_values("annee")
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    _plot_indicator_with_trend(axes[0], data, indicator_trends, "jours_tres_chauds_30", "Jours très chauds : tmax ≥ 30 °C", "Nombre de jours par an", "#C0392B", "jours/décennie")
    _plot_indicator_with_trend(axes[1], data, indicator_trends, "jours_de_gel", "Jours de gel : tmin < 0 °C", "Nombre de jours par an", "#2874A6", "jours/décennie")
    figure.suptitle("Évolution des extrêmes thermiques", fontsize=15, y=1.02)
    _save_figure(figure, output_path)


def plot_precipitation_trends(
    annual_indicators: pd.DataFrame,
    indicator_trends: pd.DataFrame,
    output_path: str = "outputs/tendances_precipitations.png",
) -> None:
    """Trace les précipitations annuelles et les jours de forte pluie."""
    required = ["annee", "precipitation_totale_annuelle", "jours_forte_pluie_20mm"]
    missing = [column for column in required if column not in annual_indicators.columns]
    if missing:
        raise ValueError(f"Colonnes annuelles absentes : {missing}")
    data = annual_indicators.sort_values("annee")
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    _plot_indicator_with_trend(axes[0], data, indicator_trends, "precipitation_totale_annuelle", "Cumul annuel de précipitations", "Précipitations (mm/an)", "#2874A6", "mm/décennie")
    _plot_indicator_with_trend(axes[1], data, indicator_trends, "jours_forte_pluie_20mm", "Jours de forte pluie : précipitations ≥ 20 mm", "Nombre de jours par an", "#6C3483", "jours/décennie")
    figure.suptitle("Évolution des précipitations", fontsize=15, y=1.02)
    _save_figure(figure, output_path)


def save_markdown_report(content: str, output_path: str = "outputs/rapport_meteo.md") -> None:
    """Enregistre le rapport Markdown localement."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Rapport Markdown enregistré localement : {output}")
