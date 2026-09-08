from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_test_predictions(
    test_data: pd.DataFrame,
    predictions: pd.Series,
    target_col: str = "target_tmean_j1",
    date_col: str = "date",
    model_name: str = "Random Forest",
    output_path: str = "outputs/predictions_random_forest.png",
    rolling_window: int = 30,
) -> None:
    """Crée un graphique local comparant valeurs réelles et prédictions."""
    if date_col not in test_data.columns:
        raise ValueError(f"Colonne de date absente : {date_col}")
    if target_col not in test_data.columns:
        raise ValueError(f"Colonne cible absente : {target_col}")

    plot_data = pd.DataFrame(
        {
            "date": pd.to_datetime(test_data[date_col]),
            "reel": test_data[target_col].to_numpy(),
            "prediction": predictions.to_numpy(),
        }
    ).sort_values("date")

    plot_data["reel_moyenne_mobile"] = (
        plot_data["reel"].rolling(window=rolling_window, min_periods=1).mean()
    )
    plot_data["prediction_moyenne_mobile"] = (
        plot_data["prediction"].rolling(window=rolling_window, min_periods=1).mean()
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 7))
    plt.plot(
        plot_data["date"],
        plot_data["reel"],
        color="steelblue",
        alpha=0.18,
        linewidth=0.7,
        label="Température réelle quotidienne",
    )
    plt.plot(
        plot_data["date"],
        plot_data["prediction"],
        color="darkorange",
        alpha=0.18,
        linewidth=0.7,
        label="Prédiction quotidienne",
    )
    plt.plot(
        plot_data["date"],
        plot_data["reel_moyenne_mobile"],
        color="blue",
        linewidth=2,
        label=f"Réel — moyenne mobile {rolling_window} jours",
    )
    plt.plot(
        plot_data["date"],
        plot_data["prediction_moyenne_mobile"],
        color="red",
        linewidth=2,
        label=f"Prédiction — moyenne mobile {rolling_window} jours",
    )
    plt.title(f"Prédiction de température moyenne à J+1 — {model_name}")
    plt.xlabel("Date")
    plt.ylabel("Température moyenne (°C)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Graphique enregistré localement : {output}")


def plot_annual_climate_summary(
    annual_statistics: pd.DataFrame,
    output_path: str = "outputs/climat_annuel.png",
) -> None:
    """Affiche température moyenne et précipitations pour chaque année."""
    required_columns = [
        "annee",
        "temperature_moyenne_annuelle",
        "precipitation_totale_annuelle",
    ]
    missing_columns = [
        column for column in required_columns if column not in annual_statistics.columns
    ]
    if missing_columns:
        raise ValueError(f"Colonnes annuelles absentes : {missing_columns}")

    data = annual_statistics.sort_values("annee").copy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, (temperature_axis, precipitation_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 9),
        sharex=True,
    )

    temperature_axis.plot(
        data["annee"],
        data["temperature_moyenne_annuelle"],
        color="firebrick",
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    temperature_axis.set_title("Température moyenne annuelle")
    temperature_axis.set_ylabel("Température (°C)")
    temperature_axis.grid(alpha=0.25)

    precipitation_axis.bar(
        data["annee"],
        data["precipitation_totale_annuelle"],
        color="steelblue",
        width=0.8,
    )
    precipitation_axis.set_title("Précipitations annuelles")
    precipitation_axis.set_xlabel("Année")
    precipitation_axis.set_ylabel("Précipitations (mm)")
    precipitation_axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Synthèse climatique annuelle", fontsize=15, y=0.99)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Graphique enregistré localement : {output}")


def plot_monthly_climate_summary(
    monthly_statistics: pd.DataFrame,
    output_path: str = "outputs/climatologie_mensuelle.png",
) -> None:
    """Affiche les moyennes mensuelles calculées sur toute la période."""
    required_columns = [
        "mois",
        "temperature_moyenne_mensuelle",
        "precipitation_totale_mensuelle",
    ]
    missing_columns = [
        column for column in required_columns if column not in monthly_statistics.columns
    ]
    if missing_columns:
        raise ValueError(f"Colonnes mensuelles absentes : {missing_columns}")

    data = (
        monthly_statistics.groupby("mois", as_index=False)
        .agg(
            temperature_moyenne=("temperature_moyenne_mensuelle", "mean"),
            precipitation_moyenne=("precipitation_totale_mensuelle", "mean"),
        )
        .sort_values("mois")
    )

    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, (temperature_axis, precipitation_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 9),
        sharex=True,
    )

    temperature_axis.plot(
        data["mois"],
        data["temperature_moyenne"],
        color="firebrick",
        marker="o",
        linewidth=2,
    )
    temperature_axis.fill_between(
        data["mois"],
        data["temperature_moyenne"],
        color="tomato",
        alpha=0.20,
    )
    temperature_axis.set_title("Température moyenne par mois calendaire")
    temperature_axis.set_ylabel("Température (°C)")
    temperature_axis.grid(alpha=0.25)

    precipitation_axis.bar(
        data["mois"],
        data["precipitation_moyenne"],
        color="steelblue",
        width=0.7,
    )
    precipitation_axis.set_title("Précipitations mensuelles moyennes")
    precipitation_axis.set_xlabel("Mois")
    precipitation_axis.set_ylabel("Précipitations (mm)")
    precipitation_axis.grid(axis="y", alpha=0.25)
    precipitation_axis.set_xticks(range(1, 13), month_labels)

    figure.suptitle("Climatologie mensuelle de la période analysée", fontsize=15, y=0.99)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Graphique enregistré localement : {output}")


def save_markdown_report(
    content: str,
    output_path: str = "outputs/rapport_meteo.md",
) -> None:
    """Enregistre un rapport Markdown localement."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Rapport Markdown enregistré localement : {output}")
