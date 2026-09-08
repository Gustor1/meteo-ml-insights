from pathlib import Path

import matplotlib.pyplot as plt


def plot_monthly_holt_winters_forecast(
    monthly_forecasting: dict,
    output_path: str = "outputs/prevision_mensuelle_holt_winters.png",
    historical_months: int = 60,
) -> None:
    """Trace l'historique, le test mensuel et la projection Holt-Winters."""
    monthly_series = monthly_forecasting["monthly_series"]
    train_series = monthly_forecasting["train_series"]
    test_series = monthly_forecasting["test_series"]
    baseline_predictions = monthly_forecasting["baseline_predictions"]
    holt_winters_predictions = monthly_forecasting["holt_winters_predictions"]
    future_forecast = monthly_forecasting["future_forecast"]
    metrics = monthly_forecasting["comparison_metrics"]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    historical_start = max(0, len(train_series) - historical_months)
    historical_train = train_series.iloc[historical_start:]

    figure, axis = plt.subplots(figsize=(15, 7))
    axis.plot(
        historical_train.index,
        historical_train,
        color="lightgray",
        linewidth=1.5,
        label=f"Historique d'entraînement ({len(historical_train)} derniers mois)",
    )
    axis.plot(
        test_series.index,
        test_series,
        color="#1F4E79",
        marker="o",
        markersize=3.5,
        linewidth=2.2,
        label="Températures mensuelles observées",
    )
    axis.plot(
        baseline_predictions.index,
        baseline_predictions,
        color="#7F8C8D",
        marker="o",
        markersize=3,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Baseline : même mois année précédente "
            f"(MAE = {metrics['Baseline même mois année précédente']['mae']:.2f} °C)"
        ),
    )
    axis.plot(
        holt_winters_predictions.index,
        holt_winters_predictions,
        color="#C0392B",
        marker="o",
        markersize=3,
        linestyle="--",
        linewidth=2,
        label=f"Holt-Winters — test (MAE = {metrics['Holt-Winters additif']['mae']:.2f} °C)",
    )
    axis.plot(
        future_forecast.index,
        future_forecast,
        color="#7D3C98",
        marker="o",
        markersize=4,
        linestyle="--",
        linewidth=2.5,
        label="Projection Holt-Winters",
    )
    axis.axvline(
        test_series.index.min(),
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Début du test",
    )
    axis.axvline(
        future_forecast.index.min(),
        color="#7D3C98",
        linestyle=":",
        linewidth=1.2,
        label="Début de la projection",
    )
    axis.set_title("Prévision mensuelle : baseline, Holt-Winters et projection")
    axis.set_xlabel("Date")
    axis.set_ylabel("Température moyenne mensuelle (°C)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    axis.text(
        0.02,
        0.02,
        "La projection est statistique et mensuelle.\n"
        "Elle ne constitue pas une prévision météo quotidienne.",
        transform=axis.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.9,
        },
    )
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Graphique enregistré localement : {output}")
