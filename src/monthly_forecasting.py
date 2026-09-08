import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.evaluation import calculate_regression_metrics


def build_complete_monthly_temperature_series(
    df: pd.DataFrame,
    date_col: str = "date",
    temp_col: str = "tmean",
    minimum_coverage: float = 0.99,
) -> tuple[pd.Series, pd.DataFrame]:
    """Construit une série mensuelle de température en contrôlant la couverture."""
    if date_col not in df.columns or temp_col not in df.columns:
        raise ValueError(f"Colonnes nécessaires absentes : {date_col}, {temp_col}")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage doit être compris entre 0 et 1.")

    data = df[[date_col, temp_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[temp_col] = pd.to_numeric(data[temp_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    monthly_temperature = data[temp_col].resample("MS").mean()
    coverage = pd.DataFrame({
        "jours_observes": data.resample("MS").size(),
    })
    coverage["jours_attendus"] = coverage.index.days_in_month
    coverage["taux_couverture"] = coverage["jours_observes"] / coverage["jours_attendus"]

    valid_months = coverage["taux_couverture"] >= minimum_coverage
    monthly_temperature = monthly_temperature.loc[valid_months].copy()
    monthly_temperature.name = "temperature_moyenne_mensuelle"

    expected_index = pd.date_range(monthly_temperature.index.min(), monthly_temperature.index.max(), freq="MS")
    if not monthly_temperature.index.equals(expected_index):
        raise ValueError(
            "La série mensuelle contient des mois incomplets ou absents entre deux mois valides. "
            "La prévision Holt-Winters exige une série mensuelle continue."
        )
    if monthly_temperature.isna().any():
        raise ValueError("La série mensuelle contient des températures manquantes.")

    return monthly_temperature.asfreq("MS"), coverage.reset_index(names="date")


def chronological_monthly_train_test_split(
    monthly_series: pd.Series,
    test_months: int = 36,
    minimum_train_months: int = 60,
) -> tuple[pd.Series, pd.Series]:
    """Réserve les derniers mois à une évaluation chronologique."""
    if test_months < 12:
        raise ValueError("test_months doit être au moins égal à 12.")
    if len(monthly_series) < minimum_train_months + test_months:
        raise ValueError(
            f"Série mensuelle trop courte : {len(monthly_series)} mois disponibles, "
            f"{minimum_train_months + test_months} nécessaires."
        )
    return monthly_series.iloc[:-test_months].copy(), monthly_series.iloc[-test_months:].copy()


def seasonal_year_lag_baseline(train_series: pd.Series, test_series: pd.Series) -> pd.Series:
    """Prédit chaque mois par la valeur du même mois de l'année précédente."""
    combined = pd.concat([train_series, test_series])
    predictions = combined.shift(12).loc[test_series.index]
    if predictions.isna().any():
        raise ValueError("La baseline saisonnière contient des valeurs manquantes.")
    predictions.name = "baseline_meme_mois_annee_precedente"
    return predictions


def fit_holt_winters_additive(train_series: pd.Series, forecast_months: int) -> tuple[object, pd.Series]:
    """Ajuste Holt-Winters additif et prévoit le nombre de mois demandé."""
    if forecast_months < 1:
        raise ValueError("forecast_months doit être strictement positif.")
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    )
    fitted_model = model.fit(optimized=True, use_brute=True)
    forecast = fitted_model.forecast(forecast_months)
    forecast.name = "prediction_holt_winters"
    return fitted_model, forecast


def run_monthly_holt_winters_analysis(
    df: pd.DataFrame,
    test_months: int = 36,
    forecast_months: int = 12,
    minimum_coverage: float = 0.99,
) -> dict:
    """Évalue Holt-Winters mensuel puis projette les mois suivant la série."""
    monthly_series, monthly_coverage = build_complete_monthly_temperature_series(
        df,
        minimum_coverage=minimum_coverage,
    )
    train_series, test_series = chronological_monthly_train_test_split(
        monthly_series,
        test_months=test_months,
    )
    baseline_predictions = seasonal_year_lag_baseline(train_series, test_series)
    _, holt_winters_predictions = fit_holt_winters_additive(train_series, len(test_series))
    holt_winters_predictions.index = test_series.index

    comparison_metrics = {
        "Baseline même mois année précédente": calculate_regression_metrics(test_series, baseline_predictions),
        "Holt-Winters additif": calculate_regression_metrics(test_series, holt_winters_predictions),
    }
    final_model, future_forecast = fit_holt_winters_additive(monthly_series, forecast_months)

    return {
        "monthly_series": monthly_series,
        "monthly_coverage": monthly_coverage,
        "train_series": train_series,
        "test_series": test_series,
        "baseline_predictions": baseline_predictions,
        "holt_winters_predictions": holt_winters_predictions,
        "comparison_metrics": comparison_metrics,
        "final_model": final_model,
        "future_forecast": future_forecast,
        "test_months": test_months,
        "forecast_months": forecast_months,
        "minimum_coverage": minimum_coverage,
    }
