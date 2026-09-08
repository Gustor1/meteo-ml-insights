from src.loader import load_weather_csv
from src.features import build_j1_temperature_dataset
from src.forecasting import (
    naive_persistence_forecast,
    fit_linear_regression,
)
from src.evaluation import (
    chronological_train_test_split,
    calculate_regression_metrics,
    print_split_summary,
    print_metrics,
)

df = load_weather_csv("data/raw/meteo_strasbourg_propre.csv")

model_data = build_j1_temperature_dataset(df)

train_data, test_data = chronological_train_test_split(
    model_data,
    test_size=0.20,
)

feature_columns = [
    "tmin",
    "tmax",
    "tmean",
    "precipitation_mm",
    "vent_ms",
    "humidite_min",
    "humidite_max",
    "insolation_min",
    "rayonnement_pm",
    "rayonnement_total",
    "jour_annee_sin",
    "jour_annee_cos",
    "tmean_lag_1",
    "tmean_lag_7",
    "tmean_lag_30",
]

print_split_summary(train_data, test_data)

y_test = test_data["target_tmean_j1"]

naive_predictions = naive_persistence_forecast(
    test_data,
    current_temp_col="tmean",
)

naive_metrics = calculate_regression_metrics(
    y_true=y_test,
    y_pred=naive_predictions,
)

print_metrics("Baseline naïve : demain = aujourd'hui", naive_metrics)

linear_model, linear_predictions = fit_linear_regression(
    train_data=train_data,
    test_data=test_data,
    feature_columns=feature_columns,
)

linear_metrics = calculate_regression_metrics(
    y_true=y_test,
    y_pred=linear_predictions,
)

print_metrics("Régression linéaire", linear_metrics)

mae_gain = naive_metrics["mae"] - linear_metrics["mae"]
rmse_gain = naive_metrics["rmse"] - linear_metrics["rmse"]

print("\n=== COMPARAISON ===")
print(f"Gain de MAE par rapport à la baseline : {mae_gain:+.3f} °C")
print(f"Gain de RMSE par rapport à la baseline : {rmse_gain:+.3f} °C")

if mae_gain > 0:
    print("Conclusion : la régression linéaire améliore la baseline naïve.")
else:
    print(
        "Conclusion : la régression linéaire ne bat pas la baseline naïve. "
        "La baseline reste donc le meilleur choix à ce stade."
    )