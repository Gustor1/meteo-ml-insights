from src.loader import load_weather_csv
from src.features import build_j1_temperature_dataset
from src.forecasting import (
    naive_persistence_forecast,
    fit_linear_regression,
    fit_random_forest,
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

y_test = test_data["target_tmean_j1"]

print_split_summary(train_data, test_data)

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

forest_model, forest_predictions = fit_random_forest(
    train_data=train_data,
    test_data=test_data,
    feature_columns=feature_columns,
)

forest_metrics = calculate_regression_metrics(
    y_true=y_test,
    y_pred=forest_predictions,
)

print_metrics("Random Forest", forest_metrics)

print("\n=== COMPARAISON DES MODÈLES ===")

linear_mae_gain = naive_metrics["mae"] - linear_metrics["mae"]
forest_mae_gain = naive_metrics["mae"] - forest_metrics["mae"]
forest_vs_linear_gain = linear_metrics["mae"] - forest_metrics["mae"]

print(f"Gain MAE régression vs baseline : {linear_mae_gain:+.3f} °C")
print(f"Gain MAE Random Forest vs baseline : {forest_mae_gain:+.3f} °C")
print(f"Gain MAE Random Forest vs régression : {forest_vs_linear_gain:+.3f} °C")

if forest_metrics["mae"] < linear_metrics["mae"]:
    print(
        "\nConclusion : la Random Forest est meilleure que la "
        "régression linéaire sur cette période de test."
    )
else:
    print(
        "\nConclusion : la Random Forest ne bat pas la régression linéaire. "
        "La régression linéaire reste le meilleur modèle à ce stade."
    )