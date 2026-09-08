from src.loader import load_weather_csv
from src.features import build_j1_temperature_dataset
from src.forecasting import (
    naive_persistence_forecast,
    fit_random_forest,
)
from src.evaluation import (
    chronological_train_test_split,
    calculate_regression_metrics,
    print_metrics,
)
from src.reporting import plot_test_predictions

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

forest_model, forest_predictions = fit_random_forest(
    train_data=train_data,
    test_data=test_data,
    feature_columns=feature_columns,
)

forest_metrics = calculate_regression_metrics(
    y_true=test_data["target_tmean_j1"],
    y_pred=forest_predictions,
)

print_metrics("Random Forest", forest_metrics)

plot_test_predictions(
    test_data=test_data,
    predictions=forest_predictions,
    target_col="target_tmean_j1",
    date_col="date",
    model_name="Random Forest",
    output_path="outputs/predictions_random_forest.png",
    rolling_window=30,
)

naive_predictions = naive_persistence_forecast(
    test_data,
    current_temp_col="tmean",
)

plot_test_predictions(
    test_data=test_data,
    predictions=naive_predictions,
    target_col="target_tmean_j1",
    date_col="date",
    model_name="Baseline naïve",
    output_path="outputs/predictions_baseline_naive.png",
    rolling_window=30,
)