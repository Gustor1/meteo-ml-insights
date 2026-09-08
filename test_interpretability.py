from src.loader import load_weather_csv
from src.features import build_j1_temperature_dataset
from src.forecasting import fit_random_forest
from src.evaluation import (
    chronological_train_test_split,
    calculate_regression_metrics,
    compute_permutation_importance,
    print_metrics,
    print_permutation_importance,
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

importance_df = compute_permutation_importance(
    model=forest_model,
    test_data=test_data,
    feature_columns=feature_columns,
    target_col="target_tmean_j1",
    n_repeats=20,
)

print_permutation_importance(importance_df)