from src.loader import load_weather_csv
from src.features import build_j1_temperature_dataset
from src.forecasting import naive_persistence_forecast
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