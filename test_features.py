from src.loader import load_weather_csv
from src.features import build_j1_temperature_dataset

df = load_weather_csv("data/raw/meteo_strasbourg_propre.csv")

model_data = build_j1_temperature_dataset(df)

print(f"Lignes d'origine : {len(df)}")
print(f"Lignes utilisables pour le modèle : {len(model_data)}")
print(f"Colonnes créées : {list(model_data.columns)}")

print("\nVariables qui serviront au modèle :")

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

available_features = [
    column for column in feature_columns
    if column in model_data.columns
]

print(available_features)

print("\nCible à prédire : target_tmean_j1")

columns_for_model = available_features + ["target_tmean_j1"]
missing_for_model = model_data[columns_for_model].isna().sum().sum()

print(
    f"Valeurs manquantes dans les variables réellement utilisées "
    f"par le modèle : {missing_for_model}"
)