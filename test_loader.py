from src.loader import load_weather_csv, detect_column_mapping, summarize_mapping

df = load_weather_csv("data/raw/meteo_strasbourg_propre.csv")
print(f"Lignes : {len(df)}, Colonnes : {len(df.columns)}")

mapping = detect_column_mapping(df)
summarize_mapping(mapping)

from src.loader import load_weather_csv, detect_column_mapping
from src.validator import build_quality_report, print_quality_report

df = load_weather_csv("data/raw/meteo_strasbourg_propre.csv")
mapping = detect_column_mapping(df)

report = build_quality_report(df, mapping)
print_quality_report(report)

from src.validator import check_missing_values_timeline

timeline = check_missing_values_timeline(df, "rayonnement_am")
print("\n=== Répartition temporelle des valeurs manquantes : rayonnement_am ===")
print(f"Première date manquante : {timeline['premiere_date_manquante']}")
print(f"Dernière date manquante : {timeline['derniere_date_manquante']}")
print(f"Nombre d'années concernées : {timeline['nb_annees_concernees']}")
print(f"Années concernées : {timeline['annees_concernees']}")

from src.loader import load_weather_csv, detect_column_mapping, get_usable_mapping

df = load_weather_csv("data/raw/meteo_strasbourg_propre.csv")
mapping = detect_column_mapping(df)
usable_mapping = get_usable_mapping(mapping)

print("Variables retenues pour le ML :", list(usable_mapping.keys()))

from src.features import compute_temperature_trend, describe_trend

trend = compute_temperature_trend(df)
print(trend)
print()
print(describe_trend(trend))