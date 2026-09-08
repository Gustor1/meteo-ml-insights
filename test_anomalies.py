from src.loader import load_weather_csv
from src.anomalies import (
    add_seasonal_zscores,
    build_annual_statistics,
    build_monthly_statistics,
    detect_monthly_anomalies,
    get_daily_extremes,
    get_top_periods,
)

df = load_weather_csv("data/raw/meteo_strasbourg_propre.csv")

monthly = build_monthly_statistics(df)
monthly_with_zscores = add_seasonal_zscores(monthly)

annual = build_annual_statistics(df)

anomalies = detect_monthly_anomalies(
    monthly_with_zscores,
    zscore_threshold=2.0,
)

extremes = get_daily_extremes(df)

hottest_months = get_top_periods(
    monthly_with_zscores,
    value_col="temperature_moyenne_mensuelle",
    n=5,
    ascending=False,
)

coldest_months = get_top_periods(
    monthly_with_zscores,
    value_col="temperature_moyenne_mensuelle",
    n=5,
    ascending=True,
)

wettest_months = get_top_periods(
    monthly_with_zscores,
    value_col="precipitation_totale_mensuelle",
    n=5,
    ascending=False,
)

warmest_years = get_top_periods(
    annual,
    value_col="temperature_moyenne_annuelle",
    n=5,
    ascending=False,
)

print("=== STATISTIQUES MENSUELLES ===")
print(f"Nombre de mois analysés : {len(monthly_with_zscores)}")

print("\n=== ANOMALIES SAISONNIÈRES ===")
print(f"Nombre de mois signalés avec |z-score| >= 2 : {len(anomalies)}")

print("\n=== JOURS EXTRÊMES ===")
print(
    f"Jours chauds (tmax >= {extremes['seuil_jour_chaud_c']} °C) : "
    f"{extremes['jours_chauds']}"
)
print(
    f"Jours de gel (tmin < {extremes['seuil_gel_c']} °C) : "
    f"{extremes['jours_de_gel']}"
)
print(
    f"Jours de forte pluie "
    f"(précipitations >= {extremes['seuil_forte_pluie_mm']} mm) : "
    f"{extremes['jours_forte_pluie']}"
)

print("\n=== CINQ MOIS LES PLUS CHAUDS ===")
print(
    hottest_months[
        [
            "date",
            "temperature_moyenne_mensuelle",
            "zscore_temperature_saisonnier",
            "percentile_temperature_mois",
        ]
    ].to_string(index=False)
)

print("\n=== CINQ MOIS LES PLUS FROIDS ===")
print(
    coldest_months[
        [
            "date",
            "temperature_moyenne_mensuelle",
            "zscore_temperature_saisonnier",
            "percentile_temperature_mois",
        ]
    ].to_string(index=False)
)

print("\n=== CINQ MOIS LES PLUS HUMIDES ===")
print(
    wettest_months[
        [
            "date",
            "precipitation_totale_mensuelle",
            "zscore_precipitation_saisonnier",
            "percentile_precipitation_mois",
        ]
    ].to_string(index=False)
)

print("\n=== CINQ ANNÉES LES PLUS CHAUDES ===")
print(
    warmest_years[
        [
            "annee",
            "temperature_moyenne_annuelle",
            "precipitation_totale_annuelle",
        ]
    ].to_string(index=False)
)