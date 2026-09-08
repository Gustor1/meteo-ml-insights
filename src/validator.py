import pandas as pd


def check_period_coverage(df: pd.DataFrame, date_col: str = "date") -> dict:
    """Vérifie la période couverte et détecte les dates manquantes dans la série."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    invalid_dates = dates.isna().sum()

    full_range = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
    missing_dates = full_range.difference(dates.dropna())

    return {
        "date_min": dates.min(),
        "date_max": dates.max(),
        "jours_attendus": len(full_range),
        "jours_presents": dates.notna().sum(),
        "dates_invalides": int(invalid_dates),
        "dates_manquantes": len(missing_dates),
    }


def check_duplicates(df: pd.DataFrame, date_col: str = "date") -> int:
    """Compte les lignes en doublon sur la colonne date."""
    return int(df.duplicated(subset=[date_col]).sum())


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Compte les valeurs manquantes par colonne."""
    return df.isna().sum()


def check_physical_consistency(df: pd.DataFrame, mapping: dict) -> dict:
    """Vérifie des règles physiques simples, sans rien supprimer."""
    issues = {}

    tmin_col, tmax_col = mapping.get("tmin"), mapping.get("tmax")
    if tmin_col and tmax_col:
        issues["tmin_superieur_tmax"] = int((df[tmin_col] > df[tmax_col]).sum())

    precip_col = mapping.get("precip_mm")
    if precip_col:
        issues["precipitation_negative"] = int((df[precip_col] < 0).sum())

    hum_min_col, hum_max_col = mapping.get("humidite_min"), mapping.get("humidite_max")
    for col_name, col in [("humidite_min", hum_min_col), ("humidite_max", hum_max_col)]:
        if col:
            issues[f"{col_name}_hors_plage"] = int(((df[col] < 0) | (df[col] > 100)).sum())

    return issues


def build_quality_report(df: pd.DataFrame, mapping: dict) -> dict:
    """Assemble un rapport de qualité complet, purement descriptif."""
    date_col = mapping.get("date", "date")

    report = {
        "nb_lignes": len(df),
        "nb_colonnes": len(df.columns),
        "periode": check_period_coverage(df, date_col),
        "doublons_date": check_duplicates(df, date_col),
        "valeurs_manquantes": check_missing_values(df).to_dict(),
        "incoherences_physiques": check_physical_consistency(df, mapping),
    }
    return report


def print_quality_report(report: dict) -> None:
    """Affiche le rapport de façon lisible, sans jamais afficher de valeurs brutes."""
    print("=== RAPPORT DE QUALITÉ ===")
    print(f"Lignes : {report['nb_lignes']}, Colonnes : {report['nb_colonnes']}")

    periode = report["periode"]
    print(f"\nPériode : {periode['date_min']} -> {periode['date_max']}")
    print(f"Jours attendus : {periode['jours_attendus']}, présents : {periode['jours_presents']}")
    print(f"Dates invalides : {periode['dates_invalides']}, dates manquantes : {periode['dates_manquantes']}")

    print(f"\nDoublons sur la date : {report['doublons_date']}")

    print("\nValeurs manquantes par colonne :")
    for col, n in report["valeurs_manquantes"].items():
        if n > 0:
            print(f"  {col} : {n}")

    print("\nIncohérences physiques détectées :")
    for check, n in report["incoherences_physiques"].items():
        marker = "⚠️ " if n > 0 else "✅ "
        print(f"  {marker}{check} : {n}")

def check_missing_values_timeline(df: pd.DataFrame, column: str, date_col: str = "date") -> dict:
    """Analyse la répartition temporelle des valeurs manquantes d'une colonne donnée."""
    missing = df[df[column].isna()][date_col]
    missing_dates = pd.to_datetime(missing)

    return {
        "colonne": column,
        "premiere_date_manquante": missing_dates.min(),
        "derniere_date_manquante": missing_dates.max(),
        "nb_annees_concernees": missing_dates.dt.year.nunique(),
        "annees_concernees": sorted(missing_dates.dt.year.unique().tolist()),
    }