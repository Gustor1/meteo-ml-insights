import pandas as pd
from pathlib import Path

# Alias possibles pour chaque variable météo (insensible à la casse)
# Adapté au schéma réel du fichier :
# date,annee,mois,jour,jour_annee,saison,tmin,tmax,tmean,precipitation_mm,
# code_precipitation,vent_ms,humidite_min,humidite_max,insolation_min,
# rayonnement_am,rayonnement_pm,rayonnement_total
COLUMN_ALIASES = {
    "date": ["date"],
    "annee": ["annee"],
    "mois": ["mois"],
    "jour": ["jour"],
    "jour_annee": ["jour_annee"],
    "saison": ["saison"],
    "tmin": ["tmin"],
    "tmax": ["tmax"],
    "tmean": ["tmean"],
    "precip_mm": ["precipitation_mm"],
    "precip_code": ["code_precipitation"],
    "vent": ["vent_ms"],
    "humidite_min": ["humidite_min"],
    "humidite_max": ["humidite_max"],
    "insolation": ["insolation_min"],
    "rayonnement_am": ["rayonnement_am"],
    "rayonnement_pm": ["rayonnement_pm"],
    "rayonnement_total": ["rayonnement_total"],
}

EXCLUDED_VARIABLES = {
    "rayonnement_am": "Capteur discontinué depuis 2008 (~43% de valeurs manquantes, "
                       "concentrées sur 19 des 34 années). Exclu des variables prédictives "
                       "car indisponible de façon fiable sur la période récente et future.",
}


def load_weather_csv(filepath: str, sep: str = None) -> pd.DataFrame:
    """Charge un CSV météo local. Détecte le séparateur si non précisé."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    if sep is None:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")

    return df


def detect_column_mapping(df: pd.DataFrame) -> dict:
    """Essaie de deviner à quelle variable correspond chaque colonne du DataFrame."""
    mapping = {}
    normalized_cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}

    for variable, aliases in COLUMN_ALIASES.items():
        found = None
        for original_col, norm_col in normalized_cols.items():
            if norm_col in aliases:
                found = original_col
                break
        mapping[variable] = found

    return mapping


def summarize_mapping(mapping: dict) -> None:
    """Affiche ce qui a été détecté et ce qui manque, pour vérification humaine."""
    print("Mapping détecté automatiquement :")
    for variable, col in mapping.items():
        status = col if col else "NON TROUVÉ — à mapper manuellement"
        print(f"  {variable:20s} -> {status}")


def get_missing_variables(mapping: dict) -> list:
    """Retourne la liste des variables attendues mais non trouvées dans le fichier."""
    return [variable for variable, col in mapping.items() if col is None]

def get_usable_mapping(mapping: dict) -> dict:
    """Retourne le mapping en excluant les variables jugées non fiables pour le ML."""
    return {
        variable: col
        for variable, col in mapping.items()
        if variable not in EXCLUDED_VARIABLES
    }