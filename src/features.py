import pandas as pd
import numpy as np
from scipy import stats


def compute_temperature_trend(
    df: pd.DataFrame,
    date_col: str = "date",
    temp_col: str = "tmean",
) -> dict:
    """Calcule la tendance linéaire de la température moyenne en °C par décennie."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    temps = pd.to_numeric(df[temp_col], errors="coerce")

    valid = dates.notna() & temps.notna()

    if valid.sum() < 2:
        raise ValueError(
            "Au moins deux dates et températures valides sont nécessaires "
            "pour calculer une tendance."
        )

    days_since_start = (dates - dates[valid].min()).dt.days

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        days_since_start[valid],
        temps[valid],
    )

    slope_per_decade = slope * 365.25 * 10

    return {
        "pente_par_jour": float(slope),
        "pente_par_decennie": round(float(slope_per_decade), 3),
        "p_value": float(p_value),
        "significatif": bool(p_value < 0.05),
        "r_squared": round(float(r_value ** 2), 4),
        "nb_observations": int(valid.sum()),
    }


def describe_trend(trend: dict) -> str:
    """
    Génère une conclusion déterministe à partir des résultats calculés.

    Cette fonction décrit une association temporelle statistique :
    elle n'identifie pas une cause de l'évolution observée.
    """
    pente = trend["pente_par_decennie"]
    signe = "hausse" if pente > 0 else "baisse"
    valeur = abs(pente)

    if trend["significatif"]:
        if trend["p_value"] < 0.001:
            seuil_p = "< 0,001"
        else:
            seuil_p = f"= {trend['p_value']:.3f}".replace(".", ",")

        phrase = (
            f"La température moyenne présente une {signe} de "
            f"{valeur:.2f} °C par décennie sur la période étudiée "
            f"({trend['nb_observations']} jours analysés).\n"
            f"Cette tendance est statistiquement significative (p {seuil_p})."
        )

        if trend["r_squared"] < 0.05:
            r_squared_fr = f"{trend['r_squared']:.4f}".replace(".", ",")
            phrase += (
                f"\nCette tendance de long terme n'explique cependant qu'une part "
                f"marginale (R² = {r_squared_fr}) des variations quotidiennes de "
                f"température, qui restent fortement variables d'un jour à l'autre."
            )

    else:
        p_value_fr = f"{trend['p_value']:.3f}".replace(".", ",")

        phrase = (
            f"La température moyenne présente une {signe} de "
            f"{valeur:.2f} °C par décennie sur la période étudiée, "
            f"mais cette tendance n'est pas statistiquement significative "
            f"(p = {p_value_fr}).\n"
            f"Il n'est pas possible d'affirmer une évolution claire sur cette période."
        )

    return phrase

def build_j1_temperature_dataset(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "tmean",
) -> pd.DataFrame:
    """
    Construit le jeu de données pour prédire la température moyenne à J+1.

    Chaque ligne contient uniquement des informations disponibles au jour t.
    La cible target_tmean_j1 correspond à la température moyenne du jour t+1.
    """
    required_columns = [date_col, target_col]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes obligatoires absentes : {missing_columns}"
        )

    data = df.copy()

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.sort_values(date_col).reset_index(drop=True)

    # Variables calendaires cycliques :
    # décembre et janvier sont ainsi considérés comme proches.
    day_of_year = data[date_col].dt.dayofyear

    data["jour_annee_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["jour_annee_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Variables retardées : uniquement des températures déjà connues
    # avant le jour courant ou au plus tard au jour courant.
    data["tmean_lag_1"] = data[target_col].shift(1)
    data["tmean_lag_7"] = data[target_col].shift(7)
    data["tmean_lag_30"] = data[target_col].shift(30)

    # La cible est la température moyenne du lendemain.
    data["target_tmean_j1"] = data[target_col].shift(-1)

    # Variables candidates connues au jour t.
    candidate_features = [
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

    # Garde seulement les colonnes réellement disponibles dans le CSV.
    feature_columns = [
        column for column in candidate_features
        if column in data.columns
    ]

    # Pour cette première version simple :
    # on retire uniquement les lignes incomplètes nécessaires au modèle.
    # Aucune modification n'est faite dans le CSV source.
    columns_needed = feature_columns + ["target_tmean_j1"]
    model_data = data.dropna(subset=columns_needed).copy()

    return model_data