import numpy as np
import pandas as pd


def build_monthly_statistics(
    df: pd.DataFrame,
    date_col: str = "date",
    temp_col: str = "tmean",
    precip_col: str = "precipitation_mm",
) -> pd.DataFrame:
    """
    Agrège les données quotidiennes en statistiques mensuelles.

    - temperature_moyenne_mensuelle : moyenne des tmean quotidiennes.
    - precipitation_totale_mensuelle : somme des précipitations quotidiennes.
    """
    required_columns = [date_col, temp_col, precip_col]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes nécessaires absentes : {missing_columns}"
        )

    data = df[
        [date_col, temp_col, precip_col]
    ].copy()

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce",
    )

    data[temp_col] = pd.to_numeric(
        data[temp_col],
        errors="coerce",
    )

    data[precip_col] = pd.to_numeric(
        data[precip_col],
        errors="coerce",
    )

    data = data.dropna(subset=[date_col])
    data = data.sort_values(date_col)
    data = data.set_index(date_col)

    monthly = data.resample("MS").agg(
        temperature_moyenne_mensuelle=(
            temp_col,
            "mean",
        ),
        precipitation_totale_mensuelle=(
            precip_col,
            lambda values: values.sum(min_count=1),
        ),
    )

    monthly = monthly.reset_index()

    monthly["annee"] = monthly[date_col].dt.year
    monthly["mois"] = monthly[date_col].dt.month

    return monthly


def build_annual_statistics(
    df: pd.DataFrame,
    date_col: str = "date",
    temp_col: str = "tmean",
    precip_col: str = "precipitation_mm",
) -> pd.DataFrame:
    """
    Agrège les données quotidiennes en statistiques annuelles.

    - temperature_moyenne_annuelle : moyenne des tmean quotidiennes.
    - precipitation_totale_annuelle : somme des précipitations quotidiennes.
    """
    required_columns = [date_col, temp_col, precip_col]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes nécessaires absentes : {missing_columns}"
        )

    data = df[
        [date_col, temp_col, precip_col]
    ].copy()

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce",
    )

    data[temp_col] = pd.to_numeric(
        data[temp_col],
        errors="coerce",
    )

    data[precip_col] = pd.to_numeric(
        data[precip_col],
        errors="coerce",
    )

    data = data.dropna(subset=[date_col])
    data = data.sort_values(date_col)
    data = data.set_index(date_col)

    annual = data.resample("YS").agg(
        temperature_moyenne_annuelle=(
            temp_col,
            "mean",
        ),
        precipitation_totale_annuelle=(
            precip_col,
            lambda values: values.sum(min_count=1),
        ),
    )

    annual = annual.reset_index()

    annual["annee"] = annual[date_col].dt.year

    return annual


def add_seasonal_zscores(
    monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ajoute des z-scores mensuels saisonniers.

    Chaque mois est comparé uniquement aux mêmes mois des autres années :
    janvier avec janvier, juillet avec juillet, etc.
    """
    data = monthly_df.copy()

    temperature_col = "temperature_moyenne_mensuelle"
    precipitation_col = "precipitation_totale_mensuelle"

    temperature_stats = (
        data.groupby("mois")[temperature_col]
        .agg(["mean", "std", "count"])
        .rename(
            columns={
                "mean": "temperature_moyenne_reference",
                "std": "temperature_ecart_type_reference",
                "count": "temperature_nb_mois_reference",
            }
        )
    )

    precipitation_stats = (
        data.groupby("mois")[precipitation_col]
        .agg(["mean", "std", "count"])
        .rename(
            columns={
                "mean": "precipitation_moyenne_reference",
                "std": "precipitation_ecart_type_reference",
                "count": "precipitation_nb_mois_reference",
            }
        )
    )

    data = data.merge(
        temperature_stats,
        left_on="mois",
        right_index=True,
        how="left",
    )

    data = data.merge(
        precipitation_stats,
        left_on="mois",
        right_index=True,
        how="left",
    )

    data["zscore_temperature_saisonnier"] = np.where(
        data["temperature_ecart_type_reference"] > 0,
        (
            data[temperature_col]
            - data["temperature_moyenne_reference"]
        )
        / data["temperature_ecart_type_reference"],
        np.nan,
    )

    data["zscore_precipitation_saisonnier"] = np.where(
        data["precipitation_ecart_type_reference"] > 0,
        (
            data[precipitation_col]
            - data["precipitation_moyenne_reference"]
        )
        / data["precipitation_ecart_type_reference"],
        np.nan,
    )

    data["percentile_temperature_mois"] = (
        data.groupby("mois")[temperature_col]
        .rank(pct=True)
    )

    data["percentile_precipitation_mois"] = (
        data.groupby("mois")[precipitation_col]
        .rank(pct=True)
    )

    return data


def detect_monthly_anomalies(
    monthly_df: pd.DataFrame,
    zscore_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Identifie les mois inhabituellement chauds, froids, secs ou humides.

    Une anomalie est signalée lorsque le z-score saisonnier est
    supérieur ou égal au seuil en valeur absolue.
    """
    required_columns = [
        "zscore_temperature_saisonnier",
        "zscore_precipitation_saisonnier",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in monthly_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Les z-scores doivent être calculés avant la détection "
            f"des anomalies : {missing_columns}"
        )

    data = monthly_df.copy()

    data["anomalie_chaude"] = (
        data["zscore_temperature_saisonnier"]
        >= zscore_threshold
    )

    data["anomalie_froide"] = (
        data["zscore_temperature_saisonnier"]
        <= -zscore_threshold
    )

    data["anomalie_humide"] = (
        data["zscore_precipitation_saisonnier"]
        >= zscore_threshold
    )

    data["anomalie_seche"] = (
        data["zscore_precipitation_saisonnier"]
        <= -zscore_threshold
    )

    anomalies = data[
        data[
            [
                "anomalie_chaude",
                "anomalie_froide",
                "anomalie_humide",
                "anomalie_seche",
            ]
        ].any(axis=1)
    ].copy()

    return anomalies.sort_values("date").reset_index(drop=True)


def get_daily_extremes(
    df: pd.DataFrame,
    tmin_col: str = "tmin",
    tmax_col: str = "tmax",
    precip_col: str = "precipitation_mm",
    hot_day_threshold: float = 30.0,
    frost_day_threshold: float = 0.0,
    heavy_rain_threshold: float = 20.0,
) -> dict:
    """
    Compte les jours extrêmes selon des seuils explicites et configurables.

    - Jour chaud : tmax >= 30 °C.
    - Jour de gel : tmin < 0 °C.
    - Forte précipitation : pluie >= 20 mm.
    """
    required_columns = [
        tmin_col,
        tmax_col,
        precip_col,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes nécessaires absentes : {missing_columns}"
        )

    return {
        "seuil_jour_chaud_c": hot_day_threshold,
        "jours_chauds": int(
            (df[tmax_col] >= hot_day_threshold).sum()
        ),
        "seuil_gel_c": frost_day_threshold,
        "jours_de_gel": int(
            (df[tmin_col] < frost_day_threshold).sum()
        ),
        "seuil_forte_pluie_mm": heavy_rain_threshold,
        "jours_forte_pluie": int(
            (df[precip_col] >= heavy_rain_threshold).sum()
        ),
    }


def get_top_periods(
    statistics_df: pd.DataFrame,
    value_col: str,
    n: int = 5,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Retourne les n périodes les plus élevées ou les plus faibles.

    Exemple :
    - mois les plus chauds : ascending=False.
    - mois les plus froids : ascending=True.
    """
    if value_col not in statistics_df.columns:
        raise ValueError(f"Colonne absente : {value_col}")

    return (
        statistics_df
        .sort_values(value_col, ascending=ascending)
        .head(n)
        .copy()
    )