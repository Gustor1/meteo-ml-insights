import numpy as np
import pandas as pd
from scipy.stats import linregress


SEASON_MONTHS = {
    "Hiver": [12, 1, 2],
    "Printemps": [3, 4, 5],
    "Été": [6, 7, 8],
    "Automne": [9, 10, 11],
}


def build_annual_coverage(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Calcule le taux de couverture pour chaque année civile."""
    if date_col not in df.columns:
        raise ValueError(f"Colonne de date absente : {date_col}")

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    observed = dates.groupby(dates.dt.year).nunique().rename("jours_observes")
    coverage = observed.to_frame()
    coverage.index.name = "annee"
    coverage["jours_attendus"] = [
        366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365
        for year in coverage.index
    ]
    coverage["taux_couverture"] = coverage["jours_observes"] / coverage["jours_attendus"]
    return coverage.reset_index()


def filter_complete_years(
    df: pd.DataFrame,
    date_col: str = "date",
    minimum_coverage: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conserve uniquement les années dont la couverture atteint le seuil."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage doit être compris entre 0 et 1.")

    coverage = build_annual_coverage(df, date_col=date_col)
    complete_years = coverage.loc[
        coverage["taux_couverture"] >= minimum_coverage,
        "annee",
    ]
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.loc[data[date_col].dt.year.isin(complete_years)].copy()
    return data, coverage


def calculate_linear_trend(
    data: pd.DataFrame,
    value_col: str,
    time_col: str = "annee",
) -> dict:
    """Calcule une tendance linéaire et ses indicateurs statistiques."""
    required_columns = [time_col, value_col]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Colonnes absentes : {missing_columns}")

    valid_data = data[[time_col, value_col]].dropna()
    if len(valid_data) < 3:
        raise ValueError("Au moins trois observations sont nécessaires pour une tendance.")

    result = linregress(valid_data[time_col], valid_data[value_col])
    return {
        "pente_par_unite": float(result.slope),
        "pente_par_decennie": float(result.slope * 10),
        "intercept": float(result.intercept),
        "p_value": float(result.pvalue),
        "r_squared": float(result.rvalue ** 2),
        "erreur_standard": float(result.stderr),
        "nb_observations": int(len(valid_data)),
        "significatif": bool(result.pvalue < 0.05),
    }


def build_annual_climate_indicators(
    df: pd.DataFrame,
    date_col: str = "date",
    tmin_col: str = "tmin",
    tmax_col: str = "tmax",
    tmean_col: str = "tmean",
    precip_col: str = "precipitation_mm",
) -> pd.DataFrame:
    """Calcule indicateurs annuels de température, extrêmes et précipitations."""
    required_columns = [date_col, tmin_col, tmax_col, tmean_col, precip_col]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes nécessaires absentes : {missing_columns}")

    data = df[required_columns].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    data["annee"] = data[date_col].dt.year

    return (
        data.groupby("annee", as_index=False)
        .agg(
            temperature_moyenne_annuelle=(tmean_col, "mean"),
            temperature_minimale_moyenne_annuelle=(tmin_col, "mean"),
            temperature_maximale_moyenne_annuelle=(tmax_col, "mean"),
            temperature_minimale_record=(tmin_col, "min"),
            temperature_maximale_record=(tmax_col, "max"),
            jours_chauds_25=(tmax_col, lambda values: (values >= 25).sum()),
            jours_tres_chauds_30=(tmax_col, lambda values: (values >= 30).sum()),
            jours_canicule_35=(tmax_col, lambda values: (values >= 35).sum()),
            nuits_tropicales_20=(tmin_col, lambda values: (values >= 20).sum()),
            jours_de_gel=(tmin_col, lambda values: (values < 0).sum()),
            jours_sans_degel=(tmax_col, lambda values: (values < 0).sum()),
            precipitation_totale_annuelle=(precip_col, "sum"),
            jours_pluie_1mm=(precip_col, lambda values: (values >= 1).sum()),
            jours_pluie_10mm=(precip_col, lambda values: (values >= 10).sum()),
            jours_forte_pluie_20mm=(precip_col, lambda values: (values >= 20).sum()),
            precipitation_maximale_journaliere=(precip_col, "max"),
            jours_observes=(date_col, "nunique"),
        )
        .sort_values("annee")
        .reset_index(drop=True)
    )


def build_indicator_trends(
    annual_indicators: pd.DataFrame,
    indicator_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Calcule les tendances annuelles pour une liste d'indicateurs."""
    if "annee" not in annual_indicators.columns:
        raise ValueError("La colonne annee est nécessaire.")

    if indicator_columns is None:
        indicator_columns = [
            "temperature_moyenne_annuelle",
            "jours_tres_chauds_30",
            "jours_canicule_35",
            "nuits_tropicales_20",
            "jours_de_gel",
            "jours_sans_degel",
            "precipitation_totale_annuelle",
            "jours_pluie_10mm",
            "jours_forte_pluie_20mm",
            "precipitation_maximale_journaliere",
        ]

    rows = []
    for indicator in indicator_columns:
        if indicator not in annual_indicators.columns:
            continue
        trend = calculate_linear_trend(annual_indicators, value_col=indicator)
        rows.append({"indicateur": indicator, **trend})
    return pd.DataFrame(rows)


def build_seasonal_temperature_statistics(
    df: pd.DataFrame,
    date_col: str = "date",
    temp_col: str = "tmean",
    minimum_coverage: float = 0.99,
) -> pd.DataFrame:
    """Calcule les températures moyennes saisonnières sur des saisons complètes."""
    required_columns = [date_col, temp_col]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes nécessaires absentes : {missing_columns}")

    data = df[required_columns].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])
    data["mois"] = data[date_col].dt.month
    data["annee"] = data[date_col].dt.year
    data["saison"] = np.select(
        [
            data["mois"].isin(SEASON_MONTHS["Hiver"]),
            data["mois"].isin(SEASON_MONTHS["Printemps"]),
            data["mois"].isin(SEASON_MONTHS["Été"]),
        ],
        ["Hiver", "Printemps", "Été"],
        default="Automne",
    )
    data["annee_saison"] = np.where(data["mois"] == 12, data["annee"] + 1, data["annee"])

    seasonal = (
        data.groupby(["annee_saison", "saison"], as_index=False)
        .agg(
            temperature_moyenne_saisonniere=(temp_col, "mean"),
            jours_observes=(date_col, "nunique"),
        )
        .rename(columns={"annee_saison": "annee"})
    )

    def expected_days(row) -> int:
        months = SEASON_MONTHS[row["saison"]]
        years = [row["annee"] - 1 if month == 12 else row["annee"] for month in months]
        return sum(pd.Period(year=year, month=month, freq="M").days_in_month for year, month in zip(years, months))

    seasonal["jours_attendus"] = seasonal.apply(expected_days, axis=1)
    seasonal["taux_couverture"] = seasonal["jours_observes"] / seasonal["jours_attendus"]
    return seasonal.loc[seasonal["taux_couverture"] >= minimum_coverage].sort_values(["saison", "annee"]).reset_index(drop=True)


def build_seasonal_temperature_trends(seasonal_statistics: pd.DataFrame) -> pd.DataFrame:
    """Calcule une tendance de température séparée pour chaque saison."""
    required_columns = ["annee", "saison", "temperature_moyenne_saisonniere"]
    missing_columns = [column for column in required_columns if column not in seasonal_statistics.columns]
    if missing_columns:
        raise ValueError(f"Colonnes nécessaires absentes : {missing_columns}")

    rows = []
    for season in ["Hiver", "Printemps", "Été", "Automne"]:
        season_data = seasonal_statistics.loc[seasonal_statistics["saison"] == season]
        if len(season_data) < 3:
            continue
        trend = calculate_linear_trend(season_data, value_col="temperature_moyenne_saisonniere")
        rows.append({"saison": season, **trend})
    return pd.DataFrame(rows)
