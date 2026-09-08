import pandas as pd

from src.climate_indicators import (
    build_annual_climate_indicators,
    build_annual_coverage,
    build_indicator_trends,
    build_seasonal_temperature_statistics,
    build_seasonal_temperature_trends,
    filter_complete_years,
)


def test_filters_incomplete_calendar_years():
    complete_2020 = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    incomplete_2021 = pd.date_range("2021-01-01", periods=10, freq="D")
    weather = pd.DataFrame(
        {
            "date": complete_2020.append(incomplete_2021),
            "tmin": 2.0,
            "tmax": 12.0,
            "tmean": 7.0,
            "precipitation_mm": 1.0,
        }
    )

    coverage = build_annual_coverage(weather)
    complete_weather, _ = filter_complete_years(weather, minimum_coverage=0.99)

    assert coverage.loc[coverage["annee"] == 2020, "taux_couverture"].iloc[0] == 1.0
    assert coverage.loc[coverage["annee"] == 2021, "taux_couverture"].iloc[0] < 0.99
    assert set(complete_weather["date"].dt.year.unique()) == {2020}


def test_builds_annual_thermal_and_precipitation_indicators():
    weather = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "tmin": [-1.0, 0.0, 20.0, 21.0, 3.0],
            "tmax": [-0.5, 25.0, 30.0, 35.0, 36.0],
            "tmean": [0.0, 10.0, 25.0, 28.0, 20.0],
            "precipitation_mm": [0.0, 1.0, 10.0, 20.0, 25.0],
        }
    )

    annual = build_annual_climate_indicators(weather)
    row = annual.iloc[0]

    assert row["jours_chauds_25"] == 4
    assert row["jours_tres_chauds_30"] == 3
    assert row["jours_canicule_35"] == 2
    assert row["nuits_tropicales_20"] == 2
    assert row["jours_de_gel"] == 1
    assert row["jours_sans_degel"] == 1
    assert row["jours_pluie_1mm"] == 4
    assert row["jours_pluie_10mm"] == 3
    assert row["jours_forte_pluie_20mm"] == 2
    assert row["precipitation_maximale_journaliere"] == 25.0


def test_calculates_positive_indicator_trend():
    annual_indicators = pd.DataFrame(
        {
            "annee": [2018, 2019, 2020, 2021, 2022],
            "jours_tres_chauds_30": [2, 3, 4, 5, 6],
        }
    )

    trends = build_indicator_trends(
        annual_indicators,
        indicator_columns=["jours_tres_chauds_30"],
    )
    row = trends.iloc[0]

    assert row["indicateur"] == "jours_tres_chauds_30"
    assert row["pente_par_decennie"] == 10.0
    assert row["significatif"]


def test_groups_december_with_following_winter_and_calculates_seasonal_trends():
    winter_2020 = pd.date_range("2019-12-01", "2020-02-29", freq="D")
    winter_2021 = pd.date_range("2020-12-01", "2021-02-28", freq="D")
    weather = pd.DataFrame(
        {
            "date": winter_2020.append(winter_2021),
            "tmean": [2.0] * len(winter_2020) + [4.0] * len(winter_2021),
        }
    )

    seasonal = build_seasonal_temperature_statistics(weather)
    winters = seasonal.loc[seasonal["saison"] == "Hiver"].sort_values("annee")
    trends = build_seasonal_temperature_trends(seasonal)

    assert winters["annee"].tolist() == [2020, 2021]
    assert winters["temperature_moyenne_saisonniere"].tolist() == [2.0, 4.0]
    assert trends.empty
