import pandas as pd

from src.anomalies import (
    add_seasonal_zscores,
    build_monthly_statistics,
    detect_monthly_anomalies,
    get_daily_extremes,
)
from src.evaluation import (
    expanding_window_splits,
    summarize_walk_forward_results,
)


def build_synthetic_monthly_weather() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
    data = pd.DataFrame(
        {
            "date": dates,
            "tmean": 10.0,
            "precipitation_mm": 10.0,
        }
    )
    outlier = data["date"] == pd.Timestamp("2024-07-01")
    data.loc[outlier, "tmean"] = 30.0
    data.loc[outlier, "precipitation_mm"] = 100.0
    return data


def test_detects_hot_and_wet_seasonal_anomaly():
    weather = build_synthetic_monthly_weather()
    monthly = build_monthly_statistics(weather)
    monthly_with_zscores = add_seasonal_zscores(monthly)
    anomalies = detect_monthly_anomalies(monthly_with_zscores, zscore_threshold=2.0)

    july_2024 = anomalies.loc[anomalies["date"] == pd.Timestamp("2024-07-01")].iloc[0]

    assert july_2024["anomalie_chaude"]
    assert july_2024["anomalie_humide"]
    assert july_2024["zscore_temperature_saisonnier"] >= 2.0
    assert july_2024["zscore_precipitation_saisonnier"] >= 2.0


def test_counts_daily_extremes_with_explicit_thresholds():
    weather = pd.DataFrame(
        {
            "tmin": [-1.0, 0.0, 3.0, -2.0],
            "tmax": [29.9, 30.0, 35.0, 10.0],
            "precipitation_mm": [19.9, 20.0, 25.0, 0.0],
        }
    )

    extremes = get_daily_extremes(weather)

    assert extremes["jours_chauds"] == 2
    assert extremes["jours_de_gel"] == 2
    assert extremes["jours_forte_pluie"] == 2


def test_expanding_windows_never_use_future_observations_for_training():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=60, freq="D"),
            "target_tmean_j1": range(60),
        }
    )

    folds = expanding_window_splits(
        data,
        n_splits=3,
        test_window_size=10,
        min_train_size=20,
    )

    assert len(folds) == 3
    for _, train_data, test_data in folds:
        assert len(train_data) >= 20
        assert len(test_data) == 10
        assert train_data["date"].max() < test_data["date"].min()


def test_walk_forward_summary_aggregates_metrics_by_model():
    results = pd.DataFrame(
        {
            "fenetre": [1, 2, 1, 2],
            "modele": ["Baseline naïve", "Baseline naïve", "Random Forest", "Random Forest"],
            "mae": [2.0, 2.2, 1.5, 1.7],
            "rmse": [2.5, 2.7, 2.0, 2.1],
            "bias": [0.1, -0.1, 0.0, -0.2],
        }
    )

    summary = summarize_walk_forward_results(results)
    forest = summary.loc[summary["modele"] == "Random Forest"].iloc[0]

    assert forest["mae_moyenne"] == 1.6
    assert forest["nombre_fenetres"] == 2
