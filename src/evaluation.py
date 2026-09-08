import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error


def chronological_train_test_split(
    data: pd.DataFrame,
    test_size: float = 0.20,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les données dans l'ordre chronologique."""
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être compris strictement entre 0 et 1.")
    if date_col not in data.columns:
        raise ValueError(f"Colonne de date absente : {date_col}")

    sorted_data = data.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(sorted_data) * (1 - test_size))
    return sorted_data.iloc[:split_index].copy(), sorted_data.iloc[split_index:].copy()


def expanding_window_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_window_size: int = 365,
    min_train_size: int = 3650,
    date_col: str = "date",
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """
    Crée des fenêtres de validation temporelle à entraînement expansif.

    Chaque fenêtre teste des observations postérieures à toutes les données
    utilisées pour l'entraînement. test_window_size et min_train_size sont
    exprimés en nombre de lignes, donc approximativement en jours pour une
    série quotidienne complète.
    """
    if date_col not in data.columns:
        raise ValueError(f"Colonne de date absente : {date_col}")
    if n_splits < 2:
        raise ValueError("n_splits doit être au moins égal à 2.")
    if test_window_size < 1 or min_train_size < 1:
        raise ValueError("Les tailles de fenêtre doivent être strictement positives.")

    sorted_data = data.sort_values(date_col).reset_index(drop=True)
    required_size = min_train_size + n_splits * test_window_size
    if len(sorted_data) < required_size:
        raise ValueError(
            "La série est trop courte pour cette validation glissante : "
            f"{len(sorted_data)} lignes disponibles, {required_size} nécessaires."
        )

    first_test_start = len(sorted_data) - n_splits * test_window_size
    splits = []
    for fold_number in range(1, n_splits + 1):
        test_start = first_test_start + (fold_number - 1) * test_window_size
        test_end = test_start + test_window_size
        train_data = sorted_data.iloc[:test_start].copy()
        test_data = sorted_data.iloc[test_start:test_end].copy()
        splits.append((fold_number, train_data, test_data))
    return splits


def calculate_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """Calcule MAE, RMSE et biais de prévision."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    bias = (y_pred - y_true).mean()
    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "bias": round(float(bias), 3),
    }


def summarize_walk_forward_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les métriques calculées sur les fenêtres temporelles."""
    required_columns = ["modele", "mae", "rmse", "bias"]
    missing_columns = [column for column in required_columns if column not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes absentes pour le résumé : {missing_columns}")

    summary = (
        results_df.groupby("modele", as_index=False)
        .agg(
            mae_moyenne=("mae", "mean"),
            mae_ecart_type=("mae", "std"),
            rmse_moyen=("rmse", "mean"),
            biais_moyen=("bias", "mean"),
            nombre_fenetres=("fenetre", "nunique"),
        )
        .sort_values("mae_moyenne")
        .reset_index(drop=True)
    )

    numeric_columns = ["mae_moyenne", "mae_ecart_type", "rmse_moyen", "biais_moyen"]
    summary[numeric_columns] = summary[numeric_columns].round(3)
    return summary


def print_split_summary(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    date_col: str = "date",
) -> None:
    """Affiche le découpage choisi, sans afficher de données brutes."""
    print("=== DÉCOUPAGE CHRONOLOGIQUE ===")
    print(f"Train : {len(train_data)} lignes | {train_data[date_col].min().date()} -> {train_data[date_col].max().date()}")
    print(f"Test : {len(test_data)} lignes | {test_data[date_col].min().date()} -> {test_data[date_col].max().date()}")


def print_metrics(model_name: str, metrics: dict) -> None:
    """Affiche les métriques d'un modèle de manière lisible."""
    print(f"\n=== {model_name.upper()} ===")
    print(f"MAE : {metrics['mae']:.3f} °C")
    print(f"RMSE : {metrics['rmse']:.3f} °C")
    print(f"Biais : {metrics['bias']:+.3f} °C")


def compute_permutation_importance(
    model,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target_tmean_j1",
    n_repeats: int = 20,
) -> pd.DataFrame:
    """Calcule l'importance par permutation sur le jeu de test."""
    result = permutation_importance(
        estimator=model,
        X=test_data[feature_columns],
        y=test_data[target_col],
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "variable": feature_columns,
            "augmentation_mae_moyenne": result.importances_mean,
            "ecart_type": result.importances_std,
        }
    )
    return importance_df.sort_values("augmentation_mae_moyenne", ascending=False).reset_index(drop=True)


def print_permutation_importance(importance_df: pd.DataFrame) -> None:
    """Affiche les importances de permutation de manière lisible."""
    print("\n=== IMPORTANCE PAR PERMUTATION ===")
    print("Une valeur positive indique de combien la MAE augmente lorsque la variable est mélangée.")
    for _, row in importance_df.iterrows():
        print(f"{row['variable']:22s} : {row['augmentation_mae_moyenne']:+.3f} °C (écart-type : {row['ecart_type']:.3f})")
