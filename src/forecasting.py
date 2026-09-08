import pandas as pd


def naive_persistence_forecast(
    data: pd.DataFrame,
    current_temp_col: str = "tmean",
) -> pd.Series:
    """
    Baseline de persistance.

    Hypothèse :
    la température moyenne de demain sera égale à celle d'aujourd'hui.

    Pour chaque ligne t :
    prédiction(t+1) = tmean(t)
    """
    if current_temp_col not in data.columns:
        raise ValueError(
            f"Colonne nécessaire à la baseline absente : {current_temp_col}"
        )

    return data[current_temp_col].copy()

from sklearn.linear_model import LinearRegression


def fit_linear_regression(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target_tmean_j1",
) -> tuple[LinearRegression, pd.Series]:
    """
    Entraîne une régression linéaire sur la période train,
    puis prédit exclusivement la période test.

    Les données train et test doivent déjà être séparées
    chronologiquement avant l'appel de cette fonction.
    """
    required_columns = feature_columns + [target_col]

    missing_train = [
        column for column in required_columns
        if column not in train_data.columns
    ]
    missing_test = [
        column for column in feature_columns
        if column not in test_data.columns
    ]

    if missing_train:
        raise ValueError(
            f"Colonnes absentes dans les données train : {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Colonnes absentes dans les données test : {missing_test}"
        )

    if train_data[required_columns].isna().any().any():
        raise ValueError(
            "Les données train contiennent des valeurs manquantes "
            "dans les variables du modèle."
        )

    if test_data[feature_columns].isna().any().any():
        raise ValueError(
            "Les données test contiennent des valeurs manquantes "
            "dans les variables du modèle."
        )

    X_train = train_data[feature_columns]
    y_train = train_data[target_col]

    X_test = test_data[feature_columns]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = pd.Series(
        model.predict(X_test),
        index=test_data.index,
        name="prediction_regression_lineaire",
    )

    return model, predictions

from sklearn.ensemble import RandomForestRegressor


def fit_random_forest(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target_tmean_j1",
) -> tuple[RandomForestRegressor, pd.Series]:
    """
    Entraîne une Random Forest sur les données train
    et prédit exclusivement les données test.

    Les paramètres sont volontairement raisonnables pour un ordinateur CPU :
    - 300 arbres : suffisamment stable pour une première comparaison
    - n_jobs=-1 : utilise les cœurs CPU disponibles
    - random_state=42 : résultat reproductible
    """
    required_columns = feature_columns + [target_col]

    missing_train = [
        column for column in required_columns
        if column not in train_data.columns
    ]

    missing_test = [
        column for column in feature_columns
        if column not in test_data.columns
    ]

    if missing_train:
        raise ValueError(
            f"Colonnes absentes dans les données train : {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Colonnes absentes dans les données test : {missing_test}"
        )

    if train_data[required_columns].isna().any().any():
        raise ValueError(
            "Les données train contiennent des valeurs manquantes "
            "dans les variables du modèle."
        )

    if test_data[feature_columns].isna().any().any():
        raise ValueError(
            "Les données test contiennent des valeurs manquantes "
            "dans les variables du modèle."
        )

    X_train = train_data[feature_columns]
    y_train = train_data[target_col]

    X_test = test_data[feature_columns]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = pd.Series(
        model.predict(X_test),
        index=test_data.index,
        name="prediction_random_forest",
    )

    return model, predictions