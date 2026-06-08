# ============================================================
# XGBOOST — fonction universelle train/predict
# ============================================================
import numpy as np
import pandas as pd
import utils_series
from xgboost import XGBRegressor

# ── Hyperparamètres fixes ─────────────────────────────────────────────────────
_XGB_FIXED: dict = {
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "objective"        : "reg:squarederror",
    "random_state"     : 42,
    "verbosity"        : 0,
}

# ── Helpers (privés) ──────────────────────────────────────────────────────────

def _extract_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    """Accepte une Series ou un DataFrame avec colonne 'quantite_y'."""
    if isinstance(data, pd.Series):
        return data
    return data["quantite_y"]


def _build_features(series: pd.Series) -> pd.DataFrame:
    """Features : semaine ISO + lag_1."""
    df = pd.DataFrame({"y": series}, index=series.index)
    df["week"]  = df.index.isocalendar().week.astype(int)
    df["lag_1"] = df["y"].shift(1)
    return df


def _make_future_index(last_date: pd.Timestamp, n_weeks: int = 52) -> pd.DatetimeIndex:
    """Génère n_weeks dates hebdomadaires après last_date."""
    return pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                         periods=n_weeks, freq="W")


# ── Fonction principale ───────────────────────────────────────────────────────

def run_xgboost(
    n_estimators : int,
    max_depth    : int,
    learning_rate: float,
    data_train   : pd.DataFrame | pd.Series,
    data_test    : pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame | dict:
    """
    Deux modes selon les arguments fournis :

    MODE PRÉDICTION (data_test=None)
    ─────────────────────────────────
    - Entraîne sur l'ensemble de data_train
    - Retourne un DataFrame de prédictions sur 52 semaines
      Colonnes : ['ds', 'yhat']

    MODE ÉVALUATION (data_test fourni)
    ────────────────────────────────────
    - Entraîne sur data_train, prédit sur data_test
    - Retourne les métriques seasonal_metrics (dict)

    Paramètres
    ----------
    n_estimators  : nombre d'arbres
    max_depth     : profondeur max des arbres
    learning_rate : taux d'apprentissage
    data_train    : Series ou DataFrame (col 'quantite_y' + index datetime)
    data_test     : idem, optionnel
    """

    y_train = _extract_series(data_train)

    model = XGBRegressor(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        learning_rate     = learning_rate,
        subsample         = _XGB_FIXED["subsample"],
        colsample_bytree  = _XGB_FIXED["colsample_bytree"],
        objective         = _XGB_FIXED["objective"],
        random_state      = _XGB_FIXED["random_state"],
        verbosity         = _XGB_FIXED["verbosity"],
    )

    # ── Entraînement ─────────────────────────────────────────────────────────
    df_train = _build_features(y_train).dropna()
    X_train  = df_train[["week", "lag_1"]].values
    y_arr    = df_train["y"].values
    model.fit(X_train, y_arr)

    # ── MODE ÉVALUATION ──────────────────────────────────────────────────────
    if data_test is not None:
        y_test   = _extract_series(data_test)
        full     = pd.concat([y_train, y_test])
        df_full  = _build_features(full)
        df_t     = df_full.loc[y_test.index].fillna(0)
        X_test   = df_t[["week", "lag_1"]].values
        y_pred   = np.clip(model.predict(X_test), 0, None)
        metrics  = utils_series.seasonal_metrics(y_test, y_pred)
        return metrics

    # ── MODE PRÉDICTION ──────────────────────────────────────────────────────
    future_idx = _make_future_index(y_train.index.max(), n_weeks=52)
    preds      = []
    lag_val    = float(y_train.iloc[-1])          # dernier y connu

    for date in future_idx:
        week   = int(date.isocalendar()[1])
        X_step = np.array([[week, lag_val]])
        yhat   = float(np.clip(model.predict(X_step), 0, None))
        preds.append({"ds": date, "yhat": yhat})
        lag_val = yhat                             # lag glissant sur les prédictions

    return pd.DataFrame(preds).set_index("ds")