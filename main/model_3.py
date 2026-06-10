import numpy as np
import pandas as pd
import utils_series

# ── Helpers (privés) ──────────────────────────────────────────────────────────
def _extract_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    """Accepte une Series ou un DataFrame avec colonne 'quantite_y'."""
    if isinstance(data, pd.Series):
        return data
    return data["quantite_y"]

def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
    """Convertit une Series indexée datetime → DataFrame Prophet (ds, y)."""
    return (
        series
        .reset_index()
        .rename(columns={series.index.name or "index": "ds", series.name or 0: "y"})
        [["ds", "y"]]
    )

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
   
   
# ============================================================
# BASELINE — fonction universelle train/predict
# ============================================================

# ── Fonction principale ───────────────────────────────────────────────────────
def run_baseline(
    data_train : pd.DataFrame | pd.Series,
    data_test  : pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame | dict:
    """
    Deux modes selon les arguments fournis :

    MODE PRÉDICTION (data_test=None)
    ─────────────────────────────────
    - Entraîne sur l'ensemble de data_train
    - Retourne un DataFrame de prédictions sur 52 semaines
      Colonnes : ['yhat'] + index datetime

    MODE ÉVALUATION (data_test fourni)
    ────────────────────────────────────
    - Entraîne sur data_train, prédit sur data_test
    - Retourne les métriques seasonal_metrics (dict)

    Paramètres
    ----------
    data_train : Series ou DataFrame (col 'quantite_y' + index datetime)
    data_test  : idem, optionnel
    """

    y_train    = _extract_series(data_train)
    weekly_avg = y_train.groupby(y_train.index.isocalendar().week).mean()

    # ── MODE ÉVALUATION ──────────────────────────────────────────────────────
    if data_test is not None:
        y_test  = _extract_series(data_test)
        y_pred  = y_test.index.isocalendar().week.map(weekly_avg).fillna(0)
        y_pred  = np.clip(y_pred.values, 0, None)
        metrics = utils_series.seasonal_metrics(y_test, y_pred)
        return metrics

    # ── MODE PRÉDICTION ──────────────────────────────────────────────────────
    future_idx = _make_future_index(y_train.index.max(), n_weeks=52)
    weeks      = future_idx.isocalendar().week
    y_pred     = np.clip(weeks.map(weekly_avg).fillna(0).values, 0, None)

    return pd.DataFrame({"yhat": y_pred}, index=future_idx).rename_axis("ds")



 
 
 
    
# ============================================================
# PROPHET — fonction universelle train/predict
# ============================================================
from prophet import Prophet


# ── Hyperparamètres fixes (validés par expérimentation) ──────────────────────
_PROPHET_FIXED: dict = {
    "growth"                  : "flat",
    "seasonality_mode"        : "additive",
    "weekly_seasonality"      : True,
    "daily_seasonality"       : False,
    "yearly_seasonality"      : True,
    "interval_width"          : 0.95,
    "changepoint_prior_scale" : 0.08,
    "holidays_prior_scale"    : 1,
}
    
    
# ── Fonction principale ───────────────────────────────────────────────────────
def run_prophet(
    seasonality_prior_scale : float,
    data_train              : pd.DataFrame | pd.Series,
    data_test               : pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame | dict:
    """
    Deux modes selon les arguments fournis :

    MODE PRÉDICTION (data_test=None)
    ─────────────────────────────────
    - Entraîne sur l'ensemble de data_train
    - Retourne un DataFrame de prédictions sur 52 semaines
      Colonnes : ['yhat', 'yhat_lower', 'yhat_upper'] + index datetime

    MODE ÉVALUATION (data_test fourni)
    ────────────────────────────────────
    - Entraîne sur data_train, prédit sur data_test
    - Retourne les métriques seasonal_metrics (dict)

    Paramètres
    ----------
    seasonality_prior_scale : seul hyperparamètre tunable (validé : [3, 10])
    data_train              : Series ou DataFrame (col 'quantite_y' + index datetime)
    data_test               : idem, optionnel
    """

    y_train  = _extract_series(data_train)
    df_train = _to_prophet_df(y_train)

    # ── Instanciation du modèle ───────────────────────────────────────────────
    m = Prophet(
        growth                 = _PROPHET_FIXED["growth"],
        seasonality_mode       = _PROPHET_FIXED["seasonality_mode"],
        weekly_seasonality     = _PROPHET_FIXED["weekly_seasonality"],
        daily_seasonality      = _PROPHET_FIXED["daily_seasonality"],
        yearly_seasonality     = _PROPHET_FIXED["yearly_seasonality"],
        interval_width         = _PROPHET_FIXED["interval_width"],
        changepoint_prior_scale= _PROPHET_FIXED["changepoint_prior_scale"],
        seasonality_prior_scale= seasonality_prior_scale,
        holidays_prior_scale   = _PROPHET_FIXED["holidays_prior_scale"],
    )
    m.fit(df_train)

    # ── MODE ÉVALUATION ──────────────────────────────────────────────────────
    if data_test is not None:
        y_test   = _extract_series(data_test)
        future   = pd.DataFrame({"ds": y_test.index})
        forecast = m.predict(future)
        y_pred   = np.clip(forecast["yhat"].values, 0, None)
        metrics  = utils_series.seasonal_metrics(y_test, y_pred)
        return metrics

    # ── MODE PRÉDICTION ──────────────────────────────────────────────────────
    future   = m.make_future_dataframe(periods=52, freq="W", include_history=False)
    forecast = m.predict(future)

    df_pred = (
        forecast[["ds", "yhat"]]
        .assign(
            yhat       = lambda d: np.clip(d["yhat"],       0, None),
        )
        .set_index("ds")
    )
    return df_pred


# ============================================================
# XGBOOST — fonction universelle train/predict
# ============================================================
from xgboost import XGBRegressor


# ── Hyperparamètres fixes ─────────────────────────────────────────────────────
_XGB_FIXED: dict = {
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "objective"        : "reg:squarederror",
    "random_state"     : 42,
    "verbosity"        : 0,
}

def _build_features_xg(series: pd.Series) -> pd.DataFrame:
    """
    Features :
      - week         : semaine ISO (1..53)
      - week_sin/cos : encodage polaire de la semaine (saisonnalité circulaire)
      - lag_1        : y_t-1
      - lag_52       : y_t-52 (signal saisonnier annuel)
      - rolling_52   : moyenne glissante sur 52 semaines
    """
    df = pd.DataFrame({"y": series}, index=series.index)

    # semaine ISO + encodage circulaire
    df["week"]     = df.index.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    # lags
    df["lag_1"]  = df["y"].shift(1)
    df["lag_52"] = df["y"].shift(52)

    # rolling mean annuel (min_periods=26 : accepte dès 6 mois de données)
    df["rolling_52"] = df["y"].shift(1).rolling(window=52, min_periods=26).mean()

    return df

FEATURES = ["week", "week_sin", "week_cos", "lag_1", "lag_52", "rolling_52"]


def _make_future_index(last_date: pd.Timestamp, n_weeks: int = 52) -> pd.DatetimeIndex:
    """Génère n_weeks dates hebdomadaires après last_date."""
    return pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=n_weeks,
        freq="W"
    )


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
      Colonnes : ['yhat'] + index datetime

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
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = _XGB_FIXED["subsample"],
        colsample_bytree = _XGB_FIXED["colsample_bytree"],
        objective        = _XGB_FIXED["objective"],
        random_state     = _XGB_FIXED["random_state"],
        verbosity        = _XGB_FIXED["verbosity"],
    )

    # ── Entraînement ─────────────────────────────────────────────────────────
    df_train = _build_features_xg(y_train).dropna()
    X_train  = df_train[FEATURES].values
    y_arr    = df_train["y"].values
    model.fit(X_train, y_arr)

    # ── MODE ÉVALUATION ──────────────────────────────────────────────────────
    if data_test is not None:
        y_test  = _extract_series(data_test)
        full    = pd.concat([y_train, y_test])
        df_full = _build_features_xg(full)
        df_t    = df_full.loc[y_test.index].fillna(0)
        X_test  = df_t[FEATURES].values
        y_pred  = np.clip(model.predict(X_test), 0, None)
        metrics = utils_series.seasonal_metrics(y_test, y_pred)
        return metrics

    # ── MODE PRÉDICTION ──────────────────────────────────────────────────────
    # Historique glissant : on part du train complet et on append les prédictions
    # au fur et à mesure pour recalculer lag_1, lag_52, rolling_52 proprement
    future_idx   = _make_future_index(y_train.index.max(), n_weeks=52)
    history      = y_train.copy()

    preds = []
    for date in future_idx:
        # Reconstruire les features sur history + 1 point fictif pour obtenir
        # les lags à la date cible
        tmp_series = pd.concat([history, pd.Series([np.nan], index=[date])])
        df_tmp     = _build_features_xg(tmp_series)
        row        = df_tmp.loc[date, FEATURES].fillna(0).values.reshape(1, -1)

        yhat = float(np.clip(model.predict(row), 0, None))
        preds.append({"ds": date, "yhat": yhat})

        # Append la prédiction à l'historique pour les lags suivants
        history = pd.concat([history, pd.Series([yhat], index=[date])])

    return pd.DataFrame(preds).set_index("ds")